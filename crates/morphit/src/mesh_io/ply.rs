//! Stanford PLY reader (ASCII, binary little- and big-endian).
//!
//! Reads `x y z` of the `vertex` element and the `vertex_indices` (or
//! `vertex_index`) list of the `face` element; every other element and
//! property is parsed and skipped, so binary files with extra data (normals,
//! colors, edges) still decode correctly.

use glam::DVec3;

use super::{Soup, fan};

#[derive(Clone, Copy, Debug, PartialEq)]
enum Scalar {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl Scalar {
    fn parse(name: &str) -> Option<Scalar> {
        Some(match name {
            "char" | "int8" => Scalar::I8,
            "uchar" | "uint8" => Scalar::U8,
            "short" | "int16" => Scalar::I16,
            "ushort" | "uint16" => Scalar::U16,
            "int" | "int32" => Scalar::I32,
            "uint" | "uint32" => Scalar::U32,
            "float" | "float32" => Scalar::F32,
            "double" | "float64" => Scalar::F64,
            _ => return None,
        })
    }
}

#[derive(Debug)]
enum Kind {
    Scalar(Scalar),
    List { count: Scalar, item: Scalar },
}

#[derive(Debug)]
struct Property {
    name: String,
    kind: Kind,
}

#[derive(Debug)]
struct Element {
    name: String,
    count: usize,
    props: Vec<Property>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Format {
    Ascii,
    LittleEndian,
    BigEndian,
}

/// A source of numbers in file order.
trait Values {
    fn next(&mut self, ty: Scalar) -> Result<f64, String>;
}

struct Ascii<'a> {
    tokens: std::str::SplitAsciiWhitespace<'a>,
}

impl Values for Ascii<'_> {
    fn next(&mut self, _ty: Scalar) -> Result<f64, String> {
        let t = self.tokens.next().ok_or("unexpected end of data")?;
        t.parse::<f64>().map_err(|_| format!("bad number `{t}`"))
    }
}

struct Binary<'a> {
    data: &'a [u8],
    pos: usize,
    big: bool,
}

impl Binary<'_> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N], String> {
        let end = self.pos + N;
        let bytes = self.data.get(self.pos..end).ok_or("unexpected end of data")?;
        self.pos = end;
        let mut a: [u8; N] = bytes.try_into().expect("slice length is N");
        if self.big {
            a.reverse();
        }
        Ok(a)
    }
}

impl Values for Binary<'_> {
    fn next(&mut self, ty: Scalar) -> Result<f64, String> {
        Ok(match ty {
            Scalar::I8 => i8::from_le_bytes(self.take()?) as f64,
            Scalar::U8 => u8::from_le_bytes(self.take()?) as f64,
            Scalar::I16 => i16::from_le_bytes(self.take()?) as f64,
            Scalar::U16 => u16::from_le_bytes(self.take()?) as f64,
            Scalar::I32 => i32::from_le_bytes(self.take()?) as f64,
            Scalar::U32 => u32::from_le_bytes(self.take()?) as f64,
            Scalar::F32 => f32::from_le_bytes(self.take()?) as f64,
            Scalar::F64 => f64::from_le_bytes(self.take()?),
        })
    }
}

/// Split off the header: returns its lines and the byte offset of the body.
fn header(bytes: &[u8]) -> Result<(Vec<String>, usize), String> {
    let mut lines = Vec::new();
    let mut pos = 0;
    while pos < bytes.len() {
        let end = bytes[pos..].iter().position(|&b| b == b'\n').map_or(bytes.len(), |i| pos + i);
        let line = std::str::from_utf8(&bytes[pos..end]).map_err(|_| "header is not text")?;
        let line = line.trim_end_matches('\r').trim().to_string();
        pos = (end + 1).min(bytes.len());
        if line == "end_header" {
            return Ok((lines, pos));
        }
        lines.push(line);
    }
    Err("missing end_header".into())
}

fn parse_header(lines: &[String]) -> Result<(Format, Vec<Element>), String> {
    if lines.first().map(String::as_str) != Some("ply") {
        return Err("not a PLY file (missing `ply` magic)".into());
    }
    let mut format = None;
    let mut elements: Vec<Element> = Vec::new();
    for line in &lines[1..] {
        let words: Vec<&str> = line.split_ascii_whitespace().collect();
        match words.as_slice() {
            [] | ["comment", ..] | ["obj_info", ..] => {}
            ["format", f, ..] => {
                format = Some(match *f {
                    "ascii" => Format::Ascii,
                    "binary_little_endian" => Format::LittleEndian,
                    "binary_big_endian" => Format::BigEndian,
                    other => return Err(format!("unknown format `{other}`")),
                });
            }
            ["element", name, count] => {
                let count = count.parse().map_err(|_| format!("bad element count `{count}`"))?;
                elements.push(Element { name: name.to_string(), count, props: Vec::new() });
            }
            ["property", "list", count, item, name] => {
                let el = elements.last_mut().ok_or("property before any element")?;
                let count = Scalar::parse(count).ok_or_else(|| format!("unknown type `{count}`"))?;
                let item = Scalar::parse(item).ok_or_else(|| format!("unknown type `{item}`"))?;
                el.props.push(Property { name: name.to_string(), kind: Kind::List { count, item } });
            }
            ["property", ty, name] => {
                let el = elements.last_mut().ok_or("property before any element")?;
                let ty = Scalar::parse(ty).ok_or_else(|| format!("unknown type `{ty}`"))?;
                el.props.push(Property { name: name.to_string(), kind: Kind::Scalar(ty) });
            }
            _ => return Err(format!("unrecognized header line `{line}`")),
        }
    }
    Ok((format.ok_or("missing format line")?, elements))
}

pub(super) fn parse(bytes: &[u8]) -> Result<Soup, String> {
    let (lines, body) = header(bytes)?;
    let (format, elements) = parse_header(&lines)?;
    let mut ascii;
    let mut binary;
    let values: &mut dyn Values = match format {
        Format::Ascii => {
            let text = std::str::from_utf8(&bytes[body..]).map_err(|_| "ASCII body is not text")?;
            ascii = Ascii { tokens: text.split_ascii_whitespace() };
            &mut ascii
        }
        Format::LittleEndian | Format::BigEndian => {
            binary = Binary { data: bytes, pos: body, big: format == Format::BigEndian };
            &mut binary
        }
    };

    let mut vertices = Vec::new();
    let mut faces = Vec::new();
    let mut poly = Vec::new();
    for el in &elements {
        let is_vertex = el.name == "vertex";
        let is_face = el.name == "face";
        let coord = |axis: &str| el.props.iter().position(|p| p.name == axis);
        let xyz = [coord("x"), coord("y"), coord("z")];
        if is_vertex && xyz.iter().any(Option::is_none) {
            return Err("vertex element lacks x, y or z".into());
        }
        let face_list = el.props.iter().position(|p| p.name == "vertex_indices" || p.name == "vertex_index");
        let mut v = [0.0; 3];
        for _ in 0..el.count {
            for (k, prop) in el.props.iter().enumerate() {
                match prop.kind {
                    Kind::Scalar(ty) => {
                        let x = values.next(ty)?;
                        if is_vertex && let Some(axis) = xyz.iter().position(|&i| i == Some(k)) {
                            v[axis] = x;
                        }
                    }
                    Kind::List { count, item } => {
                        let n = values.next(count)?;
                        if !(n >= 0.0 && n.fract() == 0.0) {
                            return Err(format!("bad list length {n}"));
                        }
                        let keep = is_face && face_list == Some(k);
                        poly.clear();
                        for _ in 0..n as usize {
                            let i = values.next(item)?;
                            if keep {
                                if !(i >= 0.0 && i.fract() == 0.0 && i <= u32::MAX as f64) {
                                    return Err(format!("bad vertex index {i}"));
                                }
                                poly.push(i as u32);
                            }
                        }
                        if keep {
                            fan(&poly, &mut faces);
                        }
                    }
                }
            }
            if is_vertex {
                vertices.push(DVec3::from_array(v));
            }
        }
    }
    if vertices.is_empty() {
        return Err("no vertex element".into());
    }
    Ok((vertices, faces))
}

#[cfg(test)]
mod tests {
    use super::*;

    const CUBE_V: [[f64; 3]; 8] = [
        [0., 0., 0.],
        [1., 0., 0.],
        [1., 1., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [1., 0., 1.],
        [1., 1., 1.],
        [0., 1., 1.],
    ];
    /// Outward-wound quads.
    const CUBE_F: [[u32; 4]; 6] =
        [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]];

    fn ascii_cube() -> Vec<u8> {
        let mut s = String::from(
            "ply\nformat ascii 1.0\ncomment made by hand\nelement vertex 8\nproperty float x\n\
             property float y\nproperty float z\nproperty uchar red\nelement face 6\n\
             property list uchar int vertex_indices\nend_header\n",
        );
        for v in CUBE_V {
            s += &format!("{} {} {} 255\n", v[0], v[1], v[2]);
        }
        for f in CUBE_F {
            s += &format!("4 {} {} {} {}\n", f[0], f[1], f[2], f[3]);
        }
        s.into_bytes()
    }

    fn binary_cube(big: bool) -> Vec<u8> {
        let fmt = if big { "binary_big_endian" } else { "binary_little_endian" };
        let mut b = format!(
            "ply\r\nformat {fmt} 1.0\r\nelement vertex 8\r\nproperty double x\r\nproperty double y\r\n\
             property double z\r\nproperty float nx\r\nelement face 6\r\n\
             property list uchar uint vertex_index\r\nelement edge 1\r\nproperty int a\r\nend_header\r\n"
        )
        .into_bytes();
        let put = |b: &mut Vec<u8>, le: &[u8], be: &[u8]| b.extend_from_slice(if big { be } else { le });
        for v in CUBE_V {
            for x in v {
                put(&mut b, &x.to_le_bytes(), &x.to_be_bytes());
            }
            put(&mut b, &0.5f32.to_le_bytes(), &0.5f32.to_be_bytes());
        }
        for f in CUBE_F {
            b.push(4);
            for i in f {
                put(&mut b, &i.to_le_bytes(), &i.to_be_bytes());
            }
        }
        put(&mut b, &7i32.to_le_bytes(), &7i32.to_be_bytes());
        b
    }

    fn check_cube(soup: Soup) {
        let (v, f) = soup;
        assert_eq!(v.len(), 8);
        assert_eq!(f.len(), 12);
        let m = crate::Mesh::from_vertices(v, f, None).unwrap();
        assert!((m.volume() - 1.0).abs() < 1e-12);
        assert!(!m.winding_flipped());
    }

    #[test]
    fn ascii_and_binary_cubes() {
        check_cube(parse(&ascii_cube()).unwrap());
        check_cube(parse(&binary_cube(false)).unwrap());
        check_cube(parse(&binary_cube(true)).unwrap());
    }

    #[test]
    fn malformed_files_are_rejected() {
        assert!(parse(b"solid x\n").unwrap_err().contains("end_header"));
        assert!(parse(b"foo\nend_header\n").unwrap_err().contains("magic"));
        let truncated = binary_cube(false);
        assert!(parse(&truncated[..truncated.len() - 10]).unwrap_err().contains("end of data"));
        let no_z =
            b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nend_header\n0 0\n";
        assert!(parse(no_z).unwrap_err().contains("x, y or z"));
    }
}
