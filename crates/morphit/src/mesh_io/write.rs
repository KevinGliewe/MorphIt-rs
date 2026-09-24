//! Mesh writers: Wavefront OBJ (text, exact f64 round trip) and binary STL
//! (f32, as the format stores it). Used to export the prepared mesh.

use std::fmt::Write as _;
use std::path::Path;
use std::str::FromStr;

use crate::error::{Error, Result};
use crate::mesh::Mesh;

/// A file format [`Mesh::to_bytes`] and [`Mesh::save`] can write.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MeshFormat {
    /// Wavefront OBJ: `v`/`f` lines, coordinates in shortest round-trip form.
    #[default]
    Obj,
    /// Binary STL: one record per triangle, f32 coordinates and face normals.
    Stl,
}

impl MeshFormat {
    /// Every format, in the order of [`MeshFormat::extension`] names.
    pub const ALL: [MeshFormat; 2] = [MeshFormat::Obj, MeshFormat::Stl];

    /// The format named by a file extension (case-insensitive, with or without a dot).
    pub fn from_extension(ext: &str) -> Result<MeshFormat> {
        ext.parse()
    }

    /// Lower-case file extension without the dot.
    pub fn extension(self) -> &'static str {
        match self {
            MeshFormat::Obj => "obj",
            MeshFormat::Stl => "stl",
        }
    }

    /// Media type for downloads.
    pub fn mime(self) -> &'static str {
        match self {
            MeshFormat::Obj => "model/obj",
            MeshFormat::Stl => "model/stl",
        }
    }
}

impl FromStr for MeshFormat {
    type Err = Error;

    fn from_str(s: &str) -> Result<MeshFormat> {
        match s.trim().trim_start_matches('.').to_ascii_lowercase().as_str() {
            "obj" => Ok(MeshFormat::Obj),
            "stl" => Ok(MeshFormat::Stl),
            other => Err(Error::Mesh(format!("cannot write mesh format `{other}`; supported: obj, stl"))),
        }
    }
}

impl Mesh {
    /// The mesh as Wavefront OBJ text. Coordinates use Rust's shortest
    /// round-trip formatting, so loading the text gives the same f64 values.
    pub fn to_obj(&self) -> String {
        let mut s = String::with_capacity(self.vertices().len() * 48 + self.faces().len() * 24);
        let _ =
            writeln!(s, "# MorphIt mesh: {} vertices, {} faces", self.vertices().len(), self.faces().len());
        for v in self.vertices() {
            let _ = writeln!(s, "v {} {} {}", v.x, v.y, v.z);
        }
        for f in self.faces() {
            let _ = writeln!(s, "f {} {} {}", f[0] + 1, f[1] + 1, f[2] + 1);
        }
        s
    }

    /// The mesh as binary STL (f32 coordinates, face normals).
    pub fn to_stl(&self) -> Vec<u8> {
        let tris: Vec<stl_io::Triangle> = self
            .triangles()
            .zip(self.face_normals())
            .map(|(t, n)| stl_io::Triangle {
                normal: stl_io::Vector::new([n.x as f32, n.y as f32, n.z as f32]),
                vertices: t.map(|v| stl_io::Vector::new([v.x as f32, v.y as f32, v.z as f32])),
            })
            .collect();
        let mut out = Vec::with_capacity(84 + tris.len() * 50);
        stl_io::write_stl(&mut out, tris.iter()).expect("writing to a Vec cannot fail");
        out
    }

    /// The mesh in `format`.
    pub fn to_bytes(&self, format: MeshFormat) -> Vec<u8> {
        match format {
            MeshFormat::Obj => self.to_obj().into_bytes(),
            MeshFormat::Stl => self.to_stl(),
        }
    }

    /// Write the mesh to `path`, choosing the format by extension (`.obj`, `.stl`).
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or_default();
        let format = MeshFormat::from_extension(ext)?;
        std::fs::write(path, self.to_bytes(format))
            .map_err(|e| Error::Io(format!("cannot write {}: {e}", path.display())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shapes::box_mesh;
    use glam::DVec3;

    fn odd_box() -> Mesh {
        // Coordinates without short decimal forms.
        box_mesh(DVec3::new(0.1, -1.0 / 3.0, 2.0f64.sqrt()), DVec3::new(0.7, 0.3, 2.5))
    }

    #[test]
    fn obj_round_trips_exactly() {
        let m = odd_box();
        let back = Mesh::load_from_bytes(m.to_obj().as_bytes(), "obj", None).unwrap();
        // The loader may renumber vertices; the triangles are bit-identical.
        assert_eq!(back.triangles().collect::<Vec<_>>(), m.triangles().collect::<Vec<_>>());
        assert_eq!(back.volume(), m.volume());
    }

    #[test]
    fn stl_round_trips_in_f32() {
        let m = odd_box();
        let bytes = m.to_stl();
        assert_eq!(bytes.len(), 84 + 50 * m.faces().len());
        let back = Mesh::load_from_bytes(&bytes, "stl", None).unwrap();
        assert_eq!(back.faces().len(), m.faces().len());
        assert!((back.volume() - m.volume()).abs() < 1e-6 * m.volume());
    }

    #[test]
    fn formats_parse_and_save_by_extension() {
        assert_eq!("OBJ".parse::<MeshFormat>().unwrap(), MeshFormat::Obj);
        assert_eq!(MeshFormat::from_extension(".stl").unwrap(), MeshFormat::Stl);
        assert!(MeshFormat::from_extension("ply").is_err());
        assert_eq!((MeshFormat::Stl.extension(), MeshFormat::Obj.mime()), ("stl", "model/obj"));

        let dir = tempfile::tempdir().unwrap();
        let m = odd_box();
        for name in ["a.obj", "b.STL"] {
            let p = dir.path().join(name);
            m.save(&p).unwrap();
            assert!((Mesh::load(&p).unwrap().volume() - m.volume()).abs() < 1e-6);
        }
        assert!(m.save(dir.path().join("c.ply")).is_err());
    }

    #[cfg(feature = "union")]
    #[test]
    fn a_prepared_mesh_keeps_its_volume_through_a_file() {
        let (union, r) = crate::mesh_prep::tests::two_overlapping_boxes().prepared();
        assert_eq!(r.action, "unioned");
        let back = Mesh::load_from_bytes(union.to_obj().as_bytes(), "obj", None).unwrap();
        assert_eq!(back.volume(), union.volume());
    }
}
