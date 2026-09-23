//! A packed object as a simulator model, ported from `create_object_urdf.py`:
//!
//! - URDF: one link per sphere, each fixed to a light `base` link at its
//!   center relative to the sphere centroid, plus the web API's
//!   `morphit:centroid` comment, which records the centroid so the spheres can
//!   be mapped back into the mesh frame ([`spheres_from_object_urdf`]).
//! - MJCF (MuJoCo XML): a `base` body at the centroid with one sphere body
//!   per sphere; MuJoCo derives mass and inertia from each geom's density.
//!
//! Both split `total_mass` over the spheres in proportion to r³. A free
//! (dynamic) object has a floating base; an anchored one is welded to the
//! world (URDF: a `world` link and a fixed joint; MJCF: no free joint).

use std::collections::HashMap;

use crate::xml::{child, children, find};
use crate::{Error, Result};

/// Generator settings; [`Default`] is the web API's `URDF_DEFAULTS`.
#[derive(Clone, Debug)]
pub struct ObjectModelOptions {
    /// `<robot name>` or `<mujoco model>`.
    pub robot_name: String,
    pub color_rgba: [f64; 4],
    /// Decimal places of every number written.
    pub decimals: usize,
    /// Total mass, split over the spheres in proportion to r³.
    pub total_mass: f64,
    /// URDF only: mass and inertia diagonal of the `base` link.
    pub base_mass: f64,
    pub base_inertia_diag: f64,
    /// Radii are clamped to at least this.
    pub min_radius: f64,
    /// Weld the object to the world instead of letting it float.
    pub anchored: bool,
}

impl Default for ObjectModelOptions {
    fn default() -> Self {
        ObjectModelOptions {
            robot_name: "object".into(),
            color_rgba: crate::color::DEFAULT_SPHERE_RGBA,
            decimals: 6,
            total_mass: 1.0,
            base_mass: 0.001,
            base_inertia_diag: 1e-5,
            min_radius: 1e-6,
            anchored: false,
        }
    }
}

/// Generated model text and the centroid its sphere positions are relative to.
#[derive(Clone, Debug, PartialEq)]
pub struct ObjectModel {
    pub text: String,
    pub centroid: [f64; 3],
}

fn fnum(v: f64, d: usize) -> String {
    format!("{v:.d$}")
}

fn xyz(v: [f64; 3], d: usize) -> String {
    format!("{} {} {}", fnum(v[0], d), fnum(v[1], d), fnum(v[2], d))
}

/// Spheres ready to write (Python's `load_inputs`): radii padded with the
/// last value or truncated to the number of centers and clamped, centers
/// relative to their centroid, masses in proportion to r³.
struct Inputs {
    radii: Vec<f64>,
    rel_centers: Vec<[f64; 3]>,
    masses: Vec<f64>,
    centroid: [f64; 3],
}

fn load_inputs(centers: &[[f64; 3]], radii: &[f64], opts: &ObjectModelOptions) -> Result<Inputs> {
    if centers.is_empty() || radii.is_empty() {
        return Err(Error::Invalid("JSON must contain non-empty 'centers' and 'radii' arrays.".into()));
    }
    let n = centers.len();
    let mut radii: Vec<f64> = radii.iter().copied().take(n).collect();
    let last = *radii.last().expect("non-empty");
    radii.resize(n, last);
    let radii: Vec<f64> = radii.into_iter().map(|r| opts.min_radius.max(r)).collect();

    // Python sums left to right; keep that order for identical rounding.
    let mut c = [0.0; 3];
    for p in centers {
        for k in 0..3 {
            c[k] += p[k];
        }
    }
    let centroid = c.map(|s| s / n as f64);
    let rel_centers =
        centers.iter().map(|p| [p[0] - centroid[0], p[1] - centroid[1], p[2] - centroid[2]]).collect();
    // Python's `r ** 3` is C `pow`, like `powf`.
    let vols: Vec<f64> = radii.iter().map(|r| r.powf(3.0)).collect();
    let vtot: f64 = vols.iter().sum();
    let masses = vols.iter().map(|v| opts.total_mass * v / vtot).collect();
    Ok(Inputs { radii, rel_centers, masses, centroid })
}

/// Write the URDF for `centers`/`radii` (mesh frame) and embed the centroid
/// comment.
pub fn write_object_urdf(
    centers: &[[f64; 3]],
    radii: &[f64],
    opts: &ObjectModelOptions,
) -> Result<ObjectModel> {
    let Inputs { radii, rel_centers, masses, centroid } = load_inputs(centers, radii, opts)?;
    let d = opts.decimals;
    let rgba = opts.color_rgba.iter().map(|&x| fnum(x, d)).collect::<Vec<_>>().join(" ");
    let base_m = fnum(opts.base_mass, d);
    let base_i = fnum(opts.base_inertia_diag, d);
    let mut x: Vec<String> = vec![
        r#"<?xml version="1.0"?>"#.into(),
        format!(r#"<robot name="{}">"#, opts.robot_name),
        r#"  <material name="default_color">"#.into(),
        format!(r#"    <color rgba="{rgba}"/>"#),
        "  </material>".into(),
    ];
    if opts.anchored {
        x.push(r#"  <link name="world"/>"#.into());
    }
    x.extend([
        r#"  <link name="base">"#.into(),
        "    <inertial>".into(),
        format!(r#"      <mass value="{base_m}"/>"#),
        format!(
            r#"      <inertia ixx="{base_i}" ixy="0.0" ixz="0.0" iyy="{base_i}" iyz="0.0" izz="{base_i}"/>"#
        ),
        "    </inertial>".into(),
        "  </link>".into(),
    ]);
    if opts.anchored {
        x.push(r#"  <joint name="world_to_base" type="fixed">"#.into());
        x.push(r#"    <parent link="world"/>"#.into());
        x.push(r#"    <child link="base"/>"#.into());
        x.push(format!(r#"    <origin xyz="{}" rpy="0 0 0"/>"#, xyz(centroid, d)));
        x.push("  </joint>".into());
    }
    for (i, ((rel, &r), &m)) in rel_centers.iter().zip(&radii).zip(&masses).enumerate() {
        let rad = fnum(r, d);
        let inertia = fnum((2.0 / 5.0) * m * r.powf(2.0), d);
        let name = format!("sphere_{i}");
        x.push(format!(r#"  <link name="{name}">"#));
        x.push("    <visual>".into());
        x.push(r#"      <origin xyz="0 0 0" rpy="0 0 0"/>"#.into());
        x.push(format!(r#"      <geometry><sphere radius="{rad}"/></geometry>"#));
        x.push(r#"      <material name="default_color"/>"#.into());
        x.push("    </visual>".into());
        x.push("    <collision>".into());
        x.push(r#"      <origin xyz="0 0 0" rpy="0 0 0"/>"#.into());
        x.push(format!(r#"      <geometry><sphere radius="{rad}"/></geometry>"#));
        x.push("    </collision>".into());
        x.push("    <inertial>".into());
        x.push(format!(r#"      <mass value="{}"/>"#, fnum(m, d)));
        x.push(format!(
            r#"      <inertia ixx="{inertia}" ixy="0.0" ixz="0.0" iyy="{inertia}" iyz="0.0" izz="{inertia}"/>"#
        ));
        x.push("    </inertial>".into());
        x.push("  </link>".into());
        x.push(format!(r#"  <joint name="{name}_fixed" type="fixed">"#));
        x.push(r#"    <parent link="base"/>"#.into());
        x.push(format!(r#"    <child link="{name}"/>"#));
        x.push(format!(r#"    <origin xyz="{}" rpy="0 0 0"/>"#, xyz(*rel, d)));
        x.push("  </joint>".into());
    }
    x.push("</robot>\n".into());
    let text = inject_centroid_comment(&x.join("\n"), centroid);
    Ok(ObjectModel { text, centroid })
}

/// Write the MJCF (MuJoCo XML) model for `centers`/`radii` (mesh frame): a
/// `base` body at the centroid (with a free joint unless anchored) holding
/// one body per sphere, whose geom density gives the sphere its mass.
pub fn write_object_mjcf(
    centers: &[[f64; 3]],
    radii: &[f64],
    opts: &ObjectModelOptions,
) -> Result<ObjectModel> {
    let Inputs { radii, rel_centers, masses, centroid } = load_inputs(centers, radii, opts)?;
    let d = opts.decimals;
    let rgba = opts.color_rgba.iter().map(|&x| fnum(x, d)).collect::<Vec<_>>().join(" ");
    let mut x: Vec<String> = vec![
        r#"<?xml version="1.0"?>"#.into(),
        format!(r#"<mujoco model="{}">"#, opts.robot_name),
        r#"  <option gravity="0 0 -9.8"/>"#.into(),
        r#"  <compiler inertiafromgeom="true"/>"#.into(),
        "  <worldbody>".into(),
        format!(r#"    <body name="base" pos="{}">"#, xyz(centroid, d)),
    ];
    if !opts.anchored {
        x.push(r#"      <freejoint name="base_free"/>"#.into());
    }
    for (i, ((rel, &r), &m)) in rel_centers.iter().zip(&radii).zip(&masses).enumerate() {
        let vol = (4.0 / 3.0) * std::f64::consts::PI * r.powf(3.0);
        let density = if vol > 0.0 { m / vol } else { 0.0 };
        x.push(format!(r#"      <body name="sphere_{i}" pos="{}">"#, xyz(*rel, d)));
        x.push(format!(
            r#"        <geom name="sphere_{i}_geom" type="sphere" size="{}" density="{}" rgba="{rgba}"/>"#,
            fnum(r, d),
            fnum(density, d)
        ));
        x.push("      </body>".into());
    }
    x.push("    </body>".into());
    x.push("  </worldbody>".into());
    x.push("</mujoco>\n".into());
    Ok(ObjectModel { text: x.join("\n"), centroid })
}

/// `  <!-- morphit:centroid x y z -->` right after the `<robot ...>` tag (or
/// on a first line when there is none).
pub fn inject_centroid_comment(urdf: &str, c: [f64; 3]) -> String {
    let comment = format!("  <!-- morphit:centroid {:.6} {:.6} {:.6} -->", c[0], c[1], c[2]);
    match robot_tag_end(urdf) {
        Some(end) => format!("{}\n{comment}{}", &urdf[..end], &urdf[end..]),
        None => format!("{comment}\n{urdf}"),
    }
}

/// End of the first match of `<robot\b[^>]*>`.
fn robot_tag_end(s: &str) -> Option<usize> {
    let mut from = 0;
    while let Some(i) = s[from..].find("<robot") {
        let at = from + i + "<robot".len();
        let boundary = s[at..].chars().next().is_none_or(|c| !(c.is_alphanumeric() || c == '_'));
        if boundary && let Some(j) = s[at..].find('>') {
            return Some(at + j + 1);
        }
        from = at;
    }
    None
}

/// The centroid of the first `<!-- morphit:centroid x y z -->` comment
/// (numbers as `-?\d+(\.\d+)?`, the API's regex), or `None`.
pub fn extract_centroid(urdf: &str) -> Option<[f64; 3]> {
    let mut from = 0;
    while let Some(i) = urdf[from..].find("<!--") {
        let start = from + i + 4;
        if let Some(c) = parse_centroid_comment(&urdf[start..]) {
            return Some(c);
        }
        from = start;
    }
    None
}

fn parse_centroid_comment(s: &str) -> Option<[f64; 3]> {
    let s = s.trim_start();
    let mut s = s.strip_prefix("morphit:centroid")?;
    let mut out = [0.0; 3];
    for slot in &mut out {
        let t = s.trim_start();
        if t.len() == s.len() {
            return None; // `\s+` needs at least one whitespace
        }
        let len = number_len(t)?;
        *slot = t[..len].parse().ok()?;
        s = &t[len..];
    }
    s.trim_start().starts_with("-->").then_some(out)
}

/// Length of a leading `-?\d+(?:\.\d+)?`.
fn number_len(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    let mut i = usize::from(b.first() == Some(&b'-'));
    let digits = |from: usize| b[from..].iter().take_while(|c| c.is_ascii_digit()).count();
    let int = digits(i);
    if int == 0 {
        return None;
    }
    i += int;
    if b.get(i) == Some(&b'.') {
        let frac = digits(i + 1);
        if frac > 0 {
            i += 1 + frac;
        }
    }
    Some(i)
}

/// Sphere centers (mesh frame) and radii from an object URDF: radii from
/// each top-level link's `collision/geometry/sphere`, centers from the
/// top-level joints whose child is such a link, plus the centroid. Errors
/// carry the API's 400 messages.
pub fn spheres_from_object_urdf(urdf: &str) -> Result<(Vec<[f64; 3]>, Vec<f64>)> {
    let centroid = extract_centroid(urdf).ok_or_else(|| {
        Error::Invalid("URDF has no morphit:centroid comment; not a MorphIt object URDF".into())
    })?;
    let doc = roxmltree::Document::parse(urdf)
        .map_err(|e| Error::Invalid(format!("URDF is not valid XML: {e}")))?;
    let root = doc.root_element();
    let bad = |what: &str| Error::Invalid(format!("URDF is not valid XML: bad {what}"));

    let mut radii_by_link = HashMap::new();
    for link in children(root, "link") {
        let sphere = find(link, &["collision", "geometry", "sphere"]);
        if let Some(s) = sphere {
            let r: f64 =
                s.attribute("radius").and_then(|r| r.trim().parse().ok()).ok_or_else(|| bad("radius"))?;
            radii_by_link.insert(link.attribute("name").unwrap_or(""), r);
        }
    }
    let mut centers = Vec::new();
    let mut radii = Vec::new();
    for joint in children(root, "joint") {
        let Some(r) =
            child(joint, "child").and_then(|c| radii_by_link.get(c.attribute("link").unwrap_or("")))
        else {
            continue;
        };
        let xyz = child(joint, "origin").and_then(|o| o.attribute("xyz")).unwrap_or("0 0 0");
        let v: Vec<f64> = xyz
            .split_whitespace()
            .map(str::parse)
            .collect::<std::result::Result<_, _>>()
            .map_err(|_| bad("origin"))?;
        let [x, y, z] = v[..] else { return Err(bad("origin")) };
        centers.push([x + centroid[0], y + centroid[1], z + centroid[2]]);
        radii.push(*r);
    }
    if centers.is_empty() {
        return Err(Error::Invalid("no spheres found in URDF".into()));
    }
    Ok((centers, radii))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn template_and_centroid() {
        let opts = ObjectModelOptions { robot_name: "cube".into(), ..Default::default() };
        let u = write_object_urdf(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], &[0.5], &opts).unwrap();
        assert_eq!(u.centroid, [1.0, 0.0, 0.0]);
        let lines: Vec<&str> = u.text.lines().collect();
        assert_eq!(lines[0], r#"<?xml version="1.0"?>"#);
        assert_eq!(lines[1], r#"<robot name="cube">"#);
        assert_eq!(lines[2], "  <!-- morphit:centroid 1.000000 0.000000 0.000000 -->");
        assert_eq!(lines[4], r#"    <color rgba="0.200000 0.600000 1.000000 1.000000"/>"#);
        assert!(u.text.contains(r#"<joint name="sphere_1_fixed" type="fixed">"#));
        assert!(u.text.contains(r#"<origin xyz="-1.000000 0.000000 0.000000" rpy="0 0 0"/>"#));
        // Equal radii: half the mass each, I = 0.4 * 0.5 * 0.25.
        assert!(u.text.contains(r#"<mass value="0.500000"/>"#));
        assert!(u.text.contains(r#"ixx="0.050000" ixy="0.0""#));
        assert!(u.text.ends_with("</robot>\n"));
        let (c, r) = spheres_from_object_urdf(&u.text).unwrap();
        assert_eq!(c, vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
        assert_eq!(r, vec![0.5, 0.5]);
    }

    #[test]
    fn centroid_comment_parsing() {
        assert_eq!(extract_centroid("<!-- morphit:centroid 1 -2.5 3.25 -->"), Some([1.0, -2.5, 3.25]));
        assert_eq!(extract_centroid("x<!--x--><!--morphit:centroid\t1.0 2.0 3.0-->"), Some([1.0, 2.0, 3.0]));
        assert_eq!(extract_centroid("<!-- morphit:centroid 1e-7 2 3 -->"), None);
        assert_eq!(extract_centroid("<!-- morphit:centroid 1 2 -->"), None);
        assert_eq!(extract_centroid("<!-- morphit:centroid1 2 3 -->"), None);
        let with = inject_centroid_comment("<?xml?>\n<robot name=\"a\">\n</robot>", [0.1, -0.2, 0.3]);
        assert_eq!(
            with,
            "<?xml?>\n<robot name=\"a\">\n  <!-- morphit:centroid 0.100000 -0.200000 0.300000 -->\n</robot>"
        );
        assert_eq!(
            inject_centroid_comment("<robots/>", [0.0; 3]),
            "  <!-- morphit:centroid 0.000000 0.000000 0.000000 -->\n<robots/>"
        );
    }

    #[test]
    fn reader_errors() {
        let msg = |s: &str| spheres_from_object_urdf(s).unwrap_err().to_string();
        assert_eq!(msg("<robot/>"), "URDF has no morphit:centroid comment; not a MorphIt object URDF");
        assert!(msg("<!-- morphit:centroid 0 0 0 --><robot>").starts_with("URDF is not valid XML: "));
        assert_eq!(msg("<robot><!-- morphit:centroid 0 0 0 --></robot>"), "no spheres found in URDF");
    }

    /// The bundled example URDFs were written by the Python generator; our
    /// writer must reproduce each one byte for byte from its own spheres.
    #[test]
    fn reproduces_the_bundled_example_urdfs() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../web/examples");
        let Ok(entries) = std::fs::read_dir(&dir) else {
            eprintln!("skipping: {} not found", dir.display());
            return;
        };
        let mut checked = 0;
        for e in entries.flatten() {
            let p = e.path();
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            if !name.ends_with(".urdf") || name.ends_with(".spherical.urdf") {
                continue;
            }
            let text = std::fs::read_to_string(&p).unwrap();
            let (centers, radii) = spheres_from_object_urdf(&text).unwrap();
            let robot_name = text.split("<robot name=\"").nth(1).unwrap().split('"').next().unwrap();
            let opts = ObjectModelOptions { robot_name: robot_name.into(), ..Default::default() };
            let u = write_object_urdf(&centers, &radii, &opts).unwrap();
            // The file stores centers and radii rounded to 6 decimals, so
            // masses and inertias recomputed from them drift slightly; the
            // byte-exact comparison is `tests/object_urdf_parity.rs`.
            let a: Vec<&str> = u.text.lines().collect();
            let b: Vec<&str> = text.lines().collect();
            assert_eq!(a.len(), b.len(), "{name}");
            for (x, y) in a.iter().zip(&b) {
                assert!(lines_match(x, y), "{name}:\n rust   {x}\n python {y}");
            }
            checked += 1;
        }
        assert!(checked >= 19, "checked {checked}");
    }

    /// Equal except for small numeric drift.
    fn lines_match(a: &str, b: &str) -> bool {
        if a == b {
            return true;
        }
        let tok = |s: &str| s.split(['"', ' ']).map(str::to_string).collect::<Vec<_>>();
        let (ta, tb) = (tok(a), tok(b));
        ta.len() == tb.len()
            && ta.iter().zip(&tb).all(|(x, y)| {
                x == y
                    || match (x.parse::<f64>(), y.parse::<f64>()) {
                        (Ok(p), Ok(q)) => (p - q).abs() <= 1.5e-6f64.max(1e-3 * q.abs()),
                        _ => false,
                    }
            })
    }
}
