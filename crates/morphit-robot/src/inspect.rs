//! Stage 1 of the robot pipeline, ported from `discover.py`: find the URDF
//! in a robot package and classify every `<collision>` element. Pure
//! metadata; no mesh is loaded.
//!
//! Element lookups follow ElementTree: tags match without a namespace, links
//! are all `<link>` descendants of the root, visuals and collisions are
//! direct children of their link.

use std::path::{Path, PathBuf};

use roxmltree::Node;
use serde::{Deserialize, Serialize};

use crate::paths::{canonical, join_lexical, parent_rel, walk_files};
use crate::vfs::MemPackage;
use crate::xml::{child, children, find};
use crate::{Error, Result, py_list, py_repr};

/// Mesh extensions the packing stage accepts.
pub const SUPPORTED_MESH_EXTENSIONS: [&str; 4] = [".obj", ".stl", ".ply", ".dae"];

/// What the pipeline does with one `<collision>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Action {
    /// A mesh that will be sphere-packed.
    #[serde(rename = "pack")]
    Pack,
    /// A box, cylinder or capsule that is stripped.
    #[serde(rename = "remove-primitive")]
    RemovePrimitive,
    /// An existing sphere that is stripped.
    #[serde(rename = "remove-already-sphere")]
    RemoveAlreadySphere,
    /// Mesh not found, unsupported or malformed.
    #[serde(rename = "error")]
    Error,
}

/// One `<collision>` element (field order is the Python dataclass order, so
/// the JSON keys come out identically).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CollisionItem {
    pub link_name: String,
    /// Index among the `<collision>` children of its link.
    pub collision_index: usize,
    /// `mesh`, `box`, `sphere`, `cylinder`, `capsule` or `unknown`.
    pub geometry_type: String,
    pub action: Action,
    /// Resolved absolute mesh path.
    pub mesh_path: Option<String>,
    /// The raw `filename` attribute.
    pub mesh_filename: Option<String>,
    pub mesh_scale: [f64; 3],
    pub origin_xyz: [f64; 3],
    pub origin_rpy: [f64; 3],
    pub warning: Option<String>,
}

impl CollisionItem {
    fn new(link_name: &str, collision_index: usize, geometry_type: &str, action: Action) -> Self {
        CollisionItem {
            link_name: link_name.to_string(),
            collision_index,
            geometry_type: geometry_type.to_string(),
            action,
            mesh_path: None,
            mesh_filename: None,
            mesh_scale: [1.0; 3],
            origin_xyz: [0.0; 3],
            origin_rpy: [0.0; 3],
            warning: None,
        }
    }
}

/// Everything the packing stage needs.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct InspectionReport {
    pub robot_dir: String,
    pub urdf_path: String,
    pub urdfs_in_folder: Vec<String>,
    pub collisions: Vec<CollisionItem>,
    pub visual_count: usize,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

impl InspectionReport {
    /// The items that get packed.
    pub fn to_pack(&self) -> impl Iterator<Item = &CollisionItem> {
        self.collisions.iter().filter(|c| c.action == Action::Pack)
    }
}

/// Every `.urdf` file below `robot_dir`, sorted.
pub fn find_urdfs(robot_dir: &Path) -> Vec<PathBuf> {
    walk_files(robot_dir).into_iter().filter(|p| p.extension().is_some_and(|e| e == "urdf")).collect()
}

fn file_name(p: &Path) -> String {
    p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

/// Pick the URDF: by basename when `requested` is given, else the only one.
/// Errors carry the Python messages.
pub fn select_urdf(urdfs: &[PathBuf], requested: Option<&str>) -> Result<PathBuf> {
    let available = || {
        let mut names: Vec<String> = urdfs.iter().map(|u| file_name(u)).collect();
        names.sort();
        names.dedup();
        py_list(&names)
    };
    if let Some(req) = requested.filter(|r| !r.is_empty()) {
        let matches: Vec<&PathBuf> = urdfs.iter().filter(|u| file_name(u) == req).collect();
        return match matches.len() {
            0 => {
                Err(Error::Invalid(format!("--urdf {} not found. Available: {}", py_repr(req), available())))
            }
            1 => Ok(matches[0].clone()),
            _ => {
                let paths: Vec<String> = matches.iter().map(|p| p.display().to_string()).collect();
                Err(Error::Invalid(format!(
                    "--urdf {} matches multiple paths: {}",
                    py_repr(req),
                    py_list(&paths)
                )))
            }
        };
    }
    match urdfs {
        [] => Err(Error::Invalid("No .urdf files found in folder.".into())),
        [one] => Ok(one.clone()),
        _ => Err(Error::Invalid(format!(
            "Multiple URDFs found; specify with --urdf NAME.urdf. Available: {}",
            available()
        ))),
    }
}

/// Resolve a `<mesh filename=...>` value (see `resolve_mesh_path` in
/// `discover.py`): `package://X/p` tries `robot_dir/X/p`, `robot_dir/p`, then
/// a unique basename match anywhere below `robot_dir`; `file://p` is taken
/// as is; absolute paths must exist; relative ones are relative to the URDF.
pub fn resolve_mesh_path(filename: &str, robot_dir: &Path, urdf_path: &Path) -> Option<PathBuf> {
    if let Some(rest) = filename.strip_prefix("package://") {
        let (pkg, rel) = rest.split_once('/')?;
        for candidate in [robot_dir.join(pkg).join(rel), robot_dir.join(rel)] {
            if candidate.exists() {
                return Some(canonical(&candidate));
            }
        }
        let base = Path::new(rel).file_name()?;
        let mut matches = walk_all(robot_dir).into_iter().filter(|p| p.file_name() == Some(base));
        let first = matches.next()?;
        return matches.next().is_none().then(|| canonical(&first));
    }
    if let Some(rest) = filename.strip_prefix("file://") {
        let p = Path::new(rest);
        return Some(if p.exists() {
            canonical(p)
        } else {
            std::path::absolute(p).unwrap_or(p.to_path_buf())
        });
    }
    let p = Path::new(filename);
    if p.is_absolute() {
        return p.exists().then(|| canonical(p));
    }
    let candidate = urdf_path.parent().unwrap_or(Path::new("")).join(p);
    candidate.exists().then(|| canonical(&candidate))
}

/// Files and directories below `dir` (what `rglob(name)` can match).
fn walk_all(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else { continue };
        for e in entries.flatten() {
            let p = e.path();
            if e.file_type().is_ok_and(|t| t.is_dir()) {
                stack.push(p.clone());
            }
            out.push(p);
        }
    }
    out
}

/// [`select_urdf`] over the URDFs of an in-memory package; returns the key.
pub fn select_urdf_in(pkg: &MemPackage, requested: Option<&str>) -> Result<String> {
    let urdfs: Vec<PathBuf> = pkg.urdfs().into_iter().map(PathBuf::from).collect();
    let chosen = select_urdf(&urdfs, requested)?;
    Ok(chosen.to_string_lossy().replace('\\', "/"))
}

/// [`resolve_mesh_path`] inside an in-memory package, with the same rules;
/// the result is a package key (except for `file://`, taken as is).
/// Absolute paths name package keys from its root.
pub fn resolve_mesh_path_in(pkg: &MemPackage, filename: &str, urdf_rel: &str) -> Option<String> {
    if let Some(rest) = filename.strip_prefix("package://") {
        let (name, rel) = rest.split_once('/')?;
        for candidate in [join_lexical(name, rel), join_lexical("", rel)].into_iter().flatten() {
            if pkg.exists(&candidate) {
                return Some(candidate);
            }
        }
        let base = rel.rsplit('/').next().filter(|b| !b.is_empty())?;
        let is_base = |k: &str| k.rsplit('/').next() == Some(base);
        let dirs = pkg.dirs();
        let mut matches = pkg.files().map(str::to_string).chain(dirs).filter(|k| is_base(k));
        let first = matches.next()?;
        return matches.next().is_none().then_some(first);
    }
    if let Some(rest) = filename.strip_prefix("file://") {
        return Some(rest.to_string());
    }
    if filename.starts_with('/') {
        let key = join_lexical("", filename)?;
        return pkg.exists(&key).then_some(key);
    }
    let key = join_lexical(parent_rel(urdf_rel), filename)?;
    pkg.exists(&key).then_some(key)
}

/// Unresolved xacro: a `<xacro:` element or a `${...}` substitution.
pub fn looks_like_xacro(text: &str) -> bool {
    if text.contains("<xacro:") {
        return true;
    }
    let mut rest = text;
    while let Some(i) = rest.find("${") {
        let after = &rest[i + 2..];
        match after.find('}') {
            Some(j) if j > 0 => return true,
            _ => rest = after,
        }
    }
    false
}

fn floats3(s: &str) -> Option<[f64; 3]> {
    let v: Vec<f64> = s.split_whitespace().map(|t| t.parse().ok()).collect::<Option<_>>()?;
    v.try_into().ok()
}

/// `(xyz, rpy)` of an `<origin>`, zero when absent or malformed.
pub(crate) fn parse_origin(origin: Option<Node>) -> ([f64; 3], [f64; 3]) {
    let Some(o) = origin else { return ([0.0; 3], [0.0; 3]) };
    let get = |k: &str| floats3(o.attribute(k).unwrap_or("0 0 0")).unwrap_or([0.0; 3]);
    (get("xyz"), get("rpy"))
}

fn parse_scale(s: Option<&str>) -> [f64; 3] {
    let Some(s) = s.filter(|s| !s.is_empty()) else { return [1.0; 3] };
    let parts: Vec<&str> = s.split_whitespace().collect();
    match parts.len() {
        1 => parts[0].parse().map(|v| [v; 3]).unwrap_or([1.0; 3]),
        3 => floats3(s).unwrap_or([1.0; 3]),
        _ => [1.0; 3],
    }
}

/// ElementTree's tag string: `{namespace}name` or `name`.
fn et_tag(n: Node) -> String {
    match n.tag_name().namespace() {
        Some(ns) => format!("{{{ns}}}{}", n.tag_name().name()),
        None => n.tag_name().name().to_string(),
    }
}

/// Build the inspection report for one URDF. Problems with the file itself
/// (xacro, not XML, not a `<robot>`) end up in `errors`, not in `Err`; `Err`
/// is only returned when the file cannot be read.
pub fn inspect_urdf(robot_dir: &Path, urdf_path: &Path) -> Result<InspectionReport> {
    let robot_dir = canonical(robot_dir);
    let urdf_path = canonical(urdf_path);
    let report = InspectionReport {
        robot_dir: robot_dir.display().to_string(),
        urdf_path: urdf_path.display().to_string(),
        urdfs_in_folder: find_urdfs(&robot_dir).iter().map(|p| p.display().to_string()).collect(),
        ..Default::default()
    };
    let bytes = std::fs::read(&urdf_path)
        .map_err(|e| Error::Io(format!("cannot read {}: {e}", urdf_path.display())))?;
    let resolve = |f: &str| resolve_mesh_path(f, &robot_dir, &urdf_path).map(|p| p.display().to_string());
    Ok(inspect_urdf_text(&String::from_utf8_lossy(&bytes), report, &resolve))
}

/// [`inspect_urdf`] for a URDF of an in-memory package. `robot_dir` is empty
/// and every path in the report is a package key.
pub fn inspect_urdf_in(pkg: &MemPackage, urdf_rel: &str) -> Result<InspectionReport> {
    let bytes =
        pkg.read(urdf_rel).ok_or_else(|| Error::Io(format!("cannot read {urdf_rel}: not in the package")))?;
    let report = InspectionReport {
        robot_dir: String::new(),
        urdf_path: urdf_rel.to_string(),
        urdfs_in_folder: pkg.urdfs(),
        ..Default::default()
    };
    let resolve = |f: &str| resolve_mesh_path_in(pkg, f, urdf_rel);
    Ok(inspect_urdf_text(&String::from_utf8_lossy(bytes), report, &resolve))
}

/// Classify the collisions of the URDF `text` into `report` (whose path
/// fields the caller has filled). `resolve` maps a `<mesh filename>` to the
/// mesh's path.
pub fn inspect_urdf_text(
    text: &str,
    mut report: InspectionReport,
    resolve: &dyn Fn(&str) -> Option<String>,
) -> InspectionReport {
    if looks_like_xacro(text) {
        report.errors.push(
            "This file looks like unresolved xacro source: it contains <xacro:...> elements or ${...} \
             substitutions that have not been expanded. Run `xacro foo.xacro > foo.urdf` first and drop \
             the resolved URDF instead."
                .into(),
        );
        return report;
    }
    let opts = roxmltree::ParsingOptions { allow_dtd: true, ..Default::default() };
    let doc = match roxmltree::Document::parse_with_options(text, opts) {
        Ok(d) => d,
        Err(e) => {
            report.errors.push(format!("URDF is not valid XML: {e}"));
            return report;
        }
    };
    let root = doc.root_element();
    if et_tag(root) != "robot" {
        report.errors.push(format!(
            "Top-level element is <{}>, expected <robot>. Is this actually a URDF?",
            et_tag(root)
        ));
        return report;
    }

    let links = root.descendants().skip(1).filter(|n| n.is_element() && et_tag(*n) == "link");
    for link in links {
        let link_name = link.attribute("name").unwrap_or("<unnamed>");
        report.visual_count +=
            children(link, "visual").filter(|v| find(*v, &["geometry", "mesh"]).is_some()).count();
        for (ci, coll) in children(link, "collision").enumerate() {
            let (xyz, rpy) = parse_origin(child(coll, "origin"));
            let item = |geometry: &str, action: Action, warning: Option<String>| CollisionItem {
                origin_xyz: xyz,
                origin_rpy: rpy,
                warning,
                ..CollisionItem::new(link_name, ci, geometry, action)
            };
            let Some(geom) = child(coll, "geometry") else {
                let w = "<collision> has no <geometry> child.".to_string();
                report.collisions.push(item("unknown", Action::Error, Some(w)));
                continue;
            };
            if let Some(mesh) = child(geom, "mesh") {
                let filename = mesh.attribute("filename").unwrap_or("").to_string();
                let scale = parse_scale(mesh.attribute("scale"));
                let mut it = item("mesh", Action::Pack, None);
                it.mesh_filename = Some(filename.clone());
                it.mesh_scale = scale;
                match resolve(&filename) {
                    None => {
                        it.action = Action::Error;
                        it.warning = Some(format!(
                            "Mesh not found: {filename}. Drop the package root so package:// URIs resolve."
                        ));
                    }
                    Some(resolved) => {
                        let suffix = Path::new(&resolved)
                            .extension()
                            .map(|e| format!(".{}", e.to_string_lossy().to_lowercase()))
                            .unwrap_or_default();
                        it.mesh_path = Some(resolved);
                        if !SUPPORTED_MESH_EXTENSIONS.contains(&suffix.as_str()) {
                            it.action = Action::Error;
                            let supported: Vec<String> =
                                SUPPORTED_MESH_EXTENSIONS.iter().map(|s| py_repr(s)).collect();
                            it.warning = Some(format!(
                                "Unsupported mesh extension {}; only ({}) are packed.",
                                py_repr(&suffix),
                                supported.join(", ")
                            ));
                        }
                    }
                }
                report.collisions.push(it);
                continue;
            }
            let prim =
                ["sphere", "box", "cylinder", "capsule"].into_iter().find(|p| child(geom, p).is_some());
            report.collisions.push(match prim {
                Some("sphere") => item(
                    "sphere",
                    Action::RemoveAlreadySphere,
                    Some(
                        "Existing <sphere> collision will be removed; the mesh-derived spheres are the \
                         canonical collision in the output URDF."
                            .into(),
                    ),
                ),
                Some(p) => item(
                    p,
                    Action::RemovePrimitive,
                    Some(format!(
                        "<{p}> collision will be removed. Mesh collisions are the source of truth; \
                         redundant primitives are stripped."
                    )),
                ),
                None => item(
                    "unknown",
                    Action::Error,
                    Some("Unknown collision geometry (not mesh / box / sphere / cylinder / capsule).".into()),
                ),
            });
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, rel: &str, text: &str) -> PathBuf {
        let p = dir.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, text).unwrap();
        p
    }

    const URDF: &str = r#"<?xml version="1.0"?>
<robot name="r">
  <!-- a comment -->
  <link name="a">
    <visual><geometry><mesh filename="package://pkg/meshes/a.stl"/></geometry></visual>
    <collision>
      <origin xyz="0.1 0 0" rpy="0 0 1.5"/>
      <geometry><mesh filename="package://pkg/meshes/a.stl" scale="2"/></geometry>
    </collision>
    <collision><geometry><box size="1 1 1"/></geometry></collision>
    <collision><geometry><sphere radius="1"/></geometry></collision>
    <collision><geometry><mesh filename="../meshes/missing.stl"/></geometry></collision>
    <collision><geometry><mesh filename="../meshes/b.3ds"/></geometry></collision>
    <collision/>
    <collision><geometry><plane/></geometry></collision>
  </link>
  <link name="b"><collision><geometry><mesh filename="c.obj"/></geometry></collision></link>
</robot>"#;

    #[test]
    fn classifies_collisions() {
        let t = tempfile::tempdir().unwrap();
        let root = t.path();
        write(root, "pkg/meshes/a.stl", "solid");
        write(root, "pkg/meshes/b.3ds", "");
        write(root, "pkg/urdf/c.obj", "");
        let urdf = write(root, "pkg/urdf/r.urdf", URDF);
        let r = inspect_urdf(root, &urdf).unwrap();
        assert!(r.errors.is_empty(), "{:?}", r.errors);
        assert_eq!(r.visual_count, 1);
        assert_eq!(r.urdfs_in_folder.len(), 1);
        let a: Vec<(Action, &str)> =
            r.collisions.iter().map(|c| (c.action, c.geometry_type.as_str())).collect();
        assert_eq!(
            a,
            vec![
                (Action::Pack, "mesh"),
                (Action::RemovePrimitive, "box"),
                (Action::RemoveAlreadySphere, "sphere"),
                (Action::Error, "mesh"),
                (Action::Error, "mesh"),
                (Action::Error, "unknown"),
                (Action::Error, "unknown"),
                (Action::Pack, "mesh"),
            ]
        );
        let first = &r.collisions[0];
        assert_eq!(first.origin_xyz, [0.1, 0.0, 0.0]);
        assert_eq!(first.origin_rpy, [0.0, 0.0, 1.5]);
        assert_eq!(first.mesh_scale, [2.0; 3]);
        assert!(first.mesh_path.as_ref().unwrap().ends_with("a.stl"));
        assert_eq!(
            r.collisions[3].warning.as_deref(),
            Some("Mesh not found: ../meshes/missing.stl. Drop the package root so package:// URIs resolve.")
        );
        assert_eq!(
            r.collisions[4].warning.as_deref(),
            Some("Unsupported mesh extension '.3ds'; only ('.obj', '.stl', '.ply', '.dae') are packed.")
        );
        assert_eq!(r.collisions[7].collision_index, 0);
        assert_eq!(r.to_pack().count(), 2);
        // JSON keys in Python order.
        let j = serde_json::to_string(&r.collisions[1]).unwrap();
        assert!(j.starts_with(r#"{"link_name":"a","collision_index":1,"geometry_type":"box","action":"remove-primitive","mesh_path":null"#), "{j}");
    }

    #[test]
    fn file_level_errors() {
        let t = tempfile::tempdir().unwrap();
        let x = write(t.path(), "x.urdf", r#"<robot><xacro:include filename="a"/></robot>"#);
        assert!(
            inspect_urdf(t.path(), &x).unwrap().errors[0]
                .starts_with("This file looks like unresolved xacro")
        );
        let x = write(t.path(), "x.urdf", r#"<robot><link name="${n}"/></robot>"#);
        assert_eq!(inspect_urdf(t.path(), &x).unwrap().errors.len(), 1);
        let x = write(t.path(), "x.urdf", "<robot><link></robot>");
        assert!(inspect_urdf(t.path(), &x).unwrap().errors[0].starts_with("URDF is not valid XML: "));
        let x = write(t.path(), "x.urdf", "<sdf/>");
        assert_eq!(
            inspect_urdf(t.path(), &x).unwrap().errors[0],
            "Top-level element is <sdf>, expected <robot>. Is this actually a URDF?"
        );
        assert!(!looks_like_xacro("cost ${} and $x"));
    }

    #[test]
    fn urdf_selection() {
        let a = PathBuf::from("x/a.urdf");
        let b = PathBuf::from("y/b.urdf");
        let b2 = PathBuf::from("z/b.urdf");
        assert_eq!(select_urdf(std::slice::from_ref(&a), None).unwrap(), a);
        assert_eq!(select_urdf(&[], None).unwrap_err().to_string(), "No .urdf files found in folder.");
        assert_eq!(
            select_urdf(&[b.clone(), a.clone()], None).unwrap_err().to_string(),
            "Multiple URDFs found; specify with --urdf NAME.urdf. Available: ['a.urdf', 'b.urdf']"
        );
        assert_eq!(select_urdf(&[a.clone(), b.clone()], Some("b.urdf")).unwrap(), b);
        assert_eq!(
            select_urdf(std::slice::from_ref(&a), Some("c.urdf")).unwrap_err().to_string(),
            "--urdf 'c.urdf' not found. Available: ['a.urdf']"
        );
        assert!(
            select_urdf(&[b, b2], Some("b.urdf")).unwrap_err().to_string().contains("matches multiple paths")
        );
    }

    #[test]
    fn mesh_resolution() {
        let t = tempfile::tempdir().unwrap();
        let root = t.path();
        let urdf = write(root, "pkg/urdf/r.urdf", "");
        write(root, "pkg/meshes/a.stl", "");
        write(root, "meshes/b.stl", "");
        write(root, "deep/er/c.stl", "");
        write(root, "d1/dup.stl", "");
        write(root, "d2/dup.stl", "");
        let r = |f: &str| resolve_mesh_path(f, root, &urdf).map(|p| p.display().to_string());
        assert!(r("package://pkg/meshes/a.stl").unwrap().ends_with("a.stl"));
        assert!(r("package://other/meshes/b.stl").unwrap().ends_with("b.stl"));
        assert!(r("package://x/c.stl").unwrap().ends_with("c.stl"));
        assert_eq!(r("package://x/dup.stl"), None);
        assert_eq!(r("package://nopath"), None);
        assert!(r("../meshes/a.stl").unwrap().ends_with("a.stl"));
        assert_eq!(r("missing.stl"), None);
        assert!(r("file:///nowhere/x.stl").is_some());
    }

    #[test]
    fn in_memory_packages_resolve_like_directories() {
        let mut pkg = MemPackage::new();
        for f in [
            "pkg/meshes/a.stl",
            "meshes/b.stl",
            "deep/er/c.stl",
            "d1/dup.stl",
            "d2/dup.stl",
            "pkg/urdf/c.obj",
        ] {
            pkg.insert(f, b"".to_vec()).unwrap();
        }
        pkg.insert("pkg/urdf/r.urdf", URDF.as_bytes().to_vec()).unwrap();
        let r = |f: &str| resolve_mesh_path_in(&pkg, f, "pkg/urdf/r.urdf");
        assert_eq!(r("package://pkg/meshes/a.stl").as_deref(), Some("pkg/meshes/a.stl"));
        assert_eq!(r("package://other/meshes/b.stl").as_deref(), Some("meshes/b.stl"));
        assert_eq!(r("package://x/c.stl").as_deref(), Some("deep/er/c.stl"));
        assert_eq!(r("package://x/dup.stl"), None);
        assert_eq!(r("package://x/er").as_deref(), Some("deep/er"), "directories match too");
        assert_eq!(r("package://nopath"), None);
        assert_eq!(r("../meshes/a.stl").as_deref(), Some("pkg/meshes/a.stl"));
        assert_eq!(r("/meshes/b.stl").as_deref(), Some("meshes/b.stl"));
        assert_eq!(r("../../../x.stl"), None);
        assert_eq!(r("missing.stl"), None);

        assert_eq!(select_urdf_in(&pkg, None).unwrap(), "pkg/urdf/r.urdf");
        let report = inspect_urdf_in(&pkg, "pkg/urdf/r.urdf").unwrap();
        assert_eq!(report.urdfs_in_folder, ["pkg/urdf/r.urdf"]);
        assert_eq!(report.collisions[0].mesh_path.as_deref(), Some("pkg/meshes/a.stl"));
        assert_eq!(report.collisions[7].mesh_path.as_deref(), Some("pkg/urdf/c.obj"));
        assert_eq!(report.to_pack().count(), 2);
        assert!(inspect_urdf_in(&pkg, "nope.urdf").is_err());
    }

    #[test]
    fn bundled_robot_examples() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../web/examples");
        if !dir.is_dir() {
            eprintln!("skipping: {} not found", dir.display());
            return;
        }
        // (folder, urdf, pack, remove-primitive)
        for (folder, urdf, pack, prim) in [
            ("kinova_description", "m1n4s200_standalone.urdf", 9, 0),
            ("ur5", "ur5_gripper.urdf", 7, 4),
            ("valkyrie", "valkyrie_A.urdf", 55, 2),
        ] {
            let root = dir.join(folder);
            let path = select_urdf(&find_urdfs(&root), Some(urdf)).unwrap();
            let r = inspect_urdf(&root, &path).unwrap();
            assert!(r.errors.is_empty(), "{folder}: {:?}", r.errors);
            let count = |a| r.collisions.iter().filter(|c| c.action == a).count();
            assert_eq!(count(Action::Error), 0, "{folder}: {:?}", r.collisions);
            assert_eq!((count(Action::Pack), count(Action::RemovePrimitive)), (pack, prim), "{folder}");
            assert!(r.visual_count > 0);
        }
    }
}
