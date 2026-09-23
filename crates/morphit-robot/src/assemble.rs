//! Stage 3 of the robot pipeline, ported from `create_robot_urdf.py`: rewrite
//! the robot URDF so every packed `<collision>` is replaced by sphere child
//! links on fixed joints, and redundant primitive or sphere collisions are
//! removed.
//!
//! Python round-trips the file through ElementTree. Here the original text is
//! kept and only edited: the removed `<collision>` elements are cut out by
//! their byte ranges and the new links (then joints) are inserted before
//! `</robot>`. Comments, attribute order and formatting of everything else
//! stay as they were; the element structure equals Python's output.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;

use morphit::glam::{DMat3, DVec3};
use roxmltree::{Document, Node};

use crate::color::vary_color;
use crate::inspect::{Action, CollisionItem, InspectionReport};
use crate::pack::json_filename;
use crate::xml::children;
use crate::{Error, Result};

const SPHERE_INERTIAL_MASS: f64 = 0.001;
const SPHERE_INERTIAL_DIAG: f64 = 1e-5;
const DRAKE_NS: &str = "http://drake.mit.edu";

/// What the rewrite changed.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RewriteStats {
    pub links_with_collisions_replaced: usize,
    pub mesh_collisions_replaced: usize,
    pub primitive_collisions_removed: usize,
    pub sphere_collisions_removed: usize,
    pub sphere_children_added: usize,
    /// `(link, collision_index, reason)` for pack items that were not replaced.
    pub skipped_pack_items: Vec<(String, usize, String)>,
}

impl RewriteStats {
    /// `link[idx]: reason; ...`, as the API's 400 message lists them.
    pub fn skipped_summary(&self) -> String {
        self.skipped_pack_items
            .iter()
            .map(|(l, i, r)| format!("{l}[{i}]: {r}"))
            .collect::<Vec<_>>()
            .join("; ")
    }
}

/// URDF `rpy` (Rz(yaw) · Ry(pitch) · Rx(roll)) as a matrix.
pub fn rpy_to_matrix(rpy: [f64; 3]) -> DMat3 {
    let [r, p, y] = rpy;
    let (sr, cr) = r.sin_cos();
    let (sp, cp) = p.sin_cos();
    let (sy, cy) = y.sin_cos();
    DMat3::from_cols(
        DVec3::new(cy * cp, sy * cp, -sp),
        DVec3::new(cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, cp * sr),
        DVec3::new(cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr),
    )
}

/// Lift a sphere center from the mesh frame into the link frame.
pub fn apply_collision_transform(c: [f64; 3], xyz: [f64; 3], rpy: [f64; 3]) -> [f64; 3] {
    let r = rpy_to_matrix(rpy);
    // Same term order as the Python code, for identical rounding.
    let row = |i: usize| r.row(i);
    let rot = [0, 1, 2].map(|i| row(i).x * c[0] + row(i).y * c[1] + row(i).z * c[2]);
    [rot[0] + xyz[0], rot[1] + xyz[1], rot[2] + xyz[2]]
}

fn fmt6(v: f64) -> String {
    format!("{v:.6}")
}

fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => o.push_str("&amp;"),
            '<' => o.push_str("&lt;"),
            '>' => o.push_str("&gt;"),
            '"' => o.push_str("&quot;"),
            c => o.push(c),
        }
    }
    o
}

/// Sphere centers and radii.
pub type SphereSet = (Vec<[f64; 3]>, Vec<f64>);

/// `centers` and `radii` of a sphere JSON (a saved `PackResult`).
pub fn load_spheres(path: &Path) -> Result<SphereSet> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::Io(format!("cannot read {}: {e}", path.display())))?;
    load_spheres_str(&text, &path.display().to_string())
}

/// [`load_spheres`] from the JSON text; `name` labels the errors.
pub fn load_spheres_str(text: &str, name: &str) -> Result<SphereSet> {
    let v: serde_json::Value = serde_json::from_str(text).map_err(|e| Error::Io(format!("{name}: {e}")))?;
    let bad = || Error::Io(format!("{name}: missing or malformed centers/radii"));
    let centers: Vec<[f64; 3]> =
        serde_json::from_value(v.get("centers").cloned().ok_or_else(bad)?).map_err(|_| bad())?;
    let radii: Vec<f64> =
        serde_json::from_value(v.get("radii").cloned().ok_or_else(bad)?).map_err(|_| bad())?;
    if centers.len() != radii.len() {
        return Err(Error::Io(format!(
            "{name}: centers ({}) and radii ({}) length mismatch",
            centers.len(),
            radii.len()
        )));
    }
    Ok((centers, radii))
}

/// Where [`rewrite_urdf_text`] gets the spheres of each packed collision.
pub trait SphereSource {
    /// The spheres of `link[index]`, or `None` if it was not packed.
    fn spheres(&self, link: &str, index: usize) -> Result<Option<SphereSet>>;
    /// The skip reason reported when [`SphereSource::spheres`] is `None`.
    fn missing(&self, link: &str, index: usize) -> String;
}

/// Sphere JSONs `<dir>/<link>_<index>.json`, as the pack stage writes them.
pub struct DirSpheres<'a>(pub &'a Path);

impl SphereSource for DirSpheres<'_> {
    fn spheres(&self, link: &str, index: usize) -> Result<Option<SphereSet>> {
        let json = self.0.join(json_filename(link, index));
        if !json.exists() {
            return Ok(None);
        }
        load_spheres(&json).map(Some)
    }

    fn missing(&self, link: &str, index: usize) -> String {
        format!("missing JSON: {}", self.0.join(json_filename(link, index)).display())
    }
}

/// Spheres held in memory, keyed by `(link, collision index)`.
#[derive(Clone, Debug, Default)]
pub struct MemSpheres(pub HashMap<(String, usize), SphereSet>);

impl SphereSource for MemSpheres {
    fn spheres(&self, link: &str, index: usize) -> Result<Option<SphereSet>> {
        Ok(self.0.get(&(link.to_string(), index)).cloned())
    }

    fn missing(&self, link: &str, index: usize) -> String {
        format!("missing JSON: {}", json_filename(link, index))
    }
}

/// The byte range to cut for `node`: the element, plus its line when nothing
/// else shares that line.
fn cut_range(text: &str, node: Node) -> std::ops::Range<usize> {
    let r = node.range();
    let line_start = text[..r.start].rfind('\n').map_or(0, |i| i + 1);
    let line_end = text[r.end..].find('\n').map_or(text.len(), |i| r.end + i + 1);
    let blank = |s: &str| s.chars().all(|c| c == ' ' || c == '\t' || c == '\r' || c == '\n');
    if blank(&text[line_start..r.start]) && blank(&text[r.end..line_end]) { line_start..line_end } else { r }
}

/// Raw text of a `drake:proximity_properties` child of `coll`, with a
/// namespace declaration added when the prefix is not bound at the root.
fn drake_block(text: &str, coll: Node, root: Node) -> Option<String> {
    let props = coll.children().find(|c| {
        c.is_element()
            && c.tag_name().name() == "proximity_properties"
            && c.tag_name().namespace() == Some(DRAKE_NS)
    })?;
    let raw = &text[props.range()];
    let prefix = raw[1..].split(|c: char| c == ':' || c.is_whitespace() || c == '>' || c == '/').next()?;
    if root.lookup_prefix(DRAKE_NS) == Some(prefix) || raw.contains(&format!("xmlns:{prefix}=")) {
        return Some(raw.to_string());
    }
    let at = 1 + raw[1..].find(|c: char| c.is_whitespace() || c == '>' || c == '/')?;
    Some(format!("{} xmlns:{prefix}=\"{DRAKE_NS}\"{}", &raw[..at], &raw[at..]))
}

fn sphere_link(name: &str, radius: f64, rgba: [f64; 4], drake: Option<&str>) -> String {
    let n = esc(name);
    let r = fmt6(radius);
    let m = fmt6(SPHERE_INERTIAL_MASS);
    let i = fmt6(SPHERE_INERTIAL_DIAG);
    let rgba = rgba.iter().map(|&c| fmt6(c)).collect::<Vec<_>>().join(" ");
    let mut s = String::new();
    let _ = write!(
        s,
        "  <link name=\"{n}\">\n    <inertial>\n      <origin xyz=\"0 0 0\" rpy=\"0 0 0\" />\n      \
         <mass value=\"{m}\" />\n      <inertia ixx=\"{i}\" ixy=\"0\" ixz=\"0\" iyy=\"{i}\" iyz=\"0\" izz=\"{i}\" />\n    \
         </inertial>\n    <visual>\n      <origin xyz=\"0 0 0\" rpy=\"0 0 0\" />\n      <geometry>\n        \
         <sphere radius=\"{r}\" />\n      </geometry>\n      <material name=\"color_{n}\">\n        \
         <color rgba=\"{rgba}\" />\n      </material>\n    </visual>\n    <collision>\n      \
         <origin xyz=\"0 0 0\" rpy=\"0 0 0\" />\n      <geometry>\n        <sphere radius=\"{r}\" />\n      \
         </geometry>\n"
    );
    if let Some(d) = drake {
        let _ = writeln!(s, "      {d}");
    }
    s += "    </collision>\n  </link>\n";
    s
}

fn fixed_joint(parent: &str, child: &str, xyz: [f64; 3]) -> String {
    let (p, c) = (esc(parent), esc(child));
    format!(
        "  <joint name=\"{p}_to_{c}\" type=\"fixed\">\n    <parent link=\"{p}\" />\n    <child link=\"{c}\" />\n    \
         <origin xyz=\"{} {} {}\" rpy=\"0 0 0\" />\n  </joint>\n",
        fmt6(xyz[0]),
        fmt6(xyz[1]),
        fmt6(xyz[2])
    )
}

/// Rewrite `report.urdf_path` into `output_path` using the sphere JSONs in
/// `spheres_dir`. Returns the new text and what changed. Pack items whose
/// JSON is missing (or whose link or collision cannot be found) are left in
/// place and listed in [`RewriteStats::skipped_pack_items`].
pub fn rewrite_urdf(
    report: &InspectionReport,
    spheres_dir: &Path,
    output_path: &Path,
    base_color: [f64; 4],
    color_variation: f64,
) -> Result<(String, RewriteStats)> {
    let urdf_path = Path::new(&report.urdf_path);
    let text = std::fs::read_to_string(urdf_path)
        .map_err(|e| Error::Io(format!("cannot read {}: {e}", urdf_path.display())))?;
    let (out, stats) =
        rewrite_urdf_text(&text, report, &DirSpheres(spheres_dir), base_color, color_variation)?;
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| Error::Io(format!("cannot create {}: {e}", parent.display())))?;
    }
    std::fs::write(output_path, &out)
        .map_err(|e| Error::Io(format!("cannot write {}: {e}", output_path.display())))?;
    Ok((out, stats))
}

/// The rewrite itself: the URDF `text` (the file `report` describes) with
/// the spheres from `source`. See [`rewrite_urdf`].
pub fn rewrite_urdf_text(
    text: &str,
    report: &InspectionReport,
    source: &dyn SphereSource,
    base_color: [f64; 4],
    color_variation: f64,
) -> Result<(String, RewriteStats)> {
    let opts = roxmltree::ParsingOptions { allow_dtd: true, ..Default::default() };
    let doc = Document::parse_with_options(text, opts)
        .map_err(|e| Error::Invalid(format!("URDF is not valid XML: {e}")))?;
    let root = doc.root_element();

    let actionable: Vec<&CollisionItem> = report
        .collisions
        .iter()
        .filter(|c| matches!(c.action, Action::Pack | Action::RemovePrimitive | Action::RemoveAlreadySphere))
        .collect();
    let mut pack_links: Vec<&str> = Vec::new();
    for c in &actionable {
        if c.action == Action::Pack && !pack_links.contains(&c.link_name.as_str()) {
            pack_links.push(&c.link_name);
        }
    }
    let n = pack_links.len();
    let colors: HashMap<&str, [f64; 4]> = pack_links
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let c = if n > 1 && color_variation > 0.0 {
                vary_color(base_color, (i as f64 / (n - 1) as f64) * 2.0 - 1.0, color_variation)
            } else {
                base_color
            };
            (*name, c)
        })
        .collect();
    let mut by_link: Vec<(&str, Vec<&CollisionItem>)> = Vec::new();
    for item in actionable {
        match by_link.iter_mut().find(|(l, _)| *l == item.link_name) {
            Some((_, v)) => v.push(item),
            None => by_link.push((&item.link_name, vec![item])),
        }
    }

    let links_in_doc: Vec<Node> =
        root.descendants().skip(1).filter(|n| n.is_element() && n.has_tag_name("link")).collect();
    let mut stats = RewriteStats::default();
    let mut cuts: Vec<std::ops::Range<usize>> = Vec::new();
    let mut new_links = String::new();
    let mut new_joints = String::new();
    for (link_name, mut items) in by_link {
        let Some(link) = links_in_doc.iter().find(|l| l.attribute("name") == Some(link_name)) else {
            for it in items.iter().filter(|it| it.action == Action::Pack) {
                stats.skipped_pack_items.push((
                    link_name.into(),
                    it.collision_index,
                    format!("link <{link_name}> not found in URDF"),
                ));
            }
            continue;
        };
        let collisions: Vec<Node> = children(*link, "collision").collect();
        items.sort_by_key(|it| std::cmp::Reverse(it.collision_index));
        let mut changed = false;
        for item in items {
            let Some(target) = collisions.get(item.collision_index) else {
                if item.action == Action::Pack {
                    stats.skipped_pack_items.push((
                        link_name.into(),
                        item.collision_index,
                        format!("collision_index out of range (link has {} collisions)", collisions.len()),
                    ));
                }
                continue;
            };
            match item.action {
                Action::Pack => {
                    let Some((centers, radii)) = source.spheres(link_name, item.collision_index)? else {
                        stats.skipped_pack_items.push((
                            link_name.into(),
                            item.collision_index,
                            source.missing(link_name, item.collision_index),
                        ));
                        continue;
                    };
                    let drake = drake_block(text, *target, root);
                    cuts.push(cut_range(text, *target));
                    stats.mesh_collisions_replaced += 1;
                    changed = true;
                    let color = colors.get(link_name).copied().unwrap_or(base_color);
                    for (k, (c, r)) in centers.iter().zip(&radii).enumerate() {
                        let child = format!("{link_name}_sphere{}", k + 1);
                        let at = apply_collision_transform(*c, item.origin_xyz, item.origin_rpy);
                        new_links += &sphere_link(&child, *r, color, drake.as_deref());
                        new_joints += &fixed_joint(link_name, &child, at);
                        stats.sphere_children_added += 1;
                    }
                }
                Action::RemovePrimitive => {
                    cuts.push(cut_range(text, *target));
                    stats.primitive_collisions_removed += 1;
                    changed = true;
                }
                Action::RemoveAlreadySphere => {
                    cuts.push(cut_range(text, *target));
                    stats.sphere_collisions_removed += 1;
                    changed = true;
                }
                Action::Error => {}
            }
        }
        stats.links_with_collisions_replaced += usize::from(changed);
    }

    // Splice: cut the removed collisions, insert the new elements before
    // the root's end tag.
    let root_range = root.range();
    let root_text = &text[root_range.clone()];
    let self_closing = root_text.ends_with("/>");
    let insert_at = if self_closing {
        root_range.end - 2
    } else {
        root_range.start + root_text.rfind("</").expect("non-empty element has an end tag")
    };
    cuts.sort_by_key(|r| r.start);
    let mut out = String::with_capacity(text.len() + new_links.len() + new_joints.len());
    if !text.trim_start().starts_with("<?xml") {
        out += "<?xml version='1.0' encoding='utf-8'?>\n";
    }
    let mut pos = 0;
    for r in &cuts {
        out += &text[pos..r.start];
        pos = r.end;
    }
    out += &text[pos..insert_at];
    let additions = new_links + &new_joints;
    if self_closing {
        out += ">\n";
        out += &additions;
        out += "</robot>";
        out += &text[root_range.end..];
    } else {
        if !additions.is_empty() && !out.ends_with('\n') {
            out.push('\n');
        }
        out += &additions;
        out += &text[insert_at..];
    }
    Ok((out, stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::DEFAULT_SPHERE_RGBA;
    use crate::inspect::inspect_urdf;
    use crate::xml::{child, find};

    const URDF: &str = r#"<?xml version="1.0"?>
<robot name="r" xmlns:drake="http://drake.mit.edu">
  <!-- keep me -->
  <link name="a">
    <visual><geometry><mesh filename="a.stl"/></geometry></visual>
    <collision>
      <origin xyz="1 2 3" rpy="0 0 1.5707963267948966"/>
      <geometry><mesh filename="a.stl"/></geometry>
      <drake:proximity_properties><drake:mu_static value="0.5"/></drake:proximity_properties>
    </collision>
    <collision><geometry><box size="1 1 1"/></geometry></collision>
  </link>
  <link name="b">
    <collision><geometry><sphere radius="0.1"/></geometry></collision>
    <collision><geometry><mesh filename="b.stl"/></geometry></collision>
  </link>
  <link name="c"><collision><geometry><mesh filename="a.stl"/></geometry></collision></link>
  <joint name="j" type="fixed"><parent link="a"/><child link="b"/></joint>
</robot>
"#;

    fn setup() -> (tempfile::TempDir, InspectionReport) {
        let t = tempfile::tempdir().unwrap();
        for f in ["a.stl", "b.stl"] {
            std::fs::write(t.path().join(f), "solid").unwrap();
        }
        let urdf = t.path().join("r.urdf");
        std::fs::write(&urdf, URDF).unwrap();
        let report = inspect_urdf(t.path(), &urdf).unwrap();
        (t, report)
    }

    fn write_json(dir: &Path, name: &str, centers: &str, radii: &str) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join(name), format!(r#"{{"centers": {centers}, "radii": {radii}}}"#)).unwrap();
    }

    #[test]
    fn replaces_packed_collisions_with_spheres() {
        let (t, report) = setup();
        let spheres = t.path().join("out/spheres");
        write_json(&spheres, "a_0.json", "[[1, 0, 0], [0, 0, 0]]", "[0.1, 0.2]");
        write_json(&spheres, "b_1.json", "[[0, 0, 1]]", "[0.3]");
        write_json(&spheres, "c_0.json", "[[0, 0, 0]]", "[0.4]");
        let out_path = t.path().join("out/r_spherical.urdf");
        let (text, stats) = rewrite_urdf(&report, &spheres, &out_path, DEFAULT_SPHERE_RGBA, 1.0).unwrap();
        assert_eq!(std::fs::read_to_string(&out_path).unwrap(), text);
        assert!(stats.skipped_pack_items.is_empty(), "{stats:?}");
        assert_eq!(
            (
                stats.mesh_collisions_replaced,
                stats.primitive_collisions_removed,
                stats.sphere_collisions_removed
            ),
            (3, 1, 1)
        );
        assert_eq!((stats.sphere_children_added, stats.links_with_collisions_replaced), (4, 3));

        assert!(text.contains("<!-- keep me -->"));
        assert!(text.starts_with(r#"<?xml version="1.0"?>"#));
        let doc = Document::parse(&text).unwrap();
        let root = doc.root_element();
        let link = |name: &str| children(root, "link").find(|l| l.attribute("name") == Some(name)).unwrap();
        for l in ["a", "b", "c"] {
            assert_eq!(children(link(l), "collision").count(), 0, "{l}");
        }
        assert_eq!(children(link("a"), "visual").count(), 1);
        let names: Vec<&str> = children(root, "link").filter_map(|l| l.attribute("name")).collect();
        assert_eq!(names, ["a", "b", "c", "a_sphere1", "a_sphere2", "b_sphere1", "c_sphere1"]);
        // Links come before all new joints.
        let order: Vec<&str> =
            root.children().filter(|n| n.is_element()).map(|n| n.tag_name().name()).collect();
        assert_eq!(&order[order.len() - 4..], ["joint", "joint", "joint", "joint"]);

        let s1 = link("a_sphere1");
        assert_eq!(
            find(s1, &["collision", "geometry", "sphere"]).unwrap().attribute("radius"),
            Some("0.100000")
        );
        let drake = child(s1, "collision")
            .unwrap()
            .children()
            .find(|c| c.tag_name().name() == "proximity_properties");
        assert_eq!(drake.unwrap().tag_name().namespace(), Some(DRAKE_NS));
        assert_eq!(find(s1, &["visual", "material"]).unwrap().attribute("name"), Some("color_a_sphere1"));
        // Hue spread over three packed links: a at one end, c at the other.
        let rgba = |l: &str| {
            find(link(l), &["visual", "material", "color"]).unwrap().attribute("rgba").unwrap().to_string()
        };
        assert_ne!(rgba("a_sphere1"), rgba("c_sphere1"));
        assert_eq!(rgba("a_sphere1"), rgba("a_sphere2"));

        // Center (1,0,0) rotated 90 degrees about z, then moved by (1,2,3).
        let j = children(root, "joint").find(|j| j.attribute("name") == Some("a_to_a_sphere1")).unwrap();
        assert_eq!(child(j, "origin").unwrap().attribute("xyz"), Some("1.000000 3.000000 3.000000"));
        assert_eq!(child(j, "parent").unwrap().attribute("link"), Some("a"));
        assert_eq!(child(j, "child").unwrap().attribute("link"), Some("a_sphere1"));
    }

    #[test]
    fn missing_jsons_are_reported() {
        let (t, report) = setup();
        let spheres = t.path().join("out/spheres");
        write_json(&spheres, "b_1.json", "[[0, 0, 1]]", "[0.3]");
        let (text, stats) =
            rewrite_urdf(&report, &spheres, &t.path().join("o.urdf"), DEFAULT_SPHERE_RGBA, 0.0).unwrap();
        let skipped: Vec<(&str, usize)> =
            stats.skipped_pack_items.iter().map(|(l, i, _)| (l.as_str(), *i)).collect();
        assert_eq!(skipped, [("a", 0), ("c", 0)]);
        assert!(stats.skipped_summary().starts_with("a[0]: missing JSON: "));
        // The unpacked mesh collision stays, the box next to it goes.
        let doc = Document::parse(&text).unwrap();
        let a = children(doc.root_element(), "link").next().unwrap();
        assert_eq!(children(a, "collision").count(), 1);

        let mut gone = report.clone();
        gone.collisions[0].link_name = "zzz".into();
        gone.collisions[1].link_name = "zzz".into();
        gone.collisions[3].collision_index = 7;
        let (_, stats) =
            rewrite_urdf(&gone, &spheres, &t.path().join("o.urdf"), DEFAULT_SPHERE_RGBA, 0.0).unwrap();
        let reasons: Vec<&str> = stats.skipped_pack_items.iter().map(|s| s.2.as_str()).collect();
        assert!(reasons.contains(&"link <zzz> not found in URDF"), "{reasons:?}");
        assert!(reasons.contains(&"collision_index out of range (link has 2 collisions)"), "{reasons:?}");
    }

    #[test]
    fn declaration_and_prefix_handling() {
        let t = tempfile::tempdir().unwrap();
        std::fs::write(t.path().join("m.stl"), "solid").unwrap();
        let urdf = t.path().join("r.urdf");
        std::fs::write(
            &urdf,
            r#"<robot name="x"><link name="l"><collision><geometry><mesh filename="m.stl"/></geometry><d:proximity_properties xmlns:d="http://drake.mit.edu"><d:x/></d:proximity_properties></collision></link></robot>"#,
        )
        .unwrap();
        let report = inspect_urdf(t.path(), &urdf).unwrap();
        write_json(&t.path().join("s"), "l_0.json", "[[0, 0, 0]]", "[1]");
        let (text, _) =
            rewrite_urdf(&report, &t.path().join("s"), &t.path().join("o.urdf"), DEFAULT_SPHERE_RGBA, 0.0)
                .unwrap();
        assert!(text.starts_with("<?xml version='1.0' encoding='utf-8'?>\n<robot"));
        let doc = Document::parse(&text).unwrap();
        assert!(doc.descendants().any(|n| n.tag_name().name() == "proximity_properties"));
    }

    #[test]
    fn in_memory_rewrite_equals_the_disk_rewrite() {
        let (t, report) = setup();
        let spheres = t.path().join("out/spheres");
        write_json(&spheres, "a_0.json", "[[1, 0, 0], [0, 0, 0]]", "[0.1, 0.2]");
        write_json(&spheres, "b_1.json", "[[0, 0, 1]]", "[0.3]");
        let (disk, disk_stats) =
            rewrite_urdf(&report, &spheres, &t.path().join("o.urdf"), DEFAULT_SPHERE_RGBA, 1.0).unwrap();
        let mut mem = MemSpheres::default();
        mem.0.insert(("a".into(), 0), (vec![[1.0, 0.0, 0.0], [0.0; 3]], vec![0.1, 0.2]));
        mem.0.insert(("b".into(), 1), (vec![[0.0, 0.0, 1.0]], vec![0.3]));
        let (text, stats) = rewrite_urdf_text(URDF, &report, &mem, DEFAULT_SPHERE_RGBA, 1.0).unwrap();
        assert_eq!(text, disk);
        assert_eq!(stats.skipped_summary(), "c[0]: missing JSON: c_0.json");
        assert_eq!(stats.sphere_children_added, disk_stats.sphere_children_added);
        assert_eq!(
            load_spheres_str("{}", "x").unwrap_err().to_string(),
            "x: missing or malformed centers/radii"
        );
    }

    #[test]
    fn rpy_matches_python() {
        // _rpy_to_matrix((0.1, 0.2, 0.3)) @ (1, 2, 3) + (0.5, 0, 0), from Python.
        let c = apply_collision_transform([1.0, 2.0, 3.0], [0.5, 0.0, 0.0], [0.1, 0.2, 0.3]);
        assert_eq!(c, [1.5411536583867151, 2.0916086087501053, 2.9225284408248977]);
    }
}
