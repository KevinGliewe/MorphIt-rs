//! The rewritten URDF must carry the same sphere links, joints, colors and
//! remaining collisions as Python's `rewrite_urdf` on bundled robots.
//! The checked-in fixture comes from the Python reference; sphere JSONs follow
//! the same synthetic formula here.

use std::collections::BTreeMap;
use std::path::Path;

use morphit_robot::assemble::rewrite_urdf;
use morphit_robot::color::hex_to_rgba;
use morphit_robot::inspect::{Action, find_urdfs, inspect_urdf, select_urdf};
use roxmltree::{Document, Node};
use serde_json::{Value, json};

fn find<'a, 'i>(n: Node<'a, 'i>, path: &[&str]) -> Node<'a, 'i> {
    path.iter()
        .fold(n, |at, tag| at.children().find(|c| c.is_element() && c.tag_name().name() == *tag).unwrap())
}

fn summarize(text: &str) -> Value {
    let doc = Document::parse(text).unwrap();
    let root = doc.root_element();
    let mut collisions = BTreeMap::new();
    let mut links = Vec::new();
    for link in root.descendants().skip(1).filter(|n| n.has_tag_name("link")) {
        let name = link.attribute("name").unwrap();
        collisions.insert(name.to_string(), link.children().filter(|c| c.has_tag_name("collision")).count());
        if name.contains("_sphere") {
            links.push(json!({
                "name": name,
                "radius": find(link, &["collision", "geometry", "sphere"]).attribute("radius"),
                "material": find(link, &["visual", "material"]).attribute("name"),
                "rgba": find(link, &["visual", "material", "color"]).attribute("rgba"),
            }));
        }
    }
    let joints: Vec<Value> = root
        .children()
        .filter(|j| j.has_tag_name("joint") && j.attribute("name").unwrap_or("").contains("_to_"))
        .map(|j| {
            json!({
                "name": j.attribute("name"),
                "parent": find(j, &["parent"]).attribute("link"),
                "child": find(j, &["child"]).attribute("link"),
                "xyz": find(j, &["origin"]).attribute("xyz"),
            })
        })
        .collect();
    json!({"collisions": collisions, "links": links, "joints": joints})
}

#[test]
fn rewrites_match_python() {
    let base = Path::new(env!("CARGO_MANIFEST_DIR"));
    let Ok(text) = std::fs::read_to_string(base.join("tests/fixtures/py_assemble.json")) else {
        eprintln!("skipping: fixture not found (generated from the Python reference)");
        return;
    };
    let cases: Value = serde_json::from_str(&text).unwrap();
    for (name, case) in cases.as_object().unwrap() {
        let root = base.join("../../web/examples").join(case["folder"].as_str().unwrap());
        let urdf = select_urdf(&find_urdfs(&root), case["urdf"].as_str()).unwrap();
        let report = inspect_urdf(&root, &urdf).unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let sdir = tmp.path().join("spheres");
        std::fs::create_dir_all(&sdir).unwrap();
        for (k, item) in report.collisions.iter().filter(|c| c.action == Action::Pack).enumerate() {
            let k = k as f64;
            let spheres = json!({
                "centers": [[0.01 * k, 0.02, -0.01 * k], [0.0, 0.0, 0.05]],
                "radii": [0.01, 0.02 + 0.001 * k],
            });
            let file = sdir.join(format!("{}_{}.json", item.link_name, item.collision_index));
            std::fs::write(file, spheres.to_string()).unwrap();
        }
        let color = hex_to_rgba(case["color"].as_str().unwrap()).unwrap();
        let variation = case["variation"].as_f64().unwrap();
        let (out, stats) =
            rewrite_urdf(&report, &sdir, &tmp.path().join("out.urdf"), color, variation).unwrap();
        assert!(stats.skipped_pack_items.is_empty(), "{name}");
        let got = summarize(&out);
        let want = &case["summary"];
        assert_eq!(got["collisions"], want["collisions"], "{name}");
        assert_eq!(got["links"], want["links"], "{name}");
        assert_eq!(got["joints"], want["joints"], "{name}");
    }
}
