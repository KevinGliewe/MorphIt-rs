//! Inspection reports must equal Python's `discover.inspect_urdf` on every
//! bundled robot package. The checked-in fixture (from the Python reference)
//! has paths relative to the package folder; the same is done here.

use std::path::Path;

use morphit_robot::inspect::{InspectionReport, find_urdfs, inspect_urdf, select_urdf};
use serde_json::Value;

fn relative(report: &mut InspectionReport, root: &Path) {
    let root = morphit_robot::paths::canonical(root);
    let rel = |p: &str| Path::new(p).strip_prefix(&root).unwrap().to_string_lossy().replace('\\', "/");
    report.robot_dir = ".".into();
    report.urdf_path = rel(&report.urdf_path);
    report.urdfs_in_folder = report.urdfs_in_folder.iter().map(|p| rel(p)).collect();
    for c in &mut report.collisions {
        c.mesh_path = c.mesh_path.as_deref().map(rel);
    }
}

#[test]
fn reports_match_python() {
    let base = Path::new(env!("CARGO_MANIFEST_DIR"));
    let Ok(text) = std::fs::read_to_string(base.join("tests/fixtures/py_inspect.json")) else {
        eprintln!("skipping: fixture not found (generated from the Python reference)");
        return;
    };
    let examples = base.join("../../web/examples");
    let cases: Value = serde_json::from_str(&text).unwrap();
    for (name, case) in cases.as_object().unwrap() {
        let root = examples.join(case["folder"].as_str().unwrap());
        let urdf = select_urdf(&find_urdfs(&root), case["urdf"].as_str()).unwrap();
        let mut report = inspect_urdf(&root, &urdf).unwrap();
        relative(&mut report, &root);
        let got = serde_json::to_value(&report).unwrap();
        let want = &case["report"];
        for key in ["robot_dir", "urdf_path", "urdfs_in_folder", "visual_count", "warnings", "errors"] {
            assert_eq!(got[key], want[key], "{name}: {key}");
        }
        let (g, w) = (got["collisions"].as_array().unwrap(), want["collisions"].as_array().unwrap());
        assert_eq!(g.len(), w.len(), "{name}: collision count");
        for (a, b) in g.iter().zip(w) {
            assert_eq!(a, b, "{name}");
        }
    }
}
