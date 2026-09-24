//! Mesh preparation parity against Python's `mesh_prep.prepare_mesh`.
//!
//! The OBJ cases under `tests/fixtures/mesh_prep/` and Python's report for
//! each in `tests/fixtures/py_mesh_prep.json` were generated from the Python
//! reference and are checked in. Here the same files go through
//! `morphit::prepare_mesh`; every decision and count must match, and unions
//! must agree in volume and face count. The test skips with a message when the
//! fixture is absent, and needs the `union` feature.

#![cfg(feature = "union")]

use std::path::PathBuf;
use std::sync::Arc;

use morphit::{Mesh, MeshPrepOptions, prepare_mesh};
use serde_json::Value;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures")
}

fn rel_close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * b.abs().max(1e-12)
}

#[test]
fn reports_match_python() {
    let path = fixture_dir().join("py_mesh_prep.json");
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!("skipping: {} not found (generated from the Python reference)", path.display());
        return;
    };
    let cases: Value = serde_json::from_str(&text).unwrap();
    let cases = cases.as_object().unwrap();
    assert!(cases.len() >= 8);
    for (name, case) in cases {
        let mesh = Arc::new(Mesh::load(fixture_dir().join(case["file"].as_str().unwrap())).unwrap());
        let (out, got) = prepare_mesh(&mesh, MeshPrepOptions::DEFAULT);
        let want = &case["report"];
        let ctx = format!("{name}: rust {got:?}\npython {want}");
        for key in ["action", "reason"] {
            assert_eq!(serde_json::to_value(&got).unwrap()[key], want[key], "{key} differs\n{ctx}");
        }
        for key in ["n_bodies", "n_closed", "n_open", "n_degenerate_dropped", "faces_before"] {
            assert_eq!(serde_json::to_value(&got).unwrap()[key], want[key], "{key} differs\n{ctx}");
        }
        assert_eq!(got.overlapping, want["overlapping"].as_bool().unwrap(), "{ctx}");
        assert!(rel_close(got.volume_before, want["volume_before"].as_f64().unwrap(), 1e-9), "{ctx}");
        let want_warnings: Vec<&str> =
            want["warnings"].as_array().unwrap().iter().map(|w| w.as_str().unwrap()).collect();
        assert_eq!(got.warnings, want_warnings, "{ctx}");

        if got.is_unioned() {
            let u = &case["union"];
            // manifold-rust ports Manifold 3.5.0 bit for bit; Python runs the
            // C++ 3.5.3. Both start from the same f32 vertices.
            assert!(rel_close(out.volume(), u["volume"].as_f64().unwrap(), 1e-9), "{ctx}");
            assert!(rel_close(got.volume_after, want["volume_after"].as_f64().unwrap(), 1e-9), "{ctx}");
            assert_eq!(out.faces().len() as u64, u["faces"].as_u64().unwrap(), "{ctx}");
            assert_eq!(got.faces_after as u64, want["faces_after"].as_u64().unwrap(), "{ctx}");
            assert_eq!(out.vertices().len() as u64, u["vertices"].as_u64().unwrap(), "{ctx}");
        } else {
            assert!(Arc::ptr_eq(&out, &mesh), "{ctx}");
            assert_eq!(got.faces_after as u64, want["faces_after"].as_u64().unwrap(), "{ctx}");
        }
    }
}
