//! Mesh from bytes, object URDF/MJCF export and quality metrics through the C API.

mod common;

use std::ptr::{null, null_mut};

use common::*;
use morphit_capi::*;
use serde_json::Value;

const LINK0: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../morphit/tests/fixtures/link0.obj");
const OBJECT_FIXTURE: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/../morphit-robot/tests/fixtures/py_object_model.json");

#[test]
fn mesh_from_bytes_matches_file() {
    let bytes = std::fs::read(LINK0).unwrap();
    let (ext, name) = (c("obj"), c("link0.obj"));
    let mut m = null_mut();
    ok(unsafe { morphit_mesh_from_bytes(bytes.as_ptr(), bytes.len(), ext.as_ptr(), name.as_ptr(), &mut m) });
    let mut info = morphit_mesh_info::default();
    ok(unsafe { morphit_mesh_get_info(m, &mut info) });
    assert_eq!(info.num_faces, 200);
    unsafe { morphit_mesh_free(m) };

    let bad = b"not a mesh";
    let status = unsafe { morphit_mesh_from_bytes(bad.as_ptr(), bad.len(), ext.as_ptr(), null(), &mut m) };
    assert_eq!(status, morphit_status::MORPHIT_ERR_MESH);
    assert!(m.is_null());
    let status = unsafe { morphit_mesh_from_bytes(bytes.as_ptr(), bytes.len(), null(), null(), &mut m) };
    assert_eq!(status, morphit_status::MORPHIT_ERR_NULL_ARG);
}

#[test]
fn object_models_match_python() {
    let mut checked = 0;
    let cases: Value = serde_json::from_str(&std::fs::read_to_string(OBJECT_FIXTURE).unwrap()).unwrap();
    for (name, case) in cases.as_object().unwrap() {
        let centers: Vec<f64> = case["centers"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|p| p.as_array().unwrap().clone())
            .map(|v| v.as_f64().unwrap())
            .collect();
        let radii: Vec<f64> = case["radii"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        // Some cases pair unequal numbers of centers and radii (Python zips them
        // but takes the centroid of all centers); the C API has one count for
        // both, so only the matched cases apply. crates/morphit-robot covers the rest.
        let count = radii.len();
        if centers.len() != 3 * count {
            continue;
        }
        let rgba: Vec<f64> = case["rgba"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let robot_name = c(case["robot_name"].as_str().unwrap());
        let mut opts = std::mem::MaybeUninit::<morphit_object_options>::uninit();
        ok(unsafe { morphit_object_options_default(opts.as_mut_ptr()) });
        let mut opts = unsafe { opts.assume_init() };
        opts.name = robot_name.as_ptr();
        opts.rgba = [rgba[0], rgba[1], rgba[2], rgba[3]];
        for (anchored, key) in [(0, ""), (1, "_anchored")] {
            opts.anchored = anchored;
            let mut centroid = [0.0; 3];
            let centroid_ptr = centroid.as_mut_ptr();
            let urdf = string_out(|b, cap, n| unsafe {
                morphit_object_urdf(centers.as_ptr(), radii.as_ptr(), count, &opts, b, cap, n, centroid_ptr)
            });
            assert_eq!(urdf, case[format!("urdf{key}")].as_str().unwrap(), "{name}: urdf{key}");
            let want: Vec<f64> =
                case["centroid"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
            assert_eq!(centroid.to_vec(), want, "{name}");
            let mjcf = string_out(|b, cap, n| unsafe {
                morphit_object_mjcf(centers.as_ptr(), radii.as_ptr(), count, &opts, b, cap, n, null_mut())
            });
            assert_eq!(mjcf, case[format!("mjcf{key}")].as_str().unwrap(), "{name}: mjcf{key}");
        }
        checked += 1;
    }
    assert!(checked >= 1);
}

#[test]
fn object_model_defaults_and_errors() {
    let centers = [0.0, 0.0, 0.0, 0.1, 0.0, 0.0];
    let radii = [0.02, 0.03];
    // NULL options = defaults.
    let urdf = string_out(|b, cap, n| unsafe {
        morphit_object_urdf(centers.as_ptr(), radii.as_ptr(), 2, null(), b, cap, n, null_mut())
    });
    assert!(urdf.contains("<robot name=\"object\""));
    let mut n = 0;
    let status = unsafe {
        morphit_object_urdf(centers.as_ptr(), radii.as_ptr(), 0, null(), null_mut(), 0, &mut n, null_mut())
    };
    assert_eq!(status, morphit_status::MORPHIT_ERR_INVALID_ARG);
    let status =
        unsafe { morphit_object_urdf(null(), radii.as_ptr(), 2, null(), null_mut(), 0, &mut n, null_mut()) };
    assert_eq!(status, morphit_status::MORPHIT_ERR_NULL_ARG);
}

#[test]
fn quality_metrics() {
    let mesh = box_mesh(1.0, 1.0, 1.0);
    let cfg = small_config("MorphIt-B", 8, 30, 3);
    let s = new_session(mesh, cfg);
    ok(unsafe { morphit_run(s, None, null_mut()) });

    let mut opts = std::mem::MaybeUninit::<morphit_quality_options>::uninit();
    ok(unsafe { morphit_quality_options_default(opts.as_mut_ptr()) });
    let mut opts = unsafe { opts.assume_init() };
    assert_eq!((opts.surface_samples, opts.volume_samples), (50_000, 50_000));
    opts.surface_samples = 2000;
    opts.volume_samples = 2000;

    let mut from_session = morphit_quality_metrics::default();
    ok(unsafe { morphit_session_evaluate(s, &opts, &mut from_session) });
    assert_eq!(from_session.actual_n, state(s).num_spheres);
    assert!(from_session.r_in > 0.0 && from_session.r_in <= 1.0);

    // The same through the arrays and the session's prepared mesh.
    let mut prepared = null_mut();
    ok(unsafe { morphit_session_mesh(s, &mut prepared) });
    let (c, r, _) = spheres(s);
    let mut from_arrays = morphit_quality_metrics::default();
    ok(unsafe {
        morphit_evaluate_packing(prepared, c.as_ptr(), r.as_ptr(), null(), r.len(), &opts, &mut from_arrays)
    });
    assert_eq!(from_arrays.actual_n, from_session.actual_n);
    assert_eq!(from_arrays.r_in, from_session.r_in);

    unsafe {
        morphit_mesh_free(prepared);
        morphit_session_free(s);
        morphit_config_free(cfg);
        morphit_mesh_free(mesh);
    }
}
