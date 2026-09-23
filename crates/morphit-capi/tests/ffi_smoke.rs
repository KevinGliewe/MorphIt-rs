//! Exercises every exported function through its C ABI, single-threaded.

mod common;

use std::ffi::CStr;
use std::ptr::{null, null_mut};

use common::*;
use morphit_capi::morphit_status::*;
use morphit_capi::*;

#[test]
fn version_is_set() {
    let v = unsafe { CStr::from_ptr(morphit_version()) }.to_str().unwrap();
    assert_eq!(v, env!("CARGO_PKG_VERSION"));
}

#[test]
fn null_arguments_and_error_messages() {
    let mut m = null_mut();
    assert_eq!(unsafe { morphit_mesh_load(null(), &mut m) }, MORPHIT_ERR_NULL_ARG);
    assert!(last_error().contains("path"), "{}", last_error());
    assert!(m.is_null());
    let p = c("x.obj");
    assert_eq!(unsafe { morphit_mesh_load(p.as_ptr(), null_mut()) }, MORPHIT_ERR_NULL_ARG);
    assert_eq!(unsafe { morphit_step(null_mut(), null_mut()) }, MORPHIT_ERR_NULL_ARG);
    assert_eq!(unsafe { morphit_state(null(), null_mut()) }, MORPHIT_ERR_NULL_ARG);
    // A successful call clears the message.
    let mut cfg = null_mut();
    ok(unsafe { morphit_config_new(null(), &mut cfg) });
    assert_eq!(last_error(), "");
    unsafe { morphit_config_free(cfg) };
    // Freeing NULL is a no-op.
    unsafe {
        morphit_mesh_free(null_mut());
        morphit_config_free(null_mut());
        ok(morphit_session_free(null_mut()));
    }
}

#[test]
fn mesh_loading_errors() {
    let mut m = null_mut();
    let missing = c("definitely/missing.obj");
    assert_eq!(unsafe { morphit_mesh_load(missing.as_ptr(), &mut m) }, MORPHIT_ERR_IO);
    let dir = tempfile::tempdir().unwrap();
    let ply = dir.path().join("a.ply");
    std::fs::write(&ply, "ply").unwrap();
    let ply = c(ply.to_str().unwrap());
    assert_eq!(unsafe { morphit_mesh_load(ply.as_ptr(), &mut m) }, MORPHIT_ERR_MESH);
    // Flat mesh: no volume.
    let xyz = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
    let tri = [0u32, 1, 2];
    assert_eq!(
        unsafe { morphit_mesh_from_arrays(xyz.as_ptr(), 3, tri.as_ptr(), 1, &mut m) },
        MORPHIT_ERR_MESH
    );
    assert!(m.is_null());
}

#[test]
fn mesh_from_file_info_and_contains() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../morphit/tests/fixtures/link0.obj");
    let p = c(path);
    let mut m = null_mut();
    ok(unsafe { morphit_mesh_load(p.as_ptr(), &mut m) });
    let mut info = morphit_mesh_info::default();
    ok(unsafe { morphit_mesh_get_info(m, &mut info) });
    assert!(info.num_faces > 100 && info.volume > 0.0);
    unsafe { morphit_mesh_free(m) };

    let b = box_mesh(2.0, 1.0, 1.0);
    ok(unsafe { morphit_mesh_get_info(b, &mut info) });
    assert!((info.volume - 2.0).abs() < 1e-12);
    assert!((info.center_mass[0] - 1.0).abs() < 1e-12);
    assert!((info.inertia[0] - 2.0 * 2.0 / 12.0).abs() < 1e-12, "Ixx = m (b^2 + c^2) / 12");
    let pts = [0.5, 0.5, 0.5, 3.0, 0.5, 0.5, 1.0, 0.5, 0.5];
    let mut inside = [9u8; 3];
    ok(unsafe { morphit_mesh_contains(b, pts.as_ptr(), 3, inside.as_mut_ptr()) });
    assert_eq!(inside, [1, 0, 1]);
    unsafe { morphit_mesh_free(b) };
}

#[test]
fn config_accessors() {
    let mut cfg = null_mut();
    let bad = c("MorphIt-Q");
    assert_eq!(unsafe { morphit_config_new(bad.as_ptr(), &mut cfg) }, MORPHIT_ERR_CONFIG);
    assert!(last_error().contains("unknown preset"));
    let v = c("MorphIt-V");
    ok(unsafe { morphit_config_new(v.as_ptr(), &mut cfg) });

    let k = c("training.coverage_weight");
    let mut f = 0.0;
    ok(unsafe { morphit_config_get_f64(cfg, k.as_ptr(), &mut f) });
    assert_eq!(f, 5000.0);
    let lr = c("training.center_lr");
    ok(unsafe { morphit_config_set_f64(cfg, lr.as_ptr(), 0.004) });
    ok(unsafe { morphit_config_get_f64(cfg, lr.as_ptr(), &mut f) });
    assert_eq!(f, 0.004);
    assert_eq!(unsafe { morphit_config_set_f64(cfg, lr.as_ptr(), f64::NAN) }, MORPHIT_ERR_CONFIG);

    let n = c("model.num_spheres");
    ok(unsafe { morphit_config_set_f64(cfg, n.as_ptr(), 12.0) });
    let mut i = 0i64;
    ok(unsafe { morphit_config_get_i64(cfg, n.as_ptr(), &mut i) });
    assert_eq!(i, 12);
    assert_eq!(unsafe { morphit_config_set_f64(cfg, n.as_ptr(), 12.5) }, MORPHIT_ERR_CONFIG);

    let psm = c("model.per_sphere_mass");
    ok(unsafe { morphit_config_set_bool(cfg, psm.as_ptr(), 1) });
    ok(unsafe { morphit_config_get_i64(cfg, psm.as_ptr(), &mut i) });
    assert_eq!(i, 1);

    let rd = c("results_dir");
    let val = c("out/here");
    ok(unsafe { morphit_config_set_str(cfg, rd.as_ptr(), val.as_ptr()) });

    let bogus = c("training.bogus");
    assert_eq!(unsafe { morphit_config_set_i64(cfg, bogus.as_ptr(), 1) }, MORPHIT_ERR_CONFIG);
    assert!(last_error().contains("bogus"));

    let upd = c(r#"{"training.iterations": 7, "model.num_spheres": 3}"#);
    ok(unsafe { morphit_config_set_json(cfg, upd.as_ptr()) });
    let bad_upd = c(r#"{"training.iterations": 9, "nope.x": 1}"#);
    assert_eq!(unsafe { morphit_config_set_json(cfg, bad_upd.as_ptr()) }, MORPHIT_ERR_CONFIG);
    let it = c("training.iterations");
    ok(unsafe { morphit_config_get_i64(cfg, it.as_ptr(), &mut i) });
    assert_eq!(i, 7, "failed batch update leaves the config unchanged");

    let json = string_out(|b, cap, need| unsafe { morphit_config_to_json(cfg, b, cap, need) });
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["results_dir"], "out/here");
    assert_eq!(v["model"]["num_spheres"], 3);

    let mut small = [0 as std::ffi::c_char; 4];
    let mut needed = 0;
    assert_eq!(
        unsafe { morphit_config_to_json(cfg, small.as_mut_ptr(), small.len(), &mut needed) },
        MORPHIT_ERR_BUFFER_TOO_SMALL
    );
    assert_eq!(needed, json.len() + 1);

    let mut copy = null_mut();
    ok(unsafe { morphit_config_clone(cfg, &mut copy) });
    let j = c(&json);
    let mut from_json = null_mut();
    ok(unsafe { morphit_config_from_json(j.as_ptr(), &mut from_json) });
    ok(unsafe { morphit_config_get_i64(from_json, it.as_ptr(), &mut i) });
    assert_eq!(i, 7);
    unsafe {
        morphit_config_free(cfg);
        morphit_config_free(copy);
        morphit_config_free(from_json);
    }
}

#[test]
fn step_by_step_session() {
    let mesh = box_mesh(1.0, 0.8, 0.6);
    let cfg = small_config("MorphIt-B", 6, 30, 3);
    let s = new_session(mesh, cfg);
    unsafe {
        morphit_mesh_free(mesh);
        morphit_config_free(cfg);
    }
    let st = state(s);
    assert_eq!(st.state, morphit_session_state::MORPHIT_STATE_RUNNING);
    assert_eq!((st.iteration, st.total_iterations, st.num_spheres), (0, 30, 6));
    assert!(st.total_loss.is_nan());
    let mut info = morphit_step_info::default();
    assert_eq!(unsafe { morphit_get_last_step(s, &mut info) }, MORPHIT_ERR_STATE);

    let mut steps = 0;
    let mut dc = 0;
    loop {
        let r = unsafe { morphit_step(s, &mut info) };
        if r == MORPHIT_DONE {
            break;
        }
        ok(r);
        assert_eq!(info.iteration, steps);
        assert!(info.total_loss.is_finite());
        let sum: f64 = info.weighted_losses.iter().sum();
        assert!((sum - info.total_loss).abs() <= 1e-9 * info.total_loss.abs());
        dc += info.density_control_fired;
        steps += 1;
        assert_eq!(info.done != 0, steps == 30);
    }
    assert_eq!(steps, 30);
    assert_eq!(dc, 1, "density control at iteration 20");
    let st = state(s);
    assert_eq!(st.state, morphit_session_state::MORPHIT_STATE_COMPLETED);
    assert_eq!(st.density_control_passes, 1);

    let mut last = morphit_step_info::default();
    ok(unsafe { morphit_get_last_step(s, &mut last) });
    assert_eq!(last.iteration, 29);

    let mut pruned = 99usize;
    ok(unsafe { morphit_finalize(s, &mut pruned) });
    assert_eq!(state(s).state, morphit_session_state::MORPHIT_STATE_FINALIZED);
    assert_eq!(state(s).num_spheres, 6 - pruned);
    assert_eq!(unsafe { morphit_step(s, null_mut()) }, MORPHIT_DONE);

    let (centers, radii, masses) = spheres(s);
    assert_eq!(centers.len(), 3 * radii.len());
    assert_eq!(masses.len(), radii.len());
    let mut n = 0;
    let mut tiny = [0.0; 2];
    assert_eq!(
        unsafe { morphit_get_spheres(s, tiny.as_mut_ptr(), null_mut(), null_mut(), 0, &mut n) },
        MORPHIT_ERR_BUFFER_TOO_SMALL
    );
    let mut r2 = vec![0.0; radii.len()];
    let mut w = 0;
    ok(unsafe { morphit_get_radii(s, r2.as_mut_ptr(), r2.len(), &mut w) });
    assert_eq!((r2, w), (radii.clone(), radii.len()));
    let mut c2 = vec![0.0; centers.len()];
    ok(unsafe { morphit_get_centers(s, c2.as_mut_ptr(), c2.len(), &mut w) });
    assert_eq!(c2, centers);
    let mut m2 = vec![0.0; masses.len()];
    ok(unsafe { morphit_get_masses(s, m2.as_mut_ptr(), m2.len(), &mut w) });
    assert_eq!(m2, masses);

    let json = string_out(|b, cap, need| unsafe { morphit_result_json(s, b, cap, need) });
    let r = morphit::PackResult::from_json_str(&json).unwrap();
    assert_eq!(r.radii, radii);
    assert_eq!(r.config.random_seed, Some(3));

    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("nested").join("res.json");
    let p = c(out.to_str().unwrap());
    ok(unsafe { morphit_result_save(s, p.as_ptr()) });
    assert_eq!(morphit::PackResult::load(&out).unwrap(), r);

    let hist = string_out(|b, cap, need| unsafe { morphit_history_json(s, b, cap, need) });
    let h: serde_json::Value = serde_json::from_str(&hist).unwrap();
    assert_eq!(h["total_loss"].as_array().unwrap().len(), 30);
    let cj = string_out(|b, cap, need| unsafe { morphit_session_config_json(s, b, cap, need) });
    assert!(cj.contains("\"num_spheres\": 6"));
    ok(unsafe { morphit_session_free(s) });
}

#[test]
fn run_finalizes_and_reports_progress() {
    unsafe extern "C" fn count(info: *const morphit_step_info, ud: *mut std::ffi::c_void) -> std::ffi::c_int {
        let n = unsafe { &mut *(ud as *mut u64) };
        assert_eq!(unsafe { (*info).iteration }, *n);
        *n += 1;
        0
    }
    let mesh = box_mesh(1.0, 1.0, 1.0);
    let cfg = small_config("MorphIt-S", 5, 25, 1);
    let s = new_session(mesh, cfg);
    let mut calls = 0u64;
    ok(unsafe { morphit_run(s, Some(count), &mut calls as *mut u64 as *mut _) });
    assert_eq!(calls, 25);
    assert_eq!(state(s).state, morphit_session_state::MORPHIT_STATE_FINALIZED);
    // Running a finished session is a no-op success.
    ok(unsafe { morphit_run(s, None, null_mut()) });
    unsafe {
        ok(morphit_session_free(s));
        morphit_mesh_free(mesh);
        morphit_config_free(cfg);
    }
}

#[test]
fn callback_cancel_leaves_session_resumable() {
    unsafe extern "C" fn stop_at_5(
        info: *const morphit_step_info,
        _: *mut std::ffi::c_void,
    ) -> std::ffi::c_int {
        (unsafe { (*info).iteration } == 4) as std::ffi::c_int
    }
    let mesh = box_mesh(1.0, 1.0, 1.0);
    let cfg = small_config("MorphIt-B", 4, 12, 2);
    let s = new_session(mesh, cfg);
    assert_eq!(unsafe { morphit_run(s, Some(stop_at_5), null_mut()) }, MORPHIT_ERR_CANCELLED);
    assert!(last_error().contains("callback"));
    let st = state(s);
    assert_eq!((st.iteration, st.running), (5, 0));
    assert_eq!(st.state, morphit_session_state::MORPHIT_STATE_RUNNING);
    ok(unsafe { morphit_run(s, None, null_mut()) });
    assert_eq!(state(s).iteration, 12);
    unsafe {
        ok(morphit_session_free(s));
        morphit_mesh_free(mesh);
        morphit_config_free(cfg);
    }
}

#[test]
fn invalid_session_config_is_rejected() {
    let mesh = box_mesh(1.0, 1.0, 1.0);
    let cfg = small_config("MorphIt-B", 4, 12, 2);
    let k = c("training.flatness_weight");
    ok(unsafe { morphit_config_set_f64(cfg, k.as_ptr(), 1.0) });
    let mut s = null_mut();
    assert_eq!(unsafe { morphit_session_new(mesh, cfg, &mut s) }, MORPHIT_ERR_CONFIG);
    assert!(s.is_null());
    assert!(last_error().contains("flatness"));
    unsafe {
        morphit_mesh_free(mesh);
        morphit_config_free(cfg);
    }
}

#[test]
fn thread_pool_cannot_be_reconfigured_after_use() {
    assert_eq!(morphit_set_num_threads(0), MORPHIT_ERR_INVALID_ARG);
    // Running a session initializes the global pool.
    let mesh = box_mesh(1.0, 1.0, 1.0);
    let cfg = small_config("MorphIt-B", 3, 2, 2);
    let s = new_session(mesh, cfg);
    ok(unsafe { morphit_run(s, None, null_mut()) });
    assert_eq!(morphit_set_num_threads(2), MORPHIT_ERR_THREADPOOL);
    unsafe {
        ok(morphit_session_free(s));
        morphit_mesh_free(mesh);
        morphit_config_free(cfg);
    }
}

#[cfg(feature = "test-hooks")]
#[test]
fn panics_become_status_codes() {
    assert_eq!(morphit__test_panic(), MORPHIT_ERR_PANIC);
    assert!(last_error().contains("deliberate test panic"));
}

#[test]
fn device_enumeration_and_session_device() {
    let mut count = usize::MAX;
    ok(unsafe { morphit_device_count(&mut count) });
    assert_ne!(count, usize::MAX);
    assert_eq!(unsafe { morphit_device_count(null_mut()) }, MORPHIT_ERR_NULL_ARG);
    // Out-of-range index is an argument error; valid ones follow the buffer convention.
    let mut needed = 0usize;
    assert_eq!(unsafe { morphit_device_name(count, null_mut(), 0, &mut needed) }, MORPHIT_ERR_INVALID_ARG);
    for i in 0..count {
        let name = string_out(|b, cap, n| unsafe { morphit_device_name(i, b, cap, n) });
        assert!(name.contains('(') && !name.is_empty());
        let mut small = [0 as std::ffi::c_char; 2];
        assert_eq!(
            unsafe { morphit_device_name(i, small.as_mut_ptr(), 2, &mut needed) },
            MORPHIT_ERR_BUFFER_TOO_SMALL
        );
        assert_eq!(needed, name.len() + 1);
    }

    let mesh = box_mesh(1.0, 1.0, 1.0);
    let cfg = small_config("MorphIt-B", 4, 3, 2);
    let k = c("model.device");
    let v = c("cpu");
    ok(unsafe { morphit_config_set_str(cfg, k.as_ptr(), v.as_ptr()) });
    let s = new_session(mesh, cfg);
    assert_eq!(string_out(|b, cap, n| unsafe { morphit_session_device(s, b, cap, n) }), "cpu");
    ok(unsafe { morphit_session_free(s) });

    // A bogus device name is rejected when the session is created.
    let v = c("tpu");
    ok(unsafe { morphit_config_set_str(cfg, k.as_ptr(), v.as_ptr()) });
    let mut bad = null_mut();
    assert_eq!(unsafe { morphit_session_new(mesh, cfg, &mut bad) }, MORPHIT_ERR_CONFIG);
    assert!(last_error().contains("model.device"));
    unsafe { morphit_config_free(cfg) };
    unsafe { morphit_mesh_free(mesh) };
}

#[test]
fn mesh_prep_is_reported() {
    let json_of = |s: *const morphit_session| -> serde_json::Value {
        serde_json::from_str(&string_out(|b, cap, n| unsafe { morphit_session_mesh_prep_json(s, b, cap, n) }))
            .unwrap()
    };
    let mut needed = 0usize;
    assert_eq!(
        unsafe { morphit_session_mesh_prep_json(std::ptr::null(), null_mut(), 0, &mut needed) },
        MORPHIT_ERR_NULL_ARG
    );

    let single = box_mesh(1.0, 1.0, 1.0);
    let cfg = small_config("MorphIt-B", 4, 3, 2);
    let s = new_session(single, cfg);
    let v = json_of(s);
    assert_eq!(v["action"], "unchanged");
    assert_eq!(v["reason"], "single body");
    ok(unsafe { morphit_session_free(s) });

    // Without the `union` feature the step reports `skipped` instead.
    let expected = if cfg!(feature = "union") { "unioned" } else { "skipped" };
    let boxes = overlapping_boxes();
    let s = new_session(boxes, cfg);
    let v = json_of(s);
    assert_eq!(v["action"], expected, "{v}");
    assert_eq!(v["n_bodies"], 2);
    ok(unsafe { morphit_run(s, None, null_mut()) });
    let r: serde_json::Value =
        serde_json::from_str(&string_out(|b, cap, n| unsafe { morphit_result_json(s, b, cap, n) })).unwrap();
    assert_eq!(r["mesh_prep"]["action"], expected);
    ok(unsafe { morphit_session_free(s) });

    let k = c("model.union_overlapping_bodies");
    ok(unsafe { morphit_config_set_bool(cfg, k.as_ptr(), 0) });
    let s = new_session(boxes, cfg);
    assert_eq!(json_of(s)["action"], "disabled");
    ok(unsafe { morphit_session_free(s) });

    unsafe { morphit_config_free(cfg) };
    unsafe { morphit_mesh_free(boxes) };
    unsafe { morphit_mesh_free(single) };
}
