//! The robot pipeline through the C API: package, inspect, pack links, assemble.

mod common;

use std::io::Write;
use std::path::Path;
use std::ptr::{null, null_mut};

use common::*;
use morphit_capi::*;

const KINOVA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/examples/kinova_description");

fn from_folder() -> *mut morphit_robot {
    let path = c(KINOVA);
    let mut r = null_mut();
    ok(unsafe { morphit_robot_from_folder(path.as_ptr(), &mut r) });
    r
}

fn inspect(r: *const morphit_robot) -> *mut morphit_robot_report {
    let mut rep = null_mut();
    ok(unsafe { morphit_robot_inspect(r, null(), &mut rep) });
    rep
}

fn params(spheres: usize, iterations: usize) -> morphit_pack_params {
    let mut p = std::mem::MaybeUninit::<morphit_pack_params>::uninit();
    ok(unsafe { morphit_pack_params_default(p.as_mut_ptr()) });
    let mut p = unsafe { p.assume_init() };
    p.num_spheres = spheres;
    p.iterations = iterations;
    p.seed = 0;
    p
}

fn zip_folder(root: &Path) -> Vec<u8> {
    // A stored (uncompressed) zip written by hand keeps the test free of a zip writer dependency.
    let mut files = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for e in std::fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                stack.push(p);
            } else {
                let rel = p.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/");
                files.push((rel, std::fs::read(&p).unwrap()));
            }
        }
    }
    let crc = |data: &[u8]| {
        let mut c = 0xFFFF_FFFFu32;
        for &b in data {
            c ^= b as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { (c >> 1) ^ 0xEDB8_8320 } else { c >> 1 };
            }
        }
        !c
    };
    let (mut out, mut central) = (Vec::new(), Vec::new());
    for (name, data) in &files {
        let offset = out.len() as u32;
        let (crc, size, nlen) = (crc(data), data.len() as u32, name.len() as u16);
        let local = |sig: u32, w: &mut Vec<u8>| {
            w.write_all(&sig.to_le_bytes()).unwrap();
        };
        local(0x0403_4b50, &mut out);
        out.write_all(&[20, 0, 0, 0, 0, 0, 0, 0, 0, 0]).unwrap();
        out.write_all(&crc.to_le_bytes()).unwrap();
        out.write_all(&size.to_le_bytes()).unwrap();
        out.write_all(&size.to_le_bytes()).unwrap();
        out.write_all(&nlen.to_le_bytes()).unwrap();
        out.write_all(&0u16.to_le_bytes()).unwrap();
        out.write_all(name.as_bytes()).unwrap();
        out.write_all(data).unwrap();
        local(0x0201_4b50, &mut central);
        central.write_all(&[20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0]).unwrap();
        central.write_all(&crc.to_le_bytes()).unwrap();
        central.write_all(&size.to_le_bytes()).unwrap();
        central.write_all(&size.to_le_bytes()).unwrap();
        central.write_all(&nlen.to_le_bytes()).unwrap();
        central.write_all(&[0; 12]).unwrap();
        central.write_all(&offset.to_le_bytes()).unwrap();
        central.write_all(name.as_bytes()).unwrap();
    }
    let (cd_offset, cd_size, n) = (out.len() as u32, central.len() as u32, files.len() as u16);
    out.extend(central);
    out.write_all(&0x0605_4b50u32.to_le_bytes()).unwrap();
    out.write_all(&[0, 0, 0, 0]).unwrap();
    out.write_all(&n.to_le_bytes()).unwrap();
    out.write_all(&n.to_le_bytes()).unwrap();
    out.write_all(&cd_size.to_le_bytes()).unwrap();
    out.write_all(&cd_offset.to_le_bytes()).unwrap();
    out.write_all(&0u16.to_le_bytes()).unwrap();
    out
}

#[test]
fn inspect_pack_assemble() {
    let r = from_folder();
    let rep = inspect(r);
    let json = string_out(|b, cap, n| unsafe { morphit_report_json(rep, b, cap, n) });
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    let want = v["collisions"].as_array().unwrap().iter().filter(|c| c["action"] == "pack").count();
    let mut count = 0;
    ok(unsafe { morphit_report_pack_count(rep, &mut count) });
    assert_eq!(count, want);
    assert!(count >= 2);

    let p = params(4, 10);
    for i in 0..2 {
        let mut ci = usize::MAX;
        let ci_ptr: *mut usize = &mut ci;
        let link = string_out(|b, cap, n| unsafe { morphit_report_pack_item(rep, i, b, cap, n, ci_ptr) });
        assert!(!link.is_empty() && ci != usize::MAX);
        let mut s = null_mut();
        ok(unsafe { morphit_robot_pack_link(r, rep, i, &p, c("cpu").as_ptr(), &mut s) });
        ok(unsafe { morphit_run(s, None, null_mut()) });
        assert_eq!(state(s).state, morphit_session_state::MORPHIT_STATE_FINALIZED);
        ok(unsafe { morphit_robot_set_link_result(r, rep, i, s) });
        ok(unsafe { morphit_session_free(s) });
    }
    let mut stats = morphit_assemble_stats::default();
    let stats_ptr: *mut morphit_assemble_stats = &mut stats;
    let color = c("#3399ff");
    let urdf = string_out(|b, cap, n| unsafe {
        morphit_robot_assemble(r, rep, color.as_ptr(), 0.5, b, cap, n, stats_ptr)
    });
    assert!(urdf.contains("<sphere"));
    assert_eq!(stats.mesh_collisions_replaced, 2);

    ok(unsafe { morphit_robot_clear_link_results(r) });
    let urdf =
        string_out(|b, cap, n| unsafe { morphit_robot_assemble(r, rep, null(), 0.0, b, cap, n, stats_ptr) });
    assert!(!urdf.contains("<sphere"));
    assert_eq!(stats.mesh_collisions_replaced, 0);
    unsafe {
        morphit_report_free(rep);
        morphit_robot_free(r);
    }
}

#[test]
fn zip_and_files_give_the_same_report() {
    let zip = zip_folder(Path::new(KINOVA));
    let mut from_zip = null_mut();
    ok(unsafe { morphit_robot_from_zip(zip.as_ptr(), zip.len(), &mut from_zip) });
    let mut added = null_mut();
    ok(unsafe { morphit_robot_new(&mut added) });
    let root = Path::new(KINOVA);
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for e in std::fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                stack.push(p);
            } else {
                let rel = c(&p.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/"));
                let data = std::fs::read(&p).unwrap();
                ok(unsafe { morphit_robot_add_file(added, rel.as_ptr(), data.as_ptr(), data.len()) });
            }
        }
    }
    let json = |r| {
        let rep = inspect(r);
        let j = string_out(|b, cap, n| unsafe { morphit_report_json(rep, b, cap, n) });
        unsafe { morphit_report_free(rep) };
        j
    };
    let folder = from_folder();
    assert_eq!(json(from_zip), json(folder));
    assert_eq!(json(added), json(folder));
    unsafe {
        morphit_robot_free(from_zip);
        morphit_robot_free(added);
        morphit_robot_free(folder);
    }
}

#[test]
fn invalid_requests() {
    let r = from_folder();
    let rep = inspect(r);
    let mut s = null_mut();
    let mut p = params(0, 10);
    assert_eq!(
        unsafe { morphit_robot_pack_link(r, rep, 0, &p, null(), &mut s) },
        morphit_status::MORPHIT_ERR_INVALID_ARG
    );
    assert!(last_error().contains("num_spheres"));
    p.num_spheres = 4;
    let obj = c("MorphIt-Obj");
    p.variant = obj.as_ptr();
    assert_eq!(
        unsafe { morphit_robot_pack_link(r, rep, 0, &p, null(), &mut s) },
        morphit_status::MORPHIT_ERR_INVALID_ARG
    );
    let mut n = 0;
    ok(unsafe { morphit_report_pack_count(rep, &mut n) });
    assert_eq!(
        unsafe { morphit_robot_pack_link(r, rep, n, null(), null(), &mut s) },
        morphit_status::MORPHIT_ERR_INVALID_ARG
    );
    let missing = c("missing.urdf");
    let mut rep2 = null_mut();
    assert_ne!(unsafe { morphit_robot_inspect(r, missing.as_ptr(), &mut rep2) }, OK);
    assert!(rep2.is_null());
    let bad = c("/definitely/not/here");
    let mut r2 = null_mut();
    assert_eq!(unsafe { morphit_robot_from_folder(bad.as_ptr(), &mut r2) }, morphit_status::MORPHIT_ERR_IO);
    let (link, json) = (c("link"), c("{}"));
    assert_eq!(
        unsafe { morphit_robot_set_link_result_json(r, link.as_ptr(), 0, json.as_ptr()) },
        morphit_status::MORPHIT_ERR_IO
    );
    unsafe {
        morphit_report_free(rep);
        morphit_robot_free(r);
    }
}

#[test]
fn robot_sessions_follow_the_session_rules() {
    let r = from_folder();
    let rep = inspect(r);
    let p = params(4, 1000);
    let mut s = null_mut();
    ok(unsafe { morphit_robot_pack_link(r, rep, 0, &p, c("cpu").as_ptr(), &mut s) });
    // A run cancelled from its own callback leaves the session resumable.
    unsafe extern "C" fn stop(_: *const morphit_step_info, _: *mut std::ffi::c_void) -> std::ffi::c_int {
        1
    }
    assert_eq!(unsafe { morphit_run(s, Some(stop), null_mut()) }, morphit_status::MORPHIT_ERR_CANCELLED);
    assert_eq!(state(s).iteration, 1);
    ok(unsafe { morphit_robot_set_link_result(r, rep, 0, s) });
    ok(unsafe { morphit_session_free(s) });
    unsafe {
        morphit_report_free(rep);
        morphit_robot_free(r);
    }
}
