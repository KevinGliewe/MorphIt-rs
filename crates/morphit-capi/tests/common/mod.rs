#![allow(dead_code)]

use std::ffi::{CStr, CString};
use std::ptr::null_mut;

use morphit_capi::*;

pub const OK: morphit_status = morphit_status::MORPHIT_OK;

pub fn c(s: &str) -> CString {
    CString::new(s).unwrap()
}

pub fn last_error() -> String {
    unsafe { CStr::from_ptr(morphit_last_error()) }.to_string_lossy().into_owned()
}

#[track_caller]
pub fn ok(status: morphit_status) {
    assert_eq!(status, OK, "call failed: {}", last_error());
}

/// Axis-aligned box mesh built through the C API.
pub fn box_mesh(sx: f64, sy: f64, sz: f64) -> *mut morphit_mesh {
    let v = [
        [0.0, 0.0, 0.0],
        [sx, 0.0, 0.0],
        [sx, sy, 0.0],
        [0.0, sy, 0.0],
        [0.0, 0.0, sz],
        [sx, 0.0, sz],
        [sx, sy, sz],
        [0.0, sy, sz],
    ];
    let xyz: Vec<f64> = v.iter().flatten().copied().collect();
    let tris: [u32; 36] = [
        0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1,
        6, 5,
    ];
    let mut m = null_mut();
    ok(unsafe { morphit_mesh_from_arrays(xyz.as_ptr(), 8, tris.as_ptr(), 12, &mut m) });
    m
}

/// Two unit boxes overlapping in a 0.6 x 0.8 x 0.9 block (union volume 1.568).
pub fn overlapping_boxes() -> *mut morphit_mesh {
    let corners = |o: [f64; 3]| -> Vec<f64> {
        let mut v = Vec::new();
        for &(x, y, z) in &[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
        ] {
            v.extend([x + o[0], y + o[1], z + o[2]]);
        }
        v
    };
    let tris: [u32; 36] = [
        0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1,
        6, 5,
    ];
    let mut xyz = corners([0.0; 3]);
    xyz.extend(corners([0.4, 0.2, 0.1]));
    let faces: Vec<u32> = tris.iter().copied().chain(tris.iter().map(|i| i + 8)).collect();
    let mut m = null_mut();
    ok(unsafe { morphit_mesh_from_arrays(xyz.as_ptr(), 16, faces.as_ptr(), 24, &mut m) });
    m
}

/// A small, fast config.
pub fn small_config(preset: &str, spheres: i64, iterations: i64, seed: i64) -> *mut morphit_config {
    let mut cfg = null_mut();
    let p = c(preset);
    ok(unsafe { morphit_config_new(p.as_ptr(), &mut cfg) });
    for (k, v) in [
        ("model.num_spheres", spheres),
        ("training.iterations", iterations),
        ("random_seed", seed),
        ("model.num_inside_samples", 400),
        ("model.num_surface_samples", 400),
        ("training.density_control_min_interval", 20),
        ("training.density_control_warmup_steps", 2),
    ] {
        let k = c(k);
        ok(unsafe { morphit_config_set_i64(cfg, k.as_ptr(), v) });
    }
    cfg
}

pub fn new_session(mesh: *const morphit_mesh, cfg: *const morphit_config) -> *mut morphit_session {
    let mut s = null_mut();
    ok(unsafe { morphit_session_new(mesh, cfg, &mut s) });
    s
}

pub fn state(s: *const morphit_session) -> morphit_state_info {
    let mut st = std::mem::MaybeUninit::<morphit_state_info>::uninit();
    ok(unsafe { morphit_state(s, st.as_mut_ptr()) });
    unsafe { st.assume_init() }
}

/// (centers, radii, masses) read atomically.
pub fn spheres(s: *const morphit_session) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut n = 0usize;
    ok(unsafe { morphit_get_spheres(s, null_mut(), null_mut(), null_mut(), 0, &mut n) });
    // Leave headroom: the count cannot grow, but keep the call honest.
    let mut c = vec![0.0; 3 * n];
    let mut r = vec![0.0; n];
    let mut m = vec![0.0; n];
    let mut got = 0usize;
    ok(unsafe { morphit_get_spheres(s, c.as_mut_ptr(), r.as_mut_ptr(), m.as_mut_ptr(), n, &mut got) });
    c.truncate(3 * got);
    r.truncate(got);
    m.truncate(got);
    (c, r, m)
}

pub fn string_out(f: impl Fn(*mut std::ffi::c_char, usize, *mut usize) -> morphit_status) -> String {
    let mut needed = 0usize;
    ok(f(null_mut(), 0, &mut needed));
    let mut buf = vec![0 as std::ffi::c_char; needed];
    ok(f(buf.as_mut_ptr(), buf.len(), &mut needed));
    unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned()
}

/// Raw pointers are not `Send`; tests move handles between threads on purpose.
pub struct SendPtr<T>(pub *mut T);
impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SendPtr<T> {}
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
