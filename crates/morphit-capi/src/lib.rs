//! Thread-safe C API for MorphIt.
//!
//! # Threading contract
//!
//! - Every function may be called from any thread.
//! - Meshes are immutable and may be shared by any number of sessions and
//!   threads. A session keeps its own reference, so a mesh handle may be freed
//!   right after `morphit_session_new`.
//! - Configs lock internally; sessions copy the config when they are created.
//! - Different sessions are fully independent and may run concurrently.
//! - On one session, the read functions (`morphit_state`, `morphit_get_*`,
//!   `morphit_result_*`) may be called concurrently with `morphit_run` /
//!   `morphit_step` and never wait for an iteration to finish. `morphit_cancel`
//!   may be called from any thread at any time.
//! - `morphit_step`, `morphit_run`, `morphit_finalize` and
//!   `morphit_session_free` return `MORPHIT_ERR_BUSY` while a `morphit_run` is
//!   active on the same session (including from inside its progress callback).
//! - A handle must not be freed while another thread is still using it.
//!
//! Errors: functions return a `morphit_status`; `morphit_last_error()` returns a
//! message for the most recent failed call on the calling thread.
//!
//! Output buffers: functions that return arrays or strings take a buffer, its
//! capacity and a pointer receiving the required size. Passing `NULL` with
//! capacity 0 queries the size; a too-small buffer yields
//! `MORPHIT_ERR_BUFFER_TOO_SMALL` and still reports the required size.

#![allow(non_camel_case_types)]
// Every exported function documents its pointer requirements in the header.
#![allow(clippy::missing_safety_doc)]

mod ffi;
mod handles;
mod log;

use std::ffi::{c_char, c_int, c_void};
use std::ptr::null_mut;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use morphit::{Config, Mesh, Session};
use serde_json::Value;

pub use crate::ffi::morphit_status;
use crate::ffi::morphit_status::*;
use crate::ffi::{FfiError, FfiResult, cstr, guard, out, write_slice, write_string};
use crate::handles::RunningGuard;
pub use crate::handles::{
    MORPHIT_LOSS_COUNT, morphit_config, morphit_loss_id, morphit_mesh, morphit_mesh_info, morphit_session,
    morphit_session_state, morphit_state_info, morphit_step_info,
};
pub use crate::log::morphit_log_fn;

/// Called after every iteration of `morphit_run` with no locks held. Return
/// nonzero to cancel the run. `info` is only valid during the call.
pub type morphit_progress_fn =
    Option<unsafe extern "C" fn(info: *const morphit_step_info, user_data: *mut c_void) -> c_int>;

static VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");

unsafe fn mesh_ref<'a>(p: *const morphit_mesh) -> FfiResult<&'a morphit_mesh> {
    unsafe { p.as_ref() }.ok_or_else(|| FfiError::null("mesh"))
}

unsafe fn config_ref<'a>(p: *const morphit_config) -> FfiResult<&'a morphit_config> {
    unsafe { p.as_ref() }.ok_or_else(|| FfiError::null("config"))
}

unsafe fn session_ref<'a>(p: *const morphit_session) -> FfiResult<&'a morphit_session> {
    unsafe { p.as_ref() }.ok_or_else(|| FfiError::null("session"))
}

fn busy() -> FfiError {
    FfiError::new(MORPHIT_ERR_BUSY, "morphit_run is active on this session")
}

// ---------------------------------------------------------------------------
// Library
// ---------------------------------------------------------------------------

/// Library version string, e.g. "0.1.0". Static storage; do not free.
#[unsafe(no_mangle)]
pub extern "C" fn morphit_version() -> *const c_char {
    VERSION.as_ptr().cast()
}

/// Message for the most recent failed call on this thread, or "" if the most
/// recent call succeeded. Valid until the next API call on this thread.
#[unsafe(no_mangle)]
pub extern "C" fn morphit_last_error() -> *const c_char {
    ffi::last_error_ptr()
}

/// Set the number of worker threads shared by all sessions. Must be called
/// before the first session is created; returns `MORPHIT_ERR_THREADPOOL` once
/// the pool exists. By default one worker per logical CPU is used.
#[unsafe(no_mangle)]
pub extern "C" fn morphit_set_num_threads(num_threads: usize) -> morphit_status {
    guard(|| {
        if num_threads == 0 {
            return Err(FfiError::invalid("num_threads must be at least 1"));
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .map_err(|e| FfiError::new(MORPHIT_ERR_THREADPOOL, e.to_string()))?;
        Ok(MORPHIT_OK)
    })
}

/// Forward library log messages with level <= `max_level` (1 = error ...
/// 5 = trace) to `callback`, which may be invoked from any thread. Pass NULL to
/// stop forwarding. Fails with `MORPHIT_ERR_STATE` if the host process already
/// installed its own global `tracing` subscriber.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_set_log_callback(
    callback: morphit_log_fn,
    user_data: *mut c_void,
    max_level: c_int,
) -> morphit_status {
    guard(|| {
        if log::set_callback(callback, user_data, max_level) {
            Ok(MORPHIT_OK)
        } else {
            Err(FfiError::new(MORPHIT_ERR_STATE, "a global tracing subscriber is already installed"))
        }
    })
}

/// Number of GPU adapters usable for `model.device = "gpu:N"` (0 in a build
/// without GPU support). Enumerating adapters happens once per process.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_device_count(count: *mut usize) -> morphit_status {
    guard(|| {
        *unsafe { out(count, "count") }? = morphit::list_devices().len();
        Ok(MORPHIT_OK)
    })
}

/// Name and kind of GPU adapter `index`, e.g. `"NVIDIA RTX A2000 (discrete, Vulkan)"`,
/// as a NUL-terminated string (buffer convention as for the JSON functions).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_device_name(
    index: usize,
    buf: *mut c_char,
    capacity: usize,
    needed: *mut usize,
) -> morphit_status {
    guard(|| {
        let list = morphit::list_devices();
        let d = list.get(index).ok_or_else(|| {
            FfiError::invalid(format!("device index {index} out of range (count {})", list.len()))
        })?;
        let s = format!("{} ({}, {})", d.name, d.kind, d.backend);
        unsafe { write_string(&s, buf, capacity, needed) }
    })
}

// ---------------------------------------------------------------------------
// Meshes
// ---------------------------------------------------------------------------

/// Load an .obj or .stl file (UTF-8 path). On success `*out` receives a mesh
/// to be released with `morphit_mesh_free`; on failure it is set to NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_mesh_load(
    path: *const c_char,
    out_mesh: *mut *mut morphit_mesh,
) -> morphit_status {
    guard(|| {
        let o = unsafe { out(out_mesh, "out") }?;
        *o = null_mut();
        let path = unsafe { cstr(path, "path") }?;
        let mesh = Mesh::load(path)?;
        *o = Box::into_raw(Box::new(morphit_mesh { mesh: Arc::new(mesh) }));
        Ok(MORPHIT_OK)
    })
}

/// Build a mesh from `num_vertices` xyz triples and `num_triangles` index
/// triples (zero-based, counter-clockwise seen from outside). The data is copied.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_mesh_from_arrays(
    xyz: *const f64,
    num_vertices: usize,
    triangles: *const u32,
    num_triangles: usize,
    out_mesh: *mut *mut morphit_mesh,
) -> morphit_status {
    guard(|| {
        let o = unsafe { out(out_mesh, "out") }?;
        *o = null_mut();
        if xyz.is_null() {
            return Err(FfiError::null("xyz"));
        }
        if triangles.is_null() {
            return Err(FfiError::null("triangles"));
        }
        let nv = num_vertices.checked_mul(3).ok_or_else(|| FfiError::invalid("num_vertices too large"))?;
        let nt = num_triangles.checked_mul(3).ok_or_else(|| FfiError::invalid("num_triangles too large"))?;
        // SAFETY: the caller provides arrays of the stated sizes.
        let (v, t) =
            unsafe { (std::slice::from_raw_parts(xyz, nv), std::slice::from_raw_parts(triangles, nt)) };
        let mesh = Mesh::from_flat(v, t)?;
        *o = Box::into_raw(Box::new(morphit_mesh { mesh: Arc::new(mesh) }));
        Ok(MORPHIT_OK)
    })
}

/// Fill `*out` with the mesh's properties.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_mesh_get_info(
    mesh: *const morphit_mesh,
    out_info: *mut morphit_mesh_info,
) -> morphit_status {
    guard(|| {
        let m = &unsafe { mesh_ref(mesh) }?.mesh;
        let o = unsafe { out(out_info, "out") }?;
        let (lo, hi) = m.bounds();
        let i = m.moment_inertia();
        let mut inertia = [0.0; 9];
        for r in 0..3 {
            for c in 0..3 {
                inertia[r * 3 + c] = i.col(c)[r];
            }
        }
        *o = morphit_mesh_info {
            num_vertices: m.vertices().len(),
            num_faces: m.faces().len(),
            volume: m.volume(),
            area: m.area(),
            scale: m.scale(),
            bounds_min: lo.to_array(),
            bounds_max: hi.to_array(),
            center_mass: m.center_mass().to_array(),
            inertia,
            winding_flipped: m.winding_flipped() as i32,
        };
        Ok(MORPHIT_OK)
    })
}

/// Point-in-mesh test for `num_points` xyz triples; `out` receives 1 (inside)
/// or 0 per point. Uses a ray direction that is robust for grid-aligned points.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_mesh_contains(
    mesh: *const morphit_mesh,
    xyz: *const f64,
    num_points: usize,
    out_inside: *mut u8,
) -> morphit_status {
    guard(|| {
        let m = &unsafe { mesh_ref(mesh) }?.mesh;
        if num_points == 0 {
            return Ok(MORPHIT_OK);
        }
        if xyz.is_null() {
            return Err(FfiError::null("xyz"));
        }
        if out_inside.is_null() {
            return Err(FfiError::null("out"));
        }
        let n3 = num_points.checked_mul(3).ok_or_else(|| FfiError::invalid("num_points too large"))?;
        // SAFETY: the caller provides arrays of the stated sizes.
        let pts = unsafe { std::slice::from_raw_parts(xyz, n3) };
        let pts: Vec<morphit::glam::DVec3> =
            pts.chunks_exact(3).map(|c| morphit::glam::DVec3::new(c[0], c[1], c[2])).collect();
        let res: Vec<u8> = m.contains_robust_many(&pts).into_iter().map(u8::from).collect();
        unsafe { std::ptr::copy_nonoverlapping(res.as_ptr(), out_inside, res.len()) };
        Ok(MORPHIT_OK)
    })
}

/// Release a mesh. NULL is ignored. Sessions created from it stay valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_mesh_free(mesh: *mut morphit_mesh) {
    if !mesh.is_null() {
        let _ = std::panic::catch_unwind(|| drop(unsafe { Box::from_raw(mesh) }));
    }
}

// ---------------------------------------------------------------------------
// Configs
// ---------------------------------------------------------------------------

fn new_config(config: Config) -> *mut morphit_config {
    Box::into_raw(Box::new(morphit_config { config: std::sync::Mutex::new(config) }))
}

/// Create a config from a preset name ("MorphIt-V", "MorphIt-S", "MorphIt-B",
/// "MorphIt-Obj", "MorphIt-Obj-mass"); NULL selects "MorphIt-B".
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_config_new(
    preset: *const c_char,
    out_config: *mut *mut morphit_config,
) -> morphit_status {
    guard(|| {
        let o = unsafe { out(out_config, "out") }?;
        *o = null_mut();
        let name = if preset.is_null() { "MorphIt-B" } else { unsafe { cstr(preset, "preset") }? };
        *o = new_config(Config::from_preset_name(name)?);
        Ok(MORPHIT_OK)
    })
}

/// Create a config from nested JSON in the result-file layout
/// (`{"model": {...}, "training": {...}, "random_seed": 1}`); missing fields
/// take their defaults.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_config_from_json(
    json: *const c_char,
    out_config: *mut *mut morphit_config,
) -> morphit_status {
    guard(|| {
        let o = unsafe { out(out_config, "out") }?;
        *o = null_mut();
        let json = unsafe { cstr(json, "json") }?;
        let config = Config::from_json_str(json)?;
        config.validate()?;
        *o = new_config(config);
        Ok(MORPHIT_OK)
    })
}

/// Copy a config.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_config_clone(
    config: *const morphit_config,
    out_config: *mut *mut morphit_config,
) -> morphit_status {
    guard(|| {
        let o = unsafe { out(out_config, "out") }?;
        *o = null_mut();
        let c = unsafe { config_ref(config) }?.lock().clone();
        *o = new_config(c);
        Ok(MORPHIT_OK)
    })
}

unsafe fn config_set(config: *const morphit_config, key: *const c_char, value: Value) -> FfiResult {
    let c = unsafe { config_ref(config) }?;
    let key = unsafe { cstr(key, "key") }?;
    c.lock().set(key, value)?;
    Ok(MORPHIT_OK)
}

/// Set a numeric value by dotted key, e.g. "training.center_lr". An integral
/// value may also be used for integer keys such as "model.num_spheres".
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_config_set_f64(
    config: *mut morphit_config,
    key: *const c_char,
    value: f64,
) -> morphit_status {
    guard(|| {
        if !value.is_finite() {
            return Err(FfiError::new(MORPHIT_ERR_CONFIG, "value must be finite"));
        }
        unsafe { config_set(config, key, Value::from(value)) }
    })
}

/// Set an integer value by dotted key, e.g. "model.num_spheres" or "random_seed".
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_config_set_i64(
    config: *mut morphit_config,
    key: *const c_char,
    value: i64,
) -> morphit_status {
    guard(|| unsafe { config_set(config, key, Value::from(value)) })
}

/// Set a boolean value (0 = false) by dotted key, e.g. "model.per_sphere_mass".
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_config_set_bool(
    config: *mut morphit_config,
    key: *const c_char,
    value: c_int,
) -> morphit_status {
    guard(|| unsafe { config_set(config, key, Value::from(value != 0)) })
}

/// Set a string value by dotted key.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_config_set_str(
    config: *mut morphit_config,
    key: *const c_char,
    value: *const c_char,
) -> morphit_status {
    guard(|| {
        let v = unsafe { cstr(value, "value") }?.to_string();
        unsafe { config_set(config, key, Value::from(v)) }
    })
}

/// Apply a JSON object of dotted-key updates, e.g.
/// `{"training.iterations": 100, "model.num_spheres": 20}`. All or nothing.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_config_set_json(
    config: *mut morphit_config,
    json_updates: *const c_char,
) -> morphit_status {
    guard(|| {
        let c = unsafe { config_ref(config) }?;
        let json = unsafe { cstr(json_updates, "json_updates") }?;
        c.lock().apply_updates_json(json)?;
        Ok(MORPHIT_OK)
    })
}

/// Read a numeric value by dotted key.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_config_get_f64(
    config: *const morphit_config,
    key: *const c_char,
    out_value: *mut f64,
) -> morphit_status {
    guard(|| {
        let c = unsafe { config_ref(config) }?;
        let key = unsafe { cstr(key, "key") }?;
        let o = unsafe { out(out_value, "out") }?;
        let v = c.lock().get(key)?;
        *o = v
            .as_f64()
            .ok_or_else(|| FfiError::new(MORPHIT_ERR_CONFIG, format!("`{key}` is not a number")))?;
        Ok(MORPHIT_OK)
    })
}

/// Read an integer (or boolean, as 0/1) value by dotted key.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_config_get_i64(
    config: *const morphit_config,
    key: *const c_char,
    out_value: *mut i64,
) -> morphit_status {
    guard(|| {
        let c = unsafe { config_ref(config) }?;
        let key = unsafe { cstr(key, "key") }?;
        let o = unsafe { out(out_value, "out") }?;
        let v = c.lock().get(key)?;
        *o = match v {
            Value::Bool(b) => b as i64,
            v => v
                .as_i64()
                .ok_or_else(|| FfiError::new(MORPHIT_ERR_CONFIG, format!("`{key}` is not an integer")))?,
        };
        Ok(MORPHIT_OK)
    })
}

/// Write the config as nested JSON (NUL-terminated) into `buf`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_config_to_json(
    config: *const morphit_config,
    buf: *mut c_char,
    capacity: usize,
    needed: *mut usize,
) -> morphit_status {
    guard(|| {
        let c = unsafe { config_ref(config) }?;
        let s = serde_json::to_string_pretty(&*c.lock()).expect("config serializes");
        unsafe { write_string(&s, buf, capacity, needed) }
    })
}

/// Release a config. NULL is ignored.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_config_free(config: *mut morphit_config) {
    if !config.is_null() {
        let _ = std::panic::catch_unwind(|| drop(unsafe { Box::from_raw(config) }));
    }
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

/// Create a session: validates the config, samples the mesh and initializes
/// the spheres (this can take a moment for large sample counts). The mesh and
/// config may be freed afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_session_new(
    mesh: *const morphit_mesh,
    config: *const morphit_config,
    out_session: *mut *mut morphit_session,
) -> morphit_status {
    guard(|| {
        let o = unsafe { out(out_session, "out") }?;
        *o = null_mut();
        let m = Arc::clone(&unsafe { mesh_ref(mesh) }?.mesh);
        let c = unsafe { config_ref(config) }?.lock().clone();
        let session = Session::new(c, m)?;
        *o = Box::into_raw(Box::new(morphit_session::new(session)));
        Ok(MORPHIT_OK)
    })
}

/// Run one iteration. Returns `MORPHIT_OK` after running it (with
/// `info->done` set on the last one) or `MORPHIT_DONE` if no iterations remain.
/// `info` may be NULL. Step-by-step callers finish with `morphit_finalize`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_step(
    session: *mut morphit_session,
    info: *mut morphit_step_info,
) -> morphit_status {
    guard(|| {
        let s = unsafe { session_ref(session) }?;
        if s.is_running() {
            return Err(busy());
        }
        let step = {
            let mut sess = s.lock()?;
            if sess.is_done() {
                return Ok(MORPHIT_DONE);
            }
            let step = sess.step()?;
            s.publish(&sess);
            step
        };
        if let Some(o) = unsafe { info.as_mut() } {
            *o = morphit_step_info::from(&step);
        }
        Ok(MORPHIT_OK)
    })
}

/// Run all remaining iterations, then finalize (remove spheres whose centers
/// escaped the mesh), like Python's `train()`. `progress` (may be NULL) is
/// called after every iteration without any lock held; it may call the read
/// functions and `morphit_cancel` on this session. Returns `MORPHIT_OK`, or
/// `MORPHIT_ERR_CANCELLED` if cancelled (the session is then not finalized and
/// can be resumed with another `morphit_run`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_run(
    session: *mut morphit_session,
    progress: morphit_progress_fn,
    user_data: *mut c_void,
) -> morphit_status {
    guard(|| {
        let s = unsafe { session_ref(session) }?;
        if s.running.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_err() {
            return Err(busy());
        }
        let _running = RunningGuard(&s.running);
        let result = (|| {
            loop {
                if s.cancel.swap(false, Ordering::AcqRel) {
                    return Err(FfiError::new(MORPHIT_ERR_CANCELLED, "cancelled by morphit_cancel"));
                }
                let step = {
                    let mut sess = s.lock()?;
                    if sess.is_done() {
                        break;
                    }
                    let step = sess.step()?;
                    s.publish(&sess);
                    step
                };
                if let Some(cb) = progress {
                    let c = morphit_step_info::from(&step);
                    // SAFETY: host-provided callback; `c` outlives the call.
                    if unsafe { cb(&c, user_data) } != 0 {
                        return Err(FfiError::new(
                            MORPHIT_ERR_CANCELLED,
                            "cancelled by the progress callback",
                        ));
                    }
                }
            }
            let mut sess = s.lock()?;
            sess.finalize();
            s.publish(&sess);
            Ok(MORPHIT_OK)
        })();
        // Cancels that arrived during this run are consumed by it.
        s.cancel.store(false, Ordering::Release);
        result
    })
}

/// Ask the active `morphit_run` on this session (or the next one, if none is
/// active) to stop before its next iteration. Never blocks.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_cancel(session: *mut morphit_session) -> morphit_status {
    guard(|| {
        let s = unsafe { session_ref(session) }?;
        s.cancel.store(true, Ordering::Release);
        Ok(MORPHIT_OK)
    })
}

/// Remove spheres whose centers ended outside the mesh and end the session.
/// Idempotent. `pruned` (may be NULL) receives the number removed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_finalize(
    session: *mut morphit_session,
    pruned: *mut usize,
) -> morphit_status {
    guard(|| {
        let s = unsafe { session_ref(session) }?;
        if s.is_running() {
            return Err(busy());
        }
        let n = {
            let mut sess = s.lock()?;
            let n = sess.finalize();
            s.publish(&sess);
            n
        };
        if let Some(p) = unsafe { pruned.as_mut() } {
            *p = n;
        }
        Ok(MORPHIT_OK)
    })
}

/// Release a session. NULL is ignored. Returns `MORPHIT_ERR_BUSY` (and does
/// not free) while `morphit_run` is active on it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_session_free(session: *mut morphit_session) -> morphit_status {
    guard(|| {
        if session.is_null() {
            return Ok(MORPHIT_OK);
        }
        let s = unsafe { session_ref(session) }?;
        if s.is_running() {
            return Err(busy());
        }
        drop(unsafe { Box::from_raw(session) });
        Ok(MORPHIT_OK)
    })
}

// ---------------------------------------------------------------------------
// Reading results (never blocks on a running iteration)
// ---------------------------------------------------------------------------

/// Progress and status of a session.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_state(
    session: *const morphit_session,
    out_state: *mut morphit_state_info,
) -> morphit_status {
    guard(|| {
        let s = unsafe { session_ref(session) }?;
        let o = unsafe { out(out_state, "out") }?;
        let snap = s.snapshot();
        *o = morphit_state_info {
            state: snap.state.into(),
            iteration: snap.iteration as u64,
            total_iterations: snap.total_iterations as u64,
            num_spheres: snap.radii.len(),
            total_loss: snap.total_loss,
            running: s.is_running() as i32,
            density_control_passes: snap.density_control_passes as u64,
            pruned: snap.pruned,
            seed: snap.seed,
        };
        Ok(MORPHIT_OK)
    })
}

/// Current number of spheres.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_sphere_count(
    session: *const morphit_session,
    count: *mut usize,
) -> morphit_status {
    guard(|| {
        let s = unsafe { session_ref(session) }?;
        *unsafe { out(count, "count") }? = s.snapshot().radii.len();
        Ok(MORPHIT_OK)
    })
}

/// Consistent copy of all sphere data from one moment: `centers` receives 3
/// doubles per sphere, `radii` and `masses` one each. Any of the three may be
/// NULL to skip it; `capacity` is in spheres; `count` receives the number of
/// spheres. All three NULL with capacity 0 is a size query.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_get_spheres(
    session: *const morphit_session,
    centers: *mut f64,
    radii: *mut f64,
    masses: *mut f64,
    capacity: usize,
    count: *mut usize,
) -> morphit_status {
    guard(|| {
        let s = unsafe { session_ref(session) }?;
        let snap = s.snapshot();
        let n = snap.radii.len();
        if let Some(c) = unsafe { count.as_mut() } {
            *c = n;
        }
        if centers.is_null() && radii.is_null() && masses.is_null() {
            return Ok(MORPHIT_OK);
        }
        if capacity < n {
            return Err(FfiError::new(
                MORPHIT_ERR_BUFFER_TOO_SMALL,
                format!("capacity {capacity} spheres, {n} needed"),
            ));
        }
        unsafe {
            if !centers.is_null() {
                write_slice(&snap.centers, centers, capacity * 3, null_mut())?;
            }
            if !radii.is_null() {
                write_slice(&snap.radii, radii, capacity, null_mut())?;
            }
            if !masses.is_null() {
                write_slice(&snap.masses, masses, capacity, null_mut())?;
            }
        }
        Ok(MORPHIT_OK)
    })
}

/// Sphere centers as xyz triples; `capacity` and `*written` count doubles (3 per sphere).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_get_centers(
    session: *const morphit_session,
    buf: *mut f64,
    capacity: usize,
    written: *mut usize,
) -> morphit_status {
    guard(|| {
        let s = unsafe { session_ref(session) }?;
        unsafe { write_slice(&s.snapshot().centers, buf, capacity, written) }
    })
}

/// Sphere radii; `capacity` and `*written` count doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_get_radii(
    session: *const morphit_session,
    buf: *mut f64,
    capacity: usize,
    written: *mut usize,
) -> morphit_status {
    guard(|| {
        let s = unsafe { session_ref(session) }?;
        unsafe { write_slice(&s.snapshot().radii, buf, capacity, written) }
    })
}

/// Sphere masses (learned, or density x volume); `capacity` and `*written` count doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_get_masses(
    session: *const morphit_session,
    buf: *mut f64,
    capacity: usize,
    written: *mut usize,
) -> morphit_status {
    guard(|| {
        let s = unsafe { session_ref(session) }?;
        unsafe { write_slice(&s.snapshot().masses, buf, capacity, written) }
    })
}

/// Details of the most recent iteration; `MORPHIT_ERR_STATE` before the first.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_get_last_step(
    session: *const morphit_session,
    info: *mut morphit_step_info,
) -> morphit_status {
    guard(|| {
        let s = unsafe { session_ref(session) }?;
        let o = unsafe { out(info, "info") }?;
        *o = s
            .snapshot()
            .last_step
            .ok_or_else(|| FfiError::new(MORPHIT_ERR_STATE, "no iteration has run yet"))?;
        Ok(MORPHIT_OK)
    })
}

/// Current spheres as JSON in the Python MorphIt result schema
/// (`centers`, `radii`, `masses`, `mesh_path`, `num_spheres`, `per_sphere_mass`,
/// `mesh_prep`, `config`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_result_json(
    session: *const morphit_session,
    buf: *mut c_char,
    capacity: usize,
    needed: *mut usize,
) -> morphit_status {
    guard(|| {
        let s = unsafe { session_ref(session) }?;
        let json = s.snapshot().result().to_json_string();
        unsafe { write_string(&json, buf, capacity, needed) }
    })
}

/// Write the current result JSON to a file (UTF-8 path; parent directories are created).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_result_save(
    session: *const morphit_session,
    path: *const c_char,
) -> morphit_status {
    guard(|| {
        let s = unsafe { session_ref(session) }?;
        let path = unsafe { cstr(path, "path") }?;
        s.snapshot().result().save(path)?;
        Ok(MORPHIT_OK)
    })
}

/// The session's effective config as nested JSON (includes the mesh path).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_session_config_json(
    session: *const morphit_session,
    buf: *mut c_char,
    capacity: usize,
    needed: *mut usize,
) -> morphit_status {
    guard(|| {
        let s = unsafe { session_ref(session) }?;
        let json = serde_json::to_string_pretty(&*s.snapshot().config).expect("config serializes");
        unsafe { write_string(&json, buf, capacity, needed) }
    })
}

/// The device the session's distance searches run on: `"cpu"` or
/// `"gpu:N <adapter name>"`. Never blocks.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_session_device(
    session: *const morphit_session,
    buf: *mut c_char,
    capacity: usize,
    needed: *mut usize,
) -> morphit_status {
    guard(|| {
        let s = unsafe { session_ref(session) }?;
        unsafe { write_string(&s.device, buf, capacity, needed) }
    })
}

/// What mesh preparation did when the session was created, as JSON with the
/// keys of Python's `MeshPrepReport` (`action` is `unchanged`, `unioned`,
/// `skipped` or `disabled`; see `model.union_overlapping_bodies`). Never blocks.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_session_mesh_prep_json(
    session: *const morphit_session,
    buf: *mut c_char,
    capacity: usize,
    needed: *mut usize,
) -> morphit_status {
    guard(|| {
        let s = unsafe { session_ref(session) }?;
        let json = serde_json::to_string_pretty(&*s.mesh_prep).expect("report serializes");
        unsafe { write_string(&json, buf, capacity, needed) }
    })
}

/// Per-iteration training history as JSON (Python `ConvergenceTracker` layout).
/// Unlike the other read functions this waits for a running iteration to
/// finish, and the size may grow between a size query and the copy.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_history_json(
    session: *const morphit_session,
    buf: *mut c_char,
    capacity: usize,
    needed: *mut usize,
) -> morphit_status {
    guard(|| {
        let s = unsafe { session_ref(session) }?;
        let json = serde_json::to_string(&s.lock()?.history().to_json()).expect("history serializes");
        unsafe { write_string(&json, buf, capacity, needed) }
    })
}

/// Internal: panics inside the API guard (tests only).
/// cbindgen:ignore
#[cfg(feature = "test-hooks")]
#[unsafe(no_mangle)]
pub extern "C" fn morphit__test_panic() -> morphit_status {
    guard(|| panic!("deliberate test panic"))
}
