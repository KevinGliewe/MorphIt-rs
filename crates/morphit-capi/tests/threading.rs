//! Concurrency guarantees of the C API.

mod common;

use std::ffi::{c_int, c_void};
use std::ptr::null_mut;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use common::*;
use morphit_capi::morphit_status::*;
use morphit_capi::*;

fn run_to_end(mesh: *const morphit_mesh, seed: i64) -> (Vec<f64>, Vec<f64>) {
    let cfg = small_config("MorphIt-B", 10, 60, seed);
    let s = new_session(mesh, cfg);
    unsafe { morphit_config_free(cfg) };
    ok(unsafe { morphit_run(s, None, null_mut()) });
    let (c, r, _) = spheres(s);
    ok(unsafe { morphit_session_free(s) });
    (c, r)
}

#[test]
fn parallel_sessions_on_a_shared_mesh_match_sequential_runs() {
    let mesh = box_mesh(1.0, 0.7, 0.5);
    let seeds = [11, 12, 13, 14];
    let sequential: Vec<_> = seeds.iter().map(|&seed| run_to_end(mesh, seed)).collect();
    let m = SendPtr(mesh);
    let handles: Vec<_> = seeds
        .iter()
        .map(|&seed| {
            std::thread::spawn(move || {
                let m = m;
                run_to_end(m.0, seed)
            })
        })
        .collect();
    let parallel: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(parallel, sequential, "concurrent sessions are independent and deterministic");
    unsafe { morphit_mesh_free(mesh) };
}

/// Long-running session for cancellation tests.
fn long_session(mesh: *const morphit_mesh) -> *mut morphit_session {
    let cfg = small_config("MorphIt-B", 8, 1_000_000, 5);
    let s = new_session(mesh, cfg);
    unsafe { morphit_config_free(cfg) };
    s
}

fn spawn_run(s: *mut morphit_session) -> std::thread::JoinHandle<morphit_status> {
    let p = SendPtr(s);
    std::thread::spawn(move || {
        let p = p;
        unsafe { morphit_run(p.0, None, null_mut()) }
    })
}

fn wait_for(s: *const morphit_session, pred: impl Fn(&morphit_state_info) -> bool) {
    let start = Instant::now();
    while !pred(&state(s)) {
        assert!(start.elapsed() < Duration::from_secs(60), "timed out");
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn readers_poll_while_running_and_cancel_stops_the_run() {
    let mesh = box_mesh(1.0, 1.0, 1.0);
    let s = long_session(mesh);
    let worker = spawn_run(s);
    wait_for(s, |st| st.running == 1 && st.iteration >= 3);

    let mut slowest = Duration::ZERO;
    let mut last_iter = 0;
    for _ in 0..200 {
        let t = Instant::now();
        let st = state(s);
        let (c, r, m) = spheres(s);
        slowest = slowest.max(t.elapsed());
        assert_eq!(c.len(), 3 * r.len());
        assert_eq!(m.len(), r.len());
        assert!(r.iter().all(|x| x.is_finite() && *x > 0.0));
        assert!(st.iteration >= last_iter, "progress is monotonic");
        last_iter = st.iteration;
    }
    assert!(slowest < Duration::from_millis(250), "readers must not wait for iterations: {slowest:?}");

    ok(unsafe { morphit_cancel(s) });
    assert_eq!(worker.join().unwrap(), MORPHIT_ERR_CANCELLED);
    let st = state(s);
    assert_eq!(st.running, 0);
    assert_eq!(st.state, morphit_session_state::MORPHIT_STATE_RUNNING);
    assert!(st.iteration < st.total_iterations);
    ok(unsafe { morphit_finalize(s, null_mut()) });
    assert_eq!(state(s).state, morphit_session_state::MORPHIT_STATE_FINALIZED);
    ok(unsafe { morphit_session_free(s) });
    unsafe { morphit_mesh_free(mesh) };
}

#[test]
fn conflicting_calls_during_a_run_are_busy() {
    let mesh = box_mesh(1.0, 1.0, 1.0);
    let s = long_session(mesh);
    let worker = spawn_run(s);
    wait_for(s, |st| st.running == 1);
    assert_eq!(unsafe { morphit_step(s, null_mut()) }, MORPHIT_ERR_BUSY);
    assert_eq!(unsafe { morphit_run(s, None, null_mut()) }, MORPHIT_ERR_BUSY);
    assert_eq!(unsafe { morphit_finalize(s, null_mut()) }, MORPHIT_ERR_BUSY);
    assert_eq!(unsafe { morphit_session_free(s) }, MORPHIT_ERR_BUSY);
    ok(unsafe { morphit_cancel(s) });
    assert_eq!(worker.join().unwrap(), MORPHIT_ERR_CANCELLED);
    ok(unsafe { morphit_session_free(s) });
    unsafe { morphit_mesh_free(mesh) };
}

#[test]
fn cancel_before_run_stops_it_before_the_first_iteration() {
    let mesh = box_mesh(1.0, 1.0, 1.0);
    let s = long_session(mesh);
    ok(unsafe { morphit_cancel(s) });
    assert_eq!(unsafe { morphit_run(s, None, null_mut()) }, MORPHIT_ERR_CANCELLED);
    assert_eq!(state(s).iteration, 0);
    // The cancel was consumed: a step works again.
    ok(unsafe { morphit_step(s, null_mut()) });
    ok(unsafe { morphit_session_free(s) });
    unsafe { morphit_mesh_free(mesh) };
}

struct CallbackCtx {
    session: *mut morphit_session,
    calls: AtomicU64,
    busy_seen: AtomicU64,
}

unsafe extern "C" fn reentrant(info: *const morphit_step_info, ud: *mut c_void) -> c_int {
    let ctx = unsafe { &*(ud as *const CallbackCtx) };
    let n = ctx.calls.fetch_add(1, Ordering::SeqCst) + 1;
    // Reads from inside the callback see this iteration's state.
    let st = state(ctx.session);
    assert_eq!(st.iteration, unsafe { (*info).iteration } + 1);
    // Mutating calls are refused instead of deadlocking.
    if unsafe { morphit_step(ctx.session, null_mut()) } == MORPHIT_ERR_BUSY {
        ctx.busy_seen.fetch_add(1, Ordering::SeqCst);
    }
    if n == 7 {
        unsafe { morphit_cancel(ctx.session) };
    }
    0
}

#[test]
fn callback_can_read_and_cancel_its_own_session() {
    let mesh = box_mesh(1.0, 1.0, 1.0);
    let s = long_session(mesh);
    let ctx = CallbackCtx { session: s, calls: AtomicU64::new(0), busy_seen: AtomicU64::new(0) };
    let r = unsafe { morphit_run(s, Some(reentrant), &ctx as *const _ as *mut c_void) };
    assert_eq!(r, MORPHIT_ERR_CANCELLED);
    assert_eq!(ctx.calls.load(Ordering::SeqCst), 7);
    assert_eq!(ctx.busy_seen.load(Ordering::SeqCst), 7);
    assert_eq!(state(s).iteration, 7);
    ok(unsafe { morphit_session_free(s) });
    unsafe { morphit_mesh_free(mesh) };
}

#[test]
fn concurrent_config_edits_are_safe() {
    let cfg = small_config("MorphIt-B", 5, 5, 1);
    let p = SendPtr(cfg);
    let workers: Vec<_> = (0..8)
        .map(|t| {
            std::thread::spawn(move || {
                let p = p;
                let key = c(if t % 2 == 0 { "training.center_lr" } else { "model.num_spheres" });
                for i in 0..300 {
                    let st = if t % 2 == 0 {
                        unsafe { morphit_config_set_f64(p.0, key.as_ptr(), 0.001 * (1 + i % 5) as f64) }
                    } else {
                        unsafe { morphit_config_set_i64(p.0, key.as_ptr(), 1 + (i % 30) as i64) }
                    };
                    assert_eq!(st, OK);
                    let mut v = 0.0;
                    assert_eq!(unsafe { morphit_config_get_f64(p.0, key.as_ptr(), &mut v) }, OK);
                    assert!(v > 0.0);
                }
            })
        })
        .collect();
    for w in workers {
        w.join().unwrap();
    }
    unsafe { morphit_config_free(cfg) };
}

#[test]
fn error_messages_are_per_thread() {
    let t = std::thread::spawn(|| {
        let mut m = null_mut();
        let p = c("nowhere/missing.obj");
        assert_eq!(unsafe { morphit_mesh_load(p.as_ptr(), &mut m) }, MORPHIT_ERR_IO);
        last_error()
    });
    let msg = t.join().unwrap();
    assert!(msg.contains("missing.obj"));
    // This thread never failed, so it sees no message.
    let mut cfg = null_mut();
    ok(unsafe { morphit_config_new(std::ptr::null(), &mut cfg) });
    assert_eq!(last_error(), "");
    unsafe { morphit_config_free(cfg) };
}

/// Like `run_to_end` but on the requested device; `None` when no GPU is usable.
fn run_on(mesh: *const morphit_mesh, seed: i64, device: &str) -> Option<(Vec<f64>, Vec<f64>)> {
    let cfg = small_config("MorphIt-B", 10, 60, seed);
    let k = c("model.device");
    let v = c(device);
    ok(unsafe { morphit_config_set_str(cfg, k.as_ptr(), v.as_ptr()) });
    let mut s = null_mut();
    let st = unsafe { morphit_session_new(mesh, cfg, &mut s) };
    unsafe { morphit_config_free(cfg) };
    if st != OK {
        eprintln!("no GPU ({}); skipping", last_error());
        return None;
    }
    let dev = string_out(|b, cap, n| unsafe { morphit_session_device(s, b, cap, n) });
    assert!(dev.starts_with(device), "session device {dev} for {device}");
    ok(unsafe { morphit_run(s, None, null_mut()) });
    let (c, r, _) = spheres(s);
    ok(unsafe { morphit_session_free(s) });
    Some((c, r))
}

#[test]
fn parallel_gpu_sessions_match_cpu_runs() {
    let mesh = box_mesh(1.0, 0.7, 0.5);
    let seeds = [21, 22, 23, 24];
    if run_on(mesh, seeds[0], "gpu").is_none() {
        unsafe { morphit_mesh_free(mesh) };
        return;
    }
    let cpu: Vec<_> = seeds.iter().map(|&seed| run_on(mesh, seed, "cpu").unwrap()).collect();
    let m = SendPtr(mesh);
    let handles: Vec<_> = seeds
        .iter()
        .map(|&seed| {
            std::thread::spawn(move || {
                let m = m;
                run_on(m.0, seed, "gpu").unwrap()
            })
        })
        .collect();
    let gpu: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(gpu, cpu, "GPU sessions running concurrently equal the CPU results bit for bit");
    unsafe { morphit_mesh_free(mesh) };
}
