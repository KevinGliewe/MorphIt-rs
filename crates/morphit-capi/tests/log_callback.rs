//! The log callback installs a process-wide subscriber, so it lives in its own
//! test binary.

mod common;

use std::ffi::{CStr, c_char, c_int, c_void};
use std::ptr::null_mut;
use std::sync::Mutex;

use common::*;
use morphit_capi::*;

static LINES: Mutex<Vec<(c_int, String)>> = Mutex::new(Vec::new());

unsafe extern "C" fn collect(level: c_int, message: *const c_char, _: *mut c_void) {
    let text = unsafe { CStr::from_ptr(message) }.to_string_lossy().into_owned();
    LINES.lock().unwrap().push((level, text));
}

#[test]
fn log_messages_reach_the_callback() {
    ok(unsafe { morphit_set_log_callback(Some(collect), null_mut(), 3) });
    let mesh = box_mesh(1.0, 1.0, 1.0);
    let cfg = small_config("MorphIt-B", 6, 25, 4);
    let s = new_session(mesh, cfg);
    ok(unsafe { morphit_run(s, None, null_mut()) });
    let lines = LINES.lock().unwrap().clone();
    assert!(
        lines.iter().any(|(lvl, m)| *lvl == 3
            && m.contains("density control re-packed spheres")
            && m.contains("culled=")),
        "{lines:?}"
    );
    assert!(lines.iter().all(|(lvl, _)| *lvl <= 3), "debug messages are filtered out");

    // Removing the callback stops forwarding.
    ok(unsafe { morphit_set_log_callback(None, null_mut(), 0) });
    let before = LINES.lock().unwrap().len();
    let s2 = new_session(mesh, cfg);
    ok(unsafe { morphit_run(s2, None, null_mut()) });
    assert_eq!(LINES.lock().unwrap().len(), before);
    unsafe {
        ok(morphit_session_free(s));
        ok(morphit_session_free(s2));
        morphit_mesh_free(mesh);
        morphit_config_free(cfg);
    }
}
