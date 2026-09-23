//! Status codes, the thread-local error message, the panic guard, and helpers
//! for copying results into caller-provided buffers.

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};

/// Result of every fallible call. Negative values are errors; call
/// `morphit_last_error()` on the same thread for a message.
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum morphit_status {
    /// Success.
    MORPHIT_OK = 0,
    /// `morphit_step`: the session has no iterations left; nothing was run.
    MORPHIT_DONE = 1,
    /// A required pointer argument was NULL.
    MORPHIT_ERR_NULL_ARG = -1,
    /// An argument was invalid (bad UTF-8, zero count, out-of-range index, ...).
    MORPHIT_ERR_INVALID_ARG = -2,
    /// A file could not be read or written.
    MORPHIT_ERR_IO = -3,
    /// The mesh could not be parsed or is unusable.
    MORPHIT_ERR_MESH = -4,
    /// Unknown preset or key, wrong value type, or invalid value.
    MORPHIT_ERR_CONFIG = -5,
    /// Operation not valid in the session's current state.
    MORPHIT_ERR_STATE = -6,
    /// `morphit_run` is active on this session (from another thread or from
    /// inside its own progress callback).
    MORPHIT_ERR_BUSY = -7,
    /// The run was cancelled by `morphit_cancel` or the progress callback.
    MORPHIT_ERR_CANCELLED = -8,
    /// The output buffer is too small; the required size was reported.
    MORPHIT_ERR_BUFFER_TOO_SMALL = -9,
    /// An internal error (Rust panic) was caught at the API boundary.
    MORPHIT_ERR_PANIC = -10,
    /// The worker thread pool is already running and cannot be reconfigured.
    MORPHIT_ERR_THREADPOOL = -11,
}

use morphit_status::*;

/// Error carried from an API function body to the guard.
pub(crate) struct FfiError {
    pub status: morphit_status,
    pub message: String,
}

impl FfiError {
    pub fn new(status: morphit_status, message: impl Into<String>) -> Self {
        FfiError { status, message: message.into() }
    }

    pub fn null(what: &str) -> Self {
        FfiError::new(MORPHIT_ERR_NULL_ARG, format!("`{what}` must not be NULL"))
    }

    pub fn invalid(message: impl Into<String>) -> Self {
        FfiError::new(MORPHIT_ERR_INVALID_ARG, message)
    }
}

impl From<morphit::Error> for FfiError {
    fn from(e: morphit::Error) -> Self {
        let status = match &e {
            morphit::Error::Io(_) => MORPHIT_ERR_IO,
            morphit::Error::Mesh(_) => MORPHIT_ERR_MESH,
            morphit::Error::Config { .. } => MORPHIT_ERR_CONFIG,
            morphit::Error::State(_) => MORPHIT_ERR_STATE,
            morphit::Error::Cancelled => MORPHIT_ERR_CANCELLED,
        };
        FfiError::new(status, e.to_string())
    }
}

pub(crate) type FfiResult<T = morphit_status> = Result<T, FfiError>;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_last_error(message: &str) {
    let c = CString::new(message.replace('\0', "\u{FFFD}")).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = c);
}

fn clear_last_error() {
    LAST_ERROR.with(|e| {
        let mut e = e.borrow_mut();
        if !e.as_bytes().is_empty() {
            *e = CString::default();
        }
    });
}

/// Pointer to this thread's last error message ("" after a successful call).
pub(crate) fn last_error_ptr() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

/// Run an API body: convert errors to status codes, record the message, and
/// turn panics into `MORPHIT_ERR_PANIC` instead of unwinding into C.
pub(crate) fn guard(body: impl FnOnce() -> FfiResult) -> morphit_status {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(status)) => {
            clear_last_error();
            status
        }
        Ok(Err(e)) => {
            set_last_error(&e.message);
            e.status
        }
        Err(payload) => {
            let msg = payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".into());
            set_last_error(&format!("internal error: {msg}"));
            MORPHIT_ERR_PANIC
        }
    }
}

/// Borrow a NUL-terminated UTF-8 string argument.
///
/// # Safety
/// `p` must be NULL or point to a NUL-terminated string valid for the call.
pub(crate) unsafe fn cstr<'a>(p: *const c_char, what: &str) -> FfiResult<&'a str> {
    if p.is_null() {
        return Err(FfiError::null(what));
    }
    // SAFETY: caller guarantees a valid NUL-terminated string.
    unsafe { CStr::from_ptr(p) }
        .to_str()
        .map_err(|_| FfiError::invalid(format!("`{what}` is not valid UTF-8")))
}

/// Mutable reference to a required output argument.
///
/// # Safety
/// `p` must be NULL or valid for writes of `T`.
pub(crate) unsafe fn out<'a, T>(p: *mut T, what: &str) -> FfiResult<&'a mut T> {
    // SAFETY: caller guarantees validity when non-null.
    unsafe { p.as_mut() }.ok_or_else(|| FfiError::null(what))
}

/// Copy `data` into `buf` (capacity `cap` elements), reporting the required
/// count through `count`. `buf == NULL && cap == 0` is a size query.
///
/// # Safety
/// `buf` must be NULL or valid for `cap` writes; `count` NULL or valid.
pub(crate) unsafe fn write_slice<T: Copy>(
    data: &[T],
    buf: *mut T,
    cap: usize,
    count: *mut usize,
) -> FfiResult {
    if let Some(c) = unsafe { count.as_mut() } {
        *c = data.len();
    }
    if buf.is_null() {
        return if cap == 0 { Ok(MORPHIT_OK) } else { Err(FfiError::null("buffer")) };
    }
    if cap < data.len() {
        return Err(FfiError::new(
            MORPHIT_ERR_BUFFER_TOO_SMALL,
            format!("buffer holds {cap} elements, {} needed", data.len()),
        ));
    }
    // SAFETY: buf is valid for `cap >= data.len()` writes and does not overlap `data`.
    unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), buf, data.len()) };
    Ok(MORPHIT_OK)
}

/// Copy a string plus its NUL terminator into `buf`; `needed` receives the
/// size in bytes including the terminator. `buf == NULL && cap == 0` is a size query.
///
/// # Safety
/// As for [`write_slice`].
pub(crate) unsafe fn write_string(s: &str, buf: *mut c_char, cap: usize, needed: *mut usize) -> FfiResult {
    let mut bytes = Vec::with_capacity(s.len() + 1);
    bytes.extend(s.bytes().map(|b| b as c_char));
    bytes.push(0);
    unsafe { write_slice(&bytes, buf, cap, needed) }
}
