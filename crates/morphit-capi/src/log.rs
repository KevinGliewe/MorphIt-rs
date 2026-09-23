//! Optional forwarding of the library's `tracing` events to a C callback.
//! Nothing is printed unless a callback is installed.

use std::ffi::{CString, c_char, c_int, c_void};
use std::fmt::Write as _;
use std::sync::{OnceLock, RwLock};

use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};

/// Receives one formatted log line. `level`: 1 = error, 2 = warn, 3 = info,
/// 4 = debug, 5 = trace. `message` is only valid during the call.
#[allow(non_camel_case_types)]
pub type morphit_log_fn =
    Option<unsafe extern "C" fn(level: c_int, message: *const c_char, user_data: *mut c_void)>;

#[derive(Clone, Copy)]
struct Sink {
    callback: unsafe extern "C" fn(c_int, *const c_char, *mut c_void),
    user_data: usize,
    max_level: c_int,
}

static SINK: RwLock<Option<Sink>> = RwLock::new(None);
static INSTALLED: OnceLock<bool> = OnceLock::new();

fn level_number(l: &Level) -> c_int {
    match *l {
        Level::ERROR => 1,
        Level::WARN => 2,
        Level::INFO => 3,
        Level::DEBUG => 4,
        Level::TRACE => 5,
    }
}

#[derive(Default)]
struct Line {
    message: String,
    fields: String,
}

impl Visit for Line {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message = value.to_string();
        } else {
            let _ = write!(self.fields, " {}={}", field.name(), value);
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.message = format!("{value:?}");
        } else {
            let _ = write!(self.fields, " {}={:?}", field.name(), value);
        }
    }
}

struct CallbackLayer;

impl<S: Subscriber> Layer<S> for CallbackLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        // Copy the sink out so the callback runs without holding the lock.
        let sink = match SINK.read() {
            Ok(g) => *g,
            Err(p) => *p.into_inner(),
        };
        let Some(sink) = sink else { return };
        let level = level_number(event.metadata().level());
        if level > sink.max_level {
            return;
        }
        let mut line = Line::default();
        event.record(&mut line);
        let text =
            CString::new(format!("{}{}", line.message, line.fields).replace('\0', " ")).unwrap_or_default();
        // SAFETY: the host promised a valid callback when installing it.
        unsafe { (sink.callback)(level, text.as_ptr(), sink.user_data as *mut c_void) };
    }
}

/// Install, replace or (with `None`) remove the log callback.
/// Returns false if another global tracing subscriber already owns the process.
pub(crate) fn set_callback(callback: morphit_log_fn, user_data: *mut c_void, max_level: c_int) -> bool {
    let new = callback.map(|cb| Sink { callback: cb, user_data: user_data as usize, max_level });
    match SINK.write() {
        Ok(mut g) => *g = new,
        Err(p) => *p.into_inner() = new,
    }
    if callback.is_none() {
        return true;
    }
    *INSTALLED.get_or_init(|| {
        tracing::subscriber::set_global_default(tracing_subscriber::registry().with(CallbackLayer)).is_ok()
    })
}
