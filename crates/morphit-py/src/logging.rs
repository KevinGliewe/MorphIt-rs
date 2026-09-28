//! Library log messages for Python's `logging`.
//!
//! A `tracing` layer queues records; Python drains the queue from its own
//! thread (`morphit_rs.enable_logging`). Worker threads never touch the GIL,
//! so logging cannot deadlock against a Python thread waiting for them.

use std::collections::VecDeque;
use std::fmt::Write as _;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use pyo3::prelude::*;
use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::prelude::*;

/// Oldest records are dropped beyond this many undrained ones.
const CAPACITY: usize = 10_000;

static QUEUE: Mutex<VecDeque<(u8, String, String)>> = Mutex::new(VecDeque::new());
static INSTALLED: AtomicBool = AtomicBool::new(false);

/// The event's message, then its fields as `key=value`.
#[derive(Default)]
struct Message {
    text: String,
    fields: String,
}

impl Visit for Message {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            let _ = write!(self.text, "{value:?}");
        } else {
            let _ = write!(self.fields, " {}={value:?}", field.name());
        }
    }
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.text.push_str(value);
        } else {
            let _ = write!(self.fields, " {}={value}", field.name());
        }
    }
}

struct QueueLayer;

impl<S: Subscriber> Layer<S> for QueueLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let mut msg = Message::default();
        event.record(&mut msg);
        let text = format!("{}{}", msg.text, msg.fields);
        // Python logging levels.
        let level = match *event.metadata().level() {
            Level::ERROR => 40,
            Level::WARN => 30,
            Level::INFO => 20,
            Level::DEBUG | Level::TRACE => 10,
        };
        let mut q = QUEUE.lock().unwrap_or_else(|p| p.into_inner());
        if q.len() >= CAPACITY {
            q.pop_front();
        }
        q.push_back((level, event.metadata().target().to_string(), text));
    }
}

/// Start queueing library records at `level` and above (`"DEBUG"`, `"INFO"`,
/// `"WARNING"`, `"ERROR"`). Only the first call installs the collector.
#[pyfunction]
#[pyo3(signature = (level = "INFO"))]
pub(crate) fn _install_logging(level: &str) -> PyResult<bool> {
    let filter = match level.to_ascii_uppercase().as_str() {
        "DEBUG" | "TRACE" => LevelFilter::DEBUG,
        "INFO" => LevelFilter::INFO,
        "WARNING" | "WARN" => LevelFilter::WARN,
        "ERROR" | "CRITICAL" => LevelFilter::ERROR,
        other => return Err(crate::errors::value(format!("unknown log level {other:?}"))),
    };
    if INSTALLED.swap(true, Ordering::AcqRel) {
        return Ok(false);
    }
    // wgpu reports unrelated system layers as errors; keep only this library's records.
    let targets = tracing_subscriber::filter::Targets::new()
        .with_target("morphit", filter)
        .with_target("morphit_robot", filter);
    let installed = tracing_subscriber::registry().with(QueueLayer.with_filter(targets)).try_init().is_ok();
    Ok(installed)
}

/// Take the queued records as `(level, target, message)`.
#[pyfunction]
pub(crate) fn _drain_logs() -> Vec<(u8, String, String)> {
    QUEUE.lock().unwrap_or_else(|p| p.into_inner()).drain(..).collect()
}
