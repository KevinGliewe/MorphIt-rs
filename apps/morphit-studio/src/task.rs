//! Background work that runs the same code natively and in the browser.
//!
//! Natively a task gets its own thread (the optimizer uses rayon inside it).
//! In the browser everything shares the page's thread: a task runs as a
//! local future and gives the event loop back through [`Yielder`], so Bevy
//! keeps drawing frames while the optimizer works.

use std::future::Future;
use std::sync::{Arc, Mutex};

/// A result that a task delivers and a Bevy system picks up.
pub struct Slot<T>(Arc<Mutex<Option<T>>>);

impl<T> Clone for Slot<T> {
    fn clone(&self) -> Self {
        Slot(self.0.clone())
    }
}

impl<T> Default for Slot<T> {
    fn default() -> Self {
        Slot(Arc::new(Mutex::new(None)))
    }
}

impl<T> Slot<T> {
    pub fn put(&self, v: T) {
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) = Some(v);
    }

    pub fn take(&self) -> Option<T> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).take()
    }
}

/// Run `f` in the background, delivering its output to the returned slot.
#[cfg(not(target_arch = "wasm32"))]
pub fn spawn<T: Send + 'static>(f: impl Future<Output = T> + Send + 'static) -> Slot<T> {
    let slot = Slot::default();
    let out = slot.clone();
    std::thread::spawn(move || out.put(pollster::block_on(f)));
    slot
}

/// Run `f` in the background, delivering its output to the returned slot.
#[cfg(target_arch = "wasm32")]
pub fn spawn<T: 'static>(f: impl Future<Output = T> + 'static) -> Slot<T> {
    let slot = Slot::default();
    let out = slot.clone();
    wasm_bindgen_futures::spawn_local(async move { out.put(f.await) });
    slot
}

/// Hands control back to the browser now and then. Natively a no-op: the
/// work has a thread of its own.
pub struct Yielder {
    #[cfg(target_arch = "wasm32")]
    since: web_time::Instant,
}

// Derivable natively, where the struct is empty.
#[allow(clippy::derivable_impls)]
impl Default for Yielder {
    fn default() -> Self {
        Yielder {
            #[cfg(target_arch = "wasm32")]
            since: web_time::Instant::now(),
        }
    }
}

/// Milliseconds of work between yields in the browser.
#[cfg(target_arch = "wasm32")]
const SLICE_MS: f64 = 10.0;

impl Yielder {
    /// Yield if this slice of work is used up.
    pub async fn maybe_yield(&mut self) {
        #[cfg(target_arch = "wasm32")]
        if self.since.elapsed().as_secs_f64() * 1e3 >= SLICE_MS {
            self.yield_now().await;
        }
    }

    /// Yield unconditionally (e.g. before a long blocking step, so the UI
    /// can show what is about to happen).
    pub async fn yield_now(&mut self) {
        #[cfg(target_arch = "wasm32")]
        {
            sleep_ms(0).await;
            self.since = web_time::Instant::now();
        }
    }

    /// Wait a little (while paused).
    pub async fn idle(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::sleep(std::time::Duration::from_millis(30));
        #[cfg(target_arch = "wasm32")]
        {
            sleep_ms(30).await;
            self.since = web_time::Instant::now();
        }
    }
}

/// A `setTimeout` promise.
#[cfg(target_arch = "wasm32")]
async fn sleep_ms(ms: i32) {
    let p = js_sys::Promise::new(&mut |resolve, _| {
        let w = web_sys::window().expect("a window");
        let _ = w.set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, ms);
    });
    let _ = wasm_bindgen_futures::JsFuture::from(p).await;
}
