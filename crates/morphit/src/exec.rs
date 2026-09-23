//! Running the async core synchronously.
//!
//! The optimizer is written as `async` so that GPU readbacks can await the
//! browser's `mapAsync`. Everywhere else the futures never suspend: the CPU
//! computes inline and the native GPU readback blocks inside the future. The
//! sync API ([`crate::Session::step`], [`crate::loss::evaluate`], ...) drives
//! them with [`block_on`].

use std::future::Future;

/// Drive `f` to completion on the calling thread.
///
/// Natively with the `gpu` feature this is `pollster::block_on`. Otherwise
/// the future must complete on its first poll; it panics if it does not
/// (only a WebGPU readback suspends, and [`crate::Session::step`] refuses
/// that case before it starts).
pub(crate) fn block_on<F: Future>(f: F) -> F::Output {
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    {
        pollster::block_on(f)
    }
    #[cfg(not(all(feature = "gpu", not(target_arch = "wasm32"))))]
    {
        let mut f = std::pin::pin!(f);
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        match f.as_mut().poll(&mut cx) {
            std::task::Poll::Ready(v) => v,
            std::task::Poll::Pending => {
                panic!("a WebGPU search must be driven asynchronously (use Session::step_async)")
            }
        }
    }
}
