//! MorphIt for JavaScript: the sphere-packing optimizer compiled to
//! WebAssembly, with an API modelled on the C API (`morphit-capi`).
//!
//! ```js
//! import init, { Config, Mesh, Session, initGpu } from "./pkg/morphit_wasm.js";
//! await init();
//! await initGpu();                      // optional: WebGPU for the searches
//! const mesh = Mesh.fromBytes(bytes, "obj", "bunny.obj");
//! const config = Config.fromPreset("MorphIt-B");
//! config.set("model.num_spheres", 20);
//! const session = new Session(config, mesh);
//! while (!session.isDone) {
//!   await session.stepMany({ budgetMs: 12 });  // keep the page responsive
//!   draw(session.centers(), session.radii());
//! }
//! session.finalize();
//! const json = session.resultJson();
//! ```
//!
//! Everything runs on the calling thread. `step`/`stepMany` return promises
//! because a WebGPU readback resolves only after control returns to the
//! browser; on the CPU they resolve immediately.

mod config;
mod dto;
mod export;
mod mesh;
mod robot;
mod session;

use wasm_bindgen::prelude::*;

pub use config::JsConfig;
pub use export::{aggregate, evaluate, link_quality, object_mjcf, object_urdf, spheres_from_urdf};
pub use mesh::JsMesh;
pub use robot::JsRobotPackage;
pub use session::JsSession;

#[wasm_bindgen(typescript_custom_section)]
const TS_TYPES: &str = include_str!("types.d.ts");

/// Library version.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Route Rust panics to `console.error` and `tracing` events at `level`
/// (`error`, `warn`, `info`, `debug`, `trace`; default `warn`) to the
/// console. Call once; later calls only keep the first level.
#[wasm_bindgen(js_name = initLogging)]
pub fn init_logging(level: Option<String>) {
    console_error_panic_hook::set_once();
    let level = match level.as_deref().unwrap_or("warn") {
        "error" => tracing::Level::ERROR,
        "info" => tracing::Level::INFO,
        "debug" => tracing::Level::DEBUG,
        "trace" => tracing::Level::TRACE,
        _ => tracing::Level::WARN,
    };
    let config = tracing_wasm::WASMLayerConfigBuilder::new().set_max_level(level).build();
    let _ = std::panic::catch_unwind(|| tracing_wasm::set_as_global_default_with_config(config));
}

/// Discover WebGPU adapters and open a device on each. Await this once
/// before creating sessions that should use the GPU (`model.device` `auto`
/// or `gpu`). Resolves to an empty list when the browser has no WebGPU; the
/// sessions then run on the CPU.
#[wasm_bindgen(js_name = initGpu, unchecked_return_type = "Promise<GpuInfo[]>")]
pub async fn init_gpu() -> Result<JsValue, JsError> {
    let list: Vec<dto::GpuInfoDto> = morphit::init_gpu().await.into_iter().map(Into::into).collect();
    dto::to_js(&list)
}

/// The GPU adapters found by [`init_gpu`] (empty before it has run).
#[wasm_bindgen(js_name = listDevices, unchecked_return_type = "GpuInfo[]")]
pub fn list_devices() -> Result<JsValue, JsError> {
    let list: Vec<dto::GpuInfoDto> = morphit::list_devices().into_iter().map(Into::into).collect();
    dto::to_js(&list)
}

/// A JS exception carrying `e`'s message.
pub(crate) fn err(e: impl std::fmt::Display) -> JsError {
    JsError::new(&e.to_string())
}
