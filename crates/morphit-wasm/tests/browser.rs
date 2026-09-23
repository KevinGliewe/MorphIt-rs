//! WebGPU in a browser: sessions whose searches run on the GPU must equal
//! CPU sessions bit for bit, as natively. Skips when the browser has no
//! WebGPU adapter.
//!
//! `NO_HEADLESS=1 cargo test -p morphit-wasm --target wasm32-unknown-unknown --test browser`
//! then open the printed URL in a WebGPU browser (or run with a WebDriver).
#![cfg(target_arch = "wasm32")]

use std::sync::Arc;

use morphit::{Config, Mesh, Preset, Session};
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

const LINK0: &[u8] = include_bytes!("../../morphit/tests/fixtures/link0.obj");

fn config(device: &str) -> Config {
    let mut c = Config::from_preset(Preset::B);
    c.model.num_spheres = 24;
    c.model.num_inside_samples = 3000;
    c.model.num_surface_samples = 3000;
    c.model.device = device.to_string();
    c.training.iterations = 30;
    c.training.density_control_min_interval = 12;
    c.training.density_control_warmup_steps = 3;
    c.random_seed = Some(11);
    c
}

#[wasm_bindgen_test]
async fn webgpu_session_equals_cpu() {
    let gpus = morphit::init_gpu().await;
    web_sys_log(&format!("WebGPU adapters: {gpus:?}"));
    if gpus.is_empty() {
        web_sys_log("skipping: no WebGPU adapter");
        return;
    }
    let mesh = Arc::new(Mesh::load_from_bytes(LINK0, "obj", None).unwrap());
    let mut gpu = Session::new(config("gpu"), mesh.clone()).unwrap();
    let mut cpu = Session::new(config("cpu"), mesh).unwrap();
    assert!(gpu.needs_async());
    assert!(gpu.step().is_err(), "the sync step refuses WebGPU");
    assert_eq!(gpu.iteration(), 0, "and leaves the session untouched");
    while !cpu.is_done() {
        let a = gpu.step_async().await.unwrap();
        let b = cpu.step_async().await.unwrap();
        assert_eq!((a.iteration, a.total_loss), (b.iteration, b.total_loss));
    }
    assert!(gpu.is_done());
    assert!(gpu.density_passes() > 0);
    assert_eq!(gpu.gpu_error(), None, "the searches must really have run on WebGPU");
    let mut ha = gpu.history().to_json();
    let mut hb = cpu.history().to_json();
    for h in [&mut ha, &mut hb] {
        h.as_object_mut().unwrap().remove("time_per_iteration");
    }
    assert_eq!(ha, hb);
    gpu.finalize();
    cpu.finalize();
    let (mut a, mut b) = (gpu.result(), cpu.result());
    a.config.model.device.clear();
    b.config.model.device.clear();
    assert_eq!(a, b);
    web_sys_log(&format!("WebGPU session on {} equals the CPU session", gpu.device()));
}

fn web_sys_log(s: &str) {
    wasm_bindgen_test::console_log!("{s}");
}
