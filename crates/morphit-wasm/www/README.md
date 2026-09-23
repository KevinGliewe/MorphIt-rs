# morphit-wasm example page

A single page that packs a mesh in the browser with the WebAssembly build:
pick an `.obj`/`.stl`/`.ply`/`.dae`, choose preset, sphere count, iterations
and device (WebGPU adapters appear when the browser has WebGPU), watch the
spheres (top view) and the loss curve, and download the result JSON, URDF
or MJCF.

Build the bindings into `pkg/` (from the repository root), then serve this
folder with any static file server:

```sh
cargo build -p morphit-wasm --target wasm32-unknown-unknown --profile wasm-release
wasm-bindgen --target web --out-dir crates/morphit-wasm/www/pkg \
  target/wasm32-unknown-unknown/wasm-release/morphit_wasm.wasm
python -m http.server 8080 --directory crates/morphit-wasm/www
```

`wasm-bindgen` must be the version in `Cargo.lock`
(`cargo install wasm-bindgen-cli --version =0.2.128 --locked`);
`wasm-pack build crates/morphit-wasm --target web --out-dir www/pkg` works too.
