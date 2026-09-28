// Node entry of the morphit-rs package: the wasm-bindgen `nodejs` build loads
// and instantiates the module synchronously. `init()` is a no-op here so the
// same `import init, { ... } from "morphit-rs"; await init();` works in
// browsers, bundlers and Node.
export * from "./morphit_wasm.js";

export default async function init() {}
