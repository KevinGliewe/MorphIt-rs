#!/usr/bin/env sh
# Build the npm package `morphit-rs` (the WebAssembly bindings, crates/morphit-wasm):
#   web/   wasm-bindgen --target web     browsers and bundlers (await init())
#   node/  wasm-bindgen --target nodejs  Node.js (loads synchronously)
# package.json routes `import "morphit-rs"` to the right one. Needs
# wasm-bindgen-cli at the version in Cargo.lock; wasm-opt (binaryen) is used
# when installed.
#   tools/build_npm.sh [out dir]      (default target/npm/morphit-rs)
set -eu
cd "$(dirname "$0")/.."
OUT=${1:-target/npm/morphit-rs}
PROFILE=wasm-release
VERSION=$(sh tools/ci/version.sh)

cargo build -p morphit-wasm --target wasm32-unknown-unknown --profile "$PROFILE" --locked
WASM="target/wasm32-unknown-unknown/$PROFILE/morphit_wasm.wasm"
rm -rf "$OUT"
mkdir -p "$OUT"
wasm-bindgen --target web --out-dir "$OUT/web" "$WASM"
wasm-bindgen --target nodejs --out-dir "$OUT/node" "$WASM"
if command -v wasm-opt >/dev/null 2>&1; then
  for f in "$OUT/web/morphit_wasm_bg.wasm" "$OUT/node/morphit_wasm_bg.wasm"; do
    wasm-opt -O3 --enable-bulk-memory --enable-nontrapping-float-to-int --enable-sign-ext \
      --enable-mutable-globals -o "$f" "$f"
  done
fi
# The nodejs build is CommonJS inside a "type": "module" package.
printf '{ "type": "commonjs" }\n' > "$OUT/node/package.json"
cp crates/morphit-wasm/npm/index.mjs "$OUT/node/index.mjs"
cp crates/morphit-wasm/npm/README.md LICENSE "$OUT/"
cat > "$OUT/package.json" <<EOF
{
  "name": "morphit-rs",
  "version": "$VERSION",
  "description": "Approximate triangle meshes and robots with spheres: the MorphIt optimizer in Rust, as WebAssembly with TypeScript types",
  "license": "MIT",
  "repository": { "type": "git", "url": "git+https://github.com/KevinGliewe/MorphIt-rs.git", "directory": "crates/morphit-wasm" },
  "homepage": "https://github.com/KevinGliewe/MorphIt-rs#readme",
  "keywords": ["sphere-packing", "collision", "robotics", "urdf", "mjcf", "mesh", "webassembly", "webgpu", "morphit"],
  "type": "module",
  "main": "./node/index.mjs",
  "module": "./web/morphit_wasm.js",
  "types": "./web/morphit_wasm.d.ts",
  "exports": {
    ".": {
      "types": "./web/morphit_wasm.d.ts",
      "node": "./node/index.mjs",
      "default": "./web/morphit_wasm.js"
    },
    "./morphit_wasm_bg.wasm": "./web/morphit_wasm_bg.wasm",
    "./package.json": "./package.json"
  },
  "files": ["web/", "node/", "README.md", "LICENSE"],
  "sideEffects": ["./node/*"]
}
EOF
echo "built $OUT (morphit-rs $VERSION)"
