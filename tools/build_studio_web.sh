#!/usr/bin/env sh
# Build MorphIt Studio for the browser without Trunk: the wasm binary, its JS
# bindings, a static page and the example library in target/studio-dist.
# Serve that folder with any static file server, e.g.
#   python -m http.server 8080 --directory target/studio-dist
# Needs wasm-bindgen-cli matching Cargo.lock:
#   cargo install wasm-bindgen-cli --version =0.2.128 --locked
set -eu
cd "$(dirname "$0")/.."
PROFILE="${PROFILE:-wasm-release}"
OUT=target/studio-dist
cargo build -p morphit-studio --target wasm32-unknown-unknown --profile "$PROFILE" "$@"
rm -rf "$OUT" && mkdir -p "$OUT"
wasm-bindgen --target web --no-typescript --out-dir "$OUT" \
  "target/wasm32-unknown-unknown/$PROFILE/morphit-studio.wasm"
if command -v wasm-opt >/dev/null 2>&1; then
  wasm-opt -Oz --enable-bulk-memory --enable-nontrapping-float-to-int \
    -o "$OUT/morphit-studio_bg.wasm" "$OUT/morphit-studio_bg.wasm"
fi
# The Trunk page without its build directives, plus the loader.
sed -e '/data-trunk/d' -e '/<!-- Trunk:/d' apps/morphit-studio/index.html \
  | sed -e 's#</body>#  <script type="module">import init from "./morphit-studio.js"; init();</script>\n</body>#' \
  > "$OUT/index.html"
cp -r web/examples "$OUT/examples"
rm -rf "$OUT"/examples/*.spheres
echo "built $OUT ($(du -sh "$OUT" | cut -f1))"
