#!/usr/bin/env sh
# Build the GitHub Pages site in target/pages: MorphIt Studio at the root and
# the morphit-wasm example page (with its JS package) under wasm/.
#   tools/build_pages.sh && python -m http.server 8080 --directory target/pages
# Needs wasm-bindgen-cli matching Cargo.lock (tools/ci/wasm_bindgen_version.sh);
# wasm-opt is used when present.
set -eu
cd "$(dirname "$0")/.."
PROFILE="${PROFILE:-wasm-release}"
OUT=target/pages

sh tools/build_studio_web.sh
rm -rf "$OUT" && mkdir -p "$OUT/wasm"
cp -r target/studio-dist/. "$OUT/"

cargo build -p morphit-wasm --target wasm32-unknown-unknown --profile "$PROFILE" --locked
wasm-bindgen --target web --out-dir "$OUT/wasm/pkg" \
  "target/wasm32-unknown-unknown/$PROFILE/morphit_wasm.wasm"
if command -v wasm-opt >/dev/null 2>&1; then
  # The flags of [package.metadata.wasm-pack.profile.release] in crates/morphit-wasm.
  wasm-opt -O3 --enable-bulk-memory --enable-nontrapping-float-to-int --enable-sign-ext \
    --enable-mutable-globals -o "$OUT/wasm/pkg/morphit_wasm_bg.wasm" "$OUT/wasm/pkg/morphit_wasm_bg.wasm"
fi
cp crates/morphit-wasm/www/index.html crates/morphit-wasm/www/main.js "$OUT/wasm/"
echo "built $OUT ($(du -sh "$OUT" | cut -f1))"
