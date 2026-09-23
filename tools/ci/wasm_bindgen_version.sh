#!/usr/bin/env sh
# Print the wasm-bindgen version pinned in Cargo.lock; wasm-bindgen-cli must match it.
set -eu
cd "$(dirname "$0")/../.."
awk '/^name = "wasm-bindgen"$/ { getline; gsub(/version = |"/, ""); print; exit }' Cargo.lock
