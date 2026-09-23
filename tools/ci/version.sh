#!/usr/bin/env sh
# Print the workspace version (the version of every crate).
set -eu
cd "$(dirname "$0")/../.."
cargo pkgid -p morphit | sed 's/.*[#@]//'
