#!/usr/bin/env sh
# Check a morphit-capi release archive the way a C project uses it: unpack it,
# build crates/morphit-capi/examples/c against it with find_package(morphit),
# once with the shared and once with the static library, and pack a mesh.
#   tools/ci/test_capi_package.sh dist/morphit-capi-<v>-<target>.(zip|tar.gz)
set -eu
[ $# -eq 1 ] || { echo "usage: $0 <morphit-capi archive>" >&2; exit 2; }
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
ARCHIVE=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

case "$ARCHIVE" in
  *.zip)
    if command -v 7z >/dev/null 2>&1; then 7z x -bso0 -bsp0 -o"$WORK" "$ARCHIVE"
    else unzip -q "$ARCHIVE" -d "$WORK"; fi ;;
  *) tar -xzf "$ARCHIVE" -C "$WORK" ;;
esac
PREFIX=$(find "$WORK" -mindepth 1 -maxdepth 1 -type d -name 'morphit-capi-*' | head -n 1)
[ -f "$PREFIX/lib/cmake/morphit/morphit-config.cmake" ] || { echo "no CMake package in $ARCHIVE" >&2; exit 1; }
MESH="$ROOT/crates/morphit/tests/fixtures/link0.obj"
# Native paths for CMake under Git Bash on Windows.
if command -v cygpath >/dev/null 2>&1; then
  PREFIX=$(cygpath -m "$PREFIX") ROOT=$(cygpath -m "$ROOT") WORK=$(cygpath -m "$WORK") MESH=$(cygpath -m "$MESH")
fi

for kind in shared static; do
  build="$WORK/build-$kind"
  static=OFF
  [ "$kind" = shared ] || static=ON
  cmake -S "$ROOT/crates/morphit-capi/examples/c" -B "$build" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$PREFIX" -DMORPHIT_STATIC=$static
  grep -q "^morphit_DIR:PATH=.*/lib/cmake/morphit$" "$build/CMakeCache.txt" \
    || { echo "the examples did not use the package" >&2; exit 1; }
  cmake --build "$build" --config Release
  exe=$(find "$build" -type f \( -name pack -o -name pack.exe \) | head -n 1)
  "$exe" "$MESH" MorphIt-B 8 10 1 "$build/out.json"
  grep -q '"radii"' "$build/out.json"
  echo "$kind library: ok"
done
