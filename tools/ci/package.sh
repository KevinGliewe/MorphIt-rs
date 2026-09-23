#!/usr/bin/env sh
# Pack the release binaries of one target into archives in <outdir>:
#   morphit-cli-<v>-<target>      the `morphit` command-line tool
#   morphit-studio-<v>-<target>   MorphIt Studio with its example library
#   morphit-server-<v>-<target>   the HTTP server with the web UI (web/)
#   morphit-capi-<v>-<target>     C library (shared + static), morphit.h and a CMake package
# Each holds README.md and LICENSE; Windows targets get .zip, others .tar.gz,
# every archive a .sha256 next to it. Run after
#   cargo build --release --locked --target <target> -p morphit-cli -p morphit-studio -p morphit-server -p morphit-capi
set -eu
[ $# -eq 3 ] || { echo "usage: $0 <target> <version> <outdir>" >&2; exit 2; }
TARGET=$1 VERSION=$2
cd "$(dirname "$0")/../.."
mkdir -p "$3" && OUTDIR=$(cd "$3" && pwd)
BIN="target/$TARGET/release"
[ -d "$BIN" ] || BIN=target/release   # built without --target
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

case "$TARGET" in
  *windows*) EXE=.exe; SHARED=morphit_capi.dll; STATIC=morphit_capi.lib; EXTRA=morphit_capi.dll.lib ;;
  *apple*)   EXE=;     SHARED=libmorphit_capi.dylib; STATIC=libmorphit_capi.a; EXTRA= ;;
  *)         EXE=;     SHARED=libmorphit_capi.so; STATIC=libmorphit_capi.a; EXTRA= ;;
esac

# stage <name> <files...>: a folder <name>-<v>-<target> with the files, README and LICENSE.
stage() {
  dir="$STAGE/$1-$VERSION-$TARGET"
  shift
  mkdir -p "$dir"
  cp README.md LICENSE "$dir/"
  for f in "$@"; do cp -r "$f" "$dir/"; done
  echo "$dir"
}

archive() {
  dir=$1 name=$(basename "$1")
  case "$TARGET" in
    *windows*)
      out="$OUTDIR/$name.zip"
      rm -f "$out"
      if command -v 7z >/dev/null 2>&1; then
        (cd "$STAGE" && 7z a -tzip -bso0 -bsp0 "$out" "$name")
      else
        (cd "$STAGE" && powershell -NoProfile -Command "Compress-Archive -Path '$name' -DestinationPath '$(cygpath -w "$out" 2>/dev/null || echo "$out")'")
      fi ;;
    *)
      out="$OUTDIR/$name.tar.gz"
      tar -C "$STAGE" -czf "$out" "$name" ;;
  esac
  (cd "$OUTDIR" && sha256sum "$(basename "$out")" > "$(basename "$out").sha256")
  echo "packed $(basename "$out")"
}

archive "$(stage morphit-cli "$BIN/morphit$EXE")"

dir=$(stage morphit-studio "$BIN/morphit-studio$EXE")
cp -r web/examples "$dir/examples"
rm -rf "$dir"/examples/*.spheres
archive "$dir"

dir=$(stage morphit-server "$BIN/morphit-server$EXE")
cp -r web "$dir/web"
archive "$dir"

# C library: headers, shared + static libraries and a CMake package
# (find_package(morphit CONFIG), see crates/morphit-capi/cmake).
dir=$(stage morphit-capi)
mkdir -p "$dir/include" "$dir/lib/cmake/morphit"
cp crates/morphit-capi/include/morphit.h "$dir/include/"
case "$TARGET" in
  *windows*) mkdir -p "$dir/bin" && cp "$BIN/$SHARED" "$dir/bin/" ;;
  *)         cp "$BIN/$SHARED" "$dir/lib/" ;;
esac
cp "$BIN/$STATIC" "$dir/lib/"
[ -z "$EXTRA" ] || cp "$BIN/$EXTRA" "$dir/lib/"
cp crates/morphit-capi/cmake/morphit-config.cmake "$dir/lib/cmake/morphit/"
sed "s/@VERSION@/$VERSION/" crates/morphit-capi/cmake/morphit-config-version.cmake.in   > "$dir/lib/cmake/morphit/morphit-config-version.cmake"
# The system libraries the static library needs, as rustc reports them.
target_arg=
[ "$BIN" = target/release ] || target_arg="--target $TARGET"
# shellcheck disable=SC2086
libs=$(cargo rustc --release --locked -q $target_arg -p morphit-capi --crate-type staticlib   -- --print native-static-libs 2>&1 | sed -n 's/.*native-static-libs: //p' | tail -n 1)
[ -n "$libs" ] || { echo "rustc did not report native-static-libs" >&2; exit 1; }
{
  echo "# System libraries of the static MorphIt library ($TARGET), from rustc --print native-static-libs."
  printf 'set(MORPHIT_NATIVE_STATIC_LIBS'
  framework=
  for word in $libs; do
    if [ -n "$framework" ]; then printf ' "-framework %s"' "$word"; framework=; continue; fi
    case "$word" in
      -framework) framework=1 ;;
      -l*) printf ' %s' "${word#-l}" ;;
      /*) ;;   # MSVC /defaultlib: the consumer's CRT choice applies
      *) printf ' %s' "$word" ;;
    esac
  done
  echo ')'
} > "$dir/lib/cmake/morphit/morphit-native-libs.cmake"
archive "$dir"
