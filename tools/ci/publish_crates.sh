#!/usr/bin/env sh
# Publish the workspace crates to crates.io in dependency order, skipping a
# crate whose version is already there (so a rerun after a partial failure
# is safe). `--dry-run [cargo args]` packages and verifies every crate without
# uploading (locally add --allow-dirty for uncommitted changes).
# The token comes from CARGO_REGISTRY_TOKEN.
set -eu
cd "$(dirname "$0")/../.."
if [ "${1:-}" = "--dry-run" ]; then
  # Packages every crate against the others' local packages (cargo >= 1.90).
  shift
  # act copies the working tree with its uncommitted changes.
  [ -z "${ACT:-}" ] || set -- --allow-dirty "$@"
  exec cargo publish --workspace --dry-run --locked "$@"
fi
VERSION=$(sh tools/ci/version.sh)
for crate in morphit morphit-robot morphit-capi morphit-cli morphit-server; do
  code=$(curl -s -o /dev/null -w '%{http_code}' -A "morphit-release (github actions)" \
    "https://crates.io/api/v1/crates/$crate/$VERSION")
  if [ "$code" = 200 ]; then
    echo "$crate $VERSION is already on crates.io; skipping"
    continue
  fi
  echo "publishing $crate $VERSION"
  cargo publish -p "$crate" --locked
done
