#!/usr/bin/env sh
# Publish the workspace crates to crates.io in dependency order, skipping a
# crate whose version is already there (so a rerun after a partial failure
# is safe). `--dry-run [cargo args]` packages and verifies every crate without
# uploading; uncommitted changes are allowed (the README rewrite needs it).
# The token comes from CARGO_REGISTRY_TOKEN.
#
# The crates share the root README.md. crates.io resolves its relative links
# against the crate folder (crates/<name>/), where assets/ and the other files
# do not exist, so the packaged copy gets absolute links pinned to the release
# tag v<version>: images to raw.githubusercontent.com, files to github.com.
# The original README.md is restored on exit.
set -eu
cd "$(dirname "$0")/../.."
VERSION=$(sh tools/ci/version.sh)
REPO=https://github.com/KevinGliewe/MorphIt-rs
RAW=https://raw.githubusercontent.com/KevinGliewe/MorphIt-rs/v$VERSION

readme_backup=$(mktemp)
cp README.md "$readme_backup"
trap 'cp "$readme_backup" README.md; rm -f "$readme_backup"' EXIT
# Images (src/srcset="path") and markdown links to files ](path); anchors (#...)
# and absolute URLs (containing ':') stay as they are.
sed -E -i \
  -e "s@(src|srcset)=\"([^\"#:]+)\"@\1=\"$RAW/\2\"@g" \
  -e "s@\]\(([^)#:]+)\)@]($REPO/blob/v$VERSION/\1)@g" \
  README.md
if grep -nE '(src|srcset)="[^"#:]+"|\]\([^)#:]+\)' README.md; then
  echo "README.md still has relative links (above)" >&2
  exit 1
fi

if [ "${1:-}" = "--dry-run" ]; then
  # Packages every crate against the others' local packages (cargo >= 1.90).
  # The README rewrite makes the tree dirty.
  shift
  cargo publish --workspace --dry-run --locked --allow-dirty "$@"
  exit 0
fi

# The README links point at the tag, so it must exist before anything is published.
git ls-remote --exit-code --tags origin "refs/tags/v$VERSION" >/dev/null \
  || { echo "tag v$VERSION is not on origin; push it before publishing" >&2; exit 1; }
for crate in morphit morphit-robot morphit-capi morphit-cli morphit-server; do
  code=$(curl -s -o /dev/null -w '%{http_code}' -A "morphit-release (github actions)" \
    "https://crates.io/api/v1/crates/$crate/$VERSION")
  if [ "$code" = 200 ]; then
    echo "$crate $VERSION is already on crates.io; skipping"
    continue
  fi
  echo "publishing $crate $VERSION"
  cargo publish -p "$crate" --locked --allow-dirty
done
