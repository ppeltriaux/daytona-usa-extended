#!/usr/bin/env bash
# Build the custom Model 2 MAME plus the unidasm disassembler from source, into build/mame (git-ignored).
# Adapted from the virtua-racing project's tools/build-mame.sh (Model 1 there).
#
# The build is MAME 0.289 (`mame0289`) with tools/mame-daytona-rom-space.patch applied: the `daytona` ROM regions grown
# as far as the game's address decoding reaches, the new areas loaded from five optional files (docs/formats/
# model2-rom-images.md). The stock set runs on it unchanged. The binary is
# build/mame/model2; dtools.config.custom_mame() finds it through the stamp written at the end (the patch's SHA-256).
# The installed `mame` stays the reference for every other test.
#
# Licence: this script and tools/mame-daytona-rom-space.patch are changes to MAME and fall under MAME's licence
# (GPL-2.0; https://github.com/mamedev/mame, COPYING). The rest of this repository (the converter, dtools/) is MIT.
# The release set (`dtools build`, README.md "Quickstart") needs this build: stock MAME 0.289 will not run it.
#
# Idempotent: clones once, checks out the pinned tag, applies the patch unless it is already applied (a different
# patch: the file is restored first), rebuilds only what changed.
# Env: MAME_TAG (default mame0289), MAME_REPO (default GitHub; e.g. file:///path/to/a/local/mame/clone),
# MAME_JOBS (default: CPU count - 2, so the machine stays usable).
set -euo pipefail
TAG="${MAME_TAG:-mame0289}"
REPO="${MAME_REPO:-https://github.com/mamedev/mame.git}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="$ROOT/build/mame"
PATCH="$ROOT/tools/mame-daytona-rom-space.patch"
STAMP="$DIR/.daytona-rom-space"
DRIVER=src/mame/sega/model2.cpp
JOBS="${MAME_JOBS:-$(( $(sysctl -n hw.ncpu) > 3 ? $(sysctl -n hw.ncpu) - 2 : 1 ))}"
command -v brew >/dev/null || { echo "Homebrew required"; exit 1; }
for dep in sdl3 pkgconf; do
  brew list --formula "$dep" >/dev/null 2>&1 || { echo "missing: brew install $dep"; exit 1; }
done
mkdir -p "$ROOT/build"
if [ ! -d "$DIR/.git" ]; then
  git clone --depth 1 --branch "$TAG" "$REPO" "$DIR"
fi
cd "$DIR"
git fetch --depth 1 origin "refs/tags/$TAG:refs/tags/$TAG" 2>/dev/null || true
if [ "$(git rev-parse HEAD)" != "$(git rev-parse "$TAG^{commit}")" ]; then
  git checkout -q -- "$DRIVER"                       # the patch is re-applied below
  git checkout -q "$TAG"
fi
rm -f "$STAMP"                                       # no stamp until this patch is built
if git apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "patch already applied: $PATCH"
else
  git checkout -q -- "$DRIVER"                       # another patch, or none: start from the tag's driver
  git apply "$PATCH"
  echo "patch applied: $PATCH"
fi
echo "Compiling MAME (SUBTARGET=model2, $JOBS jobs, nice 10): from scratch this takes about 25-60 minutes and keeps"
echo "the machine busy; a rebuild after a patch change recompiles one driver (a few minutes)."
nice -n 10 make -j"$JOBS" SUBTARGET=model2 SOURCES="$DRIVER" TOOLS=1 REGENIE=1 NOWERROR=1
ls -la "$DIR/model2" "$DIR/unidasm"
shasum -a 256 "$PATCH" | cut -d' ' -f1 > "$STAMP"
echo "built: $DIR/model2 (stamp $STAMP)"
