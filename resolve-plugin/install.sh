#!/usr/bin/env bash
# Build, verify, self-test and optionally activate the relocatable Resolve plugin.
#
# Usage:
#   resolve-plugin/install.sh --engine-bundle ABSOLUTE_PATH [options]
#
# Options:
#   --no-install          assemble and self-test only
#   --skip-ui-build       reuse ui/dist, but still require and verify a complete UI
#   --engine-bundle PATH  relocatable, manifest-bearing engine directory (required)
#   --sdk-node PATH       WorkflowIntegration.node from the installed Resolve SDK
#   --desktop-only        explicitly assemble without WorkflowIntegration.node
#   --dest PATH           alternate Workflow Integration Plugins directory
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PLUGIN_ID="com.hawavoclean.resolve"
VERSION=""
SRC_DIR="$SCRIPT_DIR/$PLUGIN_ID"
UI_DIR="$REPO_ROOT/ui"
TOOLCHAIN_DIR="$SCRIPT_DIR/toolchain"
STAGES_DIR="$REPO_ROOT/build/resolve-plugin/stages"
DEFAULT_DEST="/Library/Application Support/Blackmagic Design/DaVinci Resolve/Workflow Integration Plugins"
SDK_NODE_DEFAULT="/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Workflow Integrations/Examples/SamplePlugin/WorkflowIntegration.node"

ENGINE_BUNDLE=""
SDK_NODE="$SDK_NODE_DEFAULT"
DEST="$DEFAULT_DEST"
DO_INSTALL=1
BUILD_UI=1
DESKTOP_ONLY=0

usage() { awk 'NR > 1 && !/^#/ { exit } NR > 1 { sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"; }
say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

verify_engine_bundle() {
  local root="$1" actual listed rel target
  [ -z "$(find "$root" ! -type d ! -type f ! -type l -print -quit)" ] || die "engine bundle contains a special filesystem node"
  (cd "$root" && shasum -a 256 --strict --status -c ENGINE-SHA256SUMS) || die "engine bundle checksum verification failed"

  actual="$(mktemp)"
  (cd "$root" && find . -type f ! -name ENGINE-SHA256SUMS -print | LC_ALL=C sort) > "$actual"
  listed="$(mktemp)"
  sed -E 's/^[0-9a-fA-F]{64}  //' "$root/ENGINE-SHA256SUMS" | LC_ALL=C sort > "$listed"
  if ! cmp -s "$actual" "$listed"; then
    command rm -f "$actual" "$listed"
    die "engine regular-file inventory does not match ENGINE-SHA256SUMS"
  fi
  command rm -f "$actual" "$listed"

  actual="$(mktemp)"
  (cd "$root" && find . -type l -print | LC_ALL=C sort) > "$actual"
  listed="$(mktemp)"
  cut -f1 "$root/ENGINE-SYMLINKS" | LC_ALL=C sort > "$listed"
  if ! cmp -s "$actual" "$listed"; then
    command rm -f "$actual" "$listed"
    die "engine symlink inventory does not match ENGINE-SYMLINKS"
  fi
  command rm -f "$actual" "$listed"
  while IFS=$'\t' read -r rel target; do
    [ -n "$rel" ] || continue
    case "$rel" in ./*) ;; *) die "unsafe engine symlink path: $rel" ;; esac
    case "$target" in /*|../*|*/../*|*/..) die "escaping engine symlink target: $target" ;; esac
    [ -L "$root/${rel#./}" ] && [ "$(readlink "$root/${rel#./}")" = "$target" ] || die "engine symlink differs from its manifest: $rel"
  done < "$root/ENGINE-SYMLINKS"
}

write_checksum_manifest() {
  local root="$1" output="$2" rel
  local -a batch=()
  (
    cd "$root"
    while IFS= read -r rel; do
      batch+=("$rel")
      if [ "${#batch[@]}" -ge 256 ]; then
        shasum -a 256 "${batch[@]}"
        batch=()
      fi
    done < <(find . -type f ! -name "$output" -print | LC_ALL=C sort)
    [ "${#batch[@]}" -eq 0 ] || shasum -a 256 "${batch[@]}"
  ) > "$root/$output"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --no-install) DO_INSTALL=0 ;;
    --skip-ui-build) BUILD_UI=0 ;;
    --desktop-only) DESKTOP_ONLY=1 ;;
    --engine-bundle) [ "$#" -ge 2 ] || die "--engine-bundle needs a path"; ENGINE_BUNDLE="$2"; shift ;;
    --engine-bundle=*) ENGINE_BUNDLE="${1#--engine-bundle=}" ;;
    --sdk-node) [ "$#" -ge 2 ] || die "--sdk-node needs a path"; SDK_NODE="$2"; shift ;;
    --sdk-node=*) SDK_NODE="${1#--sdk-node=}" ;;
    --dest) [ "$#" -ge 2 ] || die "--dest needs a path"; DEST="$2"; shift ;;
    --dest=*) DEST="${1#--dest=}" ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done

[ -n "$ENGINE_BUNDLE" ] || die "--engine-bundle is required; repo-local virtual environments are not installable artifacts"
case "$ENGINE_BUNDLE" in /*) ;; *) die "--engine-bundle must be absolute" ;; esac
[ -d "$ENGINE_BUNDLE" ] && [ ! -L "$ENGINE_BUNDLE" ] || die "engine bundle is not a real directory: $ENGINE_BUNDLE"
for rel in ENGINE-MANIFEST.json ENGINE-SHA256SUMS ENGINE-SYMLINKS hawavoclean-engine python/bin/python3.11 site-packages/hawavoclean; do
  [ -e "$ENGINE_BUNDLE/$rel" ] || die "engine bundle is incomplete: missing $rel"
done
[ -x "$ENGINE_BUNDLE/hawavoclean-engine" ] || die "engine launcher is not executable"
verify_engine_bundle "$ENGINE_BUNDLE"

for f in manifest.xml package.json main.js preload.js session-auth.js; do
  [ -f "$SRC_DIR/$f" ] || die "missing shell source $SRC_DIR/$f"
done
[ -f "$TOOLCHAIN_DIR/package-lock.json" ] || die "Resolve build-tool lock is missing"
command -v node >/dev/null 2>&1 || die "Node.js is required to run the locked build toolchain"
command -v npm >/dev/null 2>&1 || die "npm is required only to bootstrap the integrity-locked build toolchain"
say "Bootstrapping exact pnpm from the Resolve toolchain lock"
npm --prefix "$TOOLCHAIN_DIR" ci --ignore-scripts --audit=false --fund=false
PNPM_CLI="$TOOLCHAIN_DIR/node_modules/pnpm/bin/pnpm.mjs"
[ "$(node "$PNPM_CLI" --version)" = "11.22.0" ] || die "locked pnpm bootstrap did not produce 11.22.0"
VERSION="$(node -e 'process.stdout.write(require(process.argv[1]).version)' "$SRC_DIR/package.json")"
[ "$VERSION" = "3.3.0" ] || die "Resolve package identity is not the expected v3.3.0 release"

if [ "$BUILD_UI" -eq 1 ]; then
  say "Installing UI dependencies from the frozen pnpm lock"
  (cd "$UI_DIR" && node "$PNPM_CLI" install --frozen-lockfile)
  say "Building the UI"
  (cd "$UI_DIR" && node "$PNPM_CLI" build)
else
  say "Reusing the existing UI build (--skip-ui-build)"
fi
[ -f "$UI_DIR/dist/index.html" ] || die "a complete ui/dist build is required"
[ -d "$UI_DIR/dist/assets" ] || die "ui/dist/assets is missing"
# Everything the app ships goes through the bundler into dist/assets, so
# index.html is the only file that may sit at the top of dist/. Anything else
# arrived unprocessed — a stray public/ file, a leftover from a debug session —
# and this script would otherwise package it into the plugin verbatim. A 580 kB
# debug script reached dist/ exactly that way; `publicDir: false` in
# ui/vite.config.ts closes the usual door, and this closes the rest.
STRAY="$(find "$UI_DIR/dist" -type f -not -path '*/assets/*' -not -name 'index.html')"
[ -z "$STRAY" ] || die "unexpected files at the top of ui/dist (rebuild it): $(echo "$STRAY" | tr '\n' ' ')"

say "Installing the exact standalone Electron test runtime from pnpm-lock.yaml"
(cd "$SRC_DIR" && node "$PNPM_CLI" install --frozen-lockfile)

mkdir -p "$STAGES_DIR"
ASSEMBLY_ROOT="$(mktemp -d "$STAGES_DIR/.assembly.XXXXXX")"
STAGE="$ASSEMBLY_ROOT/$PLUGIN_ID"
mkdir "$STAGE"

cleanup_assembly() {
  if [ -d "$ASSEMBLY_ROOT" ]; then
    case "$ASSEMBLY_ROOT" in "$STAGES_DIR/.assembly."*) find "$ASSEMBLY_ROOT" -depth -delete ;; esac
  fi
}
trap cleanup_assembly EXIT

say "Assembling a relocatable plugin"
for f in manifest.xml package.json main.js preload.js session-auth.js; do cp "$SRC_DIR/$f" "$STAGE/$f"; done
cp -R "$UI_DIR/dist/." "$STAGE/"
cp -R "$ENGINE_BUNDLE" "$STAGE/engine"
chmod a+rx "$STAGE/engine/hawavoclean-engine"

if [ "$DESKTOP_ONLY" -eq 0 ]; then
  [ -f "$SDK_NODE" ] || die "WorkflowIntegration.node not found at $SDK_NODE; pass --sdk-node or use --desktop-only explicitly"
  cp "$SDK_NODE" "$STAGE/WorkflowIntegration.node"
else
  say "Desktop-only stage requested; Resolve native bridge intentionally omitted"
fi

cat > "$STAGE/engine.json" <<'EOF'
{
  "command": ["./engine/hawavoclean-engine", "serve"],
  "cwd": ".",
  "env": {"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
}
EOF
printf '%s\n' "$PLUGIN_ID" > "$STAGE/PLUGIN_ID"
printf '%s\n' "$VERSION" > "$STAGE/VERSION"

say "Running engine integrity and preflight checks"
"$STAGE/engine/hawavoclean-engine" --version | grep -Fxq "hawavoclean $VERSION" || die "engine version does not match plugin version $VERSION"
"$STAGE/engine/hawavoclean-engine" doctor >/dev/null || die "engine doctor preflight failed"

(cd "$STAGE" && find . -type l -print | LC_ALL=C sort | while IFS= read -r rel; do printf '%s\t%s\n' "$rel" "$(readlink "$rel")"; done) > "$STAGE/SYMLINKS"
write_checksum_manifest "$STAGE" SHA256SUMS

say "Running the real staged Electron → engine → health → shutdown self-test"
"$SCRIPT_DIR/dev/stage-selftest.sh" "$STAGE"

STAGE_DIGEST="$(shasum -a 256 "$STAGE/SHA256SUMS" | awk '{print $1}')"
FINAL_ROOT="$STAGES_DIR/${STAGE_DIGEST:0:20}"
FINAL_STAGE="$FINAL_ROOT/$PLUGIN_ID"
if [ -d "$FINAL_STAGE" ] && cmp -s "$STAGE/SHA256SUMS" "$FINAL_STAGE/SHA256SUMS" && cmp -s "$STAGE/SYMLINKS" "$FINAL_STAGE/SYMLINKS"; then
  say "Reusing identical content-addressed stage"
else
  [ ! -e "$FINAL_ROOT" ] || die "content-addressed stage collision: $FINAL_ROOT"
  mkdir "$FINAL_ROOT"
  mv "$STAGE" "$FINAL_STAGE"
fi
trap - EXIT
cleanup_assembly
"$SCRIPT_DIR/activate.sh" --stage "$FINAL_STAGE" --verify-only >/dev/null

say "Verified stage: $FINAL_STAGE"
if [ "$DO_INSTALL" -eq 0 ]; then
  printf '%s\n' "$FINAL_STAGE"
  exit 0
fi

if [ -d "$DEST" ] && [ -w "$DEST" ]; then
  exec "$SCRIPT_DIR/activate.sh" --stage "$FINAL_STAGE" --dest "$DEST"
fi

cat >&2 <<EOF

The Resolve plugins directory needs elevated write permission. The stage has
already passed checksum, engine, doctor and Electron lifecycle tests. Activate
it transactionally with:

  sudo "$SCRIPT_DIR/activate.sh" --stage "$FINAL_STAGE" --dest "$DEST"

The activator verifies the copied bytes, backs up the prior plugin, atomically
renames the candidate into place, verifies again, and rolls back on failure.
EOF
exit 2
