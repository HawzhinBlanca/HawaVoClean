#!/usr/bin/env bash
# HawaVoClean — build the UI, assemble the Resolve Workflow Integration plugin, install it.
#
#   resolve-plugin/install.sh [--no-install] [--skip-ui] [--engine PATH] [--sdk-node PATH]
#
# Steps
#   1. Build the UI bundle:  pnpm --dir ui install --frozen-lockfile (fallback: install) && pnpm --dir ui build
#   2. Assemble build/resolve-plugin/com.hawavoclean.resolve/ =
#        manifest.xml package.json main.js preload.js   (shell sources, resolve-plugin/com.hawavoclean.resolve/)
#      + index.html assets/ ...                          (ui/dist contents)
#      + WorkflowIntegration.node                        (copied from the Resolve SDK, never committed)
#      + engine.json                                     (generated: {"command":[ENGINE,"serve"],"cwd":REPO,"env":{}})
#   3. Install into "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Workflow Integration Plugins/".
#      If that directory is not writable, the exact sudo command is printed and the script exits 2
#      (it never elevates on its own).
#
# Flags
#   --no-install       assemble only (steps 1–2)
#   --skip-ui          skip step 1; reuse an existing ui/dist if present, otherwise assemble without the UI
#   --engine PATH      hawavoclean executable to write into engine.json (default: <repo>/.venv/bin/hawavoclean)
#   --sdk-node PATH    WorkflowIntegration.node to ship (default: the SDK's Examples/SamplePlugin copy)
#   -h, --help
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PLUGIN_ID="com.hawavoclean.resolve"
SRC_DIR="$SCRIPT_DIR/$PLUGIN_ID"
UI_DIR="$REPO_ROOT/ui"
STAGE_ROOT="$REPO_ROOT/build/resolve-plugin"
STAGE="$STAGE_ROOT/$PLUGIN_ID"
DEST="/Library/Application Support/Blackmagic Design/DaVinci Resolve/Workflow Integration Plugins"
SDK_NODE_DEFAULT="/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Workflow Integrations/Examples/SamplePlugin/WorkflowIntegration.node"

ENGINE="$REPO_ROOT/.venv/bin/hawavoclean"
SDK_NODE="$SDK_NODE_DEFAULT"
DO_INSTALL=1
BUILD_UI=1

usage() { awk 'NR > 1 && !/^#/ { exit } NR > 1 { sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --no-install) DO_INSTALL=0 ;;
    --skip-ui) BUILD_UI=0 ;;
    --engine) [ $# -ge 2 ] || { echo "--engine needs a PATH" >&2; exit 1; }; ENGINE="$2"; shift ;;
    --engine=*) ENGINE="${1#--engine=}" ;;
    --sdk-node) [ $# -ge 2 ] || { echo "--sdk-node needs a PATH" >&2; exit 1; }; SDK_NODE="$2"; shift ;;
    --sdk-node=*) SDK_NODE="${1#--sdk-node=}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
  shift
done

say()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

for f in manifest.xml package.json main.js preload.js; do
  [ -f "$SRC_DIR/$f" ] || die "missing shell source $SRC_DIR/$f"
done

# ---------------------------------------------------------------------------------------------
# 1. UI build
# ---------------------------------------------------------------------------------------------
if [ "$BUILD_UI" -eq 1 ]; then
  [ -f "$UI_DIR/package.json" ] || die "$UI_DIR/package.json not found (pass --skip-ui to assemble without the UI)"
  command -v pnpm >/dev/null 2>&1 || die "pnpm not found on PATH (corepack enable / npm i -g pnpm)"
  say "Installing UI dependencies (pnpm --dir $UI_DIR install --frozen-lockfile)"
  if ! pnpm --dir "$UI_DIR" install --frozen-lockfile; then
    warn "frozen-lockfile install failed; retrying without --frozen-lockfile"
    pnpm --dir "$UI_DIR" install
  fi
  say "Building UI (pnpm --dir $UI_DIR build)"
  pnpm --dir "$UI_DIR" build
  [ -f "$UI_DIR/dist/index.html" ] || die "UI build did not produce $UI_DIR/dist/index.html"
else
  say "Skipping UI build (--skip-ui)"
fi

# ---------------------------------------------------------------------------------------------
# 2. Assemble
# ---------------------------------------------------------------------------------------------
case "$STAGE" in
  */build/resolve-plugin/$PLUGIN_ID) ;;
  *) die "refusing to clean unexpected staging path: $STAGE" ;;
esac
say "Assembling $STAGE"
rm -rf "$STAGE"
mkdir -p "$STAGE"

for f in manifest.xml package.json main.js preload.js; do
  cp "$SRC_DIR/$f" "$STAGE/$f"
done

if [ -f "$UI_DIR/dist/index.html" ]; then
  cp -R "$UI_DIR/dist/." "$STAGE/"
  say "UI bundle: $(cd "$UI_DIR/dist" && find . -type f | wc -l | tr -d ' ') files from $UI_DIR/dist"
else
  warn "no UI bundle at $UI_DIR/dist — the shell will show its 'UI bundle not found' page"
fi

if [ -f "$SDK_NODE" ]; then
  cp "$SDK_NODE" "$STAGE/WorkflowIntegration.node"
  say "WorkflowIntegration.node copied from $SDK_NODE"
else
  if [ "$DO_INSTALL" -eq 1 ]; then
    die "WorkflowIntegration.node not found at $SDK_NODE (is DaVinci Resolve Studio installed? pass --sdk-node PATH)"
  fi
  warn "WorkflowIntegration.node not found at $SDK_NODE — assembling a desktop-only (host 'electron') plugin"
fi

if [ ! -x "$ENGINE" ]; then
  warn "engine executable $ENGINE does not exist or is not executable yet; engine.json will still point at it"
fi
if command -v node >/dev/null 2>&1; then
  node -e '
    const fs = require("fs");
    const [out, engine, cwd] = process.argv.slice(1);
    fs.writeFileSync(out, JSON.stringify({ command: [engine, "serve"], cwd, env: {} }, null, 2) + "\n");
  ' "$STAGE/engine.json" "$ENGINE" "$REPO_ROOT"
else
  esc() { printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'; }
  printf '{\n  "command": ["%s", "serve"],\n  "cwd": "%s",\n  "env": {}\n}\n' "$(esc "$ENGINE")" "$(esc "$REPO_ROOT")" > "$STAGE/engine.json"
fi
say "engine.json: $(tr -d '\n' < "$STAGE/engine.json" | tr -s ' ')"

say "Staged plugin contents:"
( cd "$STAGE" && ls -1 | sed 's/^/    /' )

# ---------------------------------------------------------------------------------------------
# 3. Install
# ---------------------------------------------------------------------------------------------
if [ "$DO_INSTALL" -eq 0 ]; then
  say "Assembly only (--no-install). Plugin path:"
  echo "$STAGE"
  exit 0
fi

TARGET="$DEST/$PLUGIN_ID"
if [ -d "$DEST" ] && [ -w "$DEST" ] && { [ ! -e "$TARGET" ] || [ -w "$TARGET" ]; }; then
  say "Installing into $TARGET"
  rm -rf "$TARGET"
  cp -R "$STAGE" "$TARGET"
  say "Installed. Restart DaVinci Resolve Studio, then open Workspace → Workflow Integrations → HawaVoClean."
  echo "$TARGET"
  exit 0
fi

cat >&2 <<EOF

The plugins directory is not writable by $(id -un):
    $DEST
Run this to install (it replaces any existing $PLUGIN_ID):

    sudo mkdir -p "$DEST" && sudo rm -rf "$TARGET" && sudo cp -R "$STAGE" "$TARGET"

Then restart DaVinci Resolve Studio and open Workspace → Workflow Integrations → HawaVoClean.
Staged plugin path: $STAGE
EOF
exit 2
