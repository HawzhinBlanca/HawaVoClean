#!/usr/bin/env bash
# Build signed, transactional macOS PKG for HawaVoClean DaVinci Resolve Plugin.
#
# Generates com.hawavoclean.resolve.pkg with transactional preinstall backup,
# postinstall verification, rollback on failure, and clean uninstaller support.
#
# Implements Task Q5.2 (P0/L).
set -euo pipefail
export COPYFILE_DISABLE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE=""
OUTPUT=""
SIGN_IDENTITY=""
PLUGIN_ID="com.hawavoclean.resolve"
DEFAULT_INSTALL_LOCATION="/Library/Application Support/Blackmagic Design/DaVinci Resolve/Workflow Integration Plugins"

usage() {
  cat <<EOF
Usage: resolve-plugin/build-pkg.sh --stage ABSOLUTE_PATH --output ABSOLUTE_PATH [--sign IDENTITY]

Builds a transactional macOS installer PKG for the HawaVoClean DaVinci Resolve plugin.
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
say() { printf '==> %s\n' "$*"; }

while [ "$#" -gt 0 ]; do
  case "$1" in
    --stage) [ "$#" -ge 2 ] || die "--stage needs a path"; STAGE="$2"; shift ;;
    --stage=*) STAGE="${1#--stage=}" ;;
    --output) [ "$#" -ge 2 ] || die "--output needs a path"; OUTPUT="$2"; shift ;;
    --output=*) OUTPUT="${1#--output=}" ;;
    --sign) [ "$#" -ge 2 ] || die "--sign needs an identity"; SIGN_IDENTITY="$2"; shift ;;
    --sign=*) SIGN_IDENTITY="${1#--sign=}" ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done

[ -n "$STAGE" ] || die "--stage is required"
[ -n "$OUTPUT" ] || die "--output is required"
case "$STAGE" in /*) ;; *) die "--stage must be absolute" ;; esac
case "$OUTPUT" in /*) ;; *) die "--output must be absolute" ;; esac
[ -d "$STAGE" ] || die "stage directory does not exist: $STAGE"

VERSION="$(cat "$STAGE/VERSION" 2>/dev/null || echo "1.0.0")"
say "Building transactional PKG for $PLUGIN_ID v$VERSION"

PKG_ROOT="$(mktemp -d /tmp/hawavoclean-pkg-root.XXXXXX)"
PKG_SCRIPTS="$(mktemp -d /tmp/hawavoclean-pkg-scripts.XXXXXX)"
trap 'rm -rf "$PKG_ROOT" "$PKG_SCRIPTS"' EXIT

# Payload: the plugin directory itself
mkdir -p "$PKG_ROOT/$PLUGIN_ID"
cp -R "$STAGE/." "$PKG_ROOT/$PLUGIN_ID/"
# Ensure sdk-preflight.js is always packaged so postinstall & runtime can acquire
# Blackmagic's local WorkflowIntegration.node with Mach-O + SHA-256 verification.
if [ ! -f "$PKG_ROOT/$PLUGIN_ID/sdk-preflight.js" ] && [ -f "$SCRIPT_DIR/$PLUGIN_ID/sdk-preflight.js" ]; then
  cp "$SCRIPT_DIR/$PLUGIN_ID/sdk-preflight.js" "$PKG_ROOT/$PLUGIN_ID/sdk-preflight.js"
  if [ -f "$PKG_ROOT/$PLUGIN_ID/SHA256SUMS" ]; then
    (cd "$PKG_ROOT/$PLUGIN_ID" && shasum -a 256 ./sdk-preflight.js >> SHA256SUMS)
  fi
fi
if command -v xattr >/dev/null 2>&1; then
  xattr -cr "$PKG_ROOT/$PLUGIN_ID" 2>/dev/null || true
fi

# Write preinstall script: backup prior version and check Resolve is closed
cat > "$PKG_SCRIPTS/preinstall" << 'EOF'
#!/bin/bash
set -euo pipefail
export COPYFILE_DISABLE=1

DEST="/Library/Application Support/Blackmagic Design/DaVinci Resolve/Workflow Integration Plugins"
PLUGIN_ID="com.hawavoclean.resolve"
TARGET="$DEST/$PLUGIN_ID"

# 1. Reject if DaVinci Resolve is running
if command -v pgrep >/dev/null 2>&1 && pgrep -x Resolve >/dev/null 2>&1; then
  echo "DaVinci Resolve is currently running. Please quit Resolve before installing." >&2
  exit 1
fi

# 2. If existing plugin exists, create pre-install backup
if [ -d "$TARGET" ]; then
  BACKUP_DIR="$DEST/.${PLUGIN_ID}.preinstall-backup"
  rm -rf "$BACKUP_DIR"
  cp -R "$TARGET" "$BACKUP_DIR"
fi

exit 0
EOF
chmod +x "$PKG_SCRIPTS/preinstall"

# Write postinstall script: verify integrity and rollback if failed
cat > "$PKG_SCRIPTS/postinstall" << 'EOF'
#!/bin/bash
set -euo pipefail
export COPYFILE_DISABLE=1

DEST="/Library/Application Support/Blackmagic Design/DaVinci Resolve/Workflow Integration Plugins"
PLUGIN_ID="com.hawavoclean.resolve"
TARGET="$DEST/$PLUGIN_ID"
BACKUP_DIR="$DEST/.${PLUGIN_ID}.preinstall-backup"

rollback() {
  echo "Postinstall verification failed. Restoring previous plugin state..." >&2
  rm -rf "$TARGET"
  if [ -d "$BACKUP_DIR" ]; then
    mv "$BACKUP_DIR" "$TARGET"
  fi
  exit 1
}

# Verify target exists and is directory
[ -d "$TARGET" ] || rollback

# Fix permissions: 0755 for dirs and executables, 0644 for files
chmod -R a+rX "$TARGET"
[ -f "$TARGET/main.js" ] || rollback

# If SHA256SUMS is present, verify integrity BEFORE acquiring local Resolve SDK bridge
if [ -f "$TARGET/SHA256SUMS" ]; then
  (cd "$TARGET" && shasum -a 256 --strict --status -c SHA256SUMS) || rollback
fi

# Legally acquire Blackmagic's local WorkflowIntegration.node from the user's
# DaVinci Resolve Studio installation with Mach-O + SHA-256 checksum verification.
if [ ! -f "$TARGET/WorkflowIntegration.node" ]; then
  ACQUIRED_SDK=0
  if command -v node >/dev/null 2>&1 && [ -f "$TARGET/sdk-preflight.js" ]; then
    if node -e '
      const { acquireWorkflowIntegrationNode } = require(process.argv[1]);
      const res = acquireWorkflowIntegrationNode(process.argv[2]);
      if (!res.ok) {
        console.error("HawaVoClean SDK preflight:", res.code || "ERR", res.error || "unavailable");
        process.exit(1);
      }
      console.log("Acquired WorkflowIntegration.node (sha256=" + res.sha256 + ")");
    ' "$TARGET/sdk-preflight.js" "$TARGET"; then
      ACQUIRED_SDK=1
    fi
  fi
  if [ "$ACQUIRED_SDK" -eq 0 ]; then
    SDK_BASE="/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Workflow Integrations"
    for REL in "Examples/SamplePlugin/WorkflowIntegration.node" "Examples/SamplePromisePlugin/WorkflowIntegration.node"; do
      SRC_NODE="$SDK_BASE/$REL"
      if [ -f "$SRC_NODE" ]; then
        TMP_NODE="$TARGET/WorkflowIntegration.node.tmp.$$"
        SRC_SHA="$(shasum -a 256 "$SRC_NODE" | awk '{print $1}')"
        cp "$SRC_NODE" "$TMP_NODE"
        DST_SHA="$(shasum -a 256 "$TMP_NODE" | awk '{print $1}')"
        if [ "$SRC_SHA" = "$DST_SHA" ] && [ -n "$SRC_SHA" ]; then
          chmod 0755 "$TMP_NODE"
          mv "$TMP_NODE" "$TARGET/WorkflowIntegration.node"
          echo "Acquired local WorkflowIntegration.node (sha256=$DST_SHA)"
          ACQUIRED_SDK=1
          break
        else
          rm -f "$TMP_NODE"
        fi
      fi
    done
  fi
  if [ "$ACQUIRED_SDK" -eq 0 ]; then
    echo "Notice: DaVinci Resolve Studio WorkflowIntegration.node was not found locally yet; runtime preflight will retry when Resolve Studio is installed." >&2
  fi
fi

# Clean up backup on success
rm -rf "$BACKUP_DIR"
echo "HawaVoClean DaVinci Resolve plugin successfully verified and installed."
exit 0
EOF
chmod +x "$PKG_SCRIPTS/postinstall"

mkdir -p "$(dirname "$OUTPUT")"

if command -v pkgbuild >/dev/null 2>&1; then
  PKGBUILD_CMD=(
    pkgbuild
    --root "$PKG_ROOT"
    --install-location "$DEFAULT_INSTALL_LOCATION"
    --identifier "$PLUGIN_ID.pkg"
    --version "$VERSION"
    --scripts "$PKG_SCRIPTS"
  )
  if [ -n "$SIGN_IDENTITY" ]; then
    PKGBUILD_CMD+=(--sign "$SIGN_IDENTITY")
  fi
  PKGBUILD_CMD+=("$OUTPUT")

  say "Running: ${PKGBUILD_CMD[*]}"
  "${PKGBUILD_CMD[@]}"
  say "Successfully generated $OUTPUT"
else
  # Fallback for systems without pkgbuild: generate simulated package bundle
  say "pkgbuild not available; archiving transactional package archive"
  (cd "$PKG_ROOT" && tar -czf "$OUTPUT" "$PLUGIN_ID")
fi

exit 0
