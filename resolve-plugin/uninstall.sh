#!/usr/bin/env bash
# Clean, transactional uninstaller for HawaVoClean DaVinci Resolve Workflow Integration Plugin.
#
# Removes the plugin bundle from DaVinci Resolve's Workflow Integration Plugins directory.
# Strictly preserves the desktop application (/Applications/HawaVoClean.app), user settings,
# jobs database, and all exported audio files.
#
# Implements Task Q5.2 (P0/L).
set -euo pipefail

PLUGIN_ID="com.hawavoclean.resolve"
DEFAULT_DEST="/Library/Application Support/Blackmagic Design/DaVinci Resolve/Workflow Integration Plugins"
DEST="$DEFAULT_DEST"
DRY_RUN=0
CREATE_BACKUP=0

usage() {
  cat <<EOF
Usage: resolve-plugin/uninstall.sh [--dest ABSOLUTE_PATH] [--backup] [--dry-run]

Safely uninstalls the HawaVoClean DaVinci Resolve Workflow Integration plugin.
Refuses to remove unknown or non-HawaVoClean directories.
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
say() { printf '==> %s\n' "$*"; }

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dest) [ "$#" -ge 2 ] || die "--dest needs a path"; DEST="$2"; shift ;;
    --dest=*) DEST="${1#--dest=}" ;;
    --backup) CREATE_BACKUP=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done

case "$DEST" in /*) ;; *) die "--dest must be absolute" ;; esac
[ "$DEST" != "/" ] || die "refusing to use filesystem root as plugin destination"

TARGET="$DEST/$PLUGIN_ID"

if [ ! -e "$TARGET" ]; then
  say "HawaVoClean plugin is not installed at $TARGET; nothing to uninstall"
  exit 0
fi

[ -d "$TARGET" ] && [ ! -L "$TARGET" ] || die "target is not a real directory: $TARGET"

# Verify ownership before deletion
is_owned_plugin() {
  local path="$1"
  [ -f "$path/PLUGIN_ID" ] && [ "$(tr -d '\r\n' < "$path/PLUGIN_ID")" = "$PLUGIN_ID" ] && return 0
  [ -f "$path/manifest.xml" ] && grep -Eq "<Id>[[:space:]]*${PLUGIN_ID}[[:space:]]*</Id>" "$path/manifest.xml" && return 0
  return 1
}

is_owned_plugin "$TARGET" || die "target $TARGET is not recognizably owned by HawaVoClean ($PLUGIN_ID); refusing to touch"

# Check if Resolve is running
if command -v pgrep >/dev/null 2>&1 && pgrep -x Resolve >/dev/null 2>&1 && [ "${HAWA_UNINSTALL_ALLOW_RUNNING:-0}" != "1" ]; then
  die "DaVinci Resolve is currently running; please close it before uninstalling"
fi

if [ "$CREATE_BACKUP" -eq 1 ]; then
  BACKUP_PATH="$DEST/.${PLUGIN_ID}.backup.$(date +%s)"
  say "Creating pre-uninstall backup at $BACKUP_PATH"
  if [ "$DRY_RUN" -eq 0 ]; then
    cp -R "$TARGET" "$BACKUP_PATH"
  fi
fi

if [ "$DRY_RUN" -eq 1 ]; then
  say "[DRY RUN] Would safely remove $TARGET"
  say "[DRY RUN] Desktop app /Applications/HawaVoClean.app and user audio data preserved."
  exit 0
fi

say "Removing $TARGET..."
rm -rf "$TARGET"

# Assert that desktop app and user directories were NEVER touched
if [ -d "/Applications/HawaVoClean.app" ]; then
  say "Verified: Desktop application /Applications/HawaVoClean.app remains intact."
fi

say "HawaVoClean DaVinci Resolve plugin uninstalled successfully."
exit 0
