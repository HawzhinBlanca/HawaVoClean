#!/usr/bin/env bash
# Transactionally activate one already-assembled HawaVoClean Resolve plugin.
#
# This script is intentionally build-tool free so it is safe to run through
# sudo after install.sh has assembled and tested an unprivileged stage.
set -euo pipefail

PLUGIN_ID="com.hawavoclean.resolve"
DEFAULT_DEST="/Library/Application Support/Blackmagic Design/DaVinci Resolve/Workflow Integration Plugins"
STAGE=""
DEST="$DEFAULT_DEST"
VERIFY_ONLY=0

usage() {
  cat <<EOF
Usage: resolve-plugin/activate.sh --stage ABSOLUTE_PATH [--dest ABSOLUTE_PATH] [--verify-only]

Copies a verified stage into a same-filesystem transaction, backs up an
existing HawaVoClean plugin, atomically activates the candidate, verifies the
installed bytes, and restores the backup on any failure.
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
say() { printf '==> %s\n' "$*"; }

while [ "$#" -gt 0 ]; do
  case "$1" in
    --stage) [ "$#" -ge 2 ] || die "--stage needs a path"; STAGE="$2"; shift ;;
    --stage=*) STAGE="${1#--stage=}" ;;
    --dest) [ "$#" -ge 2 ] || die "--dest needs a path"; DEST="$2"; shift ;;
    --dest=*) DEST="${1#--dest=}" ;;
    --verify-only) VERIFY_ONLY=1 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done

[ -n "$STAGE" ] || die "--stage is required"
case "$STAGE" in /*) ;; *) die "--stage must be absolute" ;; esac
case "$DEST" in /*) ;; *) die "--dest must be absolute" ;; esac
[ "$DEST" != "/" ] || die "refusing to use the filesystem root as a plugin destination"
[ -d "$STAGE" ] && [ ! -L "$STAGE" ] || die "stage is not a real directory: $STAGE"

is_owned_plugin() {
  local path="$1"
  [ -f "$path/PLUGIN_ID" ] && [ "$(tr -d '\r\n' < "$path/PLUGIN_ID")" = "$PLUGIN_ID" ] && return 0
  [ -f "$path/manifest.xml" ] && grep -Eq "<Id>[[:space:]]*${PLUGIN_ID}[[:space:]]*</Id>" "$path/manifest.xml"
}

remove_transaction_tree() {
  local path="$1"
  case "$path" in "$DEST/.${PLUGIN_ID}.transaction."*) ;; *) die "unsafe transaction cleanup path: $path" ;; esac
  [ ! -e "$path" ] || find "$path" -depth -delete
}

verify_tree() {
  local root="$1" listed actual target rel actual_files listed_files
  [ -d "$root" ] && [ ! -L "$root" ] || return 1
  for rel in PLUGIN_ID VERSION SHA256SUMS SYMLINKS manifest.xml package.json main.js preload.js engine.json index.html engine/hawavoclean-engine; do
    [ -e "$root/$rel" ] || { printf 'missing staged file: %s\n' "$rel" >&2; return 1; }
  done
  [ -x "$root/engine/hawavoclean-engine" ] || { printf 'engine launcher is not executable\n' >&2; return 1; }
  [ "$(tr -d '\r\n' < "$root/PLUGIN_ID")" = "$PLUGIN_ID" ] || return 1
  grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$' "$root/VERSION" || return 1
  [ -z "$(find "$root" ! -type d ! -type f ! -type l -print -quit)" ] || {
    printf 'stage contains a special filesystem node\n' >&2
    return 1
  }
  (cd "$root" && shasum -a 256 --strict --status -c SHA256SUMS) || return 1

  # A checksum manifest is also an allowlist: added regular files must fail,
  # rather than silently surviving activation outside the integrity boundary.
  actual_files="$(mktemp)"
  (cd "$root" && find . -type f ! -name SHA256SUMS -print | LC_ALL=C sort) > "$actual_files"
  listed_files="$(mktemp)"
  sed -E 's/^[0-9a-fA-F]{64}  //' "$root/SHA256SUMS" | LC_ALL=C sort > "$listed_files"
  if ! cmp -s "$actual_files" "$listed_files"; then
    command rm -f "$actual_files" "$listed_files"
    printf 'stage regular-file inventory does not match SHA256SUMS\n' >&2
    return 1
  fi
  command rm -f "$actual_files" "$listed_files"

  # Every symlink is declared, relative, and confined to the plugin tree.
  actual="$(mktemp)"
  (cd "$root" && find . -type l -print | LC_ALL=C sort) > "$actual"
  listed="$(mktemp)"
  cut -f1 "$root/SYMLINKS" | LC_ALL=C sort > "$listed"
  if ! cmp -s "$actual" "$listed"; then
    command rm -f "$actual" "$listed"
    printf 'stage symlink inventory does not match SYMLINKS\n' >&2
    return 1
  fi
  command rm -f "$actual" "$listed"
  while IFS=$'\t' read -r rel target; do
    [ -n "$rel" ] || continue
    case "$rel" in ./*) ;; *) printf 'unsafe symlink path: %s\n' "$rel" >&2; return 1 ;; esac
    case "$target" in /*|../*|*/../*|*/..) printf 'escaping symlink target: %s\n' "$target" >&2; return 1 ;; esac
    [ -L "$root/${rel#./}" ] || return 1
    [ "$(readlink "$root/${rel#./}")" = "$target" ] || return 1
  done < "$root/SYMLINKS"
}

if [ "$VERIFY_ONLY" -eq 1 ]; then
  verify_tree "$STAGE" || die "staged plugin failed content verification"
  printf '%s\n' "$STAGE"
  exit 0
fi

mkdir -p "$DEST"
[ -d "$DEST" ] && [ ! -L "$DEST" ] || die "destination is not a real directory: $DEST"
DEST="$(cd "$DEST" && pwd -P)"
TARGET="$DEST/$PLUGIN_ID"

if command -v pgrep >/dev/null 2>&1 && pgrep -x Resolve >/dev/null 2>&1 && [ "${HAWA_ACTIVATE_ALLOW_RUNNING:-0}" != "1" ]; then
  die "DaVinci Resolve is running; close it before installing or upgrading the plugin"
fi

# Recover the only two power-loss states that can leave a transaction behind.
shopt -s nullglob
stale=("$DEST/.${PLUGIN_ID}.transaction."*)
shopt -u nullglob
[ "${#stale[@]}" -le 1 ] || die "multiple stale installer transactions need manual inspection"
if [ "${#stale[@]}" -eq 1 ]; then
  old_txn="${stale[0]}"
  if [ ! -e "$TARGET" ] && [ -d "$old_txn/previous" ] && is_owned_plugin "$old_txn/previous"; then
    say "Recovering the prior plugin from an interrupted transaction"
    mv "$old_txn/previous" "$TARGET"
  elif [ ! -e "$TARGET" ]; then
    die "stale transaction has no recoverable previous plugin: $old_txn"
  elif verify_tree "$TARGET"; then
    say "Completing an interrupted activation whose installed plugin verifies"
  elif [ -d "$old_txn/previous" ] && is_owned_plugin "$old_txn/previous" && verify_tree "$old_txn/previous" && is_owned_plugin "$TARGET"; then
    say "Restoring the prior plugin after an interrupted activation failed verification"
    mv "$TARGET" "$old_txn/failed-candidate"
    mv "$old_txn/previous" "$TARGET"
  else
    die "stale transaction and installed plugin need manual inspection: $old_txn"
  fi
  remove_transaction_tree "$old_txn"
fi

verify_tree "$STAGE" || die "staged plugin failed content verification"

if [ -e "$TARGET" ]; then
  [ -d "$TARGET" ] && [ ! -L "$TARGET" ] || die "existing target is not a real directory"
  is_owned_plugin "$TARGET" || die "existing target is not recognizably owned by HawaVoClean"
  if verify_tree "$TARGET" && cmp -s "$STAGE/SHA256SUMS" "$TARGET/SHA256SUMS" && cmp -s "$STAGE/SYMLINKS" "$TARGET/SYMLINKS"; then
    say "Exact plugin version is already installed; nothing changed"
    printf '%s\n' "$TARGET"
    exit 0
  fi
fi

TRANSACTION="$(mktemp -d "$DEST/.${PLUGIN_ID}.transaction.XXXXXX")"
ACTIVE=0
HAD_PREVIOUS=0

rollback_on_exit() {
  local status="$?"
  trap - EXIT
  if [ "$status" -ne 0 ]; then
    printf 'error: activation failed; restoring the prior plugin\n' >&2
    if [ "$ACTIVE" -eq 1 ] && [ -e "$TARGET" ]; then
      mv "$TARGET" "$TRANSACTION/failed-candidate" || true
    fi
    if [ "$HAD_PREVIOUS" -eq 1 ] && [ -d "$TRANSACTION/previous" ] && [ ! -e "$TARGET" ]; then
      mv "$TRANSACTION/previous" "$TARGET" || true
    fi
  fi
  remove_transaction_tree "$TRANSACTION"
  exit "$status"
}
trap rollback_on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

mkdir "$TRANSACTION/candidate"
cp -R "$STAGE/." "$TRANSACTION/candidate/"
chmod -R a+rX "$TRANSACTION/candidate"
verify_tree "$TRANSACTION/candidate" || die "copied candidate failed content verification"
sync

if [ -e "$TARGET" ]; then
  mv "$TARGET" "$TRANSACTION/previous"
  HAD_PREVIOUS=1
fi
[ "${HAWA_INSTALL_FAILPOINT:-}" != "after_backup" ] || die "injected failure after backup"

mv "$TRANSACTION/candidate" "$TARGET"
ACTIVE=1
sync
[ "${HAWA_INSTALL_FAILPOINT:-}" != "after_activate" ] || die "injected failure after activation"
if [ "${HAWA_INSTALL_FAILPOINT:-}" = "corrupt_after_activate" ]; then
  printf '\nINJECTED CORRUPTION\n' >> "$TARGET/main.js"
fi
verify_tree "$TARGET" || die "activated plugin failed final verification"
[ "${HAWA_INSTALL_FAILPOINT:-}" != "after_verify" ] || die "injected failure after verification"

ACTIVE=0
HAD_PREVIOUS=0
trap - EXIT
remove_transaction_tree "$TRANSACTION"
say "Installed and verified HawaVoClean $(tr -d '\r\n' < "$TARGET/VERSION")"
printf '%s\n' "$TARGET"
