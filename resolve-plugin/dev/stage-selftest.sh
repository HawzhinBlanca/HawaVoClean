#!/usr/bin/env bash
# Prove that an immutable staged plugin can start, authenticate and stop its engine.
set -euo pipefail

[ "$#" -eq 1 ] || { echo "usage: stage-selftest.sh STAGED_PLUGIN" >&2; exit 2; }
STAGE="$1"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_PLUGIN="$(cd "$HERE/../com.hawavoclean.resolve" && pwd)"
ELECTRON="$SOURCE_PLUGIN/node_modules/.bin/electron"
[ -x "$ELECTRON" ] || { echo "exact Electron test runtime is not installed" >&2; exit 1; }
[ -f "$STAGE/engine.json" ] && [ -f "$STAGE/index.html" ] || { echo "stage is incomplete" >&2; exit 1; }

LOG="$(mktemp)"
cleanup() { command rm -f "$LOG"; }
trap cleanup EXIT

(cd "$STAGE" && env HAWA_SELFTEST=1 ELECTRON_ENABLE_LOGGING=0 "$ELECTRON" . >"$LOG" 2>&1) &
PID="$!"
WAITED=0
while kill -0 "$PID" 2>/dev/null; do
  if [ "$WAITED" -ge 90 ]; then
    kill -9 "$PID" 2>/dev/null || true
    echo "staged Electron self-test exceeded 90 seconds" >&2
    sed 's/^/  | /' "$LOG" >&2
    exit 1
  fi
  sleep 1
  WAITED=$((WAITED + 1))
done
wait "$PID" 2>/dev/null || true

RESULT="$(grep '^HAWA_SELFTEST_RESULT ' "$LOG" | head -1 | sed 's/^HAWA_SELFTEST_RESULT //')"
[ -n "$RESULT" ] || { echo "staged shell emitted no success result" >&2; sed 's/^/  | /' "$LOG" >&2; exit 1; }
ENGINE_PID="$(node - "$RESULT" <<'JS'
const value = JSON.parse(process.argv[2]);
if (!value.hasBridge) throw new Error('preload bridge missing');
if (value.host !== 'electron') throw new Error(`unexpected standalone host ${value.host}`);
if (Object.keys(value.endpoint ?? {}).join(',') !== 'baseUrl' || value.endpointHasCredential !== false) throw new Error('renderer endpoint exposed credential material');
if (value.health?.status !== 200 || value.health?.body?.ok !== true) throw new Error('authenticated health check failed');
if (value.sessionBootstrapBlocked !== true) throw new Error('renderer could reach session bootstrap');
if (value.runtime?.electron !== '43.4.1') throw new Error(`unexpected Electron runtime ${value.runtime?.electron}`);
for (const probe of ['inlineScriptBlocked', 'remoteFetchBlocked', 'popupBlocked', 'geolocationDenied', 'pathTraversalBlocked', 'serviceWorkerBlocked', 'blobWorkerRan']) {
  if (value.security?.[probe] !== true) throw new Error(`security probe failed: ${probe}`);
}
if (!value.security.csp.includes("object-src 'none'") || value.security.csp.includes("'unsafe-eval'")) {
  throw new Error('staged CSP is missing a required boundary');
}
if ((value.securityEvents?.blockedWindows ?? 0) < 1 || (value.securityEvents?.blockedPermissions ?? 0) < 1) {
  throw new Error('main process did not observe denied window/permission requests');
}
if (!Number.isInteger(value.enginePid) || value.enginePid <= 1) throw new Error('engine PID missing');
process.stdout.write(String(value.enginePid));
JS
)"
if kill -0 "$ENGINE_PID" 2>/dev/null; then
  echo "engine $ENGINE_PID survived staged shell shutdown" >&2
  exit 1
fi
grep -q 'engine exited' "$LOG" || { echo "shell did not observe engine exit" >&2; exit 1; }
echo "== staged shell self-test passed (engine $ENGINE_PID stopped)"
