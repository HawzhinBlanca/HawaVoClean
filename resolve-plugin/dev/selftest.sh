#!/usr/bin/env bash
# Standalone smoke test for the HawaVoClean shell using the real Electron and the fake engine.
#
#   resolve-plugin/dev/selftest.sh            # runs all scenarios
#   resolve-plugin/dev/selftest.sh ok crash   # runs a subset
#
# Scenarios (FAKE_ENGINE_MODE):
#   ok        engine starts; the renderer gets the endpoint through window.hawa, calls
#             /api/health with the token, gets 401 without it; engine is gone after quit.
#   crash     engine dies before ready -> error page (fast fail, no 60 s wait).
#   hang      engine never reports ready -> ready timeout (shortened to 2 s) -> error page.
#   stubborn  engine ignores /api/shutdown and SIGTERM -> shell must SIGKILL it on quit.
#
# Requires: `npm ci` done in resolve-plugin/com.hawavoclean.resolve (Electron 43.4.1).
# Creates a temporary engine.json + index.html next to main.js and removes them afterwards
# (pre-existing ones are backed up and restored).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(cd "$HERE/../com.hawavoclean.resolve" && pwd)"
FAKE_ENGINE="$HERE/fake-engine.mjs"
NODE_BIN="$(command -v node)"
ELECTRON_BIN="$PLUGIN_DIR/node_modules/.bin/electron"
SCENARIOS=("$@")
[ ${#SCENARIOS[@]} -eq 0 ] && SCENARIOS=(ok crash hang stubborn)

if [ ! -x "$ELECTRON_BIN" ]; then
  echo "electron not installed; run: npm --prefix $PLUGIN_DIR ci" >&2
  exit 1
fi

ENGINE_JSON="$PLUGIN_DIR/engine.json"
INDEX_HTML="$PLUGIN_DIR/index.html"
BACKUP_DIR="$(mktemp -d)"
[ -f "$ENGINE_JSON" ] && cp "$ENGINE_JSON" "$BACKUP_DIR/engine.json"
[ -f "$INDEX_HTML" ] && cp "$INDEX_HTML" "$BACKUP_DIR/index.html"

cleanup() {
  command rm -f "$ENGINE_JSON" "$INDEX_HTML"
  [ -f "$BACKUP_DIR/engine.json" ] && cp "$BACKUP_DIR/engine.json" "$ENGINE_JSON"
  [ -f "$BACKUP_DIR/index.html" ] && cp "$BACKUP_DIR/index.html" "$INDEX_HTML"
  [ ! -d "$BACKUP_DIR" ] || find "$BACKUP_DIR" -depth -delete
}
trap cleanup EXIT

cat > "$ENGINE_JSON" <<EOF
{"command": ["$NODE_BIN", "$FAKE_ENGINE", "serve"], "cwd": "$HERE", "env": {"FAKE_ENGINE_MARKER": "selftest"}}
EOF
cat > "$INDEX_HTML" <<'EOF'
<!doctype html><html><head><meta charset="utf-8"><title>HawaVoClean selftest</title></head>
<body style="background:#0e1013;color:#ddd;font-family:sans-serif"><h1>HawaVoClean shell self-test page</h1>
<p id="s">loading</p><script>document.getElementById('s').textContent = 'hawa host: ' + (window.hawa ? window.hawa.host : 'none');</script></body></html>
EOF

# run_electron <timeout_s> <logfile> [ENV=VAL ...]
run_electron() {
  local timeout_s="$1" logfile="$2"; shift 2
  ( cd "$PLUGIN_DIR" && env "$@" HAWA_SELFTEST=1 ELECTRON_ENABLE_LOGGING=0 "$ELECTRON_BIN" . >"$logfile" 2>&1 ) &
  local pid=$! waited=0
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$waited" -ge "$timeout_s" ]; then
      echo "  !! electron still running after ${timeout_s}s; killing" >&2
      kill -9 "$pid" 2>/dev/null || true
      return 124
    fi
    sleep 1; waited=$((waited + 1))
  done
  wait "$pid" 2>/dev/null || true
  return 0
}

json_field() { # json_field <json> <js-expr-on-o>
  "$NODE_BIN" -e 'const o=JSON.parse(process.argv[1]); const v=eval(process.argv[2]); process.stdout.write(v===undefined?"":String(typeof v==="object"?JSON.stringify(v):v))' "$1" "$2"
}

engine_pids() { pgrep -f "fake-engine.mjs serve" || true; }

fail=0
pass() { echo "  PASS: $*"; }
failmsg() { echo "  FAIL: $*"; fail=1; }

for sc in "${SCENARIOS[@]}"; do
  echo "== scenario: $sc"
  log="$(mktemp)"
  if [ -n "$(engine_pids)" ]; then echo "  (stray fake engines from a previous run: $(engine_pids); killing)"; pkill -9 -f "fake-engine.mjs serve" || true; sleep 0.5; fi
  case "$sc" in
    ok)
      run_electron 60 "$log" FAKE_ENGINE_MODE=ok || true
      line="$(grep '^HAWA_SELFTEST_RESULT ' "$log" | head -1 | sed 's/^HAWA_SELFTEST_RESULT //')"
      hostline="$(grep '^HAWA_SELFTEST_HOST ' "$log" | head -1 | awk '{print $2}')"
      if [ -z "$line" ]; then failmsg "no HAWA_SELFTEST_RESULT line"; sed 's/^/    | /' "$log"; else
        echo "  result: $line"
        [ "$(json_field "$line" 'o.hasBridge')" = "true" ] && pass "window.hawa present" || failmsg "window.hawa missing"
        [ "$(json_field "$line" 'o.host')" = "electron" ] && pass "host === 'electron' (Resolve absent)" || failmsg "host = $(json_field "$line" 'o.host')"
        [ "$hostline" = "electron" ] && pass "main reports host electron" || failmsg "main host line: $hostline"
        [ "$(json_field "$line" 'o.hasResolve')" = "false" ] && pass "no resolve bridge in electron host" || failmsg "resolve bridge unexpectedly present"
        [ "$(json_field "$line" 'o.keys')" = '["engine","files","host"]' ] && pass "bridge keys = engine,files,host" || failmsg "bridge keys = $(json_field "$line" 'o.keys')"
        [ "$(json_field "$line" 'o.pathForFileNull')" = "true" ] && pass "pathForFile(in-memory File) -> null" || failmsg "pathForFile did not return null"
        base="$(json_field "$line" 'o.endpoint.baseUrl')"; tok="$(json_field "$line" 'o.endpoint.token')"
        [[ "$base" =~ ^http://127\.0\.0\.1:[0-9]+$ ]] && pass "endpoint baseUrl $base" || failmsg "bad baseUrl: $base"
        [[ "$tok" =~ ^[0-9a-f]{32}$ ]] && pass "token is 32 hex" || failmsg "bad token: $tok"
        [ "$(json_field "$line" 'o.health.status')" = "200" ] && [ "$(json_field "$line" 'o.health.body.ok')" = "true" ] && pass "renderer fetched /api/health with token (200 ok)" || failmsg "health: $(json_field "$line" 'o.health')"
        [ "$(json_field "$line" 'o.unauthStatus')" = "401" ] && pass "no token -> 401" || failmsg "unauth status $(json_field "$line" 'o.unauthStatus')"
        epid="$(json_field "$line" 'o.enginePid')"
        if [ -n "$epid" ] && ! kill -0 "$epid" 2>/dev/null; then pass "engine pid $epid is gone after quit"; else failmsg "engine pid $epid still alive"; fi
        grep -q 'shutdown requested; exiting' "$log" && pass "engine received POST /api/shutdown" || failmsg "no shutdown log line"
      fi
      ;;
    crash)
      run_electron 30 "$log" FAKE_ENGINE_MODE=crash || true
      line="$(grep '^HAWA_SELFTEST_ERROR ' "$log" | head -1 | sed 's/^HAWA_SELFTEST_ERROR //')"
      if [ -z "$line" ]; then failmsg "no HAWA_SELFTEST_ERROR line"; sed 's/^/    | /' "$log"; else
        echo "  error: $line"
        echo "$(json_field "$line" 'o.detail')" | grep -q 'exit code 3' && pass "fast fail on early exit (code 3)" || failmsg "detail: $(json_field "$line" 'o.detail')"
        echo "$(json_field "$line" 'o.stderrTail')" | grep -q 'ModuleNotFoundError' && pass "stderr tail captured for the error page" || failmsg "stderr tail missing"
        grep -q '^HAWA_SELFTEST_RESULT' "$log" && failmsg "unexpected RESULT line" || pass "UI not loaded on failure"
      fi
      ;;
    hang)
      run_electron 30 "$log" FAKE_ENGINE_MODE=hang HAWA_ENGINE_TIMEOUT_MS=2000 || true
      line="$(grep '^HAWA_SELFTEST_ERROR ' "$log" | head -1 | sed 's/^HAWA_SELFTEST_ERROR //')"
      if [ -z "$line" ]; then failmsg "no HAWA_SELFTEST_ERROR line"; sed 's/^/    | /' "$log"; else
        echo "  error: $line"
        echo "$(json_field "$line" 'o.detail')" | grep -q 'did not report ready within 2 s' && pass "ready timeout fired" || failmsg "detail: $(json_field "$line" 'o.detail')"
      fi
      sleep 1
      [ -z "$(engine_pids)" ] && pass "hung engine killed after timeout" || { failmsg "hung engine still alive: $(engine_pids)"; pkill -9 -f "fake-engine.mjs serve" || true; }
      ;;
    stubborn)
      run_electron 60 "$log" FAKE_ENGINE_MODE=stubborn || true
      line="$(grep '^HAWA_SELFTEST_RESULT ' "$log" | head -1 | sed 's/^HAWA_SELFTEST_RESULT //')"
      if [ -z "$line" ]; then failmsg "no HAWA_SELFTEST_RESULT line"; sed 's/^/    | /' "$log"; else
        epid="$(json_field "$line" 'o.enginePid')"
        grep -q 'ignored SIGTERM; sending SIGKILL' "$log" && pass "shell escalated to SIGKILL" || failmsg "no SIGKILL escalation logged"
        if [ -n "$epid" ] && ! kill -0 "$epid" 2>/dev/null; then pass "stubborn engine pid $epid is gone after quit"; else failmsg "stubborn engine $epid survived"; pkill -9 -f "fake-engine.mjs serve" || true; fi
      fi
      ;;
    *) failmsg "unknown scenario $sc" ;;
  esac
  rm -f "$log"
done

[ -z "$(engine_pids)" ] && echo "== no orphan fake engines left" || { echo "== ORPHANS: $(engine_pids)"; fail=1; }
if [ "$fail" -eq 0 ]; then echo "== ALL PASS"; else echo "== FAILURES"; exit 1; fi
