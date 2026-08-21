# HawaVoClean UI contract (v1)

One web UI bundle, three shells, one engine. This document is the binding contract
between the three parts. Anything not written here is the implementer's call, but
every name, path and shape written here is fixed.

## Parts and owners

| Part | Location | Runtime |
|------|----------|---------|
| Engine bridge | `src/hawavoclean/server/` + `hawavoclean serve` + `hawavoclean process --progress-json` | Python (FastAPI + uvicorn), binds 127.0.0.1 only |
| UI bundle | `ui/` (Vite 8 + React 19 + TypeScript), builds to `ui/dist/` with `base: './'` | Supported browser, the pinned standalone Electron test runtime, or Resolve's vendor-owned embedded runtime |
| Resolve / desktop shell | `resolve-plugin/com.hawavoclean.resolve/` (`manifest.xml`, `package.json`, `main.js`, `preload.js`) + `resolve-plugin/install.sh` | DaVinci Resolve Studio 21 host, or exact standalone Electron 43.4.1 for controlled testing |

## 1. Engine HTTP API

Base URL: `http://127.0.0.1:{port}`. The server prints exactly one JSON line to **stdout** when
it is listening and then nothing else on stdout (all logs go to stderr):

```json
{"event":"ready","port":54321,"pid":12345,"version":"3.3.0"}
```

CLI: `hawavoclean serve [--host 127.0.0.1] [--port 0] --token TOKEN [--ui-dir DIR]`.
`--port 0` = OS-assigned. `--ui-dir` = directory containing `index.html` + `assets/` to serve at `/`
(optional; when absent `/` returns 404 JSON). Token: every `/api/*` request must carry the token as
header `X-Hawa-Token: TOKEN` **or** query `?token=TOKEN` (query form exists for `EventSource` and
`<audio src>` which cannot set headers). Missing/wrong token → 401 `{"error":"unauthorized"}`.
CORS: allow all origins (`*`), all methods, headers `X-Hawa-Token, Content-Type`. The UI in Resolve
loads from `file://` (origin `null`), so this is required.

All errors: JSON `{"error": "<code>", "message": "<human text>"}` with 4xx/5xx.

### `GET /api/health`
```json
{"ok":true,"version":"3.3.0","profiles":["studio","lowband","production"],"engine_pid":12345}
```

### `POST /api/analyze`
Request `{"path": "/abs/file.wav", "buckets": 1200}` (buckets optional, default 1200, max 8000).
Response `AudioAnalysis`:
```json
{
  "path": "/abs/file.wav",
  "duration_s": 61.2, "sample_rate": 48000, "channels": 1,
  "peaks": {"min": [..buckets floats in -1..1..], "max": [..]},
  "rms_db": [..buckets floats, dBFS, -120 for silence..],
  "spectrum": {"freqs_hz": [..N..], "db": [..N..]},
  "loudness": {"integrated_lufs": -23.1, "true_peak_dbtp": -3.2},
  "noise_floor_db": -52.3
}
```
Rules: mono mix (mean of channels) for peaks/rms/spectrum. Spectrum = long-term average magnitude,
1/12-octave bands from 40 Hz to min(20 kHz, nyquist), in dB relative to full scale (a full-scale
sine at a band centre ≈ 0 dB); values below −120 clamp to −120. `noise_floor_db` = 10th-percentile
of the per-bucket `rms_db` over buckets above −120. Loudness via the existing
`hawavoclean.finishing.loudness.measure_loudness_and_peaks`. Decoding via the existing
`hawavoclean.audio` decode path (so any container ffmpeg can read works).

### `POST /api/jobs`
Request `{"input_path": "/abs/in.m4a.mp4", "profile": "studio", "output_path": "/abs/out.wav" (optional), "overwrite": false}`
Default `output_path`: same directory as input, `<stem>_studio.wav` for `studio`,
`<stem>_lowband.wav` for `lowband`, and `<stem>_clean.wav` for `production`, where `<stem>` strips
**all** audio/container suffixes (`Flute 09.m4a.mp4` → `Flute 09`, same rule as
`cli._clean_stem`). Response `202`:
```json
{"job_id":"j_8f2a...","output_path":"/abs/Flute 09_studio.wav","report_path":"/abs/Flute 09_studio.hawavoclean.json"}
```
Execution: the server spawns a **child process** `python -m hawavoclean.cli process IN -o OUT --profile P [--overwrite] --progress-json`
(same isolation pattern as `cli._run_one_isolated`), reads progress JSON lines from the child's stdout,
and tails stderr into `message` on failure. One job runs at a time; additional jobs queue (FIFO).

### `GET /api/jobs/{job_id}` → `JobStatus`
```json
{
  "job_id":"j_8f2a...",
  "state":"queued|running|done|failed|cancelled",
  "stage":"preflight|decode|segment|enhance|guard|finish|publish|done|error",
  "progress":0.0,                      // 0..1 overall
  "message":"Enhancing unit 3/5",
  "unit":{"index":3,"total":5},        // optional
  "input_path":"...","output_path":"...","report_path":"...",
  "profile":"studio",
  "started_at":"2026-08-19T12:00:00Z","finished_at":null,
  "error":{"code":"PREFLIGHT_FAILURE","message":"..."},   // only when failed
  "report":{...HawaVoCleanReport JSON...}                 // only when done
}
```
Unknown id → 404.

### `GET /api/jobs/{job_id}/events?token=` (Server-Sent Events)
`Content-Type: text/event-stream`. On connect: one `event: status` with the current `JobStatus`, then one
`event: status` per change (throttle to ≥50 ms apart, but never drop the final one), then
`event: end` with `{}` after the job reaches done/failed/cancelled, then close. Keep-alive comment
`: ping` every 15 s.

### `POST /api/jobs/{job_id}/cancel`
Terminates the child (SIGTERM, SIGKILL after 5 s). Status becomes `cancelled`. `200 {"ok":true}`.
Cancelling a finished job is a no-op 200.

### `GET /api/audio?path=&token=`
Streams the file with HTTP Range support (206) and a sensible `Content-Type` (`audio/wav`,
`audio/mp4` for `.m4a/.mp4`, `audio/mpeg`, `audio/flac`, `audio/aac`, `video/quicktime` for `.mov`).
Path policy (also applies to `/api/analyze` and `POST /api/jobs`): absolute path, must resolve under
`Path.home()` **or** `/Volumes` **or** the HawaVoClean work dir; otherwise 403. Missing file → 404.

### `POST /api/upload` (multipart, field `file`)
Saves to `<work_dir>/uploads/<uuid>/<original name>` and returns `{"path": "/abs/..."}`. Web-only fallback
(browsers cannot give local paths).

### `POST /api/shutdown`
Responds `{"ok":true}` then exits the server process within 1 s.

## 2. `hawavoclean process --progress-json`

New flag on the existing `process` subcommand. When set, the process emits **one JSON object per line on
stdout** and nothing else on stdout (human logs stay on stderr):

```json
{"event":"progress","stage":"preflight","progress":0.02,"message":"Preflight checks passed"}
{"event":"progress","stage":"decode","progress":0.05,"message":"Decoded 61.2 s @ 48 kHz, 1 ch"}
{"event":"progress","stage":"segment","progress":0.08,"message":"5 units"}
{"event":"progress","stage":"enhance","progress":0.23,"message":"Enhancing unit 1/5","unit":{"index":1,"total":5}}
{"event":"progress","stage":"guard","progress":0.31,"message":"Unit 1/5: ENHANCED","unit":{"index":1,"total":5}}
{"event":"progress","stage":"finish","progress":0.85,"message":"Finishing: EQ/limiter/loudness"}
{"event":"progress","stage":"publish","progress":0.98,"message":"Publishing master"}
{"event":"done","progress":1.0,"output_path":"/abs/out.wav","report_path":"/abs/out.hawavoclean.json"}
```
or on failure `{"event":"error","code":"<ExitCode name>","message":"..."}` (exit code unchanged).
Progress weights: preflight→0.02, decode→0.05, segment→0.08, enhancement+guard spans 0.08→0.80
linearly over units, finish→0.80–0.95, publish→0.98, done→1.0. Implemented via a new optional
`on_progress: Callable[[ProgressEvent], None] | None` parameter on `run_pipeline` (and threaded into
`_run_after_preflight`); `ProgressEvent` is a small frozen dataclass in `hawavoclean/progress.py`.
Callback exceptions must never break the pipeline (catch, log, continue).

## 3. Renderer bridge: `window.hawa`

Provided by the Electron preload via `contextBridge.exposeInMainWorld('hawa', …)`. In a plain browser
`window.hawa` is undefined and the UI builds a `web` fallback (engine = `location.origin`, token from
`?token=` query or `localStorage['hawa.token']`). TypeScript shape (authoritative copy lives in
`ui/src/bridge/types.ts`):

```ts
export type HawaHost = 'resolve' | 'electron' | 'web';
export interface ResolveClip { mediaId: string; name: string; filePath: string; durationS?: number; }
export interface HawaBridge {
  host: HawaHost;
  engine: { getEndpoint(): Promise<{ baseUrl: string; token: string }> };
  files: {
    pickAudio(): Promise<string | null>;            // native open dialog → absolute path
    pathForFile(file: File): string | null;         // dropped File → absolute path (Electron webUtils)
    revealInFinder(path: string): Promise<void>;
  };
  resolve?: {                                        // present only when host === 'resolve'
    getSelectedClip(): Promise<ResolveClip | null>;  // MediaPool.GetSelectedClips()[0], else current video item's MediaPoolItem
    importMedia(path: string): Promise<ResolveClip | null>;        // MediaPool.ImportMedia([path])[0]
    replaceClip(mediaId: string, path: string): Promise<boolean>;  // ReplaceClipPreserveSubClip, fallback ReplaceClip
    appendToTimeline(mediaId: string): Promise<boolean>;
    getContext(): Promise<{ project: string | null; timeline: string | null; page: string | null }>;
  };
}
```
IPC channel names (main ⇄ preload): `hawa:engine:endpoint`, `hawa:files:pick`, `hawa:files:reveal`,
`hawa:resolve:selected`, `hawa:resolve:import`, `hawa:resolve:replace`, `hawa:resolve:append`,
`hawa:resolve:context`. All `ipcMain.handle` (promise-based). Errors are thrown as `Error(message)`.

## 4. Shell behaviour (main.js)

* Reads `engine.json` next to `main.js`. The shipped file is relocatable:
  `{"command":["./engine/hawavoclean-engine","serve"],"cwd":".","env":{"PYTHONNOUSERSITE":"1","PYTHONDONTWRITEBYTECODE":"1"}}`.
  Relative executable and working-directory paths are resolved below the plugin directory and may not
  escape it. Absolute paths remain available only for explicit developer configurations.
  Spawns `command + ["--port","0","--token",TOKEN,"--ui-dir",__dirname]` with a fresh random 32-hex TOKEN,
  waits for the `ready` stdout line (timeout 60 s → show an error page with the stderr tail).
* Registers a standard secure `hawa://app` protocol on a private in-memory session. Its handler serves
  only canonical regular files below the plugin root; CORS is enabled while service workers and CSP
  bypass are disabled. `BrowserWindow` loads `hawa://app/index.html` at 1280×820 (min 960×640), dark
  background `#0e1013`.
* Window preferences are fixed: `sandbox:true`, `contextIsolation:true`, `nodeIntegration:false`,
  `webSecurity:true`, `allowRunningInsecureContent:false`, `webviewTag:false`. All popups, foreign
  navigation and webview attachments are denied. Renderer requests are limited to the app protocol,
  its confined backing files, and exactly the spawned `http://127.0.0.1:<port>` engine.
* The session denies all device permissions and every permission except sanitized clipboard write
  from the exact main renderer. Every privileged IPC handler validates the exact main frame and app
  URL before acting. `HAWA_DEVTOOLS=1` opens devtools for explicit standalone diagnostics.
* `WorkflowIntegration.node` is `require`d in **main** (sandboxed preload cannot load native modules);
  `Initialize('com.hawavoclean.resolve')` failing ⇒ `host='electron'` and no `resolve` bridge (standalone run).
  Registers `ResolveQuit` callback → quit.
* On quit: `POST /api/shutdown`, then SIGTERM/SIGKILL the child after 3 s; `WorkflowIntegration.CleanUp()`.

## 5. Files on disk

```
resolve-plugin/
  com.hawavoclean.resolve/      # shell sources (no built UI, no .node committed)
    manifest.xml  package.json  main.js  preload.js  engine.json.example
  install.sh                    # locked build, immutable staging and staged lifecycle self-test
  activate.sh                   # privileged, build-tool-free transactional activation + rollback
scripts/build_resolve_engine.py # exact-wheel → relocatable macOS arm64 CPython 3.11 engine
ui/                             # Vite + React + TS sources; ui/dist is gitignored
src/hawavoclean/server/         # app.py (FastAPI), jobs.py (child-process job manager), analysis.py, auth/paths helpers
src/hawavoclean/progress.py     # ProgressEvent + ProgressCallback types
```
The content-addressed stage and installed plugin contain
`manifest.xml package.json main.js preload.js engine.json index.html assets/ engine/ PLUGIN_ID VERSION
SHA256SUMS SYMLINKS`, plus `WorkflowIntegration.node` for a Resolve build. The installed directory is
`/Library/Application Support/Blackmagic Design/DaVinci Resolve/Workflow Integration Plugins/com.hawavoclean.resolve/`:
the engine is self-contained and does not refer to the source checkout or a mutable virtual environment.

`manifest.xml`:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<BlackmagicDesign>
  <Plugin>
    <Id>com.hawavoclean.resolve</Id>
    <Name>HawaVoClean</Name>
    <Version>3.3.0</Version>
    <Description>HawaVoClean voice restoration for DaVinci Resolve Studio.</Description>
    <FilePath>main.js</FilePath>
  </Plugin>
</BlackmagicDesign>
```

## 6. Non-negotiables

* Engine binds 127.0.0.1 only, token required, path policy enforced. No arbitrary file reads.
* The Resolve-owned Electron version is not inferred from the standalone lock. Capture and assess it
  separately as specified in `docs/resolve-runtime-risk.md`; never describe a vulnerable host as clean.
* The UI never drives per-frame drawing through React state: waveform in a Worker via `OffscreenCanvas`
  (WebGL2), spectrum/analyser via `requestAnimationFrame` + refs. React owns layout/controls only.
* Dependencies: Python extra `ui = [fastapi, uvicorn, python-multipart]`; UI deps: react, react-dom,
  motion (MIT); no UI kit, no Tailwind. Everything must work offline (no Google Fonts, no CDN).
* All existing gates stay green: `ruff check`, `ruff format --check`, `mypy --strict src`, `pytest`
  (default `-m 'not fuzz'`), coverage ≥ 90 % branch for touched modules, `scripts/mutation_gate.py`.

---

## Addendum 1 — windowed peaks (`POST /api/peaks`)

Added for goal box E3/B3 (deep zoom must show true sample detail, not interpolated buckets).

Request:
```json
{"path": "/abs/file.wav", "start_s": 12.0, "end_s": 18.5, "buckets": 1600}
```
* `path` — same path policy as `/api/analyze` (403/404 otherwise).
* `start_s` / `end_s` — floats, `0 <= start_s < end_s`. `end_s` is clamped to the file duration;
  `start_s >= duration` → 400 `{"error":"bad_request"}`.
* `buckets` — 1..8000, default 1200. Clamped down to the number of samples in the window.

Response `PeaksWindow`:
```json
{
  "path": "/abs/file.wav",
  "start_s": 12.0, "end_s": 18.5,
  "sample_rate": 48000, "channels": 1, "duration_s": 94.6,
  "samples_per_bucket": 195,
  "peaks": {"min": [..buckets floats -1..1..], "max": [..buckets floats..]},
  "rms_db": [..buckets floats, dBFS, -120 for silence..]
}
```
Semantics identical to `/api/analyze`'s waveform fields (mono mix, every bucket covers ≥1 sample),
but computed over the requested window only.

**Memory rule (normative):** the handler MUST NOT decode the whole file to serve a window. Decode
only the requested span (ffmpeg `-ss <start> -t <len>` before `-i` for fast seek, or
`soundfile.read(start=, stop=)` on the fallback path). Peak RSS for a 5-second window out of a
3-hour file must stay within a few MB of the idle server (measured, not assumed).

`samples_per_bucket` lets the client decide when it has reached 1 sample/bucket (no more detail to
fetch). Clients should re-query on zoom/pan and cache by `(path, start_s, end_s, buckets)`.
