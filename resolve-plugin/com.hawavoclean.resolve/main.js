'use strict';
/*
 * HawaVoClean — DaVinci Resolve Workflow Integration / desktop shell (main process).
 *
 * Contract: docs/ui-contract.md sections 3, 4 and 5.
 *
 *  - Spawns the Python engine (`hawavoclean serve`) described by engine.json (next to this
 *    file), waits for its single `{"event":"ready",...}` stdout line, and hands the renderer
 *    the endpoint + token over IPC (`hawa:engine:endpoint`).
 *  - Loads index.html (the built UI bundle) into a sandboxed, context-isolated BrowserWindow.
 *  - Talks to Resolve through WorkflowIntegration.node (required here in main; the sandboxed
 *    preload cannot load native modules). If the module is missing or Resolve is not running
 *    the shell degrades to host 'electron' and exposes no `resolve` bridge.
 *  - On quit: POST /api/shutdown, then SIGTERM, SIGKILL after 3 s, WorkflowIntegration.CleanUp().
 *
 * Only Node/Electron built-ins are used at runtime (no npm dependencies).
 *
 * Environment knobs:
 *   HAWA_DEVTOOLS=1            open DevTools
 *   HAWA_SELFTEST=1            headless smoke test: load UI, query the bridge, print a result line, quit
 *   HAWA_ENGINE_TIMEOUT_MS=N   override the 60 s ready timeout (tests only)
 */

const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const path = require('node:path');
const fs = require('node:fs');
const os = require('node:os');
const http = require('node:http');
const crypto = require('node:crypto');
const { spawn } = require('node:child_process');

const PLUGIN_ID = 'com.hawavoclean.resolve';
const APP_TITLE = 'HawaVoClean';
const ENGINE_READY_TIMEOUT_MS = positiveInt(process.env.HAWA_ENGINE_TIMEOUT_MS, 60_000);
const RESOLVE_INIT_TIMEOUT_MS = 8_000;
const SHUTDOWN_POST_TIMEOUT_MS = 1_000;
const SHUTDOWN_SIGTERM_GRACE_MS = 3_000;
const QUIT_HARD_DEADLINE_MS = 10_000;
const STDERR_TAIL_LINES = 80;
const DEVTOOLS = process.env.HAWA_DEVTOOLS === '1';
const SELFTEST = process.env.HAWA_SELFTEST === '1';

// Directories where common CLI tools (ffmpeg etc.) live on macOS but which are missing from
// the minimal PATH a GUI-launched process inherits from Resolve.
const EXTRA_PATH_DIRS = ['/opt/homebrew/bin', '/usr/local/bin'];

function positiveInt(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : fallback;
}

function log(...args) {
  // Goes to the Electron main process stderr (visible when run standalone; Resolve discards it).
  console.error('[hawa-shell]', ...args);
}

// ---------------------------------------------------------------------------------------------
// Engine process management
// ---------------------------------------------------------------------------------------------

const engine = {
  config: null, // {command: string[], cwd: string, env: object}
  child: null,
  token: crypto.randomBytes(16).toString('hex'), // fresh random 32-hex token per run
  port: null,
  pid: null,
  version: null,
  stderrTail: [], // last STDERR_TAIL_LINES lines of stderr (plus unexpected stdout)
  ready: null, // Promise<{baseUrl, token}>
  failure: null, // Error once the engine is known to be unusable
  exited: false,
  exitInfo: null, // {code, signal}
  settled: false,
};

let resolveReady = null; // resolve() of engine.ready
let rejectReady = null; // reject() of engine.ready

function pushTail(line) {
  engine.stderrTail.push(line);
  if (engine.stderrTail.length > STDERR_TAIL_LINES) {
    engine.stderrTail.splice(0, engine.stderrTail.length - STDERR_TAIL_LINES);
  }
}

function readEngineConfig() {
  const file = path.join(__dirname, 'engine.json');
  if (!fs.existsSync(file)) {
    throw new Error(
      `engine.json not found at ${file}. Copy engine.json.example to engine.json and point "command" at your hawavoclean executable (or run resolve-plugin/install.sh).`,
    );
  }
  let raw;
  try {
    raw = JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch (err) {
    throw new Error(`engine.json is not valid JSON: ${err.message}`);
  }
  if (!Array.isArray(raw.command) || raw.command.length === 0 || !raw.command.every((s) => typeof s === 'string' && s.length > 0)) {
    throw new Error('engine.json: "command" must be a non-empty array of strings, e.g. ["/abs/.venv/bin/hawavoclean","serve"].');
  }
  const cwd = typeof raw.cwd === 'string' && raw.cwd.length > 0 ? raw.cwd : os.homedir();
  const env = raw.env && typeof raw.env === 'object' && !Array.isArray(raw.env) ? raw.env : {};
  for (const [k, v] of Object.entries(env)) {
    if (typeof v !== 'string') throw new Error(`engine.json: env.${k} must be a string.`);
  }
  return { command: raw.command.slice(), cwd, env, file };
}

function buildEngineEnv(config) {
  const env = { ...process.env, ...config.env };
  const commandDir = path.isAbsolute(config.command[0]) ? path.dirname(config.command[0]) : null;
  const wanted = [commandDir, ...EXTRA_PATH_DIRS].filter(Boolean);
  const current = (env.PATH || '').split(path.delimiter).filter(Boolean);
  const prepend = wanted.filter((d) => !current.includes(d));
  env.PATH = [...prepend, ...current].join(path.delimiter);
  if (!('PYTHONUNBUFFERED' in env)) env.PYTHONUNBUFFERED = '1';
  return env;
}

function failEngine(message) {
  if (engine.settled) return;
  engine.settled = true;
  const err = new Error(message);
  engine.failure = err;
  if (rejectReady) rejectReady(err);
  log('engine failure:', message);
  // Make sure a half-started child does not linger.
  if (engine.child && !engine.exited) {
    try {
      engine.child.kill('SIGTERM');
    } catch {
      /* ignore */
    }
    const c = engine.child;
    setTimeout(() => {
      if (!engine.exited) {
        try {
          c.kill('SIGKILL');
        } catch {
          /* ignore */
        }
      }
    }, SHUTDOWN_SIGTERM_GRACE_MS).unref();
  }
  showErrorPage('The HawaVoClean engine could not be started', message);
}

function handleReadyLine(obj) {
  if (engine.settled) return;
  const port = Number(obj.port);
  if (!Number.isInteger(port) || port <= 0 || port > 65535) {
    failEngine(`Engine reported an invalid port in its ready line: ${JSON.stringify(obj)}`);
    return;
  }
  engine.settled = true;
  engine.port = port;
  engine.pid = Number.isInteger(obj.pid) ? obj.pid : engine.child ? engine.child.pid : null;
  engine.version = typeof obj.version === 'string' ? obj.version : null;
  const endpoint = { baseUrl: `http://127.0.0.1:${port}`, token: engine.token };
  log(`engine ready: ${endpoint.baseUrl} (pid ${engine.pid}, version ${engine.version || '?'})`);
  if (resolveReady) resolveReady(endpoint);
}

function startEngine() {
  engine.ready = new Promise((resolve, reject) => {
    resolveReady = resolve;
    rejectReady = reject;
  });
  // Swallow "unhandled rejection" noise; consumers (IPC handler) attach their own handlers.
  engine.ready.catch(() => {});

  let config;
  try {
    config = readEngineConfig();
  } catch (err) {
    failEngine(err.message);
    return;
  }
  engine.config = config;

  const [exe, ...rest] = config.command;
  const args = [...rest, '--port', '0', '--token', engine.token, '--ui-dir', __dirname];
  log(`spawning engine: ${exe} ${args.map((a) => (a === engine.token ? '<token>' : a)).join(' ')} (cwd ${config.cwd})`);

  let child;
  try {
    child = spawn(exe, args, {
      cwd: config.cwd,
      env: buildEngineEnv(config),
      stdio: ['ignore', 'pipe', 'pipe'],
      windowsHide: true,
    });
  } catch (err) {
    failEngine(`Could not spawn engine (${exe}): ${err.message}`);
    return;
  }
  engine.child = child;

  // ---- stdout: exactly one JSON "ready" line is expected; tolerate stray lines before it. ----
  let stdoutBuf = '';
  child.stdout.setEncoding('utf8');
  child.stdout.on('data', (chunk) => {
    stdoutBuf += chunk;
    let nl;
    while ((nl = stdoutBuf.indexOf('\n')) >= 0) {
      const line = stdoutBuf.slice(0, nl).replace(/\r$/, '');
      stdoutBuf = stdoutBuf.slice(nl + 1);
      if (!line.trim()) continue;
      if (engine.settled) {
        pushTail(`[stdout] ${line}`);
        continue;
      }
      let obj = null;
      try {
        obj = JSON.parse(line);
      } catch {
        obj = null;
      }
      if (obj && typeof obj === 'object' && obj.event === 'ready') {
        handleReadyLine(obj);
      } else {
        pushTail(`[stdout] ${line}`);
      }
    }
    if (stdoutBuf.length > 65536) stdoutBuf = stdoutBuf.slice(-65536);
  });

  // ---- stderr: keep a tail for the error page and mirror to our own stderr. ----
  let stderrBuf = '';
  child.stderr.setEncoding('utf8');
  child.stderr.on('data', (chunk) => {
    process.stderr.write(chunk);
    stderrBuf += chunk;
    let nl;
    while ((nl = stderrBuf.indexOf('\n')) >= 0) {
      pushTail(stderrBuf.slice(0, nl).replace(/\r$/, ''));
      stderrBuf = stderrBuf.slice(nl + 1);
    }
    if (stderrBuf.length > 65536) stderrBuf = stderrBuf.slice(-65536);
  });

  child.on('error', (err) => {
    failEngine(`Could not start engine (${exe}): ${err.message}`);
  });

  child.on('exit', (code, signal) => {
    engine.exited = true;
    engine.exitInfo = { code, signal };
    const how = signal ? `signal ${signal}` : `exit code ${code}`;
    log(`engine exited (${how})`);
    if (!engine.settled) {
      failEngine(`Engine exited (${how}) before it reported ready.`);
    } else if (!engine.failure && quitState === 'running') {
      engine.failure = new Error(`Engine exited unexpectedly (${how}).`);
      showErrorPage('The HawaVoClean engine stopped unexpectedly', engine.failure.message);
    }
  });

  const timer = setTimeout(() => {
    if (!engine.settled) {
      failEngine(`Engine did not report ready within ${Math.round(ENGINE_READY_TIMEOUT_MS / 1000)} s.`);
    }
  }, ENGINE_READY_TIMEOUT_MS);
  timer.unref();
  engine.ready.then(() => clearTimeout(timer), () => clearTimeout(timer));
}

function postShutdown(port, token, timeoutMs) {
  return new Promise((resolve) => {
    let done = false;
    const finish = () => {
      if (!done) {
        done = true;
        resolve();
      }
    };
    try {
      const req = http.request(
        {
          host: '127.0.0.1',
          port,
          method: 'POST',
          path: '/api/shutdown',
          headers: { 'X-Hawa-Token': token, 'Content-Length': '0' },
          timeout: timeoutMs,
        },
        (res) => {
          res.resume();
          res.on('end', finish);
          res.on('error', finish);
        },
      );
      req.on('timeout', () => {
        req.destroy();
        finish();
      });
      req.on('error', finish);
      req.end();
    } catch {
      finish();
    }
    setTimeout(finish, timeoutMs + 200).unref();
  });
}

function waitForExit(timeoutMs) {
  return new Promise((resolve) => {
    if (!engine.child || engine.exited) return resolve(true);
    const child = engine.child;
    const t = setTimeout(() => {
      child.removeListener('exit', onExit);
      resolve(engine.exited);
    }, timeoutMs);
    function onExit() {
      clearTimeout(t);
      resolve(true);
    }
    child.once('exit', onExit);
  });
}

async function stopEngine() {
  const child = engine.child;
  if (!child || engine.exited) return;
  if (engine.port) {
    await postShutdown(engine.port, engine.token, SHUTDOWN_POST_TIMEOUT_MS);
    if (await waitForExit(1_000)) return;
  }
  try {
    child.kill('SIGTERM');
  } catch {
    /* ignore */
  }
  if (await waitForExit(SHUTDOWN_SIGTERM_GRACE_MS)) return;
  log('engine ignored SIGTERM; sending SIGKILL');
  try {
    child.kill('SIGKILL');
  } catch {
    /* ignore */
  }
  await waitForExit(1_000);
}

// ---------------------------------------------------------------------------------------------
// Resolve (WorkflowIntegration.node)
// ---------------------------------------------------------------------------------------------

let WorkflowIntegration = null;
let wiInitialized = false; // Initialize() succeeded — CleanUp() must run on quit even if later steps failed
let resolveObj = null; // Resolve scripting object (promise flavour when available)
let host = 'electron'; // 'resolve' | 'electron'
const knownItems = new Map(); // mediaId -> MediaPoolItem handed out earlier (verified before use)

function loadWorkflowIntegration() {
  const file = path.join(__dirname, 'WorkflowIntegration.node');
  if (!fs.existsSync(file)) {
    log('WorkflowIntegration.node not present; running as standalone Electron app');
    return null;
  }
  try {
    // eslint-disable-next-line import/no-dynamic-require, global-require
    return require(file);
  } catch (err) {
    log(`WorkflowIntegration.node could not be loaded (${err.message}); running standalone`);
    return null;
  }
}

function withTimeout(promise, ms, fallbackValue) {
  return new Promise((resolve) => {
    const t = setTimeout(() => resolve(fallbackValue), ms);
    Promise.resolve(promise).then(
      (v) => {
        clearTimeout(t);
        resolve(v);
      },
      () => {
        clearTimeout(t);
        resolve(fallbackValue);
      },
    );
  });
}

async function initResolve() {
  WorkflowIntegration = loadWorkflowIntegration();
  if (!WorkflowIntegration) return false;
  try {
    let ok = false;
    if (typeof WorkflowIntegration.InitializePromise === 'function') {
      ok = await withTimeout(WorkflowIntegration.InitializePromise(PLUGIN_ID), RESOLVE_INIT_TIMEOUT_MS, false);
    } else if (typeof WorkflowIntegration.Initialize === 'function') {
      ok = WorkflowIntegration.Initialize(PLUGIN_ID);
    }
    if (!ok) {
      log('WorkflowIntegration.Initialize failed (Resolve not running?); running standalone');
      return false;
    }
    wiInitialized = true;
    if (typeof WorkflowIntegration.GetResolvePromise === 'function') {
      resolveObj = await withTimeout(WorkflowIntegration.GetResolvePromise(), RESOLVE_INIT_TIMEOUT_MS, null);
    } else {
      resolveObj = WorkflowIntegration.GetResolve();
    }
    if (!resolveObj) {
      log('WorkflowIntegration.GetResolve returned nothing; running standalone');
      return false;
    }
    try {
      WorkflowIntegration.RegisterCallback('ResolveQuit', () => {
        log('ResolveQuit received; quitting');
        app.quit();
      });
    } catch (err) {
      log(`RegisterCallback(ResolveQuit) failed: ${err.message}`);
    }
    log('connected to DaVinci Resolve');
    return true;
  } catch (err) {
    log(`Resolve initialisation threw (${err.message}); running standalone`);
    resolveObj = null;
    return false;
  }
}

function cleanupResolve() {
  // Gate on "Initialize succeeded", not on host: GetResolve may have failed after a successful
  // Initialize, and the registration inside Resolve must still be cleaned up.
  if (!WorkflowIntegration || !wiInitialized) return;
  try {
    WorkflowIntegration.CleanUp();
  } catch (err) {
    log(`WorkflowIntegration.CleanUp failed: ${err.message}`);
  }
  resolveObj = null;
}

// --- Resolve helpers. `await` works on both the sync and the promise flavour of the API. ---

function requireResolve() {
  if (host !== 'resolve' || !resolveObj) {
    throw new Error('DaVinci Resolve is not available in this host.');
  }
  return resolveObj;
}

async function getProject() {
  const resolve = requireResolve();
  const pm = await resolve.GetProjectManager();
  if (!pm) return null;
  return (await pm.GetCurrentProject()) || null;
}

async function getMediaPool() {
  const project = await getProject();
  if (!project) return null;
  return (await project.GetMediaPool()) || null;
}

function parseTimecodeSeconds(tc, fps) {
  if (typeof tc !== 'string') return null;
  const m = tc.trim().match(/^(\d+)[:;](\d+)[:;](\d+)[:;.](\d+)$/);
  if (!m) return null;
  const rate = Number(fps);
  if (!Number.isFinite(rate) || rate <= 0) return null;
  const [h, mi, s, f] = m.slice(1).map(Number);
  return h * 3600 + mi * 60 + s + f / rate;
}

async function clipInfo(item) {
  const mediaId = String((await item.GetMediaId()) ?? '');
  const name = String((await item.GetName()) ?? '');
  const filePath = String((await item.GetClipProperty('File Path')) ?? '');
  const info = { mediaId, name, filePath };
  try {
    const tc = await item.GetClipProperty('Duration');
    const fps = await item.GetClipProperty('FPS');
    const seconds = parseTimecodeSeconds(tc, fps);
    if (seconds != null) info.durationS = seconds;
  } catch {
    /* duration is optional */
  }
  if (mediaId) knownItems.set(mediaId, item);
  return info;
}

async function itemHasId(item, mediaId) {
  try {
    return String(await item.GetMediaId()) === mediaId;
  } catch {
    return false;
  }
}

async function findItemByMediaId(mediaId) {
  if (typeof mediaId !== 'string' || !mediaId) throw new Error('mediaId must be a non-empty string.');
  const cached = knownItems.get(mediaId);
  if (cached && (await itemHasId(cached, mediaId))) return cached;
  knownItems.delete(mediaId);

  const mediaPool = await getMediaPool();
  if (!mediaPool) return null;

  // Cheap path: the selection.
  const selected = (await mediaPool.GetSelectedClips()) || [];
  for (const item of selected) {
    if (await itemHasId(item, mediaId)) {
      knownItems.set(mediaId, item);
      return item;
    }
  }

  // Full walk of the media pool tree.
  const root = await mediaPool.GetRootFolder();
  const stack = root ? [root] : [];
  while (stack.length) {
    const folder = stack.pop();
    const clips = (await folder.GetClipList()) || [];
    for (const item of clips) {
      if (await itemHasId(item, mediaId)) {
        knownItems.set(mediaId, item);
        return item;
      }
    }
    const subs = (await folder.GetSubFolderList()) || [];
    for (const sub of subs) stack.push(sub);
  }
  return null;
}

async function resolveGetSelectedClip() {
  const mediaPool = await getMediaPool();
  if (!mediaPool) return null;
  const selected = (await mediaPool.GetSelectedClips()) || [];
  if (selected.length > 0 && selected[0]) return clipInfo(selected[0]);
  const project = await getProject();
  const timeline = project ? await project.GetCurrentTimeline() : null;
  if (!timeline) return null;
  const videoItem = await timeline.GetCurrentVideoItem();
  if (!videoItem) return null;
  const item = await videoItem.GetMediaPoolItem();
  return item ? clipInfo(item) : null;
}

async function resolveImportMedia(filePath) {
  if (typeof filePath !== 'string' || !path.isAbsolute(filePath)) throw new Error('importMedia: path must be an absolute path.');
  const mediaPool = await getMediaPool();
  if (!mediaPool) throw new Error('No project is open in DaVinci Resolve.');
  const items = (await mediaPool.ImportMedia([filePath])) || [];
  if (!items.length || !items[0]) return null;
  return clipInfo(items[0]);
}

async function resolveReplaceClip(mediaId, filePath) {
  if (typeof filePath !== 'string' || !path.isAbsolute(filePath)) throw new Error('replaceClip: path must be an absolute path.');
  const item = await findItemByMediaId(mediaId);
  if (!item) throw new Error(`Clip ${mediaId} was not found in the media pool.`);
  if (typeof item.ReplaceClipPreserveSubClip === 'function') {
    try {
      if (await item.ReplaceClipPreserveSubClip(filePath)) return true;
    } catch (err) {
      log(`ReplaceClipPreserveSubClip failed (${err.message}); falling back to ReplaceClip`);
    }
  }
  return Boolean(await item.ReplaceClip(filePath));
}

async function resolveAppendToTimeline(mediaId) {
  const item = await findItemByMediaId(mediaId);
  if (!item) throw new Error(`Clip ${mediaId} was not found in the media pool.`);
  const mediaPool = await getMediaPool();
  if (!mediaPool) throw new Error('No project is open in DaVinci Resolve.');
  const result = await mediaPool.AppendToTimeline([item]);
  return Array.isArray(result) ? result.length > 0 : Boolean(result);
}

async function resolveGetContext() {
  const resolve = requireResolve();
  const ctx = { project: null, timeline: null, page: null };
  try {
    const project = await getProject();
    if (project) {
      ctx.project = (await project.GetName()) ?? null;
      const timeline = await project.GetCurrentTimeline();
      if (timeline) ctx.timeline = (await timeline.GetName()) ?? null;
    }
  } catch (err) {
    log(`getContext(project): ${err.message}`);
  }
  try {
    ctx.page = (await resolve.GetCurrentPage()) ?? null;
  } catch (err) {
    log(`getContext(page): ${err.message}`);
  }
  return ctx;
}

// ---------------------------------------------------------------------------------------------
// Window + pages
// ---------------------------------------------------------------------------------------------

let mainWindow = null;
let errorShown = false;

const AUDIO_EXTENSIONS = ['wav', 'aif', 'aiff', 'flac', 'mp3', 'm4a', 'aac', 'ogg', 'opus', 'caf', 'wma', 'mp4', 'mov', 'mkv', 'webm', 'm4v'];

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

function errorPageHtml(title, detail) {
  const cmd = engine.config ? engine.config.command.join(' ') : '(engine.json missing or invalid)';
  const cwd = engine.config ? engine.config.cwd : '';
  const tail = engine.stderrTail.length ? engine.stderrTail.join('\n') : '(no output on stderr)';
  return `<!doctype html><html><head><meta charset="utf-8"><title>${escapeHtml(APP_TITLE)} — error</title>
<style>
  html,body{margin:0;background:#0e1013;color:#e6e8eb;font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
  main{max-width:900px;margin:0 auto;padding:40px 32px}
  h1{font-size:20px;font-weight:600;margin:0 0 8px;color:#ff7a7a}
  p{margin:0 0 12px;color:#b7bcc4}
  code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;color:#dfe3e8}
  pre{background:#15181d;border:1px solid #262b33;border-radius:8px;padding:14px;overflow:auto;max-height:46vh;white-space:pre-wrap;word-break:break-word;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;color:#c9ced6}
  dl{display:grid;grid-template-columns:max-content 1fr;gap:4px 16px;margin:16px 0}
  dt{color:#7d848f}
  h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:#7d848f;margin:24px 0 8px}
  ul{margin:0 0 0 18px;color:#b7bcc4}
</style></head><body><main>
<h1>${escapeHtml(title)}</h1>
<p>${escapeHtml(detail)}</p>
<dl>
  <dt>Command</dt><dd><code>${escapeHtml(cmd)}</code></dd>
  <dt>Working dir</dt><dd><code>${escapeHtml(cwd)}</code></dd>
  <dt>Plugin dir</dt><dd><code>${escapeHtml(__dirname)}</code></dd>
</dl>
<h2>Engine output (stderr tail)</h2>
<pre>${escapeHtml(tail)}</pre>
<h2>What to check</h2>
<ul>
  <li>The engine path in <code>engine.json</code> exists and is executable, and its virtualenv has the <code>ui</code> extra installed (<code>uv sync --extra ui</code> / <code>pip install -e '.[ui]'</code>).</li>
  <li><code>ffmpeg</code> is on the PATH (Homebrew: <code>/opt/homebrew/bin</code>).</li>
  <li>Close this window and relaunch the plugin from Resolve's <b>Workspace → Workflow Integrations → HawaVoClean</b> (or re-run <code>npm start</code>).</li>
</ul>
</main></body></html>`;
}

function showErrorPage(title, detail) {
  if (errorShown) return;
  errorShown = true;
  const html = errorPageHtml(title, detail);
  const present = () => {
    if (quitState !== 'running') return;
    if (!mainWindow || mainWindow.isDestroyed()) createWindow({ loadUi: false });
    mainWindow.loadURL('data:text/html;charset=utf-8,' + encodeURIComponent(html)).catch(() => {});
    if (SELFTEST) {
      console.log('HAWA_SELFTEST_ERROR ' + JSON.stringify({ title, detail, stderrTail: engine.stderrTail.slice(-5) }));
      setTimeout(() => app.quit(), 200);
    }
  };
  if (app.isReady()) present();
  else app.whenReady().then(present);
}

function createWindow({ loadUi = true } = {}) {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 820,
    minWidth: 960,
    minHeight: 640,
    title: APP_TITLE,
    backgroundColor: '#0e1013',
    show: !SELFTEST,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      sandbox: true,
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  mainWindow.on('closed', () => {
    mainWindow = null;
  });
  // Keep the UI inside the window: external links go to the system browser.
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:/i.test(url)) shell.openExternal(url).catch(() => {});
    return { action: 'deny' };
  });
  // Deny-by-default navigation: the only renderer-initiated navigation allowed is a reload of
  // the page that is already loaded. http(s) links open in the system browser; anything else
  // (foreign file: URLs — e.g. a file dropped on a spot the UI does not handle — data:, custom
  // schemes) is blocked so the privileged preload never attaches to unexpected content.
  const contents = mainWindow.webContents;
  contents.on('will-navigate', (event, url) => {
    if (url === contents.getURL()) return; // allow reload
    event.preventDefault();
    if (/^https?:/i.test(url)) shell.openExternal(url).catch(() => {});
    else log(`blocked navigation to ${url}`);
  });
  if (DEVTOOLS) mainWindow.webContents.openDevTools({ mode: 'detach' });

  if (loadUi) {
    const indexHtml = path.join(__dirname, 'index.html');
    if (fs.existsSync(indexHtml)) {
      mainWindow.loadFile(indexHtml).catch((err) => log(`loadFile failed: ${err.message}`));
    } else {
      const html = `<!doctype html><html><head><meta charset="utf-8"><title>${APP_TITLE}</title></head>
<body style="margin:0;background:#0e1013;color:#e6e8eb;font:14px/1.5 -apple-system,Helvetica,Arial,sans-serif">
<main style="max-width:760px;margin:0 auto;padding:48px 32px">
<h1 style="font-size:20px;font-weight:600;margin:0 0 8px">UI bundle not found</h1>
<p style="color:#b7bcc4">There is no <code>index.html</code> next to <code>main.js</code>. Build the UI and assemble the plugin with
<code>resolve-plugin/install.sh</code> (or copy <code>ui/dist/*</code> into <code>${escapeHtml(__dirname)}</code>).</p>
<p style="color:#7d848f">Host: ${escapeHtml(host)}</p></main></body></html>`;
      mainWindow.loadURL('data:text/html;charset=utf-8,' + encodeURIComponent(html)).catch(() => {});
    }
  }
  return mainWindow;
}

// ---------------------------------------------------------------------------------------------
// IPC (channel names are fixed by the contract, section 3)
// ---------------------------------------------------------------------------------------------

function registerIpc() {
  // The one synchronous call: preload needs `host` as a plain value at script-evaluation time.
  ipcMain.on('hawa:host', (event) => {
    event.returnValue = host;
  });

  ipcMain.handle('hawa:engine:endpoint', async () => {
    if (engine.failure) throw engine.failure;
    return engine.ready; // rejects with the failure Error if the engine never comes up
  });

  ipcMain.handle('hawa:files:pick', async (event) => {
    const owner = BrowserWindow.fromWebContents(event.sender) || mainWindow || undefined;
    const result = await dialog.showOpenDialog(owner, {
      title: 'Choose an audio or video file',
      buttonLabel: 'Open',
      properties: ['openFile', 'treatPackageAsDirectory'],
      filters: [
        { name: 'Audio and video', extensions: AUDIO_EXTENSIONS },
        { name: 'All files', extensions: ['*'] },
      ],
    });
    if (result.canceled || !result.filePaths || result.filePaths.length === 0) return null;
    return result.filePaths[0];
  });

  ipcMain.handle('hawa:files:reveal', async (_event, filePath) => {
    if (typeof filePath !== 'string' || !path.isAbsolute(filePath)) throw new Error('revealInFinder: path must be an absolute path.');
    shell.showItemInFolder(filePath);
  });

  ipcMain.handle('hawa:resolve:selected', async () => resolveGetSelectedClip());
  ipcMain.handle('hawa:resolve:import', async (_event, filePath) => resolveImportMedia(filePath));
  ipcMain.handle('hawa:resolve:replace', async (_event, mediaId, filePath) => resolveReplaceClip(mediaId, filePath));
  ipcMain.handle('hawa:resolve:append', async (_event, mediaId) => resolveAppendToTimeline(mediaId));
  ipcMain.handle('hawa:resolve:context', async () => resolveGetContext());
}

// ---------------------------------------------------------------------------------------------
// Self-test hook (HAWA_SELFTEST=1): exercise the bridge from the renderer, print, quit.
// ---------------------------------------------------------------------------------------------

async function runSelfTest() {
  const js = `(async () => {
    const out = { hasBridge: typeof window.hawa === 'object' && window.hawa !== null };
    if (!out.hasBridge) return out;
    out.host = window.hawa.host;
    out.keys = Object.keys(window.hawa).sort();
    out.hasResolve = 'resolve' in window.hawa;
    out.pathForFileNull = window.hawa.files.pathForFile(new File(['x'], 'x.wav')) === null;
    const ep = await window.hawa.engine.getEndpoint();
    out.endpoint = ep;
    const r = await fetch(ep.baseUrl + '/api/health', { headers: { 'X-Hawa-Token': ep.token } });
    out.health = { status: r.status, body: await r.json() };
    const r2 = await fetch(ep.baseUrl + '/api/health');
    out.unauthStatus = r2.status;
    // Test pages may publish extra probe results here (e.g. module script / worker checks).
    if (window.__hawaSelftestPage !== undefined) {
      out.page = await Promise.resolve(window.__hawaSelftestPage);
    }
    return out;
  })()`;
  try {
    const result = await mainWindow.webContents.executeJavaScript(js, true);
    result.enginePid = engine.child ? engine.child.pid : null;
    console.log('HAWA_SELFTEST_RESULT ' + JSON.stringify(result));
  } catch (err) {
    console.log('HAWA_SELFTEST_ERROR ' + JSON.stringify({ title: 'selftest threw', detail: String(err && err.message ? err.message : err) }));
  }
  app.quit();
}

// ---------------------------------------------------------------------------------------------
// App lifecycle
// ---------------------------------------------------------------------------------------------

let quitState = 'running'; // 'running' | 'pending' | 'done'

async function shutdownAll() {
  try {
    await stopEngine();
  } catch (err) {
    log(`stopEngine: ${err.message}`);
  }
  cleanupResolve();
}

app.on('before-quit', (event) => {
  if (quitState === 'done') return;
  event.preventDefault();
  if (quitState === 'pending') return;
  quitState = 'pending';
  const hardDeadline = setTimeout(() => {
    log('shutdown deadline hit; exiting');
    app.exit(0);
  }, QUIT_HARD_DEADLINE_MS);
  hardDeadline.unref();
  shutdownAll().finally(() => {
    clearTimeout(hardDeadline);
    quitState = 'done';
    app.quit();
  });
});

app.on('window-all-closed', () => {
  // A plugin window that is closed should end the plugin process on every platform.
  app.quit();
});

for (const sig of ['SIGINT', 'SIGTERM', 'SIGHUP']) {
  process.on(sig, () => {
    log(`${sig} received; quitting`);
    app.quit();
  });
}

process.on('uncaughtException', (err) => {
  log(`uncaught exception: ${err && err.stack ? err.stack : err}`);
});

app.setName(APP_TITLE);

app.whenReady().then(async () => {
  registerIpc();
  startEngine();
  // Decide the host before the first page loads: preload reads it synchronously.
  host = (await initResolve()) ? 'resolve' : 'electron';
  log(`host = ${host}`);
  if (errorShown) return; // engine failed fast; the error page is already up.
  createWindow();
  if (SELFTEST) {
    console.log('HAWA_SELFTEST_HOST ' + host);
    mainWindow.webContents.once('did-finish-load', () => {
      runSelfTest();
    });
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0 && quitState === 'running') createWindow();
});
