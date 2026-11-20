'use strict';
/*
 * HawaVoClean — DaVinci Resolve Workflow Integration / desktop shell (main process).
 *
 * Contract: docs/ui-contract.md sections 3, 4 and 5.
 *
 *  - Spawns the Python engine (`hawavoclean serve`) described by engine.json (next to this
 *    file), waits for its single `{"event":"ready",...}` stdout line, mints a short-lived
 *    session in main, and hands the renderer only the credential-free endpoint over IPC.
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
 *   HAWA_SELFTEST_HARD_CRASH=1 leave the self-test host alive so the harness can SIGKILL it
 *   HAWA_ENGINE_TIMEOUT_MS=N   override the 60 s ready timeout (tests only)
 */

const isElectronRuntime = Boolean(process.versions && process.versions.electron);
const { app, BrowserWindow, ipcMain, dialog, shell, session, protocol } = isElectronRuntime
  ? require('electron')
  : {};
const path = require('node:path');
const fs = require('node:fs');
const os = require('node:os');
const http = require('node:http');
const crypto = require('node:crypto');
const { spawn } = require('node:child_process');
const { fileURLToPath, pathToFileURL } = require('node:url');
const {
  exactEngineOrigin,
  isEngineApiRequest,
  rendererEngineEndpoint,
  requestEngineSession,
  requestNativeSourceRegistration,
  sessionNeedsRenewal,
  withEngineAuthorization,
  withoutRendererCredentials,
} = require('./session-auth.js');
const {
  discoverDesktopEngine,
  findDesktopEngineExecutable,
  verifyEngineExecutable,
} = require('./engine-discovery.js');
const {
  discoverWorkflowIntegrationSdk,
  acquireWorkflowIntegrationNode,
} = require('./sdk-preflight.js');
const {
  TimelineTransactionManager,
  validateBatchCleanTrackPayload,
  validateApplyGuardMarkersPayload,
} = require('./timeline-transactions.js');

const PLUGIN_ID = 'com.hawavoclean.resolve';
const APP_TITLE = 'HawaVoClean';
const APP_SCHEME = 'hawa';
const APP_ENTRY_URL = `${APP_SCHEME}://app/index.html`;
// No `persist:` prefix: Electron guarantees an in-memory session. This keeps cookies, cache and
// service workers out of Resolve's default session and discards the plugin session at process exit.
const SESSION_PARTITION = 'hawavoclean-isolated';
const ENGINE_READY_TIMEOUT_MS = positiveInt(process.env.HAWA_ENGINE_TIMEOUT_MS, 60_000);
const RESOLVE_INIT_TIMEOUT_MS = 8_000;
const SHUTDOWN_POST_TIMEOUT_MS = 1_000;
const SHUTDOWN_SIGTERM_GRACE_MS = 3_000;
const QUIT_HARD_DEADLINE_MS = 10_000;
const STDERR_TAIL_LINES = 80;
const DEVTOOLS = process.env.HAWA_DEVTOOLS === '1';
const SELFTEST = process.env.HAWA_SELFTEST === '1';
const SELFTEST_HARD_CRASH = SELFTEST && process.env.HAWA_SELFTEST_HARD_CRASH === '1';
const RUNTIME_EVIDENCE = Object.freeze({
  electron: process.versions.electron || null,
  chromium: process.versions.chrome || null,
  node: process.versions.node || null,
  v8: process.versions.v8 || null,
});
const securityEvents = {
  blockedPermissions: 0,
  blockedNavigations: 0,
  blockedWindows: 0,
  blockedRequests: 0,
  blockedWebviews: 0,
};

// Must run before `ready`. The app scheme is standard so relative Vite assets resolve, secure so
// browser APIs behave normally, and explicitly CORS-enabled to avoid the custom-protocol class of
// cross-origin read flaws. Service workers and CSP bypass remain disabled.
if (isElectronRuntime && protocol && typeof protocol.registerSchemesAsPrivileged === 'function') {
  protocol.registerSchemesAsPrivileged([
    {
      scheme: APP_SCHEME,
      privileges: {
        standard: true,
        secure: true,
        supportFetchAPI: true,
        corsEnabled: true,
        stream: true,
        codeCache: true,
        allowServiceWorkers: false,
        bypassCSP: false,
      },
    },
  ]);
}

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

function pathInside(root, candidate) {
  const relative = path.relative(path.resolve(root), path.resolve(candidate));
  return relative === '' || (!relative.startsWith('..' + path.sep) && relative !== '..' && !path.isAbsolute(relative));
}

function isTrustedAppUrl(value) {
  if (typeof value !== 'string' || !value) return false;
  try {
    const url = new URL(value);
    return url.protocol === `${APP_SCHEME}:` && url.hostname === 'app' && url.pathname === '/index.html';
  } catch {
    return false;
  }
}

function ipcSenderUrl(event) {
  if (event.senderFrame && typeof event.senderFrame.url === 'string') return event.senderFrame.url;
  return event.sender.getURL();
}

function isTrustedIpcSender(event) {
  if (!mainWindow || mainWindow.isDestroyed() || event.sender !== mainWindow.webContents) return false;
  if (event.senderFrame && event.sender.mainFrame && event.senderFrame !== event.sender.mainFrame) return false;
  return isTrustedAppUrl(ipcSenderUrl(event));
}

function requireTrustedIpcSender(event) {
  if (!isTrustedIpcSender(event)) throw new Error('IPC sender is not the trusted HawaVoClean renderer.');
}

function trustedClipboardRequest(contents, permission, requestingUrl) {
  return (
    permission === 'clipboard-sanitized-write' &&
    mainWindow &&
    !mainWindow.isDestroyed() &&
    contents === mainWindow.webContents &&
    isTrustedAppUrl(requestingUrl || contents.getURL())
  );
}

function configureSessionSecurity() {
  const appSession = session.fromPartition(SESSION_PARTITION);
  const requestAuthorization = new Map();
  appSession.protocol.handle(APP_SCHEME, (request) => {
    try {
      const url = new URL(request.url);
      if (request.method !== 'GET' || url.hostname !== 'app') {
        return new Response('not found', { status: 404, headers: { 'content-type': 'text/plain' } });
      }
      const decodedPath = decodeURIComponent(url.pathname);
      if (decodedPath.includes('\0')) throw new Error('NUL in app path');
      const candidate = path.resolve(__dirname, '.' + decodedPath);
      const realRoot = fs.realpathSync(__dirname);
      const realCandidate = fs.realpathSync(candidate);
      if (!pathInside(realRoot, realCandidate) || !fs.statSync(realCandidate).isFile()) {
        throw new Error('app path escaped plugin root');
      }
      return appSession.fetch(pathToFileURL(realCandidate).href);
    } catch {
      return new Response('not found', { status: 404, headers: { 'content-type': 'text/plain' } });
    }
  });
  appSession.setPermissionRequestHandler((contents, permission, callback, details) => {
    const allowed = trustedClipboardRequest(contents, permission, details && details.requestingUrl);
    if (!allowed) {
      securityEvents.blockedPermissions += 1;
      log(`blocked permission ${permission}`);
    }
    callback(Boolean(allowed));
  });
  appSession.setPermissionCheckHandler((contents, permission, requestingOrigin, details) => {
    const url = (details && details.requestingUrl) || requestingOrigin || (contents && contents.getURL()) || '';
    const allowed = trustedClipboardRequest(contents, permission, url);
    if (!allowed) securityEvents.blockedPermissions += 1;
    return Boolean(allowed);
  });
  if (typeof appSession.setDevicePermissionHandler === 'function') {
    appSession.setDevicePermissionHandler(() => false);
  }
  appSession.webRequest.onBeforeRequest(
    { urls: [`${APP_SCHEME}://*/*`, 'file://*/*', 'http://*/*', 'https://*/*'] },
    (details, callback) => {
      let allowed = false;
      try {
        const url = new URL(details.url);
        if (url.protocol === `${APP_SCHEME}:`) {
          allowed = url.hostname === 'app';
        } else if (url.protocol === 'file:') {
          allowed = pathInside(__dirname, fileURLToPath(url));
        } else if (url.protocol === 'http:') {
          allowed = isEngineApiRequest(details.url, engineOrigin()) && url.pathname !== '/api/session';
        }
      } catch {
        allowed = false;
      }
      if (!allowed) {
        securityEvents.blockedRequests += 1;
        log(`blocked renderer request to ${details.url}`);
      }
      callback({ cancel: !allowed });
    },
  );
  appSession.webRequest.onBeforeSendHeaders(
    { urls: ['http://127.0.0.1:*/*'] },
    (details, callback) => {
      if (!isEngineApiRequest(details.url, engineOrigin())) {
        callback({ requestHeaders: details.requestHeaders });
        return;
      }
      if (details.method === 'OPTIONS') {
        callback({ requestHeaders: withoutRendererCredentials(details.requestHeaders) });
        return;
      }
      engineAuthorization().then(
        (authorization) => {
          try {
            requestAuthorization.set(details.id, authorization);
            callback({ requestHeaders: withEngineAuthorization(details.requestHeaders, authorization) });
          } catch {
            callback({ cancel: true });
          }
        },
        () => callback({ cancel: true }),
      );
    },
  );
  appSession.webRequest.onCompleted({ urls: ['http://127.0.0.1:*/*'] }, (details) => {
    const authorization = requestAuthorization.get(details.id);
    requestAuthorization.delete(details.id);
    if (
      authorization !== undefined &&
      details.statusCode === 401 &&
      isEngineApiRequest(details.url, engineOrigin())
    ) {
      invalidateEngineSession(authorization);
    }
  });
  appSession.webRequest.onErrorOccurred({ urls: ['http://127.0.0.1:*/*'] }, (details) => {
    requestAuthorization.delete(details.id);
  });

  const STRICT_CSP = "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; media-src 'self' blob: http://127.0.0.1:*; connect-src http://127.0.0.1:* ws://127.0.0.1:*; font-src 'self'; worker-src 'self' blob:; object-src 'none'; frame-src 'none'; base-uri 'none'; form-action 'none';";
  appSession.webRequest.onHeadersReceived((details, callback) => {
    const responseHeaders = { ...(details.responseHeaders || {}) };
    responseHeaders['Content-Security-Policy'] = [STRICT_CSP];
    responseHeaders['X-Frame-Options'] = ['DENY'];
    responseHeaders['X-Content-Type-Options'] = ['nosniff'];
    callback({ responseHeaders });
  });
}

// ---------------------------------------------------------------------------------------------
// Engine process management
// ---------------------------------------------------------------------------------------------

const ALLOWED_RESOLVE_ENGINE_TRANSITIONS = Object.freeze({
  idle: Object.freeze(['discovering', 'stopped']),
  discovering: Object.freeze(['bootstrapping', 'failed', 'stopped']),
  bootstrapping: Object.freeze(['ready', 'discovering', 'failed', 'stopped']),
  ready: Object.freeze(['failed', 'stopped']),
  failed: Object.freeze(['stopped']),
  stopped: Object.freeze([]),
});

const engine = {
  phase: 'idle', // 'idle' | 'discovering' | 'bootstrapping' | 'ready' | 'failed' | 'stopped'
  config: null, // {command: string[], cwd: string, env: object}
  child: null,
  token: crypto.randomBytes(16).toString('hex'), // fresh random 32-hex token per run
  port: null,
  pid: null,
  version: null,
  stderrTail: [], // last STDERR_TAIL_LINES lines of stderr (plus unexpected stdout)
  ready: null, // Promise<{baseUrl}>; renderer-facing and deliberately credential-free
  endpoint: null, // immutable {baseUrl}
  session: null, // main-only {authorization, expiresAtMs, refreshAtMs}
  sessionPromise: null,
  failure: null, // Error once the engine is known to be unusable
  exited: false,
  exitInfo: null, // {code, signal}
  settled: false,
  bootstrapping: false,
  shared: false,
};

function transitionEnginePhase(next) {
  if (engine.phase === next) return;
  const allowed = ALLOWED_RESOLVE_ENGINE_TRANSITIONS[engine.phase] || [];
  if (!allowed.includes(next)) {
    throw new Error(`Illegal Resolve engine transition: ${engine.phase} -> ${next}`);
  }
  engine.phase = next;
}

let resolveReady = null; // resolve() of engine.ready
let rejectReady = null; // reject() of engine.ready

function engineOrigin() {
  if (engine.phase !== 'ready' && engine.phase !== 'bootstrapping') return null;
  return engine.endpoint ? exactEngineOrigin(engine.endpoint.baseUrl) : null;
}

async function engineAuthorization() {
  if (engine.failure) throw engine.failure;
  const endpoint = engine.endpoint || (await engine.ready);
  if (!sessionNeedsRenewal(engine.session, Date.now()) && engine.session !== null) {
    return engine.session.authorization;
  }
  if (engine.sessionPromise === null) {
    engine.sessionPromise = requestEngineSession(endpoint, engine.token);
  }
  const pending = engine.sessionPromise;
  try {
    const capability = await pending;
    engine.session = capability;
    return capability.authorization;
  } finally {
    if (engine.sessionPromise === pending) engine.sessionPromise = null;
  }
}

async function registerNativePath(filePath) {
  if (typeof filePath !== 'string' || !path.isAbsolute(filePath)) {
    throw new Error('Selected source path must be absolute.');
  }
  const extension = path.extname(filePath).slice(1).toLowerCase();
  if (!AUDIO_EXTENSIONS.includes(extension)) {
    throw new Error('Selected source must use a supported audio or video extension.');
  }
  if (engine.failure) throw engine.failure;
  const endpoint = engine.endpoint || (await engine.ready);
  const registration = await requestNativeSourceRegistration(endpoint, engine.token, filePath);
  return registration.path;
}

function invalidateEngineSession(authorization) {
  if (authorization !== undefined && engine.session && engine.session.authorization !== authorization) return;
  engine.session = null;
}

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
      'The HawaVoClean audio engine could not be located. Launch HawaVoClean Desktop from /Applications/HawaVoClean.app or reinstall from the HawaVoClean installer package.',
    );
  }
  let raw;
  try {
    raw = JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch (err) {
    throw new Error(`engine.json is not valid JSON: ${err.message}`);
  }
  if (!Array.isArray(raw.command) || raw.command.length === 0 || !raw.command.every((s) => typeof s === 'string' && s.length > 0)) {
    throw new Error('engine.json: "command" must be a non-empty array of strings, e.g. ["./engine/hawavoclean-engine","serve"].');
  }
  const cwdValue = typeof raw.cwd === 'string' && raw.cwd.length > 0 ? raw.cwd : os.homedir();
  const cwd = path.isAbsolute(cwdValue) ? cwdValue : path.resolve(__dirname, cwdValue);
  if (!path.isAbsolute(cwdValue) && (path.relative(__dirname, cwd).startsWith('..' + path.sep) || path.relative(__dirname, cwd) === '..')) {
    throw new Error('engine.json: a relative cwd may not escape the plugin directory.');
  }
  const command = raw.command.slice();
  if (!path.isAbsolute(command[0])) {
    if (!command[0].startsWith('./')) {
      throw new Error('engine.json: a relative executable must start with ./ and stay inside the plugin.');
    }
    command[0] = path.resolve(__dirname, command[0]);
    if (path.relative(__dirname, command[0]).startsWith('..' + path.sep) || path.relative(__dirname, command[0]) === '..') {
      throw new Error('engine.json: a relative executable may not escape the plugin directory.');
    }
  }
  const env = raw.env && typeof raw.env === 'object' && !Array.isArray(raw.env) ? raw.env : {};
  for (const [k, v] of Object.entries(env)) {
    if (typeof v !== 'string') throw new Error(`engine.json: env.${k} must be a string.`);
  }
  return { command, cwd, env, file };
}

function buildEngineEnv(config) {
  const env = { ...process.env, ...config.env };
  // A detached broker would otherwise survive a hard Resolve/Electron crash,
  // because that path runs none of this process's shutdown handlers.
  env.HAWAVOCLEAN_PARENT_PID = String(process.pid);
  const commandDir = path.isAbsolute(config.command[0]) ? path.dirname(config.command[0]) : null;
  const wanted = [commandDir, ...EXTRA_PATH_DIRS].filter(Boolean);
  const current = (env.PATH || '').split(path.delimiter).filter(Boolean);
  const prepend = wanted.filter((d) => !current.includes(d));
  env.PATH = [...prepend, ...current].join(path.delimiter);
  if (!('PYTHONUNBUFFERED' in env)) env.PYTHONUNBUFFERED = '1';
  return env;
}

function signalEngineTree(child, signal) {
  if (!child || !Number.isInteger(child.pid) || child.pid <= 0) return false;
  try {
    // Resolve support is macOS-only. A detached POSIX child leads a process
    // group, so the negative pid reaches the broker and every decoder/model
    // descendant even when the leader has already stopped responding.
    if (process.platform !== 'win32') process.kill(-child.pid, signal);
    else child.kill(signal);
    return true;
  } catch {
    return false;
  }
}

function failEngine(message) {
  if (engine.settled || engine.phase === 'stopped') return;
  engine.settled = true;
  engine.bootstrapping = false;
  engine.endpoint = null;
  engine.session = null;
  transitionEnginePhase('failed');
  const err = new Error(message);
  engine.failure = err;
  if (rejectReady) rejectReady(err);
  log('engine failure:', message);
  // Make sure a half-started child does not linger.
  if (engine.child) {
    signalEngineTree(engine.child, 'SIGTERM');
    const c = engine.child;
    setTimeout(() => {
      // The leader's exit is not proof that a decoder/model descendant is
      // gone. Signalling the retained process-group id is safe after exit.
      signalEngineTree(c, 'SIGKILL');
    }, SHUTDOWN_SIGTERM_GRACE_MS).unref();
  }
  showErrorPage('The HawaVoClean engine could not be started', message);
}

function handleReadyLine(obj) {
  if (engine.settled || engine.bootstrapping || engine.phase !== 'discovering') return;
  const port = Number(obj.port);
  if (!Number.isInteger(port) || port <= 0 || port > 65535) {
    failEngine(`Engine reported an invalid port in its ready line: ${JSON.stringify(obj)}`);
    return;
  }
  transitionEnginePhase('bootstrapping');
  engine.bootstrapping = true;
  engine.port = port;
  engine.pid = Number.isInteger(obj.pid) ? obj.pid : engine.child ? engine.child.pid : null;
  engine.version = typeof obj.version === 'string' ? obj.version : null;
  engine.endpoint = rendererEngineEndpoint(port);
  log(`engine ready: ${engine.endpoint.baseUrl} (pid ${engine.pid}, version ${engine.version || '?'}); minting renderer session`);
  requestEngineSession(engine.endpoint, engine.token).then(
    (capability) => {
      if (engine.settled || engine.phase !== 'bootstrapping') return;
      engine.session = capability;
      engine.bootstrapping = false;
      engine.settled = true;
      transitionEnginePhase('ready');
      log('engine renderer session ready');
      if (resolveReady) resolveReady(engine.endpoint);
    },
    (error) => {
      engine.bootstrapping = false;
      failEngine(`Could not establish an engine renderer session: ${error.message}`);
    },
  );
}

function spawnLocalEngine(discovery) {
  let config;
  if (discovery && discovery.type === 'installed_desktop_engine') {
    config = {
      command: discovery.command,
      cwd: discovery.cwd,
      env: discovery.env,
      file: discovery.executable,
    };
  } else {
    try {
      config = readEngineConfig();
    } catch (err) {
      failEngine((discovery && discovery.message) || err.message);
      return;
    }
  }
  engine.config = config;

  const [exe, ...rest] = config.command;
  const args = [...rest, '--port', '0', '--token-stdin', '--ui-dir', __dirname];
  log(`spawning engine: ${exe} ${args.join(' ')} (cwd ${config.cwd})`);

  let child;
  try {
    child = spawn(exe, args, {
      cwd: config.cwd,
      env: buildEngineEnv(config),
      stdio: ['pipe', 'pipe', 'pipe'],
      windowsHide: true,
      detached: process.platform !== 'win32',
    });
  } catch (err) {
    failEngine(`Could not spawn engine (${exe}): ${err.message}`);
    return;
  }
  engine.child = child;
  child.stdin.on('error', () => {});
  child.stdin.end(`${engine.token}\n`);

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
      engine.endpoint = null;
      engine.session = null;
      transitionEnginePhase('failed');
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

/**
 * Resolves the installed HawaVoClean Desktop engine executable (`/Applications/HawaVoClean.app`)
 * after a stale rendezvous failure, before falling back to plugin-local `engine.json`.
 */
function resolveDesktopEngineFallback(options = {}) {
  const findExe = options.findDesktopEngineExecutable || findDesktopEngineExecutable;
  const verifyExe = options.verifyEngineExecutable || verifyEngineExecutable;
  const executable = options.engineExecutable || findExe();
  if (!executable || !fs.existsSync(executable)) {
    return null;
  }
  verifyExe(executable);
  return {
    type: 'installed_desktop_engine',
    executable,
    command: [executable, 'serve'],
    cwd: os.homedir(),
    env: {
      PYTHONNOUSERSITE: '1',
      PYTHONDONTWRITEBYTECODE: '1',
    },
  };
}

function startEngine() {
  if (engine.phase !== 'idle') return;
  transitionEnginePhase('discovering');
  engine.ready = new Promise((resolve, reject) => {
    resolveReady = resolve;
    rejectReady = reject;
  });
  // Swallow "unhandled rejection" noise; consumers (IPC handler) attach their own handlers.
  engine.ready.catch(() => {});

  let discovery;
  try {
    discovery = discoverDesktopEngine();
  } catch (err) {
    failEngine(`Desktop engine discovery failed: ${err.message}`);
    return;
  }

  if (discovery.type === 'active_rendezvous') {
    log(`connecting to active desktop engine rendezvous: ${discovery.origin} (pid ${discovery.pid})`);
    const freshToken = engine.token;
    engine.endpoint = { baseUrl: discovery.origin };
    engine.token = discovery.token;
    engine.pid = discovery.pid;
    engine.bootstrapping = true;
    engine.shared = true;
    transitionEnginePhase('bootstrapping');

    requestEngineSession(engine.endpoint, engine.token).then(
      (capability) => {
        if (engine.settled || engine.phase !== 'bootstrapping') return;
        engine.session = capability;
        engine.bootstrapping = false;
        engine.settled = true;
        transitionEnginePhase('ready');
        log('engine renderer session ready via desktop rendezvous');
        if (resolveReady) resolveReady(engine.endpoint);
      },
      (err) => {
        if (engine.settled || engine.phase !== 'bootstrapping') return;
        log(`stale desktop rendezvous (${err.message}); falling back to Desktop app engine before local engine.json`);
        engine.endpoint = null;
        engine.session = null;
        engine.pid = null;
        engine.token = freshToken;
        engine.bootstrapping = false;
        engine.shared = false;
        transitionEnginePhase('discovering');
        let desktopFallback = null;
        try {
          desktopFallback = resolveDesktopEngineFallback();
        } catch (fallbackErr) {
          log(`Desktop engine fallback rejected (${fallbackErr.message}); checking local engine.json`);
        }
        spawnLocalEngine(desktopFallback);
      },
    );
    return;
  }

  spawnLocalEngine(discovery);
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
  const port = engine.port;
  const token = engine.token;
  engine.endpoint = null;
  engine.session = null;
  transitionEnginePhase('stopped');
  if (engine.shared) {
    log('shared desktop engine: leaving process intact on Resolve exit');
    return;
  }
  const child = engine.child;
  if (!child) return;
  if (!engine.exited && port) {
    await postShutdown(port, token, SHUTDOWN_POST_TIMEOUT_MS);
    if (await waitForExit(1_000)) {
      signalEngineTree(child, 'SIGKILL');
      return;
    }
  }
  signalEngineTree(child, 'SIGTERM');
  if (!engine.exited && await waitForExit(SHUTDOWN_SIGTERM_GRACE_MS)) {
    signalEngineTree(child, 'SIGKILL');
    return;
  }
  log('engine ignored SIGTERM; sending SIGKILL');
  signalEngineTree(child, 'SIGKILL');
  if (!engine.exited) await waitForExit(1_000);
}

// ---------------------------------------------------------------------------------------------
// Resolve (WorkflowIntegration.node)
// ---------------------------------------------------------------------------------------------

let WorkflowIntegration = null;
let wiInitialized = false; // Initialize() succeeded — CleanUp() must run on quit even if later steps failed
let resolveObj = null; // Resolve scripting object (promise flavour when available)
let host = 'electron'; // 'resolve' | 'electron'
const knownItems = new Map(); // mediaId -> MediaPoolItem handed out earlier (verified before use)

let lastSdkPreflight = null;

function getUserSdkCacheDir() {
  if (process.platform === 'darwin') {
    return path.join(os.homedir(), 'Library', 'Application Support', 'HawaVoClean', 'resolve-sdk');
  }
  return path.join(os.homedir(), '.config', 'hawavoclean', 'resolve-sdk');
}

function loadWorkflowIntegration(options = {}) {
  const pluginDir = options.pluginDir || __dirname;
  const localFile = path.join(pluginDir, 'WorkflowIntegration.node');
  const requireFn = options.requireFn || require;
  const discoverFn = options.discoverWorkflowIntegrationSdk || discoverWorkflowIntegrationSdk;
  const acquireFn = options.acquireWorkflowIntegrationNode || acquireWorkflowIntegrationNode;
  const cacheDir = options.cacheDir || getUserSdkCacheDir();

  if (fs.existsSync(localFile)) {
    try {
      // eslint-disable-next-line import/no-dynamic-require, global-require
      return requireFn(localFile);
    } catch (err) {
      log(`WorkflowIntegration.node could not be loaded (${err.message}); running standalone`);
      return null;
    }
  }

  // Legally discover and acquire Blackmagic's local WorkflowIntegration.node
  // from the user's licensed DaVinci Resolve Studio installation with checksum verification.
  const discovery = discoverFn(options.sdkOptions);
  lastSdkPreflight = discovery;
  if (!discovery.ok || !discovery.path) {
    log(`WorkflowIntegration.node unavailable (${discovery.code}): ${discovery.repairGuidance || 'running standalone'}`);
    return null;
  }

  let candidatePath = null;
  const acquired = acquireFn(pluginDir, {
    ...(options.sdkOptions || {}),
    overrideNodePath: discovery.path,
  });
  if (acquired.ok && acquired.installedPath) {
    log(`acquired local WorkflowIntegration.node into plugin directory (sha256=${acquired.sha256 || 'verified'})`);
    candidatePath = acquired.installedPath;
  } else {
    // Plugin directory under /Library/... is typically root-owned at runtime;
    // stage a checksum-verified copy in the user's Application Support cache or load directly.
    const cached = acquireFn(cacheDir, {
      ...(options.sdkOptions || {}),
      overrideNodePath: discovery.path,
    });
    if (cached.ok && cached.installedPath) {
      log(`acquired local WorkflowIntegration.node into user cache (sha256=${cached.sha256 || 'verified'})`);
      candidatePath = cached.installedPath;
    } else {
      log(`using verified local Resolve SDK WorkflowIntegration.node directly (${discovery.path})`);
      candidatePath = discovery.path;
    }
  }

  try {
    // eslint-disable-next-line import/no-dynamic-require, global-require
    return requireFn(candidatePath);
  } catch (err) {
    log(`WorkflowIntegration.node could not be loaded from ${candidatePath} (${err.message}); running standalone`);
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

function parseTimecodeFrames(tc, fps) {
  if (typeof tc !== 'string') return null;
  const trimmed = tc.trim();
  const m = trimmed.match(/^(\d+)[:;](\d+)[:;](\d+)([:;])(\d+)$/);
  if (!m) return null;
  const rate = Number(fps);
  if (!Number.isFinite(rate) || rate <= 0) return null;
  const h = Number(m[1]);
  const mi = Number(m[2]);
  const s = Number(m[3]);
  const isDropFrame = trimmed.includes(';');
  const f = Number(m[5]);
  const nominalRate = Math.round(rate);
  if (!Number.isFinite(nominalRate) || nominalRate <= 0) return null;
  const totalNdf = ((h * 3600) + (mi * 60) + s) * nominalRate + f;
  if (isDropFrame && (Math.abs(rate - 29.97) < 0.05 || Math.abs(rate - 59.94) < 0.05)) {
    const dropFrames = Math.abs(rate - 59.94) < 0.05 ? 4 : 2;
    const totalMinutes = h * 60 + mi;
    return totalNdf - dropFrames * (totalMinutes - Math.floor(totalMinutes / 10));
  }
  return totalNdf;
}

async function getTimelineFps(timeline) {
  if (timeline && typeof timeline.GetSetting === 'function') {
    try {
      const raw = await timeline.GetSetting('timelineFrameRate');
      const parsed = Number(raw);
      if (Number.isFinite(parsed) && parsed > 0 && parsed <= 1000) {
        return parsed;
      }
    } catch (_) {
      /* fallback below */
    }
  }
  return 24;
}

async function getTimelinePlayheadFrame(timeline, fps) {
  if (!timeline) return null;
  if (typeof timeline.GetCurrentTimecode === 'function') {
    try {
      const tc = await timeline.GetCurrentTimecode();
      const frame = parseTimecodeFrames(tc, fps);
      if (frame !== null) return frame;
    } catch (_) {}
  }
  if (typeof timeline.GetStartFrame === 'function') {
    try {
      const start = Number(await timeline.GetStartFrame());
      if (Number.isFinite(start) && start >= 0) return Math.round(start);
    } catch (_) {}
  }
  return null;
}

async function mediaItemHasAudio(item) {
  if (!item || typeof item.GetClipProperty !== 'function') return true;
  try {
    const channels = await item.GetClipProperty('Audio Ch');
    if (channels !== undefined && channels !== null && String(channels).trim() !== '') {
      const n = Number(channels);
      if (Number.isFinite(n)) return n > 0;
    }
  } catch (_) {}
  return true;
}

async function buildTimelineClipInfo(timelineItem, { trackIndex, mediaType, fps }) {
  if (!timelineItem || typeof timelineItem.GetMediaPoolItem !== 'function') return null;
  const item = await timelineItem.GetMediaPoolItem();
  if (!item) return null;
  const info = await clipInfo(item);
  const startRaw = typeof timelineItem.GetStart === 'function' ? Number(await timelineItem.GetStart()) : Number.NaN;
  const endRaw = typeof timelineItem.GetEnd === 'function' ? Number(await timelineItem.GetEnd()) : Number.NaN;
  const leftRaw = typeof timelineItem.GetLeftOffset === 'function' ? Number(await timelineItem.GetLeftOffset()) : Number.NaN;
  const durRaw = typeof timelineItem.GetDuration === 'function' ? Number(await timelineItem.GetDuration()) : Number.NaN;

  const startFrame = Number.isFinite(startRaw) && startRaw >= 0 ? Math.round(startRaw) : 0;
  const leftOffset = Number.isFinite(leftRaw) && leftRaw >= 0 ? Math.round(leftRaw) : 0;
  let durationFrames = null;
  if (Number.isFinite(durRaw) && durRaw > 0) {
    durationFrames = Math.round(durRaw);
  } else if (Number.isFinite(endRaw) && Number.isFinite(startRaw) && endRaw > startRaw) {
    durationFrames = Math.round(endRaw - startRaw);
  }

  return {
    ...info,
    startFrame,
    leftOffset,
    ...(durationFrames !== null ? { durationFrames } : {}),
    trackIndex,
    mediaType,
    fps,
  };
}

async function findActiveAudioTimelineItem(timeline, playheadFrame) {
  if (
    !timeline ||
    typeof timeline.GetTrackCount !== 'function' ||
    typeof timeline.GetItemListInTrack !== 'function'
  ) {
    return null;
  }

  let audioTracks = 0;
  try {
    audioTracks = Number(await timeline.GetTrackCount('audio')) || 0;
  } catch (_) {
    return null;
  }

  let nearest = null;
  let nearestDistance = Number.POSITIVE_INFINITY;

  for (let trackIdx = 1; trackIdx <= audioTracks; trackIdx += 1) {
    let items = [];
    try {
      items = (await timeline.GetItemListInTrack('audio', trackIdx)) || [];
    } catch (_) {
      items = [];
    }
    for (const tItem of items) {
      if (!tItem) continue;
      let start = Number.NaN;
      let end = Number.NaN;
      try {
        if (typeof tItem.GetStart === 'function') start = Number(await tItem.GetStart());
        if (typeof tItem.GetEnd === 'function') end = Number(await tItem.GetEnd());
        if (!Number.isFinite(end) && Number.isFinite(start) && typeof tItem.GetDuration === 'function') {
          const dur = Number(await tItem.GetDuration());
          if (Number.isFinite(dur) && dur >= 0) end = start + dur;
        }
      } catch (_) {
        continue;
      }

      if (playheadFrame === null) {
        return { item: tItem, trackIndex: trackIdx };
      }

      if (Number.isFinite(start)) {
        const effectiveEnd = Number.isFinite(end) && end > start ? end : start + 1;
        if (playheadFrame >= start && playheadFrame < effectiveEnd) {
          return { item: tItem, trackIndex: trackIdx };
        }
        const distance = playheadFrame < start
          ? start - playheadFrame
          : playheadFrame - effectiveEnd;
        if (distance < nearestDistance) {
          nearestDistance = distance;
          nearest = { item: tItem, trackIndex: trackIdx };
        }
      } else if (!nearest) {
        nearest = { item: tItem, trackIndex: trackIdx };
      }
    }
  }

  return nearest;
}

async function findTimelineMatchForMediaId(timeline, mediaId, playheadFrame) {
  if (!timeline || !mediaId) return null;
  for (const [trackType, mediaType] of [['audio', 'audio'], ['video', 'video']]) {
    if (typeof timeline.GetTrackCount !== 'function' || typeof timeline.GetItemListInTrack !== 'function') continue;
    let count = 0;
    try {
      count = Number(await timeline.GetTrackCount(trackType)) || 0;
    } catch (_) {
      continue;
    }
    let best = null;
    let bestDistance = Number.POSITIVE_INFINITY;
    for (let trackIdx = 1; trackIdx <= count; trackIdx += 1) {
      let items = [];
      try {
        items = (await timeline.GetItemListInTrack(trackType, trackIdx)) || [];
      } catch (_) {
        continue;
      }
      for (const tItem of items) {
        if (!tItem || typeof tItem.GetMediaPoolItem !== 'function') continue;
        try {
          const mp = await tItem.GetMediaPoolItem();
          if (!mp || String((await mp.GetMediaId()) ?? '') !== mediaId) continue;
          const start = typeof tItem.GetStart === 'function' ? Number(await tItem.GetStart()) : Number.NaN;
          const end = typeof tItem.GetEnd === 'function' ? Number(await tItem.GetEnd()) : Number.NaN;
          if (playheadFrame !== null && Number.isFinite(start)) {
            const effectiveEnd = Number.isFinite(end) && end > start ? end : start + 1;
            if (playheadFrame >= start && playheadFrame < effectiveEnd) {
              return { item: tItem, trackIndex: trackIdx, mediaType };
            }
            const dist = playheadFrame < start ? start - playheadFrame : playheadFrame - effectiveEnd;
            if (dist < bestDistance) {
              bestDistance = dist;
              best = { item: tItem, trackIndex: trackIdx, mediaType };
            }
          } else if (!best) {
            best = { item: tItem, trackIndex: trackIdx, mediaType };
          }
        } catch (_) {}
      }
    }
    if (best) return best;
  }
  return null;
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

async function resolveGetSelectedClip(deps = {}) {
  const getProjectFn = deps.getProject || getProject;
  const getMediaPoolFn = deps.getMediaPool || getMediaPool;

  const mediaPool = await getMediaPoolFn();
  const project = await getProjectFn();
  const timeline = project && typeof project.GetCurrentTimeline === 'function'
    ? await project.GetCurrentTimeline()
    : null;
  const fps = await getTimelineFps(timeline);
  const playheadFrame = await getTimelinePlayheadFrame(timeline, fps);

  if (mediaPool && typeof mediaPool.GetSelectedClips === 'function') {
    const selected = (await mediaPool.GetSelectedClips()) || [];
    if (selected.length > 0 && selected[0]) {
      const base = await clipInfo(selected[0]);
      const tlMatch = await findTimelineMatchForMediaId(timeline, base.mediaId, playheadFrame);
      if (tlMatch) {
        const enriched = await buildTimelineClipInfo(tlMatch.item, {
          trackIndex: tlMatch.trackIndex,
          mediaType: tlMatch.mediaType,
          fps,
        });
        if (enriched) return enriched;
      }
      return { ...base, fps };
    }
  }

  if (!timeline) return null;

  const videoItem = typeof timeline.GetCurrentVideoItem === 'function'
    ? await timeline.GetCurrentVideoItem()
    : null;
  let videoFallbackInfo = null;
  if (videoItem && typeof videoItem.GetMediaPoolItem === 'function') {
    const mpItem = await videoItem.GetMediaPoolItem();
    if (mpItem) {
      const hasAudio = await mediaItemHasAudio(mpItem);
      const videoTrackIdx = typeof videoItem.GetTrackIndex === 'function'
        ? (Number(await videoItem.GetTrackIndex()) || 1)
        : 1;
      const built = await buildTimelineClipInfo(videoItem, {
        trackIndex: videoTrackIdx,
        mediaType: 'video',
        fps,
      });
      if (built && hasAudio) {
        return built;
      }
      videoFallbackInfo = built;
    }
  }

  // Inspect audio timeline tracks near the playhead when there is no video item or the video clip has no audio.
  const activeAudio = await findActiveAudioTimelineItem(timeline, playheadFrame);
  if (activeAudio) {
    const audioInfo = await buildTimelineClipInfo(activeAudio.item, {
      trackIndex: activeAudio.trackIndex,
      mediaType: 'audio',
      fps,
    });
    if (audioInfo) return audioInfo;
  }

  return videoFallbackInfo;
}

async function resolveImportMedia(filePath) {
  if (typeof filePath !== 'string' || !path.isAbsolute(filePath)) throw new Error('importMedia: path must be an absolute path.');
  const mediaPool = await getMediaPool();
  if (!mediaPool) throw new Error('No project is open in DaVinci Resolve.');
  const items = (await mediaPool.ImportMedia([filePath])) || [];
  if (!items.length || !items[0]) return null;
  return clipInfo(items[0]);
}

let timelineTxManager = null;
function getTimelineTxManager() {
  if (!timelineTxManager) {
    timelineTxManager = new TimelineTransactionManager({
      getProject,
      getMediaPool,
      findItemByMediaId,
      log,
    });
  }
  return timelineTxManager;
}

async function resolveReplaceClip(mediaId, filePath) {
  const mgr = getTimelineTxManager();
  const res = await mgr.transactionalReplace(mediaId, filePath);
  return res.success;
}

async function resolveAppendToTimeline(target) {
  const item = await findItemByMediaId(target);
  if (item) {
    const mediaPool = await getMediaPool();
    if (!mediaPool) throw new Error('No project is open in DaVinci Resolve.');
    const result = await mediaPool.AppendToTimeline([item]);
    return Array.isArray(result) ? result.length > 0 : Boolean(result);
  }
  const mgr = getTimelineTxManager();
  const res = await mgr.transactionalAppend(target);
  return res.success;
}

async function resolveNewTrack(mediaId, filePath, options) {
  const mgr = getTimelineTxManager();
  return mgr.transactionalNewTrack(mediaId, filePath, options);
}

async function resolveBatchCleanTrack(payload) {
  const validated = validateBatchCleanTrackPayload(payload);
  const mgr = getTimelineTxManager();
  return mgr.transactionalBatchTrackClean(validated);
}

async function resolveApplyGuardMarkers(payload) {
  const validated = validateApplyGuardMarkersPayload(payload);
  const mgr = getTimelineTxManager();
  return mgr.applyGuardMarkers(validated);
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

const AUDIO_EXTENSIONS = ['wav', 'wave', 'aif', 'aiff', 'aifc', 'flac', 'mp3', 'm4a', 'mp4'];

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

function errorPageHtml(title, detail) {
  const cmd = engine.config ? engine.config.command.join(' ') : '(HawaVoClean engine not configured)';
  const cwd = engine.config ? engine.config.cwd : '';
  const tail = engine.stderrTail.length ? engine.stderrTail.join('\n') : '(no diagnostic output)';
  return `<!doctype html><html><head><meta charset="utf-8"><meta name="referrer" content="no-referrer"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'"><title>${escapeHtml(APP_TITLE)} — error</title>
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
  <dt>Engine</dt><dd><code>${escapeHtml(cmd)}</code></dd>
  <dt>Working dir</dt><dd><code>${escapeHtml(cwd)}</code></dd>
  <dt>Plugin dir</dt><dd><code>${escapeHtml(__dirname)}</code></dd>
</dl>
<h2>Diagnostic log</h2>
<pre>${escapeHtml(tail)}</pre>
<h2>How to fix this</h2>
<ul>
  <li>Launch <b>HawaVoClean Desktop</b> from <code>/Applications/HawaVoClean.app</code>, or reinstall HawaVoClean from the official installer package.</li>
  <li>If the plugin files are missing or damaged, quit DaVinci Resolve Studio and re-run the HawaVoClean Resolve Plugin installer package (<code>.pkg</code>).</li>
  <li>Close this window and reopen <b>Workspace → Workflow Integrations → HawaVoClean</b> in DaVinci Resolve Studio.</li>
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
      partition: SESSION_PARTITION,
      webSecurity: true,
      allowRunningInsecureContent: false,
      webviewTag: false,
      navigateOnDragDrop: false,
      safeDialogs: true,
      spellcheck: false,
    },
  });
  mainWindow.on('closed', () => {
    mainWindow = null;
  });
  // This product has no trusted external-navigation surface. A compromised renderer must not
  // turn shell.openExternal into a URL launcher, so every popup is denied without side effects.
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    securityEvents.blockedWindows += 1;
    log(`blocked new window to ${url}`);
    return { action: 'deny' };
  });
  // Deny-by-default navigation: the only renderer-initiated navigation allowed is a reload of
  // the page that is already loaded. Foreign file:, http(s), data: and custom schemes are blocked
  // so the privileged preload never attaches to unexpected content.
  const contents = mainWindow.webContents;
  contents.on('will-navigate', (event, url) => {
    if (url === contents.getURL()) return; // allow reload
    event.preventDefault();
    securityEvents.blockedNavigations += 1;
    log(`blocked navigation to ${url}`);
  });
  contents.on('will-attach-webview', (event) => {
    securityEvents.blockedWebviews += 1;
    event.preventDefault();
  });
  if (DEVTOOLS) mainWindow.webContents.openDevTools({ mode: 'detach' });

  if (loadUi) {
    const indexHtml = path.join(__dirname, 'index.html');
    if (fs.existsSync(indexHtml)) {
      mainWindow.loadURL(APP_ENTRY_URL).catch((err) => log(`app URL load failed: ${err.message}`));
    } else {
      const html = `<!doctype html><html><head><meta charset="utf-8"><meta name="referrer" content="no-referrer"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'"><title>${APP_TITLE}</title></head>
<body style="margin:0;background:#0e1013;color:#e6e8eb;font:14px/1.5 -apple-system,Helvetica,Arial,sans-serif">
<main style="max-width:760px;margin:0 auto;padding:48px 32px">
<h1 style="font-size:20px;font-weight:600;margin:0 0 8px">Plugin installation incomplete</h1>
<p style="color:#b7bcc4">The HawaVoClean interface files could not be found in this plugin folder. Quit DaVinci Resolve Studio and reinstall HawaVoClean from the installer package, or launch HawaVoClean Desktop from <code>/Applications/HawaVoClean.app</code>.</p>
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
    event.returnValue = isTrustedIpcSender(event) ? host : null;
  });

  ipcMain.handle('hawa:engine:endpoint', async (event) => {
    requireTrustedIpcSender(event);
    if (engine.failure) throw engine.failure;
    return engine.ready; // rejects with the failure Error if the engine never comes up
  });

  ipcMain.handle('hawa:files:pick', async (event) => {
    requireTrustedIpcSender(event);
    const owner = BrowserWindow.fromWebContents(event.sender) || mainWindow || undefined;
    const result = await dialog.showOpenDialog(owner, {
      title: 'Choose an audio or video file',
      buttonLabel: 'Open',
      properties: ['openFile', 'treatPackageAsDirectory'],
      filters: [
        { name: 'Audio and video', extensions: AUDIO_EXTENSIONS },
      ],
    });
    if (result.canceled || !result.filePaths || result.filePaths.length === 0) return null;
    return registerNativePath(result.filePaths[0]);
  });

  ipcMain.handle('hawa:files:register-dropped', async (event, filePath) => {
    requireTrustedIpcSender(event);
    return registerNativePath(filePath);
  });

  ipcMain.handle('hawa:files:reveal', async (event, filePath) => {
    requireTrustedIpcSender(event);
    if (typeof filePath !== 'string' || !path.isAbsolute(filePath)) throw new Error('revealInFinder: path must be an absolute path.');
    shell.showItemInFolder(filePath);
  });

  ipcMain.handle('hawa:resolve:selected', async (event) => {
    requireTrustedIpcSender(event);
    const clip = await resolveGetSelectedClip();
    if (!clip) return null;
    return { ...clip, filePath: await registerNativePath(clip.filePath) };
  });
  ipcMain.handle('hawa:resolve:import', async (event, filePath) => {
    requireTrustedIpcSender(event);
    return resolveImportMedia(filePath);
  });
  ipcMain.handle('hawa:resolve:replace', async (event, mediaId, filePath) => {
    requireTrustedIpcSender(event);
    return resolveReplaceClip(mediaId, filePath);
  });
  ipcMain.handle('hawa:resolve:append', async (event, mediaId) => {
    requireTrustedIpcSender(event);
    return resolveAppendToTimeline(mediaId);
  });
  ipcMain.handle('hawa:resolve:new-track', async (event, mediaId, filePath, options) => {
    requireTrustedIpcSender(event);
    return resolveNewTrack(mediaId, filePath, options);
  });
  ipcMain.handle('hawa:resolve:batch-clean-track', async (event, payload) => {
    requireTrustedIpcSender(event);
    return resolveBatchCleanTrack(payload);
  });
  ipcMain.handle('hawa:resolve:apply-guard-markers', async (event, payload) => {
    requireTrustedIpcSender(event);
    return resolveApplyGuardMarkers(payload);
  });
  ipcMain.handle('hawa:resolve:context', async (event) => {
    requireTrustedIpcSender(event);
    return resolveGetContext();
  });
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
    out.endpointHasCredential = Object.keys(ep).some((key) => /token|auth|session|secret/i.test(key)) || /[?&](?:token|auth|session)=/i.test(ep.baseUrl);
    const r = await fetch(ep.baseUrl + '/api/health', {
      headers: {
        'X-Hawa-Token': 'renderer-controlled-root',
        Authorization: 'Bearer renderer-controlled-session',
        Range: 'bytes=0-1023',
      },
    });
    out.health = { status: r.status, body: await r.json() };
    try {
      await fetch(ep.baseUrl + '/api/session', { method: 'POST' });
      out.sessionBootstrapBlocked = false;
    } catch {
      out.sessionBootstrapBlocked = true;
    }
    const csp = document.querySelector('meta[http-equiv="Content-Security-Policy"]')?.content || '';
    out.security = { csp };
    delete window.__hawaInjectedInlineRan;
    const inline = document.createElement('script');
    inline.textContent = 'window.__hawaInjectedInlineRan = true';
    document.head.appendChild(inline);
    await new Promise((resolve) => setTimeout(resolve, 0));
    inline.remove();
    out.security.inlineScriptBlocked = window.__hawaInjectedInlineRan !== true;
    try {
      await fetch('https://example.invalid/hawavoclean-must-not-connect');
      out.security.remoteFetchBlocked = false;
    } catch {
      out.security.remoteFetchBlocked = true;
    }
    out.security.popupBlocked = window.open('https://example.invalid/hawavoclean-must-not-open') === null;
    try {
      const permission = await navigator.permissions.query({ name: 'geolocation' });
      out.security.geolocationDenied = permission.state === 'denied';
    } catch {
      out.security.geolocationDenied = true;
    }
    try {
      const traversal = await fetch('${APP_SCHEME}://app/%2e%2e%2fpackage.json');
      out.security.pathTraversalBlocked = !traversal.ok;
    } catch {
      out.security.pathTraversalBlocked = true;
    }
    try {
      await navigator.serviceWorker.register('./hawavoclean-must-not-register.js');
      out.security.serviceWorkerBlocked = false;
    } catch {
      out.security.serviceWorkerBlocked = true;
    }
    out.security.blobWorkerRan = await new Promise((resolve) => {
      const workerUrl = URL.createObjectURL(new Blob(['postMessage("ok")'], { type: 'text/javascript' }));
      const worker = new Worker(workerUrl);
      const timer = setTimeout(() => { worker.terminate(); URL.revokeObjectURL(workerUrl); resolve(false); }, 2000);
      worker.onmessage = (event) => { clearTimeout(timer); worker.terminate(); URL.revokeObjectURL(workerUrl); resolve(event.data === 'ok'); };
      worker.onerror = () => { clearTimeout(timer); worker.terminate(); URL.revokeObjectURL(workerUrl); resolve(false); };
    });
    // Test pages may publish extra probe results here (e.g. module script / worker checks).
    if (window.__hawaSelftestPage !== undefined) {
      out.page = await Promise.resolve(window.__hawaSelftestPage);
    }
    return out;
  })()`;
  try {
    const result = await mainWindow.webContents.executeJavaScript(js, true);
    result.enginePid = engine.child ? engine.child.pid : null;
    result.runtime = RUNTIME_EVIDENCE;
    result.securityEvents = { ...securityEvents };
    console.log('HAWA_SELFTEST_RESULT ' + JSON.stringify(result));
  } catch (err) {
    console.log('HAWA_SELFTEST_ERROR ' + JSON.stringify({ title: 'selftest threw', detail: String(err && err.message ? err.message : err) }));
  }
  if (!SELFTEST_HARD_CRASH) app.quit();
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

if (isElectronRuntime) {
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
    configureSessionSecurity();
    log(`runtime evidence: ${JSON.stringify(RUNTIME_EVIDENCE)}; sandbox=true contextIsolation=true nodeIntegration=false`);
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
}

module.exports = {
  loadWorkflowIntegration,
  resolveDesktopEngineFallback,
  resolveGetSelectedClip,
  getTimelineFps,
  parseTimecodeFrames,
  findActiveAudioTimelineItem,
  errorPageHtml,
  getLastSdkPreflight: () => lastSdkPreflight,
};
