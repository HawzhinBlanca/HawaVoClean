import fs from 'node:fs';
import path from 'node:path';

import {
  app,
  BrowserWindow,
  dialog,
  ipcMain,
  type IpcMainInvokeEvent,
  protocol,
  session,
  shell,
} from 'electron';

import {
  AUDIO_EXTENSIONS,
  type AppInfo,
  type DiagnosticsState,
  IPC,
  parseExportRequest,
  requireAbsolutePath,
  requireSupportedMediaPath,
  safeClearSessionStorage,
  safeSuggestedExportName,
  SESSION_PARTITION,
  type UpdateState,
} from './contracts.js';
import { EngineBroker } from './engine.js';
import { DiagnosticsManager, type RedactedDiagnosticReport } from './diagnostics.js';
import { UpdateManager } from './updates/index.js';
import { packagedProofSelfTestAllowed } from './proof.js';
import {
  APP_ENTRY_URL,
  APP_SCHEME,
  isAllowedRendererRequest,
  isEngineApiRequest,
  isTrustedIpcSenderEvent,
  isTrustedRendererUrl,
  resolveAppAsset,
  withoutRendererCredentials,
  withEngineAuthorization,
} from './security.js';

const APP_TITLE = 'HawaVoClean';
const QUIT_DEADLINE_MS = 10_000;
const DEVTOOLS = !app.isPackaged && process.env.HAWAVOCLEAN_DESKTOP_DEVTOOLS === '1';
const SELFTEST_REQUESTED = process.env.HAWAVOCLEAN_DESKTOP_SELFTEST === '1';
const PACKAGED_PROOF_SELFTEST = packagedProofSelfTestAllowed(process.resourcesPath, app.isPackaged);
const SELFTEST = SELFTEST_REQUESTED && (!app.isPackaged || PACKAGED_PROOF_SELFTEST);
const REJECTED_PACKAGED_SELFTEST = SELFTEST_REQUESTED && app.isPackaged && !PACKAGED_PROOF_SELFTEST;
const SELFTEST_HARD_CRASH = SELFTEST && process.env.HAWAVOCLEAN_DESKTOP_SELFTEST_HARD_CRASH === '1';

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

app.setName(APP_TITLE);
app.setAppUserModelId('com.hawavoclean.desktop');

const repositoryRoot = path.resolve(__dirname, '..', '..');
const brandingRoot = app.isPackaged
  ? path.join(process.resourcesPath, 'branding')
  : path.join(__dirname, '..', 'build');
const brandedIconPath = path.join(brandingRoot, 'icon.png');

if (fs.existsSync(brandedIconPath)) {
  app.setAboutPanelOptions({
    applicationName: APP_TITLE,
    applicationVersion: app.getVersion(),
    copyright: 'Copyright © Hawzhin',
    version: app.getVersion(),
    iconPath: brandedIconPath,
  });
}

const uiRoot = app.isPackaged ? path.join(process.resourcesPath, 'ui') : path.join(repositoryRoot, 'ui', 'dist');
const broker = new EngineBroker({
  packaged: app.isPackaged,
  resourcesPath: process.resourcesPath,
  repositoryRoot,
  userData: app.getPath('userData'),
  temp: app.getPath('temp'),
  platform: process.platform,
});
const diagnosticsManager = new DiagnosticsManager();
const updateManager = new UpdateManager({
  currentVersion: app.getVersion() || '3.3.0',
  feedUrl: process.env.HAWAVOCLEAN_UPDATE_FEED_URL,
  publicKey: process.env.HAWAVOCLEAN_UPDATE_PUBLIC_KEY,
  updateDir: path.join(app.getPath('userData'), 'updates'),
  dbPath: path.join(app.getPath('userData'), 'state', 'jobs.sqlite3'),
});

let mainWindow: BrowserWindow | null = null;
let quitState: 'running' | 'stopping' | 'done' = 'running';

function isTrustedIpcSender(event: IpcMainInvokeEvent): boolean {
  return isTrustedIpcSenderEvent(
    event,
    mainWindow?.webContents,
    !mainWindow || mainWindow.isDestroyed(),
    event.sender.getURL(),
  );
}

function requireTrustedIpcSender(event: IpcMainInvokeEvent): void {
  if (!isTrustedIpcSender(event)) throw new Error('IPC sender is not the trusted HawaVoClean renderer.');
}

function ownerFor(event: IpcMainInvokeEvent): BrowserWindow | null {
  return BrowserWindow.fromWebContents(event.sender) ?? mainWindow;
}

function openDialogFor(event: IpcMainInvokeEvent, options: Electron.OpenDialogOptions) {
  const owner = ownerFor(event);
  return owner ? dialog.showOpenDialog(owner, options) : dialog.showOpenDialog(options);
}

function saveDialogFor(event: IpcMainInvokeEvent, options: Electron.SaveDialogOptions) {
  const owner = ownerFor(event);
  return owner ? dialog.showSaveDialog(owner, options) : dialog.showSaveDialog(options);
}

function configureSession(): void {
  const appSession = session.fromPartition(SESSION_PARTITION);
  // Ensure legacy partition disk storage is cleaned and initial in-memory cache is flushed
  void safeClearSessionStorage(app.getPath('userData'), {
    clearCache: () => appSession.clearCache(),
    clearStorageData: () =>
      appSession.clearStorageData({
        storages: ['cookies', 'localstorage', 'indexdb', 'serviceworkers', 'cachestorage'],
      }),
    clearCodeCaches: () => appSession.clearCodeCaches({}),
  });
  const requestAuthorization = new Map<number, string>();
  appSession.protocol.handle(APP_SCHEME, async (request) => {
    if (request.method !== 'GET') return new Response('not found', { status: 404 });
    const asset = resolveAppAsset(uiRoot, request.url);
    if (asset === null) return new Response('not found', { status: 404 });
    const contentType =
      new Map([
        ['.html', 'text/html; charset=utf-8'],
        ['.js', 'text/javascript; charset=utf-8'],
        ['.css', 'text/css; charset=utf-8'],
        ['.json', 'application/json; charset=utf-8'],
        ['.svg', 'image/svg+xml'],
        ['.png', 'image/png'],
        ['.woff2', 'font/woff2'],
      ]).get(path.extname(asset).toLowerCase()) ?? 'application/octet-stream';
    const bytes = Uint8Array.from(await fs.promises.readFile(asset));
    return new Response(bytes, {
      status: 200,
      headers: {
        'content-type': contentType,
        'cache-control': asset.endsWith('index.html') ? 'no-store' : 'public, max-age=31536000, immutable',
        'x-content-type-options': 'nosniff',
      },
    });
  });

  const trustedClipboardWrite = (
    contents: Electron.WebContents | null,
    permission: string,
    requestingUrl: string,
  ): boolean =>
    permission === 'clipboard-sanitized-write' &&
    contents !== null &&
    mainWindow !== null &&
    !mainWindow.isDestroyed() &&
    contents === mainWindow.webContents &&
    isTrustedRendererUrl(requestingUrl);

  appSession.setPermissionRequestHandler((contents, permission, callback, details) => {
    callback(trustedClipboardWrite(contents, permission, details.requestingUrl));
  });
  appSession.setPermissionCheckHandler((contents, permission, requestingOrigin, details) => {
    const requestingUrl = details.requestingUrl ?? requestingOrigin;
    return trustedClipboardWrite(contents, permission, requestingUrl);
  });
  appSession.setDevicePermissionHandler(() => false);
  appSession.webRequest.onBeforeRequest({ urls: ['<all_urls>'] }, (details, callback) => {
    const allowed = isAllowedRendererRequest(details.url, broker.origin);
    callback({ cancel: !allowed });
  });
  appSession.webRequest.onBeforeSendHeaders({ urls: ['<all_urls>'] }, (details, callback) => {
    if (!isEngineApiRequest(details.url, broker.origin)) {
      callback({ requestHeaders: details.requestHeaders });
      return;
    }
    if (details.method === 'OPTIONS') {
      callback({ requestHeaders: withoutRendererCredentials(details.requestHeaders) });
      return;
    }
    void broker.authorizationHeader().then(
      (authorization) => {
        try {
          const requestHeaders = withEngineAuthorization(details.requestHeaders, authorization);
          requestAuthorization.set(details.id, authorization);
          callback({ requestHeaders });
        } catch {
          callback({ cancel: true });
        }
      },
      () => callback({ cancel: true }),
    );
  });
  appSession.webRequest.onCompleted({ urls: ['<all_urls>'] }, (details) => {
    const authorization = requestAuthorization.get(details.id);
    requestAuthorization.delete(details.id);
    if (
      authorization !== undefined &&
      details.statusCode === 401 &&
      isEngineApiRequest(details.url, broker.origin)
    ) {
      broker.invalidateSession(authorization);
    }
  });
  appSession.webRequest.onErrorOccurred({ urls: ['<all_urls>'] }, (details) => {
    requestAuthorization.delete(details.id);
  });
}

function registerIpc(): void {
  ipcMain.handle(IPC.engineEndpoint, async (event) => {
    requireTrustedIpcSender(event);
    return broker.endpoint();
  });

  ipcMain.handle(IPC.pickAudioFiles, async (event) => {
    requireTrustedIpcSender(event);
    const result = await openDialogFor(event, {
      title: 'Choose audio or video files',
      buttonLabel: 'Open',
      properties: ['openFile', 'multiSelections', 'treatPackageAsDirectory'],
      filters: [
        { name: 'Audio and video', extensions: [...AUDIO_EXTENSIONS] },
      ],
    });
    if (result.canceled) return [];
    const registered: string[] = [];
    for (const filePath of result.filePaths) {
      registered.push(
        (await broker.registerNativeSource(requireSupportedMediaPath(filePath, 'pickAudio'))).path,
      );
    }
    return registered;
  });

  ipcMain.handle(IPC.registerDroppedFile, async (event, value: unknown) => {
    requireTrustedIpcSender(event);
    const filePath = requireSupportedMediaPath(value, 'registerDroppedFile');
    const registered = await broker.registerNativeSource(filePath);
    return { sourceId: registered.sourceId, path: registered.path };
  });

  ipcMain.handle(IPC.pickFolder, async (event) => {
    requireTrustedIpcSender(event);
    const result = await openDialogFor(event, {
      title: 'Choose a folder',
      buttonLabel: 'Choose',
      properties: ['openDirectory', 'createDirectory', 'promptToCreate'],
    });
    return result.canceled ? null : (result.filePaths[0] ?? null);
  });

  ipcMain.handle(IPC.chooseExportPath, async (event, value: unknown) => {
    requireTrustedIpcSender(event);
    const request = parseExportRequest(value);
    const defaultPath = path.join(app.getPath('documents'), safeSuggestedExportName(request));
    const result = await saveDialogFor(event, {
      title: request.kind === 'master' ? 'Save Master WAV' : 'Save Full Processing Record',
      buttonLabel: 'Save',
      defaultPath,
      filters:
        request.kind === 'master'
          ? [{ name: 'WAV audio', extensions: ['wav'] }]
          : [{ name: 'ZIP archive', extensions: ['zip'] }],
      properties: ['showOverwriteConfirmation', 'createDirectory'],
    });
    if (result.canceled || !result.filePath) return null;
    const destination = result.filePath;
    if (request.sourcePath) {
      if (!fs.existsSync(request.sourcePath)) {
        throw new Error('chooseExportPath: source file no longer exists.');
      }
      fs.copyFileSync(request.sourcePath, destination);
    }
    return destination;
  });

  ipcMain.handle(IPC.reveal, async (event, value: unknown) => {
    requireTrustedIpcSender(event);
    const filePath = requireAbsolutePath(value, 'revealInFolder');
    if (!fs.existsSync(filePath)) throw new Error('revealInFolder: file no longer exists.');
    shell.showItemInFolder(filePath);
  });

  ipcMain.handle(IPC.appInfo, (event): AppInfo => {
    requireTrustedIpcSender(event);
    return Object.freeze({
      name: app.getName(),
      version: app.getVersion(),
      locale: app.getLocale(),
      platform: process.platform,
      arch: process.arch,
      packaged: app.isPackaged,
    });
  });

  ipcMain.handle(
    IPC.diagnosticsState,
    (event, payload?: unknown): DiagnosticsState | RedactedDiagnosticReport => {
      requireTrustedIpcSender(event);
      if (payload && typeof payload === 'object' && 'action' in payload) {
        const action = (payload as { action: string; optIn?: boolean }).action;
        if (action === 'setOptIn') {
          return diagnosticsManager.setOptIn(Boolean((payload as { optIn?: boolean }).optIn));
        }
        if (action === 'export') {
          return diagnosticsManager.generateReport({
            appName: app.name,
            appVersion: app.getVersion(),
            packaged: app.isPackaged,
            engineStatus: broker.origin ? 'running' : 'stopped',
            enginePid: broker.pid,
          });
        }
      }
      return diagnosticsManager.getState();
    },
  );

  ipcMain.handle(
    IPC.updateState,
    async (
      event,
      payload?: { action?: 'check' | 'apply' | 'rollback'; version?: string },
    ): Promise<UpdateState> => {
      requireTrustedIpcSender(event);
      if (payload && typeof payload === 'object' && payload.action) {
        if (payload.action === 'check') {
          await updateManager.checkForUpdates();
        } else if (payload.action === 'apply') {
          await updateManager.applyUpdate();
        } else if (payload.action === 'rollback') {
          await updateManager.rollback(payload.version ?? '3.3.0');
        }
      }
      return updateManager.getState();
    },
  );

  ipcMain.handle(IPC.clearLocalData, async (event) => {
    requireTrustedIpcSender(event);
    const appSession = session.fromPartition(SESSION_PARTITION);
    return safeClearSessionStorage(app.getPath('userData'), {
      clearCache: () => appSession.clearCache(),
      clearStorageData: () =>
        appSession.clearStorageData({
          storages: ['cookies', 'localstorage', 'indexdb', 'serviceworkers', 'cachestorage'],
        }),
      clearCodeCaches: () => appSession.clearCodeCaches({}),
    });
  });
}

function createWindow(): BrowserWindow {
  const window = new BrowserWindow({
    width: 1280,
    height: 820,
    minWidth: 960,
    minHeight: 640,
    title: APP_TITLE,
    ...(fs.existsSync(brandedIconPath) ? { icon: brandedIconPath } : {}),
    backgroundColor: '#0e1013',
    show: false,
    autoHideMenuBar: process.platform !== 'darwin',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      sandbox: true,
      contextIsolation: true,
      nodeIntegration: false,
      nodeIntegrationInWorker: false,
      nodeIntegrationInSubFrames: false,
      webviewTag: false,
      webSecurity: true,
      allowRunningInsecureContent: false,
      navigateOnDragDrop: false,
      safeDialogs: true,
      spellcheck: false,
      partition: SESSION_PARTITION,
    },
  });
  mainWindow = window;
  window.once('ready-to-show', () => {
    if (!SELFTEST) window.show();
  });
  window.once('closed', () => {
    if (mainWindow === window) mainWindow = null;
  });

  window.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
  window.webContents.on('will-navigate', (event, url) => {
    if (url !== window.webContents.getURL()) event.preventDefault();
  });
  window.webContents.on('will-attach-webview', (event) => event.preventDefault());
  if (DEVTOOLS) window.webContents.openDevTools({ mode: 'detach' });
  window.webContents.once('did-finish-load', () => {
    if (SELFTEST) void runSelfTest(window);
  });
  void window.loadURL(APP_ENTRY_URL).catch((error: unknown) => {
    dialog.showErrorBox('HawaVoClean failed to load', error instanceof Error ? error.message : String(error));
    app.quit();
  });
  return window;
}

async function runSelfTest(window: BrowserWindow): Promise<void> {
  try {
    const result = (await window.webContents.executeJavaScript(
      `(async () => {
        const bridge = window.hawa;
        if (!bridge || bridge.host !== 'electron') throw new Error('desktop bridge missing');
        const proofChallenge = Array.from(crypto.getRandomValues(new Uint8Array(32)),
          (value) => value.toString(16).padStart(2, '0')).join('');
        const rendererProof = await new Promise((resolve, reject) => {
          let settled = false;
          const finish = (error, value) => {
            if (settled) return;
            settled = true;
            clearInterval(retry);
            clearTimeout(deadline);
            window.removeEventListener('hawavoclean:renderer-proof-response:v1', receive);
            if (error) reject(error);
            else resolve(value);
          };
          const receive = (event) => {
            const detail = event instanceof CustomEvent ? event.detail : null;
            if (!detail || detail.challenge !== proofChallenge) return;
            if (detail.contract !== 'hawavoclean-react-v1' || typeof detail.uiVersion !== 'string') {
              finish(new Error('React renderer returned an invalid proof response'));
              return;
            }
            finish(null, detail);
          };
          const challenge = () => window.dispatchEvent(new CustomEvent(
            'hawavoclean:renderer-proof-challenge:v1', { detail: proofChallenge }));
          window.addEventListener('hawavoclean:renderer-proof-response:v1', receive);
          const retry = setInterval(challenge, 50);
          const deadline = setTimeout(
            () => finish(new Error('React renderer proof timed out')), 5000);
          challenge();
        });
        await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
        const root = document.getElementById('root');
        const shell = root?.firstElementChild ?? null;
        if (!(root instanceof HTMLElement) || !(shell instanceof HTMLElement)) {
          throw new Error('React renderer root is missing');
        }
        const requiredSelectors = [
          'header.header h1.wordmark',
          'section.source[aria-label="Source clip"]',
          'main.main',
          'aside.right[aria-label="Analysis and controls"]',
          'button.process[aria-label="PROCESS"]',
          'footer.footer'
        ];
        const missingSelectors = requiredSelectors.filter((selector) => !shell.querySelector(selector));
        const appInfo = await bridge.app.getInfo();
        const shellStyle = getComputedStyle(shell);
        const shellRect = shell.getBoundingClientRect();
        const renderer = {
          contract: rendererProof.contract,
          challengeVerified: rendererProof.challenge === proofChallenge,
          uiVersion: rendererProof.uiVersion,
          domContract: shell.getAttribute('data-hawa-renderer-contract'),
          domUiVersion: shell.getAttribute('data-hawa-ui-version'),
          rootChildCount: root.childElementCount,
          shellIsOnlyRootChild: root.childElementCount === 1 && root.firstElementChild === shell,
          shellTag: shell.tagName,
          shellClass: shell.className,
          missingSelectors,
          brandText: shell.querySelector('h1.wordmark')?.textContent?.replace(/\\s+/g, '') ?? '',
          openFileText: Array.from(shell.querySelectorAll('section.source button'))
            .map((button) => button.textContent?.trim() ?? '')
            .find((text) => text.startsWith('Open file')) ?? '',
          processLabel: shell.querySelector('button.process')?.getAttribute('aria-label') ?? '',
          display: shellStyle.display,
          width: shellRect.width,
          height: shellRect.height,
          stylesheetCount: document.styleSheets.length,
          moduleScriptCount: document.querySelectorAll('script[type="module"][src]').length,
          documentTitle: document.title
        };
        if (
          !renderer.challengeVerified ||
          renderer.contract !== 'hawavoclean-react-v1' ||
          renderer.uiVersion !== appInfo.version ||
          renderer.domContract !== renderer.contract ||
          renderer.domUiVersion !== renderer.uiVersion ||
          !renderer.shellIsOnlyRootChild ||
          renderer.shellTag !== 'DIV' ||
          renderer.shellClass !== 'app' ||
          renderer.missingSelectors.length !== 0 ||
          !renderer.brandText.includes('HAWAVOCLEAN') ||
          !renderer.brandText.includes('v' + appInfo.version) ||
          !renderer.openFileText.startsWith('Open file') ||
          renderer.processLabel !== 'PROCESS' ||
          renderer.display !== 'grid' ||
          renderer.width < 960 ||
          renderer.height < 640 ||
          renderer.stylesheetCount < 2 ||
          renderer.moduleScriptCount < 1 ||
          renderer.documentTitle !== 'HawaVoClean'
        ) throw new Error('React renderer did not satisfy the packaged UI contract');
        const endpoint = await bridge.engine.getEndpoint();
        const response = await fetch(endpoint.baseUrl + '/api/health', {
          cache: 'no-store'
        });
        const health = await response.json();
        const postResponse = await fetch(endpoint.baseUrl + '/api/peaks', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: '{}',
          cache: 'no-store'
        });
        const mediaResponse = await fetch(endpoint.baseUrl + '/api/audio', { cache: 'no-store' });
        const mediaCacheControl = mediaResponse.headers.get('cache-control') ?? '';
        const mediaNoStore = mediaCacheControl.includes('no-store');
        const sessionClearing = await bridge.session.clearLocalData();
        let remoteBlocked = false;
        try { await fetch('https://example.invalid/hawa-selftest'); }
        catch { remoteBlocked = true; }
        return {
          host: bridge.host,
          topLevelKeys: Object.keys(bridge).sort(),
          fileKeys: Object.keys(bridge.files).sort(),
          app: appInfo,
          renderer,
          diagnostics: await bridge.diagnostics.getState(),
          updates: await bridge.updates.getState(),
          endpointKeys: Object.keys(endpoint).sort(),
          endpointHasSecret: 'token' in endpoint || 'authorization' in endpoint,
          endpointUrlHasAuth: /(?:token|authorization|session)=/i.test(endpoint.baseUrl),
          healthStatus: response.status,
          enginePid: Number.isInteger(health.engine_pid) ? health.engine_pid : null,
          authenticatedPostStatus: postResponse.status,
          authenticatedMediaStatus: mediaResponse.status,
          mediaNoStore,
          sessionClearing,
          remoteBlocked,
          popupBlocked: window.open('https://example.invalid/hawa-selftest') === null,
          nodeHidden: typeof window.require === 'undefined' && typeof window.process === 'undefined'
        };
      })()`,
      true,
    )) as unknown;
    console.log(`HAWA_DESKTOP_SELFTEST_RESULT ${JSON.stringify(result)}`);
  } catch (error) {
    console.error(`HAWA_DESKTOP_SELFTEST_ERROR ${error instanceof Error ? error.stack : String(error)}`);
    process.exitCode = 1;
  } finally {
    // The hard-crash harness deliberately kills Electron after it learns the
    // broker pid.  Calling app.quit() here would exercise graceful cleanup,
    // not the child-owned parent-death watchdog.
    if (!SELFTEST_HARD_CRASH) app.quit();
  }
}

async function stopAndQuit(): Promise<void> {
  const deadline = setTimeout(() => app.exit(1), QUIT_DEADLINE_MS);
  deadline.unref();
  try {
    await broker.stop();
  } finally {
    clearTimeout(deadline);
    quitState = 'done';
    app.quit();
  }
}

const ownsSingleInstance = app.requestSingleInstanceLock();
if (!ownsSingleInstance) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (!mainWindow) return;
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.show();
    mainWindow.focus();
  });

  app.on('before-quit', (event) => {
    if (quitState === 'done') return;
    event.preventDefault();
    if (quitState === 'stopping') return;
    quitState = 'stopping';
    void stopAndQuit();
  });

  app.on('window-all-closed', () => app.quit());
  app.on('certificate-error', (event, _webContents, _url, _error, _certificate, callback) => {
    event.preventDefault();
    callback(false);
  });
  app.whenReady().then(() => {
    if (REJECTED_PACKAGED_SELFTEST) {
      console.error(
        'HAWA_DESKTOP_SELFTEST_ERROR Packaged self-test requires a marked non-distributable full-engine proof.',
      );
      app.exit(1);
      return;
    }
    if (!fs.existsSync(path.join(uiRoot, 'index.html'))) {
      dialog.showErrorBox('HawaVoClean UI missing', `The renderer bundle was not found at ${uiRoot}.`);
      app.quit();
      return;
    }
    configureSession();
    registerIpc();
    broker.start();
    createWindow();
  });

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0 && quitState === 'running') createWindow();
  });
}
