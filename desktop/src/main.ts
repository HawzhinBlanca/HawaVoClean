import fs from 'node:fs';
import path from 'node:path';

import {
  app,
  BrowserWindow,
  dialog,
  ipcMain,
  type IpcMainInvokeEvent,
  Menu,
  type MenuItemConstructorOptions,
  protocol,
  screen,
  session,
  shell,
} from 'electron';

import {
  AUDIO_EXTENSIONS,
  type AppInfo,
  extractExternalMediaPaths,
  IPC,
  IPC_EVENTS,
  loadUserPreferences,
  MASTER_EXPORT_FILTERS,
  parseExportRequest,
  RECORD_BUNDLE_EXPORT_FILTERS,
  redactDiagnosticText,
  requireAbsolutePath,
  requireSupportedMediaPath,
  safeClearSessionStorage,
  safeSuggestedExportName,
  saveUserPreferences,
  SESSION_PARTITION,
  type WindowBounds,
} from './contracts.js';
import { EngineBroker, type NativeSourceRegistration } from './engine.js';
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
const RENDERER_CRASH_WINDOW_MS = 60_000;
const MAX_RENDERER_CRASH_RELOADS = 3;
const BOUNDS_SAVE_DEBOUNCE_MS = 300;
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

let mainWindow: BrowserWindow | null = null;
let mainWindowLoaded = false;
let quitState: 'running' | 'stopping' | 'done' = 'running';
const pendingExternalFiles: string[] = [];
const rendererCrashTimestamps: number[] = [];

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

function resolveInitialWindowBounds(): { width: number; height: number; x?: number; y?: number } {
  const fallback = { width: 1280, height: 820 };
  if (SELFTEST) return fallback;
  const prefs = loadUserPreferences(app.getPath('userData'));
  const saved: WindowBounds | undefined = prefs.windowBounds;
  if (!saved) return fallback;

  const width = Math.max(960, saved.width);
  const height = Math.max(640, saved.height);
  if (saved.x === undefined || saved.y === undefined) {
    return { width, height };
  }
  const visibleOnSomeDisplay = screen.getAllDisplays().some((display) => {
    const area = display.workArea;
    return (
      saved.x! + width > area.x + 64 &&
      saved.x! < area.x + area.width - 64 &&
      saved.y! + height > area.y + 64 &&
      saved.y! < area.y + area.height - 64
    );
  });
  return visibleOnSomeDisplay ? { width, height, x: saved.x, y: saved.y } : { width, height };
}

function persistWindowBounds(window: BrowserWindow): void {
  if (SELFTEST || window.isDestroyed() || window.isMinimized() || window.isFullScreen()) return;
  try {
    const bounds = window.getBounds();
    saveUserPreferences(app.getPath('userData'), {
      windowBounds: {
        x: bounds.x,
        y: bounds.y,
        width: bounds.width,
        height: bounds.height,
      },
    });
  } catch (error) {
    broker.log(`[prefs] failed to persist window bounds: ${String(error)}`);
  }
}

async function dispatchExternalMediaFile(rawFilePath: string): Promise<void> {
  if (SELFTEST || quitState !== 'running') return;
  let validatedPath: string;
  try {
    validatedPath = requireSupportedMediaPath(rawFilePath, 'openFile');
  } catch (error) {
    broker.log(`[open-file] ignored unsupported path: ${error instanceof Error ? error.message : String(error)}`);
    return;
  }
  if (!fs.existsSync(validatedPath)) {
    broker.log('[open-file] ignored missing file path.');
    return;
  }
  if (!mainWindow || mainWindow.isDestroyed() || !mainWindowLoaded) {
    if (!pendingExternalFiles.includes(validatedPath)) {
      pendingExternalFiles.push(validatedPath);
    }
    return;
  }
  try {
    const registered = await broker.registerNativeSource(validatedPath);
    if (!mainWindow || mainWindow.isDestroyed()) return;
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.show();
    mainWindow.focus();
    mainWindow.webContents.send(IPC_EVENTS.openFile, registered);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    broker.log(`[open-file] failed to register native source: ${message}`);
    if (mainWindow && !mainWindow.isDestroyed()) {
      dialog.showErrorBox('Could not open file', `HawaVoClean could not register the selected media file.\n\n${message}`);
    }
  }
}

function flushPendingExternalFiles(): void {
  if (SELFTEST || !mainWindowLoaded || !mainWindow || mainWindow.isDestroyed()) return;
  while (pendingExternalFiles.length > 0) {
    const next = pendingExternalFiles.shift();
    if (next) void dispatchExternalMediaFile(next);
  }
}

async function openAudioViaMenu(): Promise<void> {
  if (quitState !== 'running') return;
  const target = mainWindow && !mainWindow.isDestroyed() ? mainWindow : undefined;
  const options: Electron.OpenDialogOptions = {
    title: 'Choose audio or video files',
    buttonLabel: 'Open',
    properties: ['openFile', 'multiSelections', 'treatPackageAsDirectory'],
    filters: [{ name: 'Audio and video', extensions: [...AUDIO_EXTENSIONS] }],
  };
  const result = target ? await dialog.showOpenDialog(target, options) : await dialog.showOpenDialog(options);
  if (result.canceled || result.filePaths.length === 0) return;
  for (const filePath of result.filePaths) {
    await dispatchExternalMediaFile(filePath);
  }
}

async function exportRedactedDiagnostics(): Promise<void> {
  const target = mainWindow && !mainWindow.isDestroyed() ? mainWindow : undefined;
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const defaultPath = path.join(app.getPath('documents'), `HawaVoClean-Diagnostics-${timestamp}.json`);
  const options: Electron.SaveDialogOptions = {
    title: 'Export Redacted Diagnostics',
    buttonLabel: 'Export',
    defaultPath,
    filters: [
      { name: 'JSON diagnostics', extensions: ['json'] },
      { name: 'Text log', extensions: ['txt'] },
    ],
    properties: ['showOverwriteConfirmation', 'createDirectory'],
  };
  const result = target ? await dialog.showSaveDialog(target, options) : await dialog.showSaveDialog(options);
  if (result.canceled || !result.filePath) return;

  const recentLogLines = broker.readRecentLogLines(200).map((line) => redactDiagnosticText(line));
  const stderrTail = broker.stderrTail.map((line) => redactDiagnosticText(line));
  const payload = {
    schemaVersion: 1,
    exportedAt: new Date().toISOString(),
    redacted: true,
    app: {
      name: app.getName(),
      version: app.getVersion(),
      locale: app.getLocale(),
      platform: process.platform,
      arch: process.arch,
      packaged: app.isPackaged,
    },
    engine: {
      phase: broker.phase,
      restartCount: broker.restartCount,
      recentLogLines,
      stderrTail,
    },
  };
  try {
    fs.writeFileSync(result.filePath, JSON.stringify(payload, null, 2) + '\n', {
      encoding: 'utf8',
      mode: 0o600,
    });
    shell.showItemInFolder(result.filePath);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    broker.log(`[diagnostics] export failed: ${message}`);
    dialog.showErrorBox('Export Diagnostics Failed', message);
  }
}

function buildApplicationMenu(): void {
  const isMac = process.platform === 'darwin';
  const viewSubmenu: MenuItemConstructorOptions[] = [
    { role: 'reload' },
    { role: 'forceReload' },
    ...(DEVTOOLS ? ([{ role: 'toggleDevTools' }] as MenuItemConstructorOptions[]) : []),
    { type: 'separator' },
    { role: 'resetZoom' },
    { role: 'zoomIn' },
    { role: 'zoomOut' },
    { type: 'separator' },
    { role: 'togglefullscreen' },
  ];

  const template: MenuItemConstructorOptions[] = [
    ...(isMac
      ? ([
          {
            label: APP_TITLE,
            submenu: [
              { role: 'about' },
              { type: 'separator' },
              { role: 'services' },
              { type: 'separator' },
              { role: 'hide' },
              { role: 'hideOthers' },
              { role: 'unhide' },
              { type: 'separator' },
              { role: 'quit' },
            ],
          },
        ] as MenuItemConstructorOptions[])
      : []),
    {
      label: 'File',
      submenu: [
        {
          label: 'Open Audio/Video…',
          accelerator: 'CmdOrCtrl+O',
          click: () => {
            void openAudioViaMenu();
          },
        },
        { type: 'separator' },
        isMac ? { role: 'close' } : { role: 'quit' },
      ],
    },
    {
      label: 'Edit',
      submenu: [
        { role: 'undo' },
        { role: 'redo' },
        { type: 'separator' },
        { role: 'cut' },
        { role: 'copy' },
        { role: 'paste' },
        { role: 'selectAll' },
      ],
    },
    {
      label: 'View',
      submenu: viewSubmenu,
    },
    {
      label: 'Help',
      submenu: [
        {
          label: 'Export Redacted Diagnostics…',
          click: () => {
            void exportRedactedDiagnostics();
          },
        },
      ],
    },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

async function configureSession(): Promise<void> {
  const appSession = session.fromPartition(SESSION_PARTITION);
  // Await legacy partition disk cleanup and initial in-memory cache flush before opening any window.
  await safeClearSessionStorage(app.getPath('userData'), {
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

  ipcMain.handle(IPC.pickAudioFiles, async (event): Promise<readonly NativeSourceRegistration[]> => {
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
    const registered: NativeSourceRegistration[] = [];
    for (const filePath of result.filePaths) {
      try {
        const validated = requireSupportedMediaPath(filePath, 'pickAudio');
        const source = await broker.registerNativeSource(validated);
        registered.push({ sourceId: source.sourceId, path: source.path });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        broker.log(`[ipc:pickAudioFiles] native registration failed: ${message}`);
        throw new Error(`Could not register selected media file: ${message}`);
      }
    }
    return registered;
  });

  ipcMain.handle(IPC.registerDroppedFile, async (event, value: unknown): Promise<NativeSourceRegistration> => {
    requireTrustedIpcSender(event);
    try {
      const filePath = requireSupportedMediaPath(value, 'registerDroppedFile');
      const registered = await broker.registerNativeSource(filePath);
      return { sourceId: registered.sourceId, path: registered.path };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      broker.log(`[ipc:registerDroppedFile] native registration failed: ${message}`);
      throw error;
    }
  });

  ipcMain.handle(IPC.chooseExportPath, async (event, value: unknown) => {
    requireTrustedIpcSender(event);
    const request = parseExportRequest(value);
    const defaultPath = path.join(app.getPath('documents'), safeSuggestedExportName(request));
    const filters =
      request.kind === 'master'
        ? MASTER_EXPORT_FILTERS.map((f) => ({ name: f.name, extensions: [...f.extensions] }))
        : RECORD_BUNDLE_EXPORT_FILTERS.map((f) => ({ name: f.name, extensions: [...f.extensions] }));
    const result = await saveDialogFor(event, {
      title: request.kind === 'master' ? 'Save Master WAV' : 'Save Full Processing Record',
      buttonLabel: 'Save',
      defaultPath,
      filters,
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

  ipcMain.handle(IPC.getPreferences, (event) => {
    requireTrustedIpcSender(event);
    return loadUserPreferences(app.getPath('userData'));
  });

  ipcMain.handle(IPC.setPreferences, (event, value: unknown) => {
    requireTrustedIpcSender(event);
    return saveUserPreferences(app.getPath('userData'), value);
  });

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
  const initialBounds = resolveInitialWindowBounds();
  const window = new BrowserWindow({
    width: initialBounds.width,
    height: initialBounds.height,
    ...(initialBounds.x !== undefined ? { x: initialBounds.x } : {}),
    ...(initialBounds.y !== undefined ? { y: initialBounds.y } : {}),
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
  mainWindowLoaded = false;

  let boundsSaveTimer: ReturnType<typeof setTimeout> | null = null;
  const scheduleBoundsSave = (): void => {
    if (SELFTEST) return;
    if (boundsSaveTimer !== null) clearTimeout(boundsSaveTimer);
    boundsSaveTimer = setTimeout(() => {
      boundsSaveTimer = null;
      persistWindowBounds(window);
    }, BOUNDS_SAVE_DEBOUNCE_MS);
    boundsSaveTimer.unref();
  };
  window.on('resize', scheduleBoundsSave);
  window.on('move', scheduleBoundsSave);
  window.on('close', () => {
    if (boundsSaveTimer !== null) {
      clearTimeout(boundsSaveTimer);
      boundsSaveTimer = null;
    }
    persistWindowBounds(window);
  });

  window.once('ready-to-show', () => {
    if (!SELFTEST) window.show();
  });
  window.once('closed', () => {
    if (mainWindow === window) {
      mainWindow = null;
      mainWindowLoaded = false;
    }
  });

  window.on('unresponsive', () => {
    broker.log('[renderer] window became unresponsive');
    if (SELFTEST || quitState !== 'running' || window.isDestroyed()) return;
    void dialog
      .showMessageBox(window, {
        type: 'warning',
        title: 'HawaVoClean is not responding',
        message: 'The interface stopped responding. You can wait for it to recover or reload the window.',
        buttons: ['Wait', 'Reload Window'],
        defaultId: 0,
        cancelId: 0,
      })
      .then(({ response }) => {
        if (response === 1 && !window.isDestroyed()) {
          broker.log('[renderer] user triggered reload after unresponsive prompt');
          mainWindowLoaded = false;
          void window.loadURL(APP_ENTRY_URL);
        }
      })
      .catch(() => undefined);
  });

  window.on('responsive', () => {
    broker.log('[renderer] window became responsive again');
  });

  window.webContents.on('render-process-gone', (_event, details) => {
    broker.log(
      `[renderer] render-process-gone reason=${details.reason} exitCode=${String(details.exitCode)}`,
    );
    mainWindowLoaded = false;
    if (quitState !== 'running' || details.reason === 'clean-exit') return;
    if (SELFTEST) {
      console.error(`HAWA_DESKTOP_SELFTEST_ERROR renderer process gone: ${details.reason}`);
      process.exitCode = 1;
      app.quit();
      return;
    }
    const now = Date.now();
    while (rendererCrashTimestamps.length > 0 && now - rendererCrashTimestamps[0]! > RENDERER_CRASH_WINDOW_MS) {
      rendererCrashTimestamps.shift();
    }
    rendererCrashTimestamps.push(now);
    if (rendererCrashTimestamps.length <= MAX_RENDERER_CRASH_RELOADS) {
      broker.log(
        `[renderer] auto-recovering renderer (${rendererCrashTimestamps.length}/${MAX_RENDERER_CRASH_RELOADS})`,
      );
      if (!window.isDestroyed()) {
        void window.loadURL(APP_ENTRY_URL);
      }
      return;
    }
    if (!window.isDestroyed()) {
      void dialog
        .showMessageBox(window, {
          type: 'error',
          title: 'HawaVoClean Renderer Crashed',
          message: 'The interface crashed repeatedly and could not recover automatically.',
          buttons: ['Reload Window', 'Quit'],
          defaultId: 0,
          cancelId: 1,
        })
        .then(({ response }) => {
          if (response === 0 && !window.isDestroyed()) {
            rendererCrashTimestamps.length = 0;
            void window.loadURL(APP_ENTRY_URL);
          } else {
            app.quit();
          }
        })
        .catch(() => app.quit());
    }
  });

  window.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
  window.webContents.on('will-navigate', (event, url) => {
    if (url !== window.webContents.getURL()) event.preventDefault();
  });
  window.webContents.on('will-attach-webview', (event) => event.preventDefault());
  if (DEVTOOLS) window.webContents.openDevTools({ mode: 'detach' });
  window.webContents.on('did-finish-load', () => {
    mainWindowLoaded = true;
    if (SELFTEST) {
      void runSelfTest(window);
    } else {
      flushPendingExternalFiles();
    }
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

// Register macOS open-file early so files launched at startup are queued reliably.
app.on('open-file', (event, filePath) => {
  event.preventDefault();
  void dispatchExternalMediaFile(filePath);
});

const ownsSingleInstance = app.requestSingleInstanceLock();
if (!ownsSingleInstance) {
  app.quit();
} else {
  app.on('second-instance', (_event, argv) => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.show();
      mainWindow.focus();
    }
    for (const mediaPath of extractExternalMediaPaths(argv.slice(1))) {
      void dispatchExternalMediaFile(mediaPath);
    }
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
  app.whenReady().then(async () => {
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
    await configureSession();
    buildApplicationMenu();
    registerIpc();
    broker.start();
    if (!SELFTEST) {
      for (const mediaPath of extractExternalMediaPaths(process.argv.slice(1))) {
        if (!pendingExternalFiles.includes(mediaPath)) {
          pendingExternalFiles.push(mediaPath);
        }
      }
    }
    createWindow();
  }).catch((error: unknown) => {
    broker.log(`[startup] fatal initialization error: ${error instanceof Error ? error.message : String(error)}`);
    dialog.showErrorBox('HawaVoClean failed to start', error instanceof Error ? error.message : String(error));
    app.exit(1);
  });

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0 && quitState === 'running') createWindow();
  });
}
