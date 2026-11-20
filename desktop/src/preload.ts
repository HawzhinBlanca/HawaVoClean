import { contextBridge, ipcRenderer, webUtils } from 'electron';

// A sandboxed preload may load Electron's safe built-ins but cannot require
// sibling application modules. Keep this file self-contained; the contract
// test pins these literals against the authoritative IPC object.
const IPC = Object.freeze({
  engineEndpoint: 'hawa:engine:endpoint',
  pickAudioFiles: 'hawa:files:pick-audio',
  registerDroppedFile: 'hawa:files:register-dropped',
  chooseExportPath: 'hawa:files:choose-export',
  reveal: 'hawa:files:reveal',
  appInfo: 'hawa:app:info',
  getPreferences: 'hawa:app:get-preferences',
  setPreferences: 'hawa:app:set-preferences',
  clearLocalData: 'hawa:session:clear-local-data',
});

const IPC_EVENTS = Object.freeze({
  openFile: 'hawa:files:opened',
});

type RegisteredNativeSource = Readonly<{
  sourceId: string;
  path: string;
}>;

type ExportRequest = Readonly<{
  kind: 'master' | 'record_bundle';
  suggestedName?: string;
  sourcePath?: string;
  sourceId?: string;
}>;

type UserPreferencesPatch = Readonly<{
  workspaceMode?: 'simple' | 'pro';
  locale?: string;
  livePreviewEnabled?: boolean;
}>;

function cleanMessage(error: unknown): string {
  const raw = error instanceof Error ? error.message : String(error);
  return raw.replace(/^Error invoking remote method '[^']*':\s*(?:[A-Za-z]*Error:\s*)?/, '');
}

async function invoke<T>(channel: string, ...args: readonly unknown[]): Promise<T> {
  try {
    return (await ipcRenderer.invoke(channel, ...args)) as T;
  } catch (error) {
    throw new Error(cleanMessage(error));
  }
}

const bridge = Object.freeze({
  host: 'electron' as const,
  engine: Object.freeze({
    getEndpoint: () => invoke(IPC.engineEndpoint),
  }),
  files: Object.freeze({
    // Compatibility with the existing renderer contract.
    async pickAudio(): Promise<string | null> {
      const items = await invoke<readonly RegisteredNativeSource[]>(IPC.pickAudioFiles);
      return items[0]?.path ?? null;
    },
    async pickAudioFiles(): Promise<readonly string[]> {
      const items = await invoke<readonly RegisteredNativeSource[]>(IPC.pickAudioFiles);
      return items.map((item) => item.path);
    },
    pickAudioSources: () => invoke<readonly RegisteredNativeSource[]>(IPC.pickAudioFiles),
    chooseExportPath: (request: ExportRequest) => invoke<string | null>(IPC.chooseExportPath, request),
    async registerDroppedFile(file: File): Promise<RegisteredNativeSource | null> {
      let value: string;
      try {
        value = webUtils.getPathForFile(file);
      } catch (error) {
        console.warn('[HawaVoClean] registerDroppedFile: could not read native file path:', cleanMessage(error));
        return null;
      }
      if (typeof value !== 'string' || value.length === 0) {
        console.warn('[HawaVoClean] registerDroppedFile: no native file path available for dropped item.');
        return null;
      }
      try {
        return await invoke<RegisteredNativeSource>(IPC.registerDroppedFile, value);
      } catch (error) {
        console.error('[HawaVoClean] registerDroppedFile failed:', cleanMessage(error));
        throw error;
      }
    },
    // One-release shape compatibility. Unregistered native paths never cross
    // this bridge; callers use registerDroppedFile or upload the bytes.
    pathForFile(): null {
      return null;
    },
    // Compatibility with the existing UI name on both macOS and Windows.
    revealInFinder: (filePath: string) => invoke<void>(IPC.reveal, filePath),
    onOpenFile(listener: (source: RegisteredNativeSource) => void): () => void {
      if (typeof listener !== 'function') {
        throw new TypeError('onOpenFile listener must be a function.');
      }
      const wrapped = (_event: Electron.IpcRendererEvent, payload: unknown): void => {
        if (
          typeof payload === 'object' &&
          payload !== null &&
          typeof (payload as Record<string, unknown>).sourceId === 'string' &&
          typeof (payload as Record<string, unknown>).path === 'string'
        ) {
          listener(payload as RegisteredNativeSource);
        }
      };
      ipcRenderer.on(IPC_EVENTS.openFile, wrapped);
      return () => {
        ipcRenderer.removeListener(IPC_EVENTS.openFile, wrapped);
      };
    },
  }),
  app: Object.freeze({
    getInfo: () => invoke(IPC.appInfo),
    getPreferences: () => invoke(IPC.getPreferences),
    setPreferences: (patch: UserPreferencesPatch) => invoke(IPC.setPreferences, patch),
  }),
  session: Object.freeze({
    clearLocalData: () => invoke(IPC.clearLocalData),
  }),
});

contextBridge.exposeInMainWorld('hawa', bridge);
