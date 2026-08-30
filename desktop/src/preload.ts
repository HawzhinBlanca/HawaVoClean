import { contextBridge, ipcRenderer, webUtils } from 'electron';

// A sandboxed preload may load Electron's safe built-ins but cannot require
// sibling application modules. Keep this file self-contained; the contract
// test pins these literals against the authoritative IPC object.
const IPC = Object.freeze({
  engineEndpoint: 'hawa:engine:endpoint',
  pickAudioFiles: 'hawa:files:pick-audio',
  registerDroppedFile: 'hawa:files:register-dropped',
  pickFolder: 'hawa:files:pick-folder',
  chooseExportPath: 'hawa:files:choose-export',
  reveal: 'hawa:files:reveal',
  appInfo: 'hawa:app:info',
  diagnosticsState: 'hawa:diagnostics:state',
  updateState: 'hawa:updates:state',
});
type ExportRequest = Readonly<{
  kind: 'master' | 'record_bundle';
  suggestedName?: string;
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
      const paths = await invoke<readonly string[]>(IPC.pickAudioFiles);
      return paths[0] ?? null;
    },
    pickAudioFiles: () => invoke<readonly string[]>(IPC.pickAudioFiles),
    pickFolder: () => invoke<string | null>(IPC.pickFolder),
    chooseExportPath: (request: ExportRequest) => invoke<string | null>(IPC.chooseExportPath, request),
    async registerDroppedFile(file: File): Promise<string | null> {
      try {
        const value = webUtils.getPathForFile(file);
        if (typeof value !== 'string' || value.length === 0) return null;
        return await invoke<string>(IPC.registerDroppedFile, value);
      } catch {
        return null;
      }
    },
    // One-release shape compatibility. Unregistered native paths never cross
    // this bridge; callers use registerDroppedFile or upload the bytes.
    pathForFile(): null {
      return null;
    },
    revealInFolder: (filePath: string) => invoke<void>(IPC.reveal, filePath),
    // Compatibility with the existing UI name on both macOS and Windows.
    revealInFinder: (filePath: string) => invoke<void>(IPC.reveal, filePath),
  }),
  app: Object.freeze({
    getInfo: () => invoke(IPC.appInfo),
  }),
  diagnostics: Object.freeze({
    getState: () => invoke(IPC.diagnosticsState),
  }),
  updates: Object.freeze({
    getState: () => invoke(IPC.updateState),
  }),
});

contextBridge.exposeInMainWorld('hawa', bridge);
