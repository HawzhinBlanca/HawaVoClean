'use strict';
/*
 * HawaVoClean preload — runs in Resolve's sandboxed, context-isolated renderer.
 *
 * Exposes `window.hawa` exactly as `HawaBridge` in docs/ui-contract.md section 3 (authoritative
 * TypeScript copy: ui/src/bridge/types.ts). Only contextBridge + ipcRenderer.invoke are used; the
 * single synchronous call (`hawa:host`) is the one sanctioned exception so that `host` can be a
 * plain value at script-evaluation time.
 */

const { contextBridge, ipcRenderer, webUtils } = require('electron');

// Electron wraps handler errors as "Error invoking remote method 'chan': Error: <msg>".
// The contract wants plain `Error(message)`, so unwrap before rethrowing.
function cleanMessage(err) {
  const raw = err && typeof err.message === 'string' ? err.message : String(err);
  return raw.replace(/^Error invoking remote method '[^']*':\s*(?:[A-Za-z]*Error:\s*)?/, '');
}

async function invoke(channel, ...args) {
  try {
    return await ipcRenderer.invoke(channel, ...args);
  } catch (err) {
    throw new Error(cleanMessage(err));
  }
}

let host = 'electron';
try {
  const h = ipcRenderer.sendSync('hawa:host');
  if (h === 'resolve' || h === 'electron') host = h;
} catch {
  host = 'electron';
}

const bridge = {
  host,
  engine: {
    getEndpoint: () => invoke('hawa:engine:endpoint'),
  },
  files: {
    pickAudio: () => invoke('hawa:files:pick'),
    pathForFile: (file) => {
      try {
        const p = webUtils.getPathForFile(file);
        return typeof p === 'string' && p.length > 0 ? p : null;
      } catch {
        return null;
      }
    },
    revealInFinder: (filePath) => invoke('hawa:files:reveal', filePath),
  },
};

if (host === 'resolve') {
  bridge.resolve = {
    getSelectedClip: () => invoke('hawa:resolve:selected'),
    importMedia: (filePath) => invoke('hawa:resolve:import', filePath),
    replaceClip: (mediaId, filePath) => invoke('hawa:resolve:replace', mediaId, filePath),
    appendToTimeline: (mediaId) => invoke('hawa:resolve:append', mediaId),
    getContext: () => invoke('hawa:resolve:context'),
  };
}

contextBridge.exposeInMainWorld('hawa', bridge);
