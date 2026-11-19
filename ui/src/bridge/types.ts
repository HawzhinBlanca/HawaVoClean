// Authoritative copy of the renderer bridge shape (docs/ui-contract.md §3).
// The Electron preload exposes this as `window.hawa`; in a plain browser it is
// undefined and the UI builds a `web` fallback (see ./index.ts).

export type HawaHost = 'resolve' | 'electron' | 'web';
export interface ResolveClip { mediaId: string; name: string; filePath: string; durationS?: number; }

export interface ResolveBatchTrackClip {
  clipId?: string;
  mediaId?: string;
  id?: string;
  startFrame?: number;
  recordFrame?: number;
  sourceStartFrame?: number;
  sourceEndFrame?: number;
  durationFrames?: number;
  sourceOffsetS?: number;
  mediaType?: number;
}

export interface ResolveBatchCleanEntry {
  filePath: string;
  report?: unknown;
  reportPath?: string;
}

export interface ResolveBatchCleanTrackPayload {
  clips: ResolveBatchTrackClip[];
  cleanedByClipId: Record<string, string | ResolveBatchCleanEntry>;
  targetTrackName?: string;
  fps?: number;
  addGuardMarkers?: boolean;
}

export interface ResolveBatchCleanTrackResult {
  success: boolean;
  txId?: string;
  journalPath?: string;
  trackIndex: number;
  trackName: string;
  reusedTrack?: boolean;
  clipsPlaced: number;
  markersAdded: number;
  markers?: Array<Record<string, unknown>>;
}

export interface ResolveGuardMarker {
  frameOffset: number;
  color: string;
  name: string;
  note?: string;
  durationFrames?: number;
}

export interface ResolveApplyGuardMarkersPayload {
  report?: unknown;
  reportPath?: string;
  clipStartFrame?: number;
  fps?: number;
  sourceOffsetS?: number;
  markers?: ResolveGuardMarker[];
}

export interface ResolveApplyGuardMarkersResult {
  success: boolean;
  markersAdded: number;
  markers: Array<Record<string, unknown>>;
}

export interface ResolveBridge {
  getSelectedClip(): Promise<ResolveClip | null>;  // MediaPool.GetSelectedClips()[0], else current video item's MediaPoolItem
  importMedia(path: string): Promise<ResolveClip | null>;        // MediaPool.ImportMedia([path])[0]
  replaceClip(mediaId: string, path: string): Promise<boolean>;  // ReplaceClipPreserveSubClip, fallback ReplaceClip
  appendToTimeline(mediaId: string): Promise<boolean>;
  newTrack?(mediaId: string, path: string, options?: { trackName?: string }): Promise<{ success: boolean; trackIndex: number; trackName: string }>;
  batchCleanTrack?(payload: ResolveBatchCleanTrackPayload): Promise<ResolveBatchCleanTrackResult>;
  applyGuardMarkers?(payload: ResolveApplyGuardMarkersPayload): Promise<ResolveApplyGuardMarkersResult>;
  getContext(): Promise<{ project: string | null; timeline: string | null; page: string | null }>;
}

export interface HawaBridge {
  host: HawaHost;
  engine: { getEndpoint(): Promise<{ baseUrl: string }> };
  files: {
    pickAudio(): Promise<string | null>;            // native open dialog → absolute path
    pickAudioFiles?(): Promise<readonly string[] | string[] | null>;
    registerDroppedFile?(file: File): Promise<{ sourceId: string; path: string } | string | null>; // main registers before returning
    pathForFile(file: File): string | null;         // legacy shape; hardened shells return null
    chooseExportPath?(request: { kind: 'master' | 'record_bundle'; suggestedName?: string; sourcePath?: string }): Promise<string | null>;
    revealInFinder(path: string): Promise<void>;
  };
  resolve?: ResolveBridge;                          // present only when host === 'resolve'
}

declare global {
  interface Window {
    hawa?: HawaBridge;
  }
}
