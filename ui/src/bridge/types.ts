// Authoritative copy of the renderer bridge shape (docs/ui-contract.md §3).
// The Electron preload exposes this as `window.hawa`; in a plain browser it is
// undefined and the UI builds a `web` fallback (see ./index.ts).

export type HawaHost = 'resolve' | 'electron' | 'web';
export interface ResolveClip { mediaId: string; name: string; filePath: string; durationS?: number; }
export interface HawaBridge {
  host: HawaHost;
  engine: { getEndpoint(): Promise<{ baseUrl: string }> };
  files: {
    pickAudio(): Promise<string | null>;            // native open dialog → absolute path
    registerDroppedFile?(file: File): Promise<{ sourceId: string; path: string } | string | null>; // main registers before returning
    pathForFile(file: File): string | null;         // legacy shape; hardened shells return null
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

declare global {
  interface Window {
    hawa?: HawaBridge;
  }
}
