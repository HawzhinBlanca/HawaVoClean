import { create } from 'zustand';
import type { EngineClient } from '../api/client';
import type {
  AudioAnalysis,
  HawaVoCleanReport,
  JobStatus,
  Profile,
  UnitDecisionRecord,
} from '../api/types';
import { waveView, type ViewWindow } from '../render/viewWindow';
import type { HawaHost } from '../bridge/types';

export type EngineStatus = 'connecting' | 'ready' | 'offline';
export type SourceOrigin = 'resolve' | 'file' | 'drop' | 'upload';
export type AbMode = 'original' | 'cleaned';

export interface SourceInfo {
  path: string;
  name: string;
  origin: SourceOrigin;
  mediaId?: string;
}

export interface JobInfo {
  id: string;
  outputPath: string;
  reportPath: string;
  status: JobStatus | null;
  streamConnected: boolean;
}

export interface HoverUnit {
  unit: UnitDecisionRecord;
  x: number;
  y: number;
}

export interface AppState {
  host: HawaHost;
  engineStatus: EngineStatus;
  engineVersion: string | null;
  client: EngineClient | null;

  source: SourceInfo | null;
  analyzing: boolean;
  original: AudioAnalysis | null;
  cleaned: AudioAnalysis | null;

  profile: Profile;
  job: JobInfo | null;
  report: HawaVoCleanReport | null;
  cleanedPath: string | null;

  abMode: AbMode;
  playing: boolean;
  currentTime: number;
  duration: number;

  /**
   * Visible waveform window in seconds. Mirror of the imperative
   * `waveView` controller (render/viewWindow.ts), refreshed on a trailing
   * timer — zoom/pan themselves never re-render React.
   */
  view: ViewWindow;

  hoverUnit: HoverUnit | null;
  highlightRange: { start: number; end: number } | null;

  /**
   * The unit picked in the verdict strip. Selection is a property of the
   * report, not of the deck, so it survives A/B switching and playback; only
   * a new report or a new source clears it.
   */
  selectedUnit: UnitDecisionRecord | null;
  /** Keyboard-map overlay (`?`). */
  shortcutsOpen: boolean;

  statusLine: string;
  error: string | null;
  dragOver: boolean;

  // actions
  setHost(host: HawaHost): void;
  setEngine(status: EngineStatus, client: EngineClient | null, version: string | null): void;
  setSource(source: SourceInfo | null): void;
  setAnalyzing(v: boolean): void;
  setOriginal(a: AudioAnalysis | null): void;
  setCleaned(a: AudioAnalysis | null, path: string | null): void;
  setProfile(p: Profile): void;
  setJob(job: JobInfo | null): void;
  patchJob(patch: Partial<JobInfo>): void;
  setReport(r: HawaVoCleanReport | null): void;
  setAbMode(m: AbMode): void;
  setPlaying(v: boolean): void;
  setTime(t: number): void;
  setDuration(d: number): void;
  setView(start: number, end: number): void;
  resetView(): void;
  setHoverUnit(h: HoverUnit | null): void;
  setHighlight(r: { start: number; end: number } | null): void;
  setSelectedUnit(u: UnitDecisionRecord | null): void;
  setShortcutsOpen(v: boolean): void;
  setStatus(line: string): void;
  setError(msg: string | null): void;
  setDragOver(v: boolean): void;
  resetForNewSource(): void;
}

export const useStore = create<AppState>((set) => ({
  host: 'web',
  engineStatus: 'connecting',
  engineVersion: null,
  client: null,

  source: null,
  analyzing: false,
  original: null,
  cleaned: null,

  profile: 'studio',
  job: null,
  report: null,
  cleanedPath: null,

  abMode: 'original',
  playing: false,
  currentTime: 0,
  duration: 0,

  view: { start: 0, end: 0 },

  hoverUnit: null,
  highlightRange: null,

  selectedUnit: null,
  shortcutsOpen: false,

  statusLine: 'Connecting to engine',
  error: null,
  dragOver: false,

  setHost: (host) => set({ host }),
  setEngine: (engineStatus, client, engineVersion) => set({ engineStatus, client, engineVersion }),
  setSource: (source) => set({ source }),
  setAnalyzing: (analyzing) => set({ analyzing }),
  setOriginal: (original) =>
    set({ original, duration: original ? original.duration_s : 0 }),
  setCleaned: (cleaned, cleanedPath) => set({ cleaned, cleanedPath }),
  setProfile: (profile) => set({ profile }),
  setJob: (job) => set({ job }),
  patchJob: (patch) =>
    set((s) => (s.job ? { job: { ...s.job, ...patch } } : {})),
  // A new report invalidates any selection made against the previous one.
  setReport: (report) => set({ report, selectedUnit: null, highlightRange: null }),
  setAbMode: (abMode) => set({ abMode }),
  setPlaying: (playing) => set({ playing }),
  setTime: (currentTime) => set({ currentTime }),
  setDuration: (duration) => set({ duration }),
  setView: (start, end) => {
    waveView.set(start, end);
    set({ view: waveView.view });
  },
  resetView: () => {
    waveView.reset();
    set({ view: waveView.view });
  },
  setHoverUnit: (hoverUnit) => set({ hoverUnit }),
  setHighlight: (highlightRange) => set({ highlightRange }),
  // Selecting a unit is what lights its range in the waveform; hovering only
  // borrows the highlight and hands it back on leave.
  setSelectedUnit: (selectedUnit) =>
    set({
      selectedUnit,
      highlightRange: selectedUnit
        ? { start: selectedUnit.start_time_s, end: selectedUnit.end_time_s }
        : null,
    }),
  setShortcutsOpen: (shortcutsOpen) => set({ shortcutsOpen }),
  setStatus: (statusLine) => set({ statusLine }),
  setError: (error) => set({ error }),
  setDragOver: (dragOver) => set({ dragOver }),
  resetForNewSource: () =>
    set({
      original: null,
      cleaned: null,
      cleanedPath: null,
      job: null,
      report: null,
      abMode: 'original',
      playing: false,
      currentTime: 0,
      duration: 0,
      hoverUnit: null,
      highlightRange: null,
      selectedUnit: null,
      error: null,
      view: { start: 0, end: 0 },
    }),
}));

// Keep the store's `view` in step with the imperative controller without
// re-rendering on every wheel event: one trailing update per burst.
const VIEW_MIRROR_MS = 120;
let viewMirrorTimer: number | null = null;
waveView.subscribe(() => {
  if (viewMirrorTimer !== null) return;
  viewMirrorTimer = setTimeout(() => {
    viewMirrorTimer = null;
    const cur = useStore.getState().view;
    const next = waveView.view;
    if (cur.start !== next.start || cur.end !== next.end) useStore.setState({ view: next });
  }, VIEW_MIRROR_MS) as unknown as number;
});

/** Transient read for render loops (no subscription). */
export const getState = (): AppState => useStore.getState();
