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

/** A transfer in flight from `POST /api/upload` (goal box B2). */
export interface UploadInfo {
  name: string;
  /** Bytes on the wire so far / total. `total` is the file size. */
  loaded: number;
  total: number;
  /** `sending` while bytes move, `finishing` once the engine has them all. */
  phase: 'sending' | 'finishing';
}

/**
 * Something was dropped or picked that the tool cannot open. It is a designed
 * state, not a log line: the strip says what arrived and what it accepts.
 */
export interface DropRejection {
  /** `multi` is not a refusal — it says which of several files was taken. */
  kind: 'type' | 'folder' | 'empty' | 'multi';
  /** What was dropped, as the user named it. */
  name: string;
  /** The extra explanation for that kind (mime type, folder wording, …). */
  detail: string;
}

/**
 * One finished run, kept for the session (goal box B5). The analyses ride
 * along so re-selecting a run restores its decks and its meters without
 * asking the engine to decode anything again.
 */
export interface HistoryEntry {
  jobId: string;
  profile: Profile;
  inputPath: string;
  inputName: string;
  outcome: 'done' | 'failed' | 'cancelled';
  /** Wall-clock ms the job took, from the engine's own timestamps. */
  durationMs: number | null;
  /** Epoch ms the run reached its terminal state. */
  at: number;
  enhanced: number | null;
  unitsTotal: number | null;
  lufsIn: number | null;
  lufsOut: number | null;
  noiseIn: number | null;
  noiseOut: number | null;
  outputPath: string;
  reportPath: string;
  report: HawaVoCleanReport | null;
  /** The engine's own terminal snapshot, replayed verbatim on re-selection. */
  status: JobStatus | null;
  original: AudioAnalysis | null;
  cleaned: AudioAnalysis | null;
  /** Why the run ended badly, when it did. */
  error: string | null;
}

/** How many runs the session keeps. */
export const HISTORY_LIMIT = 8;

export interface AppState {
  host: HawaHost;
  engineStatus: EngineStatus;
  engineVersion: string | null;
  client: EngineClient | null;

  /**
   * B6 · when the engine was first seen to be gone, or null while it answers.
   * The offline banner keys on this rather than on `engineStatus` alone, so a
   * probe that is momentarily back in `connecting` does not make the banner
   * blink out and back in every few hundred milliseconds.
   */
  engineOfflineSince: number | null;
  /** Epoch ms of the next scheduled health probe, for the banner's countdown. */
  engineNextProbeAt: number | null;
  /** True only while a probe is actually on the wire. */
  engineProbing: boolean;

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

  /** In-flight upload (web mode), or null. */
  upload: UploadInfo | null;
  /** The last thing the drop well refused, until it is dismissed or replaced. */
  rejection: DropRejection | null;

  /** Finished runs this session, newest first (goal box B5). */
  history: HistoryEntry[];
  /** Which history entry the screen is currently showing. */
  currentRunId: string | null;

  statusLine: string;
  error: string | null;
  dragOver: boolean;

  // actions
  setHost(host: HawaHost): void;
  setEngine(status: EngineStatus, client: EngineClient | null, version: string | null): void;
  setEngineProbe(nextAt: number | null, probing: boolean): void;
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
  setUpload(u: UploadInfo | null): void;
  setRejection(r: DropRejection | null): void;
  pushHistory(entry: HistoryEntry): void;
  patchHistory(jobId: string, patch: Partial<HistoryEntry>): void;
  setCurrentRun(jobId: string | null): void;
  resetForNewSource(): void;
}

export const useStore = create<AppState>((set) => ({
  host: 'web',
  engineStatus: 'connecting',
  engineVersion: null,
  client: null,

  engineOfflineSince: null,
  engineNextProbeAt: null,
  engineProbing: false,

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

  upload: null,
  rejection: null,

  history: [],
  currentRunId: null,

  statusLine: 'Connecting to engine',
  error: null,
  dragOver: false,

  setHost: (host) => set({ host }),
  // Losing the engine must not lose anything else: the client object is a URL
  // and a token, not a connection, so it is kept across an outage and every
  // artefact URL, deck source and report path on screen stays valid for the
  // moment it comes back (goal box B6).
  setEngine: (engineStatus, client, engineVersion) =>
    set((s) => ({
      engineStatus,
      client,
      engineVersion,
      engineOfflineSince:
        engineStatus === 'ready' ? null : (s.engineOfflineSince ?? (engineStatus === 'offline' ? Date.now() : null)),
      ...(engineStatus === 'ready' ? { engineNextProbeAt: null, engineProbing: false } : {}),
    })),
  setEngineProbe: (engineNextProbeAt, engineProbing) => set({ engineNextProbeAt, engineProbing }),
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
  setUpload: (upload) => set({ upload }),
  // A new refusal replaces the old one; nothing queues up.
  setRejection: (rejection) => set({ rejection }),
  pushHistory: (entry) =>
    set((s) => ({
      history: [entry, ...s.history.filter((h) => h.jobId !== entry.jobId)].slice(0, HISTORY_LIMIT),
      currentRunId: entry.jobId,
    })),
  patchHistory: (jobId, patch) =>
    set((s) => ({
      history: s.history.map((h) => (h.jobId === jobId ? { ...h, ...patch } : h)),
    })),
  setCurrentRun: (currentRunId) => set({ currentRunId }),
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
      // The run list survives a new clip — it is the session's memory — but
      // nothing in it is on screen any more.
      currentRunId: null,
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
