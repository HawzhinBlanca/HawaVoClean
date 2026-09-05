import { create } from 'zustand';
import type { EngineClient } from '../api/client';
import type {
  AudioAnalysis,
  BatchSummary,
  CapabilityStatusV1,
  HawaVoCleanReport,
  JobMode,
  JobStatus,
  Profile,
  UnitDecisionRecord,
} from '../api/types';
import { waveView, type ViewWindow } from '../render/viewWindow';
import type { HawaHost } from '../bridge/types';

export type EngineStatus = 'connecting' | 'ready' | 'offline';
export type SourceOrigin = 'resolve' | 'file' | 'drop' | 'upload';
export type AbMode = 'original' | 'cleaned';

/**
 * What the engine can still serve of the run that is on screen.
 *
 * A finished run leaves three files behind, and nothing in this app owns them:
 * they can be moved, deleted or — before this iteration — written over by a
 * second run of the same profile. A cached report is therefore not evidence
 * that the files are there, so a restored run is checked (three HEADs, no
 * decode) and the answer is kept here. `null` means "not checked", which is
 * the honest state for a run whose files were written a second ago.
 */
export interface ArtifactState {
  /**
   * Can the cleaned master still be handed over *as this run's master*? False
   * covers both halves of that question: the file is not there, and the file
   * is there but is no longer the audio the report describes (truncated, or
   * bytes nothing can decode). `reason` says which.
   */
  master: boolean;
  json: boolean;
  txt: boolean;
  /** The one sentence the disabled links and the run row explain themselves with. */
  reason: string;
  /**
   * The two-word flag the run list stamps on the row. `FILE GONE` is the
   * default and was for a while the only one, which made a master that was
   * still on disk but unplayable read as deleted.
   */
  flag?: string;
}

/**
 * A deck that was asked for and could not be played, in the words the screen
 * uses. Written from the player's own fault event, so the A/B control and the
 * player can never disagree about which deck is live.
 */
export interface DeckFaultInfo {
  deck: AbMode;
  /** Small-caps label, e.g. `CLEANED DECK UNAVAILABLE`. */
  headline: string;
  /** The sentence: what failed, why, and what you are hearing instead. */
  detail: string;
}

export interface SourceInfo {
  path: string;
  name: string;
  origin: SourceOrigin;
  mediaId?: string | undefined;
  sourceId?: string | undefined;
}

export interface JobInfo {
  id: string;
  outputPath: string;
  reportPath: string;
  bundlePath?: string;
  status: JobStatus | null;
  streamConnected: boolean;
}

/**
 * Is this job still the engine's problem?
 *
 * The subtlety is `status: null`, which is the window between `createJob`
 * returning an id and the first status arriving over the stream. Written out
 * by hand as `job && job.status && !isTerminal(job.status.state)` — which is
 * how four call sites had it — that window reads as *idle*, so during it a
 * dropped file, a new load or another PROCESS press sailed past the guard and
 * silently orphaned a run the engine was already working on.
 *
 * A job with no status yet is in flight. That is what having an id means.
 */
export function jobInFlight(job: JobInfo | null | undefined): boolean {
  if (!job) return false;
  if (!job.status) return true;
  const s = job.status.state;
  return s !== 'done' && s !== 'failed' && s !== 'cancelled';
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
  /**
   * The job id of a later run that wrote to this run's output path. Two runs
   * cannot own the same file: once this is set, the files beside this row are
   * somebody else's and the row says so instead of handing them over.
   */
  supersededBy?: string | null;
  /** What the engine could still serve the last time this run was opened. */
  artifacts?: ArtifactState | null;
  /**
   * The master's byte length as the engine served it when this run's master
   * was first loaded. This is the yardstick a later re-verification measures
   * against: a 100-byte truncation answers HEAD 200 and delivers its one
   * probed byte perfectly happily — only the length says it is not the file
   * this run wrote. Null until the first healthy sighting.
   */
  masterBytes?: number | null;
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
  /**
   * Restore capability, from `GET /api/health` (contract addendum 2). The
   * engine recomputes both per probe, so they can change while the app is
   * open — a profile trained mid-session appears, a deleted one stops being
   * offered — and `setCapabilities` reconciles the selection below with them.
   */
  speakers: string[];
  restoreAvailable: boolean;
  /**
   * Versioned runtime capabilities from `GET /api/v1/capabilities` (True-10 D4.11).
   * Governs truthful route qualification, blocked reasons, and runtime providers.
   */
  capabilities: CapabilityStatusV1[] | null;
  /**
   * Explicit user consent required for generative reconstruction (HawaRestore-KD).
   */
  reconstructionConsent: boolean;
  /**
   * The next run's mode and its restore parameters. Like `profile` these are
   * a control setting, not a property of any one run: they survive a new
   * source and are read once at submit time. Invariant the control and
   * `setCapabilities` maintain together: `speakerId` is non-null whenever
   * `speakers` is non-empty, so a restore-mode submit always has the
   * `speaker_id` the engine requires. `cutoffHz` null means auto-detect.
   */
  mode: JobMode;
  speakerId: string | null;
  cutoffHz: number | null;
  job: JobInfo | null;
  report: HawaVoCleanReport | null;
  cleanedPath: string | null;

  abMode: AbMode;
  /** Whether loudness-matched A/B comparison is active (D4.2). */
  loudnessMatch: boolean;
  /** Current gain offset in dB applied to Cleaned deck during level-matched A/B. */
  gainOffsetDb: number;
  /** Whether the advanced controls drawer/disclosure is expanded (D4.2). */
  advancedOpen: boolean;
  /**
   * A deck that could not be played, and what is playing instead. Set from the
   * player's fault event, cleared when a deck is asked for again.
   */
  deckFault: DeckFaultInfo | null;
  /**
   * Which of the on-screen run's three files can still be served, or null when
   * nothing has been checked.
   */
  artifacts: ArtifactState | null;
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
  /**
   * The lit range in the waveform. `channel` is carried only when the report
   * decided on more than one — a split-speakers run's per-channel units
   * overlap in time, so the band has to say whose seconds it is lighting.
   * `state/selection.ts` is what fills it in; the store never guesses.
   */
  highlightRange: { start: number; end: number; channel?: number } | null;

  /**
   * The unit picked in the verdict strip. Selection is a property of the
   * report, not of the deck, so it survives A/B switching and playback; only
   * a new report or a new source clears it.
   */
  selectedUnit: UnitDecisionRecord | null;

  /**
   * D2 · Click-drag selection range in seconds. `null` = no active selection.
   * Set by click-drag on the waveform; cleared by Escape or a new source.
   * `I` sets in-point, `O` sets out-point (NLE convention).
   */
  selectionRange: { start: number; end: number } | null;

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
  /**
   * What the error bar's own label should say — i.e. *where* this failure came
   * from. The bar hardcoded `Engine error`, which was right for most of what
   * lands here and wrong for the rest: a clipboard permission the browser
   * refused was published to the user as ENGINE ERROR. Every caller of
   * `setError` names its source; nothing else infers one.
   */
  errorLabel: string | null;
  dragOver: boolean;
  batch: BatchSummary | null;
  activeInspectJobId: string | null;

  // actions
  setHost(host: HawaHost): void;
  setEngine(status: EngineStatus, client: EngineClient | null, version: string | null): void;
  setEngineProbe(nextAt: number | null, probing: boolean): void;
  setSource(source: SourceInfo | null): void;
  setAnalyzing(v: boolean): void;
  setOriginal(a: AudioAnalysis | null): void;
  setCleaned(a: AudioAnalysis | null, path: string | null): void;
  setProfile(p: Profile): void;
  setCapabilities(speakers: string[], restoreAvailable: boolean): void;
  setCapabilitiesV1(capabilities: CapabilityStatusV1[]): void;
  setReconstructionConsent(consent: boolean): void;
  setMode(m: JobMode): void;
  setSpeakerId(id: string | null): void;
  setCutoffHz(hz: number | null): void;
  setJob(job: JobInfo | null): void;
  patchJob(patch: Partial<JobInfo>): void;
  setReport(r: HawaVoCleanReport | null): void;
  setAbMode(m: AbMode): void;
  setLoudnessMatch(v: boolean): void;
  setGainOffsetDb(v: number): void;
  setAdvancedOpen(v: boolean): void;
  setDeckFault(f: DeckFaultInfo | null): void;
  setArtifacts(a: ArtifactState | null): void;
  setPlaying(v: boolean): void;
  setTime(t: number): void;
  setDuration(d: number): void;
  setView(start: number, end: number): void;
  resetView(): void;
  setHoverUnit(h: HoverUnit | null): void;
  setHighlight(r: { start: number; end: number; channel?: number } | null): void;
  setSelectedUnit(u: UnitDecisionRecord | null): void;
  setSelectionRange(r: { start: number; end: number } | null): void;
  setShortcutsOpen(v: boolean): void;
  setStatus(line: string): void;
  setError(msg: string | null, label?: string): void;
  setDragOver(v: boolean): void;
  setUpload(u: UploadInfo | null): void;
  setRejection(r: DropRejection | null): void;
  pushHistory(entry: HistoryEntry): void;
  patchHistory(jobId: string, patch: Partial<HistoryEntry>): void;
  setCurrentRun(jobId: string | null): void;
  setBatch(b: BatchSummary | null): void;
  patchBatch(patch: Partial<BatchSummary>): void;
  setActiveInspectJobId(id: string | null): void;
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

  profile: 'production',
  speakers: [],
  restoreAvailable: false,
  capabilities: null,
  reconstructionConsent: false,
  mode: 'natural',
  speakerId: null,
  cutoffHz: null,
  job: null,
  report: null,
  cleanedPath: null,

  abMode: 'original',
  loudnessMatch: true,
  gainOffsetDb: 0,
  advancedOpen: false,
  deckFault: null,
  artifacts: null,
  playing: false,
  currentTime: 0,
  duration: 0,

  view: { start: 0, end: 0 },

  hoverUnit: null,
  highlightRange: null,

  selectedUnit: null,
  selectionRange: null,
  shortcutsOpen: false,

  upload: null,
  rejection: null,

  history: [],
  currentRunId: null,

  statusLine: 'Connecting to engine',
  error: null,
  errorLabel: null,
  dragOver: false,
  batch: null,
  activeInspectJobId: null,

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
  // Capabilities can change under a made selection, so this reconciles rather
  // than just stores: a speaker the engine no longer offers cannot stay
  // selected (the submit it fed would 422), and when restore goes away
  // entirely the mode falls back to natural — silently, because the control
  // that would have explained it is hidden at the same moment.
  setCapabilities: (speakers, restoreAvailable) =>
    set((s) => ({
      speakers,
      restoreAvailable,
      speakerId:
        s.speakerId !== null && speakers.includes(s.speakerId)
          ? s.speakerId
          : (speakers[0] ?? null),
      mode: restoreAvailable ? s.mode : 'natural',
    })),
  setCapabilitiesV1: (capabilities) =>
    set((s) => {
      const restoreCap = capabilities.find(
        (c) => c.capability_id === 'restore_enrolled' || c.capability_id === 'restore_source',
      );
      const isRestoreBlocked = restoreCap
        ? restoreCap.maturity === 'blocked' || !restoreCap.available
        : false;
      return {
        capabilities,
        mode: isRestoreBlocked && s.mode === 'restore' ? 'natural' : s.mode,
      };
    }),
  setReconstructionConsent: (reconstructionConsent) => set({ reconstructionConsent }),
  setMode: (mode) => set({ mode }),
  setSpeakerId: (speakerId) => set({ speakerId }),
  setCutoffHz: (cutoffHz) => set({ cutoffHz }),
  setJob: (job) => set({ job }),
  patchJob: (patch) =>
    set((s) => (s.job ? { job: { ...s.job, ...patch } } : {})),
  // A new report invalidates any selection made against the previous one.
  setReport: (report) => set({ report, selectedUnit: null, highlightRange: null }),
  setAbMode: (abMode) => set({ abMode }),
  setLoudnessMatch: (loudnessMatch) => set({ loudnessMatch }),
  setGainOffsetDb: (gainOffsetDb) => set({ gainOffsetDb }),
  setAdvancedOpen: (advancedOpen) => set({ advancedOpen }),
  setDeckFault: (deckFault) => set({ deckFault }),
  setArtifacts: (artifacts) => set({ artifacts }),
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
  setSelectionRange: (selectionRange) => set({ selectionRange }),
  setShortcutsOpen: (shortcutsOpen) => set({ shortcutsOpen }),
  setStatus: (statusLine) => set({ statusLine }),
  setError: (error, label) => set({ error, errorLabel: error ? (label ?? null) : null }),
  setDragOver: (dragOver) => set({ dragOver }),
  setUpload: (upload) => set({ upload }),
  // A new refusal replaces the old one; nothing queues up.
  setRejection: (rejection) => set({ rejection }),
  // Two runs cannot own the same file. If this run wrote where an older one
  // did, the older row's artefacts are no longer its own — the report on
  // screen and the bytes behind its links would disagree — so the older row
  // is marked superseded rather than left to hand over the newer file. The
  // UI avoids the collision in the first place by naming the second run's
  // output uniquely (state/actions.ts, `uniqueOutputPath`); this is the net
  // under that, and the only thing that catches two *different* clips whose
  // names clean to the same stem.
  pushHistory: (entry) =>
    set((s) => {
      const rest = s.history.filter((h) => h.jobId !== entry.jobId);
      const collides = entry.outcome === 'done' && Boolean(entry.outputPath);
      const marked = collides
        ? rest.map((h) =>
            h.outputPath === entry.outputPath && h.outcome === 'done'
              ? { ...h, supersededBy: entry.jobId }
              : h,
          )
        : rest;
      return {
        history: [entry, ...marked].slice(0, HISTORY_LIMIT),
        currentRunId: entry.jobId,
      };
    }),
  patchHistory: (jobId, patch) =>
    set((s) => ({
      history: s.history.map((h) => (h.jobId === jobId ? { ...h, ...patch } : h)),
    })),
  setCurrentRun: (currentRunId) => set({ currentRunId }),
  setBatch: (batch) => set({ batch }),
  patchBatch: (patch) =>
    set((s) => ({
      batch: s.batch ? { ...s.batch, ...patch } : null,
    })),
  setActiveInspectJobId: (activeInspectJobId) => set({ activeInspectJobId }),
  resetForNewSource: () =>
    set({
      original: null,
      cleaned: null,
      cleanedPath: null,
      job: null,
      report: null,
      abMode: 'original',
      gainOffsetDb: 0,
      deckFault: null,
      artifacts: null,
      playing: false,
      currentTime: 0,
      duration: 0,
      hoverUnit: null,
      highlightRange: null,
      selectedUnit: null,
      selectionRange: null,
      error: null,
      errorLabel: null,
      reconstructionConsent: false,
      view: { start: 0, end: 0 },
      // The run list survives a new clip — it is the session's memory — but
      // nothing in it is on screen any more.
      currentRunId: null,
    }),
}));

export function isRouteBlocked(
  capabilities: CapabilityStatusV1[] | null | undefined,
  routeOrCapId: string,
): boolean {
  if (!capabilities) return false;
  const cap = capabilities.find((c) => c.capability_id === routeOrCapId);
  if (!cap) return false;
  return cap.maturity === 'blocked' || !cap.available;
}

export function getRouteBlockedReason(
  capabilities: CapabilityStatusV1[] | null | undefined,
  routeOrCapId: string,
): string | null {
  if (!capabilities) return null;
  const cap = capabilities.find((c) => c.capability_id === routeOrCapId);
  if (!cap) return null;
  if (cap.maturity === 'blocked' || !cap.available) {
    return cap.reason ?? `Capability ${routeOrCapId} is blocked`;
  }
  return null;
}

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
