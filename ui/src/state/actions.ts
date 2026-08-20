// Async flows that tie the bridge, the engine client, the store and the player
// together. Components call these; nothing here renders.

import { EngineClient, EngineError } from '../api/client';
import { followJob, isTerminal } from '../api/sse';
import type { AudioAnalysis, JobStatus, Profile } from '../api/types';
import { reportTxtPath } from '../api/types';
import { getPlayer } from '../audio/player';
import { getBridge } from '../bridge';
import { getState, useStore, type HistoryEntry, type JobInfo, type SourceInfo } from './store';

/**
 * B6 · health cadence.
 *
 * The engine is a separate process that can die at any moment, so its presence
 * is a *measured* fact with a cadence, not a one-off handshake. While it
 * answers, the ping is deliberately lazy — 5 s notices a death inside one
 * breath and keeps the network log readable. While it is gone the probe is
 * chased hard for the first second (a restart is usually over in well under
 * one) and then backed off to the same 5 s, so a machine left offline
 * overnight does not write a refusal a second for eight hours.
 */
const HEALTH_OK_MS = 10000;
const HEALTH_BACKOFF_MS = [400, 800, 1600, 3000, 5000] as const;

let healthTimer: number | null = null;
let probeStep = 0;
let probeInFlight = false;
let everConnected = false;
/** The pid the last successful health answer came from (a restart changes it). */
let lastEnginePid: number | null = null;
let stopFollow: (() => void) | null = null;
let analyzeAbort: AbortController | null = null;

function describeError(e: unknown): string {
  if (e instanceof EngineError) return e.message;
  if (e instanceof DOMException && e.name === 'AbortError') return 'Cancelled';
  if (e instanceof TypeError) return 'Engine unreachable';
  if (e instanceof Error) return e.message;
  return String(e);
}

export function baseName(path: string): string {
  const parts = path.split(/[\\/]/);
  return parts[parts.length - 1] || path;
}

// ---------------------------------------------------------------------------
// Engine connection

export async function connectEngine(): Promise<void> {
  const st = getState();
  st.setHost(getBridge().host);
  // A retry after a loss stays *offline* while it probes. Flipping the header
  // back to CONNECTING every few hundred milliseconds would turn the one
  // instrument that says whether the engine is there into a strobe.
  if (st.engineStatus !== 'offline') {
    st.setEngine('connecting', st.client, st.engineVersion);
    st.setStatus('Connecting to engine');
  }
  installEngineWatchers();
  await probeEngine();
}

/** The user pressing "Retry now", or the OS telling us the machine woke up. */
export function retryEngineNow(): void {
  if (getState().engineStatus === 'ready') return;
  probeStep = 0;
  clearHealthTimer();
  void probeEngine();
}

function clearHealthTimer(): void {
  if (healthTimer !== null) {
    window.clearTimeout(healthTimer);
    healthTimer = null;
  }
}

function scheduleProbe(delayMs: number): void {
  clearHealthTimer();
  const st = getState();
  st.setEngineProbe(st.engineStatus === 'ready' ? null : Date.now() + delayMs, false);
  healthTimer = window.setTimeout(() => {
    healthTimer = null;
    void probeEngine();
  }, delayMs);
}

/**
 * One probe serves both jobs: it is the keep-alive while the engine answers
 * and the reconnect while it does not. Nothing it does throws — an outage is a
 * state this UI has a design for, not an exception it leaks to the console.
 */
async function probeEngine(): Promise<void> {
  if (probeInFlight) return;
  probeInFlight = true;
  try {
    // The heartbeat does not stop for a hidden tab. It was tempting — a
    // background tab needs no liveness readout — but a run in flight still has
    // a stream attached to it, and that stream must be taken down the moment
    // the engine behind it dies rather than reopening against a refused socket
    // for as long as the user is looking elsewhere. One request every five
    // seconds is not a cost worth that bug.
    getState().setEngineProbe(getState().engineNextProbeAt, true);
    let client = getState().client;
    if (!client) client = new EngineClient(await getBridge().engine.getEndpoint());
    const health = await client.health();
    if (!health.ok) throw new Error('Engine reported not ok');

    const wasDown = getState().engineStatus !== 'ready';
    // A different pid behind the same port is a *different engine*: every job
    // the old one owned died with it, even though nothing ever looked offline.
    const restarted = lastEnginePid !== null && health.engine_pid !== lastEnginePid;
    lastEnginePid = health.engine_pid;
    probeStep = 0;
    getState().setEngine('ready', client, health.version);
    if (!everConnected) {
      everConnected = true;
      getState().setStatus(`Engine ready · v${health.version}`);
    } else if (wasDown || restarted) {
      getState().setStatus(`Engine back · v${health.version}`);
    }
    autoloadFromQuery();
    if (wasDown || restarted) await resumeAfterReconnect(client);
    scheduleProbe(HEALTH_OK_MS);
  } catch (e) {
    markOffline(describeError(e));
    const delay = HEALTH_BACKOFF_MS[Math.min(probeStep, HEALTH_BACKOFF_MS.length - 1)] ?? HEALTH_OK_MS;
    probeStep += 1;
    scheduleProbe(delay);
  } finally {
    probeInFlight = false;
  }
}

/**
 * Any call that fails in a way that smells like a dead engine re-measures
 * liveness *now* rather than waiting out the heartbeat, so the banner and the
 * disabled controls appear on the same beat as the failure the user saw.
 */
function probeSoon(e: unknown): void {
  const looksDead =
    e instanceof TypeError || (e instanceof EngineError && (e.status === 0 || e.status >= 500));
  if (!looksDead) return;
  if (getState().engineStatus !== 'ready') return;
  probeStep = 0;
  clearHealthTimer();
  void probeEngine();
}

/**
 * The engine has gone. Everything the session has loaded stays exactly where
 * it is — clip, analyses, report, run list, selection and zoom are all local
 * facts — and the only thing that changes is that the controls which need a
 * live engine stop pretending they work.
 */
function markOffline(reason: string): void {
  const st = getState();
  if (st.engineStatus !== 'offline') {
    // An `EventSource` pointed at a refused socket reopens forever. The health
    // probe owns reconnection from here, and it re-establishes the stream
    // itself once the engine answers again.
    stopFollow?.();
    stopFollow = null;
    if (st.job) st.patchJob({ streamConnected: false });
    st.setStatus(`Engine offline — ${reason}`);
  }
  st.setEngine('offline', st.client, st.engineVersion);
}

let watchersInstalled = false;
/** Anything that means "the world may have changed" gets a free probe. */
function installEngineWatchers(): void {
  if (watchersInstalled) return;
  watchersInstalled = true;
  const wake = (): void => {
    if (getState().engineStatus !== 'ready') retryEngineNow();
  };
  window.addEventListener('online', wake);
  window.addEventListener('focus', wake);
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState !== 'visible') return;
    if (getState().engineStatus !== 'ready') retryEngineNow();
    else scheduleProbe(250);
  });
}

/**
 * B6 · what a returning engine has to answer.
 *
 * A run that was in flight when the engine vanished has exactly three possible
 * truths, and "still running" is not automatically one of them: it may have
 * finished, it may have failed, or it may have died with the process that
 * owned it. `GET /api/jobs/{id}` settles which — a 404 from a live engine is a
 * positive answer, not a network error, and it means the run is gone.
 */
async function resumeAfterReconnect(client: EngineClient): Promise<void> {
  const job = getState().job;
  if (!job) return;
  if (job.status && isTerminal(job.status.state)) return;
  try {
    const status = await client.getJob(job.id);
    onJobStatus(status);
    if (!isTerminal(status.state)) followFrom(client, job.id);
  } catch (e) {
    if (e instanceof EngineError && e.status === 404) {
      onJobStatus(interruptedStatus(job));
      return;
    }
    // Anything else means we still cannot really talk to it; the health loop
    // comes back round and tries the whole reconciliation again.
  }
}

/** The terminal status a run gets when the engine that owned it went away. */
function interruptedStatus(job: JobInfo): JobStatus {
  const prev = job.status;
  const st = getState();
  return {
    job_id: job.id,
    state: 'failed',
    stage: 'error',
    progress: prev?.progress ?? 0,
    message: 'The engine stopped while this run was in flight',
    unit: prev?.unit ?? null,
    input_path: prev?.input_path ?? st.source?.path ?? '',
    output_path: prev?.output_path ?? job.outputPath,
    report_path: prev?.report_path ?? job.reportPath,
    profile: prev?.profile ?? st.profile,
    started_at: prev?.started_at ?? null,
    finished_at: new Date().toISOString(),
    error: {
      code: 'ENGINE_RESTARTED',
      message:
        'The engine stopped while this run was in flight, so it never finished. Nothing was written; press PROCESS to run it again.',
    },
    report: null,
  };
}

let autoloaded = false;
/** Dev convenience: `?file=/abs/path` loads a clip straight away (web mode). */
function autoloadFromQuery(): void {
  if (autoloaded) return;
  autoloaded = true;
  const file = new URLSearchParams(window.location.search).get('file');
  if (file && !getState().source) {
    void loadSource({ path: file, name: baseName(file), origin: 'file' });
  }
}

function requireClient(): EngineClient {
  const st = getState();
  if (!st.client || st.engineStatus !== 'ready') {
    throw new EngineError(0, 'engine_offline', 'The engine is offline — reconnecting.');
  }
  return st.client;
}

// ---------------------------------------------------------------------------
// Source selection + analysis

export async function loadSource(source: SourceInfo): Promise<void> {
  const st = getState();
  if (st.job && st.job.status && !isTerminal(st.job.status.state)) {
    st.setError('A job is still running — cancel it before loading another clip.');
    return;
  }
  stopFollow?.();
  stopFollow = null;
  analyzeAbort?.abort();
  const player = getPlayer();
  player.pause();
  player.load('cleaned', null);
  st.resetForNewSource();
  // A refusal is answered by the load that follows it. The multi-file note is
  // not a refusal — it explains *this* load ("3 files were dropped, the first
  // audio one was taken") — so it has to survive it.
  if (st.rejection && st.rejection.kind !== 'multi') st.setRejection(null);
  st.setSource(source);
  st.setStatus(`Analyzing ${source.name}`);
  st.setAnalyzing(true);
  const ac = new AbortController();
  analyzeAbort = ac;
  try {
    const client = requireClient();
    const analysis = await client.analyze(source.path, 1200, ac.signal);
    if (ac.signal.aborted) return;
    useStore.getState().setOriginal(analysis);
    player.load('original', client.fileUrl(source.path));
    player.setActive('original');
    useStore.getState().setAbMode('original');
    useStore.getState().setStatus(
      `${source.name} · ${analysis.duration_s.toFixed(1)} s · ${analysis.sample_rate} Hz · ${analysis.channels} ch`,
    );
  } catch (e) {
    if (ac.signal.aborted) return;
    probeSoon(e);
    const msg = describeError(e);
    useStore.getState().setError(`Analysis failed: ${msg}`);
    useStore.getState().setStatus(`Analysis failed: ${msg}`);
  } finally {
    if (analyzeAbort === ac) {
      analyzeAbort = null;
      useStore.getState().setAnalyzing(false);
    }
  }
}

export async function useResolveClip(): Promise<void> {
  const bridge = getBridge();
  if (!bridge.resolve) return;
  const st = getState();
  try {
    const clip = await bridge.resolve.getSelectedClip();
    if (!clip) {
      st.setStatus('No clip selected in Resolve');
      return;
    }
    await loadSource({
      path: clip.filePath,
      name: clip.name || baseName(clip.filePath),
      origin: 'resolve',
      mediaId: clip.mediaId,
    });
  } catch (e) {
    st.setError(`Resolve: ${describeError(e)}`);
  }
}

export async function openFileDialog(): Promise<void> {
  const bridge = getBridge();
  const st = getState();
  try {
    const path = await bridge.files.pickAudio();
    if (!path) return;
    await loadSource({ path, name: baseName(path), origin: 'file' });
  } catch (e) {
    st.setError(`Open file: ${describeError(e)}`);
  }
}

// ---------------------------------------------------------------------------
// B2 · what the tool will take, and what it says when it will not

/**
 * The formats the drop well advertises. The engine decodes through ffmpeg and
 * will happily open more than this, but a UI that accepts silently and fails
 * loudly is worse than one that names its list — so the list on screen, the
 * `accept` attribute and this predicate are the same list.
 */
export const ACCEPTED_EXTENSIONS = [
  'wav',
  'aiff',
  'aif',
  'aifc',
  'mp3',
  'flac',
  'm4a',
  'mp4',
  'mov',
  'aac',
  'ogg',
  'oga',
  'opus',
  'caf',
  'w64',
  'mkv',
  'webm',
] as const;

/** The short version, for the one line the drop well has room for. */
export const ACCEPTED_SHORTLIST = 'wav · aiff · mp3 · flac · m4a · mp4 · mov';

export function extensionOf(name: string): string {
  const dot = name.lastIndexOf('.');
  return dot > 0 ? name.slice(dot + 1).toLowerCase() : '';
}

export function isAcceptedFile(file: File): boolean {
  const ext = extensionOf(file.name);
  if ((ACCEPTED_EXTENSIONS as readonly string[]).includes(ext)) return true;
  // A file with no extension but an honest media MIME type is still media.
  return file.type.startsWith('audio/') || file.type.startsWith('video/');
}

/**
 * Take a whole drop and decide what to do with it. A drop carries a list, so
 * three things have to be answered rather than assumed: a folder (which arrives
 * as an entry with no readable file), several files at once, and a file whose
 * type we cannot open. Each answer is a designed state in the source strip —
 * none of them is a console message.
 */
export async function ingestDataTransfer(dt: DataTransfer): Promise<void> {
  const st = getState();
  st.setRejection(null);

  // `webkitGetAsEntry` is the only way to tell a folder from a file before
  // reading it: a dropped directory yields a File with an empty type and a
  // size the platform makes up, which is indistinguishable from a real file.
  const items = Array.from(dt.items ?? []);
  const folderNames: string[] = [];
  for (const item of items) {
    if (item.kind !== 'file') continue;
    const entry = item.webkitGetAsEntry?.();
    if (entry && entry.isDirectory) folderNames.push(entry.name);
  }

  const files = Array.from(dt.files ?? []);
  const firstFolder = folderNames[0];
  if (firstFolder !== undefined && files.length <= folderNames.length) {
    st.setRejection({
      kind: 'folder',
      name: firstFolder,
      detail:
        folderNames.length > 1
          ? `${folderNames.length} folders were dropped. Drop one audio or video file instead — batches are a job for the command line.`
          : 'Folders are not opened here. Drop one audio or video file instead — batches are a job for the command line.',
    });
    return;
  }

  const first = files[0];
  if (first === undefined) {
    st.setRejection({ kind: 'empty', name: 'nothing', detail: 'The drop carried no file.' });
    return;
  }

  const chosen = files.find(isAcceptedFile);
  if (chosen === undefined) {
    st.setRejection({
      kind: 'type',
      name: first.name,
      detail: `${first.type || `.${extensionOf(first.name) || 'no extension'}`} is not an audio or video format this tool opens.`,
    });
    return;
  }

  if (files.length > 1) {
    // Take the first one that can be opened, and say so rather than silently
    // dropping the rest on the floor.
    const ignored = files.length - 1;
    st.setRejection({
      kind: 'multi',
      name: chosen.name,
      detail: `${files.length} files were dropped — the first audio file was loaded, the other ${ignored} ${ignored === 1 ? 'was' : 'were'} ignored.`,
    });
  }
  await ingestFile(chosen);
}

/** Handles both dropped files and <input type=file> picks. */
export async function ingestFile(file: File): Promise<void> {
  const bridge = getBridge();
  const st = getState();
  if (!isAcceptedFile(file)) {
    st.setRejection({
      kind: 'type',
      name: file.name,
      detail: `${file.type || `.${extensionOf(file.name) || 'no extension'}`} is not an audio or video format this tool opens.`,
    });
    return;
  }
  if (file.size === 0) {
    st.setRejection({
      kind: 'empty',
      name: file.name,
      detail: 'The file is 0 bytes — there is nothing in it to analyze.',
    });
    return;
  }
  const local = bridge.files.pathForFile(file);
  if (local) {
    await loadSource({ path: local, name: file.name, origin: 'drop' });
    return;
  }
  await uploadFile(file);
}

let cancelUploadHandle: (() => void) | null = null;

/** True while bytes are on the wire to `POST /api/upload`. */
export function isUploading(): boolean {
  return cancelUploadHandle !== null;
}

/** Aborts the transfer in flight; the promise in `uploadFile` sees an AbortError. */
export function cancelUpload(): void {
  cancelUploadHandle?.();
}

/**
 * Web-mode ingest: the browser hands us bytes, not a path, so the file goes to
 * the engine's work directory first. The transfer reports real progress and can
 * be abandoned — a 900 MB drop on the wrong file must not be a five-minute
 * sentence.
 */
async function uploadFile(file: File): Promise<void> {
  const st = getState();
  st.setError(null);
  st.setUpload({ name: file.name, loaded: 0, total: file.size, phase: 'sending' });
  st.setStatus(`Uploading ${file.name}`);
  try {
    const client = requireClient();
    const { path } = await client.uploadWithProgress(file, {
      onProgress: (loaded, total) => {
        const cur = getState();
        if (!cur.upload) return;
        cur.setUpload({
          name: file.name,
          loaded,
          total: total || file.size,
          phase: loaded >= (total || file.size) ? 'finishing' : 'sending',
        });
      },
      onCancelHandle: (cancel) => {
        cancelUploadHandle = cancel;
      },
    });
    cancelUploadHandle = null;
    getState().setUpload(null);
    await loadSource({ path, name: file.name, origin: 'upload' });
  } catch (e) {
    cancelUploadHandle = null;
    getState().setUpload(null);
    if (e instanceof DOMException && e.name === 'AbortError') {
      getState().setStatus(`Upload cancelled · ${file.name}`);
      return;
    }
    probeSoon(e);
    const msg = describeError(e);
    getState().setError(`Upload failed: ${msg}`);
    getState().setStatus(`Upload failed: ${msg}`);
  }
}

// ---------------------------------------------------------------------------
// Processing

async function analyzeCleaned(path: string): Promise<AudioAnalysis | null> {
  try {
    return await requireClient().analyze(path, 1200);
  } catch (e) {
    probeSoon(e);
    getState().setStatus(`Cleaned analysis failed: ${describeError(e)}`);
    return null;
  }
}

/** Wall-clock length of a run from the engine's own timestamps. */
function jobDurationMs(status: JobStatus): number | null {
  if (!status.started_at || !status.finished_at) return null;
  const a = Date.parse(status.started_at);
  const b = Date.parse(status.finished_at);
  return Number.isFinite(a) && Number.isFinite(b) && b >= a ? b - a : null;
}

/** Record a terminal run in the session history (goal box B5). */
function recordRun(status: JobStatus, outcome: HistoryEntry['outcome']): void {
  const st = getState();
  const report = status.report ?? null;
  const original = st.original;
  st.pushHistory({
    jobId: status.job_id,
    profile: status.profile ?? st.profile,
    inputPath: status.input_path || st.source?.path || '',
    inputName: baseName(status.input_path || st.source?.path || st.source?.name || ''),
    outcome,
    durationMs: jobDurationMs(status),
    at: Date.now(),
    enhanced: report?.summary.enhanced ?? null,
    unitsTotal: report?.summary.units_total ?? null,
    lufsIn: original?.loudness.integrated_lufs ?? report?.input.integrated_lufs ?? null,
    lufsOut: null,
    noiseIn: original?.noise_floor_db ?? null,
    noiseOut: null,
    outputPath: status.output_path || st.job?.outputPath || '',
    reportPath: status.report_path || st.job?.reportPath || '',
    report,
    status,
    original,
    cleaned: null,
    error: outcome === 'failed' ? status.error?.message || status.message || 'Processing failed' : null,
  });
}

function onJobStatus(status: JobStatus): void {
  const st = getState();
  if (!st.job || st.job.id !== status.job_id) return;
  st.patchJob({ status });
  if (status.state === 'running' || status.state === 'queued') {
    const unit = status.unit ? ` (${status.unit.index}/${status.unit.total})` : '';
    st.setStatus(`${status.message || status.stage}${unit}`);
  }
  if (status.state === 'failed') {
    const msg = status.error?.message || status.message || 'Processing failed';
    st.setError(`${status.error?.code ? `${status.error.code}: ` : ''}${msg}`);
    st.setStatus('Processing failed');
    recordRun(status, 'failed');
  } else if (status.state === 'cancelled') {
    st.setStatus('Processing cancelled');
    recordRun(status, 'cancelled');
  } else if (status.state === 'done') {
    const report = status.report ?? null;
    st.setReport(report);
    const out = status.output_path || st.job.outputPath;
    st.setStatus(`Done · ${baseName(out)}`);
    recordRun(status, 'done');
    void (async () => {
      try {
        const client = requireClient();
        const analysis = await analyzeCleaned(out);
        const cur = getState();
        if (!cur.job || cur.job.id !== status.job_id) return;
        cur.setCleaned(analysis, out);
        // Cache the cleaned analysis on the run so re-selecting it later costs
        // nothing (goal box B5: restore without re-analyzing).
        cur.patchHistory(status.job_id, {
          cleaned: analysis,
          lufsOut: analysis?.loudness.integrated_lufs ?? null,
          noiseOut: analysis?.noise_floor_db ?? null,
        });
        const player = getPlayer();
        player.load('cleaned', client.fileUrl(out));
        player.setActive('cleaned');
        cur.setAbMode('cleaned');
        if (report) {
          const s = report.summary;
          cur.setStatus(
            `Done · ${s.enhanced ?? 0}/${s.units_total ?? 0} units enhanced · ${baseName(out)}`,
          );
        }
      } catch (e) {
        getState().setStatus(`Cleaned master unavailable: ${describeError(e)}`);
      }
    })();
  }
}

/**
 * Attach (or re-attach) the status stream for a job. One place, so a run that
 * was started here and a run that was recovered after an outage are followed
 * by exactly the same code.
 */
function followFrom(client: EngineClient, jobId: string): void {
  stopFollow?.();
  stopFollow = followJob(client, jobId, {
    onStatus: onJobStatus,
    onEnd: () => {
      stopFollow = null;
      // Make sure we have the final status even if the last event was lost.
      const cur = getState();
      if (cur.job && cur.job.id === jobId) {
        cur.patchJob({ streamConnected: false });
        if (!cur.job.status || !isTerminal(cur.job.status.state)) {
          void client
            .getJob(jobId)
            .then(onJobStatus)
            .catch(() => undefined);
        }
      }
    },
    onGone: () => {
      // The engine is up and does not know this job: it died with the process
      // that owned it. That is a terminal answer, not a lost connection.
      const cur = getState();
      if (cur.job && cur.job.id === jobId) onJobStatus(interruptedStatus(cur.job));
    },
    onConnectionChange: (connected) => {
      const cur = getState();
      if (cur.job && cur.job.id === jobId) cur.patchJob({ streamConnected: connected });
    },
  });
}

export async function startJob(): Promise<void> {
  const st = getState();
  if (!st.source) return;
  if (st.job && st.job.status && !isTerminal(st.job.status.state)) return;
  st.setError(null);
  st.setCleaned(null, null);
  st.setReport(null);
  st.setCurrentRun(null);
  getPlayer().load('cleaned', null);
  getPlayer().setActive('original');
  st.setAbMode('original');
  const profile: Profile = st.profile;
  try {
    const client = requireClient();
    const res = await client.createJob({
      input_path: st.source.path,
      profile,
      overwrite: true,
    });
    const job = {
      id: res.job_id,
      outputPath: res.output_path,
      reportPath: res.report_path,
      status: null,
      streamConnected: false,
    };
    useStore.getState().setJob(job);
    useStore.getState().setStatus('Job queued');
    followFrom(client, res.job_id);
  } catch (e) {
    probeSoon(e);
    const msg = describeError(e);
    useStore.getState().setError(`Could not start job: ${msg}`);
    useStore.getState().setStatus(`Could not start job: ${msg}`);
  }
}

export async function cancelJob(): Promise<void> {
  const st = getState();
  if (!st.job) return;
  // Cancelling needs the engine. While it is gone the run's fate is already
  // out of our hands, and the reconnect reconciles it — telling the user
  // "cancel failed" for a job nobody can reach would be noise.
  if (st.engineStatus !== 'ready') {
    st.setStatus('Cannot cancel while the engine is offline — waiting for it to come back');
    return;
  }
  try {
    st.setStatus('Cancelling');
    await requireClient().cancelJob(st.job.id);
  } catch (e) {
    probeSoon(e);
    st.setError(`Cancel failed: ${describeError(e)}`);
  }
}

// ---------------------------------------------------------------------------
// B5 · session history

/**
 * Put a finished run back on screen. Everything it needs was kept when it
 * finished, so this is a state restore, not a re-run: no `/api/analyze`, no
 * re-read of the report, no re-decode. The only reason this function ever
 * touches the engine is a run whose analysis is genuinely missing (the cleaned
 * analysis failed the first time, or the run was superseded before it landed).
 */
export async function selectRun(jobId: string): Promise<void> {
  const st = getState();
  if (st.currentRunId === jobId) return;
  const entry = st.history.find((h) => h.jobId === jobId);
  if (!entry) return;
  if (st.job && st.job.status && !isTerminal(st.job.status.state)) {
    st.setError('A job is still running — cancel it before opening another run.');
    return;
  }
  stopFollow?.();
  stopFollow = null;
  const player = getPlayer();
  player.pause();
  st.setError(null);
  st.setRejection(null);
  st.setSource({ path: entry.inputPath, name: entry.inputName, origin: 'file' });
  st.setJob({
    id: entry.jobId,
    outputPath: entry.outputPath,
    reportPath: entry.reportPath,
    status: entry.status,
    streamConnected: false,
  });
  st.setReport(entry.report);
  st.setCurrentRun(entry.jobId);
  st.resetView();

  const client = st.client;
  const restored = entry.outcome === 'done' && Boolean(entry.outputPath);

  // Only the missing halves are re-fetched.
  let original = entry.original;
  let cleaned = entry.cleaned;
  const needsOriginal = !original;
  const needsCleaned = restored && !cleaned;
  if (client && (needsOriginal || needsCleaned)) {
    getState().setAnalyzing(true);
    getState().setStatus(`Re-reading ${entry.inputName}`);
    if (needsOriginal) {
      original = await client.analyze(entry.inputPath, 1200).catch(() => null);
    }
    if (needsCleaned) {
      cleaned = await client.analyze(entry.outputPath, 1200).catch(() => null);
    }
    getState().setAnalyzing(false);
    getState().patchHistory(entry.jobId, {
      original,
      cleaned,
      lufsIn: original?.loudness.integrated_lufs ?? entry.lufsIn,
      lufsOut: cleaned?.loudness.integrated_lufs ?? entry.lufsOut,
      noiseIn: original?.noise_floor_db ?? entry.noiseIn,
      noiseOut: cleaned?.noise_floor_db ?? entry.noiseOut,
    });
    if (getState().currentRunId !== entry.jobId) return; // the user moved on
  }

  const cur = getState();
  cur.setOriginal(original);
  cur.setCleaned(restored ? cleaned : null, restored ? entry.outputPath : null);
  if (client) {
    player.load('original', client.fileUrl(entry.inputPath));
    if (restored) {
      player.load('cleaned', client.fileUrl(entry.outputPath));
      player.setActive('cleaned');
      cur.setAbMode('cleaned');
    } else {
      player.load('cleaned', null);
      player.setActive('original');
      cur.setAbMode('original');
    }
  }
  cur.setStatus(runStatusLine(entry));
}

function runStatusLine(e: HistoryEntry): string {
  if (e.outcome === 'failed') return `Failed · ${e.inputName}${e.error ? ` — ${e.error}` : ''}`;
  if (e.outcome === 'cancelled') return `Cancelled · ${e.inputName}`;
  const units =
    e.enhanced !== null && e.unitsTotal !== null
      ? ` · ${e.enhanced}/${e.unitsTotal} units enhanced`
      : '';
  return `Done${units} · ${baseName(e.outputPath) || e.inputName}`;
}

// ---------------------------------------------------------------------------
// B7 · report access

function stem(name: string): string {
  const dot = name.lastIndexOf('.');
  return dot > 0 ? name.slice(0, dot) : name;
}

function num(v: number | null | undefined, digits = 1): string | null {
  return v === null || v === undefined || !Number.isFinite(v) ? null : v.toFixed(digits);
}

/**
 * The one line a person pastes into a message when someone asks how the pass
 * went, e.g.
 * `Flute 09 · studio · 5/5 units enhanced · -24.9 -> -21.7 LUFS · noise floor
 *  -48.5 -> -84.7 dB`.
 * Every clause is dropped rather than faked when its number is missing.
 */
export function summaryLine(e: HistoryEntry): string {
  const parts: string[] = [stem(e.inputName) || 'clip', e.profile];
  if (e.outcome !== 'done') {
    parts.push(e.outcome === 'failed' ? `FAILED${e.error ? ` (${e.error})` : ''}` : 'CANCELLED');
    return parts.join(' · ');
  }
  if (e.enhanced !== null && e.unitsTotal !== null) {
    parts.push(`${e.enhanced}/${e.unitsTotal} units enhanced`);
  }
  const li = num(e.lufsIn);
  const lo = num(e.lufsOut);
  if (li && lo) parts.push(`${li} -> ${lo} LUFS`);
  const ni = num(e.noiseIn);
  const no = num(e.noiseOut);
  if (ni && no) parts.push(`noise floor ${ni} -> ${no} dB`);
  return parts.join(' · ');
}

/** The summary for what is currently on screen, or null when nothing is. */
export function currentSummaryLine(): string | null {
  const st = getState();
  const entry = st.history.find((h) => h.jobId === st.currentRunId);
  if (entry) {
    // The live analyses can be fresher than the cached ones (the cleaned
    // analysis lands a moment after the run is recorded).
    return summaryLine({
      ...entry,
      lufsIn: st.original?.loudness.integrated_lufs ?? entry.lufsIn,
      lufsOut: st.cleaned?.loudness.integrated_lufs ?? entry.lufsOut,
      noiseIn: st.original?.noise_floor_db ?? entry.noiseIn,
      noiseOut: st.cleaned?.noise_floor_db ?? entry.noiseOut,
    });
  }
  if (!st.report || !st.source) return null;
  return summaryLine({
    jobId: st.job?.id ?? '',
    profile: st.profile,
    inputPath: st.source.path,
    inputName: st.source.name,
    outcome: 'done',
    durationMs: null,
    at: Date.now(),
    enhanced: st.report.summary.enhanced ?? null,
    unitsTotal: st.report.summary.units_total ?? null,
    lufsIn: st.original?.loudness.integrated_lufs ?? null,
    lufsOut: st.cleaned?.loudness.integrated_lufs ?? null,
    noiseIn: st.original?.noise_floor_db ?? null,
    noiseOut: st.cleaned?.noise_floor_db ?? null,
    outputPath: st.cleanedPath ?? '',
    reportPath: st.job?.reportPath ?? '',
    report: st.report,
    status: st.job?.status ?? null,
    original: st.original,
    cleaned: st.cleaned,
    error: null,
  });
}

async function writeClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    /* no permission, or no clipboard API (insecure context) */
  }
  try {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.top = '0';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand('copy');
    ta.remove();
    return ok;
  } catch {
    return false;
  }
}

/** Copies the one-line summary; returns whether it actually reached the clipboard. */
export async function copyReportSummary(): Promise<boolean> {
  const line = currentSummaryLine();
  if (!line) return false;
  const ok = await writeClipboard(line);
  getState().setStatus(ok ? `Copied · ${line}` : 'Could not reach the clipboard');
  if (!ok) getState().setError('The browser refused clipboard access.');
  return ok;
}

export interface Artifacts {
  master: { url: string; name: string };
  json: { url: string; name: string };
  txt: { url: string; name: string };
}

/**
 * The three files a finished run leaves behind. `/api/audio` is the engine's
 * one file-serving route and it types each response from the file's own
 * extension, so the JSON report and the .txt sidecar come down it exactly like
 * the master does.
 */
export function artifactsFor(outputPath: string | null, reportPath: string | null): Artifacts | null {
  const client = getState().client;
  if (!client || !outputPath) return null;
  const json = reportPath || '';
  const txt = json ? reportTxtPath(json) : '';
  if (!json) return null;
  return {
    master: { url: client.fileUrl(outputPath), name: baseName(outputPath) },
    json: { url: client.fileUrl(json), name: baseName(json) },
    txt: { url: client.fileUrl(txt), name: baseName(txt) },
  };
}

// ---------------------------------------------------------------------------
// Output actions

export async function importToResolve(): Promise<void> {
  const bridge = getBridge();
  const st = getState();
  if (!bridge.resolve || !st.cleanedPath) return;
  try {
    st.setStatus('Importing to Resolve media pool');
    const clip = await bridge.resolve.importMedia(st.cleanedPath);
    st.setStatus(clip ? `Imported ${clip.name}` : 'Resolve did not import the file');
  } catch (e) {
    st.setError(`Import failed: ${describeError(e)}`);
  }
}

export async function replaceInResolve(): Promise<void> {
  const bridge = getBridge();
  const st = getState();
  if (!bridge.resolve || !st.cleanedPath || !st.source?.mediaId) return;
  try {
    st.setStatus('Replacing clip in Resolve');
    const ok = await bridge.resolve.replaceClip(st.source.mediaId, st.cleanedPath);
    st.setStatus(ok ? `Replaced ${st.source.name} with the cleaned master` : 'Resolve refused the replace');
  } catch (e) {
    st.setError(`Replace failed: ${describeError(e)}`);
  }
}

export async function revealOutput(): Promise<void> {
  const bridge = getBridge();
  const st = getState();
  if (!st.cleanedPath) return;
  try {
    await bridge.files.revealInFinder(st.cleanedPath);
  } catch (e) {
    st.setError(`Reveal failed: ${describeError(e)}`);
  }
}

// ---------------------------------------------------------------------------
// Playback glue

export function setAb(mode: 'original' | 'cleaned'): void {
  const player = getPlayer();
  if (mode === 'cleaned' && !player.hasDeck('cleaned')) return;
  player.setActive(mode);
  getState().setAbMode(mode);
}

export function togglePlay(): void {
  getPlayer().toggle();
}

export function seekTo(time: number): void {
  getPlayer().seek(time);
}
