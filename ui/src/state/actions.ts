// Async flows that tie the bridge, the engine client, the store and the player
// together. Components call these; nothing here renders.

import { EngineClient, EngineError } from '../api/client';
import { followJob, isTerminal } from '../api/sse';
import type { AudioAnalysis, JobStatus, Profile } from '../api/types';
import { getPlayer } from '../audio/player';
import { getBridge } from '../bridge';
import { getState, useStore, type SourceInfo } from './store';

const HEALTH_RETRY_MS = 1500;
const HEALTH_PING_MS = 10000;
let healthTimer: number | null = null;
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
  const bridge = getBridge();
  st.setHost(bridge.host);
  st.setEngine('connecting', st.client, st.engineVersion);
  st.setStatus('Connecting to engine');
  try {
    const ep = await bridge.engine.getEndpoint();
    const client = new EngineClient(ep);
    const health = await client.health();
    if (!health.ok) throw new Error('Engine reported not ok');
    st.setEngine('ready', client, health.version);
    st.setStatus(`Engine ready · v${health.version}`);
    if (healthTimer !== null) window.clearTimeout(healthTimer);
    healthTimer = window.setTimeout(pingEngine, HEALTH_PING_MS);
    autoloadFromQuery();
  } catch (e) {
    st.setEngine('offline', null, null);
    st.setStatus(`Engine offline — ${describeError(e)}`);
    if (healthTimer !== null) window.clearTimeout(healthTimer);
    healthTimer = window.setTimeout(() => {
      healthTimer = null;
      void connectEngine();
    }, HEALTH_RETRY_MS);
  }
}

/** Quiet liveness check while ready; flips to offline (and back) if the engine dies. */
async function pingEngine(): Promise<void> {
  healthTimer = null;
  const st = getState();
  if (st.engineStatus !== 'ready' || !st.client) return;
  try {
    await st.client.health();
    healthTimer = window.setTimeout(pingEngine, HEALTH_PING_MS);
  } catch (e) {
    st.setEngine('offline', null, null);
    st.setStatus(`Engine offline — ${describeError(e)}`);
    healthTimer = window.setTimeout(() => {
      healthTimer = null;
      void connectEngine();
    }, HEALTH_RETRY_MS);
  }
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
  const c = getState().client;
  if (!c) throw new Error('Engine is offline');
  return c;
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
    player.load('original', client.audioUrl(source.path));
    player.setActive('original');
    useStore.getState().setAbMode('original');
    useStore.getState().setStatus(
      `${source.name} · ${analysis.duration_s.toFixed(1)} s · ${analysis.sample_rate} Hz · ${analysis.channels} ch`,
    );
  } catch (e) {
    if (ac.signal.aborted) return;
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

/** Handles both dropped files and <input type=file> picks. */
export async function ingestFile(file: File): Promise<void> {
  const bridge = getBridge();
  const st = getState();
  const local = bridge.files.pathForFile(file);
  if (local) {
    await loadSource({ path: local, name: file.name, origin: 'drop' });
    return;
  }
  // Web fallback: upload to the engine work dir.
  st.setStatus(`Uploading ${file.name}`);
  try {
    const client = requireClient();
    const { path } = await client.upload(file);
    await loadSource({ path, name: file.name, origin: 'upload' });
  } catch (e) {
    st.setError(`Upload failed: ${describeError(e)}`);
  }
}

// ---------------------------------------------------------------------------
// Processing

async function analyzeCleaned(path: string): Promise<AudioAnalysis | null> {
  try {
    return await requireClient().analyze(path, 1200);
  } catch (e) {
    getState().setStatus(`Cleaned analysis failed: ${describeError(e)}`);
    return null;
  }
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
  } else if (status.state === 'cancelled') {
    st.setStatus('Processing cancelled');
  } else if (status.state === 'done') {
    const report = status.report ?? null;
    st.setReport(report);
    const out = status.output_path || st.job.outputPath;
    st.setStatus(`Done · ${baseName(out)}`);
    void (async () => {
      try {
        const client = requireClient();
        const analysis = await analyzeCleaned(out);
        const cur = getState();
        if (!cur.job || cur.job.id !== status.job_id) return;
        cur.setCleaned(analysis, out);
        const player = getPlayer();
        player.load('cleaned', client.audioUrl(out));
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

export async function startJob(): Promise<void> {
  const st = getState();
  if (!st.source) return;
  if (st.job && st.job.status && !isTerminal(st.job.status.state)) return;
  st.setError(null);
  st.setCleaned(null, null);
  st.setReport(null);
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
    stopFollow?.();
    stopFollow = followJob(client, res.job_id, {
      onStatus: onJobStatus,
      onEnd: () => {
        stopFollow = null;
        // Make sure we have the final status even if the last event was lost.
        const cur = getState();
        if (cur.job && cur.job.id === res.job_id) {
          cur.patchJob({ streamConnected: false });
          if (!cur.job.status || !isTerminal(cur.job.status.state)) {
            void client.getJob(res.job_id).then(onJobStatus).catch(() => undefined);
          }
        }
      },
      onConnectionChange: (connected) => {
        const cur = getState();
        if (cur.job && cur.job.id === res.job_id) cur.patchJob({ streamConnected: connected });
      },
    });
  } catch (e) {
    const msg = describeError(e);
    useStore.getState().setError(`Could not start job: ${msg}`);
    useStore.getState().setStatus(`Could not start job: ${msg}`);
  }
}

export async function cancelJob(): Promise<void> {
  const st = getState();
  if (!st.job) return;
  try {
    st.setStatus('Cancelling');
    await requireClient().cancelJob(st.job.id);
  } catch (e) {
    st.setError(`Cancel failed: ${describeError(e)}`);
  }
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
