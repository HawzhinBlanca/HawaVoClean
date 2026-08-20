// Async flows that tie the bridge, the engine client, the store and the player
// together. Components call these; nothing here renders.

import { EngineClient, EngineError } from '../api/client';
import { followJob, isTerminal } from '../api/sse';
import type { AudioAnalysis, JobStatus, Profile } from '../api/types';
import { reportTxtPath } from '../api/types';
import { getPlayer, type DeckFault } from '../audio/player';
import { getBridge } from '../bridge';
import { waveView } from '../render/viewWindow';
import { unitsEnhanced } from './plural';
import {
  classifyFailure,
  failureSource,
  installFailureNet,
  isCancellation,
  type UiFailure,
} from './errors';
import {
  getState,
  useStore,
  type ArtifactState,
  type DeckFaultInfo,
  type HistoryEntry,
  type JobInfo,
  type SourceInfo,
} from './store';

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

/**
 * C5 · the short form, for the status line. Never an exception: everything
 * goes through `classifyFailure`, which owns the sentences (state/errors.ts).
 */
function describeError(e: unknown, name?: string): string {
  return classifyFailure(e, name).headline;
}

/**
 * Show a failure the one designed way: the sentence in the error bar, the
 * short form in the status line, and nothing at all when it was a
 * cancellation (an aborted fetch is how this UI abandons work, not a fault).
 */
function reportFailure(f: UiFailure, statusPrefix?: string): void {
  if (isCancellation(f)) return;
  const st = getState();
  // The bar's label is the failure's own source, never a constant: see
  // `failureSource` in state/errors.ts.
  st.setError(f.detail, failureSource(f));
  st.setStatus(statusPrefix ? `${statusPrefix} · ${f.headline}` : f.headline);
}

/**
 * Why the clip currently in the strip has no analysis, or null when it has one
 * (or when nothing has failed). The store has no field for this and adding one
 * is not this agent's to add, so it lives here and is read alongside `error`,
 * which is set on exactly the same beat — a component that subscribes to
 * `error` re-renders whenever this changes.
 */
let lastSourceFailure: UiFailure | null = null;

export function sourceFailureLabel(): string {
  switch (lastSourceFailure?.kind) {
    case 'no-audio':
      return 'no audio track';
    case 'missing':
      return 'file not there';
    case 'forbidden':
      return 'out of bounds';
    case 'rate-high':
    case 'rate-low':
      return 'rate refused';
    case 'too-large':
      return 'too large';
    case 'offline':
      return 'not read yet';
    case null:
    case undefined:
      return 'not analyzed';
    default:
      return 'unreadable';
  }
}

/** Same, from a raw thrown value. Returns the classification for the caller. */
function failed(e: unknown, name?: string, statusPrefix?: string): UiFailure {
  const f = classifyFailure(e, name);
  reportFailure(f, statusPrefix);
  return f;
}

/**
 * C5 · the rates the *pipeline* will take, which are not the rates
 * `/api/analyze` will take.
 *
 * Measured against the running engine: a 192 kHz WAV analyses happily (200,
 * `sample_rate: 192000`) and then the run fails a second later with
 * `Input sample rate 192000 Hz exceeds maximum supported 48000 Hz`. Analysis
 * decodes; the pipeline pre-flights. The failure is designed, but arming
 * PROCESS on a clip that cannot be processed is leading the user on, so the
 * source strip flags the rate the moment the analysis lands.
 *
 * These mirror `MIN_SUPPORTED_SAMPLE_RATE` and `EngineConfig.max_sample_rate`
 * in the engine, where the maximum is capped by its own schema (`le=48000`).
 * The flag is advisory only — it never blocks the button, so if the engine
 * ever widens its range the worst case is a stale note, not a wall.
 */
export const PIPELINE_MIN_RATE_HZ = 8000;
export const PIPELINE_MAX_RATE_HZ = 48000;

/** Why this clip's rate will be refused by the pipeline, or null if it will not. */
export function rateWarning(sampleRate: number): string | null {
  if (!Number.isFinite(sampleRate) || sampleRate <= 0) return null;
  const khz = (v: number): string => `${(v / 1000).toFixed(v % 1000 === 0 ? 0 : 1)} kHz`;
  if (sampleRate > PIPELINE_MAX_RATE_HZ) {
    return `${khz(sampleRate)} is above this tool’s range — it reads the file, but a run will be refused. Resample to ${khz(PIPELINE_MAX_RATE_HZ)} or below first.`;
  }
  if (sampleRate < PIPELINE_MIN_RATE_HZ) {
    return `${khz(sampleRate)} is below this tool’s range — it reads the file, but a run will be refused. ${khz(PIPELINE_MIN_RATE_HZ)} is the minimum.`;
  }
  return null;
}

/**
 * C5 · the channel layouts the *pipeline* will take, which — exactly like the
 * sample rates above — are not the layouts `/api/analyze` will take.
 *
 * Measured against the running engine: a 7.1 WAV analyses happily (200,
 * `channels: 8`) and the run then fails with "Multi-channel audio with 8
 * channels is not supported without explicit split_speakers declaration."
 * The sample-rate case had a warning on its cell and this one did not, so an
 * eight-channel file looked ordinary all the way up to a refusal it could not
 * have predicted. `classifyFailure` owns the refusal (state/errors.ts); this
 * owns the warning that comes ten seconds earlier.
 *
 * Advisory only, like the rate flag: it never blocks PROCESS, so if the engine
 * ever learns to fold a 7.1 mix down itself the worst case is a stale note.
 */
export const PIPELINE_MAX_CHANNELS = 2;

/** Why this clip's channel count will be refused, or null if it will not. */
export function channelWarning(channels: number): string | null {
  if (!Number.isFinite(channels) || channels <= PIPELINE_MAX_CHANNELS) return null;
  return `${channels} channels — this tool reads the file, but a run will be refused. It cleans one voice at a time; fold the file down to mono or stereo first.`;
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
      retryFaultedDecks();
      retryStrandedAnalysis();
      // An outage is a gap in which anything on disk may have moved. The run
      // on screen gets the same verification a restore does — one ranged byte
      // per artefact plus the master-length check — no `/api/analyze`.
      void reverifyCurrentRun();
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
  probeNow();
}

/**
 * The same re-measurement, for evidence that is not an exception.
 *
 * A job in flight makes no requests of its own — the SSE stream is the only
 * thing on the wire — so when the engine dies mid-run there is no `TypeError`
 * for `probeSoon` to classify. The stream simply ends. That is evidence, and
 * this is how it gets used: probe now, on the beat the stream died, instead of
 * waiting out the heartbeat.
 */
function probeNow(): void {
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
  // C5 · the net under every flow: anything that escapes a `catch` still ends
  // in the designed error bar rather than in a console nobody has open.
  installFailureNet((f) => reportFailure(f));
  installDeckFaultNet();
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

/**
 * How many envelope buckets to ask `/api/analyze` for.
 *
 * Decoding the file is the entire cost of that call — measured on a 90 min /
 * 1.04 GB WAV, 1200 buckets and 2400 buckets both take 16.0 s — so asking for
 * one bucket per device column costs nothing and lands the whole-file envelope
 * at display resolution. The alternative (leaving the base coarse and letting
 * the waveform re-query `/api/peaks` for the whole file at fit zoom) is a
 * second full decode: 2.5 s of engine time on that same file, per clip.
 */
function envelopeBuckets(): number {
  return Math.min(8000, Math.max(1200, Math.round(waveView.maxBuckets)));
}

function requireClient(): EngineClient {
  const st = getState();
  if (!st.client || st.engineStatus !== 'ready') {
    throw new EngineError(0, 'engine_offline', 'The engine is offline — reconnecting.');
  }
  return st.client;
}

// ---------------------------------------------------------------------------
// The A/B control is not allowed to state something untrue
//
// A deck that cannot be played used to be handled entirely inside the player:
// it fell back to the other deck and said nothing. The store never heard about
// it, so the A/B switch stayed lit on CLEANED with `aria-checked="true"` while
// the ORIGINAL element was the one making sound — the screen asserting a fact
// about the audio that was the opposite of the truth, with no error anywhere.
// Every fallback is now an event, and these three lines are what happens to it:
// the switch follows the player, the screen carries a sentence, and any
// artefact the same failure condemns is marked unavailable with it.

/** A deck's name, in the words the screen uses. */
function deckName(deck: 'original' | 'cleaned'): string {
  return deck === 'cleaned' ? 'cleaned master' : 'original';
}

/** How long a deck turned out to be, in the words a person reads. */
function shortLength(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return 'no audio at all';
  // A 100-byte WAV measures 0.000396 s, which rounds to "0 ms" — a number that
  // reads like a rounding error rather than like the fact it is.
  if (seconds < 0.001) return 'less than a millisecond of audio';
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms of audio`;
  return `${seconds.toFixed(1)} s of audio`;
}

/** The player's fault, as the sentence the transport shows. */
function describeDeckFault(f: DeckFault): DeckFaultInfo {
  // A truncated deck is the one kind whose sentence needs the measurement in
  // it: the file *is* there and the browser *did* open it, so "could not be
  // loaded — it is empty" would read like a contradiction without the numbers.
  if (f.kind === 'truncated') {
    const got = shortLength(f.duration?.got ?? 0);
    const expected = f.duration?.expected ?? null;
    const against = expected ? `, where this run is ${expected.toFixed(1)} s long` : '';
    const fellShort = f.fellBackTo ? `, so the A/B is playing the ${deckName(f.fellBackTo)}` : '';
    return {
      deck: f.deck,
      headline: f.deck === 'cleaned' ? 'CLEANED DECK UNAVAILABLE' : 'ORIGINAL DECK UNAVAILABLE',
      detail: `The ${deckName(f.deck)} opened with ${got}${against} — the file on disk is empty or truncated${fellShort}.`,
    };
  }
  const why =
    f.kind === 'missing'
      ? 'it is not on disk any more'
      : f.kind === 'forbidden'
        ? 'the engine refused to serve it'
        : f.kind === 'network'
          ? 'the engine could not be reached'
          : 'nothing here could decode it';
  const fell = f.fellBackTo ? `, so the A/B is playing the ${deckName(f.fellBackTo)}` : '';
  return {
    deck: f.deck,
    headline: f.deck === 'cleaned' ? 'CLEANED DECK UNAVAILABLE' : 'ORIGINAL DECK UNAVAILABLE',
    detail: `The ${deckName(f.deck)} could not be loaded — ${why}${fell}.`,
  };
}

/**
 * What the artefact row and the run list should say about a master the deck
 * has just refused, or null when the fault condemns nothing.
 *
 * `network` is the null case and the only one: an engine that is not answering
 * says nothing at all about the file, which is still exactly where the run
 * left it. Everything else means this master cannot be handed over as this
 * run's master — either because it is not there, or because what is there is
 * not the audio the report describes — and a control that offers it anyway is
 * the same lie the A/B switch used to tell.
 */
function condemnedMaster(f: DeckFault): { reason: string; flag: string } | null {
  switch (f.kind) {
    case 'missing':
      return {
        reason:
          'The cleaned master is no longer on disk — it was moved or deleted after the run.',
        flag: 'FILE GONE',
      };
    case 'forbidden':
      return { reason: 'The engine refused to serve the cleaned master.', flag: 'NO ACCESS' };
    case 'truncated': {
      const got = shortLength(f.duration?.got ?? 0);
      return {
        reason: `The cleaned master is still on disk, but it holds ${got} — it is empty or truncated, so it is not this run's master any more.`,
        flag: 'FILE BROKEN',
      };
    }
    case 'unreadable':
      return {
        reason:
          'The cleaned master is still on disk, but nothing here can decode it — the bytes are not the audio this run wrote.',
        flag: 'FILE BROKEN',
      };
    default:
      return null;
  }
}

/** Is this fault about the file the screen is currently showing? */
function faultIsCurrent(f: DeckFault): boolean {
  const st = getState();
  const client = st.client;
  if (!client) return false;
  const path = f.deck === 'cleaned' ? st.cleanedPath : (st.source?.path ?? null);
  return Boolean(path) && client.fileUrl(path as string) === f.url;
}

let deckFaultNetInstalled = false;
function installDeckFaultNet(): void {
  if (deckFaultNetInstalled) return;
  deckFaultNetInstalled = true;
  // The player cannot tell "the engine is gone" from "these bytes are junk" on
  // its own — Chromium reports both as `MediaError.code` 4 — so it is handed
  // the one fact this module owns.
  getPlayer().setLivenessProbe(() => getState().engineStatus === 'ready');
  getPlayer().onFault((f) => {
    // The switch follows the player unconditionally — even for a fault about a
    // clip that is no longer on screen, `activeDeck` is the truth about what
    // is audible and the control renders exactly that.
    getState().setAbMode(getPlayer().activeDeck);
    if (!faultIsCurrent(f)) return;
    const st = getState();
    const said = describeDeckFault(f);
    st.setDeckFault(said);
    st.setStatus(said.detail);
    if (f.deck !== 'cleaned') return;
    // A cleaned deck that cannot be played is a master that cannot be handed
    // over either: the same fact, told once. Before this it was told only for
    // the two kinds where the engine had answered "no" — so a master truncated
    // to 100 bytes, or a PNG renamed `.wav`, left `Master WAV` an enabled
    // `<a download>` for a file the app had just declared unplayable.
    // The analysis goes with it — it describes a file that is not there any
    // more in any useful sense, and leaving it in place keeps the waveform
    // asking `/api/peaks` for it.
    const condemned = condemnedMaster(f);
    if (condemned) {
      st.setCleaned(null, st.cleanedPath);
      markArtifacts({ master: false, reason: condemned.reason, flag: condemned.flag });
    }
  });
}

/**
 * Patch the on-screen run's artefact availability, and record the same answer
 * on its history row so the run list stops offering what is not there.
 */
function markArtifacts(patch: Partial<ArtifactState> & { reason: string }): void {
  const st = getState();
  const cur = st.artifacts;
  const next: ArtifactState = {
    master: patch.master ?? cur?.master ?? true,
    json: patch.json ?? cur?.json ?? true,
    txt: patch.txt ?? cur?.txt ?? true,
    reason: patch.reason,
    ...(patch.flag ? { flag: patch.flag } : {}),
  };
  st.setArtifacts(next);
  if (st.currentRunId) st.patchHistory(st.currentRunId, { artifacts: next });
}

/**
 * B6 · a deck that failed because the engine was not answering is not a deck
 * that is gone. When the engine comes back, ask for it again rather than
 * leaving the switch greyed for the rest of the session.
 */
/**
 * B6 · a clip stranded by an outage has to come back on its own.
 *
 * `classifyFailure` tells the user, in as many words, that a failed analysis
 * "comes back on its own when it reconnects". Nothing made that true: the
 * reconnect path resumed *jobs* only, so a clip whose analyze died with the
 * engine sat at NO ANALYSIS for ever with PROCESS disabled and that sentence
 * still on screen. Now the promise is kept.
 */
function retryStrandedAnalysis(): void {
  const st = getState();
  const source = st.source;
  if (!source) return;
  if (st.original) return; // already analysed — nothing stranded
  if (isAnalyzing()) return; // a retry is already in flight
  if (lastSourceFailure?.retryable !== true) return; // a refusal is not an outage
  void loadSource(source);
}

function retryFaultedDecks(): void {
  const st = getState();
  const client = st.client;
  if (!client) return;
  const player = getPlayer();
  let retried = false;
  if (player.deckFault('original')?.kind === 'network' && st.source) {
    player.load('original', client.fileUrl(st.source.path), st.original?.duration_s ?? null);
    retried = true;
  }
  // The cleaned deck is retried on the *path*, not on the analysis. Requiring
  // `st.cleaned` meant that a run whose cleaned analysis had also died in the
  // same outage could never get its deck back, which is precisely the run most
  // likely to have lost it.
  if (player.deckFault('cleaned')?.kind === 'network' && st.cleanedPath) {
    player.load(
      'cleaned',
      client.fileUrl(st.cleanedPath),
      st.cleaned?.duration_s ?? st.original?.duration_s ?? null,
    );
    retried = true;
  }
  if (retried) st.setDeckFault(null);
}

/**
 * B5 · the run on screen is a run like any other, and its files can go too.
 *
 * `selectRun` verifies a run's artefacts when it puts it on screen, and then
 * never looks again — so deleting the master of the run you are *already*
 * looking at changed nothing: `Master WAV` stayed an enabled link whose own
 * href answered 404, the row carried no flag, and the cleaned deck played on
 * from the blob already in memory, so nothing ever probed the file. This is
 * the same verification a restore makes (one ranged byte per artefact, plus
 * the master's length against the size this run recorded — see
 * `verifyArtifacts`), run for the current run on the gestures that mean
 * "look again" — re-picking the row it is on, and the engine coming back.
 * It never calls `/api/analyze`, so it costs nothing a restore does not.
 */
async function reverifyCurrentRun(): Promise<void> {
  const st = getState();
  const id = st.currentRunId;
  if (!id) return;
  const entry = st.history.find((h) => h.jobId === id);
  if (!entry) return;
  const verified = await verifyArtifacts(st.client, entry);
  if (!verified) return;
  const avail = verified.art;
  const cur = getState();
  if (cur.currentRunId !== id) return; // the user moved on while we asked
  const was = cur.artifacts?.master ?? true;
  cur.setArtifacts(avail);
  cur.patchHistory(id, {
    artifacts: avail,
    // A healthy sighting (re)records the yardstick; a condemned one never
    // overwrites it — the recorded size is the fact the condemnation rests on.
    ...(avail.master && verified.masterBytes !== null
      ? { masterBytes: verified.masterBytes }
      : {}),
  });
  if (avail.master || !was) return;
  // The master has gone since this run was put on screen. Everything the run
  // is — its report, its numbers, its units — is still true; the deck and the
  // download are what stop being offered.
  cur.setCleaned(null, cur.cleanedPath);
  getPlayer().load('cleaned', null);
  getPlayer().setActive('original');
  cur.setAbMode('original');
  cur.setDeckFault({
    deck: 'cleaned',
    headline: 'CLEANED DECK UNAVAILABLE',
    detail: deckGoneDetail(avail),
  });
  cur.setStatus(avail.reason);
}

// ---------------------------------------------------------------------------
// Source selection + analysis

export async function loadSource(source: SourceInfo): Promise<void> {
  const st = getState();
  if (st.job && st.job.status && !isTerminal(st.job.status.state)) {
    st.setError('A job is still running — cancel it before loading another clip.', 'Busy');
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
  lastSourceFailure = null;
  st.setStatus(`Analyzing ${source.name}`);
  st.setAnalyzing(true);
  const ac = new AbortController();
  analyzeAbort = ac;
  try {
    const client = requireClient();
    const analysis = await client.analyze(source.path, envelopeBuckets(), ac.signal);
    if (ac.signal.aborted) return;
    useStore.getState().setOriginal(analysis);
    // The engine has just decoded this file end to end, so its length is
    // known: a deck that opens far shorter than that is not this clip.
    player.load('original', client.fileUrl(source.path), analysis.duration_s);
    player.setActive('original');
    useStore.getState().setAbMode('original');
    useStore.getState().setStatus(
      `${source.name} · ${analysis.duration_s.toFixed(1)} s · ${analysis.sample_rate} Hz · ${analysis.channels} ch`,
    );
  } catch (e) {
    if (ac.signal.aborted) return;
    probeSoon(e);
    lastSourceFailure = failed(e, source.name);
  } finally {
    if (analyzeAbort === ac) {
      analyzeAbort = null;
      useStore.getState().setAnalyzing(false);
    }
  }
}

/**
 * C5 · a long analyze must have a way out.
 *
 * Analysis of a three-hour file is half a minute of nothing to look at, and
 * until now the only exit was to drop a different file. `analyzeAbort` was
 * already there — it just had no button on it. Aborting leaves the clip
 * loaded but un-analysed, which is exactly the state the strip and the
 * PROCESS plate already know how to draw.
 */
export function isAnalyzing(): boolean {
  return analyzeAbort !== null;
}

export function cancelAnalysis(): void {
  const ac = analyzeAbort;
  if (!ac) return;
  ac.abort();
  analyzeAbort = null;
  const st = getState();
  st.setAnalyzing(false);
  st.setStatus(`Analysis cancelled · ${st.source?.name ?? ''}`.trim());
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
    reportFailure(classifyFailure(e), 'Resolve');
  }
}

export async function openFileDialog(): Promise<void> {
  const bridge = getBridge();
  try {
    const path = await bridge.files.pickAudio();
    if (!path) return;
    await loadSource({ path, name: baseName(path), origin: 'file' });
  } catch (e) {
    reportFailure(classifyFailure(e), 'Open file');
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
    failed(e, file.name, 'Upload failed');
  }
}

// ---------------------------------------------------------------------------
// Processing

async function analyzeCleaned(path: string): Promise<AudioAnalysis | null> {
  try {
    return await requireClient().analyze(path, envelopeBuckets());
  } catch (e) {
    probeSoon(e);
    getState().setStatus(`Cleaned analysis failed · ${describeError(e, baseName(path))}`);
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
    // The run list and the copy-summary line quote this; it has to be the
    // readable form, not the subprocess repr the engine logs.
    error:
      outcome === 'failed'
        ? jobFailureLine(status, baseName(status.input_path || st.source?.path || ''))
        : null,
  });
}

/** The engine's own words for why a run failed, in one string. */
function jobFailureMessage(status: JobStatus): string {
  return status.error?.message || status.message || 'the run failed';
}

/** The same thing, but fit to be read — used by the run list and the summary. */
function jobFailureLine(status: JobStatus, name?: string): string {
  return classifyFailure(
    new EngineError(400, status.error?.code ?? 'job_failed', jobFailureMessage(status)),
    name,
  ).headline;
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
    // C5 · a run's own failure gets the same treatment as a call's. The
    // engine reports `INVALID_USER_INPUT` plus a message written for a log;
    // neither an enum nor a subprocess repr belongs in front of a person, so
    // the pair is re-classified through the same table as everything else.
    const f = classifyFailure(
      new EngineError(400, status.error?.code ?? 'job_failed', jobFailureMessage(status)),
      st.source?.name ?? baseName(status.input_path || ''),
    );
    st.setError(f.detail, failureSource(f));
    st.setStatus(`Processing failed · ${f.headline}`);
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
        // A7 · the size the engine serves for this master *now* is the
        // yardstick every later re-verification measures against — a
        // truncation changes it, and a HEAD-style stat never notices. One
        // byte, recorded once, on the run's own row (never guarded by "is
        // this still the current job": the row belongs to that run).
        void client
          .verify(out)
          .then((p) => {
            if (p.delivered && p.size !== null) {
              getState().patchHistory(status.job_id, { masterBytes: p.size });
            }
          })
          .catch(() => undefined);
        const analysis = await analyzeCleaned(out);
        const cur = getState();
        // B5 · the history row belongs to *that* run, not to whatever is on
        // screen when its analysis lands. This patch used to sit behind the
        // "is this still the current job?" guard below, so a run whose
        // analysis returned after the user had moved on lost its cached
        // numbers permanently: the row read "— LUFS Δ" for the rest of the
        // session and re-selecting it cost a full POST /api/analyze.
        cur.patchHistory(status.job_id, {
          cleaned: analysis,
          lufsOut: analysis?.loudness.integrated_lufs ?? null,
          noiseOut: analysis?.noise_floor_db ?? null,
        });
        if (!cur.job || cur.job.id !== status.job_id) return;
        cur.setCleaned(analysis, out);
        const player = getPlayer();
        player.load(
          'cleaned',
          client.fileUrl(out),
          // What the run says this master holds: the deck is checked against
          // it, so a master that decodes to nothing cannot sit there lit.
          analysis?.duration_s ?? cur.original?.duration_s ?? null,
        );
        player.setActive('cleaned');
        cur.setAbMode('cleaned');
        if (report) {
          const s = report.summary;
          cur.setStatus(
            `Done · ${unitsEnhanced(s.enhanced ?? 0, s.units_total ?? 0)} · ${baseName(out)}`,
          );
        }
      } catch (e) {
        getState().setStatus(`Cleaned master unavailable · ${describeError(e)}`);
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
      // B6 · a stream that drops mid-job is the earliest evidence there is
      // that the engine may be gone — earlier than any request we would
      // otherwise make, because a running job makes none. Measured before
      // this line: killing the engine 3 s into a run left RUNNING / 22% /
      // "Enhancing unit 2/5" on screen for ~11 s, because the only thing that
      // noticed was the 10 s heartbeat. The stream's own death now re-measures
      // liveness on the spot; if the engine is in fact fine (a proxy hiccup,
      // a laptop lid) the probe answers `ok`, nothing changes, and `followJob`
      // reconnects as it always did. The healthy cadence is untouched.
      if (!connected) probeNow();
    },
  });
}

/**
 * `…_studio.wav` -> `…_studio-2.wav`, and never onto a name this session has
 * already used. A trailing `-N` is replaced rather than stacked, so a fourth
 * run is `-4`, not `-2-3-4`.
 */
function bumpOutputPath(base: string, taken: ReadonlySet<string>): string {
  const stem = base.replace(/\.wav$/i, '').replace(/-\d+$/, '');
  for (let n = 2; n < 1000; n++) {
    const candidate = `${stem}-${n}.wav`;
    if (!taken.has(candidate)) return candidate;
  }
  return `${stem}-${Date.now()}.wav`;
}

/**
 * The output path this run should be given, or null to let the engine name it.
 *
 * The engine's default is `<clip>_studio.wav` / `<clip>_clean.wav` beside the
 * input, which is the right name and the one the first run of a profile keeps —
 * so the common case sends no `output_path` at all and the naming rule stays
 * the engine's, with no copy of it here to drift. Only when *this session* has
 * already finished a run of the same profile on the same clip is a name chosen,
 * and it is derived from that run's real output path (the engine's own answer,
 * not a guess) with a suffix that avoids every path the session has used.
 *
 * Runs that did not finish claim nothing: a failed pass writes no master, so
 * re-running after one gets the plain name back.
 */
function uniqueOutputPath(inputPath: string, profile: Profile): string | null {
  const done = getState().history.filter((h) => h.outcome === 'done' && h.outputPath);
  const prior = done.find((h) => h.inputPath === inputPath && h.profile === profile);
  if (!prior) return null;
  return bumpOutputPath(prior.outputPath, new Set(done.map((h) => h.outputPath)));
}

export async function startJob(): Promise<void> {
  const st = getState();
  if (!st.source) return;
  if (st.job && st.job.status && !isTerminal(st.job.status.state)) return;
  st.setError(null);
  st.setCleaned(null, null);
  st.setReport(null);
  st.setCurrentRun(null);
  st.setDeckFault(null);
  st.setArtifacts(null);
  getPlayer().load('cleaned', null);
  getPlayer().setActive('original');
  st.setAbMode('original');
  const profile: Profile = st.profile;
  try {
    const client = requireClient();
    // B5 · a second run of the same profile must not land on the first run's
    // files. Left to the engine's default naming both write `<clip>_studio.wav`,
    // and the older history row then shows its own cached report beside
    // download links that hand over the newer run's bytes.
    const output = uniqueOutputPath(st.source.path, profile);
    const res = await client.createJob({
      input_path: st.source.path,
      profile,
      overwrite: true,
      ...(output ? { output_path: output } : {}),
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
    failed(e, st.source.name, 'Could not start');
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
    reportFailure(classifyFailure(e, st.source?.name), 'Cancel failed');
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
  // Re-selecting the run already on screen is a no-op — except when its files
  // were found missing, where it is the one gesture that means "look again".
  // A file that came back (a volume remounted, an undo in Finder) otherwise
  // needs the user to leave the run and return to it for no reason at all.
  const recheck = Boolean(st.artifacts && !st.artifacts.master);
  if (st.currentRunId === jobId && !recheck) {
    // Not a no-op any more. Re-picking the row you are on is the gesture that
    // means "look again", and until this line it was the one gesture that did
    // not: the files of the run *on screen* were verified once, when it was
    // opened, and never after — so deleting the master under a live run left
    // an enabled download link answering 404 with nothing on screen saying so.
    // One ranged byte per artefact, no `/api/analyze`, no re-decode.
    await reverifyCurrentRun();
    return;
  }
  const entry = st.history.find((h) => h.jobId === jobId);
  if (!entry) return;
  if (st.job && st.job.status && !isTerminal(st.job.status.state)) {
    st.setError('A job is still running — cancel it before opening another run.', 'Busy');
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
  // The profile is part of the run, not a preference that outlives it. Without
  // this a PRODUCTION row restored with the radiogroup still reading STUDIO —
  // the screen asserting both at once — and PROCESS AGAIN then ran studio,
  // which is not "again" at all.
  st.setProfile(entry.profile);
  st.setDeckFault(null);
  st.setArtifacts(null);
  st.resetView();

  const client = st.client;
  const doneRun = entry.outcome === 'done' && Boolean(entry.outputPath);

  // A7 · the swap is atomic-or-labelled, and it is settled HERE — before the
  // first engine round-trip — because everything below this line can hang or
  // die with the engine. Two halves:
  //
  // *Atomic*: the run's own facts (analyses, report, status) are session
  // memory, so the cached ones switch with the name. Before this, a restore
  // that stalled at its first request left run B's waveform, metrics and
  // status line standing under run A's name for as long as the stall lasted.
  // The post-verification code below re-states these, amending the cleaned
  // half if the master fails verification.
  st.setOriginal(entry.original);
  st.setCleaned(doneRun ? entry.cleaned : null, doneRun ? entry.outputPath : null);
  st.setStatus(runStatusLine(entry));
  // *Labelled*: a deck keeps its audio only if it is already this run's file;
  // otherwise it goes silent and unclaimed now, not after the engine answers.
  // The measured failure: engine hung mid-restore → both decks still held the
  // previous run's 12 s blobs under a 94.6 s run's name, A/B lit CLEANED.
  getPlayer().claimOnly('original', client ? client.fileUrl(entry.inputPath) : null);
  getPlayer().claimOnly(
    'cleaned',
    client && doneRun ? client.fileUrl(entry.outputPath) : null,
  );

  // B5 · a cached analysis is not proof that the file behind it is still
  // there. Before this check a restore reported "RESULT Complete 5/5" with an
  // enabled Master WAV link that 404s, because `entry.cleaned` was present and
  // nothing ever asked the engine about the *file*. One ranged byte per
  // artefact answers that (a HEAD does not — see `verifyArtifacts`) — no
  // decode, no `/api/analyze`, so the zero-analyze property of a restore is
  // untouched.
  const verified = await verifyArtifacts(client, entry);
  if (getState().currentRunId !== entry.jobId) return; // the user moved on
  const avail = verified?.art ?? null;
  if (verified && avail) {
    getState().setArtifacts(avail);
    getState().patchHistory(entry.jobId, {
      artifacts: avail,
      ...(avail.master && verified.masterBytes !== null
        ? { masterBytes: verified.masterBytes }
        : {}),
    });
  }
  // A run whose master is gone is not a run with a cleaned deck. Everything
  // else about it — its report, its numbers, its units — is still true.
  const restored = doneRun && (avail?.master ?? true);

  // Only the missing halves are re-fetched.
  let original = entry.original;
  let cleaned = entry.cleaned;
  const needsOriginal = !original;
  const needsCleaned = restored && !cleaned;
  if (client && (needsOriginal || needsCleaned)) {
    getState().setAnalyzing(true);
    getState().setStatus(`Re-reading ${entry.inputName}`);
    // C5 · a re-read that fails is not a silent nothing. The commonest cause
    // is that the file has been moved or deleted since the run — which is a
    // fact the user needs, not one to swallow into a `null`.
    let reReadFailure: UiFailure | null = null;
    const reRead = async (path: string, name: string): Promise<AudioAnalysis | null> => {
      try {
        return await client.analyze(path, envelopeBuckets());
      } catch (e) {
        reReadFailure ??= classifyFailure(e, name);
        return null;
      }
    };
    if (needsOriginal) original = await reRead(entry.inputPath, entry.inputName);
    if (needsCleaned) cleaned = await reRead(entry.outputPath, baseName(entry.outputPath));
    getState().setAnalyzing(false);
    if (reReadFailure) reportFailure(reReadFailure, 'Could not re-read this run');
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
  // The path is kept even when the master is gone — the artefact row still has
  // to be able to name the file it cannot hand over — but the analysis is not,
  // because that is what puts a cleaned deck on the waveform and keeps it
  // asking `/api/peaks` for a path that answers 404.
  cur.setCleaned(restored ? cleaned : null, doneRun ? entry.outputPath : null);
  if (client) {
    player.load('original', client.fileUrl(entry.inputPath), original?.duration_s ?? null);
    if (restored) {
      player.load(
        'cleaned',
        client.fileUrl(entry.outputPath),
        cleaned?.duration_s ?? original?.duration_s ?? null,
      );
      player.setActive('cleaned');
      cur.setAbMode('cleaned');
    } else {
      player.load('cleaned', null);
      player.setActive('original');
      cur.setAbMode('original');
    }
  }
  // A run that finished but has no master left is not an error — the pass
  // really did happen — so it is stated where the missing deck is, not in the
  // red bar reserved for things that went wrong just now.
  if (doneRun && !restored && avail) {
    cur.setDeckFault({
      deck: 'cleaned',
      headline: 'CLEANED DECK UNAVAILABLE',
      detail: deckGoneDetail(avail),
    });
  }
  // A re-read failure has already claimed the status line with the reason;
  // do not paper over it with the run's happy summary.
  if (cur.error) return;
  cur.setStatus(doneRun && !restored && avail ? avail.reason : runStatusLine(entry));
}

/**
 * The sentence the transport plate carries when a restore finds the master
 * gone.
 *
 * It is deliberately *not* `ArtifactState.reason`, which is the paragraph the
 * download row and the run list need — it has to explain all three files and
 * where the report on screen came from. Put under the A/B switch that
 * paragraph was five wrapped lines of plate inside a fixed-height column, and
 * the plate is not the place for the long version: this control's question is
 * only "what can I listen to". The long answer is still one hover away on the
 * artefact row, and on the run's own row.
 */
function deckGoneDetail(avail: ArtifactState): string {
  if (!avail.master) {
    // The flag carries *which way* the master failed verification; the plate's
    // sentence has to agree with it, or the screen says "no longer on disk"
    // about a file that is sitting right there truncated.
    if (avail.flag === 'FILE BROKEN') {
      return 'The cleaned master on disk is not the file this run wrote — it has been truncated or rewritten — so there is no cleaned deck; the A/B is playing the original.';
    }
    if (avail.flag === 'NO ACCESS') {
      return 'The cleaned master cannot be read where it stands, so there is no cleaned deck — the A/B is playing the original.';
    }
    return 'The cleaned master is no longer on disk, so there is no cleaned deck — the A/B is playing the original.';
  }
  return avail.reason;
}

/** Bytes as the condemnation sentence needs them: exact, with separators. */
function fmtBytes(n: number | null): string {
  return n === null ? 'an unknown length' : `${n.toLocaleString('en-US')} B`;
}

/** What one one-byte probe of one artefact actually established. */
type ArtifactVerdict =
  | { state: 'ok'; size: number | null }
  | { state: 'gone' } // the engine answered 404
  | { state: 'unreadable' }; // the engine committed a 2xx and could not produce the byte

/**
 * One artefact, really read. A 2xx whose byte never arrives is ambiguous —
 * a file the engine lists but cannot open (chmod 000, a volume gone under
 * it), or an engine dying between headers and body — and one more byte-sized
 * question separates them: a second answer means the file, no answer at all
 * means the engine (the throw lands in `verifyArtifacts`'s outage catch).
 * Same rule as the player's own deck probe.
 */
async function probeArtifact(client: EngineClient, path: string): Promise<ArtifactVerdict> {
  const once = async (): Promise<ArtifactVerdict | null> => {
    const p = await client.verify(path);
    if (p.status === 404) return { state: 'gone' };
    if (p.status >= 400) {
      // Any other refusal is the engine having an opinion about the request,
      // not evidence about the file; treated like the outage case.
      throw new EngineError(p.status, `http_${p.status}`, 'artefact probe refused');
    }
    return p.delivered ? { state: 'ok', size: p.size } : null;
  };
  return (await once()) ?? (await once()) ?? { state: 'unreadable' };
}

/**
 * B5/A7 · which of a finished run's three files the engine can still serve —
 * established by *reading* them, not by asking whether they exist.
 *
 * A HEAD answers 200 for a master that has been `chmod 000`ed and for one
 * truncated to 100 bytes, so the old three-HEAD check waved through exactly
 * the two attacks it was built against. Each artefact now costs one ranged
 * byte (`EngineClient.verify`), and the master's full length is additionally
 * measured against the size this run recorded when its master first loaded
 * (`HistoryEntry.masterBytes`). Nothing here calls `/api/analyze`, so
 * restoring a run stays free.
 *
 * Returns null when no answer could be had — no engine, or a run that never
 * produced anything. Silence beats a false accusation, and an outage is
 * already explained everywhere else on the screen.
 */
async function verifyArtifacts(
  client: EngineClient | null,
  entry: HistoryEntry,
): Promise<{ art: ArtifactState; masterBytes: number | null } | null> {
  if (entry.outcome !== 'done' || !entry.outputPath) return null;
  // A superseded run's files are not its own: a later run wrote over them, so
  // its report and the bytes behind its links describe two different passes.
  // That answer costs no request at all.
  if (entry.supersededBy) {
    return {
      art: {
        master: false,
        json: false,
        txt: false,
        reason:
          'A later run wrote over this run’s files — what is on disk now belongs to that run, not to this report.',
      },
      masterBytes: null,
    };
  }
  if (!client) return null;
  const jsonPath = entry.reportPath || '';
  const txtPath = jsonPath ? reportTxtPath(jsonPath) : '';
  try {
    const [m, j, t] = await Promise.all([
      probeArtifact(client, entry.outputPath),
      jsonPath ? probeArtifact(client, jsonPath) : Promise.resolve<ArtifactVerdict>({ state: 'gone' }),
      txtPath ? probeArtifact(client, txtPath) : Promise.resolve<ArtifactVerdict>({ state: 'gone' }),
    ]);
    const expected = entry.masterBytes ?? null;
    const sizeOk =
      m.state !== 'ok' || expected === null || m.size === null || m.size === expected;
    const master = m.state === 'ok' && sizeOk;
    const json = j.state === 'ok';
    const txt = t.state === 'ok';
    const masterBytes = master ? m.size : null;
    if (master && json && txt) {
      return { art: { master, json, txt, reason: '' }, masterBytes };
    }
    // A master that is present-but-wrong gets its own words: "gone" would send
    // the user looking for a file that is right where they left it.
    if (m.state === 'ok' && !sizeOk) {
      return {
        art: {
          master: false,
          json,
          txt,
          reason: `The cleaned master on disk is ${fmtBytes(m.size)} where this run wrote ${fmtBytes(
            expected,
          )} — it has been truncated or rewritten, so it is not this run’s master any more. The report on screen is this session’s own copy of the run.`,
          flag: 'FILE BROKEN',
        },
        masterBytes: null,
      };
    }
    if (m.state === 'unreadable') {
      return {
        art: {
          master: false,
          json,
          txt,
          reason:
            'The cleaned master is still listed, but the engine cannot read a byte of it — the file is unreadable where it stands. The report on screen is this session’s own copy of the run.',
          flag: 'NO ACCESS',
        },
        masterBytes: null,
      };
    }
    const gone = [
      m.state !== 'ok' ? 'the cleaned master' : null,
      !json ? 'the JSON report' : null,
      !txt ? 'the .txt summary' : null,
    ].filter((x): x is string => x !== null);
    const list =
      gone.length > 1 ? `${gone.slice(0, -1).join(', ')} and ${gone[gone.length - 1]}` : gone[0];
    return {
      art: {
        master,
        json,
        txt,
        reason: `This run’s files are not all where it left them — ${list} ${
          gone.length > 1 ? 'are' : 'is'
        } no longer on disk. The report on screen is this session’s own copy of the run.`,
      },
      masterBytes,
    };
  } catch {
    // The engine did not answer at all. That is an outage, not a deletion.
    return null;
  }
}

function runStatusLine(e: HistoryEntry): string {
  if (e.outcome === 'failed') return `Failed · ${e.inputName}${e.error ? ` — ${e.error}` : ''}`;
  if (e.outcome === 'cancelled') return `Cancelled · ${e.inputName}`;
  const units =
    e.enhanced !== null && e.unitsTotal !== null
      ? ` · ${unitsEnhanced(e.enhanced, e.unitsTotal)}`
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
    parts.push(unitsEnhanced(e.enhanced, e.unitsTotal));
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
  // Not an engine failure in any sense — the permission is the browser's, and
  // the bar used to publish it as ENGINE ERROR.
  if (!ok) getState().setError('The browser refused clipboard access.', 'Clipboard blocked');
  return ok;
}

export interface ArtifactLink {
  /** null when this file cannot be handed over; `note` then says why. */
  url: string | null;
  name: string;
  /** The explanation a disabled link carries, or null when it is live. */
  note: string | null;
}

export interface Artifacts {
  master: ArtifactLink;
  json: ArtifactLink;
  txt: ArtifactLink;
}

/**
 * The three files a finished run leaves behind. `/api/audio` is the engine's
 * one file-serving route and it types each response from the file's own
 * extension, so the JSON report and the .txt sidecar come down it exactly like
 * the master does.
 *
 * `avail` is what the engine last said about those files (state/store.ts,
 * `ArtifactState`). A link whose file is gone loses its `href` and keeps its
 * name and its reason: a button that downloads a 404 is worse than a button
 * that says why it cannot.
 */
export function artifactsFor(
  outputPath: string | null,
  reportPath: string | null,
  avail?: ArtifactState | null,
): Artifacts | null {
  const client = getState().client;
  if (!client || !outputPath) return null;
  const json = reportPath || '';
  const txt = json ? reportTxtPath(json) : '';
  if (!json) return null;
  const link = (path: string, ok: boolean): ArtifactLink => ({
    url: ok ? client.fileUrl(path) : null,
    name: baseName(path),
    note: ok ? null : (avail?.reason || 'This file is no longer where the run left it.'),
  });
  return {
    master: link(outputPath, avail?.master ?? true),
    json: link(json, avail?.json ?? true),
    txt: link(txt, avail?.txt ?? true),
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
    reportFailure(classifyFailure(e, st.cleanedPath ? baseName(st.cleanedPath) : undefined), 'Import failed');
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
    reportFailure(classifyFailure(e, st.source?.name), 'Replace failed');
  }
}

export async function revealOutput(): Promise<void> {
  const bridge = getBridge();
  const st = getState();
  if (!st.cleanedPath) return;
  try {
    await bridge.files.revealInFinder(st.cleanedPath);
  } catch (e) {
    reportFailure(classifyFailure(e, st.cleanedPath ? baseName(st.cleanedPath) : undefined), 'Reveal failed');
  }
}

// ---------------------------------------------------------------------------
// Playback glue

export function setAb(mode: 'original' | 'cleaned'): void {
  const player = getPlayer();
  // A deck that is out of service cannot be switched to, and asking for one is
  // not an error — it is a control the screen has already greyed out.
  if (!player.hasDeck(mode)) return;
  player.setActive(mode);
  getState().setAbMode(mode);
}

export function togglePlay(): void {
  getPlayer().toggle();
}

export function seekTo(time: number): void {
  getPlayer().seek(time);
}
