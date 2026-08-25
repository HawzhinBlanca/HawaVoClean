// Windowed-peaks fetching for waveform zoom: LRU cache keyed by
// (path, start, end, buckets), in-flight de-duplication, and a one-shot
// capability probe so an engine without `POST /api/peaks` degrades to the
// whole-file envelope instead of retrying forever.

import { EngineError, type EngineClient } from '../api/client';

export interface PeaksData {
  min: Float32Array;
  max: Float32Array;
  /** Linear amplitude 0..1, already converted from dBFS. */
  rms: Float32Array;
  start: number;
  end: number;
  buckets: number;
  samplesPerBucket: number;
  sampleRate: number;
}

const LRU_LIMIT = 24;
/** Requests are rounded to this many decimals so float jitter still hits the cache. */
const TIME_DECIMALS = 6;

const cache = new Map<string, PeaksData>();
/**
 * Requests in flight, so two views asking for the same window share one fetch.
 *
 * The signal is stored with the task because the sharing was unsound without
 * it: the entry held only the promise, created with the *first* caller's
 * AbortSignal. A second caller joining that entry was therefore joining a
 * request that the first caller could cancel out from under it — zoom in,
 * zoom straight back out, and the live request dies because the abandoned one
 * was aborted. A joiner now checks the owner's signal and starts its own fetch
 * rather than inheriting someone else's cancellation.
 */
interface Inflight {
  task: Promise<PeaksData>;
  signal: AbortSignal | undefined;
}

const inflight = new Map<string, Inflight>();

let supported = true;

export function peaksSupported(): boolean {
  return supported;
}

/** Test seam / clip switch: drop everything we remember. */
export function clearPeaksCache(): void {
  cache.clear();
  inflight.clear();
  // Also re-arm the capability probe. The latch exists so one 404 does not make
  // every zoom re-ask an engine that cannot answer — but the cache is cleared when
  // we point at a different engine, and that one may well support the route.
  supported = true;
}

export function roundTime(t: number): number {
  return Number(t.toFixed(TIME_DECIMALS));
}

export function peaksKey(path: string, startS: number, endS: number, buckets: number): string {
  return `${path}|${roundTime(startS)}|${roundTime(endS)}|${buckets}`;
}

function touch(key: string, value: PeaksData): void {
  cache.delete(key);
  cache.set(key, value);
  while (cache.size > LRU_LIMIT) {
    const oldest = cache.keys().next();
    if (oldest.done) break;
    cache.delete(oldest.value);
  }
}

export function peekPeaks(
  path: string,
  startS: number,
  endS: number,
  buckets: number,
): PeaksData | null {
  const key = peaksKey(path, startS, endS, buckets);
  const hit = cache.get(key);
  if (!hit) return null;
  touch(key, hit);
  return hit;
}

function toF32(a: readonly number[]): Float32Array {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] ?? 0;
  return out;
}

function rmsToLinear(db: readonly number[]): Float32Array {
  const out = new Float32Array(db.length);
  for (let i = 0; i < db.length; i++) {
    const d = db[i] ?? -120;
    out[i] = d <= -120 ? 0 : Math.min(1, 10 ** (d / 20));
  }
  return out;
}

/**
 * Fetch (or reuse) the peaks for one window. Resolves to `null` when the engine
 * has no `/api/peaks` route — callers keep drawing the whole-file envelope.
 * Aborted requests reject with the usual `AbortError`.
 */
export async function loadPeaks(
  client: EngineClient,
  path: string,
  startS: number,
  endS: number,
  buckets: number,
  signal?: AbortSignal,
): Promise<PeaksData | null> {
  if (!supported) return null;
  const s = roundTime(startS);
  const e = roundTime(endS);
  if (!(e > s) || !(buckets > 0)) return null;
  const key = peaksKey(path, s, e, buckets);
  const hit = cache.get(key);
  if (hit) {
    touch(key, hit);
    return hit;
  }
  const running = inflight.get(key);
  // Only join a request that can still finish. An aborted owner's promise is
  // already rejecting.
  if (running && !running.signal?.aborted) return settle(running.task);

  const task = (async (): Promise<PeaksData> => {
    const res = await client.peaks(path, s, e, buckets, signal);
    const data: PeaksData = {
      min: toF32(res.peaks.min),
      max: toF32(res.peaks.max),
      rms: rmsToLinear(res.rms_db),
      start: res.start_s,
      end: res.end_s,
      buckets: res.peaks.min.length,
      samplesPerBucket: res.samples_per_bucket,
      sampleRate: res.sample_rate,
    };
    touch(key, data);
    return data;
  })();
  const entry: Inflight = { task, signal };
  inflight.set(key, entry);
  try {
    return await settle(task);
  } finally {
    // Identity-checked. A bare `delete(key)` lets a dying request evict the
    // entry a *newer* request has already installed under the same key, so the
    // next caller re-fetches a window that is on its way.
    if (inflight.get(key) === entry) inflight.delete(key);
  }
}

/**
 * Await a peaks request and apply the capability latch.
 *
 * Owner and joiner both go through here. They used to differ: a joiner
 * returned the raw promise, so a 404/405 — the engine saying it has no
 * windowed-peaks route at all — reached it as a thrown EngineError instead of
 * the `null` the owner gets, and did not set the latch that stops the app
 * asking again.
 */
async function settle(task: Promise<PeaksData>): Promise<PeaksData | null> {
  try {
    return await task;
  } catch (err) {
    // A missing route is a capability answer, not a failure to report: stop
    // asking so the flow stays free of repeated 404s.
    if (err instanceof EngineError && (err.status === 404 || err.status === 405)) {
      supported = false;
      return null;
    }
    throw err;
  }
}
