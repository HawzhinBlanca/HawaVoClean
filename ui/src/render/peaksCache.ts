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
const inflight = new Map<string, Promise<PeaksData>>();

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
  if (running) return running;

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
  inflight.set(key, task);
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
  } finally {
    inflight.delete(key);
  }
}
