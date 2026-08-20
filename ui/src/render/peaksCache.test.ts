// render/peaksCache.ts — the windowed-peaks cache behind waveform zoom. Its
// three jobs are all invisible on screen and all expensive to get wrong: hit
// the cache despite float jitter, never ask the engine the same question
// twice at once, and stop asking an engine that has no /api/peaks route.

import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { EngineClient } from '../api/client';
import type { PeaksWindow } from '../api/types';

type Mod = typeof import('./peaksCache');
type ClientMod = typeof import('../api/client');

/**
 * Fresh module per test: `supported` is a one-way latch by design. The client
 * module has to come from the same fresh graph, or the `EngineError` thrown by
 * the stub is a different class from the one `loadPeaks` tests with
 * `instanceof` and the 404 latch would look broken.
 */
async function freshModules(): Promise<{ mod: Mod; EngineError: ClientMod['EngineError'] }> {
  vi.resetModules();
  const mod = await import('./peaksCache');
  const { EngineError } = await import('../api/client');
  return { mod, EngineError };
}

function window_(startS: number, endS: number, buckets = 4): PeaksWindow {
  return {
    path: '/clip.wav',
    start_s: startS,
    end_s: endS,
    sample_rate: 48000,
    channels: 1,
    duration_s: 100,
    samples_per_bucket: 64,
    peaks: {
      min: Array.from({ length: buckets }, () => -0.5),
      max: Array.from({ length: buckets }, () => 0.5),
    },
    rms_db: [-6, -120, -200, 0].slice(0, buckets),
  };
}

interface Stub {
  client: EngineClient;
  peaks: ReturnType<typeof vi.fn>;
}

function stubClient(impl?: (s: number, e: number) => Promise<PeaksWindow>): Stub {
  const peaks = vi.fn(async (_p: string, s: number, e: number) => (impl ? impl(s, e) : window_(s, e)));
  return { client: { peaks } as unknown as EngineClient, peaks };
}

let mod: Mod;
let EngineError: ClientMod['EngineError'];
beforeEach(async () => {
  ({ mod, EngineError } = await freshModules());
});

describe('key rounding', () => {
  it('rounds times to 6 decimals so wheel jitter still hits one key', () => {
    expect(mod.roundTime(1.00000049)).toBe(1);
    expect(mod.peaksKey('/a.wav', 1.0000004, 2.0000004, 800)).toBe(
      mod.peaksKey('/a.wav', 1.0000001, 2.0000002, 800),
    );
  });

  it('keeps genuinely different windows apart', () => {
    expect(mod.peaksKey('/a.wav', 1, 2, 800)).not.toBe(mod.peaksKey('/a.wav', 1, 2.1, 800));
    expect(mod.peaksKey('/a.wav', 1, 2, 800)).not.toBe(mod.peaksKey('/b.wav', 1, 2, 800));
    expect(mod.peaksKey('/a.wav', 1, 2, 800)).not.toBe(mod.peaksKey('/a.wav', 1, 2, 801));
  });

  it('a jittered re-request is served from cache, not from the engine', async () => {
    const { client, peaks } = stubClient();
    await mod.loadPeaks(client, '/a.wav', 1.0000001, 2.0000001, 800);
    await mod.loadPeaks(client, '/a.wav', 1.00000012, 2.00000013, 800);
    expect(peaks).toHaveBeenCalledTimes(1);
  });
});

describe('payload conversion', () => {
  it('hands back typed arrays and linear RMS, with -120 dB as silence', async () => {
    const { client } = stubClient();
    const data = await mod.loadPeaks(client, '/a.wav', 0, 1, 4);
    expect(data).not.toBeNull();
    expect(data?.min).toBeInstanceOf(Float32Array);
    expect(data?.rms[0]).toBeCloseTo(10 ** (-6 / 20), 6);
    expect(data?.rms[1]).toBe(0); // -120 dB
    expect(data?.rms[2]).toBe(0); // below the floor
    expect(data?.rms[3]).toBe(1); // 0 dBFS, clamped to unity
    expect(data?.buckets).toBe(4);
  });
});

describe('argument guards', () => {
  it('asks nothing for an empty, inverted or bucket-less window', async () => {
    const { client, peaks } = stubClient();
    expect(await mod.loadPeaks(client, '/a.wav', 5, 5, 800)).toBeNull();
    expect(await mod.loadPeaks(client, '/a.wav', 9, 3, 800)).toBeNull();
    expect(await mod.loadPeaks(client, '/a.wav', 0, 1, 0)).toBeNull();
    expect(peaks).not.toHaveBeenCalled();
  });
});

describe('in-flight de-duplication', () => {
  it('collapses concurrent requests for the same window into one call', async () => {
    let release!: () => void;
    const gate = new Promise<void>((r) => {
      release = r;
    });
    const { client, peaks } = stubClient(async (s, e) => {
      await gate;
      return window_(s, e);
    });
    const a = mod.loadPeaks(client, '/a.wav', 0, 1, 800);
    const b = mod.loadPeaks(client, '/a.wav', 0, 1, 800);
    release();
    const [ra, rb] = await Promise.all([a, b]);
    expect(peaks).toHaveBeenCalledTimes(1);
    expect(ra).toBe(rb); // the identical object, not a copy
  });

  it('lets a different window through at the same time', async () => {
    const { client, peaks } = stubClient();
    await Promise.all([
      mod.loadPeaks(client, '/a.wav', 0, 1, 800),
      mod.loadPeaks(client, '/a.wav', 1, 2, 800),
    ]);
    expect(peaks).toHaveBeenCalledTimes(2);
  });

  it('clears the in-flight slot after a failure so the next drag retries', async () => {
    const boom = new EngineError(500, 'engine_fault', 'kaboom');
    const peaks = vi
      .fn()
      .mockRejectedValueOnce(boom)
      .mockImplementation(async (_p: string, s: number, e: number) => window_(s, e));
    const client = { peaks } as unknown as EngineClient;
    await expect(mod.loadPeaks(client, '/a.wav', 0, 1, 800)).rejects.toBe(boom);
    await expect(mod.loadPeaks(client, '/a.wav', 0, 1, 800)).resolves.not.toBeNull();
    expect(peaks).toHaveBeenCalledTimes(2);
  });
});

describe('LRU', () => {
  it('keeps 24 windows and evicts the least recently used', async () => {
    const { client } = stubClient();
    for (let i = 0; i < 24; i++) await mod.loadPeaks(client, '/a.wav', i, i + 1, 800);
    await mod.loadPeaks(client, '/a.wav', 100, 101, 800); // 25th
    expect(mod.peekPeaks('/a.wav', 0, 1, 800)).toBeNull(); // the oldest went
    expect(mod.peekPeaks('/a.wav', 1, 2, 800)).not.toBeNull();
    expect(mod.peekPeaks('/a.wav', 100, 101, 800)).not.toBeNull();
  });

  it('a read counts as use: the touched window survives the next eviction', async () => {
    const { client } = stubClient();
    for (let i = 0; i < 24; i++) await mod.loadPeaks(client, '/a.wav', i, i + 1, 800);
    expect(mod.peekPeaks('/a.wav', 0, 1, 800)).not.toBeNull(); // touch the oldest
    await mod.loadPeaks(client, '/a.wav', 100, 101, 800);
    expect(mod.peekPeaks('/a.wav', 0, 1, 800)).not.toBeNull();
    expect(mod.peekPeaks('/a.wav', 1, 2, 800)).toBeNull(); // it went instead
  });

  it('a cache hit re-fetches nothing', async () => {
    const { client, peaks } = stubClient();
    await mod.loadPeaks(client, '/a.wav', 0, 1, 800);
    await mod.loadPeaks(client, '/a.wav', 0, 1, 800);
    await mod.loadPeaks(client, '/a.wav', 0, 1, 800);
    expect(peaks).toHaveBeenCalledTimes(1);
  });

  it('clearPeaksCache drops everything (clip switch)', async () => {
    const { client, peaks } = stubClient();
    await mod.loadPeaks(client, '/a.wav', 0, 1, 800);
    mod.clearPeaksCache();
    expect(mod.peekPeaks('/a.wav', 0, 1, 800)).toBeNull();
    await mod.loadPeaks(client, '/a.wav', 0, 1, 800);
    expect(peaks).toHaveBeenCalledTimes(2);
  });
});

describe('capability probe', () => {
  it('stops asking an engine that has no /api/peaks route', async () => {
    const peaks = vi.fn().mockRejectedValue(new EngineError(404, 'not_found', 'no such route'));
    const client = { peaks } as unknown as EngineClient;
    expect(mod.peaksSupported()).toBe(true);
    expect(await mod.loadPeaks(client, '/a.wav', 0, 1, 800)).toBeNull();
    expect(mod.peaksSupported()).toBe(false);
    expect(await mod.loadPeaks(client, '/a.wav', 2, 3, 800)).toBeNull();
    expect(peaks).toHaveBeenCalledTimes(1); // never asked again
  });

  it('treats 405 the same way as 404', async () => {
    const peaks = vi.fn().mockRejectedValue(new EngineError(405, 'method', 'nope'));
    const client = { peaks } as unknown as EngineClient;
    expect(await mod.loadPeaks(client, '/a.wav', 0, 1, 800)).toBeNull();
    expect(mod.peaksSupported()).toBe(false);
  });

  // REGRESSION GUARD. The latch is one-way *within one engine*, but the cache is
  // cleared when the page is pointed at a different clip/engine — and that engine
  // may well have the route. An earlier build cleared the maps and left the latch
  // down, so a single 404 disabled windowed peaks for the life of the page.
  it('clearPeaksCache re-arms the latch so the next engine is probed again', async () => {
    const dead = vi.fn().mockRejectedValue(new EngineError(404, 'not_found', 'no such route'));
    const deadClient = { peaks: dead } as unknown as EngineClient;
    expect(await mod.loadPeaks(deadClient, '/a.wav', 0, 1, 800)).toBeNull();
    expect(mod.peaksSupported()).toBe(false);

    mod.clearPeaksCache();
    expect(mod.peaksSupported()).toBe(true);

    const { client, peaks } = stubClient();
    expect(await mod.loadPeaks(client, '/b.wav', 0, 1, 800)).not.toBeNull();
    expect(peaks).toHaveBeenCalledTimes(1); // it really asked, not just flipped a flag
  });

  it('does not disable itself on a server fault or an abort', async () => {
    const abort = new DOMException('Aborted', 'AbortError');
    const peaks = vi.fn().mockRejectedValue(abort);
    const client = { peaks } as unknown as EngineClient;
    await expect(mod.loadPeaks(client, '/a.wav', 0, 1, 800)).rejects.toBe(abort);
    expect(mod.peaksSupported()).toBe(true);
  });
});
