// render/viewWindow.ts — the visible waveform window. Zoom/pan run outside
// React at pointer rate, so this controller is where "the window never lies"
// has to be true: clamped to the file, pivot-preserving, and floored at one
// sample per bucket.

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { waveView } from './viewWindow';

const SR = 48000;
const BUCKETS = 1200;
/** minSpan for a 100 s clip at 1200 buckets / 48 kHz. */
const MIN_SPAN = BUCKETS / SR; // 0.025 s

beforeEach(() => {
  waveView.setSource(100, SR, BUCKETS);
});

describe('setSource', () => {
  it('opens on the whole file and derives the deepest zoom from the display', () => {
    expect(waveView.view).toEqual({ start: 0, end: 100 });
    expect(waveView.duration).toBe(100);
    expect(waveView.minSpan).toBeCloseTo(MIN_SPAN, 12);
    expect(waveView.isFull).toBe(true);
    expect(waveView.factor).toBe(1);
  });

  it('never lets minSpan exceed the file itself', () => {
    waveView.setSource(0.01, SR, BUCKETS); // clip shorter than 1200 samples
    expect(waveView.minSpan).toBeCloseTo(0.01, 12);
    expect(waveView.view).toEqual({ start: 0, end: 0.01 });
  });

  it('holds an absolute floor so a rubbish sample rate cannot collapse it', () => {
    waveView.setSource(100, 0, 0);
    expect(waveView.minSpan).toBeGreaterThanOrEqual(1e-4);
    waveView.setSource(1e-9, SR, BUCKETS);
    expect(waveView.minSpan).toBe(1e-4);
  });

  it('treats a negative duration as no clip', () => {
    waveView.setSource(-3, SR, BUCKETS);
    expect(waveView.duration).toBe(0);
    expect(waveView.view).toEqual({ start: 0, end: 0 });
  });
});

describe('set', () => {
  it('clamps the window inside the file', () => {
    waveView.set(-10, 40);
    expect(waveView.view).toEqual({ start: 0, end: 50 });
    waveView.set(80, 140);
    expect(waveView.view).toEqual({ start: 40, end: 100 });
  });

  it('clamps the span to the file length rather than showing empty air', () => {
    waveView.set(-50, 500);
    expect(waveView.view).toEqual({ start: 0, end: 100 });
  });

  it('refuses to go below minSpan', () => {
    waveView.set(50, 50 + MIN_SPAN / 10);
    expect(waveView.span).toBeCloseTo(MIN_SPAN, 12);
    expect(waveView.isMaxZoom).toBe(true);
  });

  it('reads an empty or non-finite span as "the whole file"', () => {
    waveView.set(10, 10);
    expect(waveView.view).toEqual({ start: 0, end: 100 });
    waveView.set(10, 5);
    expect(waveView.view).toEqual({ start: 0, end: 100 });
    waveView.set(Number.NaN, Number.NaN);
    expect(waveView.view).toEqual({ start: 0, end: 100 });
  });

  it('does nothing at all with no clip loaded', () => {
    waveView.clear();
    const seen = vi.fn();
    const off = waveView.subscribe(seen);
    waveView.set(1, 2);
    expect(waveView.view).toEqual({ start: 0, end: 0 });
    expect(seen).not.toHaveBeenCalled();
    off();
  });
});

describe('zoomAt', () => {
  it('keeps the time under the cursor under the cursor', () => {
    waveView.zoomAt(25, 2);
    expect(waveView.view).toEqual({ start: 12.5, end: 62.5 });
    // The pivot sat 25% into the window before and sits 25% into it after.
    expect((25 - waveView.start_s) / waveView.span).toBeCloseTo(0.25, 12);
  });

  it('holds the pivot fraction across a long zoom-in burst', () => {
    const pivot = 25;
    const frac0 = (pivot - waveView.start_s) / waveView.span;
    for (let i = 0; i < 12; i++) {
      waveView.zoomAt(pivot, 1.5);
      expect((pivot - waveView.start_s) / waveView.span).toBeCloseTo(frac0, 9);
    }
  });

  it('floors at one bucket per sample and then stops moving', () => {
    for (let i = 0; i < 200; i++) waveView.zoomAt(50, 2);
    expect(waveView.span).toBeCloseTo(MIN_SPAN, 12);
    expect(waveView.isMaxZoom).toBe(true);
    const before = waveView.view;
    const seen = vi.fn();
    const off = waveView.subscribe(seen);
    waveView.zoomAt(50, 2);
    expect(waveView.view).toEqual(before);
    expect(seen).not.toHaveBeenCalled();
    off();
  });

  it('zooms out to the whole file and no further', () => {
    waveView.set(40, 60);
    for (let i = 0; i < 20; i++) waveView.zoomAt(50, 0.5);
    expect(waveView.view).toEqual({ start: 0, end: 100 });
    expect(waveView.isFull).toBe(true);
  });

  it('pins the window at the edges when the pivot is at an edge', () => {
    waveView.zoomAt(0, 4);
    expect(waveView.start_s).toBe(0);
    expect(waveView.span).toBeCloseTo(25, 12);
    waveView.reset();
    waveView.zoomAt(100, 4);
    expect(waveView.end_s).toBe(100);
  });

  it('clamps a pivot outside the window to its edge instead of jumping', () => {
    waveView.set(40, 60);
    waveView.zoomAt(0, 2); // pivot left of the window
    expect(waveView.view).toEqual({ start: 40, end: 50 });
  });

  it('is inert without a clip or with a nonsense factor', () => {
    waveView.zoomAt(50, 0);
    expect(waveView.view).toEqual({ start: 0, end: 100 });
    waveView.zoomAt(50, -2);
    expect(waveView.view).toEqual({ start: 0, end: 100 });
    waveView.clear();
    waveView.zoomAt(1, 2);
    expect(waveView.view).toEqual({ start: 0, end: 0 });
  });
});

describe('centerOn / panBy / reset', () => {
  it('centres without changing the zoom', () => {
    waveView.set(10, 20);
    waveView.centerOn(50);
    expect(waveView.view).toEqual({ start: 45, end: 55 });
    expect(waveView.span).toBe(10);
  });

  it('clamps a centre near either end and keeps the span', () => {
    waveView.set(10, 20);
    waveView.centerOn(1);
    expect(waveView.view).toEqual({ start: 0, end: 10 });
    waveView.centerOn(99);
    expect(waveView.view).toEqual({ start: 90, end: 100 });
  });

  it('pans by a delta and stops at the ends', () => {
    waveView.set(10, 20);
    waveView.panBy(5);
    expect(waveView.view).toEqual({ start: 15, end: 25 });
    waveView.panBy(-1000);
    expect(waveView.view).toEqual({ start: 0, end: 10 });
    waveView.panBy(1000);
    expect(waveView.view).toEqual({ start: 90, end: 100 });
  });

  it('reset opens the whole file again', () => {
    waveView.set(10, 20);
    waveView.reset();
    expect(waveView.view).toEqual({ start: 0, end: 100 });
    expect(waveView.factor).toBe(1);
  });

  it('factor says how far in we are', () => {
    waveView.set(10, 20);
    expect(waveView.factor).toBeCloseTo(10, 12);
  });
});

describe('setMaxBuckets', () => {
  it('records the display width before any clip exists', () => {
    waveView.clear();
    waveView.setMaxBuckets(SR, 3000);
    expect(waveView.maxBuckets).toBe(3000);
    expect(waveView.duration).toBe(0);
  });

  it('deepens the zoom floor when the display gets wider', () => {
    waveView.setMaxBuckets(SR, 2400);
    expect(waveView.minSpan).toBeCloseTo(2400 / SR, 12);
  });

  it('opens a window that a narrower display has made illegal', () => {
    waveView.set(50, 50 + MIN_SPAN); // at the floor for 1200 buckets
    waveView.setMaxBuckets(SR, 4800); // display got wider: floor is now 0.1 s
    expect(waveView.span).toBeCloseTo(4800 / SR, 12);
    expect(waveView.start_s).toBeCloseTo(50, 12);
  });

  it('falls back to the default resolution for a nonsense width', () => {
    waveView.setMaxBuckets(SR, 0);
    expect(waveView.maxBuckets).toBe(1200);
  });
});

describe('subscribe', () => {
  it('notifies on a real change only, and stops after unsubscribe', () => {
    const seen = vi.fn();
    const off = waveView.subscribe(seen);
    waveView.set(10, 20);
    expect(seen).toHaveBeenCalledTimes(1);
    expect(seen).toHaveBeenLastCalledWith({ start: 10, end: 20 });
    waveView.set(10, 20); // same window
    expect(seen).toHaveBeenCalledTimes(1);
    off();
    waveView.set(30, 40);
    expect(seen).toHaveBeenCalledTimes(1);
  });

  it('emits on clear so every consumer drops the old clip', () => {
    const seen = vi.fn();
    const off = waveView.subscribe(seen);
    waveView.clear();
    expect(seen).toHaveBeenCalledWith({ start: 0, end: 0 });
    off();
  });
});
