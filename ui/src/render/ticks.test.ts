// render/ticks.ts — the time-ruler maths. Verified in the browser as pixels;
// verified here as numbers, across the whole span range the ruler has to
// survive (a 40 ms window up to a 2 h file).

import { describe, expect, it } from 'vitest';
import {
  chooseStep,
  chooseSteps,
  formatSeconds,
  formatTime,
  formatTimeShort,
  tickLabel,
  timeTicks,
  timeTicksIn,
} from './ticks';

const WIDTH = 1000;

/** The spans the ruler is asked for, from deepest zoom to a long file. */
const SPANS = [0.004, 0.01, 0.05, 0.2, 1, 5, 30, 120, 600, 3600, 7200];

describe('chooseSteps', () => {
  it('gives a readable label count at every span from 4 ms to 2 h', () => {
    for (const span of SPANS) {
      const [major] = chooseSteps(span, WIDTH);
      const majors = span / major;
      expect(majors, `span ${span}s`).toBeGreaterThanOrEqual(4);
      expect(majors, `span ${span}s`).toBeLessThanOrEqual(20);
    }
  });

  it('never puts two labels closer than minPx', () => {
    for (const span of SPANS) {
      for (const minPx of [48, 72, 120]) {
        const [major] = chooseSteps(span, WIDTH, minPx);
        expect(major * (WIDTH / span), `span ${span} minPx ${minPx}`).toBeGreaterThanOrEqual(minPx);
      }
    }
  });

  it('picks the smallest ladder rung that fits (no needless coarseness)', () => {
    // One rung down must violate the minPx rule, or the ladder skipped a step.
    for (const span of SPANS) {
      const [major] = chooseSteps(span, WIDTH);
      const smaller = chooseSteps(span * 4, WIDTH)[0];
      expect(smaller).toBeGreaterThanOrEqual(major);
    }
  });

  it('is monotone: a wider span never gets a finer step', () => {
    let prev = 0;
    for (const span of SPANS) {
      const [major] = chooseSteps(span, WIDTH);
      expect(major, `span ${span}s`).toBeGreaterThanOrEqual(prev);
      prev = major;
    }
  });

  it('divides each major into a whole number of minors', () => {
    for (const span of SPANS) {
      const [major, minor] = chooseSteps(span, WIDTH);
      const ratio = major / minor;
      expect(Math.abs(ratio - Math.round(ratio)), `span ${span}s`).toBeLessThan(1e-9);
      expect(Math.round(ratio)).toBeGreaterThanOrEqual(2);
    }
  });

  it('falls back to 1 s for degenerate input rather than dividing by zero', () => {
    expect(chooseSteps(0, WIDTH)).toEqual([1, 0.2]);
    expect(chooseSteps(-5, WIDTH)).toEqual([1, 0.2]);
    expect(chooseSteps(10, 0)).toEqual([1, 0.2]);
    expect(chooseSteps(Number.NaN, WIDTH)).toEqual([1, 0.2]);
  });

  it('tops out at the last rung instead of running off the ladder', () => {
    expect(chooseSteps(1e9, WIDTH)).toEqual([21600, 3600]);
  });

  it('chooseStep is the major half of chooseSteps', () => {
    for (const span of SPANS) expect(chooseStep(span, WIDTH)).toBe(chooseSteps(span, WIDTH)[0]);
  });
});

describe('timeTicksIn', () => {
  it('lays minor ticks on exact multiples of the minor step, inside the window', () => {
    const [major, minor] = chooseSteps(20, WIDTH);
    const ticks = timeTicksIn(10, 30, WIDTH);
    expect(ticks.length).toBeGreaterThan(0);
    for (const t of ticks) {
      expect(t.time).toBeGreaterThanOrEqual(10 - minor);
      expect(t.time).toBeLessThanOrEqual(30 + minor);
      const k = t.time / minor;
      expect(Math.abs(k - Math.round(k))).toBeLessThan(1e-6);
      if (t.major) {
        const m = t.time / major;
        expect(Math.abs(m - Math.round(m))).toBeLessThan(1e-6);
      }
    }
  });

  it('flags exactly the multiples of the major step as major', () => {
    const ticks = timeTicksIn(0, 1, WIDTH); // step 0.1 / minor 0.02
    const majors = ticks.filter((t) => t.major).map((t) => Number(t.time.toFixed(3)));
    expect(majors).toEqual([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]);
    expect(ticks.length).toBe(51);
  });

  it('never emits a negative time', () => {
    const ticks = timeTicksIn(-5, 5, WIDTH);
    expect(ticks.length).toBeGreaterThan(0);
    expect(ticks.every((t) => t.time >= 0)).toBe(true);
  });

  it('returns nothing for an empty or inverted window', () => {
    expect(timeTicksIn(5, 5, WIDTH)).toEqual([]);
    expect(timeTicksIn(9, 3, WIDTH)).toEqual([]);
    expect(timeTicksIn(0, 10, 0)).toEqual([]);
  });

  it('refuses to flood: a pathological minPx yields no ticks, not 5 million', () => {
    expect(timeTicksIn(0, 100, WIDTH, 0.0001)).toEqual([]);
  });

  it('deep zoom still resolves: a 4 ms window gets its own ladder', () => {
    const ticks = timeTicksIn(1.0, 1.004, WIDTH);
    expect(ticks.length).toBeGreaterThanOrEqual(10);
    expect(ticks.filter((t) => t.major).length).toBeGreaterThanOrEqual(2);
  });

  it('timeTicks is timeTicksIn from zero', () => {
    expect(timeTicks(60, WIDTH)).toEqual(timeTicksIn(0, 60, WIDTH));
  });
});

describe('tickLabel', () => {
  it('uses m:ss above one second and drops the hour field on short files', () => {
    expect(tickLabel(0, 1)).toBe('0:00');
    expect(tickLabel(90, 1)).toBe('1:30');
    expect(tickLabel(605, 5, 900)).toBe('10:05');
  });

  it('gives every label an hour field once any label on the ruler needs one', () => {
    // Hour consistency: the same instant labels differently on a 10 min ruler
    // and on a 2 h ruler, and on the 2 h ruler *all* labels carry hours.
    expect(tickLabel(65, 5, 600)).toBe('1:05');
    expect(tickLabel(65, 5, 7200)).toBe('0:01:05');
    for (const t of timeTicks(7200, WIDTH).filter((x) => x.major)) {
      expect(tickLabel(t.time, chooseStep(7200, WIDTH), 7200).split(':')).toHaveLength(3);
    }
  });

  it('renormalises a rounded 59.6 s instead of printing :60', () => {
    expect(tickLabel(119.6, 1, 200)).toBe('2:00');
    expect(tickLabel(3599.6, 1, 7200)).toBe('1:00:00');
  });

  it('scales the fraction to the step: 1, 2, 3 then 4 decimals', () => {
    expect(tickLabel(65.2, 0.5)).toBe('1:05.2');
    expect(tickLabel(65.25, 0.05)).toBe('1:05.25');
    expect(tickLabel(65.125, 0.005)).toBe('1:05.125');
    expect(tickLabel(65.0625, 0.0005)).toBe('1:05.0625');
  });

  it('pads the seconds field below ten at every zoom level', () => {
    expect(tickLabel(61.5, 0.1)).toBe('1:01.5');
    expect(tickLabel(3661.2, 0.1, 7200)).toBe('1:01:01.2');
  });

  it('signs negative times rather than printing a bare minus in the middle', () => {
    expect(tickLabel(-30, 1)).toBe('-0:30');
  });
});

describe('formatters', () => {
  it('formatTime clamps rubbish to zero and pads to mm:ss(.mmm)', () => {
    expect(formatTime(0)).toBe('00:00');
    expect(formatTime(-4)).toBe('00:00');
    expect(formatTime(Number.NaN)).toBe('00:00');
    expect(formatTime(65.4)).toBe('01:05');
    expect(formatTime(65.4, true)).toBe('01:05.400');
    expect(formatTime(125.25, true)).toBe('02:05.250');
  });

  it('formatTimeShort keeps a decimal only where it is readable', () => {
    expect(formatTimeShort(5)).toBe('5s');
    expect(formatTimeShort(5.5)).toBe('5.5s');
    expect(formatTimeShort(42.4)).toBe('42s');
    expect(formatTimeShort(65.4)).toBe('1:05');
  });

  // REGRESSION GUARD. An earlier build split the minute off first and rounded
  // the seconds remainder on its own — so 119.6 s printed `1:60`, a clock
  // reading that does not exist, and 59.6 s printed `60s`. The fix rounds to
  // whole seconds FIRST and only then splits; these cases are the difference.
  it('formatTimeShort carries a rounded-up remainder into the minute', () => {
    expect(formatTimeShort(119.6)).toBe('2:00');
    expect(formatTimeShort(179.5)).toBe('3:00');
    expect(formatTimeShort(59.6)).toBe('1:00');
    expect(formatTimeShort(3599.7)).toBe('60:00');
    expect(formatTimeShort(119.4)).toBe('1:59'); // control: nothing to carry
  });

  it('formatTimeShort never prints a sixtieth second, anywhere in 10 minutes', () => {
    for (let i = 0; i <= 6000; i++) {
      const t = i / 10;
      const out = formatTimeShort(t);
      expect(out, `t=${t.toFixed(1)}`).not.toMatch(/:60$/);
      expect(out, `t=${t.toFixed(1)}`).not.toBe('60s');
    }
  });

  it('formatSeconds shows more digits the deeper the zoom', () => {
    expect(formatSeconds(12.3456789, 120)).toBe('12.3 s');
    expect(formatSeconds(12.3456789, 12)).toBe('12.35 s');
    expect(formatSeconds(12.3456789, 1)).toBe('12.346 s');
    expect(formatSeconds(12.3456789, 0.1)).toBe('12.3457 s');
    expect(formatSeconds(-3, 1)).toBe('0.000 s');
    expect(formatSeconds(Number.NaN, 1)).toBe('0.000 s');
  });
});
