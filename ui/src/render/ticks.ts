// Shared time-ruler math (used by the waveform worker for grid lines and by
// the main thread for the ruler labels, so both agree to the pixel).
//
// Everything is range-aware: the ruler must relabel sensibly from a 6-hour
// file down to a 40 ms window, so the step ladder runs from 6 h to 100 µs and
// the label format follows the step (h:mm:ss -> m:ss -> m:ss.mmm).

/** [major step, minor step] in seconds, ascending. */
const STEPS: ReadonlyArray<readonly [number, number]> = [
  [0.0001, 0.00002],
  [0.0002, 0.00005],
  [0.0005, 0.0001],
  [0.001, 0.0002],
  [0.002, 0.0005],
  [0.005, 0.001],
  [0.01, 0.002],
  [0.02, 0.005],
  [0.05, 0.01],
  [0.1, 0.02],
  [0.2, 0.05],
  [0.5, 0.1],
  [1, 0.2],
  [2, 0.5],
  [5, 1],
  [10, 2],
  [15, 5],
  [30, 10],
  [60, 15],
  [120, 30],
  [300, 60],
  [600, 120],
  [900, 300],
  [1800, 600],
  [3600, 900],
  [7200, 1800],
  [10800, 3600],
  [21600, 3600],
];

const LAST = STEPS[STEPS.length - 1] ?? ([3600, 900] as const);

/** Hard cap so a pathological span never floods the tick list. */
const MAX_TICKS = 4000;

export interface Tick {
  time: number;
  major: boolean;
}

/** Major/minor step (seconds) for a span rendered `widthPx` wide. */
export function chooseSteps(spanS: number, widthPx: number, minPx = 72): [number, number] {
  if (!(spanS > 0) || !(widthPx > 0)) return [1, 0.2];
  const pxPerS = widthPx / spanS;
  for (const s of STEPS) {
    if (s[0] * pxPerS >= minPx) return [s[0], s[1]];
  }
  return [LAST[0], LAST[1]];
}

/** Major step only — kept for callers that just need the label cadence. */
export function chooseStep(spanS: number, widthPx: number, minPx = 72): number {
  return chooseSteps(spanS, widthPx, minPx)[0];
}

/** Ticks inside an arbitrary [startS, endS] window. */
export function timeTicksIn(startS: number, endS: number, widthPx: number, minPx = 72): Tick[] {
  const span = endS - startS;
  if (!(span > 0) || !(widthPx > 0)) return [];
  const [step, minor] = chooseSteps(span, widthPx, minPx);
  const out: Tick[] = [];
  const eps = minor * 1e-6;
  const i0 = Math.ceil((startS - eps) / minor);
  const i1 = Math.floor((endS + eps) / minor);
  if (i1 - i0 > MAX_TICKS) return [];
  for (let i = i0; i <= i1; i++) {
    const t = i * minor;
    if (t < -eps) continue;
    const r = t / step;
    out.push({ time: t, major: Math.abs(r - Math.round(r)) < 1e-6 });
  }
  return out;
}

/** Ticks over [0, durationS] (full-file view). */
export function timeTicks(durationS: number, widthPx: number, minPx = 72): Tick[] {
  return timeTicksIn(0, durationS, widthPx, minPx);
}

function pad(n: number, w: number): string {
  return String(n).padStart(w, '0');
}

/**
 * Label for a tick, formatted for the zoom level the step implies:
 * step >= 1 s   -> h:mm:ss / m:ss
 * step <  1 s   -> m:ss.d / m:ss.dd / m:ss.mmm / m:ss.dddd
 *
 * `rangeEndS` keeps one ruler internally consistent: if any label on it needs
 * an hours field, they all get one.
 */
export function tickLabel(t: number, step: number, rangeEndS = t): string {
  const sign = t < 0 ? '-' : '';
  const at = Math.abs(t);
  const h = Math.floor(at / 3600);
  const m = Math.floor((at - h * 3600) / 60);
  const s = at - h * 3600 - m * 60;
  const hours = Math.abs(rangeEndS) >= 3600;
  if (step >= 1) {
    const ws = Math.round(s);
    // Rounding can push 59.6 -> 60; renormalise.
    let mm = m;
    let hh = h;
    let ss = ws;
    if (ss === 60) {
      ss = 0;
      mm += 1;
    }
    if (mm === 60) {
      mm = 0;
      hh += 1;
    }
    return hours ? `${sign}${hh}:${pad(mm, 2)}:${pad(ss, 2)}` : `${sign}${mm}:${pad(ss, 2)}`;
  }
  const dec = step >= 0.1 ? 1 : step >= 0.01 ? 2 : step >= 0.001 ? 3 : 4;
  const str = s.toFixed(dec);
  const whole = str.split('.')[0] ?? '0';
  const frac = str.split('.')[1] ?? '';
  if (hours) return `${sign}${h}:${pad(m, 2)}:${pad(Number(whole), 2)}.${frac}`;
  return `${sign}${m}:${pad(Number(whole), 2)}.${frac}`;
}

export function formatTime(t: number, withMs = false): string {
  if (!Number.isFinite(t) || t < 0) t = 0;
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  if (withMs) {
    const whole = Math.floor(s);
    const ms = Math.floor((s - whole) * 1000);
    return `${pad(m, 2)}:${pad(whole, 2)}.${pad(ms, 3)}`;
  }
  return `${pad(m, 2)}:${pad(Math.floor(s), 2)}`;
}

export function formatTimeShort(t: number): string {
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  if (t < 10 && !Number.isInteger(t)) return `${t.toFixed(1)}s`;
  if (m === 0) return `${Math.round(s)}s`;
  return `${m}:${String(Math.round(s)).padStart(2, '0')}`;
}

/**
 * Compact seconds reading for the zoom indicator: enough precision to see the
 * window move at every zoom level, never more digits than the eye can use.
 */
export function formatSeconds(t: number, spanS: number): string {
  const v = Number.isFinite(t) ? Math.max(0, t) : 0;
  const dec = spanS >= 60 ? 1 : spanS >= 6 ? 2 : spanS >= 0.6 ? 3 : 4;
  return `${v.toFixed(dec)} s`;
}
