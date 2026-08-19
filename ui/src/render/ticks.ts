// Shared time-ruler math (used by the waveform worker for grid lines and by
// React for the labels, so both agree to the pixel).

const STEPS_S = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1200, 1800, 3600];

export interface Tick {
  time: number;
  major: boolean;
}

export function chooseStep(durationS: number, widthPx: number, minPx = 72): number {
  if (!(durationS > 0) || !(widthPx > 0)) return 1;
  const pxPerS = widthPx / durationS;
  for (const s of STEPS_S) {
    if (s * pxPerS >= minPx) return s;
  }
  return STEPS_S[STEPS_S.length - 1] ?? 3600;
}

export function timeTicks(durationS: number, widthPx: number, minPx = 72): Tick[] {
  const step = chooseStep(durationS, widthPx, minPx);
  const minor = step / (step >= 1 && Number.isInteger(step) && step % 5 === 0 ? 5 : 4);
  const out: Tick[] = [];
  const n = Math.floor(durationS / minor + 1e-6);
  for (let i = 0; i <= n; i++) {
    const t = i * minor;
    if (t > durationS + 1e-6) break;
    const major = Math.abs(t / step - Math.round(t / step)) < 1e-6;
    out.push({ time: t, major });
  }
  return out;
}

export function formatTime(t: number, withMs = false): string {
  if (!Number.isFinite(t) || t < 0) t = 0;
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  if (withMs) {
    const whole = Math.floor(s);
    const ms = Math.floor((s - whole) * 1000);
    return `${String(m).padStart(2, '0')}:${String(whole).padStart(2, '0')}.${String(ms).padStart(3, '0')}`;
  }
  return `${String(m).padStart(2, '0')}:${String(Math.floor(s)).padStart(2, '0')}`;
}

export function formatTimeShort(t: number): string {
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  if (t < 10 && !Number.isInteger(t)) return `${t.toFixed(1)}s`;
  if (m === 0) return `${Math.round(s)}s`;
  return `${m}:${String(Math.round(s)).padStart(2, '0')}`;
}
