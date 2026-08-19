// The visible time window of the waveform, owned outside React.
//
// Zoom and pan run at pointer rate (a trackpad emits wheel events faster than
// 60 Hz). Routing that through React state would re-render the tree on every
// event, so the window lives here: subscribers update the worker, the ruler,
// the overview bar and the verdict strip imperatively. The zustand store keeps
// a mirror for anything that legitimately wants to re-render, refreshed on a
// trailing timer rather than per frame.

export interface ViewWindow {
  start: number;
  end: number;
}

export type ViewListener = (view: ViewWindow) => void;

/** Never let the window collapse below this, whatever the sample rate says. */
const ABS_MIN_SPAN_S = 1e-4;

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

class WaveViewController {
  private start = 0;
  private end = 0;
  private durationS = 0;
  private minSpanS = ABS_MIN_SPAN_S;
  private readonly listeners = new Set<ViewListener>();

  get view(): ViewWindow {
    return { start: this.start, end: this.end };
  }

  get start_s(): number {
    return this.start;
  }

  get end_s(): number {
    return this.end;
  }

  get span(): number {
    return this.end - this.start;
  }

  get duration(): number {
    return this.durationS;
  }

  get minSpan(): number {
    return this.minSpanS;
  }

  /** True when the window is (within a pixel's worth) the whole file. */
  get isFull(): boolean {
    if (!(this.durationS > 0)) return true;
    return this.start <= 1e-6 && this.end >= this.durationS - 1e-6;
  }

  /** True when one more zoom step would go past 1 sample per bucket. */
  get isMaxZoom(): boolean {
    return this.span <= this.minSpanS * (1 + 1e-6);
  }

  /** How many times narrower than the whole file the window is. */
  get factor(): number {
    if (!(this.durationS > 0) || !(this.span > 0)) return 1;
    return this.durationS / this.span;
  }

  subscribe(fn: ViewListener): () => void {
    this.listeners.add(fn);
    return () => {
      this.listeners.delete(fn);
    };
  }

  private emit(): void {
    const v = this.view;
    for (const fn of this.listeners) fn(v);
  }

  /**
   * New clip: full duration, and the narrowest window we will ever allow
   * (`maxBuckets` buckets at one sample each).
   */
  setSource(durationS: number, sampleRate: number, maxBuckets: number): void {
    this.durationS = durationS > 0 ? durationS : 0;
    const sr = sampleRate > 0 ? sampleRate : 48000;
    const buckets = maxBuckets > 0 ? maxBuckets : 1200;
    this.minSpanS = Math.max(ABS_MIN_SPAN_S, Math.min(this.durationS || buckets / sr, buckets / sr));
    this.start = 0;
    this.end = this.durationS;
    this.emit();
  }

  /** Display got wider/narrower: the deepest useful zoom moves with it. */
  setMaxBuckets(sampleRate: number, maxBuckets: number): void {
    if (!(this.durationS > 0)) return;
    const sr = sampleRate > 0 ? sampleRate : 48000;
    const buckets = maxBuckets > 0 ? maxBuckets : 1200;
    this.minSpanS = Math.max(ABS_MIN_SPAN_S, Math.min(this.durationS, buckets / sr));
    if (this.span < this.minSpanS) this.set(this.start, this.start + this.minSpanS);
  }

  clear(): void {
    this.durationS = 0;
    this.start = 0;
    this.end = 0;
    this.emit();
  }

  /** Set the window, clamped to the file and to the zoom limits. */
  set(start: number, end: number): void {
    if (!(this.durationS > 0)) {
      if (this.start !== 0 || this.end !== 0) {
        this.start = 0;
        this.end = 0;
        this.emit();
      }
      return;
    }
    let span = end - start;
    if (!Number.isFinite(span) || span <= 0) span = this.durationS;
    span = clamp(span, this.minSpanS, this.durationS);
    let s = Number.isFinite(start) ? start : 0;
    s = clamp(s, 0, this.durationS - span);
    const e = s + span;
    if (s === this.start && e === this.end) return;
    this.start = s;
    this.end = e;
    this.emit();
  }

  /** Zoom about a fixed time (the cursor), `factor` > 1 zooms in. */
  zoomAt(pivotS: number, factor: number): void {
    if (!(this.durationS > 0) || !(factor > 0)) return;
    const span = this.span || this.durationS;
    const next = clamp(span / factor, this.minSpanS, this.durationS);
    if (next === span) return;
    const p = clamp(pivotS, this.start, this.end);
    const frac = span > 0 ? (p - this.start) / span : 0.5;
    this.set(p - frac * next, p - frac * next + next);
  }

  panBy(deltaS: number): void {
    if (!(this.durationS > 0) || deltaS === 0) return;
    this.set(this.start + deltaS, this.end + deltaS);
  }

  /** Centre the window on a time (overview-bar click). */
  centerOn(timeS: number): void {
    if (!(this.durationS > 0)) return;
    const span = this.span || this.durationS;
    this.set(timeS - span / 2, timeS - span / 2 + span);
  }

  reset(): void {
    this.set(0, this.durationS);
  }
}

export const waveView = new WaveViewController();
