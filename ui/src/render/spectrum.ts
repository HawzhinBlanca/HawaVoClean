// Spectrum analyser display — main-thread Canvas 2D, driven by requestAnimationFrame.
//
// Three layers, back to front:
//   1. an inset display ground with a log-frequency / dB grid (hairline, one
//      device pixel wide, so it never competes with the curves);
//   2. the long-term average spectra of the two decks (ORIGINAL amber,
//      CLEANED cyan) as gradient-filled areas with a glow and a bright core
//      stroke, plus difference shading wherever the cleaned curve sits below
//      the original — that band *is* the noise this app removed;
//   3. the live AnalyserNode overlay with proper meter ballistics (fast
//      attack / slow release) and a peak-hold line that decays over ~1.5 s.
//
// No React state is touched per frame, and a frame allocates nothing: every
// projection, gradient and label string is rebuilt only when the data, the
// canvas size or the theme changes.

export interface SpectrumCurve {
  freqs: Float32Array;
  db: Float32Array;
}

export type SpectrumDeck = 'original' | 'cleaned';

type RGB = readonly [number, number, number];

const F_MIN = 20;
const F_MAX = 20000;
const DB_MIN = -96;
const DB_MAX = 0;
const LOG_SPAN = Math.log2(F_MAX / F_MIN);
const DB_SPAN = DB_MAX - DB_MIN;

// AnalyserNode reports |X[k]| of a Blackman-windowed frame (coherent gain
// ≈ 0.42, one-sided), so a full-scale sine lands near −13.5 dB. The engine's
// long-term spectrum is referenced so that sine reads ≈ 0 dB; compensate.
const ANALYSER_OFFSET_DB = 13.5;

// Meter ballistics for the live overlay.
const ATTACK_S = 0.022; // fast attack — transients are visible
const RELEASE_S = 0.34; // slow release — the curve breathes instead of flickering
const PEAK_HOLD_S = 0.4; // peak sits still this long before it starts falling
const PEAK_FALL_DB_S = DB_SPAN / 1.5; // then it crosses the whole scale in ~1.5 s
const FADE_IN_S = 0.22;
const FADE_OUT_S = 0.45;

const MAJOR_HZ = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];
const MINOR_HZ = [
  30, 40, 60, 70, 80, 90, 300, 400, 600, 700, 800, 900, 3000, 4000, 6000, 7000, 8000, 9000,
];
const DB_MAJOR_STEP = 12;
const DB_MINOR_STEP = 6;

const MAJOR_LABEL = MAJOR_HZ.map((f) => (f >= 1000 ? `${f / 1000}k` : `${f}`));
/* Fallback ladder for a narrow display. The 1-2-5 series needs ~22px between
 * the centres of two three-character labels, and at 0.301 decades per step that
 * takes a plot about 300px wide — a 360px side column does not have it. The old
 * code discovered this one label at a time and simply dropped 10k, which left a
 * conspicuous double gap between 5k and 20k in an otherwise regular series. A
 * ladder is either complete or it steps by whole decades; it is never a series
 * with a hole in it. */
const DECADE_HZ = [20, 100, 1000, 10000];
const DECADE_LABEL = DECADE_HZ.map((f) => (f >= 1000 ? `${f / 1000}k` : `${f}`));
const DB_LINES: number[] = [];
const DB_LABEL: string[] = [];
for (let db = DB_MAX; db >= DB_MIN; db -= DB_MAJOR_STEP) {
  DB_LINES.push(db);
  DB_LABEL.push(`${db}`);
}

/** 1/12-octave band centres across the whole displayed range (live overlay). */
function bandCentres(): Float32Array {
  const n = Math.floor(12 * LOG_SPAN) + 1;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = F_MIN * 2 ** (i / 12);
  return out;
}

const BANDS = bandCentres();

const FALLBACK: Record<string, string> = {
  '--amber': '#ffb347',
  '--amber-2': '#ffd28a',
  '--cyan': '#39d0ff',
  '--cyan-2': '#9ae8ff',
  '--display': '#07080a',
  '--display-2': '#0a0c0f',
  '--fg-2': '#aab2bf',
  '--fg-3': '#6f7886',
  '--fg-4': '#454c58',
  '--font-mono': "'SF Mono', ui-monospace, Menlo, Monaco, Consolas, monospace",
  '--font-ui':
    "-apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Segoe UI', Helvetica, Arial, sans-serif",
};

function parseColor(raw: string, fb: RGB): RGB {
  const s = raw.trim();
  if (s.startsWith('#')) {
    const h = s.slice(1);
    if (h.length === 3) {
      const n = parseInt(`${h[0]}${h[0]}${h[1]}${h[1]}${h[2]}${h[2]}`, 16);
      if (Number.isFinite(n)) return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
      return fb;
    }
    if (h.length >= 6) {
      const n = parseInt(h.slice(0, 6), 16);
      if (Number.isFinite(n)) return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
    }
    return fb;
  }
  const m = s.match(/-?\d*\.?\d+/g);
  if (s.startsWith('rgb') && m && m.length >= 3) {
    return [Number(m[0]), Number(m[1]), Number(m[2])];
  }
  return fb;
}

function rgba(c: RGB, a: number): string {
  return `rgba(${c[0]},${c[1]},${c[2]},${a})`;
}

interface Pen {
  glow: string;
  halo: string;
  core: string;
  live: string;
  peak: string;
}

interface Plot {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface Paints {
  amber: RGB;
  amber2: RGB;
  cyan: RGB;
  cyan2: RGB;
  bgTop: string;
  bgBot: string;
  gridMinor: string;
  gridMajor: string;
  gridAnchor: string;
  cursor: string;
  tick: string;
  axis: string;
  faint: string;
  frame: string;
  readout: string;
  fontTick: string;
  fontAxis: string;
  fontChip: string;
}

/** Screen-space projection of one curve, rebuilt only when data/size change. */
interface Projection {
  n: number;
  xs: Float32Array;
  ys: Float32Array;
}

const EMPTY_PROJECTION: Projection = { n: 0, xs: new Float32Array(0), ys: new Float32Array(0) };

export class SpectrumRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private ro: ResizeObserver;
  private mql: MediaQueryList | null = null;
  private cssW = 0;
  private cssH = 0;
  private dpr = 1;
  private raf = 0;
  private dirty = true;

  private curves: Record<SpectrumDeck, SpectrumCurve | null> = { original: null, cleaned: null };
  private proj: Record<SpectrumDeck, Projection> = {
    original: EMPTY_PROJECTION,
    cleaned: EMPTY_PROJECTION,
  };
  private focus: SpectrumDeck = 'original';

  // difference band (cleaned vs original on a shared frequency axis)
  private diffN = 0;
  private diffX = new Float32Array(0);
  private diffYo = new Float32Array(0);
  private diffYc = new Float32Array(0);

  // live analyser
  private analyser: AnalyserNode | null = null;
  private analyserRate = 48000;
  private liveDeck: SpectrumDeck = 'original';
  private liveActive = false;
  private liveBuf: Float32Array<ArrayBuffer> | null = null;
  private liveRaw = new Float32Array(BANDS.length);
  private liveDb = new Float32Array(BANDS.length).fill(DB_MIN);
  private livePeak = new Float32Array(BANDS.length).fill(DB_MIN);
  private liveHold = new Float32Array(BANDS.length);
  private liveXs = new Float32Array(BANDS.length);
  private liveYs = new Float32Array(BANDS.length);
  private livePeakYs = new Float32Array(BANDS.length);
  private liveVisible = 0; // 0..1 fade
  private lastT = 0;

  // cached paints / gradients (rebuilt on resize or theme change)
  private paints: Paints;
  private gBg: CanvasGradient | null = null;
  private gVignette: CanvasGradient | null = null;
  private gFill: Record<SpectrumDeck, CanvasGradient | null> = { original: null, cleaned: null };
  private gDiff: CanvasGradient | null = null;
  private strokeOf: Record<SpectrumDeck, Pen>;

  // hover readout
  private cursorX: number | null = null;

  // frame instrumentation (read by perf probes; never by React)
  private frames = 0;
  private drawMsTotal = 0;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d', { alpha: false, desynchronized: true });
    if (!ctx) throw new Error('2d context unavailable');
    this.ctx = ctx;
    this.paints = this.readPaints();
    this.strokeOf = {
      original: this.strokeSet(this.paints.amber, this.paints.amber2),
      cleaned: this.strokeSet(this.paints.cyan, this.paints.cyan2),
    };
    this.ro = new ResizeObserver(() => this.syncSize());
    this.ro.observe(canvas);
    this.onDprChange = this.onDprChange.bind(this);
    this.watchDpr();
    this.syncSize();
    this.onPointerMove = this.onPointerMove.bind(this);
    this.onPointerLeave = this.onPointerLeave.bind(this);
    canvas.addEventListener('pointermove', this.onPointerMove);
    canvas.addEventListener('pointerleave', this.onPointerLeave);
    this.loop = this.loop.bind(this);
    this.raf = requestAnimationFrame(this.loop);
  }

  // ---- theme --------------------------------------------------------------

  private readPaints(): Paints {
    const cs = getComputedStyle(this.canvas);
    const raw = (name: string): string => {
      const v = cs.getPropertyValue(name).trim();
      return v || FALLBACK[name] || '';
    };
    const col = (name: string, fb: RGB): RGB => parseColor(raw(name), fb);
    const amber = col('--amber', [255, 179, 71]);
    const amber2 = col('--amber-2', [255, 210, 138]);
    const cyan = col('--cyan', [57, 208, 255]);
    const cyan2 = col('--cyan-2', [154, 232, 255]);
    const fg2 = col('--fg-2', [170, 178, 191]);
    const fg3 = col('--fg-3', [111, 120, 134]);
    const fg4 = col('--fg-4', [69, 76, 88]);
    const mono = raw('--font-mono') || FALLBACK['--font-mono'];
    const ui = raw('--font-ui') || FALLBACK['--font-ui'];
    // Grid tints are white by default but honour explicit overrides so the
    // token pass can retune the display without touching this file.
    const gridRaw = cs.getPropertyValue('--spec-grid').trim();
    const grid: RGB = gridRaw ? parseColor(gridRaw, [255, 255, 255]) : [255, 255, 255];
    return {
      amber,
      amber2,
      cyan,
      cyan2,
      bgTop: raw('--display-2') || '#0a0c0f',
      bgBot: raw('--display') || '#07080a',
      gridMinor: rgba(grid, 0.022),
      gridMajor: rgba(grid, 0.055),
      gridAnchor: rgba(grid, 0.1),
      cursor: rgba(grid, 0.2),
      tick: rgba(fg4, 0.95),
      axis: rgba(fg3, 0.9),
      faint: rgba(fg4, 0.75),
      frame: rgba(grid, 0.05),
      readout: rgba(fg2, 0.95),
      fontTick: `500 9.5px ${mono}`,
      fontAxis: `600 9px ${ui}`,
      fontChip: `600 9.5px ${mono}`,
    };
  }

  private strokeSet(base: RGB, lite: RGB): Pen {
    return {
      glow: rgba(base, 0.1),
      halo: rgba(base, 0.2),
      core: rgba(base, 0.98),
      // the live trace runs over the averaged one, so it takes the lighter
      // tint of the same hue — same deck, obviously a different reading
      live: rgba(lite, 0.95),
      peak: rgba(lite, 0.5),
    };
  }

  /** Re-read the CSS custom properties (call after a theme swap). */
  refreshTheme(): void {
    this.paints = this.readPaints();
    this.strokeOf = {
      original: this.strokeSet(this.paints.amber, this.paints.amber2),
      cleaned: this.strokeSet(this.paints.cyan, this.paints.cyan2),
    };
    this.rebuildPaints();
    this.dirty = true;
  }

  private rebuildPaints(): void {
    const ctx = this.ctx;
    const p = this.plot;
    const bg = ctx.createLinearGradient(0, 0, 0, this.cssH);
    bg.addColorStop(0, this.paints.bgTop);
    bg.addColorStop(1, this.paints.bgBot);
    this.gBg = bg;
    const vig = ctx.createRadialGradient(
      this.cssW / 2,
      this.cssH / 2,
      Math.min(this.cssW, this.cssH) * 0.25,
      this.cssW / 2,
      this.cssH / 2,
      Math.max(this.cssW, this.cssH) * 0.78,
    );
    vig.addColorStop(0, 'rgba(0,0,0,0)');
    vig.addColorStop(1, 'rgba(0,0,0,0.5)');
    this.gVignette = vig;
    this.gFill = {
      original: this.areaGradient(p, this.paints.amber),
      cleaned: this.areaGradient(p, this.paints.cyan),
    };
    const d = ctx.createLinearGradient(0, p.y, 0, p.y + p.h);
    d.addColorStop(0, rgba(this.paints.amber, 0.16));
    d.addColorStop(1, rgba(this.paints.amber, 0.05));
    this.gDiff = d;
  }

  private areaGradient(p: Plot, c: RGB): CanvasGradient {
    const g = this.ctx.createLinearGradient(0, p.y, 0, p.y + p.h);
    g.addColorStop(0, rgba(c, 0.34));
    g.addColorStop(0.45, rgba(c, 0.12));
    g.addColorStop(1, rgba(c, 0));
    return g;
  }

  // ---- size ---------------------------------------------------------------

  /**
   * Re-measure the canvas. The ResizeObserver does this on its own; call it
   * directly after changing the layout imperatively in the same frame, or when
   * observers are frozen (a backgrounded tab).
   */
  resize(): void {
    this.syncSize();
  }

  /** Dragging the window to a screen with a different DPR must re-scale. */
  private watchDpr(): void {
    this.mql?.removeEventListener('change', this.onDprChange);
    const dpr = window.devicePixelRatio || 1;
    this.mql = window.matchMedia(`(resolution: ${dpr}dppx)`);
    this.mql.addEventListener('change', this.onDprChange);
  }

  private onDprChange(): void {
    this.syncSize();
    this.watchDpr();
  }

  private syncSize(): void {
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width));
    const h = Math.max(1, Math.round(rect.height));
    const dpr = window.devicePixelRatio || 1;
    if (w === this.cssW && h === this.cssH && dpr === this.dpr) return;
    this.cssW = w;
    this.cssH = h;
    this.dpr = dpr;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.rebuildPaints();
    this.projectAll();
    this.dirty = true;
  }

  // ---- data ---------------------------------------------------------------

  setCurve(deck: SpectrumDeck, curve: SpectrumCurve | null): void {
    this.curves[deck] = curve;
    this.proj[deck] = this.project(curve);
    this.rebuildDiff();
    this.dirty = true;
  }

  setFocus(deck: SpectrumDeck): void {
    if (this.focus !== deck) {
      this.focus = deck;
      this.dirty = true;
    }
  }

  setLive(
    analyser: AnalyserNode | null,
    sampleRate: number,
    deck: SpectrumDeck,
    active: boolean,
  ): void {
    this.analyser = analyser;
    this.analyserRate = sampleRate;
    this.liveDeck = deck;
    this.liveActive = active && analyser !== null;
    if (analyser) {
      // Ballistics belong to this renderer, so the node itself smooths only
      // enough to take the edge off bin-to-bin noise.
      analyser.smoothingTimeConstant = 0.4;
      if (!this.liveBuf || this.liveBuf.length !== analyser.frequencyBinCount) {
        this.liveBuf = new Float32Array(analyser.frequencyBinCount);
      }
    }
    this.dirty = true;
  }

  dispose(): void {
    cancelAnimationFrame(this.raf);
    this.ro.disconnect();
    this.mql?.removeEventListener('change', this.onDprChange);
    this.canvas.removeEventListener('pointermove', this.onPointerMove);
    this.canvas.removeEventListener('pointerleave', this.onPointerLeave);
  }

  /** Mean cost of a drawn frame in ms, and how many frames have been drawn. */
  get stats(): { frames: number; meanDrawMs: number } {
    return {
      frames: this.frames,
      meanDrawMs: this.frames ? this.drawMsTotal / this.frames : 0,
    };
  }

  /**
   * Advance the ballistics by `dt` seconds and draw one frame immediately.
   * The app is rAF-driven; this exists so probes and tests can step the
   * display deterministically (including in a backgrounded tab, where rAF
   * is frozen).
   */
  frame(dt = 1 / 60): void {
    this.step(dt);
    this.draw();
  }

  // ---- geometry -----------------------------------------------------------

  private get plot(): Plot {
    const left = 33;
    const right = 9;
    const top = 11;
    const bottom = 17;
    return {
      x: left,
      y: top,
      w: Math.max(10, this.cssW - left - right),
      h: Math.max(10, this.cssH - top - bottom),
    };
  }

  private xOf(f: number, p: Plot): number {
    const clamped = f < F_MIN ? F_MIN : f > F_MAX ? F_MAX : f;
    return p.x + (Math.log2(clamped / F_MIN) / LOG_SPAN) * p.w;
  }

  private yOf(db: number, p: Plot): number {
    const clamped = db < DB_MIN ? DB_MIN : db > DB_MAX ? DB_MAX : db;
    return p.y + ((DB_MAX - clamped) / DB_SPAN) * p.h;
  }

  /** Snap to the centre of a device pixel so hairlines stay one pixel wide. */
  private snap(v: number): number {
    const d = this.dpr;
    return (Math.floor(v * d) + 0.5) / d;
  }

  private project(curve: SpectrumCurve | null): Projection {
    if (!curve) return EMPTY_PROJECTION;
    const n = Math.min(curve.freqs.length, curve.db.length);
    if (n < 2) return EMPTY_PROJECTION;
    const p = this.plot;
    const xs = new Float32Array(n);
    const ys = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      xs[i] = this.xOf(curve.freqs[i] ?? F_MIN, p);
      ys[i] = this.yOf(curve.db[i] ?? DB_MIN, p);
    }
    return { n, xs, ys };
  }

  private projectAll(): void {
    this.proj.original = this.project(this.curves.original);
    this.proj.cleaned = this.project(this.curves.cleaned);
    const p = this.plot;
    for (let i = 0; i < BANDS.length; i++) this.liveXs[i] = this.xOf(BANDS[i] ?? F_MIN, p);
    this.rebuildDiff();
  }

  /** dB of `curve` at `f`, linearly interpolated on a log-frequency axis. */
  private static sampleAt(curve: SpectrumCurve, f: number): number {
    const fr = curve.freqs;
    const db = curve.db;
    const n = Math.min(fr.length, db.length);
    if (n === 0) return DB_MIN;
    if (f <= (fr[0] ?? 0)) return db[0] ?? DB_MIN;
    if (f >= (fr[n - 1] ?? 0)) return db[n - 1] ?? DB_MIN;
    let lo = 0;
    let hi = n - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if ((fr[mid] ?? 0) <= f) lo = mid;
      else hi = mid;
    }
    const f0 = fr[lo] ?? 1;
    const f1 = fr[hi] ?? 1;
    const t = Math.log2(f / f0) / Math.log2(f1 / f0 || 1);
    return (db[lo] ?? DB_MIN) + ((db[hi] ?? DB_MIN) - (db[lo] ?? DB_MIN)) * t;
  }

  /** Cleaned resampled onto the original's frequency axis, in screen space. */
  private rebuildDiff(): void {
    const o = this.curves.original;
    const c = this.curves.cleaned;
    if (!o || !c) {
      this.diffN = 0;
      return;
    }
    const n = Math.min(o.freqs.length, o.db.length);
    if (n < 2) {
      this.diffN = 0;
      return;
    }
    if (this.diffX.length !== n) {
      this.diffX = new Float32Array(n);
      this.diffYo = new Float32Array(n);
      this.diffYc = new Float32Array(n);
    }
    const p = this.plot;
    const sameGrid = c.freqs.length === o.freqs.length;
    for (let i = 0; i < n; i++) {
      const f = o.freqs[i] ?? F_MIN;
      const dbO = o.db[i] ?? DB_MIN;
      const dbC = sameGrid ? (c.db[i] ?? DB_MIN) : SpectrumRenderer.sampleAt(c, f);
      this.diffX[i] = this.xOf(f, p);
      this.diffYo[i] = this.yOf(dbO, p);
      this.diffYc[i] = this.yOf(dbC, p);
    }
    this.diffN = n;
  }

  // ---- pointer ------------------------------------------------------------

  private onPointerMove(e: PointerEvent): void {
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const p = this.plot;
    const next = x >= p.x - 2 && x <= p.x + p.w + 2 ? Math.min(p.x + p.w, Math.max(p.x, x)) : null;
    if (next !== this.cursorX) {
      this.cursorX = next;
      this.dirty = true;
    }
  }

  private onPointerLeave(): void {
    if (this.cursorX !== null) {
      this.cursorX = null;
      this.dirty = true;
    }
  }

  // ---- live analyser ------------------------------------------------------

  private sampleLive(dt: number): void {
    const an = this.analyser;
    const bins = this.liveBuf;
    if (!an || !bins) return;
    an.getFloatFrequencyData(bins);
    const binHz = this.analyserRate / 2 / bins.length;
    const n = BANDS.length;
    for (let i = 0; i < n; i++) {
      const fc = BANDS[i] ?? F_MIN;
      const lo = fc * 2 ** (-1 / 24);
      const hi = fc * 2 ** (1 / 24);
      const b0 = lo / binHz;
      const b1 = hi / binHz;
      let v: number;
      if (b1 - b0 >= 1) {
        let m = -Infinity;
        const i0 = Math.max(0, Math.floor(b0));
        const i1 = Math.min(bins.length - 1, Math.ceil(b1));
        for (let k = i0; k <= i1; k++) {
          const x = bins[k] ?? -Infinity;
          if (x > m) m = x;
        }
        v = m;
      } else {
        const bc = fc / binHz;
        const k = Math.floor(bc);
        const f = bc - k;
        const a = bins[Math.max(0, Math.min(bins.length - 1, k))] ?? -Infinity;
        const b = bins[Math.max(0, Math.min(bins.length - 1, k + 1))] ?? -Infinity;
        v = a * (1 - f) + b * f;
      }
      if (!Number.isFinite(v)) v = DB_MIN;
      const t = v + ANALYSER_OFFSET_DB;
      this.liveRaw[i] = t < DB_MIN ? DB_MIN : t > DB_MAX ? DB_MAX : t;
    }
    this.applyBallistics(dt, false);
  }

  /** Fast attack / slow release + peak-hold with decay. */
  private applyBallistics(dt: number, decayOnly: boolean): void {
    const attack = 1 - Math.exp(-dt / ATTACK_S);
    const release = 1 - Math.exp(-dt / RELEASE_S);
    const fall = PEAK_FALL_DB_S * dt;
    const n = BANDS.length;
    for (let i = 0; i < n; i++) {
      const target = decayOnly ? DB_MIN : (this.liveRaw[i] ?? DB_MIN);
      const cur = this.liveDb[i] ?? DB_MIN;
      const k = target > cur ? attack : release;
      const next = cur + (target - cur) * k;
      this.liveDb[i] = next;
      const peak = this.livePeak[i] ?? DB_MIN;
      if (next >= peak) {
        this.livePeak[i] = next;
        this.liveHold[i] = PEAK_HOLD_S;
      } else {
        const hold = (this.liveHold[i] ?? 0) - dt;
        if (hold > 0) {
          this.liveHold[i] = hold;
        } else {
          this.liveHold[i] = 0;
          const dropped = peak - fall;
          this.livePeak[i] = dropped < next ? next : dropped;
        }
      }
    }
  }

  // ---- frame --------------------------------------------------------------

  private step(dt: number): void {
    if (this.liveActive) {
      this.sampleLive(dt);
      this.liveVisible = Math.min(1, this.liveVisible + dt / FADE_IN_S);
    } else if (this.liveVisible > 0) {
      this.liveVisible = Math.max(0, this.liveVisible - dt / FADE_OUT_S);
      this.applyBallistics(dt, true);
    }
  }

  private loop(ts: number): void {
    this.raf = requestAnimationFrame(this.loop);
    const dt = this.lastT ? Math.min(0.1, (ts - this.lastT) / 1000) : 1 / 60;
    this.lastT = ts;
    const animating = this.liveActive || this.liveVisible > 0;
    if (animating) this.step(dt);
    if (!animating && !this.dirty) return;
    this.dirty = false;
    this.draw();
  }

  // ---- draw ---------------------------------------------------------------

  private draw(): void {
    const t0 = performance.now();
    const ctx = this.ctx;
    const W = this.cssW;
    const H = this.cssH;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);

    ctx.fillStyle = this.gBg ?? this.paints.bgBot;
    ctx.fillRect(0, 0, W, H);

    const p = this.plot;
    this.drawGrid(p);

    const o = this.proj.original;
    const c = this.proj.cleaned;
    const hasO = o.n > 1;
    const hasC = c.n > 1;
    const live = this.liveVisible > 0.001;

    ctx.save();
    ctx.beginPath();
    ctx.rect(p.x, p.y, p.w, p.h);
    ctx.clip();

    // With both decks on screen only the monitored one is filled — two
    // stacked washes turn the overlap to mud, and the shaded difference band
    // is the thing that has to read.
    if (hasO && hasC) this.drawDifference();
    if (hasO) {
      const focused = !hasC || this.focus === 'original';
      this.drawCurve(o, 'original', focused ? 1 : 0.62, focused);
    }
    if (hasC) {
      const focused = this.focus === 'cleaned';
      this.drawCurve(c, 'cleaned', focused ? 1 : 0.72, focused);
    }
    if (live) this.drawLive(p, hasO || hasC);

    ctx.restore();

    ctx.fillStyle = this.gVignette ?? 'rgba(0,0,0,0)';
    ctx.fillRect(0, 0, W, H);

    if (!hasO && !hasC && !live) this.drawEmptyState(p);
    this.drawCorner(p, hasO || hasC, hasO && hasC);
    if (this.cursorX !== null && (hasO || hasC)) this.drawReadout(p);

    // inner bevel around the plot
    ctx.lineWidth = 1 / this.dpr;
    ctx.strokeStyle = this.paints.frame;
    ctx.strokeRect(this.snap(p.x), this.snap(p.y), p.w, p.h);

    this.frames++;
    this.drawMsTotal += performance.now() - t0;
  }

  private drawGrid(p: Plot): void {
    const ctx = this.ctx;
    const hair = 1 / this.dpr;
    ctx.lineWidth = hair;
    ctx.font = this.paints.fontTick;
    ctx.textBaseline = 'middle';

    // minor frequency lines
    ctx.strokeStyle = this.paints.gridMinor;
    ctx.beginPath();
    for (const f of MINOR_HZ) {
      const x = this.snap(this.xOf(f, p));
      ctx.moveTo(x, p.y);
      ctx.lineTo(x, p.y + p.h);
    }
    // minor dB lines
    for (let db = DB_MAX - DB_MINOR_STEP; db > DB_MIN; db -= DB_MAJOR_STEP) {
      const y = this.snap(this.yOf(db, p));
      ctx.moveTo(p.x, y);
      ctx.lineTo(p.x + p.w, y);
    }
    ctx.stroke();

    // major frequency lines (1 kHz is the anchor)
    ctx.strokeStyle = this.paints.gridMajor;
    ctx.beginPath();
    for (const f of MAJOR_HZ) {
      if (f === 1000) continue;
      const x = this.snap(this.xOf(f, p));
      ctx.moveTo(x, p.y);
      ctx.lineTo(x, p.y + p.h);
    }
    for (const db of DB_LINES) {
      if (db === DB_MAX || db === DB_MIN) continue;
      const y = this.snap(this.yOf(db, p));
      ctx.moveTo(p.x, y);
      ctx.lineTo(p.x + p.w, y);
    }
    ctx.stroke();

    ctx.strokeStyle = this.paints.gridAnchor;
    ctx.beginPath();
    const x1k = this.snap(this.xOf(1000, p));
    ctx.moveTo(x1k, p.y);
    ctx.lineTo(x1k, p.y + p.h);
    const y0 = this.snap(this.yOf(DB_MAX, p));
    ctx.moveTo(p.x, y0);
    ctx.lineTo(p.x + p.w, y0);
    ctx.stroke();

    // Frequency labels: the full 1-2-5 ladder when every neighbour clears the
    // gap, otherwise whole decades. Either way the reader sees an even series.
    ctx.fillStyle = this.paints.tick;
    ctx.textAlign = 'center';
    const baseline = p.y + p.h + 9;
    const GAP = 5;
    // The end labels are nudged inwards so they do not hang off the frame, so
    // the fit test has to be run on the positions the labels will actually be
    // drawn at — testing the ideal centres is how 10k and 20k ended up touching.
    const placed = (
      hzs: readonly number[],
      labs: readonly string[],
    ): { xs: number[]; fits: boolean } => {
      const xs: number[] = [];
      for (let i = 0; i < hzs.length; i++) {
        const half = ctx.measureText(labs[i] ?? '').width / 2;
        const x = this.xOf(hzs[i] ?? 0, p);
        xs.push(Math.min(p.x + p.w - half, Math.max(p.x + half, x)));
      }
      let fits = true;
      for (let i = 1; i < xs.length; i++) {
        const a = (xs[i - 1] ?? 0) + ctx.measureText(labs[i - 1] ?? '').width / 2;
        const b = (xs[i] ?? 0) - ctx.measureText(labs[i] ?? '').width / 2;
        if (b - a < GAP) fits = false;
      }
      return { xs, fits };
    };
    const full = placed(MAJOR_HZ, MAJOR_LABEL);
    const hz = full.fits ? MAJOR_HZ : DECADE_HZ;
    const labels = full.fits ? MAJOR_LABEL : DECADE_LABEL;
    const xs = full.fits ? full.xs : placed(DECADE_HZ, DECADE_LABEL).xs;
    for (let i = 0; i < hz.length; i++) {
      ctx.fillText(labels[i] ?? '', xs[i] ?? 0, baseline);
    }
    ctx.textAlign = 'right';
    const spacing = (p.h / (DB_LINES.length - 1)) | 0;
    const everyOther = spacing < 17;
    for (let i = 0; i < DB_LINES.length; i++) {
      if (everyOther && i % 2 === 1) continue;
      const db = DB_LINES[i] ?? 0;
      const y = this.yOf(db, p);
      ctx.fillText(DB_LABEL[i] ?? '', p.x - 6, Math.min(p.y + p.h - 4, Math.max(p.y + 4, y)));
    }

    // Axis units sit *on* the axis they name. `Hz` was in the top-right corner
    // — the far end of the dB axis, three quarters of the display away from the
    // frequency scale it labels — where it also crowded the LTAS tag. It now
    // sits in the bottom-left gutter, on the frequency labels' own baseline.
    ctx.font = this.paints.fontAxis;
    ctx.fillStyle = this.paints.axis;
    ctx.textAlign = 'left';
    ctx.fillText('dB', 2, p.y - 3);
    ctx.fillText('Hz', 2, baseline);
  }

  /** Trace a Catmull-Rom-smoothed path through a projection (no allocation). */
  private tracePath(xs: Float32Array, ys: Float32Array, n: number): void {
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.moveTo(xs[0] ?? 0, ys[0] ?? 0);
    for (let i = 0; i < n - 1; i++) {
      const x0 = xs[i > 0 ? i - 1 : 0] ?? 0;
      const y0 = ys[i > 0 ? i - 1 : 0] ?? 0;
      const x1 = xs[i] ?? 0;
      const y1 = ys[i] ?? 0;
      const x2 = xs[i + 1] ?? 0;
      const y2 = ys[i + 1] ?? 0;
      const j = i + 2 < n ? i + 2 : n - 1;
      const x3 = xs[j] ?? 0;
      const y3 = ys[j] ?? 0;
      ctx.bezierCurveTo(
        x1 + (x2 - x0) / 6,
        y1 + (y2 - y0) / 6,
        x2 - (x3 - x1) / 6,
        y2 - (y3 - y1) / 6,
        x2,
        y2,
      );
    }
  }

  private drawCurve(proj: Projection, deck: SpectrumDeck, alpha: number, fill: boolean): void {
    const ctx = this.ctx;
    const { n, xs, ys } = proj;
    if (n < 2) return;
    const p = this.plot;
    const pen = this.strokeOf[deck];

    ctx.globalAlpha = alpha;
    if (fill) {
      this.tracePath(xs, ys, n);
      ctx.lineTo(xs[n - 1] ?? 0, p.y + p.h);
      ctx.lineTo(xs[0] ?? 0, p.y + p.h);
      ctx.closePath();
      ctx.fillStyle = this.gFill[deck] ?? 'rgba(0,0,0,0)';
      ctx.fill();
    }
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    this.tracePath(xs, ys, n);
    ctx.strokeStyle = pen.glow;
    ctx.lineWidth = 6.5;
    ctx.stroke();
    ctx.strokeStyle = pen.halo;
    ctx.lineWidth = 3.2;
    ctx.stroke();
    ctx.strokeStyle = pen.core;
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  /**
   * Shade the band where the cleaned curve sits *below* the original — the
   * energy this app took out. Straight segments between 1/12-octave points
   * track the smoothed curves to well under a pixel at this scale.
   */
  private drawDifference(): void {
    const n = this.diffN;
    if (n < 2) return;
    const ctx = this.ctx;
    const xs = this.diffX;
    const yo = this.diffYo;
    const yc = this.diffYc;
    ctx.fillStyle = this.gDiff ?? 'rgba(0,0,0,0)';
    let i = 0;
    while (i < n) {
      while (i < n && (yc[i] ?? 0) - (yo[i] ?? 0) <= 0.6) i++;
      if (i >= n) break;
      let s = i;
      while (s > 0 && (yc[s - 1] ?? 0) - (yo[s - 1] ?? 0) > 0) s--;
      while (i < n && (yc[i] ?? 0) - (yo[i] ?? 0) > 0) i++;
      const e = i - 1;
      if (e - s < 1) continue;
      ctx.beginPath();
      ctx.moveTo(xs[s] ?? 0, yo[s] ?? 0);
      for (let k = s + 1; k <= e; k++) ctx.lineTo(xs[k] ?? 0, yo[k] ?? 0);
      for (let k = e; k >= s; k--) ctx.lineTo(xs[k] ?? 0, yc[k] ?? 0);
      ctx.closePath();
      ctx.fill();
    }
  }

  private drawLive(p: Plot, hasStatic: boolean): void {
    const ctx = this.ctx;
    const n = BANDS.length;
    for (let i = 0; i < n; i++) {
      this.liveYs[i] = this.yOf(this.liveDb[i] ?? DB_MIN, p);
      this.livePeakYs[i] = this.yOf(this.livePeak[i] ?? DB_MIN, p);
    }
    const pen = this.strokeOf[this.liveDeck];
    const a = this.liveVisible;

    // A filled long-term curve is already washing that area; a second wash
    // over it only makes mud, so the live trace fills only when it is alone.
    if (!hasStatic) {
      ctx.globalAlpha = a * 0.55;
      this.tracePath(this.liveXs, this.liveYs, n);
      ctx.lineTo(this.liveXs[n - 1] ?? 0, p.y + p.h);
      ctx.lineTo(this.liveXs[0] ?? 0, p.y + p.h);
      ctx.closePath();
      ctx.fillStyle = this.gFill[this.liveDeck] ?? 'rgba(0,0,0,0)';
      ctx.fill();
    }

    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.globalAlpha = a;
    this.tracePath(this.liveXs, this.liveYs, n);
    ctx.strokeStyle = pen.glow;
    ctx.lineWidth = 4.5;
    ctx.stroke();
    ctx.strokeStyle = pen.live;
    ctx.lineWidth = 1.2;
    ctx.stroke();

    // peak hold
    ctx.globalAlpha = a * 0.75;
    this.tracePath(this.liveXs, this.livePeakYs, n);
    ctx.strokeStyle = pen.peak;
    // the peak line stays a hair thinner than the live trace
    ctx.lineWidth = this.dpr >= 2 ? 0.75 : 1;
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  private drawEmptyState(p: Plot): void {
    const ctx = this.ctx;
    const cx = p.x + p.w / 2;
    const cy = p.y + p.h / 2;
    ctx.save();
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // a dashed reference line at the mid of the scale, broken by the label
    ctx.setLineDash([2 / this.dpr, 4 / this.dpr]);
    ctx.lineWidth = 1 / this.dpr;
    ctx.strokeStyle = this.paints.gridAnchor;
    const y = this.snap(cy);
    ctx.beginPath();
    ctx.moveTo(p.x + 8, y);
    ctx.lineTo(cx - 74, y);
    ctx.moveTo(cx + 74, y);
    ctx.lineTo(p.x + p.w - 8, y);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.font = this.paints.fontAxis;
    ctx.fillStyle = this.paints.axis;
    ctx.letterSpacing = '1.6px';
    ctx.fillText('NO SIGNAL', cx, cy - 1);
    ctx.font = this.paints.fontTick;
    ctx.fillStyle = this.paints.faint;
    ctx.letterSpacing = '0.6px';
    ctx.fillText('load a file to analyse', cx, cy + 14);
    ctx.letterSpacing = '0px';
    ctx.restore();
  }

  /**
   * Top-right tag: what the curves are, and — when both decks are present —
   * a key for the shaded band between them, next to the band it explains.
   */
  private drawCorner(p: Plot, hasData: boolean, hasBoth: boolean): void {
    const ctx = this.ctx;
    ctx.font = this.paints.fontAxis;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = this.paints.faint;
    ctx.letterSpacing = '0.8px';
    const right = p.x + p.w - 6;
    // the hover readout takes that corner while it is up
    if (this.cursorX === null) {
      ctx.fillText(hasData ? 'LTAS · 1/12 OCT' : '1/12 OCT', right, p.y + 8);
    }
    if (hasBoth) {
      const label = 'REMOVED';
      const y = p.y + 21;
      ctx.fillText(label, right, y);
      const w = ctx.measureText(label).width;
      const sx = right - w - 12;
      ctx.fillStyle = rgba(this.paints.amber, 0.16);
      ctx.fillRect(sx, y - 4, 8, 8);
      ctx.lineWidth = 1 / this.dpr;
      ctx.strokeStyle = rgba(this.paints.amber, 0.3);
      ctx.strokeRect(this.snap(sx), this.snap(y - 4), 8, 8);
      ctx.fillStyle = this.paints.faint;
    }
    ctx.letterSpacing = '0px';
  }

  private drawReadout(p: Plot): void {
    const x = this.cursorX;
    if (x === null) return;
    const ctx = this.ctx;
    const f = F_MIN * 2 ** (((x - p.x) / p.w) * LOG_SPAN);
    const sx = this.snap(x);

    ctx.save();
    ctx.beginPath();
    ctx.rect(p.x, p.y, p.w, p.h);
    ctx.clip();
    ctx.lineWidth = 1 / this.dpr;
    ctx.strokeStyle = this.paints.cursor;
    ctx.beginPath();
    ctx.moveTo(sx, p.y);
    ctx.lineTo(sx, p.y + p.h);
    ctx.stroke();

    const o = this.curves.original;
    const c = this.curves.cleaned;
    const dbO = o ? SpectrumRenderer.sampleAt(o, f) : null;
    const dbC = c ? SpectrumRenderer.sampleAt(c, f) : null;
    if (dbO !== null) this.dot(sx, this.yOf(dbO, p), this.paints.amber);
    if (dbC !== null) this.dot(sx, this.yOf(dbC, p), this.paints.cyan);
    ctx.restore();

    const fLabel = f >= 1000 ? `${(f / 1000).toFixed(f >= 10000 ? 1 : 2)} kHz` : `${Math.round(f)} Hz`;
    ctx.font = this.paints.fontChip;
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';
    const parts: string[] = [fLabel];
    if (dbO !== null) parts.push(`${dbO.toFixed(1)}`);
    if (dbC !== null) parts.push(`${dbC.toFixed(1)}`);
    let wTotal = 10;
    for (const s of parts) wTotal += ctx.measureText(s).width + 10;
    const bx = Math.min(p.x + p.w - wTotal - 2, Math.max(p.x + 2, x + 8));
    const by = p.y + 3;
    const bh = 15;
    ctx.fillStyle = 'rgba(6,8,11,0.88)';
    ctx.strokeStyle = this.paints.frame;
    ctx.lineWidth = 1 / this.dpr;
    ctx.beginPath();
    // roundRect is not in every embedded browser this bundle can end up in
    if (typeof ctx.roundRect === 'function') ctx.roundRect(bx, by, wTotal, bh, 3);
    else ctx.rect(bx, by, wTotal, bh);
    ctx.fill();
    ctx.stroke();
    let tx = bx + 7;
    ctx.fillStyle = this.paints.readout;
    ctx.fillText(fLabel, tx, by + bh / 2);
    tx += ctx.measureText(fLabel).width + 10;
    if (dbO !== null) {
      const s = dbO.toFixed(1);
      ctx.fillStyle = rgba(this.paints.amber2, 0.95);
      ctx.fillText(s, tx, by + bh / 2);
      tx += ctx.measureText(s).width + 10;
    }
    if (dbC !== null) {
      const s = dbC.toFixed(1);
      ctx.fillStyle = rgba(this.paints.cyan2, 0.95);
      ctx.fillText(s, tx, by + bh / 2);
    }
  }

  private dot(x: number, y: number, c: RGB): void {
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.arc(x, y, 2.4, 0, Math.PI * 2);
    ctx.fillStyle = rgba(c, 0.95);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(x, y, 4.5, 0, Math.PI * 2);
    ctx.fillStyle = rgba(c, 0.18);
    ctx.fill();
  }
}
