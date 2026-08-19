// Spectrum display — main-thread Canvas 2D, driven by requestAnimationFrame.
// Static long-term average curves (ORIGINAL amber / CLEANED cyan) plus a live
// AnalyserNode overlay while audio plays. No React state is touched per frame.

export interface SpectrumCurve {
  freqs: Float32Array;
  db: Float32Array;
}

export type SpectrumDeck = 'original' | 'cleaned';

const F_MIN = 40;
const F_MAX = 20000;
const DB_MIN = -90;
const DB_MAX = 0;
const LOG_SPAN = Math.log2(F_MAX / F_MIN);
// AnalyserNode reports |X[k]| of a Blackman-windowed frame (coherent gain
// ≈ 0.42, one-sided), so a full-scale sine lands near −13.5 dB. The engine's
// long-term spectrum is referenced so that sine reads ≈ 0 dB; compensate.
const ANALYSER_OFFSET_DB = 13.5;

const COLORS = {
  bgTop: '#0b0d10',
  bgBot: '#060709',
  grid: 'rgba(255,255,255,0.045)',
  gridMinor: 'rgba(255,255,255,0.022)',
  gridStrong: 'rgba(255,255,255,0.08)',
  label: 'rgba(170,178,191,0.75)',
  labelDim: 'rgba(111,120,134,0.7)',
  amber: '#ffb347',
  amberLite: '#ffd28a',
  cyan: '#39d0ff',
  cyanLite: '#9ae8ff',
};

const FONT = '10px "SF Mono", ui-monospace, Menlo, Monaco, Consolas, monospace';
const FONT_LABEL = '500 9.5px -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Helvetica, Arial, sans-serif';

const MAJOR_HZ = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];
const MINOR_HZ = [
  60, 70, 80, 90, 300, 400, 600, 700, 800, 900, 3000, 4000, 6000, 7000, 8000, 9000,
];

function hzLabel(f: number): string {
  if (f >= 1000) return `${f / 1000}k`;
  return `${f}`;
}

/** 1/12-octave band centres from 40 Hz to 20 kHz (inclusive-ish). */
function bandCentres(): Float32Array {
  const n = Math.floor(12 * LOG_SPAN) + 1;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = F_MIN * 2 ** (i / 12);
  return out;
}

const BANDS = bandCentres();

export class SpectrumRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private ro: ResizeObserver;
  private cssW = 0;
  private cssH = 0;
  private dpr = 1;
  private raf = 0;
  private dirty = true;

  private curves: Record<SpectrumDeck, SpectrumCurve | null> = { original: null, cleaned: null };
  private focus: SpectrumDeck = 'original';
  private analyser: AnalyserNode | null = null;
  private analyserRate = 48000;
  private liveDeck: SpectrumDeck = 'original';
  private liveActive = false;
  private liveBuf: Float32Array<ArrayBuffer> | null = null;
  private liveBands = new Float32Array(BANDS.length);
  private liveSmooth = new Float32Array(BANDS.length).fill(DB_MIN);
  private liveVisible = 0; // 0..1 fade
  private lastT = 0;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d', { alpha: false, desynchronized: true });
    if (!ctx) throw new Error('2d context unavailable');
    this.ctx = ctx;
    this.ro = new ResizeObserver(() => this.syncSize());
    this.ro.observe(canvas);
    this.syncSize();
    this.loop = this.loop.bind(this);
    this.raf = requestAnimationFrame(this.loop);
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
    this.dirty = true;
  }

  setCurve(deck: SpectrumDeck, curve: SpectrumCurve | null): void {
    this.curves[deck] = curve;
    this.dirty = true;
  }

  setFocus(deck: SpectrumDeck): void {
    if (this.focus !== deck) {
      this.focus = deck;
      this.dirty = true;
    }
  }

  setLive(analyser: AnalyserNode | null, sampleRate: number, deck: SpectrumDeck, active: boolean): void {
    this.analyser = analyser;
    this.analyserRate = sampleRate;
    this.liveDeck = deck;
    this.liveActive = active && analyser !== null;
    if (analyser && (!this.liveBuf || this.liveBuf.length !== analyser.frequencyBinCount)) {
      this.liveBuf = new Float32Array(analyser.frequencyBinCount);
    }
    this.dirty = true;
  }

  dispose(): void {
    cancelAnimationFrame(this.raf);
    this.ro.disconnect();
  }

  // ---- geometry -----------------------------------------------------------

  private get plot(): { x: number; y: number; w: number; h: number } {
    const left = 30;
    const right = 10;
    const top = 10;
    const bottom = 18;
    return {
      x: left,
      y: top,
      w: Math.max(10, this.cssW - left - right),
      h: Math.max(10, this.cssH - top - bottom),
    };
  }

  private xOf(f: number, p: { x: number; w: number }): number {
    return p.x + (Math.log2(Math.max(F_MIN, Math.min(F_MAX, f)) / F_MIN) / LOG_SPAN) * p.w;
  }

  private yOf(db: number, p: { y: number; h: number }): number {
    const t = (Math.max(DB_MIN, Math.min(DB_MAX, db)) - DB_MAX) / (DB_MIN - DB_MAX);
    return p.y + t * p.h;
  }

  // ---- live analyser ------------------------------------------------------

  private sampleLive(dt: number): boolean {
    const an = this.analyser;
    if (!an || !this.liveBuf) return false;
    an.getFloatFrequencyData(this.liveBuf);
    const bins = this.liveBuf;
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
      this.liveBands[i] = Math.max(DB_MIN, Math.min(DB_MAX, v + ANALYSER_OFFSET_DB));
    }
    // temporal smoothing on top of the analyser's own (fast attack, slow release)
    const attack = 1 - Math.exp(-dt / 0.03);
    const release = 1 - Math.exp(-dt / 0.18);
    let any = false;
    for (let i = 0; i < n; i++) {
      const target = this.liveBands[i] ?? DB_MIN;
      const cur = this.liveSmooth[i] ?? DB_MIN;
      const k = target > cur ? attack : release;
      const next = cur + (target - cur) * k;
      this.liveSmooth[i] = next;
      if (next > DB_MIN + 0.5) any = true;
    }
    return any;
  }

  // ---- draw ---------------------------------------------------------------

  private loop(ts: number): void {
    this.raf = requestAnimationFrame(this.loop);
    const dt = this.lastT ? Math.min(0.1, (ts - this.lastT) / 1000) : 1 / 60;
    this.lastT = ts;

    let animating = false;
    if (this.liveActive) {
      this.sampleLive(dt);
      this.liveVisible = Math.min(1, this.liveVisible + dt / 0.25);
      animating = true;
    } else if (this.liveVisible > 0) {
      this.liveVisible = Math.max(0, this.liveVisible - dt / 0.45);
      // let the smoothed curve decay towards the floor while fading out
      for (let i = 0; i < this.liveSmooth.length; i++) {
        const cur = this.liveSmooth[i] ?? DB_MIN;
        this.liveSmooth[i] = cur + (DB_MIN - cur) * (1 - Math.exp(-dt / 0.3));
      }
      animating = true;
    }
    if (!animating && !this.dirty) return;
    this.dirty = false;
    this.draw();
  }

  private draw(): void {
    const ctx = this.ctx;
    const dpr = this.dpr;
    const W = this.cssW;
    const H = this.cssH;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // background + vignette
    const bg = ctx.createLinearGradient(0, 0, 0, H);
    bg.addColorStop(0, COLORS.bgTop);
    bg.addColorStop(1, COLORS.bgBot);
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, W, H);
    const vig = ctx.createRadialGradient(W / 2, H / 2, Math.min(W, H) * 0.25, W / 2, H / 2, Math.max(W, H) * 0.75);
    vig.addColorStop(0, 'rgba(0,0,0,0)');
    vig.addColorStop(1, 'rgba(0,0,0,0.45)');
    ctx.fillStyle = vig;
    ctx.fillRect(0, 0, W, H);

    const p = this.plot;

    // grid
    ctx.lineWidth = 1;
    ctx.font = FONT;
    ctx.textBaseline = 'middle';
    for (const f of MINOR_HZ) {
      const x = Math.round(this.xOf(f, p)) + 0.5;
      ctx.strokeStyle = COLORS.gridMinor;
      ctx.beginPath();
      ctx.moveTo(x, p.y);
      ctx.lineTo(x, p.y + p.h);
      ctx.stroke();
    }
    ctx.textAlign = 'center';
    for (const f of MAJOR_HZ) {
      const x = Math.round(this.xOf(f, p)) + 0.5;
      ctx.strokeStyle = f === 1000 ? COLORS.gridStrong : COLORS.grid;
      ctx.beginPath();
      ctx.moveTo(x, p.y);
      ctx.lineTo(x, p.y + p.h);
      ctx.stroke();
      ctx.fillStyle = COLORS.labelDim;
      ctx.fillText(hzLabel(f), x, p.y + p.h + 9);
    }
    ctx.textAlign = 'right';
    for (let db = DB_MAX; db >= DB_MIN; db -= 10) {
      const y = Math.round(this.yOf(db, p)) + 0.5;
      ctx.strokeStyle = db % 20 === 0 ? COLORS.grid : COLORS.gridMinor;
      ctx.beginPath();
      ctx.moveTo(p.x, y);
      ctx.lineTo(p.x + p.w, y);
      ctx.stroke();
      if (db % 20 === 0 || db === DB_MIN) {
        ctx.fillStyle = COLORS.labelDim;
        ctx.fillText(`${db}`, p.x - 5, y);
      }
    }
    ctx.font = FONT_LABEL;
    ctx.textAlign = 'left';
    ctx.fillStyle = COLORS.label;
    ctx.fillText('dB', p.x - 28, p.y + 2);
    ctx.textAlign = 'right';
    ctx.fillText('Hz', p.x + p.w, p.y + p.h + 9);
    ctx.fillStyle = COLORS.labelDim;
    ctx.fillText('LTAS · 1/12 OCT', p.x + p.w - 4, p.y + 7);

    // clip to plot for curves
    ctx.save();
    ctx.beginPath();
    ctx.rect(p.x, p.y, p.w, p.h);
    ctx.clip();

    const hasCleaned = this.curves.cleaned !== null;
    const o = this.curves.original;
    const c = this.curves.cleaned;
    if (o) {
      const dim = hasCleaned ? (this.focus === 'original' ? 0.7 : 0.38) : 1;
      this.drawCurve(o.freqs, o.db, COLORS.amber, COLORS.amberLite, dim, p, true);
    }
    if (c) {
      const dim = this.focus === 'cleaned' ? 1 : 0.62;
      this.drawCurve(c.freqs, c.db, COLORS.cyan, COLORS.cyanLite, dim, p, true);
    }

    if (this.liveVisible > 0.001) {
      const col = this.liveDeck === 'cleaned' ? COLORS.cyan : COLORS.amber;
      const lite = this.liveDeck === 'cleaned' ? COLORS.cyanLite : COLORS.amberLite;
      this.drawCurve(BANDS, this.liveSmooth, col, lite, this.liveVisible, p, false, true);
    }
    ctx.restore();

    // inner bevel on plot edge
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.strokeRect(p.x + 0.5, p.y + 0.5, p.w - 1, p.h - 1);
  }

  private drawCurve(
    freqs: ArrayLike<number>,
    db: ArrayLike<number>,
    color: string,
    lite: string,
    alpha: number,
    p: { x: number; y: number; w: number; h: number },
    fill: boolean,
    live = false,
  ): void {
    const ctx = this.ctx;
    const n = Math.min(freqs.length, db.length);
    if (n < 2) return;

    // Build smoothed path (Catmull-Rom → bezier) through the band points.
    const xs = new Float32Array(n);
    const ys = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      xs[i] = this.xOf(freqs[i] ?? F_MIN, p);
      ys[i] = this.yOf(db[i] ?? DB_MIN, p);
    }
    const path = new Path2D();
    path.moveTo(xs[0] ?? 0, ys[0] ?? 0);
    for (let i = 0; i < n - 1; i++) {
      const x0 = xs[Math.max(0, i - 1)] ?? 0;
      const y0 = ys[Math.max(0, i - 1)] ?? 0;
      const x1 = xs[i] ?? 0;
      const y1 = ys[i] ?? 0;
      const x2 = xs[i + 1] ?? 0;
      const y2 = ys[i + 1] ?? 0;
      const x3 = xs[Math.min(n - 1, i + 2)] ?? 0;
      const y3 = ys[Math.min(n - 1, i + 2)] ?? 0;
      const cp1x = x1 + (x2 - x0) / 6;
      const cp1y = y1 + (y2 - y0) / 6;
      const cp2x = x2 - (x3 - x1) / 6;
      const cp2y = y2 - (y3 - y1) / 6;
      path.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, x2, y2);
    }

    ctx.globalAlpha = alpha;
    if (fill || live) {
      const area = new Path2D(path);
      area.lineTo(xs[n - 1] ?? 0, p.y + p.h);
      area.lineTo(xs[0] ?? 0, p.y + p.h);
      area.closePath();
      const g = ctx.createLinearGradient(0, p.y, 0, p.y + p.h);
      g.addColorStop(0, live ? this.rgba(color, 0.22) : this.rgba(color, 0.3));
      g.addColorStop(0.7, this.rgba(color, live ? 0.04 : 0.06));
      g.addColorStop(1, this.rgba(color, 0));
      ctx.fillStyle = g;
      ctx.fill(area);
    }
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    // glow passes
    ctx.strokeStyle = this.rgba(color, 0.12);
    ctx.lineWidth = live ? 5 : 7;
    ctx.stroke(path);
    ctx.strokeStyle = this.rgba(color, 0.22);
    ctx.lineWidth = live ? 2.5 : 3.5;
    ctx.stroke(path);
    // core line
    ctx.strokeStyle = live ? lite : color;
    ctx.lineWidth = live ? 1.1 : 1.5;
    ctx.stroke(path);
    ctx.globalAlpha = 1;
  }

  private rgbaCache = new Map<string, string>();
  private rgba(hex: string, a: number): string {
    const key = `${hex}/${a}`;
    const hit = this.rgbaCache.get(key);
    if (hit) return hit;
    const h = hex.replace('#', '');
    const n = parseInt(h, 16);
    const s = `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
    this.rgbaCache.set(key, s);
    return s;
  }
}
