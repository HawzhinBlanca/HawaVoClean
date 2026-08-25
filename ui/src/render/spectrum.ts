// Spectrum analyser display — main-thread Canvas 2D, driven by requestAnimationFrame.
//
// Three surfaces, composited back to front:
//
//   `base`   an opaque offscreen canvas holding everything that only changes
//            when the *data* changes: the inset display ground, the log-Hz/dB
//            grid and its labels, the gradient fills under each deck's
//            long-term average spectrum, the shaded REMOVED band between them,
//            the bloom around the curve strokes, and the curve cores. Rebuilt
//            only on new data / resize / theme change, then blitted once per
//            frame. That is what pays for the expensive fill and bloom work:
//            during playback the per-frame cost is one drawImage plus the live
//            trace, not a full re-render of the analysis.
//
//   `fx`     a transparent scratch the size of the display, used to composite
//            the two deck fills *correctly*: the cleaned deck's area is punched
//            out of the original's with `destination-out` before the cleaned
//            fill is laid in, so the overlap is never two washes stacked into
//            mud. What survives of the amber fill is exactly the band where the
//            cleaned curve sits below the original — the energy this app
//            removed — and it gets a hatch so it reads as a difference region
//            rather than as another deck.
//
//   `bloom`  a half-resolution transparent canvas that the curve strokes are
//            drawn into with widening, fading passes under `lighter`, then
//            composited additively (and upscaled, which softens it further).
//            Real bloom, not a halo: energy accumulates where curves overlap.
//
// On top of the blit, every frame draws the live AnalyserNode overlay — a
// thinner, brighter, faster trace than the LTAS curves, with fast-attack /
// slow-release ballistics and a peak-hold line that falls the full scale in
// ~1.5 s — and the hover readout.
//
// No React state is touched per frame, and a frame allocates nothing: every
// projection, gradient, pattern and label string is rebuilt only when the data,
// the canvas size or the theme changes.

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

/**
 * How far below the original the cleaned curve has to sit before the REMOVED
 * band is something you can actually see.
 *
 * The hatch pass uses 0.6 px of vertical separation. On the shipped plate the
 * plot is ~180 px tall over a 96 dB range, so 0.6 px is ~0.32 dB — expressed
 * in dB here because `removedCoverage()` is asked before the panel has
 * necessarily been laid out, and the answer must not depend on that.
 */
const REMOVED_MIN_DB = 0.32;

// AnalyserNode reports |X[k]| of a Blackman-windowed frame (coherent gain
// ≈ 0.42, one-sided), so a full-scale sine lands near −13.5 dB. The engine's
// long-term spectrum is referenced so that sine reads ≈ 0 dB; compensate.
const ANALYSER_OFFSET_DB = 13.5;

// Meter ballistics for the live overlay. The LTAS curves are an average over
// the whole file and never move; the live trace has to be obviously a
// different *kind* of reading, so it is quick enough to show syllables.
const ATTACK_S = 0.012; // fast attack — transients are visible
const RELEASE_S = 0.17; // release fast enough to track speech, slow enough not to strobe
const PEAK_HOLD_S = 0.35; // peak sits still this long before it starts falling
const PEAK_FALL_DB_S = DB_SPAN / 1.5; // then it crosses the whole scale in ~1.5 s
const FADE_IN_S = 0.18;
const FADE_OUT_S = 0.4;

/** Fill "hug": widening, fading strokes clipped to the area under a curve, so
 *  the wash is strongest against the curve itself and gone by the floor. */
const HUG: readonly (readonly [number, number])[] = [
  [2.5, 0.26],
  [7, 0.16],
  [17, 0.085],
  [38, 0.045],
  [80, 0.022],
];

/** Bloom passes, drawn into the half-res layer under `lighter`. */
const BLOOM: readonly (readonly [number, number])[] = [
  [12, 0.055],
  [6, 0.095],
  [2.8, 0.16],
  [1.3, 0.22],
];

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

/**
 * What the canvas paints with when a custom property cannot be resolved.
 *
 * These duplicate `styles/tokens.css`, because a canvas has no cascade to read
 * from. Duplicated values drift, and these had: --fg-3 and --fg-4 sat at
 * #6f7886 / #454c58 here — a generation older even than the #7c8593 / #545c69
 * that the D2 contrast pass rejected as illegible — long after tokens.css was
 * re-cut to #949daa / #86909f. `palette.test.ts` now pins every hex-valued
 * entry against tokens.css so the two cannot separate again.
 */
const FALLBACK: Record<string, string> = {
  '--amber': '#ffb347',
  '--amber-2': '#ffd28a',
  '--cyan': '#39d0ff',
  '--cyan-2': '#9ae8ff',
  '--display': '#07080a',
  '--display-2': '#0a0c0f',
  '--fg-2': '#aab2bf',
  '--fg-3': '#949daa',
  '--fg-4': '#86909f',
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

function mix(a: RGB, b: RGB, t: number): RGB {
  return [
    Math.round((a[0] ?? 0) + ((b[0] ?? 0) - (a[0] ?? 0)) * t),
    Math.round((a[1] ?? 0) + ((b[1] ?? 0) - (a[1] ?? 0)) * t),
    Math.round((a[2] ?? 0) + ((b[2] ?? 0) - (a[2] ?? 0)) * t),
  ];
}

const WHITE: RGB = [255, 255, 255];

interface Pen {
  base: RGB;
  core: string;
  /** live trace: the deck's hue pushed most of the way to white */
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

/** Does this browser honour `ctx.filter`? Chromium and Safari 17+ do; the
 *  bloom degrades to the multi-pass strokes alone where it does not. */
let filterSupport: boolean | null = null;
function supportsFilter(): boolean {
  if (filterSupport !== null) return filterSupport;
  try {
    const c = document.createElement('canvas').getContext('2d');
    if (!c) return (filterSupport = false);
    c.filter = 'blur(2px)';
    filterSupport = c.filter === 'blur(2px)';
  } catch {
    filterSupport = false;
  }
  return filterSupport;
}

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
  /** the cached `base` layer no longer matches the data */
  private staticDirty = true;

  // offscreen layers
  private base: HTMLCanvasElement | null = null;
  private baseCtx: CanvasRenderingContext2D | null = null;
  private fx: HTMLCanvasElement | null = null;
  private fxCtx: CanvasRenderingContext2D | null = null;
  private bloom: HTMLCanvasElement | null = null;
  private bloomCtx: CanvasRenderingContext2D | null = null;

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
  private hatch: CanvasPattern | null = null;
  private strokeOf: Record<SpectrumDeck, Pen>;

  // hover readout
  private cursorX: number | null = null;

  // frame instrumentation (read by perf probes; never by React)
  private frames = 0;
  private drawMsTotal = 0;
  private baseBuilds = 0;
  private baseMsTotal = 0;

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
    // Bound before syncSize(), because syncSize() -> invalidate() -> kick()
    // needs `this.loop` to already be the bound copy. Starting the loop with a
    // bare requestAnimationFrame here instead of kick() would leave `this.raf`
    // set by two different owners and leak a second, uncancellable loop.
    this.loop = this.loop.bind(this);
    this.syncSize();
    this.onPointerMove = this.onPointerMove.bind(this);
    this.onPointerLeave = this.onPointerLeave.bind(this);
    canvas.addEventListener('pointermove', this.onPointerMove);
    canvas.addEventListener('pointerleave', this.onPointerLeave);
    this.kick();
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
    const fg3 = col('--fg-3', [148, 157, 170]);
    const fg4 = col('--fg-4', [134, 144, 159]);
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
    // The live trace runs over the averaged one. Same deck, so the same hue —
    // but pushed most of the way to white and drawn a third thinner, which is
    // what makes an instantaneous reading read as instantaneous.
    const live = mix(lite, WHITE, 0.38);
    return {
      base,
      core: rgba(base, 0.98),
      live: rgba(live, 0.96),
      peak: rgba(live, 0.55),
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
    this.invalidate();
  }

  private invalidate(): void {
    this.staticDirty = true;
    this.dirty = true;
    this.kick();
  }

  private rebuildPaints(): void {
    const ctx = this.baseCtx ?? this.ctx;
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
    const fx = this.fxCtx ?? ctx;
    this.gFill = {
      original: this.areaGradient(fx, p, this.paints.amber),
      cleaned: this.areaGradient(fx, p, this.paints.cyan),
    };
    this.hatch = this.makeHatch(fx, this.paints.amber);
  }

  /** Body wash under a curve: present at the top of the plot, gone at the
   *  floor. The *hug* passes on top of it supply "strong near the curve". */
  private areaGradient(ctx: CanvasRenderingContext2D, p: Plot, c: RGB): CanvasGradient {
    const g = ctx.createLinearGradient(0, p.y, 0, p.y + p.h);
    g.addColorStop(0, rgba(c, 0.34));
    g.addColorStop(0.42, rgba(c, 0.15));
    g.addColorStop(0.78, rgba(c, 0.05));
    g.addColorStop(1, rgba(c, 0));
    return g;
  }

  /**
   * 45° hatch for the REMOVED band. Built at device resolution and handed back
   * a matrix that undoes the DPR scale, so the lines stay one device pixel
   * wide instead of turning into a blurred wash on a retina display.
   */
  private makeHatch(ctx: CanvasRenderingContext2D, c: RGB): CanvasPattern | null {
    const d = this.dpr;
    const tile = Math.max(4, Math.round(7 * d));
    const cv = document.createElement('canvas');
    cv.width = tile;
    cv.height = tile;
    const g = cv.getContext('2d');
    if (!g) return null;
    g.strokeStyle = rgba(c, 0.5);
    g.lineWidth = Math.max(1, Math.round(d * 0.6));
    g.beginPath();
    // two segments so the diagonal tiles seamlessly
    g.moveTo(-tile, tile);
    g.lineTo(tile, -tile);
    g.moveTo(0, 2 * tile);
    g.lineTo(2 * tile, 0);
    g.stroke();
    const pat = ctx.createPattern(cv, 'repeat');
    if (!pat) return null;
    if (typeof pat.setTransform === 'function' && typeof DOMMatrix === 'function') {
      pat.setTransform(new DOMMatrix([1 / d, 0, 0, 1 / d, 0, 0]));
    }
    return pat;
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
    // `getBoundingClientRect` reports the *painted* box, so a CSS transform
    // anywhere up the tree (a scaled overlay, a zoom animation, a
    // `scale()`-based transition on a parent panel) silently inflated the
    // backing store and left the renderer drawing several times the pixels it
    // needed. The layout box is what the canvas actually occupies, and
    // `offsetWidth`/`offsetHeight` report it untouched by any ancestor
    // transform.
    const w = Math.max(1, this.canvas.offsetWidth || 1);
    const h = Math.max(1, this.canvas.offsetHeight || 1);
    const dpr = window.devicePixelRatio || 1;
    if (w === this.cssW && h === this.cssH && dpr === this.dpr) return;
    this.cssW = w;
    this.cssH = h;
    this.dpr = dpr;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.ensureLayers();
    this.rebuildPaints();
    this.projectAll();
    this.invalidate();
  }

  /** (Re)allocate the offscreen layers for the current size / DPR. */
  private ensureLayers(): void {
    const d = this.dpr;
    const W = Math.round(this.cssW * d);
    const H = Math.round(this.cssH * d);
    if (!this.base) {
      this.base = document.createElement('canvas');
      this.baseCtx = this.base.getContext('2d', { alpha: false });
    }
    if (!this.fx) {
      this.fx = document.createElement('canvas');
      this.fxCtx = this.fx.getContext('2d');
    }
    if (!this.bloom) {
      this.bloom = document.createElement('canvas');
      this.bloomCtx = this.bloom.getContext('2d');
    }
    if (this.base) {
      this.base.width = W;
      this.base.height = H;
    }
    if (this.fx) {
      this.fx.width = W;
      this.fx.height = H;
    }
    if (this.bloom) {
      this.bloom.width = Math.max(1, Math.round(W / 2));
      this.bloom.height = Math.max(1, Math.round(H / 2));
    }
    if (this.baseCtx) {
      this.baseCtx.imageSmoothingEnabled = true;
      this.baseCtx.imageSmoothingQuality = 'high';
    }
  }

  // ---- data ---------------------------------------------------------------

  setCurve(deck: SpectrumDeck, curve: SpectrumCurve | null): void {
    this.curves[deck] = curve;
    this.proj[deck] = this.project(curve);
    this.rebuildDiff();
    this.invalidate();
  }

  /**
   * How much of the frequency axis the REMOVED band actually covers, 0..1.
   *
   * The band is not a series anyone decides to draw: it is whatever survives
   * the `destination-out` punch in `renderFills`, i.e. the frequencies where
   * the cleaned curve sits below the original. A run that took nothing out —
   * or one that came back louder everywhere, which is what a normalise-only
   * pass looks like — paints no amber at all. The key beside the plot and the
   * canvas's accessible name ask this, so that what they claim and what the
   * pixels show are the same fact rather than two independent guesses.
   *
   * Measured in dB on the curves themselves rather than in pixels on the
   * projection, so it is answerable the moment the data lands.
   */
  removedCoverage(): number {
    const o = this.curves.original;
    const c = this.curves.cleaned;
    if (!o || !c) return 0;
    const n = Math.min(o.freqs.length, o.db.length);
    if (n < 2) return 0;
    const sameGrid = c.freqs.length === o.freqs.length;
    const at = (i: number): number => {
      const dbO = o.db[i] ?? DB_MIN;
      const dbC = sameGrid
        ? (c.db[i] ?? DB_MIN)
        : SpectrumRenderer.sampleAt(c, o.freqs[i] ?? F_MIN);
      return dbO - dbC;
    };
    // Coverage is width on the log-frequency axis the plot draws, so a decade
    // of removed hiss up top does not count for less than a decade down low.
    let span = 0;
    let prev = at(0);
    for (let i = 1; i < n; i++) {
      const cur = at(i);
      if (prev > REMOVED_MIN_DB && cur > REMOVED_MIN_DB) {
        const f0 = Math.max(F_MIN, Math.min(F_MAX, o.freqs[i - 1] ?? F_MIN));
        const f1 = Math.max(F_MIN, Math.min(F_MAX, o.freqs[i] ?? F_MIN));
        if (f1 > f0) span += Math.log2(f1 / f0);
      }
      prev = cur;
    }
    return Math.max(0, Math.min(1, span / LOG_SPAN));
  }

  setFocus(deck: SpectrumDeck): void {
    if (this.focus !== deck) {
      this.focus = deck;
      this.invalidate();
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
      analyser.smoothingTimeConstant = 0.25;
      if (!this.liveBuf || this.liveBuf.length !== analyser.frequencyBinCount) {
        this.liveBuf = new Float32Array(analyser.frequencyBinCount);
      }
    }
    this.dirty = true;
    this.kick();
  }

  dispose(): void {
    cancelAnimationFrame(this.raf);
    this.raf = 0;
    this.ro.disconnect();
    this.mql?.removeEventListener('change', this.onDprChange);
    this.canvas.removeEventListener('pointermove', this.onPointerMove);
    this.canvas.removeEventListener('pointerleave', this.onPointerLeave);
    // release the backing stores rather than waiting for the GC to notice
    // three canvases per disposed renderer
    for (const c of [this.base, this.fx, this.bloom]) {
      if (c) {
        c.width = 0;
        c.height = 0;
      }
    }
    this.base = this.fx = this.bloom = null;
    this.baseCtx = this.fxCtx = this.bloomCtx = null;
  }

  /** Mean cost of a drawn frame in ms, and how many frames have been drawn. */
  get stats(): { frames: number; meanDrawMs: number; baseBuilds: number; meanBaseMs: number } {
    return {
      frames: this.frames,
      meanDrawMs: this.frames ? this.drawMsTotal / this.frames : 0,
      baseBuilds: this.baseBuilds,
      meanBaseMs: this.baseBuilds ? this.baseMsTotal / this.baseBuilds : 0,
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
    // `offsetX` is in the target's own layout coordinates, so — unlike
    // clientX minus a bounding rect — it survives an ancestor transform.
    const x = Number.isFinite(e.offsetX)
      ? e.offsetX
      : e.clientX - this.canvas.getBoundingClientRect().left;
    const p = this.plot;
    const next = x >= p.x - 2 && x <= p.x + p.w + 2 ? Math.min(p.x + p.w, Math.max(p.x, x)) : null;
    if (next !== this.cursorX) {
      this.cursorX = next;
      this.dirty = true;
      this.kick();
    }
  }

  private onPointerLeave(): void {
    if (this.cursorX !== null) {
      this.cursorX = null;
      this.dirty = true;
      this.kick();
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

  /**
   * Start the frame loop if it is not already running.
   *
   * Every path that makes the display need a frame calls this. The loop parks
   * itself when there is nothing left to do (see `loop`), so an idle app with
   * a clip loaded is not holding a 60 Hz callback open to re-check five
   * booleans — which is what it did before, for as long as the window was
   * open.
   */
  private kick(): void {
    if (this.raf) return;
    // A parked loop has no notion of when it stopped. Without clearing this,
    // the first frame after minutes of idle would see a huge `ts - lastT`,
    // clamp it to 100 ms, and advance the live fade and ballistics by that
    // whole step — a visible jump on the exact transition that woke it.
    this.lastT = 0;
    this.raf = requestAnimationFrame(this.loop);
  }

  private loop(ts: number): void {
    const dt = this.lastT ? Math.min(0.1, (ts - this.lastT) / 1000) : 1 / 60;
    this.lastT = ts;
    const animating = this.liveActive || this.liveVisible > 0;
    if (animating) this.step(dt);
    if (!animating && !this.dirty && !this.staticDirty) {
      // Nothing to draw and nothing decaying: park. `kick()` restarts us.
      this.raf = 0;
      return;
    }
    this.raf = requestAnimationFrame(this.loop);
    this.dirty = false;
    this.draw();
  }

  // ---- draw ---------------------------------------------------------------

  private draw(): void {
    const t0 = performance.now();
    if (this.staticDirty) this.renderBase();
    const ctx = this.ctx;
    const W = this.cssW;
    const H = this.cssH;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1;

    if (this.base) ctx.drawImage(this.base, 0, 0, W, H);
    else {
      ctx.fillStyle = this.paints.bgBot;
      ctx.fillRect(0, 0, W, H);
    }

    const p = this.plot;
    const hasO = this.proj.original.n > 1;
    const hasC = this.proj.cleaned.n > 1;

    if (this.liveVisible > 0.001) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(p.x, p.y, p.w, p.h);
      ctx.clip();
      this.drawLive(p);
      ctx.restore();
    }

    if (this.cursorX !== null && (hasO || hasC)) this.drawReadout(p);

    this.frames++;
    this.drawMsTotal += performance.now() - t0;
  }

  /** Everything that only changes when the analysis, the size or the theme does. */
  private renderBase(): void {
    const ctx = this.baseCtx;
    if (!ctx || !this.base) {
      this.staticDirty = false;
      return;
    }
    const t0 = performance.now();
    this.staticDirty = false;
    const W = this.cssW;
    const H = this.cssH;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1;
    ctx.fillStyle = this.gBg ?? this.paints.bgBot;
    ctx.fillRect(0, 0, W, H);

    const p = this.plot;
    this.drawGrid(ctx, p);

    const o = this.proj.original;
    const c = this.proj.cleaned;
    const hasO = o.n > 1;
    const hasC = c.n > 1;

    if (hasO || hasC) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(p.x, p.y, p.w, p.h);
      ctx.clip();
      this.renderFills(p, hasO, hasC);
      this.renderBloom(hasO, hasC);
      // crisp cores last, over their own bloom
      if (hasO) this.strokeCore(ctx, o, 'original', !hasC || this.focus === 'original' ? 1 : 0.66);
      if (hasC) this.strokeCore(ctx, c, 'cleaned', this.focus === 'cleaned' ? 1 : 0.76);
      ctx.restore();
    }

    ctx.fillStyle = this.gVignette ?? 'rgba(0,0,0,0)';
    ctx.fillRect(0, 0, W, H);

    if (!hasO && !hasC) this.drawEmptyState(ctx, p);

    // inner bevel around the plot
    ctx.lineWidth = 1 / this.dpr;
    ctx.strokeStyle = this.paints.frame;
    ctx.strokeRect(this.snap(p.x), this.snap(p.y), p.w, p.h);

    this.baseBuilds++;
    this.baseMsTotal += performance.now() - t0;
  }

  // ---- fills --------------------------------------------------------------

  /**
   * Gradient fill under each curve, composited so the overlap is never two
   * washes stacked on each other:
   *
   *   1. the original's fill goes down over its whole area;
   *   2. the cleaned deck's area is punched out of it (`destination-out`) —
   *      so whatever amber survives is exactly the band where cleaned sits
   *      *below* original, which is the energy this app removed;
   *   3. the cleaned fill is laid into the hole it just made;
   *   4. the surviving amber band gets a 45° hatch, so it reads as a
   *      difference region and not as a third deck.
   *
   * All four steps happen on the transparent `fx` layer, because step 2 needs
   * a real alpha channel; the result is composited onto the opaque base once.
   */
  private renderFills(p: Plot, hasO: boolean, hasC: boolean): void {
    const fx = this.fxCtx;
    const base = this.baseCtx;
    if (!fx || !this.fx || !base) return;
    fx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    fx.globalCompositeOperation = 'source-over';
    fx.globalAlpha = 1;
    fx.clearRect(0, 0, this.cssW, this.cssH);

    if (hasO) this.fillDeck(fx, this.proj.original, 'original', p);
    if (hasC) {
      if (hasO) {
        fx.save();
        fx.globalCompositeOperation = 'destination-out';
        fx.fillStyle = '#000';
        this.traceArea(fx, this.proj.cleaned, p);
        fx.fill();
        fx.restore();
      }
      this.fillDeck(fx, this.proj.cleaned, 'cleaned', p);
    }
    if (hasO && hasC) this.hatchRemoved(fx);

    base.drawImage(this.fx, 0, 0, this.cssW, this.cssH);
  }

  private fillDeck(
    fx: CanvasRenderingContext2D,
    proj: Projection,
    deck: SpectrumDeck,
    p: Plot,
  ): void {
    fx.save();
    this.traceArea(fx, proj, p);
    fx.clip();
    fx.fillStyle = this.gFill[deck] ?? 'rgba(0,0,0,0)';
    fx.fillRect(p.x, p.y, p.w, p.h);
    // widening, fading strokes on the curve itself, clipped to the area under
    // it — the wash is at full strength against the curve and gone by the floor
    fx.globalCompositeOperation = 'lighter';
    fx.lineJoin = 'round';
    fx.lineCap = 'round';
    const c = this.strokeOf[deck].base;
    for (const pass of HUG) {
      this.tracePath(fx, proj.xs, proj.ys, proj.n);
      fx.lineWidth = pass[0] ?? 1;
      fx.strokeStyle = rgba(c, pass[1] ?? 0);
      fx.stroke();
    }
    fx.restore();
  }

  /** 45° hatch over the surviving amber band. */
  private hatchRemoved(fx: CanvasRenderingContext2D): void {
    const n = this.diffN;
    const pat = this.hatch;
    if (n < 2 || !pat) return;
    const xs = this.diffX;
    const yo = this.diffYo;
    const yc = this.diffYc;
    fx.save();
    fx.fillStyle = pat;
    fx.globalAlpha = 0.5;
    let i = 0;
    while (i < n) {
      while (i < n && (yc[i] ?? 0) - (yo[i] ?? 0) <= 0.6) i++;
      if (i >= n) break;
      let s = i;
      while (s > 0 && (yc[s - 1] ?? 0) - (yo[s - 1] ?? 0) > 0) s--;
      while (i < n && (yc[i] ?? 0) - (yo[i] ?? 0) > 0) i++;
      const e = i - 1;
      if (e - s < 1) continue;
      fx.beginPath();
      fx.moveTo(xs[s] ?? 0, yo[s] ?? 0);
      for (let k = s + 1; k <= e; k++) fx.lineTo(xs[k] ?? 0, yo[k] ?? 0);
      for (let k = e; k >= s; k--) fx.lineTo(xs[k] ?? 0, yc[k] ?? 0);
      fx.closePath();
      fx.fill();
    }
    fx.restore();
  }

  // ---- bloom --------------------------------------------------------------

  /**
   * Half-resolution additive bloom. The strokes go down in widening, fading
   * passes under `lighter`, optionally through a real gaussian (`ctx.filter`)
   * where the browser has one, then the whole layer is composited additively
   * and upscaled — which softens it once more for free.
   */
  private renderBloom(hasO: boolean, hasC: boolean): void {
    const b = this.bloomCtx;
    const base = this.baseCtx;
    if (!b || !this.bloom || !base) return;
    const s = this.dpr / 2;
    b.setTransform(s, 0, 0, s, 0, 0);
    b.globalCompositeOperation = 'source-over';
    b.globalAlpha = 1;
    b.filter = 'none';
    b.clearRect(0, 0, this.cssW, this.cssH);
    b.globalCompositeOperation = 'lighter';
    b.lineJoin = 'round';
    b.lineCap = 'round';
    if (supportsFilter()) b.filter = 'blur(1.6px)';

    const decks: SpectrumDeck[] = [];
    if (hasO) decks.push('original');
    if (hasC) decks.push('cleaned');
    for (const deck of decks) {
      const proj = this.proj[deck];
      const c = this.strokeOf[deck].base;
      const lit = deck === this.focus || decks.length === 1;
      const k = lit ? 1 : 0.6;
      for (const pass of BLOOM) {
        this.tracePath(b, proj.xs, proj.ys, proj.n);
        b.lineWidth = pass[0] ?? 1;
        b.strokeStyle = rgba(c, (pass[1] ?? 0) * k);
        b.stroke();
      }
    }
    b.filter = 'none';

    base.save();
    base.globalCompositeOperation = 'lighter';
    base.drawImage(this.bloom, 0, 0, this.cssW, this.cssH);
    base.restore();
  }

  // ---- primitives ---------------------------------------------------------

  private drawGrid(ctx: CanvasRenderingContext2D, p: Plot): void {
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
  private tracePath(
    ctx: CanvasRenderingContext2D,
    xs: Float32Array,
    ys: Float32Array,
    n: number,
  ): void {
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

  /** The same path, closed down to the floor — the area under the curve. */
  private traceArea(ctx: CanvasRenderingContext2D, proj: Projection, p: Plot): void {
    const { n, xs, ys } = proj;
    this.tracePath(ctx, xs, ys, n);
    ctx.lineTo(xs[n - 1] ?? 0, p.y + p.h);
    ctx.lineTo(xs[0] ?? 0, p.y + p.h);
    ctx.closePath();
  }

  private strokeCore(
    ctx: CanvasRenderingContext2D,
    proj: Projection,
    deck: SpectrumDeck,
    alpha: number,
  ): void {
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    this.tracePath(ctx, proj.xs, proj.ys, proj.n);
    ctx.strokeStyle = this.strokeOf[deck].core;
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.restore();
  }

  private drawLive(p: Plot): void {
    const ctx = this.ctx;
    const n = BANDS.length;
    for (let i = 0; i < n; i++) {
      this.liveYs[i] = this.yOf(this.liveDb[i] ?? DB_MIN, p);
      this.livePeakYs[i] = this.yOf(this.livePeak[i] ?? DB_MIN, p);
    }
    const pen = this.strokeOf[this.liveDeck];
    const a = this.liveVisible;

    ctx.save();
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    // a tight additive halo so the hairline still reads over a filled curve
    ctx.globalCompositeOperation = 'lighter';
    ctx.globalAlpha = a;
    this.tracePath(ctx, this.liveXs, this.liveYs, n);
    ctx.strokeStyle = rgba(pen.base, 0.16);
    ctx.lineWidth = 5;
    ctx.stroke();
    this.tracePath(ctx, this.liveXs, this.liveYs, n);
    ctx.strokeStyle = rgba(pen.base, 0.22);
    ctx.lineWidth = 2.4;
    ctx.stroke();

    // the trace itself: a third thinner than an LTAS core and near white
    ctx.globalCompositeOperation = 'source-over';
    this.tracePath(ctx, this.liveXs, this.liveYs, n);
    ctx.strokeStyle = pen.live;
    ctx.lineWidth = 1;
    ctx.stroke();

    // peak hold
    ctx.globalAlpha = a * 0.8;
    this.tracePath(ctx, this.liveXs, this.livePeakYs, n);
    ctx.strokeStyle = pen.peak;
    // the peak line stays a hair thinner than the live trace
    ctx.lineWidth = this.dpr >= 2 ? 0.75 : 1;
    ctx.stroke();
    ctx.restore();
  }

  private drawEmptyState(ctx: CanvasRenderingContext2D, p: Plot): void {
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
