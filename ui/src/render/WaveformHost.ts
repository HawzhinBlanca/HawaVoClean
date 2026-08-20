// Main-thread side of the waveform renderer: owns the Worker, transfers the
// OffscreenCanvas, and exposes imperative setters. No React here.

import { mixRgb, parseCssRgb } from './glutil';
import {
  DEFAULT_WAVE_PALETTE,
  type WaveDeckColors,
  type WaveKind,
  type WaveMsg,
  type WaveOutMsg,
  type WavePalette,
  type WaveRgb,
  type WaveSlot,
} from './waveformProtocol';

export interface WaveformHostOptions {
  onReady?: (webgl2: boolean) => void;
  onError?: (message: string) => void;
}

const WHITE: WaveRgb = [1, 1, 1];

/**
 * Resolve the display palette from CSS custom properties.
 *
 * The worker has no DOM, so every colour it draws with is read here and posted
 * across. A hidden probe element is used rather than reading the raw token
 * text because the browser only resolves `var()` chains and non-hex colour
 * syntaxes when they are used as a real property value.
 */
function readPalette(anchor: Element): WavePalette {
  const doc = anchor.ownerDocument;
  const view = doc.defaultView;
  const host = anchor.parentElement ?? doc.body;
  if (!view || !host) return DEFAULT_WAVE_PALETTE;
  const probe = doc.createElement('span');
  probe.setAttribute('aria-hidden', 'true');
  probe.style.cssText =
    'position:absolute;left:0;top:0;width:0;height:0;opacity:0;pointer-events:none;contain:strict';
  host.appendChild(probe);
  try {
    const cs = view.getComputedStyle(probe);
    const read = (expr: string, fb: WaveRgb): WaveRgb => {
      probe.style.color = expr;
      return parseCssRgb(cs.color, fb);
    };
    const defined = (name: string): boolean => cs.getPropertyValue(name).trim() !== '';
    const deck = (
      name: string,
      edgeFallback: string,
      coreFallback: string,
      dflt: WaveDeckColors,
    ): WaveDeckColors => {
      const edge = read(`var(--wave-${name}, ${edgeFallback})`, dflt.edge);
      const core = read(`var(--wave-${name}-core, ${coreFallback})`, dflt.core);
      const rmsVar = `--wave-${name}-rms`;
      const rms = defined(rmsVar) ? read(`var(${rmsVar})`, dflt.rms) : mixRgb(core, WHITE, 0.2);
      return { core, edge, rms };
    };
    const d = DEFAULT_WAVE_PALETTE;
    return {
      bgTop: read('var(--wave-bg-top, var(--display-2, #0a0c0f))', d.bgTop),
      bgBottom: read('var(--wave-bg-bottom, var(--display, #07080a))', d.bgBottom),
      grid: read('var(--wave-grid, var(--fg, #e6e9ee))', d.grid),
      unit: read('var(--wave-unit, var(--fg-2, #aab2bf))', d.unit),
      highlight: read('var(--wave-highlight, var(--fg, #e6e9ee))', d.highlight),
      playhead: read('var(--wave-playhead, var(--fg, #e6e9ee))', d.playhead),
      original: deck('original', 'var(--amber, #ffb347)', 'var(--amber-2, #ffd28a)', d.original),
      cleaned: deck('cleaned', 'var(--cyan, #39d0ff)', 'var(--cyan-2, #9ae8ff)', d.cleaned),
    };
  } catch {
    return DEFAULT_WAVE_PALETTE;
  } finally {
    probe.remove();
  }
}

function samePalette(a: WavePalette, b: WavePalette): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

export class WaveformHost {
  private worker: Worker | null = null;
  private ro: ResizeObserver | null = null;
  private canvas: HTMLCanvasElement;
  private lastW = 0;
  private lastH = 0;
  private lastDpr = 0;
  private mql: MediaQueryList | null = null;
  private scheme: MediaQueryList | null = null;
  private themeObs: MutationObserver | null = null;
  private themeRaf = 0;
  private palette: WavePalette;
  private readonly onMql = (): void => this.syncSize();
  private readonly onTheme = (): void => this.scheduleThemeSync();

  constructor(canvas: HTMLCanvasElement, opts: WaveformHostOptions = {}) {
    this.canvas = canvas;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.palette = readPalette(canvas);
    const offscreen = canvas.transferControlToOffscreen();
    // Dev: Vite serves the worker as an ES module (it has imports). Build: the
    // worker is bundled into one classic script, which also loads from file://
    // inside Electron without module-script CORS rules.
    this.worker = import.meta.env.DEV
      ? new Worker(new URL('./waveform.worker.ts', import.meta.url), { type: 'module' })
      : new Worker(new URL('./waveform.worker.ts', import.meta.url));
    this.worker.onmessage = (ev: MessageEvent<WaveOutMsg>) => {
      const m = ev.data;
      if (m.type === 'ready') opts.onReady?.(m.webgl2);
      else if (m.type === 'error') opts.onError?.(m.message);
    };
    this.worker.onerror = (ev) => {
      opts.onError?.(ev.message || 'waveform worker failed');
    };
    this.lastW = Math.max(1, Math.round(rect.width));
    this.lastH = Math.max(1, Math.round(rect.height));
    this.lastDpr = dpr;
    this.worker.postMessage(
      {
        type: 'init',
        canvas: offscreen,
        width: this.lastW,
        height: this.lastH,
        dpr,
        palette: this.palette,
      } satisfies WaveMsg,
      [offscreen],
    );
    this.ro = new ResizeObserver(() => this.syncSize());
    this.ro.observe(canvas);
    this.watchDpr();
    this.watchTheme();
  }

  private watchDpr(): void {
    if (this.mql) this.mql.removeEventListener('change', this.onMql);
    const dpr = window.devicePixelRatio || 1;
    this.mql = window.matchMedia(`(resolution: ${dpr}dppx)`);
    this.mql.addEventListener('change', this.onMql);
  }

  /**
   * Re-read the palette when the theme can have changed: a class/attribute flip
   * on <html> (how a theme switch is normally expressed) or the OS scheme
   * preference. Reads are coalesced to one per frame and only posted when the
   * resolved colours actually differ.
   */
  private watchTheme(): void {
    this.scheme = window.matchMedia('(prefers-color-scheme: dark)');
    this.scheme.addEventListener('change', this.onTheme);
    this.themeObs = new MutationObserver(this.onTheme);
    this.themeObs.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class', 'style', 'data-theme', 'data-appearance'],
    });
  }

  private scheduleThemeSync(): void {
    if (this.themeRaf) return;
    this.themeRaf = window.requestAnimationFrame(() => {
      this.themeRaf = 0;
      this.syncTheme();
    });
  }

  private syncTheme(): void {
    if (!this.worker) return;
    const next = readPalette(this.canvas);
    if (samePalette(next, this.palette)) return;
    this.palette = next;
    this.post({ type: 'theme', palette: next });
  }

  /** Force a palette re-read (e.g. after a stylesheet swap). */
  refreshTheme(): void {
    this.syncTheme();
  }

  private syncSize(): void {
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width));
    const h = Math.max(1, Math.round(rect.height));
    const dpr = window.devicePixelRatio || 1;
    if (w === this.lastW && h === this.lastH && dpr === this.lastDpr) return;
    this.lastW = w;
    this.lastH = h;
    if (dpr !== this.lastDpr) {
      this.lastDpr = dpr;
      this.watchDpr();
    }
    this.post({ type: 'resize', width: w, height: h, dpr });
  }

  private post(msg: WaveMsg, transfer?: Transferable[]): void {
    if (!this.worker) return;
    if (transfer) this.worker.postMessage(msg, transfer);
    else this.worker.postMessage(msg);
  }

  /**
   * Hand the worker an envelope. `start`/`end` are the seconds the buckets
   * cover: the whole file for `base`, the visible window for `detail`.
   * The typed arrays are transferred, so pass copies of anything cached.
   */
  setData(
    kind: WaveKind,
    slot: WaveSlot,
    min: Float32Array,
    max: Float32Array,
    rms: Float32Array | null,
    start: number,
    end: number,
  ): void {
    const transfer: Transferable[] = [min.buffer, max.buffer];
    if (rms) transfer.push(rms.buffer);
    this.post({ type: 'data', kind, slot, min, max, rms, start, end }, transfer);
  }

  clear(kind: WaveKind, slot?: WaveSlot): void {
    this.post(slot ? { type: 'clear', kind, slot } : { type: 'clear', kind });
  }

  /** Visible time window in seconds — the only horizontal mapping the worker uses. */
  setView(start: number, end: number): void {
    this.post({ type: 'view', start, end });
  }

  setPlayhead(time: number, visible: boolean): void {
    this.post({ type: 'playhead', time, visible });
  }

  setHover(x: number | null): void {
    this.post({ type: 'hover', x });
  }

  setHighlight(range: { start: number; end: number } | null): void {
    this.post({ type: 'highlight', range });
  }

  setUnits(bounds: Float32Array): void {
    this.post({ type: 'units', bounds }, [bounds.buffer]);
  }

  setFocus(kind: WaveKind): void {
    this.post({ type: 'focus', kind });
  }

  get width(): number {
    return this.lastW;
  }

  dispose(): void {
    this.ro?.disconnect();
    this.ro = null;
    if (this.mql) this.mql.removeEventListener('change', this.onMql);
    this.mql = null;
    if (this.scheme) this.scheme.removeEventListener('change', this.onTheme);
    this.scheme = null;
    this.themeObs?.disconnect();
    this.themeObs = null;
    if (this.themeRaf) window.cancelAnimationFrame(this.themeRaf);
    this.themeRaf = 0;
    this.worker?.terminate();
    this.worker = null;
  }
}
