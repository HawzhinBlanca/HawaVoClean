// Main-thread side of the waveform renderer: owns the Worker, transfers the
// OffscreenCanvas, and exposes imperative setters. No React here.

import type { WaveKind, WaveMsg, WaveOutMsg, WaveSlot } from './waveformProtocol';

export interface WaveformHostOptions {
  onReady?: (webgl2: boolean) => void;
  onError?: (message: string) => void;
}

export class WaveformHost {
  private worker: Worker | null = null;
  private ro: ResizeObserver | null = null;
  private canvas: HTMLCanvasElement;
  private lastW = 0;
  private lastH = 0;
  private lastDpr = 0;
  private mql: MediaQueryList | null = null;
  private readonly onMql = (): void => this.syncSize();

  constructor(canvas: HTMLCanvasElement, opts: WaveformHostOptions = {}) {
    this.canvas = canvas;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
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
      { type: 'init', canvas: offscreen, width: this.lastW, height: this.lastH, dpr } satisfies WaveMsg,
      [offscreen],
    );
    this.ro = new ResizeObserver(() => this.syncSize());
    this.ro.observe(canvas);
    this.watchDpr();
  }

  private watchDpr(): void {
    if (this.mql) this.mql.removeEventListener('change', this.onMql);
    const dpr = window.devicePixelRatio || 1;
    this.mql = window.matchMedia(`(resolution: ${dpr}dppx)`);
    this.mql.addEventListener('change', this.onMql);
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
    this.worker?.terminate();
    this.worker = null;
  }
}
