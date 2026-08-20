// Message protocol between the main thread and the waveform worker.

export type WaveKind = 'original' | 'cleaned';

/** Linear sRGB triple, 0..1 — the form the shaders want. */
export type WaveRgb = [number, number, number];

/** One deck's three-stop ramp: centre-line core, peak edge, and RMS core. */
export interface WaveDeckColors {
  core: WaveRgb;
  edge: WaveRgb;
  rms: WaveRgb;
}

/**
 * Every colour the renderer uses, resolved from CSS custom properties on the
 * main thread (the worker has no DOM, so it can never read them itself).
 * `WaveformHost` re-reads and re-posts these whenever the theme changes.
 */
export interface WavePalette {
  bgTop: WaveRgb;
  bgBottom: WaveRgb;
  grid: WaveRgb;
  unit: WaveRgb;
  highlight: WaveRgb;
  playhead: WaveRgb;
  original: WaveDeckColors;
  cleaned: WaveDeckColors;
}

/** Fallbacks matching `tokens.css` at the time of writing — never a blank display. */
export const DEFAULT_WAVE_PALETTE: WavePalette = {
  bgTop: [0.043, 0.051, 0.063],
  bgBottom: [0.024, 0.027, 0.034],
  grid: [1, 1, 1],
  unit: [0.6, 0.75, 0.95],
  highlight: [1, 1, 1],
  playhead: [1, 1, 1],
  original: {
    core: [1.0, 0.855, 0.62], // #ffdb9e
    edge: [1.0, 0.702, 0.278], // #ffb347
    rms: [1.0, 0.945, 0.855], // #fff1da
  },
  cleaned: {
    core: [0.604, 0.91, 1.0], // #9ae8ff
    edge: [0.224, 0.816, 1.0], // #39d0ff
    rms: [0.85, 0.965, 1.0], // #d9f6ff
  },
};

/**
 * `base` is the whole-file envelope from `POST /api/analyze` (always present
 * once a clip is analyzed); `detail` is a windowed envelope from
 * `POST /api/peaks` covering exactly the visible range at display resolution.
 * The worker draws `detail` whenever it still covers the view, else `base`, so
 * a view change always redraws immediately from data already in hand.
 */
export type WaveSlot = 'base' | 'detail';

export interface WaveInitMsg {
  type: 'init';
  canvas: OffscreenCanvas;
  width: number;
  height: number;
  dpr: number;
  /** Optional so the first paint already uses the theme's colours. */
  palette?: WavePalette;
}
export interface WaveResizeMsg {
  type: 'resize';
  width: number;
  height: number;
  dpr: number;
}
export interface WaveDataMsg {
  type: 'data';
  kind: WaveKind;
  slot: WaveSlot;
  min: Float32Array;
  max: Float32Array;
  rms: Float32Array | null; // linear amplitude 0..1 (already converted from dB)
  /** Time span the buckets cover, in seconds. */
  start: number;
  end: number;
}
export interface WaveClearMsg {
  type: 'clear';
  kind: WaveKind;
  slot?: WaveSlot; // omitted = clear both slots
}
export interface WavePlayheadMsg {
  type: 'playhead';
  time: number;
  visible: boolean;
}
export interface WaveHoverMsg {
  type: 'hover';
  x: number | null; // CSS px within the canvas
}
/**
 * The lit range, and — on a multi-channel report — which channel's range it
 * is. `lane` is the channel's index among the report's channels and `lanes`
 * how many there are; the band is then drawn in that horizontal slice of the
 * display instead of across the whole of it, so a ch0 unit and the ch1 unit
 * overlapping it in time cannot paint the same pixels. Omitted for mono, which
 * keeps the full-height band it has always had.
 */
export interface WaveHighlightRange {
  start: number;
  end: number;
  lane?: number;
  lanes?: number;
}

export interface WaveHighlightMsg {
  type: 'highlight';
  range: WaveHighlightRange | null;
}
export interface WaveUnitsMsg {
  type: 'units';
  bounds: Float32Array; // unit boundary times (seconds), sorted
}
export interface WaveFocusMsg {
  type: 'focus';
  kind: WaveKind;
}
/** Visible time window (seconds). Drives every horizontal mapping. */
export interface WaveViewMsg {
  type: 'view';
  start: number;
  end: number;
}
/**
 * Ask the worker for its render-cost numbers. The worker keeps them anyway
 * (one `performance.now()` pair per frame); this only asks it to say them.
 */
export interface WaveStatsReqMsg {
  type: 'stats';
}
/** Palette re-read from CSS custom properties (init, and on theme change). */
export interface WaveThemeMsg {
  type: 'theme';
  palette: WavePalette;
}

export type WaveMsg =
  | WaveInitMsg
  | WaveResizeMsg
  | WaveDataMsg
  | WaveClearMsg
  | WavePlayheadMsg
  | WaveHoverMsg
  | WaveHighlightMsg
  | WaveUnitsMsg
  | WaveFocusMsg
  | WaveViewMsg
  | WaveThemeMsg
  | WaveStatsReqMsg;

export interface WaveReadyMsg {
  type: 'ready';
  webgl2: boolean;
}
export interface WaveErrorMsg {
  type: 'error';
  message: string;
}
/**
 * What one worker frame actually costs (goal box C1). Times are milliseconds
 * of worker-thread work inside `render()`, i.e. geometry rebuild + all GL
 * calls; `p95`/`max` are over the last `window` frames, `maxAll` over every
 * frame since init. Posted at most every 500 ms, and only when something was
 * drawn since the last one.
 */
export interface WaveStatsMsg {
  type: 'stats';
  frames: number;
  last: number;
  mean: number;
  p95: number;
  max: number;
  maxAll: number;
  window: number;
  /** Worker-side interval between rendered frames, ms (median over `window`). */
  interval: number;
}
export type WaveOutMsg = WaveReadyMsg | WaveErrorMsg | WaveStatsMsg;
