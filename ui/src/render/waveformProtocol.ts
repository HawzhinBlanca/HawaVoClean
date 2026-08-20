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
export interface WaveHighlightMsg {
  type: 'highlight';
  range: { start: number; end: number } | null;
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
  | WaveThemeMsg;

export interface WaveReadyMsg {
  type: 'ready';
  webgl2: boolean;
}
export interface WaveErrorMsg {
  type: 'error';
  message: string;
}
export type WaveOutMsg = WaveReadyMsg | WaveErrorMsg;
