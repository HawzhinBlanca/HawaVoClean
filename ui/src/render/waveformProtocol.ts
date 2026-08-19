// Message protocol between the main thread and the waveform worker.

export type WaveKind = 'original' | 'cleaned';

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
  | WaveViewMsg;

export interface WaveReadyMsg {
  type: 'ready';
  webgl2: boolean;
}
export interface WaveErrorMsg {
  type: 'error';
  message: string;
}
export type WaveOutMsg = WaveReadyMsg | WaveErrorMsg;
