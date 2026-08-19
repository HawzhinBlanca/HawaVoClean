// Message protocol between the main thread and the waveform worker.

export type WaveKind = 'original' | 'cleaned';

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
  min: Float32Array;
  max: Float32Array;
  rms: Float32Array | null; // linear amplitude 0..1 (already converted from dB)
  duration: number;
}
export interface WaveClearMsg {
  type: 'clear';
  kind: WaveKind;
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
export interface WaveDurationMsg {
  type: 'duration';
  duration: number;
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
  | WaveDurationMsg;

export interface WaveReadyMsg {
  type: 'ready';
  webgl2: boolean;
}
export interface WaveErrorMsg {
  type: 'error';
  message: string;
}
export type WaveOutMsg = WaveReadyMsg | WaveErrorMsg;
