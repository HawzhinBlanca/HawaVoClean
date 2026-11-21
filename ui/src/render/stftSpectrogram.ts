/**
 * Real 2D STFT Spectrogram Engine.
 *
 * Supports:
 *  - Rendering real 2D server tiles (row-major uint8 `[nBins x widthCols]`)
 *  - Genuine client-side radix-2 Cooley-Tukey FFT STFT (`computeClientStft`)
 *  - Analysis-backed 2D tile synthesis fallback (`synthesizeTileFromAnalysis`)
 *  - Removed / Delta (`Original - Cleaned`) spectrogram difference tiles (`computeDeltaTile`)
 *  - Log / Mel / Linear frequency axis (`20 Hz – 24 kHz`) with exact inverse mapping
 *  - WebGL2 texture-accelerated rendering with automatic lossless Canvas 2D `ImageData` fallback
 */

import type { AudioAnalysis } from '../api/types';
import { createProgramU, type ProgramU } from './glutil';

export type SpectrogramViewMode = 'original' | 'cleaned' | 'delta';
export type FrequencyScale = 'log' | 'mel' | 'linear';

/** Row-major 2D STFT spectrogram tile: `data[bin * widthCols + col]`, uint8 (0..255). */
export interface StftTile {
  data: Uint8Array;
  nBins: number;
  widthCols: number;
  sampleRate: number;
  startS: number;
  endS: number;
  dbFloor: number;
  dbCeil: number;
}

/** 2D Time x Frequency selection box dispatched via `hawa:spectral-selection`. */
export interface SpectralSelection {
  startS: number;
  endS: number;
  fLowHz: number;
  fHighHz: number;
}

export interface FrequencyTick {
  hz: number;
  label: string;
  /** Normalised vertical position: 0 = top (maxHz), 1 = bottom (minHz). */
  normY: number;
}

export interface StftOptions {
  fftSize?: number;
  hopSize?: number;
  dbFloor?: number;
  dbCeil?: number;
  maxCols?: number;
}

export const MIN_FREQ_HZ = 20;
export const MAX_FREQ_HZ = 24000;
export const DEFAULT_DB_FLOOR = -60;
export const DEFAULT_DB_CEIL = 0;
export const SPECTRAL_SELECTION_EVENT = 'hawa:spectral-selection';

const AXIS_TICKS_HZ: readonly number[] = [100, 500, 1000, 4000, 8000, 16000];

// ── Colourmaps (256 RGBA entries) ───────────────────────────────────────────

function buildMagmaLut(): Uint8ClampedArray {
  const map = new Uint8ClampedArray(256 * 4);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let r: number;
    let g: number;
    let b: number;
    if (t < 0.25) {
      const s = t / 0.25;
      r = Math.round(7 + s * 38);
      g = Math.round(8 + s * 9);
      b = Math.round(12 + s * 86);
    } else if (t < 0.5) {
      const s = (t - 0.25) / 0.25;
      r = Math.round(45 + s * 135);
      g = Math.round(17 + s * 22);
      b = Math.round(98 + s * 30);
    } else if (t < 0.75) {
      const s = (t - 0.5) / 0.25;
      r = Math.round(180 + s * 62);
      g = Math.round(39 + s * 96);
      b = Math.round(128 - s * 68);
    } else {
      const s = (t - 0.75) / 0.25;
      r = Math.round(242 + s * 13);
      g = Math.round(135 + s * 118);
      b = Math.round(60 + s * 136);
    }
    map[i * 4 + 0] = Math.min(255, Math.max(0, r));
    map[i * 4 + 1] = Math.min(255, Math.max(0, g));
    map[i * 4 + 2] = Math.min(255, Math.max(0, b));
    map[i * 4 + 3] = 255;
  }
  return map;
}

/** High-contrast obsidian -> deep teal -> cyan -> amber-white LUT for Removed (Delta) energy. */
function buildDeltaLut(): Uint8ClampedArray {
  const map = new Uint8ClampedArray(256 * 4);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let r: number;
    let g: number;
    let b: number;
    if (t < 0.3) {
      const s = t / 0.3;
      r = Math.round(7 + s * 8);
      g = Math.round(8 + s * 46);
      b = Math.round(12 + s * 76);
    } else if (t < 0.65) {
      const s = (t - 0.3) / 0.35;
      r = Math.round(15 + s * 42);
      g = Math.round(54 + s * 154);
      b = Math.round(88 + s * 167);
    } else {
      const s = (t - 0.65) / 0.35;
      r = Math.round(57 + s * 198);
      g = Math.round(208 + s * 36);
      b = Math.round(255 - s * 75);
    }
    map[i * 4 + 0] = Math.min(255, Math.max(0, r));
    map[i * 4 + 1] = Math.min(255, Math.max(0, g));
    map[i * 4 + 2] = Math.min(255, Math.max(0, b));
    map[i * 4 + 3] = 255;
  }
  return map;
}

export const MAGMA_LUT = buildMagmaLut();
export const DELTA_LUT = buildDeltaLut();

export function lutForMode(mode: SpectrogramViewMode): Uint8ClampedArray {
  return mode === 'delta' ? DELTA_LUT : MAGMA_LUT;
}

// ── Frequency axis mapping (Log / Mel / Linear) ─────────────────────────────

export function hzToMel(hz: number): number {
  return 2595 * Math.log10(1 + Math.max(0, hz) / 700);
}

export function melToHz(mel: number): number {
  return 700 * (Math.pow(10, Math.max(0, mel) / 2595) - 1);
}

/**
 * Map a frequency in Hz to normalised canvas Y in `[0, 1]`, where:
 *  - `0` = top of plot (`maxHz`)
 *  - `1` = bottom of plot (`minHz`)
 */
export function freqToNormY(
  hz: number,
  scale: FrequencyScale,
  minHz = MIN_FREQ_HZ,
  maxHz = MAX_FREQ_HZ,
): number {
  const lo = Math.max(1, minHz);
  const hi = Math.max(lo + 1, maxHz);
  const f = Math.max(lo, Math.min(hi, hz));
  let up: number;
  if (scale === 'linear') {
    up = (f - lo) / (hi - lo);
  } else if (scale === 'mel') {
    const mLo = hzToMel(lo);
    const mHi = hzToMel(hi);
    up = (hzToMel(f) - mLo) / (mHi - mLo);
  } else {
    up = Math.log2(f / lo) / Math.log2(hi / lo);
  }
  return Math.max(0, Math.min(1, 1 - up));
}

/** Exact inverse of {@link freqToNormY}: canvas `normY` (`0` = top, `1` = bottom) -> Hz. */
export function normYToFreq(
  normY: number,
  scale: FrequencyScale,
  minHz = MIN_FREQ_HZ,
  maxHz = MAX_FREQ_HZ,
): number {
  const lo = Math.max(1, minHz);
  const hi = Math.max(lo + 1, maxHz);
  const up = Math.max(0, Math.min(1, 1 - normY));
  if (scale === 'linear') {
    return lo + up * (hi - lo);
  }
  if (scale === 'mel') {
    const mLo = hzToMel(lo);
    const mHi = hzToMel(hi);
    return melToHz(mLo + up * (mHi - mLo));
  }
  return lo * Math.pow(hi / lo, up);
}

export function formatFreqHz(hz: number): string {
  if (!Number.isFinite(hz) || hz < 0) return '0 Hz';
  if (hz >= 1000) {
    const khz = hz / 1000;
    const str = khz >= 10 ? khz.toFixed(1) : khz.toFixed(2);
    return `${str.replace(/\.?0+$/, '')} kHz`;
  }
  return `${Math.round(hz)} Hz`;
}

export function getFrequencyTicks(
  scale: FrequencyScale,
  minHz = MIN_FREQ_HZ,
  maxHz = MAX_FREQ_HZ,
): FrequencyTick[] {
  const out: FrequencyTick[] = [];
  for (const hz of AXIS_TICKS_HZ) {
    if (hz < minHz || hz > maxHz) continue;
    out.push({
      hz,
      label: formatFreqHz(hz),
      normY: freqToNormY(hz, scale, minHz, maxHz),
    });
  }
  return out;
}

// ── Genuine Radix-2 Cooley-Tukey FFT & Client-Side STFT ─────────────────────

function nearestPow2(n: number): number {
  let p = 64;
  while (p < n && p < 16384) p <<= 1;
  return p;
}

/**
 * In-place iterative radix-2 Cooley-Tukey complex FFT.
 * `re.length` and `im.length` must be equal and a power of 2.
 */
export function fftRadix2(re: Float32Array, im: Float32Array): void {
  const n = re.length;
  if (n <= 1 || (n & (n - 1)) !== 0 || im.length !== n) {
    throw new RangeError(`fftRadix2 requires power-of-2 buffers of equal length, got ${n}/${im.length}`);
  }

  // Bit-reversal permutation
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; (j & bit) !== 0; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      const tr = re[i] ?? 0;
      re[i] = re[j] ?? 0;
      re[j] = tr;
      const ti = im[i] ?? 0;
      im[i] = im[j] ?? 0;
      im[j] = ti;
    }
  }

  // Butterfly stages
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const ang = (-2 * Math.PI) / len;
    const wLenRe = Math.cos(ang);
    const wLenIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let wRe = 1;
      let wIm = 0;
      for (let j = 0; j < half; j++) {
        const uIdx = i + j;
        const vIdx = uIdx + half;
        const uRe = re[uIdx] ?? 0;
        const uIm = im[uIdx] ?? 0;
        const vRe0 = re[vIdx] ?? 0;
        const vIm0 = im[vIdx] ?? 0;
        const vRe = vRe0 * wRe - vIm0 * wIm;
        const vIm = vRe0 * wIm + vIm0 * wRe;
        re[uIdx] = uRe + vRe;
        im[uIdx] = uIm + vIm;
        re[vIdx] = uRe - vRe;
        im[vIdx] = uIm - vIm;
        const nextWRe = wRe * wLenRe - wIm * wLenIm;
        wIm = wRe * wLenIm + wIm * wLenRe;
        wRe = nextWRe;
      }
    }
  }
}

/**
 * Compute a genuine 2D STFT magnitude spectrogram tile from mono PCM samples
 * using a Hann window and radix-2 Cooley-Tukey FFT.
 */
export function computeClientStft(
  pcm: Float32Array,
  sampleRate: number,
  startS = 0,
  opts: StftOptions = {},
): StftTile {
  const sr = sampleRate > 0 ? sampleRate : 48000;
  const fftSize = nearestPow2(opts.fftSize ?? 1024);
  const dbFloor = opts.dbFloor ?? DEFAULT_DB_FLOOR;
  const dbCeil = opts.dbCeil ?? DEFAULT_DB_CEIL;
  const dbSpan = Math.max(1, dbCeil - dbFloor);
  const maxCols = Math.max(1, opts.maxCols ?? 1024);
  const nBins = (fftSize >> 1) + 1;

  if (pcm.length === 0) {
    return {
      data: new Uint8Array(nBins),
      nBins,
      widthCols: 1,
      sampleRate: sr,
      startS,
      endS: startS,
      dbFloor,
      dbCeil,
    };
  }

  const baseHop = Math.max(1, opts.hopSize ?? fftSize >> 2);
  const rawCols = pcm.length <= fftSize ? 1 : Math.floor((pcm.length - fftSize) / baseHop) + 1;
  const widthCols = Math.min(rawCols, maxCols);
  const hop =
    widthCols > 1 && pcm.length > fftSize
      ? (pcm.length - fftSize) / (widthCols - 1)
      : baseHop;

  // Precompute Hann window and coherent gain normalization (2 / sum(w))
  // so a full-scale sine wave lands at 0 dB.
  const win = new Float32Array(fftSize);
  let winSum = 0;
  for (let i = 0; i < fftSize; i++) {
    const w = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (fftSize - 1)));
    win[i] = w;
    winSum += w;
  }
  const scaleMag = winSum > 0 ? 2 / winSum : 1 / fftSize;

  const re = new Float32Array(fftSize);
  const im = new Float32Array(fftSize);
  const data = new Uint8Array(nBins * widthCols);

  for (let col = 0; col < widthCols; col++) {
    const offset = Math.round(col * hop);
    for (let i = 0; i < fftSize; i++) {
      const idx = offset + i;
      re[i] = idx < pcm.length ? (pcm[idx] ?? 0) * (win[i] ?? 0) : 0;
      im[i] = 0;
    }
    fftRadix2(re, im);

    for (let bin = 0; bin < nBins; bin++) {
      const r = re[bin] ?? 0;
      const ii = im[bin] ?? 0;
      // DC and Nyquist bins are one-sided (half the double-sided scaling).
      const binFactor = bin === 0 || bin === nBins - 1 ? 0.5 : 1.0;
      const mag = Math.hypot(r, ii) * scaleMag * binFactor;
      const db = mag > 1e-12 ? 20 * Math.log10(mag) : dbFloor;
      const norm = Math.max(0, Math.min(1, (db - dbFloor) / dbSpan));
      data[bin * widthCols + col] = Math.round(norm * 255);
    }
  }

  return {
    data,
    nBins,
    widthCols,
    sampleRate: sr,
    startS,
    endS: startS + pcm.length / sr,
    dbFloor,
    dbCeil,
  };
}

/**
 * Synthesize a 2D STFT tile from AudioAnalysis (`peaks`, `rms_db`, and LTAS `spectrum`)
 * as a clean offline / test fallback when neither server 2D tiles nor decoded PCM are loaded.
 */
export function synthesizeTileFromAnalysis(
  analysis: Pick<AudioAnalysis, 'peaks' | 'rms_db' | 'spectrum' | 'duration_s' | 'sample_rate'>,
  opts: { nBins?: number; maxCols?: number; dbFloor?: number; dbCeil?: number } = {},
): StftTile {
  const sr = analysis.sample_rate > 0 ? analysis.sample_rate : 48000;
  const nyquist = sr / 2;
  const nBins = Math.max(16, opts.nBins ?? 129);
  const dbFloor = opts.dbFloor ?? DEFAULT_DB_FLOOR;
  const dbCeil = opts.dbCeil ?? DEFAULT_DB_CEIL;
  const dbSpan = Math.max(1, dbCeil - dbFloor);

  const nBuckets = Math.max(analysis.peaks.min.length, analysis.peaks.max.length, analysis.rms_db.length);
  const widthCols = Math.max(1, Math.min(opts.maxCols ?? 512, nBuckets > 0 ? nBuckets : 1));
  const data = new Uint8Array(nBins * widthCols);

  if (nBuckets === 0) {
    return {
      data,
      nBins,
      widthCols,
      sampleRate: sr,
      startS: 0,
      endS: Math.max(0, analysis.duration_s),
      dbFloor,
      dbCeil,
    };
  }

  // Interpolate the measured LTAS spectrum across our linear FFT bin grid.
  const freqs = analysis.spectrum.freqs_hz;
  const specDb = analysis.spectrum.db;
  const binRelDb = new Float32Array(nBins);
  let specMax = -Infinity;
  for (let i = 0; i < specDb.length; i++) {
    const v = specDb[i];
    if (v !== undefined && Number.isFinite(v) && v > specMax) specMax = v;
  }
  const hasSpectrum = freqs.length > 0 && specDb.length > 0 && Number.isFinite(specMax);

  for (let bin = 0; bin < nBins; bin++) {
    const hz = (bin / (nBins - 1)) * nyquist;
    if (!hasSpectrum) {
      // Gentle speech-like rolloff when spectrum array is empty.
      binRelDb[bin] = -18 * Math.pow(hz / Math.max(1, nyquist), 0.7);
      continue;
    }
    let dbVal = specDb[0] ?? dbFloor;
    if (hz <= (freqs[0] ?? 0)) {
      dbVal = specDb[0] ?? dbFloor;
    } else if (hz >= (freqs[freqs.length - 1] ?? nyquist)) {
      dbVal = specDb[specDb.length - 1] ?? dbFloor;
    } else {
      // Binary search for bracketing frequency points
      let lo = 0;
      let hi = freqs.length - 1;
      while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if ((freqs[mid] ?? 0) <= hz) lo = mid;
        else hi = mid;
      }
      const f0 = freqs[lo] ?? 0;
      const f1 = freqs[hi] ?? f0 + 1;
      const d0 = specDb[lo] ?? dbFloor;
      const d1 = specDb[hi] ?? d0;
      const frac = f1 > f0 ? (hz - f0) / (f1 - f0) : 0;
      dbVal = d0 + (d1 - d0) * frac;
    }
    binRelDb[bin] = dbVal - specMax;
  }

  for (let col = 0; col < widthCols; col++) {
    const bucket = Math.min(nBuckets - 1, Math.floor((col / widthCols) * nBuckets));
    const pMin = Math.abs(analysis.peaks.min[bucket] ?? 0);
    const pMax = Math.abs(analysis.peaks.max[bucket] ?? 0);
    const amp = Math.max(pMin, pMax);
    const peakDb = amp > 1e-7 ? 20 * Math.log10(amp) : dbFloor;
    const rmsRaw = analysis.rms_db[bucket];
    const colDb = rmsRaw !== undefined && Number.isFinite(rmsRaw) ? Math.max(peakDb - 6, rmsRaw) : peakDb;
    // Transients (high crest factor) carry slightly more high-frequency energy.
    const crest = Math.max(0, Math.min(18, peakDb - colDb));

    for (let bin = 0; bin < nBins; bin++) {
      const freqFrac = bin / (nBins - 1);
      const db = colDb + (binRelDb[bin] ?? 0) + crest * 0.25 * freqFrac;
      const norm = Math.max(0, Math.min(1, (db - dbFloor) / dbSpan));
      data[bin * widthCols + col] = Math.round(norm * 255);
    }
  }

  return {
    data,
    nBins,
    widthCols,
    sampleRate: sr,
    startS: 0,
    endS: Math.max(0, analysis.duration_s),
    dbFloor,
    dbCeil,
  };
}

/**
 * Compute a Removed (Delta) spectrogram tile: `max(0, Original - Cleaned)` in magnitude dB space.
 * Supports tiles with different column or bin dimensions via nearest/linear sampling.
 */
export function computeDeltaTile(original: StftTile, cleaned: StftTile): StftTile {
  const nBins = original.nBins;
  const widthCols = original.widthCols;
  const out = new Uint8Array(nBins * widthCols);

  const sameGrid = cleaned.nBins === nBins && cleaned.widthCols === widthCols;
  for (let bin = 0; bin < nBins; bin++) {
    const cBin = sameGrid
      ? bin
      : Math.min(cleaned.nBins - 1, Math.round((bin / Math.max(1, nBins - 1)) * (cleaned.nBins - 1)));
    const oRow = bin * widthCols;
    const cRow = cBin * cleaned.widthCols;
    for (let col = 0; col < widthCols; col++) {
      const cCol = sameGrid
        ? col
        : Math.min(
            cleaned.widthCols - 1,
            Math.round((col / Math.max(1, widthCols - 1)) * (cleaned.widthCols - 1)),
          );
      const oVal = original.data[oRow + col] ?? 0;
      const cVal = cleaned.data[cRow + cCol] ?? 0;
      const diff = oVal - cVal;
      // Gentle 1.5x visual contrast boost so subtle removed hum/hiss stays legible
      out[oRow + col] = diff > 0 ? Math.min(255, Math.round(diff * 1.5)) : 0;
    }
  }

  return {
    data: out,
    nBins,
    widthCols,
    sampleRate: original.sampleRate,
    startS: original.startS,
    endS: original.endS,
    dbFloor: original.dbFloor,
    dbCeil: original.dbCeil,
  };
}

// ── Lossless Canvas 2D ImageData Renderer ───────────────────────────────────

/**
 * Render a 2D `StftTile` into an `ImageData` using bilinear interpolation over
 * the active `[viewStart, viewEnd]` time window and `Log` / `Mel` / `Linear` frequency axis.
 */
export function renderTileToImageData(
  tile: StftTile,
  viewStart: number,
  viewEnd: number,
  width: number,
  height: number,
  scale: FrequencyScale,
  mode: SpectrogramViewMode = 'original',
): ImageData | null {
  if (width < 1 || height < 1 || tile.nBins < 1 || tile.widthCols < 1) return null;
  const viewSpan = viewEnd - viewStart;
  if (!(viewSpan > 0)) return null;

  const lut = lutForMode(mode);
  const nyquist = Math.max(MIN_FREQ_HZ + 1, tile.sampleRate / 2);
  const maxHz = Math.min(MAX_FREQ_HZ, nyquist);
  const tileSpan = Math.max(1e-6, tile.endS - tile.startS);

  // Precompute fractional FFT bin for each canvas row `y`.
  const rowBin = new Float32Array(height);
  for (let y = 0; y < height; y++) {
    const normY = height > 1 ? y / (height - 1) : 0;
    const hz = normYToFreq(normY, scale, MIN_FREQ_HZ, maxHz);
    rowBin[y] = Math.max(0, Math.min(tile.nBins - 1, (hz / nyquist) * (tile.nBins - 1)));
  }

  // Precompute fractional tile column for each canvas column `x`.
  const colPos = new Float32Array(width);
  for (let x = 0; x < width; x++) {
    const t = viewStart + ((x + 0.5) / width) * viewSpan;
    const u = (t - tile.startS) / tileSpan;
    colPos[x] = u * (tile.widthCols - 1);
  }

  const pixels = new Uint8ClampedArray(width * height * 4);
  const wCols = tile.widthCols;
  const maxBin = tile.nBins - 1;
  const maxCol = wCols - 1;

  for (let y = 0; y < height; y++) {
    const bF = rowBin[y] ?? 0;
    const b0 = Math.floor(bF);
    const b1 = Math.min(maxBin, b0 + 1);
    const fb = bF - b0;
    const row0 = b0 * wCols;
    const row1 = b1 * wCols;
    const yOff = y * width;

    for (let x = 0; x < width; x++) {
      const cF = colPos[x] ?? -1;
      let val = 0;
      if (cF >= 0 && cF <= maxCol) {
        const c0 = Math.floor(cF);
        const c1 = Math.min(maxCol, c0 + 1);
        const fc = cF - c0;
        const v00 = tile.data[row0 + c0] ?? 0;
        const v10 = tile.data[row0 + c1] ?? 0;
        const v01 = tile.data[row1 + c0] ?? 0;
        const v11 = tile.data[row1 + c1] ?? 0;
        const top = v00 + (v10 - v00) * fc;
        const bot = v01 + (v11 - v01) * fc;
        val = Math.round(top + (bot - top) * fb);
      }
      const ci = Math.max(0, Math.min(255, val)) * 4;
      const pi = (yOff + x) * 4;
      pixels[pi + 0] = lut[ci + 0] ?? 0;
      pixels[pi + 1] = lut[ci + 1] ?? 0;
      pixels[pi + 2] = lut[ci + 2] ?? 0;
      pixels[pi + 3] = 255;
    }
  }

  if (typeof ImageData !== 'undefined') {
    return new ImageData(pixels, width, height);
  }
  return { data: pixels, width, height, colorSpace: 'srgb' } as ImageData;
}

// ── WebGL2 Hardware-Accelerated Renderer with Automatic 2D Fallback ─────────

const VS_SRC = `#version 300 es
precision highp float;
out vec2 v_uv;
void main() {
  vec2 pos = vec2((gl_VertexID & 1) << 2, (gl_VertexID & 2) << 1) - 1.0;
  v_uv = pos * 0.5 + 0.5;
  gl_Position = vec4(pos, 0.0, 1.0);
}`;

const FS_SRC = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_stft;
uniform sampler2D u_lut;
uniform float u_viewStart;
uniform float u_viewEnd;
uniform float u_tileStart;
uniform float u_tileEnd;
uniform float u_minHz;
uniform float u_maxHz;
uniform float u_nyquist;
uniform int u_scale;

void main() {
  float t = mix(u_viewStart, u_viewEnd, v_uv.x);
  float colU = (t - u_tileStart) / max(1e-6, u_tileEnd - u_tileStart);
  if (colU < 0.0 || colU > 1.0) {
    fragColor = texture(u_lut, vec2(0.0, 0.5));
    return;
  }
  float up = clamp(v_uv.y, 0.0, 1.0);
  float hz;
  if (u_scale == 2) {
    hz = mix(u_minHz, u_maxHz, up);
  } else if (u_scale == 1) {
    float mLo = 2595.0 * log(1.0 + u_minHz / 700.0) / log(10.0);
    float mHi = 2595.0 * log(1.0 + u_maxHz / 700.0) / log(10.0);
    float mel = mix(mLo, mHi, up);
    hz = 700.0 * (pow(10.0, mel / 2595.0) - 1.0);
  } else {
    hz = u_minHz * pow(u_maxHz / u_minHz, up);
  }
  float binV = clamp(hz / max(1.0, u_nyquist), 0.0, 1.0);
  float mag = texture(u_stft, vec2(colU, binV)).r;
  fragColor = texture(u_lut, vec2(mag, 0.5));
}`;

const GL_UNIFORMS = [
  'u_stft',
  'u_lut',
  'u_viewStart',
  'u_viewEnd',
  'u_tileStart',
  'u_tileEnd',
  'u_minHz',
  'u_maxHz',
  'u_nyquist',
  'u_scale',
] as const;

/**
 * Dual-backend spectrogram surface: prefers WebGL2 texture upload for zero-copy
 * GPU frequency-axis resampling, and falls back automatically to lossless Canvas 2D
 * `ImageData` rendering when WebGL2 is unavailable (e.g. happy-dom or software mode).
 */
export class StftCanvasRenderer {
  readonly backend: 'webgl2' | '2d' | 'none';
  private readonly canvas: HTMLCanvasElement;
  private gl: WebGL2RenderingContext | null = null;
  private ctx2d: CanvasRenderingContext2D | null = null;
  private prog: ProgramU | null = null;
  private vao: WebGLVertexArrayObject | null = null;
  private stftTex: WebGLTexture | null = null;
  private lutTex: WebGLTexture | null = null;
  private lastTile: StftTile | null = null;
  private lastLutMode: SpectrogramViewMode | null = null;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    let gl: WebGL2RenderingContext | null = null;
    try {
      gl = canvas.getContext('webgl2', {
        alpha: false,
        antialias: false,
        depth: false,
        stencil: false,
        preserveDrawingBuffer: false,
      });
    } catch {
      gl = null;
    }

    if (gl) {
      try {
        this.prog = createProgramU(gl, VS_SRC, FS_SRC, GL_UNIFORMS);
        this.vao = gl.createVertexArray();
        this.stftTex = this.makeTexture(gl);
        this.lutTex = this.makeTexture(gl);
        this.gl = gl;
        this.backend = 'webgl2';
        return;
      } catch {
        this.cleanupGl(gl);
        gl = null;
      }
    }

    try {
      this.ctx2d = canvas.getContext('2d');
    } catch {
      this.ctx2d = null;
    }
    this.backend = this.ctx2d ? '2d' : 'none';
  }

  private makeTexture(gl: WebGL2RenderingContext): WebGLTexture {
    const tex = gl.createTexture();
    if (!tex) throw new Error('createTexture failed');
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return tex;
  }

  private cleanupGl(gl: WebGL2RenderingContext, releaseContext = false): void {
    const lost = typeof gl.isContextLost === 'function' && gl.isContextLost();
    if (!lost) {
      try {
        gl.activeTexture?.(gl.TEXTURE1);
        gl.bindTexture?.(gl.TEXTURE_2D, null);
        gl.activeTexture?.(gl.TEXTURE0);
        gl.bindTexture?.(gl.TEXTURE_2D, null);
        gl.bindVertexArray?.(null);
        gl.useProgram?.(null);
        if (this.stftTex) gl.deleteTexture(this.stftTex);
        if (this.lutTex) gl.deleteTexture(this.lutTex);
        if (this.vao) gl.deleteVertexArray(this.vao);
        if (this.prog) gl.deleteProgram(this.prog.p);
        if (releaseContext) {
          const ext = gl.getExtension?.('WEBGL_lose_context') as
            | { loseContext?: () => void }
            | null
            | undefined;
          ext?.loseContext?.();
        }
      } catch {
        // Ignore teardown errors if the GPU context was lost mid-cleanup.
      }
    }
    this.stftTex = null;
    this.lutTex = null;
    this.vao = null;
    this.prog = null;
  }

  clear(): void {
    if (this.gl) {
      if (typeof this.gl.isContextLost === 'function' && this.gl.isContextLost()) return;
      this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
      this.gl.clearColor(7 / 255, 8 / 255, 10 / 255, 1);
      this.gl.clear(this.gl.COLOR_BUFFER_BIT);
    } else if (this.ctx2d) {
      this.ctx2d.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
  }

  render(
    tile: StftTile,
    viewStart: number,
    viewEnd: number,
    scale: FrequencyScale,
    mode: SpectrogramViewMode,
  ): void {
    const w = this.canvas.width;
    const h = this.canvas.height;
    if (w < 1 || h < 1) return;
    if (!Number.isFinite(viewStart) || !Number.isFinite(viewEnd) || !(viewEnd > viewStart)) {
      this.clear();
      return;
    }

    if (this.gl && this.prog && this.stftTex && this.lutTex) {
      const gl = this.gl;
      if (typeof gl.isContextLost === 'function' && gl.isContextLost()) return;
      if (this.lastTile !== tile) {
        gl.bindTexture(gl.TEXTURE_2D, this.stftTex);
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
        gl.texImage2D(
          gl.TEXTURE_2D,
          0,
          gl.R8,
          tile.widthCols,
          tile.nBins,
          0,
          gl.RED,
          gl.UNSIGNED_BYTE,
          tile.data,
        );
        this.lastTile = tile;
      }
      if (this.lastLutMode !== mode) {
        const lut = lutForMode(mode);
        gl.bindTexture(gl.TEXTURE_2D, this.lutTex);
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, lut);
        this.lastLutMode = mode;
      }

      gl.viewport(0, 0, w, h);
      gl.useProgram(this.prog.p);
      gl.bindVertexArray(this.vao);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.stftTex);
      gl.uniform1i(this.prog.u.u_stft ?? null, 0);

      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this.lutTex);
      gl.uniform1i(this.prog.u.u_lut ?? null, 1);

      const nyquist = Math.max(MIN_FREQ_HZ + 1, tile.sampleRate / 2);
      const maxHz = Math.min(MAX_FREQ_HZ, nyquist);
      gl.uniform1f(this.prog.u.u_viewStart ?? null, viewStart);
      gl.uniform1f(this.prog.u.u_viewEnd ?? null, viewEnd);
      gl.uniform1f(this.prog.u.u_tileStart ?? null, tile.startS);
      gl.uniform1f(this.prog.u.u_tileEnd ?? null, tile.endS);
      gl.uniform1f(this.prog.u.u_minHz ?? null, MIN_FREQ_HZ);
      gl.uniform1f(this.prog.u.u_maxHz ?? null, maxHz);
      gl.uniform1f(this.prog.u.u_nyquist ?? null, nyquist);
      gl.uniform1i(this.prog.u.u_scale ?? null, scale === 'linear' ? 2 : scale === 'mel' ? 1 : 0);

      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.bindVertexArray(null);
      return;
    }

    if (this.ctx2d) {
      const img = renderTileToImageData(tile, viewStart, viewEnd, w, h, scale, mode);
      if (img) {
        this.ctx2d.putImageData(img, 0, 0);
      } else {
        this.ctx2d.clearRect(0, 0, w, h);
      }
    }
  }

  dispose(): void {
    if (this.gl) {
      this.cleanupGl(this.gl, true);
      this.gl = null;
    }
    this.ctx2d = null;
    this.lastTile = null;
    this.lastLutMode = null;
  }
}
