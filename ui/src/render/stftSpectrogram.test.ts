// render/stftSpectrogram.test.ts — Unit tests for the real 2D STFT spectrogram engine.

import { describe, expect, it, vi } from 'vitest';
import type { AudioAnalysis } from '../api/types';
import {
  computeClientStft,
  computeDeltaTile,
  fftRadix2,
  formatFreqHz,
  freqToNormY,
  getFrequencyTicks,
  normYToFreq,
  renderTileToImageData,
  StftCanvasRenderer,
  synthesizeTileFromAnalysis,
  type FrequencyScale,
} from './stftSpectrogram';

function makeSine(freqHz: number, sampleRate: number, durationS: number, amp = 0.8): Float32Array {
  const n = Math.round(sampleRate * durationS);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = amp * Math.sin((2 * Math.PI * freqHz * i) / sampleRate);
  }
  return out;
}

describe('fftRadix2', () => {
  it('computes exact spectrum of a unit impulse and a DC signal', () => {
    const re = new Float32Array(16);
    const im = new Float32Array(16);
    re[0] = 1;
    fftRadix2(re, im);
    for (let k = 0; k < 16; k++) {
      expect(re[k]).toBeCloseTo(1, 5);
      expect(im[k]).toBeCloseTo(0, 5);
    }

    const dcRe = new Float32Array(16).fill(0.5);
    const dcIm = new Float32Array(16);
    fftRadix2(dcRe, dcIm);
    expect(dcRe[0]).toBeCloseTo(8, 5);
    for (let k = 1; k < 16; k++) {
      expect(Math.hypot(dcRe[k] ?? 0, dcIm[k] ?? 0)).toBeCloseTo(0, 5);
    }
  });

  it('rejects non-power-of-two buffer sizes', () => {
    expect(() => fftRadix2(new Float32Array(10), new Float32Array(10))).toThrow(RangeError);
  });
});

describe('computeClientStft & computeDeltaTile', () => {
  it('concentrates spectral energy into the exact frequency bin of a pure sine wave', () => {
    const sr = 48000;
    const fftSize = 1024;
    const freqHz = 3000;
    const pcm = makeSine(freqHz, sr, 0.15, 0.9);
    const tile = computeClientStft(pcm, sr, 0, { fftSize, hopSize: 256, dbFloor: -60, dbCeil: 0 });

    expect(tile.nBins).toBe(fftSize / 2 + 1);
    expect(tile.widthCols).toBeGreaterThan(5);
    expect(tile.data.length).toBe(tile.nBins * tile.widthCols);

    const expectedBin = Math.round((freqHz / (sr / 2)) * (tile.nBins - 1));
    const midCol = Math.floor(tile.widthCols / 2);

    // Find peak bin in the middle time column
    let peakBin = 0;
    let peakVal = -1;
    for (let b = 0; b < tile.nBins; b++) {
      const v = tile.data[b * tile.widthCols + midCol] ?? 0;
      if (v > peakVal) {
        peakVal = v;
        peakBin = b;
      }
    }
    expect(Math.abs(peakBin - expectedBin)).toBeLessThanOrEqual(1);
    expect(peakVal).toBeGreaterThan(230); // ~0 dB peak

    // Distant bin (12 kHz) should be near floor
    const farBin = Math.round((12000 / (sr / 2)) * (tile.nBins - 1));
    expect(tile.data[farBin * tile.widthCols + midCol]).toBeLessThan(15);
  });

  it('computeDeltaTile isolates removed frequency components (Original - Cleaned)', () => {
    const sr = 48000;
    const fftSize = 1024;
    const speech = makeSine(1000, sr, 0.12, 0.5);
    const hum = makeSine(4000, sr, 0.12, 0.5);
    const originalPcm = new Float32Array(speech.length);
    for (let i = 0; i < speech.length; i++) {
      originalPcm[i] = (speech[i] ?? 0) + (hum[i] ?? 0);
    }

    const origTile = computeClientStft(originalPcm, sr, 0, { fftSize, hopSize: 256 });
    const cleanTile = computeClientStft(speech, sr, 0, { fftSize, hopSize: 256 });
    const deltaTile = computeDeltaTile(origTile, cleanTile);

    const midCol = Math.floor(deltaTile.widthCols / 2);
    const bin1k = Math.round((1000 / (sr / 2)) * (deltaTile.nBins - 1));
    const bin4k = Math.round((4000 / (sr / 2)) * (deltaTile.nBins - 1));

    // Preserved 1 kHz component cancels out (~0 in delta), removed 4 kHz hum is bright!
    expect(deltaTile.data[bin1k * deltaTile.widthCols + midCol]).toBeLessThan(20);
    expect(deltaTile.data[bin4k * deltaTile.widthCols + midCol]).toBeGreaterThan(180);
  });
});

describe('frequency axis mapping (Log / Mel / Linear)', () => {
  const scales: FrequencyScale[] = ['log', 'mel', 'linear'];

  it.each(scales)('round-trips freqToNormY <-> normYToFreq accurately for %s scale', (scale) => {
    for (const hz of [20, 100, 500, 1000, 4000, 8000, 16000, 24000]) {
      const y = freqToNormY(hz, scale, 20, 24000);
      expect(y).toBeGreaterThanOrEqual(0);
      expect(y).toBeLessThanOrEqual(1);
      const back = normYToFreq(y, scale, 20, 24000);
      expect(back).toBeCloseTo(hz, 1);
    }
  });

  it('produces all 6 standard axis ticks with monotonically decreasing normY (bottom to top)', () => {
    const ticks = getFrequencyTicks('log', 20, 24000);
    expect(ticks.map((t) => t.label)).toEqual([
      '100 Hz',
      '500 Hz',
      '1 kHz',
      '4 kHz',
      '8 kHz',
      '16 kHz',
    ]);
    for (let i = 1; i < ticks.length; i++) {
      expect(ticks[i]!.normY).toBeLessThan(ticks[i - 1]!.normY);
    }
  });

  it('formats Hz and kHz cleanly', () => {
    expect(formatFreqHz(85.4)).toBe('85 Hz');
    expect(formatFreqHz(1000)).toBe('1 kHz');
    expect(formatFreqHz(3750)).toBe('3.75 kHz');
    expect(formatFreqHz(16000)).toBe('16 kHz');
  });
});

describe('synthesizeTileFromAnalysis & renderTileToImageData', () => {
  const mockAnalysis: AudioAnalysis = {
    path: '/test.wav',
    sample_rate: 48000,
    channels: 1,
    duration_s: 2.0,
    loudness: { integrated_lufs: -18, true_peak_dbtp: -1 },
    noise_floor_db: -50,
    rms_db: [-22, -16, -14, -20],
    peaks: { min: [-0.4, -0.7, -0.8, -0.3], max: [0.4, 0.7, 0.8, 0.3] },
    spectrum: { freqs_hz: [100, 1000, 4000, 12000], db: [-28, -14, -26, -48] },
  };

  it('synthesizes a non-blank 2D STFT tile from AudioAnalysis and renders RGBA ImageData', () => {
    const tile = synthesizeTileFromAnalysis(mockAnalysis);
    expect(tile.nBins).toBeGreaterThanOrEqual(64);
    expect(tile.widthCols).toBe(4);
    const maxByte = Math.max(...tile.data);
    expect(maxByte).toBeGreaterThan(100);

    const img = renderTileToImageData(tile, 0, 2.0, 64, 32, 'log', 'original');
    expect(img).not.toBeNull();
    expect(img?.width).toBe(64);
    expect(img?.height).toBe(32);
    expect(img?.data.length).toBe(64 * 32 * 4);

    // Verify pixels are opaque and non-uniform
    let minLum = 255;
    let maxLum = 0;
    for (let i = 0; i < (img?.data.length ?? 0); i += 4) {
      const lum = ((img?.data[i] ?? 0) + (img?.data[i + 1] ?? 0) + (img?.data[i + 2] ?? 0)) / 3;
      if (lum < minLum) minLum = lum;
      if (lum > maxLum) maxLum = lum;
      expect(img?.data[i + 3]).toBe(255);
    }
    expect(maxLum - minLum).toBeGreaterThan(40);
  });

  it('StftCanvasRenderer falls back cleanly to 2D Canvas when WebGL2 is unavailable', () => {
    const canvas = document.createElement('canvas');
    canvas.width = 48;
    canvas.height = 24;
    const putImageData = vi.fn();
    const clearRect = vi.fn();
    vi.spyOn(canvas, 'getContext').mockImplementation(((kind: string) => {
      if (kind === 'webgl2') return null;
      if (kind === '2d') {
        return { putImageData, clearRect } as unknown as CanvasRenderingContext2D;
      }
      return null;
    }) as typeof canvas.getContext);

    const renderer = new StftCanvasRenderer(canvas);
    expect(renderer.backend).toBe('2d');

    const tile = synthesizeTileFromAnalysis(mockAnalysis);
    renderer.render(tile, 0, 2.0, 'linear', 'delta');
    expect(putImageData).toHaveBeenCalledTimes(1);

    renderer.clear();
    expect(clearRect).toHaveBeenCalledTimes(1);
    renderer.dispose();
  });
});
