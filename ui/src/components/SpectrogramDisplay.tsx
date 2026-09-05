/**
 * D3 · Scrollable time-frequency spectrogram that tracks the waveform view.
 *
 * Uses the peaks data already in the store (min/max/rms per bucket) to render
 * a spectrogram-like energy display that scrolls with the waveform. This is
 * not a true STFT spectrogram (which would require raw sample access), but
 * an RMS energy heatmap that gives a useful time-energy overview.
 *
 * When toggled visible, it shows a time-axis energy heatmap coloured by RMS
 * level, sync'd to the waveform's view window via waveView subscriptions.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { waveView } from '../render/viewWindow';
import '../render/spectrogram.css';
import { getState, useStore } from '../state/store';

// ── colour map (magma-inspired, 256 entries) ──────────────────────
function buildColourMap(): Uint8ClampedArray {
  const map = new Uint8ClampedArray(256 * 4);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    // Magma-inspired: black → deep purple → hot pink → orange → yellow
    let r: number, g: number, b: number;
    if (t < 0.25) {
      const s = t / 0.25;
      r = Math.round(s * 40);
      g = Math.round(s * 10);
      b = Math.round(30 + s * 80);
    } else if (t < 0.5) {
      const s = (t - 0.25) / 0.25;
      r = Math.round(40 + s * 140);
      g = Math.round(10 + s * 20);
      b = Math.round(110 + s * 20);
    } else if (t < 0.75) {
      const s = (t - 0.5) / 0.25;
      r = Math.round(180 + s * 60);
      g = Math.round(30 + s * 100);
      b = Math.round(130 - s * 80);
    } else {
      const s = (t - 0.75) / 0.25;
      r = Math.round(240 + s * 15);
      g = Math.round(130 + s * 125);
      b = Math.round(50 + s * 140);
    }
    map[i * 4 + 0] = Math.min(255, r);
    map[i * 4 + 1] = Math.min(255, g);
    map[i * 4 + 2] = Math.min(255, b);
    map[i * 4 + 3] = 255;
  }
  return map;
}
const COLOUR_MAP = buildColourMap();

const DB_FLOOR = -60;
const DB_CEIL = 0;
const SPEC_HEIGHT = 96;
const REDRAW_DEBOUNCE_MS = 80;

/**
 * Render an energy heatmap from peak data. Each column is a time bucket,
 * each row is an amplitude band. Brighter = louder.
 */
function renderEnergyMap(
  peakMin: number[],
  peakMax: number[],
  _rmsDb: number[],
  viewStart: number,
  viewEnd: number,
  totalDuration: number,
  width: number,
  height: number,
): ImageData | null {
  if (width < 1 || height < 1 || peakMin.length === 0) return null;

  const n = peakMin.length;
  const img = new ImageData(width, height);

  for (let x = 0; x < width; x++) {
    // Map pixel column to time, then to bucket index
    const t = viewStart + (x / width) * (viewEnd - viewStart);
    const bucketF = (t / totalDuration) * n;
    const bucket = Math.max(0, Math.min(n - 1, Math.floor(bucketF)));

    const lo = peakMin[bucket] ?? 0;
    const hi = peakMax[bucket] ?? 0;
    const amplitude = Math.max(Math.abs(lo), Math.abs(hi));

    // Convert amplitude to dB
    const db = amplitude > 0 ? 20 * Math.log10(amplitude) : DB_FLOOR;
    const normDb = Math.max(0, Math.min(1, (db - DB_FLOOR) / (DB_CEIL - DB_FLOOR)));

    // For each row, paint a gradient based on how much of the column's
    // energy reaches this "frequency" (actually amplitude) band
    for (let y = 0; y < height; y++) {
      const bandPos = (height - 1 - y) / height; // 0 = bottom (low), 1 = top (high)
      // Energy falls off as we go higher — more energy at bottom
      const energy = normDb * Math.exp(-bandPos * 3.0);
      const ci = Math.round(Math.max(0, Math.min(1, energy)) * 255) * 4;
      const pi = (y * width + x) * 4;
      img.data[pi + 0] = COLOUR_MAP[ci + 0] ?? 0;
      img.data[pi + 1] = COLOUR_MAP[ci + 1] ?? 0;
      img.data[pi + 2] = COLOUR_MAP[ci + 2] ?? 0;
      img.data[pi + 3] = 255;
    }
  }

  return img;
}

export function SpectrogramDisplay() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const redrawTimer = useRef<number | null>(null);
  const lastKey = useRef('');
  const abMode = useStore((s) => s.abMode);
  const original = useStore((s) => s.original);
  const cleaned = useStore((s) => s.cleaned);
  const [visible, setVisible] = useState(false);

  const hasData = original !== null;

  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const st = getState();
    const v = waveView.view;
    const span = v.end - v.start;
    if (!(span > 0)) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }

    // Choose which deck's peak data to visualize
    const data = st.abMode === 'cleaned' && st.cleaned ? st.cleaned : st.original;
    if (!data) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }

    const key = `${st.abMode}:${v.start.toFixed(4)}:${v.end.toFixed(4)}`;
    if (key === lastKey.current) return;
    lastKey.current = key;

    const dpr = window.devicePixelRatio || 1;
    const w = Math.round(canvas.clientWidth * dpr);
    const h = Math.round(canvas.clientHeight * dpr);
    if (canvas.width !== w) canvas.width = w;
    if (canvas.height !== h) canvas.height = h;

    const img = renderEnergyMap(
      data.peaks.min,
      data.peaks.max,
      data.rms_db,
      v.start,
      v.end,
      data.duration_s,
      w,
      h,
    );

    if (img) {
      ctx.putImageData(img, 0, 0);
    } else {
      ctx.clearRect(0, 0, w, h);
    }
  }, []);

  const scheduleRedraw = useCallback(() => {
    if (redrawTimer.current !== null) window.clearTimeout(redrawTimer.current);
    redrawTimer.current = window.setTimeout(() => {
      redrawTimer.current = null;
      redraw();
    }, REDRAW_DEBOUNCE_MS);
  }, [redraw]);

  // Subscribe to view changes
  useEffect(() => {
    if (!visible) return;
    const unsub = waveView.subscribe(scheduleRedraw);
    scheduleRedraw();
    return () => {
      unsub();
      if (redrawTimer.current !== null) {
        window.clearTimeout(redrawTimer.current);
        redrawTimer.current = null;
      }
    };
  }, [visible, scheduleRedraw]);

  // Redraw when deck changes
  useEffect(() => {
    if (visible) {
      lastKey.current = '';
      scheduleRedraw();
    }
  }, [abMode, original, cleaned, visible, scheduleRedraw]);

  if (!hasData) return null;

  return (
    <section className="panel spectrogram-panel" aria-label="Energy map">
      <div className="panel-head">
        <h2 className="panel-title">
          <span>Energy Map</span>
          <span className="sub">· {abMode === 'cleaned' && cleaned ? 'Cleaned' : 'Original'}</span>
        </h2>
        <button
          type="button"
          className="wave-fit"
          onClick={() => setVisible((v) => !v)}
          title={visible ? 'Hide energy map' : 'Show energy map'}
        >
          {visible ? 'HIDE' : 'SHOW'}
        </button>
      </div>
      {visible && (
        <div className="spectrogram-body">
          <canvas
            ref={canvasRef}
            className="spectrogram-canvas"
            role="img"
            aria-label={`Time-energy heatmap of the ${abMode} signal`}
            style={{ width: '100%', height: `${SPEC_HEIGHT}px` }}
          />
          <div className="spectrogram-legend" aria-hidden="true">
            <span className="lo">{DB_FLOOR} dB</span>
            <div className="spectrogram-gradient" />
            <span className="hi">{DB_CEIL} dB</span>
          </div>
        </div>
      )}
    </section>
  );
}
