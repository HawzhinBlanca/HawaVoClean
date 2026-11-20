/**
 * Pro Interactive 2D STFT Spectrogram Display.
 *
 * Features:
 *  - Genuine 2D STFT rendering via `StftCanvasRenderer` (WebGL2 texture upload
 *    with automatic lossless Canvas 2D `ImageData` fallback).
 *  - Supports server-provided 2D uint8 STFT tiles (`[nBins x widthCols]`),
 *    client-side radix-2 FFT STFT from PCM (`computeClientStft`), and
 *    analysis-backed 2D fallback (`synthesizeTileFromAnalysis`).
 *  - View mode switcher: Original / Cleaned / Removed (Δ).
 *  - Frequency scale switcher: Log / Mel / Linear (20 Hz – 24 kHz) with axis ticks.
 *  - Interactive 2D Time x Frequency box selection with Hz + timecode chip,
 *    Clear control, and `hawa:spectral-selection` CustomEvent dispatch.
 */
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
} from 'react';
import {
  computeClientStft,
  computeDeltaTile,
  DEFAULT_DB_CEIL,
  DEFAULT_DB_FLOOR,
  formatFreqHz,
  freqToNormY,
  getFrequencyTicks,
  MAX_FREQ_HZ,
  MIN_FREQ_HZ,
  normYToFreq,
  SPECTRAL_SELECTION_EVENT,
  StftCanvasRenderer,
  synthesizeTileFromAnalysis,
  type FrequencyScale,
  type SpectralSelection,
  type SpectrogramViewMode,
  type StftTile,
} from '../render/stftSpectrogram';
import { formatTime } from '../render/ticks';
import { waveView } from '../render/viewWindow';
import '../render/spectrogram.css';
import { applySpectralBoxRepair } from '../state/actions';
import { jobInFlight, useStore } from '../state/store';

const SPEC_HEIGHT = 160;
const REDRAW_DEBOUNCE_MS = 40;
const SERVER_FETCH_DEBOUNCE_MS = 120;
const SERVER_TILE_CACHE_MAX = 16;
const MIN_DRAG_PX = 4;

function decodeBase64ToUint8Array(b64: string): Uint8Array | null {
  if (!b64) return null;
  try {
    const binary = atob(b64);
    const out = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      out[i] = binary.charCodeAt(i);
    }
    return out;
  } catch {
    return null;
  }
}

export interface SpectrogramDisplayProps {
  /** Optional server-computed 2D STFT tile for Original (`uint8 [nBins x widthCols]`). */
  originalTile?: StftTile | null;
  /** Optional server-computed 2D STFT tile for Cleaned (`uint8 [nBins x widthCols]`). */
  cleanedTile?: StftTile | null;
  /** Optional raw mono PCM fallback for client-side radix-2 FFT STFT (Original). */
  originalPcm?: Float32Array | null;
  /** Optional raw mono PCM fallback for client-side radix-2 FFT STFT (Cleaned). */
  cleanedPcm?: Float32Array | null;
  /** Optional listener invoked whenever the 2D time x frequency box selection changes. */
  onSpectralSelectionChange?: (selection: SpectralSelection | null) => void;
}

interface DragRect {
  pointerId: number;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  startPxX: number;
  startPxY: number;
}

function clamp01(v: number): number {
  if (!Number.isFinite(v) || v <= 0) return 0;
  return v >= 1 ? 1 : v;
}

function formatTimecodeShort(seconds: number): string {
  const s = Number.isFinite(seconds) && seconds >= 0 ? seconds : 0;
  const whole = Math.floor(s);
  const hundredths = Math.floor((s - whole) * 100);
  return `${formatTime(whole, false)}.${String(hundredths).padStart(2, '0')}`;
}

function modeLabel(mode: SpectrogramViewMode): string {
  if (mode === 'cleaned') return 'Cleaned';
  if (mode === 'delta') return 'Removed (Δ)';
  return 'Original';
}

export function SpectrogramDisplay({
  originalTile = null,
  cleanedTile = null,
  originalPcm = null,
  cleanedPcm = null,
  onSpectralSelectionChange,
}: SpectrogramDisplayProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const rendererRef = useRef<StftCanvasRenderer | null>(null);
  const redrawTimer = useRef<number | null>(null);
  const fetchTimerRef = useRef<number | null>(null);
  const fetchAbortRef = useRef<AbortController | null>(null);
  const lastFetchKeyRef = useRef('');
  const tileCacheRef = useRef<Map<string, StftTile>>(new Map());
  const lastRenderKey = useRef('');
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const abMode = useStore((s) => s.abMode);
  const original = useStore((s) => s.original);
  const cleaned = useStore((s) => s.cleaned);
  const cleanedPath = useStore((s) => s.cleanedPath);
  const engineStatus = useStore((s) => s.engineStatus);
  const client = useStore((s) => s.client);
  const running = useStore((s) => jobInFlight(s.job));
  const analyzing = useStore((s) => s.analyzing);
  const storeSpectralSelection = useStore((s) => s.spectralSelection);
  const setSpectralSelection = useStore((s) => s.setSpectralSelection);
  const stageBlend = useStore((s) => s.stageBlend);

  const [visible, setVisible] = useState(false);

  useEffect(() => {
    if (stageBlend > 0) setVisible(true);
  }, [stageBlend]);
  const [viewMode, setViewMode] = useState<SpectrogramViewMode>(abMode);
  const [scale, setScale] = useState<FrequencyScale>('log');
  const [selection, setSelection] = useState<SpectralSelection | null>(null);
  const [dragRect, setDragRect] = useState<DragRect | null>(null);
  const [repairing, setRepairing] = useState(false);
  const [serverTile, setServerTile] = useState<{ key: string; tile: StftTile } | null>(null);
  const [viewTick, setViewTick] = useState(0);

  const hasData = original !== null;
  const hasCleaned = cleaned !== null || cleanedTile !== null || cleanedPcm !== null;
  const canRepair =
    engineStatus === 'ready' && Boolean(client) && !running && !analyzing && !repairing;

  // Sync spectrogram view mode with global A/B deck changes.
  useEffect(() => {
    if (abMode === 'cleaned' && hasCleaned) {
      setViewMode('cleaned');
    } else if (abMode === 'original') {
      setViewMode('original');
    }
  }, [abMode, hasCleaned]);

  // Fall back to Original if Cleaned disappears.
  useEffect(() => {
    if (!hasCleaned && viewMode !== 'original') {
      setViewMode('original');
    }
  }, [hasCleaned, viewMode]);

  // Clear stale 2D selection and server tile cache only when the source clip actually changes.
  const sourcePath = original?.path ?? null;
  const prevSourcePathRef = useRef<string | null>(sourcePath);
  useEffect(() => {
    if (prevSourcePathRef.current === sourcePath) return;
    prevSourcePathRef.current = sourcePath;
    setSelection(null);
    setDragRect(null);
    setSpectralSelection(null);
    setServerTile(null);
    lastFetchKeyRef.current = '';
    tileCacheRef.current.clear();
  }, [sourcePath, setSpectralSelection]);

  // Keep store spectralSelection synced if an external listener dispatches hawa:spectral-selection.
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const onSpectralEvent = (e: Event): void => {
      const detail = (e as CustomEvent<SpectralSelection | null>).detail ?? null;
      setSpectralSelection(detail);
      setSelection(detail);
    };
    window.addEventListener(SPECTRAL_SELECTION_EVENT, onSpectralEvent);
    return () => window.removeEventListener(SPECTRAL_SELECTION_EVENT, onSpectralEvent);
  }, [setSpectralSelection]);

  // Sync local selection if cleared from store (e.g. AudioAssistant chip dismiss).
  useEffect(() => {
    if (storeSpectralSelection === null && selection !== null) {
      setSelection(null);
    }
  }, [storeSpectralSelection, selection]);

  const effectiveMode: SpectrogramViewMode =
    viewMode === 'original' || !hasCleaned ? 'original' : viewMode;

  // Fetch real 2D STFT / Delta tiles from POST /api/v1/spectrogram when visible & ready
  // (debounced + abortable + cached), seamlessly falling back to client STFT/analysis.
  useEffect(() => {
    const abortPendingFetch = (): void => {
      if (fetchTimerRef.current !== null) {
        window.clearTimeout(fetchTimerRef.current);
        fetchTimerRef.current = null;
      }
      if (fetchAbortRef.current) {
        fetchAbortRef.current.abort();
        fetchAbortRef.current = null;
        if (lastFetchKeyRef.current && !tileCacheRef.current.has(lastFetchKeyRef.current)) {
          lastFetchKeyRef.current = '';
        }
      }
    };

    abortPendingFetch();

    if (!visible || engineStatus !== 'ready' || !client || typeof client.fetchSpectrogramTile !== 'function') {
      return;
    }
    if (!original?.path) return;
    // Explicit prop tiles override server fetching.
    if (effectiveMode === 'original' && originalTile) return;
    if (effectiveMode === 'cleaned' && cleanedTile) return;
    if (effectiveMode === 'delta' && originalTile && cleanedTile) return;

    const cleanFile = cleanedPath ?? cleaned?.path ?? null;
    if ((effectiveMode === 'cleaned' || effectiveMode === 'delta') && !cleanFile) return;

    const v = waveView.view;
    const startS = v.end > v.start ? v.start : 0;
    const endS = v.end > v.start ? v.end : (original.duration_s ?? 0);
    if (!(endS > startS)) return;

    const reqKey = `${original.path}|${effectiveMode === 'original' ? '' : (cleanFile ?? '')}|${effectiveMode}|${startS.toFixed(2)}|${endS.toFixed(2)}`;
    const cached = tileCacheRef.current.get(reqKey);
    if (cached) {
      lastFetchKeyRef.current = reqKey;
      setServerTile({ key: reqKey, tile: cached });
      return;
    }
    if (lastFetchKeyRef.current === reqKey) return;

    fetchTimerRef.current = window.setTimeout(() => {
      fetchTimerRef.current = null;
      if (!mountedRef.current) return;
      const ac = new AbortController();
      fetchAbortRef.current = ac;
      lastFetchKeyRef.current = reqKey;

      const stageW = stageRef.current?.clientWidth ?? 400;
      const widthCols = Math.max(128, Math.min(512, Math.round(stageW > 0 ? stageW : 400)));
      const nBins = 128;

      void client
        .fetchSpectrogramTile(
          {
            path: original.path,
            ...(effectiveMode !== 'original' && cleanFile ? { cleanedPath: cleanFile } : {}),
            mode: effectiveMode,
            startS,
            endS,
            widthCols,
            nBins,
            // Request linear FFT bins so StftCanvasRenderer can switch Log/Mel/Linear client-side without double-warping.
            scale: 'linear',
          },
          ac.signal,
        )
        .then((res) => {
          if (ac.signal.aborted || !mountedRef.current) return;
          if (fetchAbortRef.current === ac) fetchAbortRef.current = null;
          const bytes = decodeBase64ToUint8Array(res.data_b64);
          if (!bytes || res.n_bins < 1 || res.width_cols < 1) return;
          if (bytes.length < res.n_bins * res.width_cols) return;
          const tile: StftTile = {
            data: bytes,
            nBins: res.n_bins,
            widthCols: res.width_cols,
            sampleRate: res.sample_rate || original.sample_rate || 48000,
            startS: res.start_s,
            endS: res.end_s,
            dbFloor: res.db_floor,
            dbCeil: res.db_ceil,
          };
          const cache = tileCacheRef.current;
          if (cache.size >= SERVER_TILE_CACHE_MAX) {
            const oldest = cache.keys().next().value;
            if (oldest !== undefined) cache.delete(oldest);
          }
          cache.set(reqKey, tile);
          setServerTile({ key: reqKey, tile });
        })
        .catch(() => {
          // Soft fallback: client STFT / analysis tile stays rendered cleanly.
          if (fetchAbortRef.current === ac) fetchAbortRef.current = null;
          if (!ac.signal.aborted && lastFetchKeyRef.current === reqKey) {
            lastFetchKeyRef.current = '';
          }
        });
    }, SERVER_FETCH_DEBOUNCE_MS);

    return abortPendingFetch;
  }, [
    visible,
    engineStatus,
    client,
    original,
    cleaned,
    cleanedPath,
    effectiveMode,
    originalTile,
    cleanedTile,
    viewTick,
  ]);

  const sampleRate = original?.sample_rate ?? cleaned?.sample_rate ?? 48000;
  const maxFreqHz = Math.min(MAX_FREQ_HZ, Math.max(MIN_FREQ_HZ + 1, sampleRate / 2));

  // Resolve 2D STFT tiles: server tile > client-side radix-2 STFT PCM > analysis fallback.
  const resolvedOrigTile = useMemo<StftTile | null>(() => {
    if (originalTile) return originalTile;
    if (originalPcm && originalPcm.length > 0) {
      return computeClientStft(originalPcm, sampleRate, 0, {
        dbFloor: DEFAULT_DB_FLOOR,
        dbCeil: DEFAULT_DB_CEIL,
      });
    }
    if (original) {
      return synthesizeTileFromAnalysis(original, {
        dbFloor: DEFAULT_DB_FLOOR,
        dbCeil: DEFAULT_DB_CEIL,
      });
    }
    return null;
  }, [originalTile, originalPcm, original, sampleRate]);

  const resolvedCleanTile = useMemo<StftTile | null>(() => {
    if (cleanedTile) return cleanedTile;
    if (cleanedPcm && cleanedPcm.length > 0) {
      return computeClientStft(cleanedPcm, cleaned?.sample_rate ?? sampleRate, 0, {
        dbFloor: DEFAULT_DB_FLOOR,
        dbCeil: DEFAULT_DB_CEIL,
      });
    }
    if (cleaned) {
      return synthesizeTileFromAnalysis(cleaned, {
        dbFloor: DEFAULT_DB_FLOOR,
        dbCeil: DEFAULT_DB_CEIL,
      });
    }
    return null;
  }, [cleanedTile, cleanedPcm, cleaned, sampleRate]);

  const activeTile = useMemo<StftTile | null>(() => {
    const cleanFile = cleanedPath ?? cleaned?.path ?? '';
    const modePrefix = `${original?.path ?? ''}|${effectiveMode === 'original' ? '' : cleanFile}|${effectiveMode}|`;
    if (
      serverTile &&
      serverTile.key.startsWith(modePrefix) &&
      !(effectiveMode === 'original' && originalTile) &&
      !(effectiveMode === 'cleaned' && cleanedTile) &&
      !(effectiveMode === 'delta' && originalTile && cleanedTile)
    ) {
      return serverTile.tile;
    }
    if (effectiveMode === 'delta' && resolvedOrigTile && resolvedCleanTile) {
      return computeDeltaTile(resolvedOrigTile, resolvedCleanTile);
    }
    if (effectiveMode === 'cleaned' && resolvedCleanTile) {
      return resolvedCleanTile;
    }
    return resolvedOrigTile;
  }, [
    serverTile,
    original,
    cleaned,
    cleanedPath,
    originalTile,
    cleanedTile,
    effectiveMode,
    resolvedOrigTile,
    resolvedCleanTile,
  ]);

  const ticks = useMemo(
    () => getFrequencyTicks(scale, MIN_FREQ_HZ, maxFreqHz),
    [scale, maxFreqHz],
  );

  const getEffectiveView = useCallback((): { start: number; end: number } => {
    const v = waveView.view;
    if (v.end > v.start) return v;
    const dur = activeTile ? activeTile.endS - activeTile.startS : (original?.duration_s ?? 0);
    const start = activeTile?.startS ?? 0;
    return { start, end: start + Math.max(0, dur) };
  }, [activeTile, original]);

  const emitSelection = useCallback(
    (next: SpectralSelection | null) => {
      setSelection(next);
      setSpectralSelection(next);
      onSpectralSelectionChange?.(next);
      if (typeof window !== 'undefined') {
        window.dispatchEvent(
          new CustomEvent<SpectralSelection | null>(SPECTRAL_SELECTION_EVENT, {
            detail: next,
            bubbles: true,
          }),
        );
      }
    },
    [onSpectralSelectionChange, setSpectralSelection],
  );

  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    const renderer = rendererRef.current;
    if (!canvas || !renderer) return;

    const v = getEffectiveView();
    const span = v.end - v.start;
    if (!(span > 0) || !activeTile) {
      renderer.clear();
      return;
    }

    const dpr =
      Number.isFinite(window.devicePixelRatio) && window.devicePixelRatio > 0
        ? window.devicePixelRatio
        : 1;
    const cssW = canvas.clientWidth > 0 ? canvas.clientWidth : 400;
    const cssH = canvas.clientHeight > 0 ? canvas.clientHeight : SPEC_HEIGHT;
    const w = Math.max(1, Math.round(cssW * dpr));
    const h = Math.max(1, Math.round(cssH * dpr));
    if (canvas.width !== w) canvas.width = w;
    if (canvas.height !== h) canvas.height = h;

    const key = `${effectiveMode}:${scale}:${v.start.toFixed(4)}:${v.end.toFixed(4)}:${w}x${h}:${activeTile.widthCols}x${activeTile.nBins}`;
    if (key === lastRenderKey.current) return;
    lastRenderKey.current = key;

    renderer.render(activeTile, v.start, v.end, scale, effectiveMode);
  }, [activeTile, effectiveMode, getEffectiveView, scale]);

  const redrawRef = useRef<() => void>(() => {});
  redrawRef.current = redraw;

  const scheduleRedraw = useCallback(() => {
    if (redrawTimer.current !== null) window.clearTimeout(redrawTimer.current);
    redrawTimer.current = window.setTimeout(() => {
      redrawTimer.current = null;
      if (!mountedRef.current) return;
      redrawRef.current();
    }, REDRAW_DEBOUNCE_MS);
  }, []);

  const scheduleRedrawRef = useRef<() => void>(scheduleRedraw);
  scheduleRedrawRef.current = scheduleRedraw;

  // Mount/unmount StftCanvasRenderer ONLY when the spectrogram canvas mounts/unmounts
  // (visible or hasData toggling) so tile arrivals and scale/mode changes never tear
  // down the WebGL2 context, while loading a new clip cleanly rebinds the fresh canvas.
  useEffect(() => {
    if (!visible || !hasData) {
      setDragRect(null);
      rendererRef.current?.dispose();
      rendererRef.current = null;
      lastRenderKey.current = '';
      return;
    }
    const canvas = canvasRef.current;
    if (!canvas) return;
    const renderer = new StftCanvasRenderer(canvas);
    rendererRef.current = renderer;
    lastRenderKey.current = '';

    const unsub = waveView.subscribe(() => {
      if (!mountedRef.current) return;
      setViewTick((n) => n + 1);
      scheduleRedrawRef.current();
    });

    let ro: ResizeObserver | null = null;
    if (typeof ResizeObserver !== 'undefined' && stageRef.current) {
      ro = new ResizeObserver((entries) => {
        const entry = entries[entries.length - 1];
        if (!entry) return;
        const rect = entry.contentRect;
        if (!(rect.width > 0) || !(rect.height > 0)) return;
        lastRenderKey.current = '';
        scheduleRedrawRef.current();
      });
      ro.observe(stageRef.current);
    }

    redrawRef.current();

    return () => {
      ro?.disconnect();
      unsub();
      if (redrawTimer.current !== null) {
        window.clearTimeout(redrawTimer.current);
        redrawTimer.current = null;
      }
      renderer.dispose();
      rendererRef.current = null;
    };
  }, [visible, hasData]);

  // Redraw when active tile, view mode, or frequency scale changes.
  useEffect(() => {
    if (!visible) return;
    lastRenderKey.current = '';
    redrawRef.current();
  }, [visible, activeTile, effectiveMode, scale]);

  // ── 2D Time x Frequency Box Selection pointer handlers ────────────────────

  const getStageBox = (): { left: number; top: number; width: number; height: number } => {
    const el = stageRef.current;
    if (!el) return { left: 0, top: 0, width: 400, height: SPEC_HEIGHT };
    const rect = el.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
      return { left: rect.left, top: rect.top, width: rect.width, height: rect.height };
    }
    return {
      left: rect.left,
      top: rect.top,
      width: el.clientWidth || 400,
      height: el.clientHeight || SPEC_HEIGHT,
    };
  };

  const releaseCaptureSafe = (el: HTMLDivElement, pointerId: number): void => {
    if (typeof el.releasePointerCapture === 'function') {
      try {
        el.releasePointerCapture(pointerId);
      } catch {
        // Ignore if pointer capture was never held or already released.
      }
    }
  };

  const handlePointerDown = (e: ReactPointerEvent<HTMLDivElement>): void => {
    if (e.button !== 0) return;
    const box = getStageBox();
    if (!(box.width > 0) || !(box.height > 0)) return;
    const x = clamp01((e.clientX - box.left) / box.width);
    const y = clamp01((e.clientY - box.top) / box.height);
    if (typeof e.currentTarget.setPointerCapture === 'function') {
      try {
        e.currentTarget.setPointerCapture(e.pointerId);
      } catch {
        // Synthetic pointer events in tests may not have an active pointer id.
      }
    }
    setDragRect({
      pointerId: e.pointerId,
      x0: x,
      y0: y,
      x1: x,
      y1: y,
      startPxX: e.clientX,
      startPxY: e.clientY,
    });
  };

  const handlePointerMove = (e: ReactPointerEvent<HTMLDivElement>): void => {
    if (!dragRect || dragRect.pointerId !== e.pointerId) return;
    const box = getStageBox();
    if (!(box.width > 0) || !(box.height > 0)) return;
    const x = clamp01((e.clientX - box.left) / box.width);
    const y = clamp01((e.clientY - box.top) / box.height);
    setDragRect((prev) => (prev && prev.pointerId === e.pointerId ? { ...prev, x1: x, y1: y } : prev));
  };

  const handlePointerCancel = (e: ReactPointerEvent<HTMLDivElement>): void => {
    releaseCaptureSafe(e.currentTarget, e.pointerId);
    setDragRect((prev) => (prev && prev.pointerId === e.pointerId ? null : prev));
  };

  const commitDrag = (e: ReactPointerEvent<HTMLDivElement>): void => {
    if (!dragRect || dragRect.pointerId !== e.pointerId) return;
    releaseCaptureSafe(e.currentTarget, e.pointerId);
    const box = getStageBox();
    if (!(box.width > 0) || !(box.height > 0)) {
      setDragRect(null);
      return;
    }
    const x1 = clamp01((e.clientX - box.left) / box.width);
    const y1 = clamp01((e.clientY - box.top) / box.height);
    const dxPx = Math.abs(e.clientX - dragRect.startPxX);
    const dyPx = Math.abs(e.clientY - dragRect.startPxY);
    setDragRect(null);

    if (dxPx < MIN_DRAG_PX && dyPx < MIN_DRAG_PX) return;

    const v = getEffectiveView();
    const span = v.end - v.start;
    if (!(span > 0)) return;

    const xLo = Math.min(1 - 0.002, Math.min(dragRect.x0, x1));
    const xHi = Math.min(1, Math.max(Math.max(dragRect.x0, x1), xLo + 0.002));
    const yTop = Math.min(1 - 0.005, Math.min(dragRect.y0, y1));
    const yBot = Math.min(1, Math.max(Math.max(dragRect.y0, y1), yTop + 0.005));

    const startS = Math.max(v.start, Math.min(v.end, v.start + xLo * span));
    const endS = Math.max(v.start, Math.min(v.end, v.start + xHi * span));
    const fA = normYToFreq(yTop, scale, MIN_FREQ_HZ, maxFreqHz);
    const fB = normYToFreq(yBot, scale, MIN_FREQ_HZ, maxFreqHz);
    const fLowHz = Math.min(fA, fB);
    const fHighHz = Math.max(fA, fB);

    if (!(endS > startS) || !(fHighHz > fLowHz)) return;

    emitSelection({
      startS,
      endS,
      fLowHz,
      fHighHz,
    });
  };

  const handleStageKeyDown = (e: ReactKeyboardEvent<HTMLDivElement>): void => {
    if (e.key === 'Escape' && selection) {
      e.preventDefault();
      emitSelection(null);
    }
  };

  // Compute selection overlay CSS box (live drag preview takes precedence over committed box).
  const overlayStyle = useMemo(() => {
    if (dragRect) {
      const left = Math.min(dragRect.x0, dragRect.x1) * 100;
      const width = Math.abs(dragRect.x1 - dragRect.x0) * 100;
      const top = Math.min(dragRect.y0, dragRect.y1) * 100;
      const height = Math.abs(dragRect.y1 - dragRect.y0) * 100;
      return {
        left: `${left}%`,
        width: `${width}%`,
        top: `${top}%`,
        height: `${height}%`,
      };
    }
    if (!selection) return null;
    const v = getEffectiveView();
    const span = v.end - v.start;
    if (!(span > 0)) return null;
    const x0 = clamp01((selection.startS - v.start) / span);
    const x1 = clamp01((selection.endS - v.start) / span);
    if (x1 <= x0) return null;
    const yTop = freqToNormY(selection.fHighHz, scale, MIN_FREQ_HZ, maxFreqHz);
    const yBot = freqToNormY(selection.fLowHz, scale, MIN_FREQ_HZ, maxFreqHz);
    return {
      left: `${x0 * 100}%`,
      width: `${(x1 - x0) * 100}%`,
      top: `${Math.min(yTop, yBot) * 100}%`,
      height: `${Math.max(1, Math.abs(yBot - yTop) * 100)}%`,
    };
  }, [dragRect, selection, getEffectiveView, scale, maxFreqHz]);

  if (!hasData) return null;

  const dbFloor = activeTile?.dbFloor ?? DEFAULT_DB_FLOOR;
  const dbCeil = activeTile?.dbCeil ?? DEFAULT_DB_CEIL;

  return (
    <section className="panel spectrogram-panel" aria-label="STFT Spectrogram">
      <div className="panel-head">
        <h2 className="panel-title">
          <span>Spectrogram</span>
          <span className="sub">· {modeLabel(effectiveMode)}</span>
        </h2>

        <div className="spectrogram-controls">
          <div className="spectrogram-seg" role="group" aria-label="Spectrogram view mode">
            <button
              type="button"
              className={`spectrogram-seg-btn orig${effectiveMode === 'original' ? ' on' : ''}`}
              aria-pressed={effectiveMode === 'original'}
              onClick={() => setViewMode('original')}
            >
              Original
            </button>
            <button
              type="button"
              className={`spectrogram-seg-btn${effectiveMode === 'cleaned' ? ' on' : ''}`}
              aria-pressed={effectiveMode === 'cleaned'}
              disabled={!hasCleaned}
              onClick={() => setViewMode('cleaned')}
            >
              Cleaned
            </button>
            <button
              type="button"
              className={`spectrogram-seg-btn delta${effectiveMode === 'delta' ? ' on' : ''}`}
              aria-pressed={effectiveMode === 'delta'}
              disabled={!hasCleaned}
              onClick={() => setViewMode('delta')}
              title="Spectral difference: Original minus Cleaned"
            >
              Removed (Δ)
            </button>
          </div>

          <div className="spectrogram-seg" role="group" aria-label="Frequency scale">
            {(['log', 'mel', 'linear'] as const).map((sc) => (
              <button
                key={sc}
                type="button"
                className={`spectrogram-seg-btn${scale === sc ? ' on' : ''}`}
                aria-pressed={scale === sc}
                onClick={() => setScale(sc)}
              >
                {sc === 'log' ? 'Log' : sc === 'mel' ? 'Mel' : 'Linear'}
              </button>
            ))}
          </div>

          <button
            type="button"
            className="wave-fit"
            aria-expanded={visible}
            onClick={() => setVisible((v) => !v)}
            title={visible ? 'Hide STFT spectrogram' : 'Show STFT spectrogram'}
          >
            {visible ? 'HIDE' : 'SHOW'}
          </button>
        </div>
      </div>

      {visible && (
        <div className="spectrogram-body">
          <div className="spectrogram-plot-wrap" style={{ height: `${SPEC_HEIGHT}px` }}>
            <div className="spectrogram-axis" aria-hidden="true">
              {ticks.map((tick) => (
                <span
                  key={tick.hz}
                  className="spectrogram-axis-tick"
                  style={{ top: `${tick.normY * 100}%` }}
                >
                  {tick.label}
                </span>
              ))}
            </div>

            <div
              ref={stageRef}
              className="spectrogram-stage"
              tabIndex={0}
              role="region"
              aria-label={`Interactive 2D ${scale} spectrogram of ${modeLabel(effectiveMode)} signal. Drag to select a time and frequency region, press Escape to clear.`}
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={commitDrag}
              onPointerCancel={handlePointerCancel}
              onKeyDown={handleStageKeyDown}
            >
              <canvas
                ref={canvasRef}
                className="spectrogram-canvas"
                role="img"
                aria-label={`2D STFT spectrogram of ${modeLabel(effectiveMode)} signal (${scale} scale)`}
                style={{ width: '100%', height: `${SPEC_HEIGHT}px` }}
              />
              {ticks.map((tick) => (
                <div
                  key={tick.hz}
                  className="spectrogram-grid-line"
                  aria-hidden="true"
                  style={{ top: `${tick.normY * 100}%` }}
                />
              ))}
              {overlayStyle && (
                <div
                  className="spectrogram-selection-box"
                  aria-hidden="true"
                  style={overlayStyle}
                />
              )}
            </div>
          </div>

          <div className="spectrogram-footer">
            <div className="spectrogram-legend" aria-hidden="true">
              <span className="lo">{dbFloor} dB</span>
              <div className={`spectrogram-gradient${effectiveMode === 'delta' ? ' delta' : ''}`} />
              <span className="hi">{dbCeil} dB</span>
            </div>

            {selection && (
              <div className="spectrogram-selection-chip" role="status" aria-live="polite">
                <span>
                  {formatTimecodeShort(selection.startS)} – {formatTimecodeShort(selection.endS)} ·{' '}
                  {formatFreqHz(selection.fLowHz)} – {formatFreqHz(selection.fHighHz)}
                </span>
                <button
                  type="button"
                  className="spectrogram-repair-btn"
                  disabled={!canRepair}
                  aria-label={`Attenuate band ${formatFreqHz(selection.fLowHz)} to ${formatFreqHz(selection.fHighHz)}`}
                  title={
                    canRepair
                      ? 'Surgically attenuate this time-frequency region on the Cleaned deck'
                      : engineStatus !== 'ready'
                        ? 'Engine offline — reconnect to apply spectral repair'
                        : 'Wait for active processing to finish'
                  }
                  onClick={() => {
                    if (!canRepair) return;
                    setRepairing(true);
                    void applySpectralBoxRepair().finally(() => {
                      if (mountedRef.current) setRepairing(false);
                    });
                  }}
                >
                  {repairing ? 'Repairing…' : 'Attenuate band'}
                </button>
                <button
                  type="button"
                  className="spectrogram-clear-btn"
                  aria-label="Clear spectral selection"
                  onClick={() => emitSelection(null)}
                >
                  Clear
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
