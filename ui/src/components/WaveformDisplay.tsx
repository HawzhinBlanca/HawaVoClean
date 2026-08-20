import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent,
} from 'react';
import { getPlayer } from '../audio/player';
import { WaveformHost } from '../render/WaveformHost';
import { loadPeaks, peaksSupported } from '../render/peaksCache';
import { chooseStep, formatSeconds, formatTime, tickLabel, timeTicksIn } from '../render/ticks';
import { waveView } from '../render/viewWindow';
import type { WaveKind } from '../render/waveformProtocol';
import '../render/waveZoom.css';
import { seekTo } from '../state/actions';
import { channelName, reportChannels } from '../state/selection';
import { getState, useStore } from '../state/store';
import { VerdictStrip } from './VerdictStrip';

function toF32(a: number[]): Float32Array {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] ?? 0;
  return out;
}

function rmsToLinear(db: number[]): Float32Array {
  const out = new Float32Array(db.length);
  for (let i = 0; i < db.length; i++) {
    const d = db[i] ?? -120;
    out[i] = d <= -120 ? 0 : Math.min(1, 10 ** (d / 20));
  }
  return out;
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/** Pointer capture is best-effort: a pointer can be gone by the time we ask. */
function capture(el: Element, pointerId: number): void {
  try {
    el.setPointerCapture(pointerId);
  } catch {
    /* pointer already released */
  }
}

function release(el: Element, pointerId: number): void {
  try {
    el.releasePointerCapture(pointerId);
  } catch {
    /* already released */
  }
}

/** Buckets to ask `/api/peaks` for: one per device column, within the API cap. */
function bucketsFor(cssWidth: number): number {
  const dpr = window.devicePixelRatio || 1;
  return clamp(Math.round(cssWidth * dpr), 256, 8000);
}

const RULER_FONT = '9.5px "SF Mono", ui-monospace, Menlo, Monaco, "Roboto Mono", monospace';
const DETAIL_DEBOUNCE_MS = 120;
/** Only re-query when the window would gain real detail over the base envelope. */
const DETAIL_GAIN = 1.2;

interface DetailKey {
  path: string;
  start: number;
  end: number;
  buckets: number;
}

export function WaveformDisplay() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const displayRef = useRef<HTMLDivElement | null>(null);
  const panelRef = useRef<HTMLElement | null>(null);
  const hostRef = useRef<WaveformHost | null>(null);
  const timeRef = useRef<HTMLSpanElement | null>(null);
  const rulerRef = useRef<HTMLCanvasElement | null>(null);
  const rangeRef = useRef<HTMLSpanElement | null>(null);
  const magRef = useRef<HTMLSpanElement | null>(null);
  const selTagRef = useRef<HTMLSpanElement | null>(null);
  const selTagW = useRef(0);
  const widthRef = useRef(0);
  const ovCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const ovWinRef = useRef<HTMLDivElement | null>(null);
  const ovRef = useRef<HTMLDivElement | null>(null);

  const dragging = useRef(false);
  const panBase = useRef<{ x: number; start: number; end: number } | null>(null);
  const ovGrab = useRef<{ offset: number } | null>(null);
  const bucketsRef = useRef(1200);
  const detailTimer = useRef<number | null>(null);
  const detailAbort = useRef<AbortController | null>(null);
  const lastDetail = useRef<Record<WaveKind, DetailKey | null>>({ original: null, cleaned: null });
  const requestCount = useRef(0);

  const [width, setWidth] = useState(0);
  const [glOk, setGlOk] = useState(true);

  const original = useStore((s) => s.original);
  const cleaned = useStore((s) => s.cleaned);
  const report = useStore((s) => s.report);
  const abMode = useStore((s) => s.abMode);
  const highlight = useStore((s) => s.highlightRange);
  const analyzing = useStore((s) => s.analyzing);
  const source = useStore((s) => s.source);
  const setError = useStore((s) => s.setError);

  // A7/B4 · which channels this report decided on. A split-speakers run emits
  // a set of units per channel and those sets overlap in time, so the lit band
  // has to say *whose* seconds it is lighting; `channels.indexOf` turns the
  // channel number into the lane the renderer draws in.
  const channels = useMemo(() => reportChannels(report?.units ?? []), [report]);
  const selChannel = highlight?.channel;
  const selName =
    channels.length > 1 && selChannel !== undefined ? channelName(selChannel, channels) : null;

  // ---------------------------------------------------------------- painting
  // Everything below runs off the view controller, never off React state, so a
  // wheel burst costs one worker message plus a handful of DOM writes.

  const drawRuler = useCallback(() => {
    const c = rulerRef.current;
    if (!c) return;
    const rect = c.getBoundingClientRect();
    if (rect.width < 1 || rect.height < 1) return;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(rect.width * dpr));
    const h = Math.max(1, Math.round(rect.height * dpr));
    if (c.width !== w) c.width = w;
    if (c.height !== h) c.height = h;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    const v = waveView.view;
    const span = v.end - v.start;
    if (!(span > 0)) return;
    const step = chooseStep(span, rect.width, 72);
    const ticks = timeTicksIn(v.start, v.end, rect.width, 72);
    ctx.font = RULER_FONT;
    ctx.textBaseline = 'top';
    for (const t of ticks) {
      const x = ((t.time - v.start) / span) * rect.width;
      const px = Math.round(x);
      ctx.fillStyle = t.major ? 'rgba(255,255,255,0.42)' : 'rgba(255,255,255,0.2)';
      const th = t.major ? 7 : 4;
      ctx.fillRect(px, rect.height - th, 1, th);
      if (!t.major) continue;
      const label = tickLabel(t.time, step, v.end);
      const tw = ctx.measureText(label).width;
      if (px + 3 + tw > rect.width - 2) continue;
      // #6f7886 measured 4.5:1 on the display floor — legal, and still ghostly
      // at 9.5px. The ruler is the panel's navigation reference; it has to be
      // the second thing you read after the waveform itself.
      ctx.fillStyle = '#8e97a5';
      ctx.fillText(label, px + 3, 2);
    }
  }, []);

  const drawOverview = useCallback(() => {
    const c = ovCanvasRef.current;
    if (!c) return;
    const rect = c.getBoundingClientRect();
    if (rect.width < 1 || rect.height < 1) return;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(rect.width * dpr));
    const h = Math.max(1, Math.round(rect.height * dpr));
    if (c.width !== w) c.width = w;
    if (c.height !== h) c.height = h;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    const a = getState().original;
    if (!a) return;
    const lo = a.peaks.min;
    const hi = a.peaks.max;
    const n = Math.min(lo.length, hi.length);
    if (n === 0) return;
    const mid = rect.height / 2;
    const half = (rect.height / 2) * 0.86;
    const grad = ctx.createLinearGradient(0, 0, 0, rect.height);
    grad.addColorStop(0, 'rgba(255,179,71,0.55)');
    grad.addColorStop(0.5, 'rgba(255,201,120,0.85)');
    grad.addColorStop(1, 'rgba(255,179,71,0.55)');
    ctx.fillStyle = grad;
    const cols = Math.max(1, Math.round(rect.width));
    for (let x = 0; x < cols; x++) {
      const i0 = Math.floor((x / cols) * n);
      const i1 = Math.max(i0 + 1, Math.ceil(((x + 1) / cols) * n));
      let vmin = 0;
      let vmax = 0;
      for (let i = i0; i < i1 && i < n; i++) {
        const va = lo[i] ?? 0;
        const vb = hi[i] ?? 0;
        if (va < vmin) vmin = va;
        if (vb > vmax) vmax = vb;
      }
      const top = mid - vmax * half;
      const bot = mid - vmin * half;
      ctx.fillRect(x, top, 1, Math.max(1, bot - top));
    }
  }, []);

  /**
   * The selection band's channel tag. Its y is CSS (the lane's own slice of
   * the display); only its x depends on the view, so it moves with a single
   * style write per view change — never a React render. `data-start` carries
   * the band's start time so this can stay a zero-dependency callback like the
   * rest of the chrome.
   */
  const syncSelTag = useCallback(() => {
    const tag = selTagRef.current;
    if (!tag) return;
    const start = Number(tag.dataset.start ?? '');
    const span = waveView.span;
    const w = widthRef.current;
    if (!Number.isFinite(start) || !(span > 0) || w <= 0) return;
    const x = ((start - waveView.start_s) / span) * w;
    const tw = selTagW.current || tag.offsetWidth;
    tag.style.left = `${Math.round(clamp(x + 3, 2, Math.max(2, w - tw - 2)))}px`;
  }, []);

  const updateChrome = useCallback(() => {
    const v = waveView.view;
    const dur = waveView.duration;
    const span = v.end - v.start;
    const win = ovWinRef.current;
    if (win) {
      if (dur > 0) {
        win.style.display = '';
        win.style.left = `${(v.start / dur) * 100}%`;
        win.style.width = `${(span / dur) * 100}%`;
        win.classList.toggle('full', waveView.isFull);
      } else {
        win.style.display = 'none';
      }
    }
    const range = rangeRef.current;
    if (range) {
      range.textContent =
        dur > 0
          ? `${formatSeconds(v.start, span)} – ${formatSeconds(v.end, span)} of ${dur.toFixed(1)} s`
          : 'No clip';
    }
    const mag = magRef.current;
    if (mag) {
      const f = waveView.factor;
      mag.textContent = waveView.isMaxZoom ? '1:1' : f < 10 ? `${f.toFixed(1)}×` : `${Math.round(f)}×`;
      mag.classList.toggle('deep', waveView.isMaxZoom);
    }
    const panel = panelRef.current;
    if (panel) {
      panel.dataset.viewStart = v.start.toFixed(6);
      panel.dataset.viewEnd = v.end.toFixed(6);
      panel.dataset.viewSpan = span.toFixed(6);
      panel.dataset.duration = dur.toFixed(6);
      panel.dataset.maxZoom = String(waveView.isMaxZoom);
    }
    // D1 · the region's name *is* the readout. Written imperatively for the
    // same reason the ruler is: a zoom must not cost a React render.
    const wrap = wrapRef.current;
    if (wrap) {
      wrap.setAttribute(
        'aria-label',
        dur > 0
          ? `Waveform, showing ${formatSeconds(v.start, span)} to ${formatSeconds(v.end, span)} of ${dur.toFixed(1)} seconds`
          : 'Waveform, no clip loaded',
      );
    }
    const ov = ovRef.current;
    if (ov && dur > 0) {
      ov.setAttribute('aria-valuemin', '0');
      ov.setAttribute('aria-valuemax', dur.toFixed(3));
      ov.setAttribute('aria-valuenow', v.start.toFixed(3));
      ov.setAttribute(
        'aria-valuetext',
        `${formatSeconds(v.start, span)} to ${formatSeconds(v.end, span)}`,
      );
    }
    syncSelTag();
  }, [syncSelTag]);

  // ------------------------------------------------------------ detail fetch

  const runDetailFetch = useCallback(() => {
    detailTimer.current = null;
    // The burst has settled — a good moment to ask the worker what it cost.
    hostRef.current?.requestStats();
    const host = hostRef.current;
    const st = getState();
    const client = st.client;
    // B6 · with the engine gone the base envelope already in hand still draws
    // at every zoom; asking for detail we cannot get would only write a
    // failure into the status line on every wheel event.
    if (!host || !client || st.engineStatus !== 'ready' || !peaksSupported()) return;
    const v = waveView.view;
    const span = v.end - v.start;
    const dur = waveView.duration;
    if (!(span > 0) || !(dur > 0)) return;
    const buckets = bucketsRef.current;
    // Base envelope resolution over this window; skip the round trip when the
    // window is already drawn from as many buckets as the engine would return.
    // The base's own length is the authority — `/api/analyze` is asked for one
    // bucket per device column, so at FIT the base *is* the answer and a
    // whole-file `/api/peaks` would be a second full decode for nothing.
    const baseBuckets = st.original?.peaks.min.length ?? 0;
    if (baseBuckets <= 0) return;
    const baseHere = (baseBuckets * span) / dur;
    const sr = st.original?.sample_rate ?? 48000;
    const gain = Math.min(buckets, Math.max(1, Math.round(span * sr)));
    if (gain < baseHere * DETAIL_GAIN) return;

    const decks: Array<[WaveKind, string]> = [];
    if (st.source && st.original) decks.push(['original', st.source.path]);
    if (st.cleanedPath && st.cleaned) decks.push(['cleaned', st.cleanedPath]);
    if (decks.length === 0) return;

    detailAbort.current?.abort();
    const ac = new AbortController();
    detailAbort.current = ac;

    for (const [kind, path] of decks) {
      const prev = lastDetail.current[kind];
      if (
        prev &&
        prev.path === path &&
        prev.buckets === buckets &&
        Math.abs(prev.start - v.start) < 1e-6 &&
        Math.abs(prev.end - v.end) < 1e-6
      ) {
        continue;
      }
      requestCount.current += 1;
      const panel = panelRef.current;
      if (panel) panel.dataset.peaksRequests = String(requestCount.current);
      void loadPeaks(client, path, v.start, v.end, buckets, ac.signal)
        .then((d) => {
          if (!d || ac.signal.aborted) return;
          const now = waveView.view;
          // Apply only if it still covers what the user is looking at; the
          // worker would fall back to the base envelope otherwise anyway.
          if (d.start > now.start + 1e-4 || d.end < now.end - 1e-4) return;
          // The engine is the authority on how deep the zoom can go: once it
          // answers with one sample per bucket, that window width is the floor.
          if (d.samplesPerBucket <= 1) {
            waveView.setMaxBuckets(d.sampleRate, Math.max(1, d.buckets));
          }
          lastDetail.current[kind] = { path, start: d.start, end: d.end, buckets };
          hostRef.current?.setData(
            kind,
            'detail',
            d.min.slice(),
            d.max.slice(),
            d.rms.slice(),
            d.start,
            d.end,
          );
          const panelEl = panelRef.current;
          if (panelEl) {
            panelEl.dataset.detailBuckets = String(d.buckets);
            panelEl.dataset[kind === 'original' ? 'spbOriginal' : 'spbCleaned'] = String(
              d.samplesPerBucket,
            );
            panelEl.dataset.peaksSupported = 'true';
          }
        })
        .catch((e: unknown) => {
          if (e instanceof DOMException && e.name === 'AbortError') return;
          const panelEl = panelRef.current;
          if (panelEl) panelEl.dataset.peaksSupported = String(peaksSupported());
          // A failed detail query is cosmetic: the base envelope still draws.
          getState().setStatus(
            `Zoom detail unavailable: ${e instanceof Error ? e.message : String(e)}`,
          );
        });
    }
  }, []);

  const scheduleDetailFetch = useCallback(() => {
    if (detailTimer.current !== null) window.clearTimeout(detailTimer.current);
    detailTimer.current = window.setTimeout(runDetailFetch, DETAIL_DEBOUNCE_MS);
  }, [runDetailFetch]);

  // --------------------------------------------------- worker + view wiring

  // Worker + canvas lifecycle. The canvas is created here (not in JSX) because
  // transferControlToOffscreen() is one-shot per element and StrictMode
  // re-runs effects; a fresh element per run keeps that legal.
  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const canvas = document.createElement('canvas');
    canvas.className = 'wave-canvas';
    wrap.prepend(canvas);
    canvasRef.current = canvas;
    const host = new WaveformHost(canvas, {
      onReady: (ok) => setGlOk(ok),
      // The label is the source, not a prefix inside the sentence: this one
      // is the renderer in this page, not the engine.
      onError: (m) => setError(m, 'Waveform renderer'),
      // C1 · the renderer's own cost, on the panel next to the view state, so
      // "60 fps interaction" can be read off the running app instead of
      // inferred from a profiler trace taken once.
      onStats: (st) => {
        const panel = panelRef.current;
        if (!panel) return;
        panel.dataset.frames = String(st.frames);
        panel.dataset.frameMs = st.last.toFixed(3);
        panel.dataset.frameMeanMs = st.mean.toFixed(3);
        panel.dataset.frameP95Ms = st.p95.toFixed(3);
        panel.dataset.frameMaxMs = st.max.toFixed(3);
        panel.dataset.frameMaxAllMs = st.maxAll.toFixed(3);
        panel.dataset.frameIntervalMs = st.interval.toFixed(3);
      },
    });
    hostRef.current = host;
    host.setView(waveView.start_s, waveView.end_s);
    const ro = new ResizeObserver((entries) => {
      const e = entries[0];
      if (!e) return;
      widthRef.current = e.contentRect.width;
      setWidth(Math.round(e.contentRect.width));
    });
    ro.observe(wrap);
    widthRef.current = wrap.getBoundingClientRect().width;
    setWidth(Math.round(wrap.getBoundingClientRect().width));
    const unsub = getPlayer().subscribe((snap) => {
      const hasAudio = getPlayer().hasDeck('original') || getPlayer().hasDeck('cleaned');
      host.setPlayhead(snap.time, hasAudio);
      if (timeRef.current) {
        timeRef.current.textContent = formatTime(snap.time, true);
      }
    });
    return () => {
      unsub();
      ro.disconnect();
      host.dispose();
      hostRef.current = null;
      canvas.remove();
      canvasRef.current = null;
    };
  }, [setError]);

  // The single fast path: view change -> worker + ruler + chrome now, peaks later.
  useEffect(() => {
    const apply = (): void => {
      hostRef.current?.setView(waveView.start_s, waveView.end_s);
      drawRuler();
      updateChrome();
      scheduleDetailFetch();
    };
    apply();
    const unsub = waveView.subscribe(apply);
    return () => {
      unsub();
      if (detailTimer.current !== null) window.clearTimeout(detailTimer.current);
      detailTimer.current = null;
      detailAbort.current?.abort();
      detailAbort.current = null;
    };
  }, [drawRuler, updateChrome, scheduleDetailFetch]);

  // Data feeds
  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    lastDetail.current.original = null;
    if (original) {
      host.setData(
        'original',
        'base',
        toF32(original.peaks.min),
        toF32(original.peaks.max),
        rmsToLinear(original.rms_db),
        0,
        original.duration_s,
      );
      waveView.setSource(original.duration_s, original.sample_rate, bucketsRef.current);
    } else {
      host.clear('original');
      waveView.clear();
    }
    drawOverview();
    updateChrome();
    drawRuler();
  }, [original, drawOverview, updateChrome, drawRuler]);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    lastDetail.current.cleaned = null;
    if (cleaned) {
      host.setData(
        'cleaned',
        'base',
        toF32(cleaned.peaks.min),
        toF32(cleaned.peaks.max),
        rmsToLinear(cleaned.rms_db),
        0,
        cleaned.duration_s,
      );
      scheduleDetailFetch();
    } else {
      host.clear('cleaned');
    }
  }, [cleaned, scheduleDetailFetch]);

  useEffect(() => {
    hostRef.current?.setFocus(abMode);
  }, [abMode]);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    if (!highlight) {
      host.setHighlight(null);
      return;
    }
    // The lane is the channel's position among the report's channels, not the
    // channel number itself: a report whose only units are on channel 1 has
    // one lane, and a 0/2 pair has lanes 0 and 1.
    const lane = highlight.channel === undefined ? -1 : channels.indexOf(highlight.channel);
    host.setHighlight(
      channels.length > 1 && lane >= 0
        ? { start: highlight.start, end: highlight.end, lane, lanes: channels.length }
        : { start: highlight.start, end: highlight.end },
    );
  }, [highlight, channels]);

  // The tag's text and its band start, then one placement pass. Measuring the
  // tag here (once per selection) is what keeps `syncSelTag` free of layout
  // reads on the pan path.
  useEffect(() => {
    const tag = selTagRef.current;
    if (!tag || !highlight || !selName) return;
    tag.dataset.start = String(highlight.start);
    selTagW.current = tag.offsetWidth;
    syncSelTag();
  }, [highlight, selName, syncSelTag]);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const units = report?.units ?? [];
    const set = new Set<number>();
    for (const u of units) {
      set.add(u.start_time_s);
      set.add(u.end_time_s);
    }
    const arr = Float32Array.from([...set].sort((a, b) => a - b));
    host.setUnits(arr);
  }, [report]);

  // Width changes move the deepest useful zoom (one bucket per device column).
  useEffect(() => {
    if (width <= 0) return;
    bucketsRef.current = bucketsFor(width);
    const sr = getState().original?.sample_rate ?? 48000;
    waveView.setMaxBuckets(sr, bucketsRef.current);
    drawRuler();
    drawOverview();
    updateChrome();
    scheduleDetailFetch();
  }, [width, drawRuler, drawOverview, updateChrome, scheduleDetailFetch]);

  // ------------------------------------------------------------ interaction

  const timeAt = useCallback((clientX: number): number | null => {
    const wrap = wrapRef.current;
    if (!wrap) return null;
    const span = waveView.span;
    if (!(span > 0)) return null;
    const r = wrap.getBoundingClientRect();
    if (r.width <= 0) return null;
    const x = clamp(clientX - r.left, 0, r.width);
    return waveView.start_s + (x / r.width) * span;
  }, []);

  // Wheel: zoom about the cursor, shift/horizontal to pan, ctrl (pinch) to zoom.
  // Registered natively so preventDefault actually stops the page from scrolling.
  useEffect(() => {
    const el = displayRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent): void => {
      if (!(waveView.duration > 0)) return;
      const wrap = wrapRef.current;
      if (!wrap) return;
      const r = wrap.getBoundingClientRect();
      if (r.width <= 0) return;
      e.preventDefault();
      const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? r.height : 1;
      const dy = e.deltaY * unit;
      const dx = e.deltaX * unit;
      const span = waveView.span;
      if (e.ctrlKey) {
        // Trackpad pinch arrives as ctrl+wheel with small deltas.
        const frac = clamp((e.clientX - r.left) / r.width, 0, 1);
        waveView.zoomAt(waveView.start_s + frac * span, Math.exp(-dy * 0.01));
      } else if (e.shiftKey) {
        const d = dy !== 0 ? dy : dx;
        waveView.panBy((d / r.width) * span);
      } else if (Math.abs(dx) > Math.abs(dy)) {
        waveView.panBy((dx / r.width) * span);
      } else {
        const frac = clamp((e.clientX - r.left) / r.width, 0, 1);
        waveView.zoomAt(waveView.start_s + frac * span, Math.exp(-dy * 0.002));
      }
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, []);

  const onPointerDown = useCallback(
    (e: PointerEvent<HTMLDivElement>) => {
      if (e.button !== 0) return;
      const t = timeAt(e.clientX);
      if (t === null) return;
      dragging.current = true;
      capture(e.currentTarget, e.pointerId);
      seekTo(t);
    },
    [timeAt],
  );

  const onPointerMove = useCallback(
    (e: PointerEvent<HTMLDivElement>) => {
      const wrap = wrapRef.current;
      if (!wrap) return;
      const r = wrap.getBoundingClientRect();
      const x = e.clientX - r.left;
      if (x >= 0 && x <= r.width) hostRef.current?.setHover(x);
      else hostRef.current?.setHover(null);
      if (dragging.current) {
        const t = timeAt(e.clientX);
        if (t !== null) seekTo(t);
      }
    },
    [timeAt],
  );

  const onPointerUp = useCallback((e: PointerEvent<HTMLDivElement>) => {
    if (dragging.current) {
      dragging.current = false;
      release(e.currentTarget, e.pointerId);
    }
  }, []);

  const onPointerLeave = useCallback(() => {
    hostRef.current?.setHover(null);
  }, []);

  const resetView = useCallback(() => {
    waveView.reset();
  }, []);

  // ------------------------------------------------------------- D1 keyboard
  //
  // Zoom and pan were pointer-only: wheel, ruler drag, double-click. That left
  // the one display that carries the whole clip unreachable without a mouse.
  // Both surfaces are now focus stops with their own key handling, and both
  // call `preventDefault` on every key they consume so the global map (which
  // bails on `defaultPrevented`) never also seeks on the same press.

  /** Pan by a fraction of the visible span, keeping the zoom. */
  const panBy = useCallback((frac: number) => {
    const span = waveView.span;
    if (!(span > 0)) return;
    const d = span * frac;
    waveView.set(waveView.start_s + d, waveView.end_s + d);
  }, []);

  const onViewKey = useCallback(
    (e: ReactKeyboardEvent<HTMLDivElement>) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (!(waveView.duration > 0)) return;
      const mid = waveView.start_s + waveView.span / 2;
      switch (e.key) {
        case '+':
        case '=':
          waveView.zoomAt(mid, 1.6);
          break;
        case '-':
        case '_':
          waveView.zoomAt(mid, 1 / 1.6);
          break;
        case '0':
          waveView.reset();
          break;
        case 'Home':
          waveView.set(0, waveView.span);
          break;
        case 'End':
          waveView.set(waveView.duration - waveView.span, waveView.duration);
          break;
        case 'PageUp':
          panBy(-1);
          break;
        case 'PageDown':
          panBy(1);
          break;
        default:
          return;
      }
      e.preventDefault();
    },
    [panBy],
  );

  /**
   * The overview is `role="scrollbar"`, so it takes the keys a scrollbar takes:
   * arrows step, Page keys jump a windowful, Home/End go to the ends. Arrows
   * mean *scroll* here rather than *seek* — that is what focusing a scrollbar
   * is for — and the seek binding is one Tab away.
   */
  const onOverviewKey = useCallback(
    (e: ReactKeyboardEvent<HTMLDivElement>) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (!(waveView.duration > 0)) return;
      switch (e.key) {
        case 'ArrowLeft':
        case 'ArrowUp':
          panBy(-0.1);
          break;
        case 'ArrowRight':
        case 'ArrowDown':
          panBy(0.1);
          break;
        case 'PageUp':
          panBy(-1);
          break;
        case 'PageDown':
          panBy(1);
          break;
        case 'Home':
          waveView.set(0, waveView.span);
          break;
        case 'End':
          waveView.set(waveView.duration - waveView.span, waveView.duration);
          break;
        default:
          return;
      }
      e.preventDefault();
    },
    [panBy],
  );

  // Ruler: drag to pan.
  const onRulerDown = useCallback((e: PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0 || !(waveView.duration > 0)) return;
    panBase.current = { x: e.clientX, start: waveView.start_s, end: waveView.end_s };
    capture(e.currentTarget, e.pointerId);
    e.currentTarget.classList.add('panning');
  }, []);

  const onRulerMove = useCallback((e: PointerEvent<HTMLDivElement>) => {
    const base = panBase.current;
    const wrap = wrapRef.current;
    if (!base || !wrap) return;
    const r = wrap.getBoundingClientRect();
    if (r.width <= 0) return;
    const span = base.end - base.start;
    const dt = ((e.clientX - base.x) / r.width) * span;
    waveView.set(base.start - dt, base.end - dt);
  }, []);

  const onRulerUp = useCallback((e: PointerEvent<HTMLDivElement>) => {
    if (!panBase.current) return;
    panBase.current = null;
    e.currentTarget.classList.remove('panning');
    release(e.currentTarget, e.pointerId);
  }, []);

  // Overview: click to jump, drag to pan.
  const overviewTime = useCallback((clientX: number): number | null => {
    const el = ovRef.current;
    const dur = waveView.duration;
    if (!el || !(dur > 0)) return null;
    const r = el.getBoundingClientRect();
    if (r.width <= 0) return null;
    return clamp((clientX - r.left) / r.width, 0, 1) * dur;
  }, []);

  const onOvDown = useCallback(
    (e: PointerEvent<HTMLDivElement>) => {
      if (e.button !== 0) return;
      const t = overviewTime(e.clientX);
      if (t === null) return;
      const span = waveView.span;
      if (t >= waveView.start_s && t <= waveView.end_s) {
        ovGrab.current = { offset: t - waveView.start_s };
      } else {
        ovGrab.current = { offset: span / 2 };
        waveView.set(t - span / 2, t + span / 2);
      }
      capture(e.currentTarget, e.pointerId);
      e.currentTarget.classList.add('dragging');
    },
    [overviewTime],
  );

  const onOvMove = useCallback(
    (e: PointerEvent<HTMLDivElement>) => {
      const grab = ovGrab.current;
      if (!grab) return;
      const t = overviewTime(e.clientX);
      if (t === null) return;
      const span = waveView.span;
      waveView.set(t - grab.offset, t - grab.offset + span);
    },
    [overviewTime],
  );

  const onOvUp = useCallback((e: PointerEvent<HTMLDivElement>) => {
    if (!ovGrab.current) return;
    ovGrab.current = null;
    e.currentTarget.classList.remove('dragging');
    release(e.currentTarget, e.pointerId);
  }, []);

  const hasData = Boolean(original);

  return (
    <section className="panel wavepanel" ref={panelRef} aria-label="Waveform">
      <p className="sr-only" id="wave-view-help">
        Plus and minus zoom about the centre, zero fits the whole clip, Page Up
        and Page Down move one windowful, Home and End go to the start and the
        end. Left and right arrows seek the transport.
      </p>
      <div className="panel-head">
        <div className="panel-title">
          <span>Waveform</span>
          {source ? <span className="sub">· {source.name}</span> : null}
        </div>
        <div className="wave-legend-head">
          <span className="wave-zoom">
            <span className="range" ref={rangeRef} data-testid="zoom-range">
              No clip
            </span>
            <span className="mag" ref={magRef} data-testid="zoom-mag">
              1.0×
            </span>
            <button
              type="button"
              className="wave-fit"
              onClick={resetView}
              disabled={!hasData}
              title="Reset zoom to the whole clip (or double-click the waveform)"
            >
              FIT
            </button>
          </span>
          {/* The renderer backend is a build detail, not a readout: shipping
              `WebGL2 · worker` in the panel head is the single most
              debug-build-looking thing on the screen, and it was also what
              pushed the head into an ellipsis at 960px. Only the *failure* is
              user-facing information, so only the failure is shown. */}
          {glOk ? null : <span className="caps warn">Renderer unavailable</span>}
        </div>
      </div>
      <div className="wave-body">
        <div className="display wave-display" ref={displayRef} onDoubleClick={resetView}>
          <div
            className="ruler"
            title="Drag to pan · wheel to zoom · double-click to fit"
            onPointerDown={onRulerDown}
            onPointerMove={onRulerMove}
            onPointerUp={onRulerUp}
            onPointerCancel={onRulerUp}
          >
            <canvas className="ruler-canvas" ref={rulerRef} aria-hidden="true" />
          </div>
          <div
            ref={wrapRef}
            id="wave-canvas-wrap"
            className="wave-canvas-wrap"
            // D1 · the display is an interactive region, not a picture: it
            // seeks on click, zooms on wheel and pans on drag. It therefore
            // gets a focus stop, a role that says "a thing with controls in
            // it", and a label that is re-written on every view change with
            // the window it is currently showing — so the answer to "what am I
            // looking at" is in the accessibility tree, not only in the pixels.
            role="group"
            tabIndex={0}
            aria-label="Waveform display"
            aria-describedby="wave-view-help"
            onKeyDown={onViewKey}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerCancel={onPointerUp}
            onPointerLeave={onPointerLeave}
          >
            {hasData ? (
              <>
                <span ref={timeRef} className="wave-time">
                  00:00.000
                </span>
                {selName ? (
                  /* A7/B4 · the selection band is channel-scoped on a
                     multi-channel report, and this is what says so in words.
                     It rides at the vertical centre of the band's own lane and
                     at the band's left edge, and it is pointer-events: none,
                     so it never takes a click from the display underneath. The
                     waveform itself stays one mixed envelope — see the note in
                     the worker — so this labels the *decision*, not the
                     picture. */
                  <span
                    ref={selTagRef}
                    className="wave-selchan"
                    aria-hidden="true"
                    style={
                      {
                        '--lane': Math.max(0, channels.indexOf(selChannel ?? 0)),
                        '--lanes': channels.length,
                      } as CSSProperties
                    }
                  >
                    {selName.long}
                  </span>
                ) : null}
                <div className="wave-legend" aria-hidden="true">
                  <span className={`orig${cleaned && abMode !== 'original' ? ' off' : ''}`}>
                    <i /> Original
                  </span>
                  <span className={`clean${!cleaned || abMode !== 'cleaned' ? ' off' : ''}`}>
                    <i /> Cleaned
                  </span>
                </div>
              </>
            ) : (
              <div className="wave-empty">
                <span className="big">{analyzing ? 'Analyzing clip' : 'No clip loaded'}</span>
                <span>
                  {analyzing
                    ? 'Decoding and measuring peaks, spectrum and loudness'
                    : 'Open a file or drop one above to begin'}
                </span>
              </div>
            )}
          </div>
        </div>
        <div
          className="wave-overview"
          ref={ovRef}
          role="scrollbar"
          tabIndex={0}
          aria-label="Visible waveform window"
          aria-controls="wave-canvas-wrap"
          aria-orientation="horizontal"
          aria-valuenow={0}
          title="Drag to scroll the visible window · arrows and page keys when focused"
          onKeyDown={onOverviewKey}
          onPointerDown={onOvDown}
          onPointerMove={onOvMove}
          onPointerUp={onOvUp}
          onPointerCancel={onOvUp}
        >
          <canvas className="wave-overview-canvas" ref={ovCanvasRef} aria-hidden="true" />
          <div className="wave-overview-win full" ref={ovWinRef} />
          {hasData ? null : <div className="wave-overview-hint">Overview</div>}
        </div>
        <VerdictStrip />
      </div>
    </section>
  );
}
