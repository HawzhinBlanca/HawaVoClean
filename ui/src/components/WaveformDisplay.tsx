import { useCallback, useEffect, useMemo, useRef, useState, type PointerEvent } from 'react';
import { getPlayer } from '../audio/player';
import { WaveformHost } from '../render/WaveformHost';
import { formatTime, formatTimeShort, timeTicks } from '../render/ticks';
import { seekTo } from '../state/actions';
import { useStore } from '../state/store';
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

export function WaveformDisplay() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const hostRef = useRef<WaveformHost | null>(null);
  const timeRef = useRef<HTMLSpanElement | null>(null);
  const dragging = useRef(false);
  const [width, setWidth] = useState(0);
  const [glOk, setGlOk] = useState(true);

  const original = useStore((s) => s.original);
  const cleaned = useStore((s) => s.cleaned);
  const report = useStore((s) => s.report);
  const abMode = useStore((s) => s.abMode);
  const highlight = useStore((s) => s.highlightRange);
  const duration = useStore((s) => s.duration);
  const analyzing = useStore((s) => s.analyzing);
  const source = useStore((s) => s.source);
  const setError = useStore((s) => s.setError);

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
      onError: (m) => setError(`Waveform renderer: ${m}`),
    });
    hostRef.current = host;
    const ro = new ResizeObserver((entries) => {
      const e = entries[0];
      if (e) setWidth(Math.round(e.contentRect.width));
    });
    ro.observe(wrap);
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

  // Data feeds
  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    if (original) {
      host.setData(
        'original',
        toF32(original.peaks.min),
        toF32(original.peaks.max),
        rmsToLinear(original.rms_db),
        original.duration_s,
      );
    } else {
      host.clear('original');
    }
  }, [original]);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    if (cleaned) {
      host.setData(
        'cleaned',
        toF32(cleaned.peaks.min),
        toF32(cleaned.peaks.max),
        rmsToLinear(cleaned.rms_db),
        cleaned.duration_s,
      );
    } else {
      host.clear('cleaned');
    }
  }, [cleaned]);

  useEffect(() => {
    hostRef.current?.setFocus(abMode);
  }, [abMode]);

  useEffect(() => {
    hostRef.current?.setHighlight(highlight);
  }, [highlight]);

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

  // Pointer interaction (seek + hover)
  const timeAt = useCallback(
    (clientX: number): number | null => {
      const wrap = wrapRef.current;
      if (!wrap || !(duration > 0)) return null;
      const r = wrap.getBoundingClientRect();
      const x = Math.min(r.width, Math.max(0, clientX - r.left));
      return (x / r.width) * duration;
    },
    [duration],
  );

  const onPointerDown = useCallback(
    (e: PointerEvent<HTMLDivElement>) => {
      if (e.button !== 0) return;
      const t = timeAt(e.clientX);
      if (t === null) return;
      dragging.current = true;
      e.currentTarget.setPointerCapture(e.pointerId);
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
      try {
        e.currentTarget.releasePointerCapture(e.pointerId);
      } catch {
        /* already released */
      }
    }
  }, []);

  const onPointerLeave = useCallback(() => {
    hostRef.current?.setHover(null);
  }, []);

  const ticks = useMemo(() => (duration > 0 && width > 0 ? timeTicks(duration, width, 72) : []), [duration, width]);

  const hasData = Boolean(original);

  return (
    <section className="panel wavepanel">
      <div className="panel-head">
        <div className="panel-title">
          <span>Waveform</span>
          {source ? <span className="sub">· {source.name}</span> : null}
        </div>
        <div className="wave-legend-head">
          <span className="caps">{glOk ? 'WebGL2 · worker' : 'Renderer unavailable'}</span>
        </div>
      </div>
      <div className="wave-body">
        <div className="display wave-display">
          <div className="ruler" aria-hidden="true">
            {ticks.map((t) => {
              const x = (t.time / duration) * width;
              return (
                <span key={t.time}>
                  <i className={`tick${t.major ? ' major' : ''}`} style={{ left: x }} />
                  {t.major ? (
                    <span className="lbl" style={{ left: x }}>
                      {formatTimeShort(t.time)}
                    </span>
                  ) : null}
                </span>
              );
            })}
          </div>
          <div
            ref={wrapRef}
            className="wave-canvas-wrap"
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
                <span>{analyzing ? 'Decoding and measuring peaks, spectrum and loudness' : 'Open a file or drop one above to begin'}</span>
              </div>
            )}
          </div>
        </div>
        <VerdictStrip />
      </div>
    </section>
  );
}
