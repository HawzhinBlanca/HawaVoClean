import { useCallback, useEffect, useRef } from 'react';
import { getPlayer, type LiveMeterLevels } from '../audio/player';
import { useReducedMotion } from '../state/reducedMotion';
import { useStore } from '../state/store';

const SILENCE_FLOOR_DB = -70;
const BAR_MIN_DB = -60;
const BAR_MAX_DB = 0;
const CLIP_THRESHOLD_DB = -0.5;
const WARN_THRESHOLD_DB = -3.0;
const REDUCED_MOTION_INTERVAL_MS = 250;
const NORMAL_INTERVAL_MS = 50;

export interface LiveMeterProps {
  playing: boolean;
  /** Optional level reader override (primarily for tests). */
  readLevels?: () => LiveMeterLevels;
  /** Optional peak-hold reset callback (primarily for tests). */
  onResetPeak?: () => void;
}

function dbToPercent(db: number): number {
  if (!Number.isFinite(db) || db <= BAR_MIN_DB) return 0;
  if (db >= BAR_MAX_DB) return 100;
  return ((db - BAR_MIN_DB) / (BAR_MAX_DB - BAR_MIN_DB)) * 100;
}

function formatDb(db: number): string {
  if (!Number.isFinite(db) || db <= SILENCE_FLOOR_DB + 0.5) return '--.-';
  return db.toFixed(1);
}

/**
 * Compact broadcast-style live K-weighted loudness (Momentary / Short-Term LUFS),
 * True Peak Hold, and real-time Noise Reduction (NR dB Removed) meter.
 * Updates all bar widths and numeric readouts directly via DOM refs in rAF
 * for zero React re-renders per second during playback.
 */
export function LiveMeter({ playing, readLevels, onResetPeak }: LiveMeterProps) {
  const reducedMotion = useReducedMotion();
  const cleanedActive = useStore(
    (s) => s.abMode === 'cleaned' && (Boolean(s.cleanedPath) || s.previewActive),
  );
  const rafRef = useRef<number | null>(null);
  const lastUpdateRef = useRef<number>(0);
  const peakHoldRef = useRef<number>(SILENCE_FLOOR_DB);

  const barMRef = useRef<HTMLDivElement | null>(null);
  const barSRef = useRef<HTMLDivElement | null>(null);
  const mValRef = useRef<HTMLSpanElement | null>(null);
  const sValRef = useRef<HTMLSpanElement | null>(null);
  const nrWrapRef = useRef<HTMLSpanElement | null>(null);
  const nrValRef = useRef<HTMLSpanElement | null>(null);
  const peakBtnRef = useRef<HTMLButtonElement | null>(null);

  const sampleNow = useCallback((): LiveMeterLevels => {
    return readLevels ? readLevels() : getPlayer().getLiveMeterLevels();
  }, [readLevels]);

  const paintLevels = useCallback(
    (levels: LiveMeterLevels): void => {
      peakHoldRef.current = levels.truePeakHoldDbfs;
      const momentaryPct = dbToPercent(levels.momentaryLufs);
      const shortTermPct = dbToPercent(levels.shortTermLufs);
      const isClip = levels.truePeakHoldDbfs >= CLIP_THRESHOLD_DB;
      const isWarn = !isClip && levels.truePeakHoldDbfs >= WARN_THRESHOLD_DB;
      const fillColor = isClip
        ? 'var(--err, #ff5a5f)'
        : levels.momentaryLufs >= -12
          ? 'var(--amber, #ffb347)'
          : 'var(--cyan, #39d0ff)';

      if (barMRef.current) {
        barMRef.current.style.width = `${momentaryPct.toFixed(1)}%`;
        barMRef.current.style.background = fillColor;
      }
      if (barSRef.current) {
        barSRef.current.style.width = `${shortTermPct.toFixed(1)}%`;
      }
      if (mValRef.current) {
        mValRef.current.textContent = formatDb(levels.momentaryLufs);
      }
      if (sValRef.current) {
        sValRef.current.textContent = formatDb(levels.shortTermLufs);
      }

      if (nrWrapRef.current && nrValRef.current) {
        const nr = levels.nrDb;
        const showNr =
          (cleanedActive || Boolean(readLevels)) &&
          typeof nr === 'number' &&
          Number.isFinite(nr) &&
          nr >= 0.2;
        if (showNr) {
          nrWrapRef.current.style.display = 'inline-flex';
          nrValRef.current.textContent = `\u2212${nr.toFixed(1)} dB`;
        } else {
          nrWrapRef.current.style.display = 'none';
        }
      }

      const btn = peakBtnRef.current;
      if (btn) {
        btn.className = `live-meter-peak${isClip ? ' clip' : isWarn ? ' warn' : ''}`;
        btn.textContent = isClip ? 'CLIP' : `${formatDb(levels.truePeakHoldDbfs)} dB`;
        btn.setAttribute(
          'aria-label',
          `Reset peak hold, currently ${isClip ? 'clipping' : `${formatDb(levels.truePeakHoldDbfs)} dBFS`}`,
        );
        btn.style.borderColor = isClip
          ? 'rgba(255, 90, 95, 0.7)'
          : isWarn
            ? 'rgba(255, 179, 71, 0.55)'
            : 'var(--hair-strong, rgba(255, 255, 255, 0.11))';
        btn.style.background = isClip
          ? 'rgba(255, 90, 95, 0.22)'
          : isWarn
            ? 'rgba(255, 179, 71, 0.15)'
            : 'var(--panel-2, #1a1e24)';
        btn.style.color = isClip
          ? '#ffd0d2'
          : isWarn
            ? 'var(--amber-2, #ffd28a)'
            : 'var(--fg-2, #aab2bf)';
      }
    },
    [cleanedActive, readLevels],
  );

  useEffect(() => {
    if (!playing) {
      if (rafRef.current !== null) {
        window.cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      // Drop live momentary bar to floor on stop while preserving latched peak hold.
      paintLevels({
        momentaryLufs: SILENCE_FLOOR_DB,
        shortTermLufs: SILENCE_FLOOR_DB,
        peakDbfs: SILENCE_FLOOR_DB,
        truePeakHoldDbfs: peakHoldRef.current,
        nrDb: null,
      });
      return;
    }

    let cancelled = false;
    const intervalMs = reducedMotion ? REDUCED_MOTION_INTERVAL_MS : NORMAL_INTERVAL_MS;
    lastUpdateRef.current = 0;

    const tick = (ts: number): void => {
      rafRef.current = null;
      if (cancelled) return;
      if (lastUpdateRef.current === 0 || ts - lastUpdateRef.current >= intervalMs) {
        lastUpdateRef.current = ts;
        paintLevels(sampleNow());
      }
      if (!cancelled) {
        rafRef.current = window.requestAnimationFrame(tick);
      }
    };

    rafRef.current = window.requestAnimationFrame(tick);
    return () => {
      cancelled = true;
      if (rafRef.current !== null) {
        window.cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [playing, reducedMotion, sampleNow, paintLevels]);

  const handleResetPeak = (): void => {
    if (onResetPeak) {
      onResetPeak();
    } else {
      getPlayer().resetPeakHold();
    }
    peakHoldRef.current = SILENCE_FLOOR_DB;
    const btn = peakBtnRef.current;
    if (btn) {
      btn.className = 'live-meter-peak';
      btn.textContent = '--.- dB';
      btn.setAttribute('aria-label', 'Reset peak hold, currently --.- dBFS');
      btn.style.borderColor = 'var(--hair-strong, rgba(255, 255, 255, 0.11))';
      btn.style.background = 'var(--panel-2, #1a1e24)';
      btn.style.color = 'var(--fg-2, #aab2bf)';
    }
  };

  return (
    <div
      className="live-meter"
      role="group"
      aria-label="Live loudness and peak meter"
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '6px',
        padding: '2px 6px',
        minHeight: '28px',
        borderRadius: 'var(--r-sm, 4px)',
        background: 'var(--display, #07080a)',
        border: '1px solid var(--hair, rgba(255, 255, 255, 0.07))',
        boxShadow: 'var(--elev-0, inset 0 1px 3px rgba(0,0,0,0.7))',
        fontFamily: 'var(--font-mono, ui-monospace, monospace)',
        fontVariantNumeric: 'tabular-nums',
        fontSize: '11px',
        color: 'var(--fg-2, #aab2bf)',
        flexShrink: 0,
      }}
    >
      <div
        className="live-meter-bars"
        aria-hidden="true"
        style={{
          position: 'relative',
          width: '68px',
          height: '12px',
          borderRadius: '2px',
          background: 'var(--panel-2, #1a1e24)',
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          padding: '1px 0',
        }}
      >
        <div
          ref={barMRef}
          className="live-meter-bar-m"
          style={{
            height: '5px',
            width: '0.0%',
            background: 'var(--cyan, #39d0ff)',
            borderRadius: '1px',
            transition: reducedMotion ? 'none' : 'width 48ms linear',
          }}
        />
        <div
          ref={barSRef}
          className="live-meter-bar-s"
          style={{
            height: '3px',
            width: '0.0%',
            background: 'var(--ok, #4ade80)',
            opacity: 0.85,
            borderRadius: '1px',
            transition: reducedMotion ? 'none' : 'width 90ms linear',
          }}
        />
      </div>
      <span
        className="live-meter-readout"
        title="ITU-R BS.1770 K-weighted Momentary (400 ms) and Short-Term (3 s) LUFS"
        style={{ whiteSpace: 'nowrap', color: 'var(--fg-3, #949daa)' }}
      >
        <span style={{ color: 'var(--fg-2, #aab2bf)' }}>M</span>{' '}
        <span ref={mValRef}>--.-</span>{' '}
        <span style={{ color: 'var(--fg-2, #aab2bf)' }}>S</span>{' '}
        <span ref={sValRef}>--.-</span>
      </span>
      <span
        ref={nrWrapRef}
        className="live-meter-nr"
        title="Real-time noise reduction (dB removed compared to Original)"
        style={{
          display: 'none',
          alignItems: 'center',
          gap: '3px',
          padding: '0 5px',
          borderRadius: '3px',
          background: 'rgba(44, 212, 255, 0.12)',
          border: '1px solid rgba(44, 212, 255, 0.32)',
          color: 'var(--cyan-2, #9cecff)',
          whiteSpace: 'nowrap',
          fontSize: '10px',
        }}
      >
        <span>NR</span>
        <span ref={nrValRef}>−0.0 dB</span>
      </span>
      <button
        ref={peakBtnRef}
        type="button"
        className="live-meter-peak"
        onClick={handleResetPeak}
        aria-label="Reset peak hold, currently --.- dBFS"
        title="True peak hold (dBFS) — click to reset"
        style={{
          minWidth: '46px',
          minHeight: '24px',
          padding: '0 6px',
          borderRadius: 'var(--r-xs, 3px)',
          border: '1px solid var(--hair-strong, rgba(255, 255, 255, 0.11))',
          background: 'var(--panel-2, #1a1e24)',
          color: 'var(--fg-2, #aab2bf)',
          fontFamily: 'inherit',
          fontSize: '11px',
          fontWeight: 500,
          cursor: 'pointer',
          whiteSpace: 'nowrap',
        }}
      >
        --.- dB
      </button>
    </div>
  );
}
