import { useCallback, useEffect, useRef, useState } from 'react';
import { getPlayer, type LiveMeterLevels } from '../audio/player';
import { useReducedMotion } from '../state/reducedMotion';

const SILENCE_FLOOR_DB = -70;
const BAR_MIN_DB = -60;
const BAR_MAX_DB = 0;
const CLIP_THRESHOLD_DB = -0.5;
const WARN_THRESHOLD_DB = -3.0;
const REDUCED_MOTION_INTERVAL_MS = 250;
const NORMAL_INTERVAL_MS = 50;

const IDLE_LEVELS: LiveMeterLevels = {
  momentaryLufs: SILENCE_FLOOR_DB,
  shortTermLufs: SILENCE_FLOOR_DB,
  peakDbfs: SILENCE_FLOOR_DB,
  truePeakHoldDbfs: SILENCE_FLOOR_DB,
};

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
 * Compact broadcast-style live loudness (Momentary / Short-Term LUFS) and
 * True Peak Hold meter. Runs `requestAnimationFrame` only while `playing`,
 * respecting `prefers-reduced-motion`.
 */
export function LiveMeter({ playing, readLevels, onResetPeak }: LiveMeterProps) {
  const reducedMotion = useReducedMotion();
  const [levels, setLevels] = useState<LiveMeterLevels>(IDLE_LEVELS);
  const rafRef = useRef<number | null>(null);
  const lastUpdateRef = useRef<number>(0);

  const sampleNow = useCallback((): LiveMeterLevels => {
    return readLevels ? readLevels() : getPlayer().getLiveMeterLevels();
  }, [readLevels]);

  useEffect(() => {
    if (!playing) {
      if (rafRef.current !== null) {
        window.cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      // Drop live momentary bar to floor on stop while preserving latched peak hold.
      setLevels((prev) => ({
        momentaryLufs: SILENCE_FLOOR_DB,
        shortTermLufs: SILENCE_FLOOR_DB,
        peakDbfs: SILENCE_FLOOR_DB,
        truePeakHoldDbfs: prev.truePeakHoldDbfs,
      }));
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
        setLevels(sampleNow());
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
  }, [playing, reducedMotion, sampleNow]);

  const handleResetPeak = (): void => {
    if (onResetPeak) {
      onResetPeak();
    } else {
      getPlayer().resetPeakHold();
    }
    setLevels((prev) => ({
      ...prev,
      truePeakHoldDbfs: SILENCE_FLOOR_DB,
    }));
  };

  const momentaryPct = dbToPercent(levels.momentaryLufs);
  const shortTermPct = dbToPercent(levels.shortTermLufs);
  const isClip = levels.truePeakHoldDbfs >= CLIP_THRESHOLD_DB;
  const isWarn = !isClip && levels.truePeakHoldDbfs >= WARN_THRESHOLD_DB;

  const fillColor = isClip
    ? 'var(--err, #ff5a5f)'
    : levels.momentaryLufs >= -12
      ? 'var(--amber, #ffb347)'
      : 'var(--cyan, #39d0ff)';

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
          className="live-meter-bar-m"
          style={{
            height: '5px',
            width: `${momentaryPct.toFixed(1)}%`,
            background: fillColor,
            borderRadius: '1px',
            transition: reducedMotion ? 'none' : 'width 48ms linear',
          }}
        />
        <div
          className="live-meter-bar-s"
          style={{
            height: '3px',
            width: `${shortTermPct.toFixed(1)}%`,
            background: 'var(--ok, #4ade80)',
            opacity: 0.85,
            borderRadius: '1px',
            transition: reducedMotion ? 'none' : 'width 90ms linear',
          }}
        />
      </div>
      <span
        className="live-meter-readout"
        title="Momentary (400 ms) and Short-Term (3 s) LUFS"
        style={{ whiteSpace: 'nowrap', color: 'var(--fg-3, #949daa)' }}
      >
        <span style={{ color: 'var(--fg-2, #aab2bf)' }}>M</span> {formatDb(levels.momentaryLufs)}{' '}
        <span style={{ color: 'var(--fg-2, #aab2bf)' }}>S</span> {formatDb(levels.shortTermLufs)}
      </span>
      <button
        type="button"
        className={`live-meter-peak${isClip ? ' clip' : isWarn ? ' warn' : ''}`}
        onClick={handleResetPeak}
        aria-label={`Reset peak hold, currently ${isClip ? 'clipping' : `${formatDb(levels.truePeakHoldDbfs)} dBFS`}`}
        title="True peak hold (dBFS) — click to reset"
        style={{
          minWidth: '46px',
          minHeight: '24px',
          padding: '0 6px',
          borderRadius: 'var(--r-xs, 3px)',
          border: `1px solid ${isClip ? 'rgba(255, 90, 95, 0.7)' : isWarn ? 'rgba(255, 179, 71, 0.55)' : 'var(--hair-strong, rgba(255, 255, 255, 0.11))'}`,
          background: isClip
            ? 'rgba(255, 90, 95, 0.22)'
            : isWarn
              ? 'rgba(255, 179, 71, 0.15)'
              : 'var(--panel-2, #1a1e24)',
          color: isClip
            ? '#ffd0d2'
            : isWarn
              ? 'var(--amber-2, #ffd28a)'
              : 'var(--fg-2, #aab2bf)',
          fontFamily: 'inherit',
          fontSize: '11px',
          fontWeight: 500,
          cursor: 'pointer',
          whiteSpace: 'nowrap',
        }}
      >
        {isClip ? 'CLIP' : `${formatDb(levels.truePeakHoldDbfs)} dB`}
      </button>
    </div>
  );
}
