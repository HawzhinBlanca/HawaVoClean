import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent, type WheelEvent as ReactWheelEvent } from 'react';
import { getPlayer } from '../audio/player';
import { SPECTRUM_PLOT_INSETS, SpectrumRenderer, type SpectrumTiltSlope } from '../render/spectrum';
import '../render/spectrum.css';
import { updateRackSettings } from '../state/actions';
import { useI18n } from '../state/i18n';
import { scheduleLivePreview } from '../state/liveAudition';
import { HPF_MAX_HZ, HPF_MIN_HZ, type DeliveryPresetId } from '../state/presets';
import { useStore } from '../state/store';

const TILT_STEPS: readonly SpectrumTiltSlope[] = [0, 3, 4.5];

function evaluateDeliveryCompliance(
  preset: DeliveryPresetId,
  lufs: number | null,
  peak: number | null,
  channels: number,
): { status: 'pass' | 'warn' | 'clip'; label: string; title: string } | null {
  if (preset === 'match_input' || lufs === null || !Number.isFinite(lufs)) return null;
  const targets: Record<Exclude<DeliveryPresetId, 'match_input'>, { lufs: number; ceiling: number; name: string }> = {
    podcast: { lufs: channels <= 1 ? -19.0 : -16.0, ceiling: -1.0, name: 'Podcast' },
    ebu_r128: { lufs: -23.0, ceiling: -1.0, name: 'EBU R128' },
    atsc_a85: { lufs: -24.0, ceiling: -2.0, name: 'ATSC A/85' },
    dialogue_gated: { lufs: -27.0, ceiling: -2.0, name: 'Dialogue' },
    streaming: { lufs: -14.0, ceiling: -1.0, name: 'Streaming' },
  };
  const spec = targets[preset];
  if (!spec) return null;
  const peakOver = peak !== null && Number.isFinite(peak) && peak > spec.ceiling + 0.15;
  if (peakOver) {
    return {
      status: 'clip',
      label: 'PEAK',
      title: `${spec.name}: true peak ${peak.toFixed(1)} dBTP exceeds ${spec.ceiling.toFixed(1)} dBTP ceiling`,
    };
  }
  const delta = Math.abs(lufs - spec.lufs);
  if (delta <= 1.5) {
    return {
      status: 'pass',
      label: `${spec.name} OK`,
      title: `Compliant with ${spec.name} (${lufs.toFixed(1)} vs ${spec.lufs.toFixed(1)} LUFS target)`,
    };
  }
  return {
    status: 'warn',
    label: `${lufs - spec.lufs > 0 ? '+' : ''}${(lufs - spec.lufs).toFixed(1)} LU`,
    title: `${spec.name} target is ${spec.lufs.toFixed(1)} LUFS (currently ${lufs.toFixed(1)} LUFS)`,
  };
}

const FALLBACK_PLOT_INSETS = { left: 33, right: 9, top: 11, bottom: 17 } as const;

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'] as const;

export function hzToMusicalNote(hz: number): string {
  if (!Number.isFinite(hz) || hz <= 0) return '—';
  const midi = Math.round(69 + 12 * Math.log2(hz / 440));
  const note = NOTE_NAMES[((midi % 12) + 12) % 12] ?? 'A';
  const oct = Math.floor(midi / 12) - 1;
  return `${note}${oct}`;
}

const SPEC_F_MIN = 20;
const SPEC_F_MAX = 20000;
const SPEC_LOG_SPAN = Math.log2(SPEC_F_MAX / SPEC_F_MIN);

function hzToNormX(hz: number): number {
  const clamped = Math.min(SPEC_F_MAX, Math.max(SPEC_F_MIN, hz));
  return Math.log2(clamped / SPEC_F_MIN) / SPEC_LOG_SPAN;
}

function normXToHpfHz(normX: number): number {
  const f = SPEC_F_MIN * Math.pow(2, Math.min(1, Math.max(0, normX)) * SPEC_LOG_SPAN);
  return Math.min(HPF_MAX_HZ, Math.max(HPF_MIN_HZ, Math.round(f / 5) * 5));
}

/** The canvas carries a handle to its renderer so perf probes and scripted
 *  browser checks can step the display without going through React. */
type SpectrumCanvas = HTMLCanvasElement & { __spectrum?: SpectrumRenderer };

function toF32(a: number[]): Float32Array {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] ?? 0;
  return out;
}

type LegendKind = 'orig' | 'clean' | 'removed' | 'live';

interface LegendItemProps {
  kind: LegendKind;
  label: string;
  on: boolean;
  deck?: 'original' | 'cleaned';
  hint: string;
  /** What "off" means for this series, when "not present" is too thin. */
  offHint?: string;
}

function LegendItem({ kind, label, on, deck, hint, offHint }: LegendItemProps) {
  const cls = `item ${kind} ${on ? 'on' : 'off'}${deck ? ` deck-${deck}` : ''}`;
  return (
    <span
      className={cls}
      role="listitem"
      aria-label={`${label}: ${on ? hint : (offHint ?? 'not present')}`}
    >
      <i className="sw" aria-hidden="true" />
      <span className="txt">{label}</span>
    </span>
  );
}

const PIANO_OCTAVES: readonly { label: string; hz: number }[] = [
  { label: 'C1', hz: 32.7 },
  { label: 'C2', hz: 65.41 },
  { label: 'C3', hz: 130.81 },
  { label: 'C4', hz: 261.63 },
  { label: 'C5', hz: 523.25 },
  { label: 'C6', hz: 1046.5 },
  { label: 'C7', hz: 2093.0 },
  { label: 'C8', hz: 4186.01 },
];

export interface SpectrumDisplayProps {
  /** Show the FabFilter-style piano keyboard pitch scale and inline telemetry pills. */
  wide?: boolean;
}

export function SpectrumDisplay({ wide = false }: SpectrumDisplayProps = {}) {
  const { locale } = useI18n();
  const isCkb = locale === 'ckb';
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rendererRef = useRef<SpectrumRenderer | null>(null);
  const original = useStore((s) => s.original);
  const cleaned = useStore((s) => s.cleaned);
  const abMode = useStore((s) => s.abMode);
  const playing = useStore((s) => s.playing);
  const rackSettings = useStore((s) => s.rackSettings);
  const [removedCoverage, setRemovedCoverage] = useState(0);
  const [draggingHpf, setDraggingHpf] = useState(false);
  const [tiltSlope, setTiltSlope] = useState<SpectrumTiltSlope>(0);
  const [frozen, setFrozen] = useState(false);
  const [soloArmed, setSoloArmed] = useState(false);
  const [soloQ, setSoloQ] = useState(4.5);
  const [soloBand, setSoloBand] = useState<{
    hz: number;
    leftPct: number;
    widthPct: number;
    q: number;
  } | null>(null);
  const soloPointerIdRef = useRef<number | null>(null);
  const [hoverInfo, setHoverInfo] = useState<{
    leftPct: number;
    topPct: number;
    hz: number;
    db: number;
    note: string;
  } | null>(null);
  const displayRef = useRef<HTMLDivElement | null>(null);
  const plotOverlayRef = useRef<HTMLDivElement | null>(null);
  const insets = SPECTRUM_PLOT_INSETS ?? FALLBACK_PLOT_INSETS;

  useEffect(() => {
    const canvas = canvasRef.current as SpectrumCanvas | null;
    if (!canvas) return;
    const r = new SpectrumRenderer(canvas);
    rendererRef.current = r;
    canvas.__spectrum = r;
    return () => {
      r.dispose();
      delete canvas.__spectrum;
      rendererRef.current = null;
    };
  }, []);

  useEffect(() => {
    rendererRef.current?.setCurve(
      'original',
      original ? { freqs: toF32(original.spectrum.freqs_hz), db: toF32(original.spectrum.db) } : null,
    );
  }, [original]);

  useEffect(() => {
    rendererRef.current?.setCurve(
      'cleaned',
      cleaned ? { freqs: toF32(cleaned.spectrum.freqs_hz), db: toF32(cleaned.spectrum.db) } : null,
    );
  }, [cleaned]);

  /**
   * Honesty · the REMOVED band is drawn by subtraction, so whether it exists
   * is a property of the two curves and not of the fact that two curves are
   * loaded. This asks the renderer for the coverage it will actually paint —
   * it was possible, and it happened on a production run that enhanced no
   * units, for the key to read REMOVED lit and the canvas to promise "the
   * removed energy between them" over a plot with no amber in it anywhere.
   *
   * `setCurve` rebuilds the diff synchronously, and this effect is declared
   * after both of the effects that call it, so it reads a settled renderer.
   */
  useEffect(() => {
    setRemovedCoverage(rendererRef.current?.removedCoverage() ?? 0);
  }, [original, cleaned]);

  useEffect(() => {
    rendererRef.current?.setFocus(abMode);
  }, [abMode]);

  // Live analyser overlay follows playback; the analyser is created lazily on
  // the first play (user gesture), so re-check whenever playing flips.
  useEffect(() => {
    const player = getPlayer();
    rendererRef.current?.setLive(player.getAnalyser(), player.sampleRate, abMode, playing);
  }, [playing, abMode]);

  useEffect(() => {
    rendererRef.current?.setTiltDbPerOct?.(tiltSlope);
  }, [tiltSlope]);

  useEffect(() => {
    rendererRef.current?.setFrozen?.(frozen);
  }, [frozen]);

  // Always release real-time frequency band solo on unmount.
  useEffect(() => {
    return () => {
      getPlayer().setBandSolo?.(false);
    };
  }, []);

  const both = original !== null && cleaned !== null;
  // The band is lit by the pixels it will have, not by the pair of curves.
  const hasRemoved = both && removedCoverage > 0;
  const removedPct = Math.round(removedCoverage * 100);
  const hpfEnabled = rackSettings.hpfEnabled !== false;
  const hpfHz = rackSettings.hpfCutoffHz;
  const hpfLeftPct = Math.min(100, Math.max(0, hzToNormX(hpfHz) * 100));

  // Smooth FabFilter-style 24 dB/oct HPF roll-off curve in SVG 0..100 plot coordinates
  const hpfCurvePath = `M 0 100 C ${(hpfLeftPct * 0.42).toFixed(2)} 100, ${(hpfLeftPct * 0.78).toFixed(2)} 42, ${hpfLeftPct.toFixed(2)} 38 L 100 38`;
  const hpfFillPath = `${hpfCurvePath} L 100 100 Z`;

  // Active finishing band highlights (logarithmic 20 Hz - 20 kHz projection)
  const deplosiveRightPct = hzToNormX(160) * 100;
  const tonalLeftPct = hzToNormX(1200) * 100;
  const tonalWidthPct = (hzToNormX(4000) - hzToNormX(1200)) * 100;
  const deessLeftPct = hzToNormX(4500) * 100;
  const deessWidthPct = (hzToNormX(9500) - hzToNormX(4500)) * 100;

  const activeLufs = (cleaned ?? original)?.loudness.integrated_lufs ?? null;
  const activePeak = (cleaned ?? original)?.loudness.true_peak_dbtp ?? null;
  const activeNoise = (cleaned ?? original)?.noise_floor_db ?? null;
  const compliance =
    wide && cleaned
      ? evaluateDeliveryCompliance(
          rackSettings.deliveryPreset,
          activeLufs,
          activePeak,
          cleaned.channels ?? original?.channels ?? 1,
        )
      : null;

  const updateBandSoloAtClientX = (clientX: number, qValue = soloQ): void => {
    const plotRect = plotOverlayRef.current?.getBoundingClientRect();
    if (!plotRect || plotRect.width <= 0) return;
    const normX = Math.min(1, Math.max(0, (clientX - plotRect.left) / plotRect.width));
    const hz = SPEC_F_MIN * Math.pow(2, normX * SPEC_LOG_SPAN);
    const clampedHz = Math.max(20, Math.min(18000, hz));
    const halfBw = clampedHz / (2 * Math.max(1, qValue));
    const loPct = hzToNormX(Math.max(SPEC_F_MIN, clampedHz - halfBw)) * 100;
    const hiPct = hzToNormX(Math.min(SPEC_F_MAX, clampedHz + halfBw)) * 100;
    setSoloBand({
      hz: clampedHz,
      leftPct: loPct,
      widthPct: Math.max(1.8, hiPct - loPct),
      q: qValue,
    });
    getPlayer().setBandSolo?.(true, clampedHz, qValue);
  };

  const stopBandSolo = (): void => {
    soloPointerIdRef.current = null;
    setSoloBand(null);
    getPlayer().setBandSolo?.(false);
  };

  const onDisplayPointerDown = (e: ReactPointerEvent<HTMLDivElement>): void => {
    if (e.button !== 0 || draggingHpf || !original) return;
    if (!soloArmed && !e.altKey) return;
    const target = e.target as HTMLElement | null;
    if (target?.closest('button')) return;
    e.preventDefault();
    soloPointerIdRef.current = e.pointerId;
    e.currentTarget.setPointerCapture?.(e.pointerId);
    updateBandSoloAtClientX(e.clientX, soloQ);
  };

  const onDisplayPointerUp = (e: ReactPointerEvent<HTMLDivElement>): void => {
    if (soloPointerIdRef.current !== e.pointerId) return;
    try {
      e.currentTarget.releasePointerCapture?.(e.pointerId);
    } catch {
      // ignore
    }
    stopBandSolo();
  };

  const onDisplayWheel = (e: ReactWheelEvent<HTMLDivElement>): void => {
    if (!soloBand) return;
    e.preventDefault();
    const delta = e.deltaY < 0 ? 0.6 : -0.6;
    const nextQ = Math.round(Math.max(1.0, Math.min(12.0, soloQ + delta)) * 10) / 10;
    setSoloQ(nextQ);
    const halfBw = soloBand.hz / (2 * Math.max(1, nextQ));
    const loPct = hzToNormX(Math.max(SPEC_F_MIN, soloBand.hz - halfBw)) * 100;
    const hiPct = hzToNormX(Math.min(SPEC_F_MAX, soloBand.hz + halfBw)) * 100;
    setSoloBand({ ...soloBand, q: nextQ, leftPct: loPct, widthPct: Math.max(1.8, hiPct - loPct) });
    getPlayer().setBandSolo?.(true, soloBand.hz, nextQ);
  };

  const updateHpfFromPointer = (clientX: number): void => {
    const plotRect = plotOverlayRef.current?.getBoundingClientRect();
    let plotLeft = 0;
    let plotWidth = 0;
    if (plotRect && plotRect.width > 0) {
      plotLeft = plotRect.left;
      plotWidth = plotRect.width;
    } else {
      const dispRect = displayRef.current?.getBoundingClientRect();
      if (!dispRect || dispRect.width <= insets.left + insets.right) return;
      plotLeft = dispRect.left + insets.left;
      plotWidth = dispRect.width - insets.left - insets.right;
    }
    const normX = (clientX - plotLeft) / plotWidth;
    const nextHz = normXToHpfHz(normX);
    if (nextHz !== rackSettings.hpfCutoffHz || !hpfEnabled) {
      updateRackSettings({ ...rackSettings, hpfEnabled: true, hpfCutoffHz: nextHz });
      scheduleLivePreview();
    }
  };

  const onHpfPointerDown = (e: ReactPointerEvent<HTMLButtonElement>): void => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.currentTarget.setPointerCapture?.(e.pointerId);
    setDraggingHpf(true);
    updateHpfFromPointer(e.clientX);
  };

  const onHpfPointerMove = (e: ReactPointerEvent<HTMLButtonElement>): void => {
    if (!draggingHpf) return;
    updateHpfFromPointer(e.clientX);
  };

  const onHpfPointerUp = (e: ReactPointerEvent<HTMLButtonElement>): void => {
    if (!draggingHpf) return;
    setDraggingHpf(false);
    try {
      e.currentTarget.releasePointerCapture?.(e.pointerId);
    } catch {
      // ignore
    }
  };

  const onPlotPointerMove = (e: ReactPointerEvent<HTMLDivElement>): void => {
    if (soloPointerIdRef.current === e.pointerId) {
      updateBandSoloAtClientX(e.clientX, soloQ);
    }
    const plotRect = plotOverlayRef.current?.getBoundingClientRect();
    if (!plotRect || plotRect.width <= 0 || plotRect.height <= 0) return;
    const normX = (e.clientX - plotRect.left) / plotRect.width;
    const normY = (e.clientY - plotRect.top) / plotRect.height;
    if (normX < 0 || normX > 1 || normY < 0 || normY > 1) {
      if (hoverInfo) setHoverInfo(null);
      return;
    }
    const hz = SPEC_F_MIN * Math.pow(2, normX * SPEC_LOG_SPAN);
    const db = -(normY * 96);
    setHoverInfo({
      leftPct: normX * 100,
      topPct: normY * 100,
      hz,
      db,
      note: hzToMusicalNote(hz),
    });
  };

  const onPlotDoubleClick = (e: React.MouseEvent<HTMLDivElement>): void => {
    const plotRect = plotOverlayRef.current?.getBoundingClientRect();
    if (!plotRect || plotRect.width <= 0) return;
    const normX = (e.clientX - plotRect.left) / plotRect.width;
    const rawHz = SPEC_F_MIN * Math.pow(2, Math.min(1, Math.max(0, normX)) * SPEC_LOG_SPAN);
    if (rawHz <= HPF_MAX_HZ + 40) {
      const snapped = normXToHpfHz(normX);
      updateRackSettings({ ...rackSettings, hpfEnabled: true, hpfCutoffHz: snapped });
      scheduleLivePreview();
    }
  };

  return (
    <>
      <div className="panel-head">
        <h2 className="panel-title">
          <span>{isCkb ? 'شەبەنگ' : 'Spectrum'}</span>
          <span className="sub">· LTAS</span>
        </h2>
        <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
          {wide && original ? (
            <div className="spec-inline-telemetry" aria-label="Loudness and noise telemetry">
              <span className="sit-pill">
                <span className="sit-k">LUFS</span>
                <span className="sit-v mono">
                  {activeLufs !== null && Number.isFinite(activeLufs) ? activeLufs.toFixed(1) : '—'}
                </span>
              </span>
              <span className="sit-pill">
                <span className="sit-k">PEAK</span>
                <span className="sit-v mono">
                  {activePeak !== null && Number.isFinite(activePeak) ? `${activePeak.toFixed(1)} dBTP` : '—'}
                </span>
              </span>
              <span className="sit-pill">
                <span className="sit-k">NOISE</span>
                <span className="sit-v mono">
                  {activeNoise !== null && Number.isFinite(activeNoise) ? `${activeNoise.toFixed(1)} dB` : '—'}
                </span>
              </span>
              {compliance ? (
                <span
                  className={`sit-pill sit-compliance status-${compliance.status}`}
                  title={compliance.title}
                >
                  <span className="sit-v mono">{compliance.label}</span>
                </span>
              ) : null}
            </div>
          ) : null}
          <button
            type="button"
            className="spec-hpf-chip"
            aria-pressed={tiltSlope !== 0}
            onClick={() => {
              const idx = TILT_STEPS.indexOf(tiltSlope);
              setTiltSlope(TILT_STEPS[(idx + 1) % TILT_STEPS.length] ?? 0);
            }}
            title={
              isCkb
                ? 'لاری پیشاندانی شەبەنگ (0 / 3 / 4.5 dB/oct — ستانداردی FabFilter)'
                : 'Spectrum visual tilt around 1 kHz (0 flat / 3 pink / 4.5 dB/oct Pro-Q standard)'
            }
          >
            <span>{tiltSlope === 0 ? (isCkb ? 'تەخت' : 'Flat') : `${tiltSlope} dB/oct`}</span>
          </button>
          <button
            type="button"
            className="spec-hpf-chip"
            aria-pressed={frozen}
            disabled={!original}
            onClick={() => setFrozen((prev) => !prev)}
            title={
              isCkb
                ? 'بەستنی کێرڤی ڕاستەوخۆ بۆ پشکنینی وردی دەنگ'
                : 'Freeze live peak-hold spectrum trace for inspection'
            }
          >
            <span>{isCkb ? 'بەستن' : 'Freeze'}</span>
          </button>
          <button
            type="button"
            className="spec-hpf-chip"
            aria-pressed={soloArmed || Boolean(soloBand)}
            disabled={!original}
            onClick={() => {
              setSoloArmed((prev) => {
                if (prev) stopBandSolo();
                return !prev;
              });
            }}
            title={
              isCkb
                ? 'تاقیکردنەوەی باندی فریکوێنسی بە ڕاکێشان لەسەر شەبەنگ (یان Alt+ڕاکێشان)'
                : 'Band Solo audition: drag anywhere on the spectrum to isolate frequencies in real time (or hold Alt + drag; wheel adjusts Q)'
            }
          >
            <span>{isCkb ? 'سۆلۆ' : 'Solo'}</span>
          </button>
          <button
            type="button"
            className="spec-hpf-chip"
            aria-pressed={hpfEnabled}
            onClick={() => {
              updateRackSettings({ ...rackSettings, hpfEnabled: !hpfEnabled });
              scheduleLivePreview();
            }}
            title="Toggle High-Pass rumble filter — or drag the cyan node on the spectrum"
          >
            <span>HPF {hpfEnabled ? `${hpfHz} Hz · ${hzToMusicalNote(hpfHz)}` : 'Bypass'}</span>
          </button>
        </div>
      </div>
      <div className="spectrum-body">
        <div
          ref={displayRef}
          className="display spectrum-display"
          style={{ position: 'relative', cursor: soloArmed || soloBand ? 'crosshair' : undefined }}
          onPointerDown={onDisplayPointerDown}
          onPointerMove={onPlotPointerMove}
          onPointerUp={onDisplayPointerUp}
          onPointerCancel={onDisplayPointerUp}
          onPointerLeave={() => {
            setHoverInfo(null);
            if (soloPointerIdRef.current === null && soloBand) stopBandSolo();
          }}
          onWheel={onDisplayWheel}
          onDoubleClick={onPlotDoubleClick}
        >
          {/* D1 · a canvas with no role is an unlabelled graphic. The plot
              is not interactive — it is a picture of the two spectra — so it
              is named as one, and the key below it (a real list, in the
              accessibility tree) carries which series are actually present. */}
          <canvas
            ref={canvasRef}
            className="spectrum-canvas"
            role="img"
            aria-label={
              hasRemoved
                ? `Long-term average spectrum: original and cleaned, with the removed energy between them across ${removedPct}% of the band`
                : both
                  ? 'Long-term average spectrum: original and cleaned. Nothing was removed — the cleaned curve does not sit below the original anywhere, so there is no band between them.'
                  : original !== null
                    ? 'Long-term average spectrum of the original clip'
                    : 'Long-term average spectrum: no signal yet'
            }
          />
          {/* Plot-exact inset overlay aligned 1:1 with SpectrumRenderer.plot */}
          <div
            ref={plotOverlayRef}
            className="spec-plot-overlay"
            style={{
              position: 'absolute',
              left: `${insets.left}px`,
              right: `${insets.right}px`,
              top: `${insets.top}px`,
              bottom: `${insets.bottom}px`,
              pointerEvents: 'none',
              zIndex: 2,
            }}
          >
            {/* Active repair band overlays (FabFilter-style frequency zone shading) */}
            {rackSettings.deplosive ? (
              <div
                className="spec-band-overlay band-deplosive"
                aria-hidden="true"
                style={{ left: '0%', width: `${deplosiveRightPct.toFixed(2)}%` }}
              >
                <span>De-Plosive</span>
              </div>
            ) : null}
            {rackSettings.tonalRestoration ? (
              <div
                className="spec-band-overlay band-tonal"
                aria-hidden="true"
                style={{ left: `${tonalLeftPct.toFixed(2)}%`, width: `${tonalWidthPct.toFixed(2)}%` }}
              >
                <span>Tonal</span>
              </div>
            ) : null}
            {rackSettings.deess ? (
              <div
                className="spec-band-overlay band-deess"
                aria-hidden="true"
                style={{ left: `${deessLeftPct.toFixed(2)}%`, width: `${deessWidthPct.toFixed(2)}%` }}
              >
                <span>De-Ess</span>
              </div>
            ) : null}

            {/* Smooth 24 dB/oct Butterworth-style HPF roll-off SVG curve */}
            {hpfEnabled ? (
              <svg
                className="spec-hpf-svg"
                viewBox="0 0 100 100"
                preserveAspectRatio="none"
                aria-hidden="true"
              >
                <defs>
                  <linearGradient id="hawa-hpf-grad" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%" stopColor="rgba(44,212,255,0.22)" />
                    <stop offset={`${hpfLeftPct.toFixed(1)}%`} stopColor="rgba(44,212,255,0.06)" />
                    <stop offset="100%" stopColor="rgba(44,212,255,0)" />
                  </linearGradient>
                </defs>
                <path d={hpfFillPath} fill="url(#hawa-hpf-grad)" />
                <path
                  d={hpfCurvePath}
                  fill="none"
                  stroke="rgba(156,236,255,0.85)"
                  strokeWidth="1.5"
                  vectorEffect="non-scaling-stroke"
                />
              </svg>
            ) : null}
            {soloBand ? (
              <div
                className="spec-solo-band"
                aria-hidden="true"
                style={{
                  position: 'absolute',
                  top: 0,
                  bottom: 0,
                  left: `${soloBand.leftPct.toFixed(2)}%`,
                  width: `${soloBand.widthPct.toFixed(2)}%`,
                  background: 'linear-gradient(180deg, rgba(44,212,255,0.28), rgba(74,222,128,0.12))',
                  borderLeft: '1px solid rgba(156,236,255,0.75)',
                  borderRight: '1px solid rgba(156,236,255,0.75)',
                  boxShadow: '0 0 18px rgba(44,212,255,0.24)',
                  pointerEvents: 'none',
                  zIndex: 3,
                }}
              >
                <span
                  className="mono"
                  dir="ltr"
                  style={{
                    position: 'absolute',
                    top: '4px',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    padding: '1px 6px',
                    borderRadius: '4px',
                    background: 'rgba(5, 8, 14, 0.92)',
                    border: '1px solid rgba(44,212,255,0.55)',
                    color: 'var(--cyan-2)',
                    fontSize: '10px',
                    whiteSpace: 'nowrap',
                  }}
                >
                  SOLO{' '}
                  {soloBand.hz >= 1000
                    ? `${(soloBand.hz / 1000).toFixed(2)} kHz`
                    : `${Math.round(soloBand.hz)} Hz`}{' '}
                  · {hzToMusicalNote(soloBand.hz)} · Q {soloBand.q.toFixed(1)}
                </span>
              </div>
            ) : null}
            {hoverInfo ? (
              <>
                <div
                  className="spec-hover-line"
                  aria-hidden="true"
                  style={{ left: `${hoverInfo.leftPct.toFixed(2)}%` }}
                />
                <div
                  className="spec-hover-pill mono"
                  dir="ltr"
                  aria-hidden="true"
                  style={{
                    left: `${Math.min(82, Math.max(8, hoverInfo.leftPct)).toFixed(2)}%`,
                    top: `${Math.min(78, Math.max(10, hoverInfo.topPct)).toFixed(2)}%`,
                  }}
                >
                  {hoverInfo.hz >= 1000
                    ? `${(hoverInfo.hz / 1000).toFixed(2)} kHz`
                    : `${Math.round(hoverInfo.hz)} Hz`}{' '}
                  · {hoverInfo.note} · {hoverInfo.db.toFixed(0)} dB
                </div>
              </>
            ) : null}
            <button
              type="button"
              aria-label={`High-pass rumble filter node: ${hpfEnabled ? `${hpfHz} Hz (${hzToMusicalNote(hpfHz)})` : 'bypassed'}. Drag horizontally or press arrow keys to adjust.`}
              title={
                hpfEnabled
                  ? `HPF ${hpfHz} Hz (${hzToMusicalNote(hpfHz)}) — drag horizontally or double-click to bypass`
                  : 'HPF bypassed — double-click to enable'
              }
              onPointerDown={onHpfPointerDown}
              onPointerMove={onHpfPointerMove}
              onPointerUp={onHpfPointerUp}
              onPointerCancel={onHpfPointerUp}
              onDoubleClick={() => {
                updateRackSettings({ ...rackSettings, hpfEnabled: !hpfEnabled });
                scheduleLivePreview();
              }}
              onKeyDown={(e) => {
                if (e.key === 'ArrowRight' || e.key === 'ArrowUp') {
                  e.preventDefault();
                  updateRackSettings({
                    ...rackSettings,
                    hpfEnabled: true,
                    hpfCutoffHz: Math.min(HPF_MAX_HZ, hpfHz + 5),
                  });
                  scheduleLivePreview();
                } else if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') {
                  e.preventDefault();
                  updateRackSettings({
                    ...rackSettings,
                    hpfEnabled: true,
                    hpfCutoffHz: Math.max(HPF_MIN_HZ, hpfHz - 5),
                  });
                  scheduleLivePreview();
                }
              }}
              style={{
                position: 'absolute',
                left: `${hpfLeftPct}%`,
                top: '38%',
                transform: 'translate(-50%, -50%)',
                width: '24px',
                height: '24px',
                borderRadius: '50%',
                border: hpfEnabled ? '2px solid var(--cyan-2)' : '1.5px dashed var(--fg-4)',
                background: hpfEnabled ? 'rgba(44, 212, 255, 0.28)' : 'rgba(14, 18, 26, 0.72)',
                boxShadow: hpfEnabled ? '0 0 14px var(--cyan-glow)' : 'none',
                cursor: 'ew-resize',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                pointerEvents: 'auto',
                zIndex: 4,
              }}
            >
              <span
                aria-hidden="true"
                style={{
                  width: '6px',
                  height: '6px',
                  borderRadius: '50%',
                  background: hpfEnabled ? 'var(--cyan-2)' : 'var(--fg-4)',
                }}
              />
            </button>
          </div>
        </div>
        {wide ? (
          <div
            className="spec-piano-ruler"
            aria-hidden="true"
            style={{ marginLeft: `${insets.left}px`, marginRight: `${insets.right}px` }}
          >
            {PIANO_OCTAVES.map((oct) => (
              <span
                key={oct.label}
                className="spec-piano-note"
                style={{ left: `${(hzToNormX(oct.hz) * 100).toFixed(2)}%` }}
              >
                {oct.label}
              </span>
            ))}
          </div>
        ) : null}
        {/*
          ONE key, directly under the picture it explains.

          There used to be two: ORIGINAL / CLEANED / LIVE up in the panel head,
          and — in a different type size, a different alignment and a different
          swatch language — the `LTAS · 1/12 OCT` tag plus a REMOVED colour chip
          floating in the plot's top-right corner. Four marks, explained in two
          places, neither of them complete. They are now a single row: every
          series that can appear in the plot has exactly one entry, in the
          reading order of the picture (the two decks, the band between them,
          the live trace), and the measurement descriptor rides at the far end
          as a quiet meta note rather than as a competing legend.
        */}
        <div className="spec-legend" role="list" aria-label="Spectrum key">
          <LegendItem kind="orig" label={isCkb ? 'سەرەکی' : 'Original'} on={original !== null} hint="long-term average" />
          <LegendItem kind="clean" label={isCkb ? 'پاککراوە' : 'Cleaned'} on={cleaned !== null} hint="long-term average" />
          <LegendItem
            kind="removed"
            label={isCkb ? 'لابراو' : 'Removed'}
            on={hasRemoved}
            hint={`energy taken out, original minus cleaned, across ${removedPct}% of the band`}
            offHint={
              both
                ? 'nothing taken out — the cleaned curve does not sit below the original anywhere'
                : 'not present'
            }
          />
          <LegendItem kind="live" label={isCkb ? 'ڕاستەوخۆ' : 'Live'} on={playing} deck={abMode} hint="playing now" />
          <span className="meta" aria-hidden="true">
            1/12 oct
          </span>
        </div>
      </div>
    </>
  );
}
