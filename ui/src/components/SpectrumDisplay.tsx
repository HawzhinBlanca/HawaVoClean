import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react';
import { getPlayer } from '../audio/player';
import { SpectrumRenderer } from '../render/spectrum';
import '../render/spectrum.css';
import { updateRackSettings } from '../state/actions';
import { scheduleLivePreview } from '../state/liveAudition';
import { HPF_MAX_HZ, HPF_MIN_HZ } from '../state/presets';
import { useStore } from '../state/store';

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
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rendererRef = useRef<SpectrumRenderer | null>(null);
  const original = useStore((s) => s.original);
  const cleaned = useStore((s) => s.cleaned);
  const abMode = useStore((s) => s.abMode);
  const playing = useStore((s) => s.playing);
  const rackSettings = useStore((s) => s.rackSettings);
  const [removedCoverage, setRemovedCoverage] = useState(0);
  const [draggingHpf, setDraggingHpf] = useState(false);
  const displayRef = useRef<HTMLDivElement | null>(null);

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

  const both = original !== null && cleaned !== null;
  // The band is lit by the pixels it will have, not by the pair of curves.
  const hasRemoved = both && removedCoverage > 0;
  const removedPct = Math.round(removedCoverage * 100);
  const hpfEnabled = rackSettings.hpfEnabled !== false;
  const hpfHz = rackSettings.hpfCutoffHz;
  const hpfLeftPct = Math.min(48, Math.max(2, hzToNormX(hpfHz) * 100));

  // Smooth FabFilter-style 24 dB/oct HPF roll-off curve in SVG 0..100 coordinates
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

  const updateHpfFromPointer = (clientX: number): void => {
    const rect = displayRef.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0) return;
    const normX = (clientX - rect.left) / rect.width;
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

  return (
    <>
      <div className="panel-head">
        <h2 className="panel-title">
          <span>Spectrum</span>
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
            </div>
          ) : null}
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
          style={{ position: 'relative' }}
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
        {wide ? (
          <div className="spec-piano-ruler" aria-hidden="true">
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
          <LegendItem kind="orig" label="Original" on={original !== null} hint="long-term average" />
          <LegendItem kind="clean" label="Cleaned" on={cleaned !== null} hint="long-term average" />
          <LegendItem
            kind="removed"
            label="Removed"
            on={hasRemoved}
            hint={`energy taken out, original minus cleaned, across ${removedPct}% of the band`}
            offHint={
              both
                ? 'nothing taken out — the cleaned curve does not sit below the original anywhere'
                : 'not present'
            }
          />
          <LegendItem kind="live" label="Live" on={playing} deck={abMode} hint="playing now" />
          <span className="meta" aria-hidden="true">
            1/12 oct
          </span>
        </div>
      </div>
    </>
  );
}
