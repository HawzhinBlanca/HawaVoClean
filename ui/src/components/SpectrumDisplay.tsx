import { useEffect, useRef } from 'react';
import { getPlayer } from '../audio/player';
import { SpectrumRenderer } from '../render/spectrum';
import '../render/spectrum.css';
import { useStore } from '../state/store';

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
}

function LegendItem({ kind, label, on, deck, hint }: LegendItemProps) {
  const cls = `item ${kind} ${on ? 'on' : 'off'}${deck ? ` deck-${deck}` : ''}`;
  return (
    <span className={cls} role="listitem" aria-label={`${label}: ${on ? hint : 'not present'}`}>
      <i className="sw" aria-hidden="true" />
      <span className="txt">{label}</span>
    </span>
  );
}

export function SpectrumDisplay() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rendererRef = useRef<SpectrumRenderer | null>(null);
  const original = useStore((s) => s.original);
  const cleaned = useStore((s) => s.cleaned);
  const abMode = useStore((s) => s.abMode);
  const playing = useStore((s) => s.playing);

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

  return (
    <>
      <div className="panel-head">
        <div className="panel-title">
          <span>Spectrum</span>
          {/* the same `title · qualifier` head the waveform panel uses: what
              the panel is showing, said once, where the panel is named */}
          <span className="sub">· LTAS</span>
        </div>
      </div>
      <div className="spectrum-body">
        <div className="display spectrum-display">
          {/* D1 · a canvas with no role is an unlabelled graphic. The plot
              is not interactive — it is a picture of the two spectra — so it
              is named as one, and the key below it (a real list, in the
              accessibility tree) carries which series are actually present. */}
          <canvas
            ref={canvasRef}
            className="spectrum-canvas"
            role="img"
            aria-label={
              both
                ? 'Long-term average spectrum: original and cleaned, with the removed energy between them'
                : original !== null
                  ? 'Long-term average spectrum of the original clip'
                  : 'Long-term average spectrum: no signal yet'
            }
          />
        </div>
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
            on={both}
            hint="energy taken out, original minus cleaned"
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
