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

interface LegendItemProps {
  kind: 'orig' | 'clean' | 'live';
  label: string;
  on: boolean;
  deck?: 'original' | 'cleaned';
}

function LegendItem({ kind, label, on, deck }: LegendItemProps) {
  const cls = `item ${kind} ${on ? 'on' : 'off'}${deck ? ` deck-${deck}` : ''}`;
  return (
    <span className={cls} role="listitem" aria-label={`${label} curve ${on ? 'shown' : 'not present'}`}>
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

  return (
    <>
      <div className="panel-head">
        <div className="panel-title">
          <span>Spectrum</span>
        </div>
        <div className="spec-legend" role="list" aria-label="Spectrum curves">
          <LegendItem kind="orig" label="Original" on={original !== null} />
          <LegendItem kind="clean" label="Cleaned" on={cleaned !== null} />
          <LegendItem kind="live" label="Live" on={playing} deck={abMode} />
        </div>
      </div>
      <div className="spectrum-body">
        <div className="display spectrum-display">
          <canvas ref={canvasRef} className="spectrum-canvas" />
        </div>
      </div>
    </>
  );
}
