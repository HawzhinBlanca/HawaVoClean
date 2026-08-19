import { useEffect, useRef } from 'react';
import { getPlayer } from '../audio/player';
import { SpectrumRenderer } from '../render/spectrum';
import { useStore } from '../state/store';

function toF32(a: number[]): Float32Array {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] ?? 0;
  return out;
}

export function SpectrumDisplay() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rendererRef = useRef<SpectrumRenderer | null>(null);
  const original = useStore((s) => s.original);
  const cleaned = useStore((s) => s.cleaned);
  const abMode = useStore((s) => s.abMode);
  const playing = useStore((s) => s.playing);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const r = new SpectrumRenderer(canvas);
    rendererRef.current = r;
    return () => {
      r.dispose();
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
        <div className="spectrum-legend" aria-hidden="true">
          <span className={`orig${!original ? ' off' : ''}`}>
            <i /> Original
          </span>
          <span className={`clean${!cleaned ? ' off' : ''}`}>
            <i /> Cleaned
          </span>
          <span className={`live${!playing ? ' off' : ''}`}>
            <i /> Live
          </span>
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
