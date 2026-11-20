import { useEffect, useRef } from 'react';
import { getPlayer } from '../../audio/player';
import { useReducedMotion } from '../../state/reducedMotion';
import type { AbMode } from '../../state/store';

export interface NeuralHaloCanvasProps {
  playing: boolean;
  deck: AbMode;
  deltaMode: boolean;
  strength: number;
}

/**
 * Ambient + live audio-reactive 60fps acoustic silk wave canvas inspired by
 * Waves Clarity Vx Pro & FabFilter Pro-Q 4. Zero React re-renders per frame;
 * renders a serene static crest under prefers-reduced-motion.
 */
export function NeuralHaloCanvas({
  playing,
  deck,
  deltaMode,
  strength,
}: NeuralHaloCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const reduced = useReducedMotion() === true;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const resize = (): void => {
      const rect = canvas.getBoundingClientRect();
      const w = Math.max(1, Math.round(rect.width * dpr));
      const h = Math.max(1, Math.round(rect.height * dpr));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
    };
    resize();
    const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(resize) : null;
    ro?.observe(canvas);

    const analyser = playing ? getPlayer().getAnalyser() : null;
    const freqBuf = analyser ? new Uint8Array(Math.min(analyser.frequencyBinCount || 64, 128)) : null;

    let raf = 0;
    let phase = 0;

    const draw = (): void => {
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      let energy = 0.14 + strength * 0.12;
      if (analyser && freqBuf && typeof analyser.getByteFrequencyData === 'function') {
        analyser.getByteFrequencyData(freqBuf);
        let sum = 0;
        for (let i = 2; i < Math.min(48, freqBuf.length); i++) sum += freqBuf[i] ?? 0;
        const avg = sum / (Math.min(48, freqBuf.length) * 255);
        energy = Math.min(1, 0.18 + avg * 2.2);
      }

      const rgb = deltaMode
        ? [255, 107, 139]
        : deck === 'cleaned'
          ? [44, 212, 255]
          : [255, 180, 76];

      const cy = h * 0.54;
      const layers = [
        { amp: h * 0.24 * energy, freq: 2.2, speed: 1.0, alpha: 0.26, lw: 2.0 * dpr },
        { amp: h * 0.17 * energy, freq: 3.4, speed: -0.75, alpha: 0.18, lw: 1.4 * dpr },
        { amp: h * 0.11 * energy, freq: 5.1, speed: 1.4, alpha: 0.12, lw: 1.0 * dpr },
      ];

      for (const layer of layers) {
        ctx.beginPath();
        const steps = 64;
        for (let i = 0; i <= steps; i++) {
          const u = i / steps;
          const x = u * w;
          // Smooth Hann window envelope so edges dissolve into obsidian
          const env = Math.sin(u * Math.PI);
          const envSq = env * env;
          const binBoost =
            freqBuf && freqBuf.length > 0
              ? ((freqBuf[Math.min(freqBuf.length - 1, Math.floor(u * 40))] ?? 0) / 255) * 0.45
              : 0;
          const y =
            cy +
            Math.sin(u * Math.PI * 2 * layer.freq + phase * layer.speed) *
              (layer.amp * (1 + binBoost)) *
              envSq;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${layer.alpha})`;
        ctx.lineWidth = layer.lw;
        ctx.stroke();

        // Subtle vertical luminance fill beneath primary ribbon
        if (layer === layers[0]) {
          ctx.lineTo(w, h);
          ctx.lineTo(0, h);
          ctx.closePath();
          const grad = ctx.createLinearGradient(0, cy - layer.amp, 0, h);
          grad.addColorStop(0, `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${(0.11 * energy).toFixed(3)})`);
          grad.addColorStop(1, `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0)`);
          ctx.fillStyle = grad;
          ctx.fill();
        }
      }

      if (!reduced && playing) {
        phase += 0.042;
        raf = requestAnimationFrame(draw);
      }
    };

    draw();
    return () => {
      cancelAnimationFrame(raf);
      ro?.disconnect();
    };
  }, [playing, deck, deltaMode, strength, reduced]);

  return <canvas ref={canvasRef} className="neural-halo-canvas" aria-hidden="true" />;
}
