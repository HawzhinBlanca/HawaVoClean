import {
  useEffect,
  useId,
  useRef,
  useState,
  type ChangeEvent,
  type PointerEvent as ReactPointerEvent,
  type WheelEvent as ReactWheelEvent,
} from 'react';
import { getPlayer } from '../../audio/player';
import { useReducedMotion } from '../../state/reducedMotion';

export type KnobAccent = 'cyan' | 'amber' | 'coral' | 'emerald';
export type KnobSize = 'hero' | 'md' | 'sm';

export interface RotaryKnobProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  defaultValue?: number;
  onChange: (next: number) => void;
  formatValue?: (v: number) => string;
  subLabel?: string;
  hint?: string;
  accent?: KnobAccent;
  size?: KnobSize;
  disabled?: boolean;
  audioReactive?: boolean;
  playing?: boolean;
  ariaLabel?: string;
  readoutTestId?: string;
  inputId?: string;
}

const SIZE_PX: Record<KnobSize, number> = {
  hero: 122,
  md: 70,
  sm: 52,
};

const SWEEP_DEG = 270;
const START_DEG = -135;
const TICK_COUNT = 29;

function clamp(v: number, lo: number, hi: number): number {
  if (!Number.isFinite(v)) return lo;
  return Math.min(hi, Math.max(lo, v));
}

function quantize(v: number, min: number, max: number, step: number): number {
  const c = clamp(v, min, max);
  if (!(step > 0)) return c;
  const snapped = min + Math.round((c - min) / step) * step;
  const decimals = step < 1 ? Math.min(4, String(step).split('.')[1]?.length ?? 2) : 0;
  return clamp(Number(snapped.toFixed(decimals)), min, max);
}

function polarToCartesian(cx: number, cy: number, r: number, degFromTop: number): { x: number; y: number } {
  const rad = ((degFromTop - 90) * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}

function describeArc(cx: number, cy: number, r: number, startDeg: number, endDeg: number): string {
  const s = polarToCartesian(cx, cy, r, startDeg);
  const e = polarToCartesian(cx, cy, r, endDeg);
  const largeArc = Math.abs(endDeg - startDeg) > 180 ? 1 : 0;
  return `M ${s.x.toFixed(2)} ${s.y.toFixed(2)} A ${r.toFixed(2)} ${r.toFixed(2)} 0 ${largeArc} 1 ${e.x.toFixed(2)} ${e.y.toFixed(2)}`;
}

const PRECOMPUTED_TICKS = Array.from({ length: TICK_COUNT }, (_, i) => {
  const frac = i / (TICK_COUNT - 1);
  const deg = START_DEG + frac * SWEEP_DEG;
  const isMajor = i % 7 === 0;
  return { i, frac, deg, isMajor };
});

export function RotaryKnob({
  label,
  value,
  min,
  max,
  step = 1,
  defaultValue,
  onChange,
  formatValue,
  subLabel,
  hint,
  accent = 'cyan',
  size = 'md',
  disabled = false,
  audioReactive = false,
  playing = false,
  ariaLabel,
  readoutTestId,
  inputId,
}: RotaryKnobProps) {
  const autoId = useId();
  const id = inputId ?? autoId;
  const safeUid = autoId.replace(/[^a-zA-Z0-9_-]/g, '');
  const capGradId = `hawa-kcap-${safeUid}`;
  const rimGradId = `hawa-krim-${safeUid}`;
  const reduced = useReducedMotion() === true;

  const [dragging, setDragging] = useState(false);
  const [fineMode, setFineMode] = useState(false);
  const dragRef = useRef<{ pointerId: number; startY: number; startVal: number } | null>(null);
  const reactiveRef = useRef<SVGCircleElement | null>(null);

  const safeVal = clamp(value, min, max);
  const ratio = max > min ? (safeVal - min) / (max - min) : 0;
  const px = SIZE_PX[size];
  const vb = 120;
  const cx = 60;
  const cy = 60;
  const tickR0 = 51.5;
  const tickR1Major = 56.5;
  const tickR1Minor = 54.5;
  const trackR = 44;
  const capR = size === 'hero' ? 35 : 33;
  const strokeW = size === 'hero' ? 5.2 : 5.6;

  const trackPath = describeArc(cx, cy, trackR, START_DEG, START_DEG + SWEEP_DEG);
  const arcLen = (2 * Math.PI * trackR * SWEEP_DEG) / 360;
  const dashOffset = arcLen * (1 - ratio);
  const angleDeg = START_DEG + ratio * SWEEP_DEG;
  const tipPos = polarToCartesian(cx, cy, trackR, angleDeg);
  const notchInner = polarToCartesian(cx, cy, capR - 8, angleDeg);
  const notchOuter = polarToCartesian(cx, cy, capR - 2.5, angleDeg);

  const displayText = formatValue ? formatValue(safeVal) : String(safeVal);

  // Live audio-reactive luminous halo driven directly on DOM (zero React re-renders per frame).
  useEffect(() => {
    const ring = reactiveRef.current;
    if (!audioReactive || !playing || reduced || !ring) {
      if (ring) {
        ring.style.opacity = '0';
        ring.style.transform = 'scale(1)';
      }
      return;
    }
    const analyser = getPlayer().getAnalyser();
    if (!analyser) return;
    const buf = new Float32Array(Math.min(analyser.fftSize || 1024, 2048));
    let raf = 0;
    let smoothRms = 0;
    const tick = (): void => {
      if (typeof analyser.getFloatTimeDomainData === 'function') {
        analyser.getFloatTimeDomainData(buf);
        let sum = 0;
        for (let i = 0; i < buf.length; i++) {
          const s = buf[i] ?? 0;
          sum += s * s;
        }
        const rms = Math.sqrt(sum / Math.max(1, buf.length));
        const level = Math.min(1, rms * 4.2);
        smoothRms = smoothRms * 0.75 + level * 0.25;
        ring.style.opacity = (0.18 + smoothRms * 0.62).toFixed(3);
        ring.style.transform = `scale(${(1 + smoothRms * 0.085).toFixed(3)})`;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      ring.style.opacity = '0';
      ring.style.transform = 'scale(1)';
    };
  }, [audioReactive, playing, reduced]);

  const handlePointerDown = (e: ReactPointerEvent<HTMLDivElement>): void => {
    if (disabled || e.button !== 0) return;
    e.preventDefault();
    e.currentTarget.setPointerCapture?.(e.pointerId);
    dragRef.current = { pointerId: e.pointerId, startY: e.clientY, startVal: safeVal };
    setDragging(true);
    setFineMode(e.shiftKey);
  };

  const handlePointerMove = (e: ReactPointerEvent<HTMLDivElement>): void => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== e.pointerId || disabled) return;
    setFineMode(e.shiftKey);
    const dy = drag.startY - e.clientY;
    const sensitivityPx = e.shiftKey ? 1200 : 160;
    const next = quantize(drag.startVal + (dy / sensitivityPx) * (max - min), min, max, step);
    if (next !== safeVal) onChange(next);
  };

  const finishDrag = (e: ReactPointerEvent<HTMLDivElement>): void => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== e.pointerId) return;
    dragRef.current = null;
    setDragging(false);
    setFineMode(false);
    try {
      e.currentTarget.releasePointerCapture?.(e.pointerId);
    } catch {
      // ignore if already released
    }
  };

  const handleWheel = (e: ReactWheelEvent<HTMLDivElement>): void => {
    if (disabled) return;
    e.preventDefault();
    const dir = e.deltaY < 0 ? 1 : -1;
    const mult = e.shiftKey ? 1 : Math.max(1, Math.round((max - min) / (step * 25)));
    const next = quantize(safeVal + dir * step * mult, min, max, step);
    if (next !== safeVal) onChange(next);
  };

  const handleDoubleClick = (): void => {
    if (disabled || defaultValue === undefined) return;
    const reset = quantize(defaultValue, min, max, step);
    if (reset !== safeVal) onChange(reset);
  };

  const handleInputChange = (e: ChangeEvent<HTMLInputElement>): void => {
    if (disabled) return;
    const parsed = Number(e.target.value);
    if (!Number.isFinite(parsed)) return;
    onChange(quantize(parsed, min, max, step));
  };

  const accentStroke =
    accent === 'amber'
      ? 'var(--amber-2)'
      : accent === 'coral'
        ? 'var(--coral-2)'
        : accent === 'emerald'
          ? 'var(--ok)'
          : 'var(--cyan-2)';

  const accentCore =
    accent === 'amber'
      ? 'var(--amber)'
      : accent === 'coral'
        ? 'var(--coral)'
        : accent === 'emerald'
          ? 'var(--ok)'
          : 'var(--cyan)';

  const showTicks = size !== 'sm';

  return (
    <div
      className="rotary-knob"
      data-size={size}
      data-disabled={disabled ? 'true' : 'false'}
    >
      {dragging ? (
        <div className="rk-drag-callout" aria-hidden="true">
          <span>{displayText}</span>
          {fineMode ? <span className="rk-fine-tag">FINE</span> : null}
        </div>
      ) : null}

      <div
        className="rk-dial-wrap"
        data-dragging={dragging ? 'true' : 'false'}
        style={{ width: `${px}px`, height: `${px}px` }}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={finishDrag}
        onPointerCancel={finishDrag}
        onWheel={handleWheel}
        onDoubleClick={handleDoubleClick}
        title={
          defaultValue !== undefined
            ? `${label}: ${displayText} (drag up/down · Shift for fine · double-click to reset)`
            : `${label}: ${displayText} (drag up/down · Shift for fine)`
        }
      >
        <svg
          className="rk-svg"
          width={px}
          height={px}
          viewBox={`0 0 ${vb} ${vb}`}
          aria-hidden="true"
        >
          <defs>
            <radialGradient id={capGradId} cx="46%" cy="28%" r="74%">
              <stop offset="0%" stopColor="#2d394e" />
              <stop offset="48%" stopColor="#161d2b" />
              <stop offset="100%" stopColor="#090d14" />
            </radialGradient>
            <linearGradient id={rimGradId} x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="rgba(255,255,255,0.28)" />
              <stop offset="50%" stopColor="rgba(255,255,255,0.06)" />
              <stop offset="100%" stopColor="rgba(0,0,0,0.75)" />
            </linearGradient>
          </defs>

          {/* Subtle ambient radial energy bloom proportional to knob level */}
          <circle
            cx={cx}
            cy={cy}
            r={trackR + 3}
            fill="none"
            stroke={accentCore}
            strokeWidth={size === 'hero' ? 10 : 6}
            opacity={(0.04 + ratio * 0.12).toFixed(3)}
            style={{ filter: 'blur(5px)' }}
          />

          {/* Live audio-reactive outer activity halo */}
          {audioReactive ? (
            <circle
              ref={reactiveRef}
              className="rk-reactive-ring"
              cx={cx}
              cy={cy}
              r={57.5}
              strokeWidth={2}
            />
          ) : null}

          {/* 29 machined precision bezel tick marks */}
          {showTicks
            ? PRECOMPUTED_TICKS.map((t) => {
                const p0 = polarToCartesian(cx, cy, tickR0, t.deg);
                const p1 = polarToCartesian(cx, cy, t.isMajor ? tickR1Major : tickR1Minor, t.deg);
                const lit = t.frac <= ratio + 0.005;
                return (
                  <line
                    key={t.i}
                    x1={p0.x.toFixed(1)}
                    y1={p0.y.toFixed(1)}
                    x2={p1.x.toFixed(1)}
                    y2={p1.y.toFixed(1)}
                    stroke={lit ? accentCore : 'rgba(255,255,255,0.14)'}
                    strokeWidth={t.isMajor ? 1.6 : 1}
                    strokeLinecap="round"
                    opacity={lit ? (t.isMajor ? 0.95 : 0.7) : 0.55}
                  />
                );
              })
            : null}

          {/* Recessed arc track + luminous active arc */}
          <path className="rk-track" d={trackPath} strokeWidth={strokeW} />
          <path
            className={`rk-arc accent-${accent}`}
            d={trackPath}
            strokeWidth={strokeW}
            strokeDasharray={arcLen.toFixed(2)}
            strokeDashoffset={dashOffset.toFixed(2)}
          />

          {/* Machined bevel outer ring + concave tactile cap */}
          <circle cx={cx} cy={cy} r={capR + 1.5} fill="#06080d" />
          <circle
            className="rk-cap"
            cx={cx}
            cy={cy}
            r={capR}
            style={{ fill: `url(#${capGradId})`, stroke: `url(#${rimGradId})`, strokeWidth: 1.6 }}
          />

          {/* Subtle radial notch on cap rim + luminous bead on arc tip */}
          {size === 'hero' ? (
            <line
              x1={notchInner.x.toFixed(1)}
              y1={notchInner.y.toFixed(1)}
              x2={notchOuter.x.toFixed(1)}
              y2={notchOuter.y.toFixed(1)}
              stroke={accentStroke}
              strokeWidth={2.4}
              strokeLinecap="round"
            />
          ) : null}
          <circle
            cx={tipPos.x.toFixed(2)}
            cy={tipPos.y.toFixed(2)}
            r={size === 'hero' ? 4.4 : 3.7}
            fill={accentStroke}
            style={{ filter: `drop-shadow(0 0 4px ${accentCore})` }}
          />
        </svg>

        <div className="rk-center-val">
          <span className="rk-val-num" data-testid={readoutTestId}>
            {displayText}
          </span>
          {subLabel ? <span className="rk-val-sub">{subLabel}</span> : null}
        </div>

        <input
          id={id}
          type="range"
          className="rk-sr-input"
          min={min}
          max={max}
          step={step}
          value={safeVal}
          disabled={disabled}
          aria-label={ariaLabel ?? label}
          aria-valuetext={displayText}
          onChange={handleInputChange}
        />
      </div>

      <label htmlFor={id} className="rk-label">
        {label}
      </label>
      {hint ? <span className="rk-hint">{hint}</span> : null}
    </div>
  );
}
