import { useReducedMotion } from 'motion/react';
import { useEffect, useRef } from 'react';
import { cancelJob, startJob } from '../state/actions';
import { useStore } from '../state/store';
import { IconBolt, IconCheck, IconWarn } from './Icons';

const STAGE_LABEL: Record<string, string> = {
  preflight: 'Preflight',
  decode: 'Decoding',
  segment: 'Segmenting',
  enhance: 'Enhancing',
  guard: 'Guarding',
  finish: 'Finishing',
  publish: 'Publishing',
  done: 'Done',
  error: 'Error',
};

// Ring geometry is in viewBox units; CSS scales the whole SVG responsively.
const R = 36;
const C = 2 * Math.PI * R;

/** Time constant of the arc's chase, in seconds. ~5 frames to half-way. */
const TAU = 0.16;

/**
 * The progress arc, interpolated.
 *
 * SSE hands us progress in stage-sized steps — 0.1, then nothing for eight
 * seconds, then 0.45 — so a CSS transition can only ever chase the last step
 * it was given and then sit still. This drives the arc (and the percentage
 * readout) from a rAF loop that eases toward the target exponentially and
 * writes straight to the DOM, so the ring is smooth between updates and no
 * frame of it costs a React render.
 *
 * Under `prefers-reduced-motion` the loop still runs — the readout has to keep
 * up with the job — but it snaps to the target instead of easing, and the CSS
 * layer has already removed the transition, so nothing on screen moves that
 * the engine did not move.
 */
function useSmoothArc(target: number, live: boolean, snap: boolean) {
  const barRef = useRef<SVGCircleElement | null>(null);
  const pctRef = useRef<HTMLSpanElement | null>(null);
  const shown = useRef(0);
  const goal = useRef(target);
  goal.current = target;

  useEffect(() => {
    if (!live) {
      // Leaving the running state: hand the arc back to React. The inline
      // style the loop was writing has to go, or it would outrank the
      // `strokeDashoffset` attribute for the rest of the session and freeze
      // the ring at whatever the last streamed frame happened to be.
      shown.current = target;
      if (barRef.current) barRef.current.style.strokeDashoffset = '';
      return;
    }
    // A hidden tab gets no animation frames, so easing there would leave the
    // arc frozen at whatever it last drew. Correctness beats smoothness when
    // nobody is looking: snap, and let the loop take over on return.
    if (document.hidden) {
      shown.current = goal.current;
      if (barRef.current) barRef.current.style.strokeDashoffset = String(C * (1 - shown.current));
      if (pctRef.current) pctRef.current.textContent = String(Math.round(shown.current * 100));
    }
    let raf = 0;
    let last = performance.now();
    let lastPct = -1;
    const tick = (now: number): void => {
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      const g = goal.current;
      shown.current = snap ? g : shown.current + (g - shown.current) * (1 - Math.exp(-dt / TAU));
      if (Math.abs(g - shown.current) < 0.0005) shown.current = g;
      const bar = barRef.current;
      if (bar) bar.style.strokeDashoffset = String(C * (1 - shown.current));
      const pct = Math.round(shown.current * 100);
      if (pct !== lastPct && pctRef.current) {
        lastPct = pct;
        pctRef.current.textContent = String(pct);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // `target` is read through a ref inside the loop; it is listed so that a
    // job that ends between frames still leaves `shown` at its final value.
  }, [live, snap, target]);

  return { barRef, pctRef, initial: shown.current };
}

export function ProcessButton() {
  const source = useStore((s) => s.source);
  const original = useStore((s) => s.original);
  const engineReady = useStore((s) => s.engineStatus === 'ready');
  const job = useStore((s) => s.job);
  const analyzing = useStore((s) => s.analyzing);
  const profile = useStore((s) => s.profile);
  const reduced = useReducedMotion() === true;

  const status = job?.status ?? null;
  const state = status?.state ?? (job ? 'queued' : null);
  const running = state === 'running' || state === 'queued';
  const progress = running || state === 'done' ? Math.max(0, Math.min(1, status?.progress ?? 0)) : 0;
  const canStart = engineReady && !!source && !!original && !analyzing && !running;

  let barCls = 'bar';
  if (!running) {
    if (state === 'done') barCls += ' done';
    else if (state === 'failed') barCls += ' err';
    else barCls += ' idle';
  }
  const shown = running ? progress : state === 'done' ? 1 : state === 'failed' ? 1 : 0;
  const arc = useSmoothArc(shown, running, reduced);
  // While the loop is live it owns the attribute; otherwise React does.
  const dashOffset = running ? C * (1 - arc.initial) : C * (1 - shown);

  let title = 'PROCESS';
  const titleCls = 'title';
  let stage = profile === 'studio' ? 'Studio profile' : 'Production profile';
  let units: string | null = null;
  if (running) {
    title = 'CANCEL';
    stage = STAGE_LABEL[status?.stage ?? ''] ?? (status?.stage ?? 'Working');
    if (status?.unit) units = `unit ${status.unit.index} / ${status.unit.total}`;
    else if (status?.message) units = status.message;
  } else if (state === 'done') {
    // The title is a verb — what the key does if you press it again — so it
    // stays neutral; the *state* line carries the state colour, and it agrees
    // with the ring (green for done, red for failed) instead of fighting it.
    title = 'PROCESS AGAIN';
    stage = 'Complete';
    if (status?.report) {
      const s = status.report.summary;
      units = `${s.enhanced ?? 0} / ${s.units_total ?? 0} units enhanced`;
    }
  } else if (state === 'failed') {
    title = 'RETRY';
    stage = 'Failed';
    units = status?.error?.code ?? status?.message ?? null;
  } else if (state === 'cancelled') {
    title = 'PROCESS';
    stage = 'Cancelled';
  } else if (!source) {
    stage = 'Load a clip to begin';
  } else if (analyzing) {
    stage = 'Analyzing clip';
  }

  const onClick = (): void => {
    if (running) void cancelJob();
    else if (canStart) void startJob();
  };

  let phaseCls = '';
  if (running) phaseCls = ' running';
  else if (state === 'done') phaseCls = ' done';
  else if (state === 'failed') phaseCls = ' failed';

  return (
    <button
      type="button"
      className={`process${phaseCls}`}
      disabled={!running && !canStart}
      onClick={onClick}
      aria-label={title}
      aria-busy={running}
    >
      <span className="ring" aria-hidden="true">
        <svg viewBox="0 0 84 84" style={{ width: '100%', height: '100%' }}>
          <circle className="track" cx="42" cy="42" r={R} fill="none" strokeWidth="6" />
          {/* the lit lower lip of the channel, then the outer collar */}
          <circle className="track-lip" cx="42" cy="42" r={R - 3.2} fill="none" strokeWidth="1" />
          <circle className="track-hi" cx="42" cy="42" r={R + 3.5} fill="none" strokeWidth="1" />
          <circle
            ref={arc.barRef}
            className={barCls}
            cx="42"
            cy="42"
            r={R}
            fill="none"
            strokeWidth="5"
            strokeLinecap="round"
            strokeDasharray={C}
            strokeDashoffset={dashOffset}
          />
        </svg>
        {/*
          All four glyphs stay mounted, stacked on the ring's centre, and CSS
          picks the one the phase calls for. Cross-fading by opacity beats
          swapping the element: there is no blank frame between idle and
          running, the percentage node the rAF loop writes to never has to be
          re-acquired, and the whole transition survives `animation: none`
          under reduced motion (it degrades to an instant, legible swap).
        */}
        <span className="center">
          <span className="glyph g-idle">
            <IconBolt className="icon" style={{ color: 'var(--amber)' }} />
          </span>
          <span className="glyph g-run">
            <span className="pct">
              <span ref={arc.pctRef}>{Math.round(arc.initial * 100)}</span>
              <small>%</small>
            </span>
          </span>
          <span className="glyph g-done">
            <IconCheck className="icon" style={{ color: 'var(--ok)' }} />
          </span>
          <span className="glyph g-fail">
            <IconWarn className="icon" style={{ color: 'var(--err)' }} />
          </span>
        </span>
      </span>
      <span className="text">
        <span className={titleCls}>{title}</span>
        <span className="stage">{stage}</span>
        <span className="units">{units ?? <span className="dim">{running ? 'starting' : profile === 'studio' ? 'WPE + DeepFilterNet3' : 'Wiener decision-directed'}</span>}</span>
      </span>
    </button>
  );
}
