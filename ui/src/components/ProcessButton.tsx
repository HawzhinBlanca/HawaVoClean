
import { useEffect, useRef } from 'react';
import type { JobStage, JobStatus } from '../api/types';
import { cancelJob, startJob } from '../state/actions';
import { useReducedMotion } from '../state/reducedMotion';
import { useStore } from '../state/store';
import { IconBolt, IconCheck, IconWarn } from './Icons';
import '../styles/process.css';

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

/** The pipeline, in the order the engine walks it. The rail under the readout
 *  is a coarse, discrete reading of the same journey the arc measures finely —
 *  a meter and a stage lamp, which is how a real processor tells you both
 *  *how far* and *where*. */
const STAGE_ORDER: readonly JobStage[] = [
  'preflight',
  'decode',
  'segment',
  'enhance',
  'guard',
  'finish',
  'publish',
];

// Ring geometry is in viewBox units; CSS scales the whole SVG responsively.
// The viewBox is larger than the arc so the tick scale has somewhere to live
// outside the channel without being clipped.
const VB = 92;
const CX = VB / 2;
const R = 34;
const C = 2 * Math.PI * R;

/** 48 ticks around the bezel, every sixth one long. Built once, as two paths. */
const TICKS = ((): { minor: string; major: string } => {
  let minor = '';
  let major = '';
  for (let i = 0; i < 48; i++) {
    const a = (i * Math.PI * 2) / 48;
    const isMaj = i % 6 === 0;
    const r0 = 39.6;
    const r1 = isMaj ? 44.2 : 42.2;
    const c = Math.cos(a);
    const s = Math.sin(a);
    const seg =
      `M${(CX + c * r0).toFixed(2)} ${(CX + s * r0).toFixed(2)}` +
      `L${(CX + c * r1).toFixed(2)} ${(CX + s * r1).toFixed(2)}`;
    if (isMaj) major += seg;
    else minor += seg;
  }
  return { minor, major };
})();

/** Time constant of the arc's chase, in seconds. ~5 frames to half-way. */
const TAU = 0.16;

function pad2(n: number): string {
  return n < 10 ? `0${n}` : `${n}`;
}

/** m:ss.d — the same shape as the transport clock, always the same width. */
function fmtElapsed(sec: number): string {
  if (!Number.isFinite(sec) || sec < 0) return '—';
  const m = Math.floor(sec / 60);
  const s = sec - m * 60;
  return `${m}:${pad2(Math.floor(s))}.${Math.floor((s % 1) * 10)}`;
}

/** Engine exit codes, as a phrase that fits a readout cell. The raw code and
 *  the engine's own message stay reachable on the cell's tooltip and in the
 *  error bar; the plate says what happened in two words. */
const FAIL_REASON: Record<string, string> = {
  PREFLIGHT_FAILURE: 'Preflight',
  PUBLICATION_FAILURE: 'Publish',
  INVALID_USER_INPUT: 'Bad input',
  INTERNAL: 'Internal',
  SPAWN_FAILED: 'No worker',
  // Not an engine exit code: the verdict the UI reaches on its own when the
  // engine that owned a run went away with it (goal box B6).
  ENGINE_RESTARTED: 'Engine gone',
};

interface Cell {
  k: string;
  v: string;
  /** values that are quantities get the tabular mono face; words do not */
  num?: boolean;
  /** the long form, on the cell's tooltip */
  title?: string;
}

/**
 * The instrument face.
 *
 * SSE hands us progress in stage-sized steps — 0.1, then nothing for eight
 * seconds, then 0.45 — so a CSS transition can only ever chase the last step
 * it was given and then sit still. This drives the arc, the head of the arc,
 * the percentage and the elapsed clock from one rAF loop that eases toward the
 * target exponentially and writes straight to the DOM, so the ring sweeps
 * smoothly between updates and no frame of it costs a React render.
 *
 * Under `prefers-reduced-motion` the loop still runs — the readout has to keep
 * up with the job — but the arc snaps to the target instead of easing, and the
 * CSS layer has already removed every transition, so nothing on screen moves
 * that the engine did not move.
 */
function useFaceLoop(target: number, live: boolean, snap: boolean, startedMs: number | null) {
  const barRef = useRef<SVGCircleElement | null>(null);
  const headRef = useRef<SVGGElement | null>(null);
  const pctRef = useRef<HTMLSpanElement | null>(null);
  const clockRef = useRef<HTMLElement | null>(null);
  const shown = useRef(0);
  const goal = useRef(target);
  const start = useRef(startedMs);
  goal.current = target;
  start.current = startedMs;

  useEffect(() => {
    const paint = (v: number): void => {
      const bar = barRef.current;
      if (bar) bar.style.strokeDashoffset = String(C * (1 - v));
      const head = headRef.current;
      if (head) head.setAttribute('transform', `rotate(${(v * 360).toFixed(2)} ${CX} ${CX})`);
    };
    if (!live) {
      // Leaving the running state: hand the arc back to React. The inline
      // style the loop was writing has to go, or it would outrank the
      // `strokeDashoffset` attribute for the rest of the session and freeze
      // the ring at whatever the last streamed frame happened to be.
      shown.current = target;
      if (barRef.current) barRef.current.style.strokeDashoffset = '';
      if (headRef.current) {
        headRef.current.setAttribute('transform', `rotate(${(target * 360).toFixed(2)} ${CX} ${CX})`);
      }
      return;
    }
    // A hidden tab gets no animation frames, so easing there would leave the
    // arc frozen at whatever it last drew. Correctness beats smoothness when
    // nobody is looking: snap, and let the loop take over on return.
    if (document.hidden) {
      shown.current = goal.current;
      paint(shown.current);
      if (pctRef.current) pctRef.current.textContent = String(Math.round(shown.current * 100));
    }
    let raf = 0;
    let last = performance.now();
    let lastPct = -1;
    let lastClock = '';
    const tick = (now: number): void => {
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      const g = goal.current;
      shown.current = snap ? g : shown.current + (g - shown.current) * (1 - Math.exp(-dt / TAU));
      if (Math.abs(g - shown.current) < 0.0005) shown.current = g;
      paint(shown.current);
      const pct = Math.round(shown.current * 100);
      if (pct !== lastPct && pctRef.current) {
        lastPct = pct;
        pctRef.current.textContent = String(pct);
      }
      const s = start.current;
      if (s !== null && clockRef.current) {
        const txt = fmtElapsed((Date.now() - s) / 1000);
        if (txt !== lastClock) {
          lastClock = txt;
          clockRef.current.textContent = txt;
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // `target` is read through a ref inside the loop; it is listed so that a
    // job that ends between frames still leaves `shown` at its final value.
  }, [live, snap, target]);

  return { barRef, headRef, pctRef, clockRef, initial: shown.current };
}

function stageIndex(status: JobStatus | null): number {
  const st = status?.stage;
  if (!st) return -1;
  if (st === 'done') return STAGE_ORDER.length - 1;
  if (st === 'error') return -1;
  return STAGE_ORDER.indexOf(st);
}

export function ProcessButton() {
  const source = useStore((s) => s.source);
  const original = useStore((s) => s.original);
  const engineReady = useStore((s) => s.engineStatus === 'ready');
  const job = useStore((s) => s.job);
  const analyzing = useStore((s) => s.analyzing);
  const reduced = useReducedMotion() === true;

  const status = job?.status ?? null;
  const state = status?.state ?? (job ? 'queued' : null);
  const running = state === 'running' || state === 'queued';
  const progress = running || state === 'done' ? Math.max(0, Math.min(1, status?.progress ?? 0)) : 0;
  const canStart = engineReady && !!source && !!original && !analyzing && !running;

  // Elapsed time. The engine's own timestamps win when they parse — they are
  // the job's real wall clock, and they survive a page reload mid-run — with a
  // locally captured start as the fallback.
  const localStart = useRef<number | null>(null);
  const lastJobId = useRef<string | null>(null);
  // A new job id retires the previous run's fallback clock; carrying it over
  // would have made a second run's elapsed time start at the first run's start.
  if ((job?.id ?? null) !== lastJobId.current) {
    lastJobId.current = job?.id ?? null;
    localStart.current = running ? Date.now() : null;
  } else if (running && localStart.current === null) {
    localStart.current = Date.now();
  }
  const serverStart = status?.started_at ? Date.parse(status.started_at) : NaN;
  const startedMs = Number.isFinite(serverStart) ? serverStart : localStart.current;
  const serverEnd = status?.finished_at ? Date.parse(status.finished_at) : NaN;
  const finalSec =
    Number.isFinite(serverEnd) && startedMs !== null ? (serverEnd - startedMs) / 1000 : NaN;

  let barCls = 'bar';
  if (!running) {
    if (state === 'done') barCls += ' done';
    else if (state === 'failed') barCls += ' err';
    else barCls += ' idle';
  }
  const shown = running ? progress : state === 'done' ? 1 : state === 'failed' ? 1 : 0;
  const face = useFaceLoop(shown, running, reduced, startedMs);
  // While the loop is live it owns the attribute; otherwise React does.
  const dashOffset = running ? C * (1 - face.initial) : C * (1 - shown);
  const headDeg = (running ? face.initial : shown) * 360;

  let title = 'PROCESS';
  let cells: Cell[];
  const stageName = STAGE_LABEL[status?.stage ?? ''] ?? (status?.stage ?? 'Working');

  if (running) {
    title = 'CANCEL';
    cells = [
      { k: 'Stage', v: stageName },
      {
        k: 'Unit',
        v: status?.unit ? `${status.unit.index} / ${status.unit.total}` : '—',
        num: true,
      },
      { k: 'Elapsed', v: '0:00.0', num: true },
    ];
  } else if (state === 'done') {
    // The title is a verb — what the key does if you press it again — so it
    // stays neutral; the *state* line carries the state colour, and it agrees
    // with the ring (green for done, red for failed) instead of fighting it.
    title = 'PROCESS AGAIN';
    const s = status?.report?.summary;
    cells = [
      { k: 'Result', v: 'Complete' },
      { k: 'Units', v: s ? `${s.enhanced ?? 0} / ${s.units_total ?? 0}` : '—', num: true },
      { k: 'Took', v: fmtElapsed(finalSec), num: true },
    ];
  } else if (state === 'failed') {
    title = 'RETRY';
    const code = status?.error?.code ?? '';
    cells = [
      { k: 'Result', v: 'Failed' },
      {
        k: 'Reason',
        v: FAIL_REASON[code] ?? (code || status?.message || 'unknown'),
        title: status?.error?.message ?? code ?? undefined,
      },
      { k: 'Took', v: fmtElapsed(finalSec), num: true },
    ];
  } else if (state === 'cancelled') {
    title = 'PROCESS';
    cells = [
      { k: 'Result', v: 'Cancelled' },
      { k: 'Stopped', v: `${Math.round((status?.progress ?? 0) * 100)}%`, num: true },
      { k: 'Took', v: fmtElapsed(finalSec), num: true },
    ];
  } else {
    // Idle is a designed state too: the plate reads back the clip it is armed
    // over, not a blank face waiting for something to happen. It deliberately
    // does *not* repeat the profile and the chain — the profile control sits
    // directly above it and already says both.
    const sr = original?.sample_rate ?? 0;
    const khz = sr ? `${sr % 1000 ? (sr / 1000).toFixed(1) : String(sr / 1000)}k` : '—';
    cells = [
      {
        k: 'Ready',
        v: !engineReady
          ? 'Offline'
          : !source
            ? 'No clip'
            : analyzing
              ? 'Analyzing'
              : // a clip that failed to analyse leaves `source` set and
                // `original` null; the plate must not read "Armed" while it
                // is disabled with nothing to process
                !original
                ? 'No data'
                : 'Armed',
      },
      { k: 'Length', v: original ? fmtElapsed(original.duration_s) : '—', num: true },
      { k: 'Format', v: original ? `${khz}\u00b7${original.channels}ch` : '—', num: true },
    ];
  }

  // B6 · with the engine gone the plate is a readout, not a control: starting
  // is impossible and cancelling is meaningless (the run's fate is already out
  // of our hands, and the reconnect reconciles it). It says so on hover rather
  // than accepting a press and failing.
  const engineDown = !engineReady;
  const inert = engineDown || (!running && !canStart);
  const hint = engineDown
    ? running
      ? 'The engine is offline — this run cannot be cancelled from here. The moment it answers again, the run’s real outcome is read back from the engine.'
      : 'The engine is offline — processing resumes automatically when it reconnects.'
    : running
      ? 'Cancel this run (Esc)'
      : canStart
        ? 'Process this clip (P)'
        : !source
          ? 'Load a clip first — drop one on the strip above or press Open file'
          : analyzing
            ? 'Analysis is still running'
            : 'This clip has no analysis to process';

  const onClick = (): void => {
    if (inert) return;
    if (running) void cancelJob();
    else if (canStart) void startJob();
  };

  let phaseCls = '';
  if (running) phaseCls = ' running';
  else if (state === 'done') phaseCls = ' done';
  else if (state === 'failed') phaseCls = ' failed';
  else if (state === 'cancelled') phaseCls = ' cancelled';

  const si = stageIndex(status);
  const railCls =
    state === 'done' ? 'rail done' : state === 'failed' ? 'rail failed' : running ? 'rail run' : 'rail';

  // The button's accessible name is the *action* (what pressing it does). The
  // readout is the *state*, and an aria-label on a button suppresses its own
  // text, so the face is wired up as the description instead — otherwise the
  // stage, the unit counter and the clock would never reach a screen reader.
  return (
    <button
      type="button"
      className={`process plate${phaseCls}`}
      aria-disabled={inert || undefined}
      title={hint}
      onClick={onClick}
      aria-label={title}
      aria-busy={running}
      aria-describedby="process-readout"
    >
      <span className="ring" aria-hidden="true">
        <svg viewBox={`0 0 ${VB} ${VB}`} style={{ width: '100%', height: '100%' }}>
          <defs>
            {/* The arc is shaded along its own sweep rather than painted one
                flat colour — that, plus the glow riding its head, is what
                separates a meter from a web progress spinner. */}
            <linearGradient id="pb-arc" gradientUnits="userSpaceOnUse" x1="6" y1={VB - 6} x2={VB - 6} y2="6">
              <stop className="arc-a" offset="0" />
              <stop className="arc-b" offset="0.55" />
              <stop className="arc-c" offset="1" />
            </linearGradient>
            {/* A groove is lit from the top like everything else on this
                chassis: shadow spilling in over its upper wall, a thin lit
                lip along the lower one. */}
            <linearGradient id="pb-groove" gradientUnits="userSpaceOnUse" x1="0" y1="8" x2="0" y2={VB - 8}>
              <stop offset="0" stopColor="rgba(0,0,0,0.92)" />
              <stop offset="0.55" stopColor="rgba(0,0,0,0.28)" />
              <stop offset="1" stopColor="rgba(255,255,255,0.13)" />
            </linearGradient>
            <radialGradient id="pb-head">
              <stop className="head-a" offset="0" />
              <stop className="head-b" offset="1" />
            </radialGradient>
          </defs>

          {/* the channel: an opaque bed, then the lighting laid into it */}
          <circle className="chan" cx={CX} cy={CX} r={R} fill="none" strokeWidth="7" />
          <circle
            className="chan-lit"
            cx={CX}
            cy={CX}
            r={R}
            fill="none"
            strokeWidth="7"
            stroke="url(#pb-groove)"
          />
          <circle className="lip-in" cx={CX} cy={CX} r={R - 3.6} fill="none" strokeWidth="0.8" />
          <circle className="lip-out" cx={CX} cy={CX} r={R + 3.6} fill="none" strokeWidth="0.8" />

          {/* bezel scale */}
          <path className="tick" d={TICKS.minor} strokeWidth="0.9" />
          <path className="tick maj" d={TICKS.major} strokeWidth="1.5" />

          <circle
            ref={face.barRef}
            className={barCls}
            cx={CX}
            cy={CX}
            r={R}
            fill="none"
            strokeWidth="5"
            strokeLinecap="round"
            strokeDasharray={C}
            strokeDashoffset={dashOffset}
            stroke="url(#pb-arc)"
          />

          {/* the glow that rides the head of the arc */}
          <g ref={face.headRef} className="head" transform={`rotate(${headDeg} ${CX} ${CX})`}>
            <circle className="head-glow" cx={CX + R} cy={CX} r="9" fill="url(#pb-head)" />
            <circle className="head-dot" cx={CX + R} cy={CX} r="1.9" />
          </g>
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
              <span ref={face.pctRef}>{Math.round(face.initial * 100)}</span>
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
        <span className="title">{title}</span>
        <span className="face" id="process-readout">
          <span className={railCls} aria-hidden="true">
            {STAGE_ORDER.map((s, i) => (
              <i
                key={s}
                className={`lamp${si >= 0 && i < si ? ' past' : ''}${si === i ? ' now' : ''}`}
              />
            ))}
          </span>
          <span className="cells">
            {cells.map((c, i) => (
              <span className="cell" key={c.k} title={c.title}>
                <b>{c.k}</b>
                {running && i === 2 ? (
                  <em ref={face.clockRef} className="num clock">
                    {c.v}
                  </em>
                ) : (
                  <em className={c.num ? 'num' : undefined}>{c.v}</em>
                )}
              </span>
            ))}
          </span>
        </span>
      </span>
    </button>
  );
}
