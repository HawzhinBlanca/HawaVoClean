import { useReducedMotion } from 'motion/react';
import { useEffect, useRef, useState } from 'react';
import { useStore } from '../state/store';
import { Led, type LedState } from './Led';

const HOST_LABEL: Record<string, string> = {
  resolve: 'RESOLVE',
  electron: 'DESKTOP',
  web: 'WEB',
};

/** Length of the readout cross-fade; must match `ix-txt-*` in interaction.css. */
const SWAP_MS = 240;

/**
 * Holds the previous reading for the length of the cross-fade so the two words
 * overlap instead of the slot going blank for a frame. Under reduced motion
 * there is no cross-fade, so nothing is held and the word simply changes.
 */
function useSwap(text: string, enabled: boolean): string | null {
  const [prev, setPrev] = useState<string | null>(null);
  const last = useRef(text);
  useEffect(() => {
    if (last.current === text) return;
    const before = last.current;
    last.current = text;
    if (!enabled) return;
    setPrev(before);
    const t = window.setTimeout(() => setPrev(null), SWAP_MS);
    return () => window.clearTimeout(t);
  }, [text, enabled]);
  return prev === text ? null : prev;
}

/** m:ss.d, the same shape the transport and the plate use. */
function fmtDur(sec: number): string {
  if (!Number.isFinite(sec) || sec < 0) return '—';
  const m = Math.floor(sec / 60);
  const r = sec - m * 60;
  return `${m}:${r.toFixed(1).padStart(4, '0')}`;
}

function signed(v: number, digits = 1): string {
  const s = v > 0 ? '+' : v < 0 ? '\u2212' : '\u00b1';
  return `${s}${Math.abs(v).toFixed(digits)}`;
}

/**
 * B8 · the header used to be 46 px of empty chassis between the wordmark and
 * the engine lamp — at 1920 it was a third of the window wide and said
 * nothing. It now carries the master readout: what state the whole screen is
 * in, and the two or three numbers that state is *about*. It is the only
 * place those numbers appear at a glance-across-the-room size; the footer's
 * status line is a sentence, the plate's face is per-run, and this is the
 * headline. Below 1240 there is no room for it and it is not drawn at all.
 */
function HeaderNow() {
  const source = useStore((s) => s.source);
  const original = useStore((s) => s.original);
  const cleaned = useStore((s) => s.cleaned);
  const analyzing = useStore((s) => s.analyzing);
  const upload = useStore((s) => s.upload);
  const job = useStore((s) => s.job);
  const report = useStore((s) => s.report);

  const status = job?.status ?? null;
  const state = status?.state ?? (job ? 'queued' : null);
  const running = state === 'running' || state === 'queued';

  let tone = 'idle';
  let label = 'No clip';
  const facts: Array<{ k: string; v: string }> = [];

  if (upload) {
    tone = 'busy';
    label = 'Uploading';
    facts.push({
      k: upload.name,
      v: `${Math.round((upload.total ? upload.loaded / upload.total : 0) * 100)}%`,
    });
  } else if (analyzing) {
    tone = 'busy';
    label = 'Analyzing';
    if (source) facts.push({ k: 'Clip', v: source.name });
  } else if (running) {
    tone = 'busy';
    label = 'Running';
    facts.push({ k: 'Stage', v: status?.stage ?? 'working' });
    if (status?.unit) facts.push({ k: 'Unit', v: `${status.unit.index} / ${status.unit.total}` });
    facts.push({ k: 'Done', v: `${Math.round((status?.progress ?? 0) * 100)}%` });
  } else if (state === 'failed') {
    tone = 'fail';
    label = 'Failed';
    facts.push({ k: 'Reason', v: status?.error?.code ?? status?.message ?? 'unknown' });
  } else if (state === 'cancelled') {
    tone = 'idle';
    label = 'Cancelled';
    if (source) facts.push({ k: 'Clip', v: source.name });
  } else if (state === 'done' && report) {
    tone = 'done';
    label = 'Done';
    const sum = report.summary;
    facts.push({ k: 'Units', v: `${sum.enhanced ?? 0} / ${sum.units_total ?? 0}` });
    const li = original?.loudness.integrated_lufs;
    const lo = cleaned?.loudness.integrated_lufs;
    if (li !== null && li !== undefined && lo !== null && lo !== undefined) {
      facts.push({ k: 'LUFS', v: signed(lo - li) });
    }
    const ni = original?.noise_floor_db;
    const no = cleaned?.noise_floor_db;
    if (ni !== null && ni !== undefined && no !== null && no !== undefined) {
      facts.push({ k: 'Noise', v: `${signed(no - ni)} dB` });
    }
  } else if (source && original) {
    tone = 'ready';
    label = 'Armed';
    facts.push({ k: 'Clip', v: source.name });
    facts.push({ k: 'Length', v: fmtDur(original.duration_s) });
    facts.push({
      k: 'Format',
      v: `${(original.sample_rate / 1000).toFixed(1)} kHz · ${original.channels === 1 ? 'mono' : original.channels === 2 ? 'stereo' : `${original.channels} ch`}`,
    });
  } else if (source) {
    tone = 'idle';
    label = 'No analysis';
    facts.push({ k: 'Clip', v: source.name });
  }

  return (
    <div className="hdr-now" data-tone={tone}>
      <span className="hn-state">{label}</span>
      {facts.length > 0 ? (
        <span className="hn-facts">
          {facts.map((f) => (
            <span className="hn-kv" key={f.k}>
              <span className="hn-k">{f.k}</span>
              <span className="hn-v">{f.v}</span>
            </span>
          ))}
        </span>
      ) : null}
    </div>
  );
}

export function Header() {
  const host = useStore((s) => s.host);
  const engineStatus = useStore((s) => s.engineStatus);
  const engineVersion = useStore((s) => s.engineVersion);
  const job = useStore((s) => s.job);
  const analyzing = useStore((s) => s.analyzing);
  const reduced = useReducedMotion() === true;

  const busy =
    analyzing || (job?.status && (job.status.state === 'running' || job.status.state === 'queued'));
  let led: LedState = 'off';
  let text = 'ENGINE OFFLINE';
  if (engineStatus === 'connecting') {
    led = 'busy';
    text = 'ENGINE CONNECTING';
  } else if (engineStatus === 'ready') {
    led = busy ? 'busy' : 'ok';
    text = busy ? 'ENGINE BUSY' : 'ENGINE READY';
  } else {
    led = 'err';
    text = 'ENGINE OFFLINE';
  }

  // CONNECTING → READY → BUSY are three readings of one instrument, so they
  // cross-fade in place rather than snapping. The slot is a fixed width
  // (interaction.css), so the version chip beside it never moves.
  const prev = useSwap(text, !reduced);

  return (
    <header className="panel header">
      {/* D1 · the page had no h1 at all. The wordmark is the document's one
          top-level heading; nothing about how it is drawn changes. */}
      <h1 className="wordmark">
        <span className="name">HAWAVOCLEAN</span>
        <span className="ver">v3.2</span>
      </h1>
      <div className="header-mid">
        <span className={`badge${host === 'resolve' ? ' accent' : ''}`}>
          {HOST_LABEL[host] ?? 'WEB'}
        </span>
        <HeaderNow />
      </div>
      {/* The lamp's word changes at most a few times a session (connecting →
          ready → busy), so it is a safe polite region: no counter, no clock,
          nothing that ticks. */}
      <div className="engine" aria-live="polite">
        <Led state={led} />
        <span className="txt">
          {prev ? (
            <span className="out" key={prev} aria-hidden="true">
              {prev}
            </span>
          ) : null}
          <span className="in" key={text}>
            {text}
          </span>
        </span>
        {engineVersion ? <span className="ver">v{engineVersion}</span> : null}
      </div>
    </header>
  );
}
