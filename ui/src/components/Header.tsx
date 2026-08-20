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
      <div className="wordmark">
        <span className="name">HAWAVOCLEAN</span>
        <span className="ver">v3.2</span>
      </div>
      <div className="header-mid">
        <span className={`badge${host === 'resolve' ? ' accent' : ''}`}>
          {HOST_LABEL[host] ?? 'WEB'}
        </span>
      </div>
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
