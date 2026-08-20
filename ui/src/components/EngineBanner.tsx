// B6 · the engine is a separate process, and it can die. When it does, the
// only honest thing the screen can do is say so in words — a dark LED in the
// corner is a symptom, not a message — and then say the second, more important
// thing: that nothing was lost. Everything on screen is a local fact (the
// clip, its analysis, the report, the run list, the selection, the zoom), so
// the outage costs the user exactly the controls that need a live engine and
// nothing else.

import { useEffect, useState } from 'react';
import { isTerminal } from '../api/sse';
import { retryEngineNow } from '../state/actions';
import { useStore } from '../state/store';
import { IconWarn } from './Icons';

/** m:ss, or "a moment" for the first second of an outage. */
function fmtSince(ms: number): string {
  const s = Math.floor(ms / 1000);
  if (s < 1) return 'a moment';
  if (s < 60) return `${s} s`;
  const m = Math.floor(s / 60);
  return `${m}:${String(s - m * 60).padStart(2, '0')}`;
}

/**
 * Ticks only while the banner is mounted — i.e. only while the engine is
 * actually gone — so a healthy session pays nothing for the countdown.
 */
function useTick(active: boolean): number {
  const [, setN] = useState(0);
  useEffect(() => {
    if (!active) return;
    const id = window.setInterval(() => setN((n) => (n + 1) % 1_000_000), 250);
    return () => window.clearInterval(id);
  }, [active]);
  return Date.now();
}

export function EngineBanner() {
  const engineStatus = useStore((s) => s.engineStatus);
  const offlineSince = useStore((s) => s.engineOfflineSince);
  const nextProbeAt = useStore((s) => s.engineNextProbeAt);
  const probing = useStore((s) => s.engineProbing);
  const job = useStore((s) => s.job);
  const source = useStore((s) => s.source);
  const report = useStore((s) => s.report);
  const history = useStore((s) => s.history);

  const down = engineStatus === 'offline' || (engineStatus === 'connecting' && offlineSince !== null);
  const now = useTick(down);
  if (!down) return null;

  const inflight = Boolean(job) && (!job?.status || !isTerminal(job.status.state));
  const wait = nextProbeAt === null ? 0 : Math.max(0, nextProbeAt - now);
  const nextIn = (wait / 1000).toFixed(1);

  // What is being held for the user, named item by item. A promise that
  // "your work is safe" is worth nothing; a list is worth something.
  const kept: string[] = [];
  if (source) kept.push(source.name);
  if (report) kept.push('its report');
  if (history.length > 0) kept.push(`${history.length} run${history.length === 1 ? '' : 's'}`);

  return (
    // D1 · this used to be `role="alert" aria-live="assertive"` on the whole
    // section — with a retry countdown and an outage clock ticking inside it
    // four times a second. Every tick was a fresh interruption: a screen
    // reader user could not hear anything else while the engine was down. The
    // section is now an ordinary named region, and the announcement is one
    // short sentence in a live element of its own whose text does not tick.
    // Assertive is still the right level for it: the engine going away changes
    // what every control on the screen can do, so it earns the interruption —
    // once.
    <section className="panel offline-bar" aria-label="Engine status">
      <p className="sr-only" role="alert">
        {inflight
          ? 'Engine offline. A run was in flight; its outcome is read back when the engine returns.'
          : 'Engine offline. Nothing on screen was lost, and the app is reconnecting on its own.'}
      </p>
      <span className="ob-glyph" aria-hidden="true">
        <IconWarn size={15} />
      </span>

      <div className="ob-copy">
        <h2 className="ob-title">
          Engine offline
          {offlineSince !== null ? (
            <span className="ob-age mono"> · {fmtSince(now - offlineSince)}</span>
          ) : null}
        </h2>
        {/* Two lengths of the same sentence. A 640 px-tall window cannot spare
            three lines of prose for an alert — the waveform underneath would
            be squeezed to nothing — so the short form is shown there and the
            full one wherever there is room. Both are always in the DOM; CSS
            picks, so neither is ever half-shown or ellipsised. */}
        <p className="ob-detail ob-long">
          {inflight
            ? 'The engine stopped while a run was in flight. Nothing on screen was lost — when it answers again the run’s real outcome is read back from the engine itself, not guessed.'
            : kept.length > 0
              ? `The engine stopped answering. Nothing on screen was lost — ${kept.join(', ')} ${kept.length === 1 ? 'is' : 'are'} still loaded, and the controls that need the engine come back the moment it does.`
              : 'The engine stopped answering. Start it again and this screen reconnects on its own — nothing here needs reloading.'}
        </p>
        <p className="ob-detail ob-short">
          {inflight
            ? 'A run was in flight; its real outcome is read back when the engine returns.'
            : 'Nothing on screen was lost. Reconnecting automatically.'}
        </p>
      </div>

      {/* The countdown redraws four times a second. With the section no longer
          a live region that costs nothing — it is a clock, read when you look
          at it — so it stays in the accessibility tree where a user browsing
          the banner can find out that a retry is coming. */}
      <div className="ob-act">
        <span className={`ob-state${probing ? ' live' : ''}`}>
          {probing ? (
            'Reconnecting…'
          ) : (
            <>
              Retry in <span className="mono">{nextIn}</span> s
            </>
          )}
        </span>
        <button type="button" className="btn small ob-retry" onClick={retryEngineNow}>
          Retry now
        </button>
      </div>
    </section>
  );
}
