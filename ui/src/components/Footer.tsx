import { useEffect, useRef } from 'react';
import { useStore } from '../state/store';
import { IconCancel, IconWarn } from './Icons';
import { Led } from './Led';

/**
 * D1 · the bar is `position: fixed`, so it floats over the chassis — and
 * measured at 1440x900 it lands exactly on the second row of artefact buttons:
 * `elementFromPoint` at the centre of "Summary .txt" and "Copy summary"
 * returned the bar, not the buttons. That is the failure this codebase already
 * named for the offline banner ("an overlay can cover a control, and the
 * moment something has gone wrong is the moment the controls matter most") and
 * it also put the two labels at 1.07:1 against the bar's red.
 *
 * The bar's height depends on how far the engine's message wraps, so the strip
 * to keep clear cannot be a constant in the stylesheet. It is measured and
 * published as a custom property; `.main` gives back exactly that much
 * (interaction.css). Unmounting clears it, so a dismissed error costs nothing.
 */
function useReservedStrip(active: boolean): (el: HTMLDivElement | null) => void {
  const ro = useRef<ResizeObserver | null>(null);
  useEffect(
    () => () => {
      ro.current?.disconnect();
      document.documentElement.style.removeProperty('--errbar-h');
    },
    [],
  );
  return (el: HTMLDivElement | null): void => {
    ro.current?.disconnect();
    ro.current = null;
    if (!el || !active) {
      document.documentElement.style.removeProperty('--errbar-h');
      return;
    }
    const write = (): void => {
      document.documentElement.style.setProperty('--errbar-h', `${Math.ceil(el.offsetHeight)}px`);
    };
    write();
    ro.current = new ResizeObserver(write);
    ro.current.observe(el);
  };
}

export function Footer() {
  const statusLine = useStore((s) => s.statusLine);
  const error = useStore((s) => s.error);
  // Honesty · the label said ENGINE ERROR over everything that ever reached
  // this bar. It was measured saying it over a clipboard permission the
  // browser refused, which named the wrong culprit in the largest type on the
  // screen. Each failure now arrives with its own source (state/errors.ts,
  // `failureSource`); anything that somehow arrives without one says the
  // neutral thing rather than inventing an engine to blame.
  const errorLabel = useStore((s) => s.errorLabel);
  const setError = useStore((s) => s.setError);
  const job = useStore((s) => s.job);
  const engineStatus = useStore((s) => s.engineStatus);

  const reserve = useReservedStrip(Boolean(error));
  const led = error
    ? 'err'
    : job?.status && (job.status.state === 'running' || job.status.state === 'queued')
      ? 'busy'
      : engineStatus === 'ready'
        ? 'ok'
        : engineStatus === 'connecting'
          ? 'busy'
          : 'err';

  return (
    <>
      <footer className="panel footer">
        <Led state={led} />
        <span className="msg" title={statusLine}>
          {statusLine}
        </span>
        <span className="meta">
          {job ? `job ${job.id.slice(0, 10)}${job.streamConnected ? ' · live' : ''}` : 'idle'}
        </span>
      </footer>
      {/* The bar used to enter and leave through `AnimatePresence`. Measured on
          React 19 + motion 12: the exit animation runs, but the child is never
          unmounted — the dismissed bar stays in the DOM as a `position: fixed`,
          `z-index: 40`, opacity-0 slab 1416x38 across the bottom of the window,
          and `elementFromPoint` at its centre still returns it. Every control
          in that band (the whole artefact row at 1440x900) stopped responding
          after any error had been dismissed. The entrance is now a CSS
          animation — which the reduced-motion block already governs — and the
          dismissal is instant, which is what a dismissal should be. */}
      {error ? (
        <div className="errbar" role="alert" ref={reserve}>
          <IconWarn />
          <span className="text" title={error}>
            <b>{errorLabel ?? 'Error'}</b>
            {error}
          </span>
          <button className="dismiss" onClick={() => setError(null)} aria-label="Dismiss error">
            <IconCancel size={14} />
          </button>
        </div>
      ) : null}
    </>
  );
}
