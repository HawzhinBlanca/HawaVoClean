import { useStore } from '../state/store';
import { IconCancel, IconWarn } from './Icons';
import { Led } from './Led';

export function Footer() {
  const statusLine = useStore((s) => s.statusLine);
  const error = useStore((s) => s.error);
  const setError = useStore((s) => s.setError);
  const job = useStore((s) => s.job);
  const engineStatus = useStore((s) => s.engineStatus);

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
        <div className="errbar" role="alert">
          <IconWarn />
          <span className="text" title={error}>
            <b>Engine error</b>
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
