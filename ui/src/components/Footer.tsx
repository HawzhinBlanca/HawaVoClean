import { AnimatePresence, motion } from 'motion/react';
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
      <AnimatePresence>
        {error ? (
          <motion.div
            className="errbar"
            role="alert"
            initial={{ opacity: 0, transform: 'translateY(8px)' }}
            animate={{ opacity: 1, transform: 'translateY(0px)' }}
            exit={{ opacity: 0, transform: 'translateY(8px)' }}
            transition={{ duration: 0.18, ease: [0.22, 1, 0.36, 1] }}
          >
            <IconWarn />
            <span className="text" title={error}>
              <b>Engine error</b>
              {error}
            </span>
            <button className="dismiss" onClick={() => setError(null)} aria-label="Dismiss error">
              <IconCancel size={14} />
            </button>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </>
  );
}
