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

export function ProcessButton() {
  const source = useStore((s) => s.source);
  const original = useStore((s) => s.original);
  const engineReady = useStore((s) => s.engineStatus === 'ready');
  const job = useStore((s) => s.job);
  const analyzing = useStore((s) => s.analyzing);
  const profile = useStore((s) => s.profile);

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
  const dashOffset = C * (1 - shown);

  let title = 'PROCESS';
  let titleCls = 'title';
  let stage = profile === 'studio' ? 'Studio profile' : 'Production profile';
  let units: string | null = null;
  if (running) {
    title = 'CANCEL';
    stage = STAGE_LABEL[status?.stage ?? ''] ?? (status?.stage ?? 'Working');
    if (status?.unit) units = `unit ${status.unit.index} / ${status.unit.total}`;
    else if (status?.message) units = status.message;
  } else if (state === 'done') {
    title = 'PROCESS AGAIN';
    titleCls += ' cyan';
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

  return (
    <button
      type="button"
      className={`process${running ? ' running' : ''}`}
      disabled={!running && !canStart}
      onClick={onClick}
      aria-label={title}
    >
      <span className="ring" aria-hidden="true">
        <svg viewBox="0 0 84 84" style={{ width: '100%', height: '100%' }}>
          <circle className="track" cx="42" cy="42" r={R} fill="none" strokeWidth="6" />
          <circle className="track-hi" cx="42" cy="42" r={R + 3.5} fill="none" strokeWidth="1" />
          <circle
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
        <span className="center">
          {running ? (
            <span className="pct">
              {Math.round(progress * 100)}
              <small>%</small>
            </span>
          ) : state === 'done' ? (
            <IconCheck className="icon" style={{ color: 'var(--ok)' }} />
          ) : state === 'failed' ? (
            <IconWarn className="icon" style={{ color: 'var(--err)' }} />
          ) : (
            <IconBolt className="icon" style={{ color: 'var(--amber)' }} />
          )}
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
