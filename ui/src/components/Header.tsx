import { useStore } from '../state/store';
import { Led, type LedState } from './Led';

const HOST_LABEL: Record<string, string> = {
  resolve: 'RESOLVE',
  electron: 'DESKTOP',
  web: 'WEB',
};

export function Header() {
  const host = useStore((s) => s.host);
  const engineStatus = useStore((s) => s.engineStatus);
  const engineVersion = useStore((s) => s.engineVersion);
  const job = useStore((s) => s.job);
  const analyzing = useStore((s) => s.analyzing);

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
        <span>{text}</span>
        {engineVersion ? <span className="ver">v{engineVersion}</span> : null}
      </div>
    </header>
  );
}
