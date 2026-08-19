import { useCallback, useRef, useState, type DragEvent } from 'react';
import { getBridge } from '../bridge';
import { ingestFile, openFileDialog, useResolveClip } from '../state/actions';
import { useStore } from '../state/store';
import { IconClip, IconDrop, IconFolder } from './Icons';

function fmtDuration(s: number): string {
  const m = Math.floor(s / 60);
  const r = s - m * 60;
  return `${m}:${r.toFixed(1).padStart(4, '0')}`;
}

export function SourceStrip() {
  const bridge = getBridge();
  const hasResolve = Boolean(bridge.resolve);
  const isWeb = bridge.host === 'web';
  const source = useStore((s) => s.source);
  const original = useStore((s) => s.original);
  const analyzing = useStore((s) => s.analyzing);
  const engineReady = useStore((s) => s.engineStatus === 'ready');
  const running = useStore(
    (s) => !!s.job?.status && (s.job.status.state === 'running' || s.job.status.state === 'queued'),
  );
  const [over, setOver] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const disabled = !engineReady || running;

  const onDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault();
      setOver(false);
      if (disabled) return;
      const file = e.dataTransfer.files.item(0);
      if (file) void ingestFile(file);
    },
    [disabled],
  );

  const onDragOver = useCallback((e: DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
    setOver(true);
  }, []);

  const onDragLeave = useCallback(() => setOver(false), []);

  const onOpen = useCallback(() => {
    if (isWeb) inputRef.current?.click();
    else void openFileDialog();
  }, [isWeb]);

  return (
    <section className="panel source" onDragOver={onDragOver} onDragLeave={onDragLeave} onDrop={onDrop}>
      {hasResolve ? (
        <button className="btn" disabled={disabled} onClick={() => void useResolveClip()}>
          <IconClip />
          <span>Use clip selected in Resolve</span>
        </button>
      ) : (
        <span />
      )}
      <button className="btn" disabled={disabled} onClick={onOpen}>
        <IconFolder />
        <span>Open file…</span>
      </button>
      <div className={`dropzone${over ? ' over' : ''}`}>
        <IconDrop />
        <span className="hint">
          {over ? 'Release to load' : 'Drop an audio or video file here'}
        </span>
        {isWeb ? (
          <input
            ref={inputRef}
            type="file"
            accept="audio/*,video/*,.wav,.m4a,.mp4,.mov,.mp3,.flac,.aac,.aif,.aiff"
            disabled={disabled}
            title="Choose a file"
            onChange={(e) => {
              const f = e.currentTarget.files?.item(0);
              if (f) void ingestFile(f);
              e.currentTarget.value = '';
            }}
          />
        ) : null}
      </div>
      <div className="clipinfo">
        {source ? (
          <>
            <span className="origin">{source.origin}</span>
            <span className="name" title={source.path}>
              {source.name}
            </span>
            {original ? (
              <>
                <span className="kv">
                  <span className="k">Duration</span>
                  <span className="v">{fmtDuration(original.duration_s)}</span>
                </span>
                <span className="kv">
                  <span className="k">Rate</span>
                  <span className="v">{(original.sample_rate / 1000).toFixed(1)} kHz</span>
                </span>
                <span className="kv">
                  <span className="k">Channels</span>
                  <span className="v">{original.channels === 1 ? 'Mono' : original.channels === 2 ? 'Stereo' : `${original.channels} ch`}</span>
                </span>
              </>
            ) : analyzing ? (
              <span className="kv">
                <span className="k">Status</span>
                <span className="v">analyzing…</span>
              </span>
            ) : null}
          </>
        ) : (
          <span className="empty">No clip loaded</span>
        )}
      </div>
    </section>
  );
}
