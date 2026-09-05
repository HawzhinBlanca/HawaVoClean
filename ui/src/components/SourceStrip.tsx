import { useCallback, useRef, useState, type DragEvent } from 'react';
import { getBridge } from '../bridge';
import {
  ACCEPTED_EXTENSIONS,
  ACCEPTED_SHORTLIST,
  cancelAnalysis,
  channelWarning,
  cancelUpload,
  ingestDataTransfer,
  ingestFile,
  openFileDialog,
  rateWarning,
  sourceFailureLabel,
  useResolveClip,
} from '../state/actions';
import { useStore, jobInFlight } from '../state/store';
import { IconCancel, IconClip, IconDrop, IconFolder, IconWarn } from './Icons';

function fmtDuration(s: number): string {
  const m = Math.floor(s / 60);
  const r = s - m * 60;
  return `${m}:${r.toFixed(1).padStart(4, '0')}`;
}

/** Bytes as a person reads them; two significant decimals below 10 units. */
function fmtBytes(n: number): string {
  if (!Number.isFinite(n) || n < 0) return '—';
  if (n < 1024) return `${n} B`;
  const units = ['KB', 'MB', 'GB', 'TB'];
  let v = n / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v < 10 ? v.toFixed(1) : Math.round(v)} ${units[i]}`;
}

const ACCEPT_ATTR = ACCEPTED_EXTENSIONS.map((extension) => `.${extension}`).join(',');

/**
 * B2 · the transfer, while it is happening. An upload is the only thing in
 * this product that can take minutes with nothing to look at, so it gets a
 * real readout — bytes, percent, a determinate bar — and a real way out.
 */
function UploadProgress() {
  const upload = useStore((s) => s.upload);
  if (!upload) return null;
  const pct = upload.total > 0 ? Math.min(1, upload.loaded / upload.total) : 0;
  return (
    // D1 · this was `role="status" aria-live="polite"` around a percentage and
    // a byte counter that update on every XHR progress event — a screen reader
    // read the whole strip aloud several times a second for the length of a
    // gigabyte upload. The determinate `progressbar` below already exposes the
    // value on demand, which is the right way to publish a number that moves;
    // the app's one polite region (App.tsx) announces the transitions.
    <div className="uploading" role="group" aria-label={`Uploading ${upload.name}`}>
      <div className="up-head">
        <span className="up-name" title={upload.name}>
          {upload.name}
        </span>
        <span className="up-pct mono">{Math.round(pct * 100)}%</span>
        <button
          type="button"
          className="up-cancel"
          onClick={cancelUpload}
          title="Cancel the upload (Esc)"
          aria-label="Cancel upload"
        >
          <IconCancel size={11} />
        </button>
      </div>
      <div
        className="up-track"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(pct * 100)}
        aria-label={`Uploading ${upload.name}`}
      >
        <i style={{ transform: `scaleX(${pct})` }} />
      </div>
      <div className="up-foot">
        <span className="mono">
          {fmtBytes(upload.loaded)} / {fmtBytes(upload.total)}
        </span>
        <span className="up-phase">
          {upload.phase === 'finishing' ? 'writing to the work dir' : 'uploading'}
        </span>
      </div>
    </div>
  );
}

/**
 * C5 · the wait, and the way out of it.
 *
 * An analysis is a single request whose cost is entirely the engine's decode:
 * measured, the reply is the same size whatever the file is (27,243 B for a
 * ten-minute clip, 27,413 B for a fifty-millisecond one) and parses in 0.3 ms,
 * so the browser never has anything to do but wait. What it lacked was a door.
 * A three-hour file is half a minute of nothing, and until now the only way
 * out was to drop a different file over the top of it — so the strip now says
 * whose analysis is running and offers to abandon it.
 */
function AnalyzeProgress({ name }: { name: string }) {
  return (
    // `role="group"`, not a second `role="status" aria-live="polite"`. App.tsx
    // already announces "Analyzing <clip>" through the app's single polite
    // region; a second one narrating the same transition is the double
    // announcement D1 exists to prevent, and it is the same treatment
    // `UploadProgress` above already had. The row stays fully readable
    // whenever the user browses to it.
    <div className="analyzing-row" role="group" aria-label={`Analyzing ${name}`}>
      <span className="an-spin" aria-hidden="true" />
      <span className="an-copy">
        <span className="an-head">
          Analyzing <b title={name}>{name}</b>
        </span>
        <span className="an-detail">reading the whole file once — the engine is decoding it</span>
      </span>
      <button
        type="button"
        className="an-cancel"
        onClick={cancelAnalysis}
        title="Stop reading this clip"
        aria-label="Cancel analysis"
      >
        <IconCancel size={11} />
      </button>
    </div>
  );
}

/**
 * B2 · what happened when the drop could not be used. A refused drop names the
 * thing that was refused and the formats that would have worked; a multi-file
 * drop is not a refusal at all, it is a note about which file was taken, and it
 * is toned accordingly.
 */
function DropRejectionNote() {
  const rejection = useStore((s) => s.rejection);
  const setRejection = useStore((s) => s.setRejection);
  if (!rejection) return null;
  const info = rejection.kind === 'multi';
  return (
    <div className={`droprej${info ? ' info' : ''}`} role="status">
      <span className="dr-glyph" aria-hidden="true">
        {info ? <IconDrop size={12} /> : <IconWarn size={12} />}
      </span>
      <span className="dr-copy">
        <span className="dr-head">
          {info ? 'Loaded' : 'Cannot open'} <b title={rejection.name}>{rejection.name}</b>
        </span>
        <span className="dr-detail">
          {rejection.detail}
          {rejection.kind === 'type' ? (
            <>
              {' '}
              Accepted: <span className="dr-types">{ACCEPTED_SHORTLIST}</span>.
            </>
          ) : null}
        </span>
      </span>
      <button
        type="button"
        className="dr-dismiss"
        onClick={() => setRejection(null)}
        aria-label="Dismiss"
      >
        <IconCancel size={11} />
      </button>
    </div>
  );
}

export function SourceStrip() {
  const bridge = getBridge();
  const hasResolve = Boolean(bridge.resolve);
  const isWeb = bridge.host === 'web';
  const source = useStore((s) => s.source);
  const original = useStore((s) => s.original);
  const analyzing = useStore((s) => s.analyzing);
  const upload = useStore((s) => s.upload);
  const engineReady = useStore((s) => s.engineStatus === 'ready');
  // `jobInFlight`, not a hand-written state check: this also has to cover the
  // window between the engine accepting a job and its first status arriving,
  // where `status` is still null. Loading a clip in that window orphans the
  // run.
  const running = useStore((s) => jobInFlight(s.job));
  const engineOffline = useStore((s) => s.engineStatus === 'offline');
  // C5 · subscribing to `error` is what makes the clip-row failure chip
  // reactive: the classification behind `sourceFailureLabel()` is set on the
  // same beat as this string, so a change here is a change there.
  const error = useStore((s) => s.error);
  const [over, setOver] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const uploading = Boolean(upload);
  const disabled = !engineReady || running || uploading;
  // B6 · every refusal in this strip names its own reason. `aria-disabled`
  // rather than `disabled` on the buttons, because a disabled control takes no
  // pointer events and would swallow the tooltip that carries the reason.
  const why = engineOffline
    ? 'The engine is offline — loading a clip needs it. This comes back on its own when it reconnects.'
    : !engineReady
      ? 'Connecting to the engine…'
      : running
        ? 'A run is in flight — finish or cancel it before loading another clip.'
        : uploading
          ? 'An upload is in flight — cancel it before loading another clip.'
          : null;

  const onDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault();
      setOver(false);
      if (disabled) return;
      // The whole DataTransfer, not just files[0]: folders and multi-file
      // drops are answered, not silently mishandled.
      void ingestDataTransfer(e.dataTransfer);
    },
    [disabled],
  );

  const onDragOver = useCallback((e: DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
    setOver(true);
  }, []);

  // `dragleave` also fires when the pointer crosses from the strip into one of
  // its own children, which would flicker the highlight off and on. Only a
  // leave whose destination is outside the strip is a real leave.
  const onDragLeave = useCallback((e: DragEvent) => {
    const to = e.relatedTarget as Node | null;
    if (to && e.currentTarget.contains(to)) return;
    setOver(false);
  }, []);

  const rateNote = original ? rateWarning(original.sample_rate) : null;
  // C5 · the same pre-flight the rate cell has had, for the other property of
  // a clip this pipeline refuses. An 8-channel file used to sit here on a
  // plain cell with PROCESS armed, and the only news it ever got was the
  // failure ten seconds later.
  const chanNote = original ? channelWarning(original.channels) : null;

  const onOpen = useCallback(() => {
    if (isWeb) inputRef.current?.click();
    else void openFileDialog();
  }, [isWeb]);

  return (
    <section
      className="panel source"
      // D1 · an unnamed <section> is not a landmark, which left the drop well,
      // the file input and the clip readout outside every region on the page.
      aria-label="Source clip"
      data-uploading={uploading ? 'true' : 'false'}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      {hasResolve ? (
        <button
          className="btn"
          type="button"
          aria-disabled={disabled || undefined}
          title={why ?? 'Load the clip currently selected in Resolve'}
          onClick={() => {
            if (!disabled) void useResolveClip();
          }}
        >
          <IconClip />
          <span>Use clip selected in Resolve</span>
        </button>
      ) : (
        <span />
      )}
      <button
        className="btn"
        type="button"
        aria-disabled={disabled || undefined}
        title={why ?? 'Choose an audio or video file to clean'}
        onClick={() => {
          if (!disabled) onOpen();
        }}
      >
        <IconFolder />
        <span>Open file…</span>
      </button>
      {uploading ? (
        <UploadProgress />
      ) : analyzing && source ? (
        <AnalyzeProgress name={source.name} />
      ) : (
        <div className={`dropzone${over ? ' over' : ''}${disabled ? ' disabled' : ''}`}>
          <span className="glyph" aria-hidden="true">
            <IconDrop size={13} />
          </span>
          <span className="copy">
            <span className="hint">
              {over
                ? 'Release to load'
                : disabled
                  ? running
                    ? 'Busy — finish or cancel the job first'
                    : uploading
                      ? 'An upload is already in flight'
                      : engineOffline
                        ? 'Engine offline — reconnecting'
                        : 'Waiting for the engine'
                  : 'Drop an audio or video file here'}
            </span>
            <span className="types">
              {over
                ? 'Analysis starts immediately'
                : engineOffline
                  ? 'Nothing already loaded is lost — this well opens again the moment the engine answers'
                  : isWeb
                    ? `or click to browse · ${ACCEPTED_SHORTLIST}`
                    : ACCEPTED_SHORTLIST}
            </span>
          </span>
          {isWeb ? (
            <input
              ref={inputRef}
              type="file"
              accept={ACCEPT_ATTR}
              disabled={disabled}
              // D1 · a `title` is not a label: the control had no accessible
              // name at all in the tree.
              aria-label="Choose an audio or video file to clean"
              title="Choose a file"
              onChange={(e) => {
                const f = e.currentTarget.files?.item(0);
                if (f) void ingestFile(f);
                e.currentTarget.value = '';
              }}
            />
          ) : null}
        </div>
      )}
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
                {/* C5 · a rate the pipeline will refuse is flagged here, at the
                    moment the analysis lands, rather than a second after the
                    user has already pressed PROCESS. */}
                <span className={`kv${rateNote ? ' warn' : ''}`} title={rateNote ?? undefined}>
                  <span className="k">Rate</span>
                  <span className="v">
                    {(original.sample_rate / 1000).toFixed(1)} kHz
                    {rateNote ? <IconWarn size={10} /> : null}
                  </span>
                </span>
                <span className={`kv${chanNote ? ' warn' : ''}`} title={chanNote ?? undefined}>
                  <span className="k">Channels</span>
                  <span className="v">
                    {original.channels === 1
                      ? 'Mono'
                      : original.channels === 2
                        ? 'Stereo'
                        : `${original.channels} ch`}
                    {chanNote ? <IconWarn size={10} /> : null}
                  </span>
                </span>
              </>
            ) : analyzing ? (
              <span className="kv">
                <span className="k">Status</span>
                <span className="v">analyzing…</span>
              </span>
            ) : (
              /* C5 · a clip whose analysis failed used to leave this row
                 showing a name and nothing else, so the strip looked as if it
                 had simply not got round to it yet. The error bar carries the
                 sentence; this says, in the one place that is about the clip,
                 that the clip is the thing that is wrong. */
              <span
                className="kv bad"
                title={
                  error ??
                  'This clip has not been analyzed, so there is nothing to process yet.'
                }
              >
                <span className="k">Status</span>
                <span className="v">
                  <IconWarn size={10} /> {sourceFailureLabel()}
                </span>
              </span>
            )}
          </>
        ) : (
          <span className="empty">No clip loaded</span>
        )}
      </div>
      <DropRejectionNote />
    </section>
  );
}
