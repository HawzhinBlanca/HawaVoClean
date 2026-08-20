// B5 · the session's memory. Every run that reaches a terminal state is kept
// here with everything needed to put it back on screen — its report, its
// analyses, the paths of its artefacts — so re-selecting one is a state
// restore, not a re-run: nothing is decoded again.

import { selectRun } from '../state/actions';
import { HISTORY_LIMIT, useStore, type HistoryEntry } from '../state/store';
import { IconCancel, IconCheck, IconWarn } from './Icons';

function clockOf(at: number): string {
  const d = new Date(at);
  const p = (n: number): string => String(n).padStart(2, '0');
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

function fmtRuntime(ms: number | null): string {
  if (ms === null || !Number.isFinite(ms)) return '—';
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)} s`;
  const m = Math.floor(s / 60);
  return `${m}:${(s - m * 60).toFixed(0).padStart(2, '0')}`;
}

/** The LUFS move, signed, or an em-dash when either end is missing. */
function lufsDelta(e: HistoryEntry): string {
  if (e.lufsIn === null || e.lufsOut === null) return '—';
  const d = e.lufsOut - e.lufsIn;
  const sign = d > 0 ? '+' : d < 0 ? '−' : '±';
  return `${sign}${Math.abs(d).toFixed(1)}`;
}

function OutcomeMark({ outcome }: { outcome: HistoryEntry['outcome'] }) {
  if (outcome === 'done') return <IconCheck size={11} />;
  if (outcome === 'failed') return <IconWarn size={11} />;
  return <IconCancel size={11} />;
}

function Row({
  entry,
  current,
  blocked,
}: {
  entry: HistoryEntry;
  current: boolean;
  blocked: string | null;
}) {
  const units =
    entry.enhanced !== null && entry.unitsTotal !== null
      ? `${entry.enhanced}/${entry.unitsTotal}`
      : '—';
  // D1 · a row draws seven fragments across two lines; read one after another
  // they are unintelligible ("Flute 09.m4a.mp4 ON SCREEN studio 5/5 units +3.2
  // LUFS Δ 4.2 s took 06:37:31"). The row is a button, so it gets one name
  // that is a sentence, and the fragments are marked as the decoration of that
  // sentence.
  // A row that no longer owns its files says so on its face. Without this the
  // list's worst failure is silent: the row shows its own cached report while
  // its download links hand over a later run's bytes, so the screen and the
  // file disagree and nothing on either says which one is the run.
  const flag = entry.supersededBy
    ? 'SUPERSEDED'
    : entry.artifacts && !entry.artifacts.master
      ? // A master that is still on disk but holds no audio is not "gone", and
        // saying so sent the user looking in the wrong place. The kind of
        // failure travels with the answer (state/store.ts, `ArtifactState`).
        (entry.artifacts.flag ?? 'FILE GONE')
      : null;
  const flagWhy = entry.supersededBy
    ? 'A later run wrote over this run’s files. The report here is this session’s own copy; the files on disk belong to that later run.'
    : entry.artifacts?.reason
      ? `${entry.artifacts.reason} Open this run again to check for them once more.`
      : null;
  const say = [
    entry.inputName || 'clip',
    current ? 'currently on screen' : null,
    entry.profile,
    entry.outcome === 'done'
      ? `${units} units enhanced, LUFS change ${lufsDelta(entry)}, took ${fmtRuntime(entry.durationMs)}`
      : entry.outcome === 'failed'
        ? `failed: ${entry.error || 'unknown error'}`
        : 'cancelled',
    `at ${clockOf(entry.at)}`,
    flagWhy,
    blocked ? 'unavailable while the engine is offline' : null,
  ]
    .filter(Boolean)
    .join(', ');
  return (
    <button
      type="button"
      className="hist-row"
      data-current={current ? 'true' : 'false'}
      data-outcome={entry.outcome}
      data-flag={flag ? 'true' : undefined}
      aria-label={say}
      aria-current={current ? 'true' : undefined}
      aria-disabled={blocked ? 'true' : undefined}
      onClick={() => {
        if (blocked) return;
        void selectRun(entry.jobId);
      }}
      title={
        blocked ??
        flagWhy ??
        (entry.outcome === 'failed' && entry.error
          ? `${entry.inputName} — ${entry.error}`
          : `${entry.inputName} → ${entry.outputPath || 'no output'}`)
      }
    >
      <span className="hist-rail" aria-hidden="true" />
      <span className="hist-line1">
        <span className="hist-mark" aria-hidden="true">
          <OutcomeMark outcome={entry.outcome} />
        </span>
        {/* B8 · the filename owns line 1. The clock used to sit beside it and
            pushed the name into an ellipsis in a 300 px column — the one
            thing on the row that identifies the run. It reads perfectly well
            at the end of line 2. */}
        <span className="hist-name">{entry.inputName || 'clip'}</span>
        {current ? <span className="hist-now">ON SCREEN</span> : null}
        {flag ? <span className="hist-flag">{flag}</span> : null}
      </span>
      <span className="hist-line2">
        <span className="hist-profile">{entry.profile}</span>
        {entry.outcome === 'done' ? (
          <>
            <span className="hist-kv">
              <span className="hist-v mono">{units}</span>
              <span className="hist-k">units</span>
            </span>
            <span className="hist-kv">
              <span className="hist-v mono">{lufsDelta(entry)}</span>
              <span className="hist-k">LUFS Δ</span>
            </span>
            <span className="hist-kv">
              <span className="hist-v mono">{fmtRuntime(entry.durationMs)}</span>
              <span className="hist-k">took</span>
            </span>
          </>
        ) : (
          <span className="hist-why">
            {entry.outcome === 'failed' ? entry.error || 'failed' : 'cancelled'}
          </span>
        )}
        <span className="hist-clock mono">{clockOf(entry.at)}</span>
      </span>
    </button>
  );
}

export function JobHistory() {
  const history = useStore((s) => s.history);
  const currentRunId = useStore((s) => s.currentRunId);
  // B6 · putting a run back on screen re-points both decks at files the engine
  // serves, so it needs the engine. The list itself — every number in it — is
  // held locally and stays readable throughout an outage.
  const engineReady = useStore((s) => s.engineStatus === 'ready');
  const blocked = engineReady
    ? null
    : 'The engine is offline — this run stays in the list and opens again the moment it reconnects.';

  return (
    <section
      className="panel history"
      aria-label="Session runs"
      data-empty={history.length === 0 ? 'true' : 'false'}
    >
      <div className="panel-head">
        <div className="panel-title">
          <span>Session runs</span>
          {history.length > 0 ? (
            <span className="sub">
              · {history.length} of {HISTORY_LIMIT} kept
            </span>
          ) : null}
        </div>
      </div>
      <div className="hist-body">
        {history.length === 0 ? (
          <div className="hist-empty">
            <p className="msg">Nothing has been processed yet.</p>
            <p className="sub">
              Every finished pass is kept here for the session — its report, its cleaned deck and
              its meters — and comes back instantly when you pick it.
            </p>
          </div>
        ) : (
          <ul
            className="hist-list"
            aria-label={`${history.length} run${history.length === 1 ? '' : 's'}, newest first`}
          >
            {history.map((e) => (
              <li key={e.jobId}>
                <Row entry={e} current={e.jobId === currentRunId} blocked={blocked} />
              </li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}
