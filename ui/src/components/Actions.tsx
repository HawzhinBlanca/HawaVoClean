// B7 · getting the run out of the tool: the three artefacts a pass leaves
// behind, and the one line a person actually pastes into a message.

import { useCallback, useEffect, useRef, useState } from 'react';
import { getBridge } from '../bridge';
import {
  artifactsFor,
  copyReportSummary,
  importToResolve,
  replaceInResolve,
  revealOutput,
  saveCleanedMaster,
} from '../state/actions';
import { useStore } from '../state/store';
import { IconCheck, IconImport, IconReplace, IconReveal } from './Icons';

/**
 * `download` is ignored across origins, so the anchors also carry
 * `target="_blank"`: served by `hawavoclean serve --ui-dir` the UI and the
 * engine share an origin and the attribute wins (nothing opens, the file is
 * saved); under the Vite dev server they do not, and the tab that opens is the
 * file itself. Both paths end with the artefact in the user's hands.
 */
function ArtifactLink({
  url,
  name,
  label,
  title,
}: {
  url: string | null;
  name: string;
  label: string;
  title: string;
}) {
  if (!url) {
    // `aria-disabled` rather than `disabled`: a disabled control takes no
    // pointer events, so its `title` — the sentence that says *why* the file
    // is not there — would never be shown (goal box B6).
    return (
      <button className="btn small" type="button" aria-disabled="true" title={title}>
        <IconImport size={14} />
        <span>{label}</span>
      </button>
    );
  }
  return (
    <a
      className="btn small"
      href={url}
      download={name}
      target="_blank"
      rel="noreferrer"
      title={title}
    >
      <IconImport size={14} />
      <span>{label}</span>
    </a>
  );
}

/** Copies the one-line summary and says so, on the button itself. */
function CopySummary() {
  const report = useStore((s) => s.report);
  const source = useStore((s) => s.source);
  const [copied, setCopied] = useState(false);
  const timer = useRef<number | null>(null);
  const enabled = Boolean(report && source);

  useEffect(
    () => () => {
      if (timer.current !== null) window.clearTimeout(timer.current);
    },
    [],
  );

  const onCopy = useCallback(() => {
    void copyReportSummary().then((ok) => {
      if (!ok) return;
      setCopied(true);
      if (timer.current !== null) window.clearTimeout(timer.current);
      timer.current = window.setTimeout(() => setCopied(false), 1800);
    });
  }, []);

  return (
    <button
      className={`btn small copysum${copied ? ' copied' : ''}`}
      disabled={!enabled}
      onClick={onCopy}
      title="Copy a one-line summary of this run to the clipboard"
    >
      {copied ? <IconCheck size={14} /> : <IconReveal size={14} />}
      <span>{copied ? 'Copied' : 'Copy summary'}</span>
    </button>
  );
}

export function Actions() {
  const bridge = getBridge();
  const hasResolve = Boolean(bridge.resolve);
  const isWeb = bridge.host === 'web';
  const cleanedPath = useStore((s) => s.cleanedPath);
  const source = useStore((s) => s.source);
  const job = useStore((s) => s.job);
  // What the engine last said it can still serve of this run (state/store.ts).
  const avail = useStore((s) => s.artifacts);
  // `client` is read so the artefact URLs recompute when the engine reconnects.
  useStore((s) => s.client);
  const engineReady = useStore((s) => s.engineStatus === 'ready');
  const masterServed = !!cleanedPath && avail?.master !== false;
  const canReplace = hasResolve && masterServed && source?.origin === 'resolve' && !!source.mediaId;
  // B6 · the files are served *by* the engine. With it gone the paths are still
  // known and still on screen — they are simply not fetchable this second, and
  // saying so beats handing over a link that downloads a connection refusal.
  const artifacts = engineReady ? artifactsFor(cleanedPath, job?.reportPath ?? null, avail) : null;
  const offlineNote =
    'The engine is offline — the file is still on disk and this link comes back when it reconnects.';
  const missingNote = 'Available once a run has finished';
  const note = engineReady ? missingNote : offlineNote;
  const goneNote = avail?.reason || 'This file is no longer where the run left it.';

  if (!hasResolve && isWeb) {
    // A browser cannot reveal files; it downloads them instead. All three come
    // down `/api/audio`, which types each response from the file's extension.
    return (
      <div className="actions artifacts">
        <ArtifactLink
          url={artifacts?.master.url ?? null}
          name={artifacts?.master.name ?? 'master.wav'}
          label="Master WAV"
          title={
            artifacts
              ? (artifacts.master.note ?? `Download ${artifacts.master.name}`)
              : note
          }
        />
        <ArtifactLink
          url={artifacts?.json.url ?? null}
          name={artifacts?.json.name ?? 'report.json'}
          label="JSON report"
          title={artifacts ? (artifacts.json.note ?? `Download ${artifacts.json.name}`) : note}
        />
        <ArtifactLink
          url={artifacts?.txt.url ?? null}
          name={artifacts?.txt.name ?? 'report.txt'}
          label="Summary .txt"
          title={artifacts ? (artifacts.txt.note ?? `Download ${artifacts.txt.name}`) : note}
        />
        <CopySummary />
      </div>
    );
  }

  return (
    <div className="actions">
      {/* `aria-disabled` + a guarded handler, not the native attribute — the
          same reason the artefact links above already give: a natively
          disabled control takes no pointer events, so the `title` that says
          *why* the file is not there can never be shown, and it leaves the
          accessibility tree entirely. Half this file followed that rule and
          half did not, and the half that did not is the half whose reasons
          matter most ("the master this run wrote is no longer on disk"). The
          inline `masterServed` guard stays: importToResolve and revealOutput
          check `cleanedPath` but not availability, so without it the key goes
          live in exactly the file-is-gone state B6 exists for. */}
      {hasResolve ? (
        <>
          <button
            className="btn small"
            type="button"
            aria-disabled={!masterServed || undefined}
            onClick={() => {
              if (!masterServed) return;
              void importToResolve();
            }}
            title={masterServed ? 'Import the cleaned master into the media pool' : cleanedPath ? goneNote : missingNote}
          >
            <IconImport size={14} />
            <span>Import to Resolve</span>
          </button>
          <button
            className="btn small"
            type="button"
            aria-disabled={!canReplace || undefined}
            onClick={() => {
              if (!canReplace) return;
              void replaceInResolve();
            }}
            title={canReplace ? 'Replace the selected clip with the cleaned master' : 'Available when the source came from Resolve'}
          >
            <IconReplace size={14} />
            <span>Replace clip</span>
          </button>
        </>
      ) : null}
      <button
        className="btn small"
        type="button"
        aria-disabled={!masterServed || undefined}
        onClick={() => {
          if (!masterServed) return;
          void saveCleanedMaster();
        }}
        title={masterServed ? 'Save or export the cleaned master audio file' : cleanedPath ? goneNote : missingNote}
      >
        <IconImport size={14} />
        <span>Save Master</span>
      </button>
      <button
        className="btn small"
        type="button"
        aria-disabled={!masterServed || undefined}
        onClick={() => {
          if (!masterServed) return;
          void revealOutput();
        }}
        title={masterServed ? 'Reveal the cleaned master in Finder' : cleanedPath ? goneNote : missingNote}
      >
        <IconReveal size={14} />
        <span>Reveal in Finder</span>
      </button>
      <CopySummary />
    </div>
  );
}
