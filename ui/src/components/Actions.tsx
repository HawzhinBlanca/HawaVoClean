import { getBridge } from '../bridge';
import { baseName, importToResolve, replaceInResolve, revealOutput } from '../state/actions';
import { useStore } from '../state/store';
import { IconImport, IconReplace, IconReveal } from './Icons';

export function Actions() {
  const bridge = getBridge();
  const hasResolve = Boolean(bridge.resolve);
  const isWeb = bridge.host === 'web';
  const cleanedPath = useStore((s) => s.cleanedPath);
  const source = useStore((s) => s.source);
  const client = useStore((s) => s.client);
  const canReplace = hasResolve && !!cleanedPath && source?.origin === 'resolve' && !!source.mediaId;

  if (!hasResolve && isWeb) {
    // A browser cannot reveal files; it can download the master instead.
    const href = cleanedPath && client ? client.audioUrl(cleanedPath) : undefined;
    return (
      <div className="actions">
        {href ? (
          <a
            className="btn small"
            href={href}
            download={baseName(cleanedPath ?? 'master.wav')}
            target="_blank"
            rel="noreferrer"
            title="Download the cleaned master"
          >
            <IconImport size={14} />
            <span>Download master</span>
          </a>
        ) : (
          <button className="btn small" disabled>
            <IconImport size={14} />
            <span>Download master</span>
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="actions">
      {hasResolve ? (
        <>
          <button className="btn small" disabled={!cleanedPath} onClick={() => void importToResolve()} title="Import the cleaned master into the media pool">
            <IconImport size={14} />
            <span>Import to Resolve</span>
          </button>
          <button
            className="btn small"
            disabled={!canReplace}
            onClick={() => void replaceInResolve()}
            title={canReplace ? 'Replace the selected clip with the cleaned master' : 'Available when the source came from Resolve'}
          >
            <IconReplace size={14} />
            <span>Replace clip</span>
          </button>
        </>
      ) : null}
      <button className="btn small" disabled={!cleanedPath} onClick={() => void revealOutput()} title="Reveal the cleaned master in Finder">
        <IconReveal size={14} />
        <span>Reveal in Finder</span>
      </button>
    </div>
  );
}
