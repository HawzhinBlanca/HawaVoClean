// The app's single live region, and the only thing it says out loud.
//
// It lived inside App.tsx, where nothing could reach it: this is the one
// surface a screen-reader user has for the whole job lifecycle, and it had no
// test. It is state logic, not layout, so it belongs beside the store, the
// keymap and the failure classifier rather than in the file that describes the
// shape of the screen.

import { useEffect, useRef, useState } from 'react';
import { unitsEnhancedSpoken } from './plural';
import { useStore } from './store';

/**
 * D1 · one polite live region for the whole job lifecycle.
 *
 * The screen already *shows* progress in four places (the header readout, the
 * plate's face, the footer line, the source strip), and every one of them is
 * re-rendered on every SSE tick. Wiring any of those up as a live region would
 * announce a new percentage several times a second and make the app unusable
 * with a screen reader on. This announces only the transitions a person would
 * want told: the analysis starting and finishing, the run starting, and how it
 * ended. Same string twice in a row is not re-announced, because the state
 * only changes when it changes.
 */
export function useJobAnnouncer(): string {
  const analyzing = useStore((s) => s.analyzing);
  const sourceName = useStore((s) => s.source?.name ?? null);
  const state = useStore((s) => s.job?.status?.state ?? (s.job ? 'queued' : null));
  const report = useStore((s) => s.report);
  const errCode = useStore((s) => s.job?.status?.error?.code ?? null);
  // The boolean, not the object: `original` is a fresh analysis record on
  // every load, so selecting it would re-run this effect for no state change.
  const hasOriginal = useStore((s) => s.original !== null);
  const loadError = useStore((s) => s.error);
  const deckFault = useStore((s) => s.deckFault);
  const [msg, setMsg] = useState('');
  const saidFault = useRef<string | null>(null);

  useEffect(() => {
    if (analyzing) {
      setMsg(sourceName ? `Analyzing ${sourceName}` : 'Analyzing clip');
      return;
    }
    if (state === 'queued' || state === 'running') {
      setMsg('Processing started');
      return;
    }
    if (state === 'done') {
      const sum = report?.summary;
      setMsg(
        sum
          ? `Processing finished. ${unitsEnhancedSpoken(sum.enhanced ?? 0, sum.units_total ?? 0)}.`
          : 'Processing finished.',
      );
      return;
    }
    if (state === 'failed') {
      setMsg(`Processing failed: ${errCode ?? 'unknown error'}`);
      return;
    }
    if (state === 'cancelled') {
      setMsg('Processing cancelled');
      return;
    }
    if (!sourceName) return;
    // "ready" was said whether or not the analysis produced anything. A file
    // the engine refused, or one whose analysis was cancelled, still cleared
    // `analyzing` with a source name set — so the one region a screen-reader
    // user has announced the opposite of what happened, and the run they went
    // on to start could not work. `original` is what "ready" actually means:
    // the clip was decoded and measured.
    if (hasOriginal) {
      setMsg(`${sourceName} ready`);
      return;
    }
    // Short, and not a duplicate of the role="alert" bar in the footer, which
    // carries the full sentence. This says which clip and that it did not load.
    setMsg(loadError ? `${sourceName} not analyzed: ${loadError}` : `${sourceName} not analyzed`);
  }, [analyzing, sourceName, state, report, errCode, hasOriginal, loadError]);

  // A deck falling out of service changes what the transport can do, so it is
  // told — once, on the transition into the fault, never again while it stands.
  // It rides the same single polite region as everything else: a second live
  // region for one sentence would be a second voice for no gain.
  useEffect(() => {
    const detail = deckFault?.detail ?? null;
    if (!detail) {
      saidFault.current = null;
      return;
    }
    if (detail === saidFault.current) return;
    saidFault.current = detail;
    setMsg(detail);
  }, [deckFault]);

  return msg;
}

