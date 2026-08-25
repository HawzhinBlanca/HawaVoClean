import { useEffect, useRef, useState } from 'react';
import { isTerminal } from './api/sse';
import { Actions } from './components/Actions';
import { EngineBanner } from './components/EngineBanner';
import { Footer } from './components/Footer';
import { Header } from './components/Header';
import { JobHistory } from './components/JobHistory';
import { MetricsTiles } from './components/MetricsTiles';
import { ProcessButton } from './components/ProcessButton';
import { ProfileControl } from './components/ProfileControl';
import { RestoreControl } from './components/RestoreControl';
import { ShortcutOverlay } from './components/ShortcutOverlay';
import { SourceStrip } from './components/SourceStrip';
import { SpectrumDisplay } from './components/SpectrumDisplay';
import { Transport } from './components/Transport';
import { UnitInspector } from './components/UnitInspector';
import { WaveformDisplay } from './components/WaveformDisplay';
import { connectEngine, ingestDataTransfer } from './state/actions';
import { useKeyboardMap } from './state/keymap';
import { unitsEnhancedSpoken } from './state/plural';
import { getState, useStore } from './state/store';

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
function useJobAnnouncer(): string {
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

/** What the whole screen is doing, as one word the stylesheet can key on. */
export type Phase = 'idle' | 'analyzing' | 'ready' | 'running' | 'done' | 'failed';

function usePhase(): Phase {
  const source = useStore((s) => s.source);
  const original = useStore((s) => s.original);
  const analyzing = useStore((s) => s.analyzing);
  const state = useStore((s) => s.job?.status?.state ?? (s.job ? 'queued' : null));
  if (state === 'running' || state === 'queued') return 'running';
  if (analyzing) return 'analyzing';
  if (state === 'failed') return 'failed';
  if (state === 'done') return 'done';
  if (!source || !original) return 'idle';
  return 'ready';
}

/**
 * A drag has to be tracked at the window, not at the drop zone: `dragleave`
 * fires every time the pointer crosses into a child element, so an
 * element-level flag flickers, and a drag that ends outside the window fires
 * no `drop` at all. A depth counter over dragenter/dragleave, reset on drop
 * and on dragend, is the only version that always reverts.
 */
function useDragWatch(): boolean {
  const [over, setOver] = useState(false);
  useEffect(() => {
    let depth = 0;
    const isFiles = (e: DragEvent): boolean =>
      Array.from(e.dataTransfer?.types ?? []).includes('Files');
    const enter = (e: DragEvent): void => {
      if (!isFiles(e)) return;
      depth += 1;
      if (depth === 1) setOver(true);
    };
    const leave = (): void => {
      depth = Math.max(0, depth - 1);
      if (depth === 0) setOver(false);
    };
    const end = (): void => {
      depth = 0;
      setOver(false);
    };
    window.addEventListener('dragenter', enter);
    window.addEventListener('dragleave', leave);
    window.addEventListener('drop', end);
    window.addEventListener('dragend', end);
    return () => {
      window.removeEventListener('dragenter', enter);
      window.removeEventListener('dragleave', leave);
      window.removeEventListener('drop', end);
      window.removeEventListener('dragend', end);
    };
  }, []);
  return over;
}

export default function App() {
  useKeyboardMap();
  const phase = usePhase();
  const announcement = useJobAnnouncer();
  const dragOver = useDragWatch();
  const abMode = useStore((s) => s.abMode);
  const cleanedPath = useStore((s) => s.cleanedPath);
  const engineStatus = useStore((s) => s.engineStatus);
  const offlineSince = useStore((s) => s.engineOfflineSince);
  const deck = abMode === 'cleaned' && cleanedPath ? 'cleaned' : 'original';
  // The banner is a *row* of the shell grid, never an overlay: a bar that
  // floats over the chassis is a bar that can cover a control, and the one
  // moment the user most needs to reach the controls is the moment something
  // has gone wrong.
  const offline =
    engineStatus === 'offline' || (engineStatus === 'connecting' && offlineSince !== null);

  useEffect(() => {
    void connectEngine();
    const prevent = (e: DragEvent): void => {
      e.preventDefault();
    };
    // The whole window highlights on drag-over, so the whole window has to
    // accept the drop — anything else is a target that lies. The source strip
    // handles its own drop first and marks the event handled; this only picks
    // up the ones that landed anywhere else, and it still swallows the default
    // so the shell can never navigate away from the app.
    const onWindowDrop = (e: DragEvent): void => {
      const handled = e.defaultPrevented;
      e.preventDefault();
      if (handled || !e.dataTransfer) return;
      const st = getState();
      const busy =
        st.engineStatus !== 'ready' ||
        Boolean(st.upload) ||
        Boolean(st.job?.status && !isTerminal(st.job.status.state));
      if (busy) return;
      void ingestDataTransfer(e.dataTransfer);
    };
    window.addEventListener('dragover', prevent);
    window.addEventListener('drop', onWindowDrop);
    return () => {
      window.removeEventListener('dragover', prevent);
      window.removeEventListener('drop', onWindowDrop);
    };
  }, []);

  return (
    <div
      className="app"
      data-phase={phase}
      data-deck={deck}
      data-drag={dragOver ? 'true' : 'false'}
      data-engine={engineStatus}
      data-offline={offline ? 'true' : 'false'}
    >
      <Header />
      <EngineBanner />
      <SourceStrip />
      {/* The single place the app speaks. Polite: none of this is urgent
          enough to cut across what the user is already reading. */}
      <p className="sr-only" role="status" aria-live="polite" aria-atomic="true">
        {announcement}
      </p>
      <main className="main">
        <div className="left">
          <WaveformDisplay />
          {/* The inspector answers "why did this unit go that way"; the run
              list answers "which pass am I looking at". They share the bottom
              row because they are the same kind of question about the same
              report, and neither needs the full width. */}
          <div className="deskrow">
            <UnitInspector />
            <JobHistory />
          </div>
        </div>
        <aside className="right" aria-label="Analysis and controls">
          <section className="panel spectrum-panel" aria-label="Spectrum">
            <SpectrumDisplay />
            <MetricsTiles />
          </section>
          <section className="panel controls" aria-label="Processing controls">
            {/* The only panel with no visible title, so it is the only one
                with no heading to land on. It gets a hidden one rather than a
                visible one, because the controls below name themselves and a
                title here would be chrome for its own sake. `aria-label` on
                the section stays: it names the landmark, which is a different
                job from being a stop in the heading list. */}
            <h2 className="sr-only">Processing controls</h2>
            <ProfileControl />
            {/* Renders nothing until the engine's health answer offers
                speaker profiles (contract addendum 2). */}
            <RestoreControl />
            <ProcessButton />
            <Transport />
            <Actions />
          </section>
        </aside>
      </main>
      <Footer />
      <ShortcutOverlay />
    </div>
  );
}
