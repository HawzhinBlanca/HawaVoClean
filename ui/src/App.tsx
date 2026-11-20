import { useEffect, useState } from 'react';
import { getPlayer } from './audio/player';
import { AudioAssistant } from './components/AudioAssistant';
import { EngineBanner } from './components/EngineBanner';
import { Footer } from './components/Footer';
import { Header } from './components/Header';
import { ProWorkspace } from './components/ProWorkspace';
import { SimpleWorkspace } from './components/SimpleWorkspace';
import { SlideOverDrawer } from './components/SlideOverDrawer';
import { ShortcutOverlay } from './components/ShortcutOverlay';
import { SourceStrip } from './components/SourceStrip';
import { ToastStack } from './components/ToastStack';
import { VideoPreview, sourceHasVideo } from './components/VideoPreview';
import {
  installRendererProofResponder,
  RENDERER_PROOF_CONTRACT,
} from './rendererProof';
import {
  connectEngine,
  ingestDataTransfer,
  seekTo,
} from './state/actions';
import { useJobAnnouncer } from './state/announcer';
import { useKeyboardMap } from './state/keymap';
import { getState, jobInFlight, useStore } from './state/store';

/** What the whole screen is doing, as one word the stylesheet can key on. */
export type Phase = 'idle' | 'analyzing' | 'ready' | 'running' | 'done' | 'failed';

function usePhase(): Phase {
  const source = useStore((s) => s.source);
  const original = useStore((s) => s.original);
  const analyzing = useStore((s) => s.analyzing);
  const state = useStore((s) => s.job?.status?.state ?? (s.job ? 'queued' : null));
  if (state === 'running' || state === 'queued') return 'running';
  if (analyzing) return 'analyzing';
  if (state === 'failed' || state === 'interrupted') return 'failed';
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

function SyncedVideoPreview() {
  const source = useStore((s) => s.source);
  const original = useStore((s) => s.original);
  const client = useStore((s) => s.client);
  const duration = useStore((s) => s.duration);
  const collapsed = useStore((s) => s.videoPreviewCollapsed);
  const setCollapsed = useStore((s) => s.setVideoPreviewCollapsed);
  const [sync, setSync] = useState(() => {
    const p = getPlayer();
    return { time: p.time, playing: p.playing, duration: p.duration };
  });

  const isVideo = Boolean(source && sourceHasVideo(source.path, original));
  const videoUrl = isVideo && source && client ? client.fileUrl(source.path) : null;

  useEffect(() => {
    if (!videoUrl) return;
    return getPlayer().subscribe((snap) => {
      setSync((prev) =>
        Math.abs(prev.time - snap.time) > 0.02 ||
        prev.playing !== snap.playing ||
        prev.duration !== snap.duration
          ? { time: snap.time, playing: snap.playing, duration: snap.duration }
          : prev,
      );
    });
  }, [videoUrl]);

  if (!videoUrl || !source) return null;

  return (
    <div className="app-video-pip">
      <VideoPreview
        videoUrl={videoUrl}
        currentTime={sync.time}
        playing={sync.playing}
        duration={sync.duration > 0 ? sync.duration : duration}
        onSeek={seekTo}
        collapsed={collapsed}
        onToggleCollapse={() => setCollapsed(!collapsed)}
        title={source.name}
      />
    </div>
  );
}

export default function App() {
  useKeyboardMap();
  const phase = usePhase();
  const announcement = useJobAnnouncer();
  const dragOver = useDragWatch();
  const abMode = useStore((s) => s.abMode);
  const cleanedPath = useStore((s) => s.cleanedPath);
  const previewActive = useStore((s) => s.previewActive);
  const engineStatus = useStore((s) => s.engineStatus);
  const offlineSince = useStore((s) => s.engineOfflineSince);
  const workspaceMode = useStore((s) => s.workspaceMode);
  const assistantDrawerOpen = useStore((s) => s.assistantDrawerOpen);
  const setAssistantDrawerOpen = useStore((s) => s.setAssistantDrawerOpen);
  const deck = abMode === 'cleaned' && (cleanedPath || previewActive) ? 'cleaned' : 'original';
  // The banner is a *row* of the shell grid, never an overlay: a bar that
  // floats over the chassis is a bar that can cover a control, and the one
  // moment the user most needs to reach the controls is the moment something
  // has gone wrong.
  const offline =
    engineStatus === 'offline' || (engineStatus === 'connecting' && offlineSince !== null);

  // The desktop release gate challenges this listener from the real packaged
  // BrowserWindow.  A successful response proves that the production React
  // bundle reached an App commit; loading index.html and preload alone is not
  // enough release evidence.
  useEffect(() => installRendererProofResponder(__UI_VERSION__), []);

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
        // Not `st.job?.status && !isTerminal(...)`: a job whose first status
        // has not arrived yet has `status: null`, which that spelling reads as
        // idle — so a file dropped in that window sailed past this guard and
        // orphaned the run the engine had just accepted.
        jobInFlight(st.job);
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
      data-hawa-renderer-contract={RENDERER_PROOF_CONTRACT}
      data-hawa-ui-version={__UI_VERSION__}
      data-phase={phase}
      data-deck={deck}
      data-drag={dragOver ? 'true' : 'false'}
      data-engine={engineStatus}
      data-offline={offline ? 'true' : 'false'}
      data-workspace={workspaceMode}
    >
      <Header />
      <EngineBanner />
      <SourceStrip />
      {/* The single place the app speaks. Polite: none of this is urgent
          enough to cut across what the user is already reading. */}
      <p className="sr-only" role="status" aria-live="polite" aria-atomic="true">
        {announcement}
      </p>
      {workspaceMode === 'simple' ? (
        <SimpleWorkspace />
      ) : (
        <>
          <SyncedVideoPreview />
          <ProWorkspace />
        </>
      )}
      <Footer />
      <SlideOverDrawer
        open={assistantDrawerOpen}
        title="AI Audio Assistant"
        onClose={() => setAssistantDrawerOpen(false)}
      >
        <AudioAssistant />
      </SlideOverDrawer>
      <ToastStack />
      <ShortcutOverlay />
    </div>
  );
}
