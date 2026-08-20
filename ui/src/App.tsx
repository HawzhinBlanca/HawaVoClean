import { useEffect, useState } from 'react';
import { isTerminal } from './api/sse';
import { getPlayer } from './audio/player';
import { Actions } from './components/Actions';
import { Footer } from './components/Footer';
import { Header } from './components/Header';
import { MetricsTiles } from './components/MetricsTiles';
import { ProcessButton } from './components/ProcessButton';
import { ProfileControl } from './components/ProfileControl';
import { ShortcutOverlay } from './components/ShortcutOverlay';
import { SourceStrip } from './components/SourceStrip';
import { SpectrumDisplay } from './components/SpectrumDisplay';
import { Transport } from './components/Transport';
import { UnitInspector } from './components/UnitInspector';
import { WaveformDisplay } from './components/WaveformDisplay';
import { cancelJob, connectEngine, seekTo, setAb, startJob, togglePlay } from './state/actions';
import { clearSelection, stepUnit } from './state/selection';
import { getState, useStore } from './state/store';

const SEEK_S = 5;
const SEEK_FINE_S = 1;

/** Typing must never trigger transport keys. */
function isEditable(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  if (!el || typeof el.tagName !== 'string') return false;
  const tag = el.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return true;
  return el.isContentEditable === true;
}

/** Space on a focused control belongs to that control, not to the transport. */
function isActivatable(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  if (!el || typeof el.tagName !== 'string') return false;
  return el.tagName === 'BUTTON' || el.tagName === 'A' || el.getAttribute('role') === 'button';
}

/**
 * The whole keyboard map (goal box B1) in one place, so no two components can
 * claim the same key. Nothing here fires with Cmd/Ctrl/Alt held — those belong
 * to the browser (⌘A still selects) — and nothing fires while a text field has
 * focus or while the shortcut dialog is up.
 */
function useKeyboardMap(): void {
  useEffect(() => {
    const onKey = (e: KeyboardEvent): void => {
      if (e.defaultPrevented) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (isEditable(e.target)) return;
      const st = getState();

      if (e.key === '?') {
        e.preventDefault();
        st.setShortcutsOpen(!st.shortcutsOpen);
        return;
      }
      // While the dialog is open it owns the keyboard (it handles Esc/Tab itself).
      if (st.shortcutsOpen) return;

      const hasAudio = Boolean(st.original);
      const job = st.job;
      const running = Boolean(job) && (!job?.status || !isTerminal(job.status.state));

      switch (e.key) {
        case 'Escape': {
          e.preventDefault();
          if (running) void cancelJob();
          else if (st.selectedUnit) clearSelection();
          return;
        }
        case 'ArrowLeft':
        case 'ArrowRight': {
          if (!hasAudio) return;
          e.preventDefault();
          const step = e.shiftKey ? SEEK_FINE_S : SEEK_S;
          const dir = e.key === 'ArrowRight' ? 1 : -1;
          seekTo(Math.max(0, getPlayer().time + dir * step));
          return;
        }
        case '[':
        case ',': {
          e.preventDefault();
          stepUnit(-1);
          return;
        }
        case ']':
        case '.': {
          e.preventDefault();
          stepUnit(1);
          return;
        }
        default:
          break;
      }

      if (e.code === 'Space' || e.key === ' ') {
        if (isActivatable(e.target)) return;
        e.preventDefault();
        if (hasAudio) togglePlay();
        return;
      }

      switch (e.key.toLowerCase()) {
        case 'a':
          if (hasAudio) setAb('original');
          break;
        case 'b':
          if (st.cleanedPath) setAb('cleaned');
          break;
        case 'p': {
          const canStart =
            st.engineStatus === 'ready' && !!st.source && !!st.original && !st.analyzing && !running;
          if (canStart) void startJob();
          break;
        }
        default:
          break;
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
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
  const dragOver = useDragWatch();
  const abMode = useStore((s) => s.abMode);
  const cleanedPath = useStore((s) => s.cleanedPath);
  const deck = abMode === 'cleaned' && cleanedPath ? 'cleaned' : 'original';

  useEffect(() => {
    void connectEngine();
    // Swallow drops outside the drop zone so the shell never navigates away.
    const prevent = (e: DragEvent): void => {
      e.preventDefault();
    };
    window.addEventListener('dragover', prevent);
    window.addEventListener('drop', prevent);
    return () => {
      window.removeEventListener('dragover', prevent);
      window.removeEventListener('drop', prevent);
    };
  }, []);

  return (
    <div className="app" data-phase={phase} data-deck={deck} data-drag={dragOver ? 'true' : 'false'}>
      <Header />
      <SourceStrip />
      <main className="main">
        <div className="left">
          <WaveformDisplay />
          <UnitInspector />
        </div>
        <aside className="right">
          <section className="panel spectrum-panel">
            <SpectrumDisplay />
            <MetricsTiles />
          </section>
          <section className="panel controls">
            <ProfileControl />
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
