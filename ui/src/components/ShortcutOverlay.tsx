// The keyboard map, on screen (goal box B1). A real dialog: focus moves in on
// open, Tab cannot leave it, Esc or a click outside closes it, and focus goes
// back to whatever the user was on before.

import { useEffect, useRef } from 'react';
import { useStore } from '../state/store';
import { IconCancel } from './Icons';

interface Binding {
  keys: string[][];
  what: string;
  /** Shown dimmed when the binding only applies in a particular state. */
  when?: string;
}

interface Group {
  title: string;
  rows: Binding[];
}

const GROUPS: Group[] = [
  {
    title: 'Transport',
    rows: [
      { keys: [['Space']], what: 'Play / pause' },
      { keys: [['A']], what: 'Listen to the ORIGINAL deck' },
      { keys: [['B']], what: 'Listen to the CLEANED deck', when: 'after a run' },
      { keys: [['←'], ['→']], what: 'Seek ∓ 5 seconds' },
      { keys: [['Shift', '←'], ['Shift', '→']], what: 'Seek ∓ 1 second' },
    ],
  },
  {
    title: 'Units',
    rows: [
      { keys: [['['], [',']], what: 'Select the previous unit' },
      { keys: [[']'], ['.']], what: 'Select the next unit' },
      { keys: [['Esc']], what: 'Clear the selection', when: 'when idle' },
    ],
  },
  {
    title: 'Job',
    rows: [
      { keys: [['P']], what: 'Process the loaded clip', when: 'when idle' },
      { keys: [['Esc']], what: 'Cancel the running job', when: 'while running' },
    ],
  },
  {
    title: 'Help',
    rows: [
      { keys: [['?']], what: 'Open this panel' },
      { keys: [['Esc']], what: 'Close this panel' },
    ],
  },
  {
    title: 'Waveform · pointer',
    rows: [
      { keys: [['Wheel']], what: 'Zoom about the cursor' },
      { keys: [['Shift', 'Wheel']], what: 'Pan the visible window' },
      { keys: [['Drag', 'ruler']], what: 'Pan the visible window' },
      { keys: [['Double', 'click']], what: 'Fit the whole clip' },
      { keys: [['Click', 'segment']], what: 'Select that unit and seek to it' },
    ],
  },
];

const FOCUSABLE = 'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])';

export function ShortcutOverlay() {
  const open = useStore((s) => s.shortcutsOpen);
  const setOpen = useStore((s) => s.setShortcutsOpen);
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const closeRef = useRef<HTMLButtonElement | null>(null);
  const restoreRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!open) return;
    restoreRef.current = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === 'Escape') {
        e.preventDefault();
        e.stopPropagation();
        setOpen(false);
        return;
      }
      if (e.key !== 'Tab') return;
      const root = dialogRef.current;
      if (!root) return;
      const items = [...root.querySelectorAll<HTMLElement>(FOCUSABLE)].filter(
        (el) => !el.hasAttribute('disabled'),
      );
      if (items.length === 0) return;
      const first = items[0] as HTMLElement;
      const last = items[items.length - 1] as HTMLElement;
      const active = document.activeElement as HTMLElement | null;
      if (e.shiftKey && (active === first || !root.contains(active))) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && (active === last || !root.contains(active))) {
        e.preventDefault();
        first.focus();
      }
    };
    // Capture phase: the dialog owns Esc and Tab while it is up, so the global
    // keyboard map never sees them.
    window.addEventListener('keydown', onKey, true);
    return () => {
      window.removeEventListener('keydown', onKey, true);
      const el = restoreRef.current;
      restoreRef.current = null;
      if (el && document.contains(el)) el.focus();
    };
  }, [open, setOpen]);

  if (!open) return null;

  return (
    <div
      className="sc-scrim"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) setOpen(false);
      }}
    >
      <div
        className="panel sc-panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby="sc-title"
        ref={dialogRef}
      >
        <div className="panel-head">
          <div className="panel-title" id="sc-title">
            <span>Keyboard shortcuts</span>
            <span className="sub">· web mode</span>
          </div>
          <button
            type="button"
            className="sc-close"
            onClick={() => setOpen(false)}
            ref={closeRef}
            aria-label="Close shortcuts"
            title="Close (Esc)"
          >
            <kbd>Esc</kbd>
            <IconCancel size={12} />
          </button>
        </div>
        <div className="sc-body">
          {GROUPS.map((g) => (
            <section className="sc-group" key={g.title}>
              <h3 className="caps">{g.title}</h3>
              {g.rows.map((r, i) => (
                <div className="sc-row" key={`${g.title}-${i}`}>
                  <span className="sc-keys">
                    {r.keys.map((combo, j) => (
                      <span className="sc-combo" key={j}>
                        {j > 0 ? <span className="sep">/</span> : null}
                        {combo.map((k, n) => (
                          <span className="kwrap" key={n}>
                            {n > 0 ? <span className="plus">+</span> : null}
                            <kbd>{k}</kbd>
                          </span>
                        ))}
                      </span>
                    ))}
                  </span>
                  <span className="sc-what">
                    {r.what}
                    {r.when ? <em> · {r.when}</em> : null}
                  </span>
                </div>
              ))}
            </section>
          ))}
        </div>
        <div className="sc-foot">
          <span>
            Modifier combinations (<kbd>⌘</kbd>/<kbd>Ctrl</kbd>/<kbd>Alt</kbd>) are left to the
            browser — nothing here hijacks them.
          </span>
        </div>
      </div>
    </div>
  );
}
