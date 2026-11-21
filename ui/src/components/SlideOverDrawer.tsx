import { useEffect, type ReactNode } from 'react';

export interface SlideOverDrawerProps {
  open: boolean;
  title: string;
  onClose: () => void;
  children?: ReactNode;
}

export function SlideOverDrawer({ open, title, onClose, children }: SlideOverDrawerProps) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="drawer-backdrop"
      role="presentation"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="drawer-sheet" role="dialog" aria-modal="true" aria-label={title}>
        <div className="drawer-head">
          <h2 className="drawer-title">{title}</h2>
          <button
            type="button"
            className="drawer-close"
            onClick={onClose}
            aria-label={`Close ${title}`}
            title="Close (Esc)"
          >
            ×
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}
