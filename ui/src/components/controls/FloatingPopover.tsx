import { useEffect, useRef, type ReactNode } from 'react';

export interface FloatingPopoverProps {
  id: string;
  open: boolean;
  onClose?: () => void;
  ariaLabel: string;
  children: ReactNode;
  className?: string;
  anchored?: boolean;
}

/**
 * Contextual floating action panel using the native Popover API (`popover="auto"`)
 * when supported, with graceful inline-positioned fallback in test/older runtimes.
 */
export function FloatingPopover({
  id,
  open,
  onClose,
  ariaLabel,
  children,
  className,
  anchored = false,
}: FloatingPopoverProps) {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = ref.current as (HTMLDivElement & {
      showPopover?: () => void;
      hidePopover?: () => void;
    }) | null;
    if (!el) return;
    if (typeof el.showPopover !== 'function' || typeof el.hidePopover !== 'function') {
      return;
    }
    try {
      const isOpen = el.matches(':popover-open');
      if (open && !isOpen) el.showPopover();
      else if (!open && isOpen) el.hidePopover();
    } catch {
      // Ignore synthetic DOM environments where popover top-layer isn't wired.
    }
  }, [open]);

  useEffect(() => {
    const el = ref.current;
    if (!el || !onClose) return;
    const onToggle = (e: Event): void => {
      const te = e as Event & { newState?: string };
      if (te.newState === 'closed') onClose();
    };
    el.addEventListener('toggle', onToggle);
    return () => el.removeEventListener('toggle', onToggle);
  }, [onClose]);

  if (!open) return null;

  const classes = ['floating-popover', anchored ? 'anchored-overlay' : '', className ?? '']
    .filter(Boolean)
    .join(' ');

  return (
    <div
      ref={ref}
      id={id}
      role="group"
      aria-label={ariaLabel}
      className={classes}
      {...({ popover: 'auto' } as Record<string, string>)}
    >
      {children}
    </div>
  );
}
