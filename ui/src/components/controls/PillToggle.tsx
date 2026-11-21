import type { ReactNode } from 'react';

export interface PillToggleProps {
  label: string;
  pressed: boolean;
  onToggle: () => void;
  disabled?: boolean;
  title?: string;
  icon?: ReactNode;
  ariaLabel?: string;
  className?: string;
}

export function PillToggle({
  label,
  pressed,
  onToggle,
  disabled = false,
  title,
  icon,
  ariaLabel,
  className,
}: PillToggleProps) {
  return (
    <button
      type="button"
      className={className ? `pill-toggle ${className}` : 'pill-toggle'}
      aria-pressed={pressed}
      aria-label={ariaLabel}
      disabled={disabled}
      title={title}
      onClick={onToggle}
    >
      <span className="pill-toggle-dot" aria-hidden="true" />
      {icon ? <span aria-hidden="true">{icon}</span> : null}
      <span>{label}</span>
    </button>
  );
}
