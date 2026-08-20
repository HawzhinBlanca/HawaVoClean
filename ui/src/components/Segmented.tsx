import { useLayoutEffect, useRef, useState } from 'react';

export interface SegOption<T extends string> {
  value: T;
  label: string;
  className?: string;
  disabled?: boolean;
}

interface Props<T extends string> {
  value: T;
  options: SegOption<T>[];
  onChange: (v: T) => void;
  className?: string;
  thumbClassName?: string;
  disabled?: boolean;
  ariaLabel: string;
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  className,
  thumbClassName,
  disabled,
  ariaLabel,
}: Props<T>) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [thumb, setThumb] = useState<{ x: number; w: number } | null>(null);

  useLayoutEffect(() => {
    const root = ref.current;
    if (!root) return;
    const measure = (): void => {
      const btn = root.querySelector<HTMLButtonElement>(`button[data-value="${CSS.escape(value)}"]`);
      if (!btn) return;
      setThumb({ x: btn.offsetLeft, w: btn.offsetWidth });
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(root);
    return () => ro.disconnect();
  }, [value, options.length]);

  return (
    <div ref={ref} className={`seg${className ? ` ${className}` : ''}`} role="radiogroup" aria-label={ariaLabel}>
      {thumb ? (
        // The thumb's resting place is written straight into the style and
        // CSS carries it there (interaction.css). A frame-loop animation was
        // wrong for this one control: it is the only thing that says which
        // segment is on, and a loop that stops — a hidden tab, a reduced-motion
        // preference — would leave it parked between the two, reading as
        // neither. A committed style is always correct; the travel is a
        // bonus the compositor adds when it can.
        <span
          className={`thumb${thumbClassName ? ` ${thumbClassName}` : ''}`}
          style={{ left: 0, width: thumb.w, transform: `translateX(${thumb.x}px)` }}
        />
      ) : null}
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={o.value === value}
          data-value={o.value}
          className={`${o.value === value ? 'on' : ''}${o.className ? ` ${o.className}` : ''}`}
          disabled={disabled || o.disabled}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
