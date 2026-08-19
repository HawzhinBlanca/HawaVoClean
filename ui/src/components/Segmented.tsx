import { motion } from 'motion/react';
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
        <motion.span
          className={`thumb${thumbClassName ? ` ${thumbClassName}` : ''}`}
          initial={false}
          animate={{ x: thumb.x, width: thumb.w }}
          transition={{ type: 'spring', stiffness: 520, damping: 38, mass: 0.6 }}
          style={{ left: 0 }}
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
