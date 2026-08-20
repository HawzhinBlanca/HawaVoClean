import { useLayoutEffect, useRef, useState, type KeyboardEvent as ReactKeyboardEvent } from 'react';

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

  /**
   * Arrow keys move the selection and the focus together, wrapping, skipping
   * anything unavailable; Home/End go to the ends. Every consumed key is
   * `preventDefault`ed, which is also what keeps the global map from seeking
   * the transport on the same left/right press — it bails on
   * `defaultPrevented`.
   */
  const onKey = (e: ReactKeyboardEvent<HTMLButtonElement>, index: number): void => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const usable = options.filter((o) => !o.disabled && !disabled);
    if (usable.length === 0) return;
    let next: SegOption<T> | undefined;
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
      for (let n = 1; n <= options.length; n++) {
        const c = options[(index + n) % options.length];
        if (c && !c.disabled && !disabled) {
          next = c;
          break;
        }
      }
    } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
      const len = options.length;
      for (let n = 1; n <= len; n++) {
        const c = options[((index - n) % len + len) % len];
        if (c && !c.disabled && !disabled) {
          next = c;
          break;
        }
      }
    } else if (e.key === 'Home') {
      next = usable[0];
    } else if (e.key === 'End') {
      next = usable[usable.length - 1];
    } else {
      return;
    }
    e.preventDefault();
    if (!next || next.value === value) return;
    onChange(next.value);
    const root = ref.current;
    const btn = root?.querySelector<HTMLButtonElement>(
      `button[data-value="${CSS.escape(next.value)}"]`,
    );
    btn?.focus();
  };

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

  // Whichever option carries the group's single tab stop. Normally the checked
  // one; if that option is itself unavailable (the CLEANED deck before a run)
  // the stop moves to the first usable option, so the group never falls out of
  // the tab order altogether.
  const checked = options.find((o) => o.value === value);
  const tabValue =
    checked && !checked.disabled && !disabled
      ? checked.value
      : (options.find((o) => !o.disabled && !disabled)?.value ?? value);

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
      {options.map((o, i) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={o.value === value}
          data-value={o.value}
          className={`${o.value === value ? 'on' : ''}${o.className ? ` ${o.className}` : ''}`}
          disabled={disabled || o.disabled}
          // D1 · a radiogroup is one tab stop, not one per option: Tab lands on
          // the checked segment and the arrows move between them (WAI-ARIA
          // radio group pattern). Two stops per switch would also have meant
          // the A/B pair and the profile pair between them ate four of the
          // screen's seventeen.
          tabIndex={o.value === tabValue ? 0 : -1}
          onKeyDown={(e) => onKey(e, i)}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
