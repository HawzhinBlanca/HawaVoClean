export type LedState = 'ok' | 'busy' | 'err' | 'off';

/**
 * A hardware indicator: a lit die plus a corona. The die is the element (its
 * gradient and bevel live in app.css); the corona is the `.halo` child, which
 * is where every cadence runs — steady when ready, breathing when busy, a
 * two-flash alarm on error. Keeping the cadence off the die means the state
 * change itself can cross-fade instead of snapping, and means the whole
 * animation is transform + opacity, so `prefers-reduced-motion` can remove it
 * without removing the indicator.
 */
export function Led({ state, title }: { state: LedState; title?: string }) {
  const cls = state === 'off' ? 'led' : `led ${state}`;
  return (
    <span className={cls} title={title} aria-hidden="true">
      <span className="halo" />
    </span>
  );
}
