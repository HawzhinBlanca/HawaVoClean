export type LedState = 'ok' | 'busy' | 'err' | 'off';

export function Led({ state, title }: { state: LedState; title?: string }) {
  const cls = state === 'off' ? 'led' : `led ${state}`;
  return <span className={cls} title={title} aria-hidden="true" />;
}
