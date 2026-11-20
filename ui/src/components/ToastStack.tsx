import { useStore } from '../state/store';

export function ToastStack() {
  const toasts = useStore((s) => s.toasts);
  const dismissToast = useStore((s) => s.dismissToast);

  if (toasts.length === 0) return null;

  return (
    <div className="toast-stack" role="region" aria-label="Notifications" aria-live="polite">
      {toasts.map((t) => (
        <div key={t.id} className="toast-item" data-kind={t.kind}>
          <span>{t.text}</span>
          <button
            type="button"
            className="toast-dismiss"
            onClick={() => dismissToast(t.id)}
            aria-label={`Dismiss notification: ${t.text}`}
          >
            ×
          </button>
        </div>
      ))}
    </div>
  );
}
