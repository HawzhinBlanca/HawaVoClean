import { useRef, type KeyboardEvent as ReactKeyboardEvent } from 'react';
import type { HawaVoCleanReport, Profile } from '../api/types';
import '../styles/studio-rack.css';

export const MAX_COMPARE_TAKES = 3;
export const TAKE_SLOT_LABELS = ['Take A', 'Take B', 'Take C'] as const;

export interface TakeSnapshot {
  id: string;
  label: string;
  profile: Profile;
  cleanedPath: string;
  report: HawaVoCleanReport | null;
  mosDelta: number | null;
  lufsOut: number | null;
  pinned?: boolean | undefined;
  createdAt?: string | undefined;
}

export type ActiveTakeId = 'original' | string;

export interface TakeSwitcherProps {
  takes: readonly TakeSnapshot[];
  activeId: ActiveTakeId;
  onSelect: (id: ActiveTakeId) => void;
  onTogglePin?: ((takeId: string) => void) | undefined;
  onRemoveTake?: ((takeId: string) => void) | undefined;
  disabled?: boolean | undefined;
  className?: string | undefined;
}

const PROFILE_SHORT_LABEL: Record<Profile, string> = {
  studio_pro: 'Studio Pro',
  dereverb: 'Dereverb',
  studio: 'Studio',
  lowband: 'Low-band',
  production: 'Production',
};

/**
 * Pure helper for adding a completed take into the bounded 3-take compare ring.
 * - Replaces in place if a take with the same `id` already exists.
 * - Preserves pinned takes when evicting to stay within `maxTakes`.
 */
export function addTakeSnapshot(
  takes: readonly TakeSnapshot[],
  next: TakeSnapshot,
  maxTakes: number = MAX_COMPARE_TAKES,
): TakeSnapshot[] {
  const existingIdx = takes.findIndex((t) => t.id === next.id);
  if (existingIdx >= 0) {
    const updated = [...takes];
    updated[existingIdx] = next;
    return updated;
  }

  if (takes.length < maxTakes) {
    return [...takes, next];
  }

  const unpinnedIdx = takes.findIndex((t) => !t.pinned);
  if (unpinnedIdx === -1) {
    // All slots are pinned; keep existing takes unchanged so pinned takes are never lost.
    return [...takes];
  }

  const remaining = [...takes.slice(0, unpinnedIdx), ...takes.slice(unpinnedIdx + 1)];
  return [...remaining, next].slice(-maxTakes);
}

function formatMosDelta(delta: number | null): { text: string; polarity: 'positive' | 'negative' | 'neutral' } | null {
  if (delta === null || !Number.isFinite(delta)) return null;
  const rounded = Math.round(delta * 100) / 100;
  const sign = rounded > 0 ? '+' : '';
  const polarity = rounded > 0 ? 'positive' : rounded < 0 ? 'negative' : 'neutral';
  return { text: `${sign}${rounded.toFixed(2)} MOS`, polarity };
}

function formatLufs(lufs: number | null): string | null {
  if (lufs === null || !Number.isFinite(lufs)) return null;
  return `${lufs.toFixed(1)} LUFS`;
}

export function TakeSwitcher({
  takes,
  activeId,
  onSelect,
  onTogglePin,
  onRemoveTake,
  disabled = false,
  className,
}: TakeSwitcherProps) {
  const stripRef = useRef<HTMLDivElement | null>(null);
  const visibleTakes = takes.slice(0, MAX_COMPARE_TAKES);

  const handleStripKeyDown = (e: ReactKeyboardEvent<HTMLButtonElement>): void => {
    if (disabled || e.metaKey || e.ctrlKey || e.altKey) return;
    const container = stripRef.current;
    if (!container) return;
    const buttons = Array.from(
      container.querySelectorAll<HTMLButtonElement>('button[data-take-selector="true"]:not(:disabled)'),
    );
    if (buttons.length === 0) return;

    const currentIdx = buttons.indexOf(e.currentTarget);
    if (currentIdx === -1) return;

    let nextBtn: HTMLButtonElement | undefined;
    if (e.key === 'ArrowRight') {
      nextBtn = buttons[(currentIdx + 1) % buttons.length];
    } else if (e.key === 'ArrowLeft') {
      nextBtn = buttons[(currentIdx - 1 + buttons.length) % buttons.length];
    } else if (e.key === 'Home') {
      nextBtn = buttons[0];
    } else if (e.key === 'End') {
      nextBtn = buttons[buttons.length - 1];
    } else {
      return;
    }

    if (!nextBtn) return;
    e.preventDefault();
    const targetId = nextBtn.dataset['takeId'];
    if (targetId) {
      onSelect(targetId);
    }
    nextBtn.focus();
  };

  return (
    <div
      ref={stripRef}
      className={`take-switcher${className ? ` ${className}` : ''}`}
      role="group"
      aria-label="Multi-take comparison switcher"
    >
      <div
        className="ts-slot"
        data-kind="original"
        data-active={activeId === 'original' ? 'true' : 'false'}
      >
        <button
          type="button"
          className="ts-select-btn"
          data-take-selector="true"
          data-take-id="original"
          aria-pressed={activeId === 'original'}
          disabled={disabled}
          onKeyDown={handleStripKeyDown}
          onClick={() => onSelect('original')}
        >
          <span className="ts-slot-tag">Orig</span>
        </button>
      </div>

      {visibleTakes.map((take, index) => {
        const slotLabel = TAKE_SLOT_LABELS[index] ?? `Take ${index + 1}`;
        const isActive = activeId === take.id;
        const mos = formatMosDelta(take.mosDelta);
        const lufs = formatLufs(take.lufsOut);
        const displayName = take.label.trim() || slotLabel;
        const showSlotPrefix = displayName.toLowerCase() !== slotLabel.toLowerCase();

        return (
          <div
            key={take.id}
            className="ts-slot"
            data-kind="take"
            data-active={isActive ? 'true' : 'false'}
            data-pinned={take.pinned ? 'true' : 'false'}
          >
            <button
              type="button"
              className="ts-select-btn"
              data-take-selector="true"
              data-take-id={take.id}
              aria-pressed={isActive}
              disabled={disabled}
              title={take.cleanedPath}
              onKeyDown={handleStripKeyDown}
              onClick={() => onSelect(take.id)}
            >
              <span className="ts-slot-tag">{slotLabel}</span>
              {showSlotPrefix ? <span className="ts-label-text">{displayName}</span> : null}
              <span className="ts-profile-badge">
                {PROFILE_SHORT_LABEL[take.profile] ?? take.profile}
              </span>
              {mos ? (
                <span className="ts-metric" data-polarity={mos.polarity}>
                  {mos.text}
                </span>
              ) : null}
              {lufs ? <span className="ts-metric">{lufs}</span> : null}
            </button>

            {onTogglePin || onRemoveTake ? (
              <div className="ts-actions">
                {onTogglePin ? (
                  <button
                    type="button"
                    className="ts-icon-btn"
                    aria-label={take.pinned ? `Unpin ${slotLabel}` : `Pin ${slotLabel}`}
                    aria-pressed={Boolean(take.pinned)}
                    disabled={disabled}
                    title={take.pinned ? 'Unpin take' : 'Pin take to protect from overwrite'}
                    onClick={() => onTogglePin(take.id)}
                  >
                    <span aria-hidden="true">{take.pinned ? '★' : '☆'}</span>
                  </button>
                ) : null}
                {onRemoveTake ? (
                  <button
                    type="button"
                    className="ts-icon-btn"
                    aria-label={`Remove ${slotLabel}`}
                    disabled={disabled || Boolean(take.pinned)}
                    title={take.pinned ? 'Unpin before removing' : `Remove ${slotLabel}`}
                    onClick={() => {
                      if (!take.pinned) {
                        onRemoveTake(take.id);
                      }
                    }}
                  >
                    <span aria-hidden="true">×</span>
                  </button>
                ) : null}
              </div>
            ) : null}
          </div>
        );
      })}

      {visibleTakes.length === 0 ? (
        <span className="ts-empty-hint">Run cleanup to compare Take A–C</span>
      ) : null}
    </div>
  );
}
