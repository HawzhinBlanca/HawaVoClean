import { useState } from 'react';
import type { JobMode } from '../api/types';
import { useStore } from '../state/store';
import { Segmented } from './Segmented';

/**
 * Restore mode (docs/ui-contract.md, Addendum 2). The control exists only
 * when the engine's health answer says restore is available — an engine with
 * no speaker profiles cannot run a restore job, and a control for it would be
 * a promise the run cannot keep, so the whole block stays out of the tree
 * rather than rendering disabled.
 *
 * Natural is the default and is byte-identical to a revision-1 run; Restore
 * adds a speaker (required by the engine) and an optional manual cutoff.
 * The store guarantees a speaker is selected whenever the engine offers any
 * (`setCapabilities`), so switching to Restore is always submittable.
 */
export function RestoreControl() {
  const available = useStore((s) => s.restoreAvailable);
  const speakers = useStore((s) => s.speakers);
  const mode = useStore((s) => s.mode);
  const speakerId = useStore((s) => s.speakerId);
  const cutoffHz = useStore((s) => s.cutoffHz);
  const setMode = useStore((s) => s.setMode);
  const setSpeakerId = useStore((s) => s.setSpeakerId);
  const setCutoffHz = useStore((s) => s.setCutoffHz);
  const running = useStore(
    (s) => !!s.job?.status && (s.job.status.state === 'running' || s.job.status.state === 'queued'),
  );
  // The field shows what was typed, the store holds what will be sent: only a
  // positive finite number becomes a manual cutoff, anything else (empty, a
  // half-typed value, zero) means auto-detect. A store-driven value would snap
  // the text back mid-keystroke; local text with a parsed shadow does not.
  const [cutoffText, setCutoffText] = useState(cutoffHz === null ? '' : String(cutoffHz));
  if (!available) return null;

  const cutoffInvalid = cutoffText.trim() !== '' && cutoffHz === null;
  const sub =
    mode === 'restore'
      ? cutoffHz !== null
        ? `HawaRestore-KD above ${cutoffHz} Hz (manual cutoff)`
        : 'HawaRestore-KD above the auto-detected cutoff'
      : 'Clean only — no spectral content is invented';

  return (
    <div className="restorectl">
      <div className="pc-row">
        <span className="caps">Mode</span>
        <Segmented<JobMode>
          ariaLabel="Mode"
          value={mode}
          disabled={running}
          disabledReason="The mode is part of the run — cancel it to change this."
          onChange={setMode}
          options={[
            { value: 'natural', label: 'Natural' },
            { value: 'restore', label: 'Restore' },
          ]}
        />
      </div>
      {mode === 'restore' ? (
        <div className="rc-fields">
          <label className="rc-field">
            <span className="caps">Speaker</span>
            <select
              value={speakerId ?? ''}
              disabled={running}
              onChange={(e) => setSpeakerId(e.target.value || null)}
            >
              {speakers.map((id) => (
                <option key={id} value={id}>
                  {id}
                </option>
              ))}
            </select>
          </label>
          <label className="rc-field">
            <span className="caps">Cutoff</span>
            <input
              type="number"
              inputMode="decimal"
              min={1}
              step={100}
              placeholder="auto"
              value={cutoffText}
              disabled={running}
              aria-invalid={cutoffInvalid ? 'true' : undefined}
              title="Restoration boundary in Hz; leave empty to auto-detect"
              onChange={(e) => {
                const raw = e.target.value;
                setCutoffText(raw);
                const v = Number.parseFloat(raw);
                setCutoffHz(Number.isFinite(v) && v > 0 ? v : null);
              }}
            />
            <span className="rc-unit">Hz</span>
          </label>
        </div>
      ) : null}
      <span className="pc-sub">{sub}</span>
    </div>
  );
}
