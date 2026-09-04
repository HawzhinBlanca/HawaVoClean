import { useState } from 'react';
import type { JobMode } from '../api/types';
import { getRouteBlockedReason, isRouteBlocked, jobInFlight, useStore } from '../state/store';
import { Segmented } from './Segmented';

/**
 * Restore mode (docs/ui-contract.md, Addendum 2, True-10 D4.11). The control exists only
 * when the engine's health answer says restore is available — an engine with
 * no speaker profiles cannot run a restore job, and a control for it would be
 * a promise the run cannot keep, so the whole block stays out of the tree
 * rather than rendering disabled.
 *
 * Natural is the default and is byte-identical to a revision-1 run; Restore
 * adds a speaker (required by the engine), an optional manual cutoff, and
 * requires explicit generative reconstruction consent.
 */
export function RestoreControl() {
  const available = useStore((s) => s.restoreAvailable);
  const capabilities = useStore((s) => s.capabilities);
  const reconstructionConsent = useStore((s) => s.reconstructionConsent);
  const setReconstructionConsent = useStore((s) => s.setReconstructionConsent);
  const speakers = useStore((s) => s.speakers);
  const mode = useStore((s) => s.mode);
  const speakerId = useStore((s) => s.speakerId);
  const cutoffHz = useStore((s) => s.cutoffHz);
  const original = useStore((s) => s.original);
  const setMode = useStore((s) => s.setMode);
  const setSpeakerId = useStore((s) => s.setSpeakerId);
  const setCutoffHz = useStore((s) => s.setCutoffHz);
  // `jobInFlight`: the window between the engine accepting a job and its first
  // status reads as idle under a hand-written state check.
  const running = useStore((s) => jobInFlight(s.job));
  // The field shows what was typed, the store holds what will be sent: only a
  // positive finite number becomes a manual cutoff, anything else (empty, a
  // half-typed value, zero) means auto-detect. A store-driven value would snap
  // the text back mid-keystroke; local text with a parsed shadow does not.
  const [cutoffText, setCutoffText] = useState(cutoffHz === null ? '' : String(cutoffHz));
  if (!available) return null;

  const restoreCapId = speakerId ? 'restore_enrolled' : 'restore_source';
  const blocked = isRouteBlocked(capabilities, restoreCapId);
  const blockedReason = getRouteBlockedReason(capabilities, restoreCapId);

  // Nothing exists above Nyquist to restore, and nothing rejected a cutoff that
  // sat there: the request schema is `gt=0` with no ceiling and the DSP does
  // `np.clip(cutoff_hz, 0.0, nyquist)`, so a clip asked to restore above 24 kHz
  // at 48 kHz ran a full pass that restored nothing and said so nowhere. The
  // highest integer that still leaves a band is one below Nyquist, and that is
  // also what the input advertises as its `max`, so the browser's idea of valid
  // and this component's are one set rather than two overlapping ones.
  const maxCutoffHz = original ? Math.floor(original.sample_rate / 2) - 1 : null;
  const rateKHz = original ? Math.round(original.sample_rate / 1000) : null;
  const inRange = (v: number): boolean =>
    Number.isFinite(v) && v > 0 && (maxCutoffHz === null || v <= maxCutoffHz);

  const typed = Number.parseFloat(cutoffText);
  const cutoffInvalid = cutoffText.trim() !== '' && cutoffHz === null;
  const outOfRange = cutoffText.trim() !== '' && Number.isFinite(typed) && !inRange(typed);
  const sub =
    mode === 'restore'
      ? blocked
        ? blockedReason ?? 'Restore is not qualified for production'
        : outOfRange && maxCutoffHz !== null
          ? `${Math.round(typed)} Hz leaves nothing above it at ${rateKHz} kHz — auto-detecting instead (highest usable ${maxCutoffHz} Hz)`
          : cutoffHz !== null
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
            {
              value: 'natural',
              label: 'Natural',
              description: 'Clean only — no spectral content is invented',
            },
            {
              value: 'restore',
              label: 'Restore',
              description:
                'HawaRestore-KD rebuilds content above the cutoff, so the output contains audio the source never captured',
            },
          ]}
        />
      </div>
      {mode === 'restore' ? (
        <>
          {blocked ? (
            <div className="rc-blocked-warning" role="alert">
              <span className="rc-badge blocked">BLOCKED</span>
              <span className="rc-blocked-text">
                {blockedReason ?? 'No qualified signed Sorani Restore pack is installed'}
              </span>
            </div>
          ) : null}
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
                step="any"
                {...(maxCutoffHz !== null ? { max: maxCutoffHz } : {})}
                placeholder="auto"
                value={cutoffText}
                disabled={running}
                aria-invalid={cutoffInvalid ? 'true' : undefined}
                title="Restoration boundary in Hz; leave empty to auto-detect"
                onChange={(e) => {
                  const raw = e.target.value;
                  setCutoffText(raw);
                  const v = Number.parseFloat(raw);
                  setCutoffHz(inRange(v) ? v : null);
                }}
              />
              <span className="rc-unit">Hz</span>
            </label>
          </div>
          <label className="rc-consent">
            <input
              type="checkbox"
              checked={reconstructionConsent}
              disabled={running}
              onChange={(e) => setReconstructionConsent(e.target.checked)}
            />
            <span className="rc-consent-text">
              I acknowledge that Restore uses generative reconstruction to rebuild spectral content not captured in source audio.
            </span>
          </label>
        </>
      ) : null}
      <span className="pc-sub">{sub}</span>
    </div>
  );
}
