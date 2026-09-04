// components/RestorationCard.tsx — the restoration section of a done restore
// run's report, presented as the .txt sidecar presents it. The load-bearing
// behaviour is the honesty rule: Guard R saying FAIL means the Natural master
// shipped — the amber "reverted" language, a sentence saying so, and never
// the error red reserved for a run that actually broke.

import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import type { RestorationSection } from '../api/types';
import { RestorationCard } from './RestorationCard';

declare global {
  var IS_REACT_ACT_ENVIRONMENT: boolean | undefined;
}
globalThis.IS_REACT_ACT_ENVIRONMENT = true;

/** A restoration section as the engine writes it (RestorationReport.to_dict). */
function section(over: Partial<RestorationSection> = {}): RestorationSection {
  return {
    mode: 'restore',
    speaker_id: 'character_01',
    profile_hash: 'p'.repeat(64),
    natural_output_hash: 'n'.repeat(64),
    bandwidth: {
      effective_cutoff_hz: 7800.0,
      confidence: 0.92,
      shape: 'codec_lowpass',
      restore_recommended: true,
      cutoff_mode: 'auto',
      evidence: {
        spectral_rolloff: 7500.0,
        above_cutoff_snr_db: -14.2,
        stationarity: 0.83,
        high_band_energy_ratio_db: -41.6,
      },
    },
    restorer: {
      name: 'hawarestore-kd',
      weights_sha256: 'abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789',
      device: 'cpu',
      solver: 'midpoint',
    },
    segments: { restored: 1, reduced: 0, reverted: 0, bypassed: 0, errors: 0 },
    guard_r: {
      verdict: 'PASS',
      accepted_strength: 0.7,
      reason: 'Accepted strength 0.70: all guard layers passed',
    },
    review_timecodes: [],
    ...over,
  };
}

let host: HTMLElement;
let root: Root;

beforeEach(() => {
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(async () => {
    root.unmount();
  });
  host.remove();
});

async function render(rest: RestorationSection): Promise<void> {
  await act(async () => {
    root.render(createElement(RestorationCard, { rest }));
  });
}

describe('RestorationCard', () => {
  it('states the verdict, the speaker, the cutoff line and the weights prefix', async () => {
    await render(section());
    const pill = host.querySelector('.rest-head .pill');
    expect(pill?.textContent).toBe('PASS');
    expect(pill?.classList.contains('enhanced')).toBe(true);
    expect(host.querySelector('.rest-speaker')?.textContent).toBe('character_01');
    const text = host.textContent ?? '';
    // The cutoff line carries the same facts as the .txt summary's one line.
    expect(text).toContain('7800.0 Hz');
    expect(text).toContain('codec_lowpass');
    expect(text).toContain('conf 0.92');
    expect(text).toContain('SNR above -14.2 dB');
    expect(text).toContain('restored 1');
    // Weights are shown as the .txt shows them: a 16-hex prefix, elided.
    expect(text).toContain('hawarestore-kd · abcdef0123456789…');
    expect(text).toContain('Accepted strength 0.70');
  });

  it('a manual cutoff says it was asserted, not measured', async () => {
    await render(
      section({
        bandwidth: {
          effective_cutoff_hz: 8000.0,
          confidence: 1.0,
          shape: 'manual_override',
          cutoff_mode: 'manual',
        },
      }),
    );
    expect(host.textContent).toContain('8000.0 Hz');
    expect(host.textContent).toContain('· manual');
  });

  it('FAIL is the guard succeeding: amber reverted language, never the error class', async () => {
    await render(
      section({
        segments: { restored: 0, reduced: 0, reverted: 1, bypassed: 0, errors: 0 },
        guard_r: {
          verdict: 'FAIL',
          accepted_strength: 0.0,
          reason: 'All candidate strengths rejected; reverted to Natural-safe audio.',
        },
      }),
    );
    const card = host.querySelector('.rest-card');
    expect(card?.getAttribute('data-verdict')).toBe('reverted');
    const pill = host.querySelector('.rest-head .pill');
    expect(pill?.textContent).toBe('FAIL');
    expect(pill?.classList.contains('reverted')).toBe(true);
    expect(pill?.classList.contains('error')).toBe(false);
    // The one sentence that says what actually shipped.
    expect(host.textContent).toContain('the Natural master shipped');
    expect(host.textContent).toContain('reverted 1');
  });

  it('only the restorer itself breaking earns the error class', async () => {
    await render(
      section({
        segments: { restored: 0, reduced: 0, reverted: 0, bypassed: 0, errors: 1 },
        guard_r: { verdict: 'ERROR', accepted_strength: 0.0, reason: 'restorer raised' },
      }),
    );
    expect(host.querySelector('.rest-head .pill')?.classList.contains('error')).toBe(true);
  });

  it('NO_RESTORE reads as a pass-through, with the guard’s reason on show', async () => {
    await render(
      section({
        segments: { restored: 0, reduced: 0, reverted: 0, bypassed: 1, errors: 0 },
        guard_r: {
          verdict: 'NO_RESTORE',
          accepted_strength: 0.0,
          reason: 'No restoration candidates provided; preserved Natural audio',
        },
      }),
    );
    expect(host.querySelector('.rest-card')?.getAttribute('data-verdict')).toBe('passthrough');
    expect(host.textContent).toContain('preserved Natural audio');
  });

  it('renders Smart Safe routing decision card with candidates and fallback (True-10 D4.11)', async () => {
    await render(
      section({
        mode: 'smart_safe',
        selected_route: 'production',
        confidence: 0.94,
        fallback_route: 'production',
        reason: 'Production Wiener filter selected with high confidence',
        candidates: [
          {
            route: 'production',
            status: 'accepted',
            score: 0.94,
            confidence: 0.94,
            rank: 1,
            selected: true,
            rejection_reason: null,
          },
          {
            route: 'studio',
            status: 'rejected',
            score: 0.72,
            confidence: 0.72,
            rank: 2,
            selected: false,
            rejection_reason: 'Phase coherence margin lower than production',
          },
        ],
      }),
    );
    expect(host.textContent).toContain('Smart Safe Decision');
    expect(host.textContent).toContain('PRODUCTION');
    expect(host.textContent).toContain('94.0%');
    expect(host.textContent).toContain('Candidate Evaluation');
    expect(host.textContent).toContain('Phase coherence margin lower than production');
  });
});
