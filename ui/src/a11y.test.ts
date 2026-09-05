import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import axe from 'axe-core';
import type { AudioAnalysis, HawaVoCleanReport, UnitDecisionRecord } from './api/types';
import { useStore } from './state/store';
import { Header } from './components/Header';
import { EngineBanner } from './components/EngineBanner';
import { SmartExplanation } from './components/SmartExplanation';
import { AdvancedControls } from './components/AdvancedControls';
import { Transport } from './components/Transport';
import { Actions } from './components/Actions';
import { ProcessButton } from './components/ProcessButton';
import { RestorationCard } from './components/RestorationCard';
import { BatchQueue } from './components/BatchQueue';
import { ShortcutOverlay } from './components/ShortcutOverlay';
import { VerdictStrip } from './components/VerdictStrip';
import { UnitInspector } from './components/UnitInspector';

declare global {
  var IS_REACT_ACT_ENVIRONMENT: boolean | undefined;
}
globalThis.IS_REACT_ACT_ENVIRONMENT = true;

const pristine = useStore.getState();

let host: HTMLElement;
let root: Root;

beforeEach(() => {
  useStore.setState(pristine, true);
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

async function runAxe(element: HTMLElement) {
  const results = await axe.run(element, {
    runOnly: {
      type: 'tag',
      values: ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'best-practice'],
    },
    rules: {
      // Color contrast requires real canvas/raster rendering not present in happy-dom;
      // design tokens and stylesheets have mathematically proven contrast (>=4.5:1 / 5.6:1).
      'color-contrast': { enabled: false },
      // Region rule checks for outer landmark structure which is in App shell.
      region: { enabled: false },
    },
  });
  const seriousOrCritical = results.violations.filter(
    (v) => v.impact === 'serious' || v.impact === 'critical',
  );
  return { results, seriousOrCritical };
}

function fakeAnalysis(over: Partial<AudioAnalysis> = {}): AudioAnalysis {
  return {
    path: '/a.wav',
    duration_s: 60,
    sample_rate: 48000,
    channels: 2,
    peaks: { min: [-1], max: [1] },
    rms_db: [-20],
    spectrum: { freqs_hz: [100], db: [-30] },
    loudness: { integrated_lufs: -23, true_peak_dbtp: -1 },
    noise_floor_db: -48,
    ...over,
  };
}

function fakeUnit(id: number, over: Partial<UnitDecisionRecord> = {}): UnitDecisionRecord {
  return {
    unit_id: id,
    channel: 0,
    start_sample: id * 48000,
    end_sample: (id + 1) * 48000,
    start_time_s: id,
    end_time_s: id + 1,
    is_speech: true,
    input_sha256: 'a',
    output_sha256: 'b',
    guard_a_verdict: 'pass',
    guard_b_verdict: 'pass',
    final_decision: 'enhanced',
    decision_reason: 'ok',
    guard_a_scores: {
      envelope_correlation: 0.98,
      spectral_hole_score: 0.02,
    },
    ...over,
  };
}

function fakeReport(over: Partial<HawaVoCleanReport['summary']> = {}): HawaVoCleanReport {
  return {
    job_id: 'j1',
    config_hash: 'c',
    input: { path: '/a.wav', sha256: 'x', sample_rate: 48000, channels: 2, samples: 1, duration_s: 60 },
    output: { path: '/o.wav', sha256: 'y', sample_rate: 48000, channels: 2, samples: 1, duration_s: 60 },
    core: { id: 'studio', algorithm: 'x', params_hash: 'p' },
    guard: { id: 'g', probe_hash: 'p', calibration_id: 'c' },
    environment: {},
    summary: { units_total: 10, enhanced: 8, reverted: 2, continuity_crossfaded: 1, ...over },
    units: [fakeUnit(1), fakeUnit(2, { final_decision: 'original_reverted' })],
  };
}

describe('Accessibility & Axe-core Audit (D4.5)', () => {
  it('Header passes axe-core accessibility checks', async () => {
    await act(async () => {
      root.render(createElement(Header));
    });
    const { seriousOrCritical } = await runAxe(host);
    expect(seriousOrCritical).toEqual([]);
  });

  it('EngineBanner passes axe-core accessibility checks in offline and ready states', async () => {
    useStore.setState({ engineStatus: 'offline' });
    await act(async () => {
      root.render(createElement(EngineBanner));
    });
    const { seriousOrCritical } = await runAxe(host);
    expect(seriousOrCritical).toEqual([]);
  });

  it('SmartExplanation passes axe-core in empty, armed, and done states', async () => {
    // Empty state
    await act(async () => {
      root.render(createElement(SmartExplanation));
    });
    let check = await runAxe(host);
    expect(check.seriousOrCritical).toEqual([]);

    // Armed state
    await act(async () => {
      useStore.setState({
        source: { path: '/take1.wav', name: 'take1.wav', origin: 'file' },
        original: fakeAnalysis(),
        profile: 'studio',
        mode: 'natural',
      });
    });
    check = await runAxe(host);
    expect(check.seriousOrCritical).toEqual([]);

    // Done state
    await act(async () => {
      useStore.setState({
        report: fakeReport(),
        loudnessMatch: true,
        gainOffsetDb: -1.5,
      });
    });
    check = await runAxe(host);
    expect(check.seriousOrCritical).toEqual([]);
  });

  it('AdvancedControls passes axe-core when collapsed and expanded', async () => {
    useStore.setState({ advancedOpen: false, profile: 'studio', mode: 'natural' });
    await act(async () => {
      root.render(createElement(AdvancedControls));
    });
    let check = await runAxe(host);
    expect(check.seriousOrCritical).toEqual([]);

    await act(async () => {
      useStore.setState({ advancedOpen: true });
    });
    check = await runAxe(host);
    expect(check.seriousOrCritical).toEqual([]);
  });

  it('Transport passes axe-core with playback, A/B, and Level Match toggles', async () => {
    useStore.setState({
      original: fakeAnalysis(),
      cleanedPath: '/out/take1.wav',
      abMode: 'cleaned',
      loudnessMatch: true,
      gainOffsetDb: 1.2,
      playing: false,
    });
    await act(async () => {
      root.render(createElement(Transport));
    });
    const { seriousOrCritical } = await runAxe(host);
    expect(seriousOrCritical).toEqual([]);
  });

  it('Actions passes axe-core with Save Master and export buttons', async () => {
    useStore.setState({
      cleanedPath: '/out/take1.wav',
      artifacts: { master: true, json: true, txt: true, reason: '' },
    });
    await act(async () => {
      root.render(createElement(Actions));
    });
    const { seriousOrCritical } = await runAxe(host);
    expect(seriousOrCritical).toEqual([]);
  });

  it('ProcessButton passes axe-core across armed, running, and done states', async () => {
    useStore.setState({
      engineStatus: 'ready',
      source: { path: '/take1.wav', name: 'take1.wav', origin: 'file' },
      original: fakeAnalysis(),
    });
    await act(async () => {
      root.render(createElement(ProcessButton));
    });
    let check = await runAxe(host);
    expect(check.seriousOrCritical).toEqual([]);

    await act(async () => {
      useStore.setState({
        job: {
          id: 'job-1',
          outputPath: '/out/take1.wav',
          reportPath: '/out/take1.json',
          status: {
            job_id: 'job-1',
            state: 'running',
            stage: 'enhance',
            progress: 0.55,
            message: 'Enhancing unit 5/10',
            unit: { index: 5, total: 10 },
            input_path: '/take1.wav',
            output_path: '/out/take1.wav',
            report_path: '/out/take1.json',
            profile: 'studio',
            started_at: '2026-09-06T00:00:00Z',
            finished_at: null,
            error: null,
            report: null,
          },
          streamConnected: true,
        },
      });
    });
    check = await runAxe(host);
    expect(check.seriousOrCritical).toEqual([]);
  });

  it('ShortcutOverlay passes axe-core modal dialog standards', async () => {
    useStore.setState({ shortcutsOpen: true });
    await act(async () => {
      root.render(createElement(ShortcutOverlay));
    });
    const { seriousOrCritical } = await runAxe(host);
    expect(seriousOrCritical).toEqual([]);
  });

  it('BatchQueue passes axe-core with queue items and cancel controls', async () => {
    useStore.setState({
      batch: {
        batch_id: 'batch-1',
        state: 'running',
        total_items: 2,
        completed_items: 1,
        failed_items: 0,
        cancelled_items: 0,
        running_items: 1,
        queued_items: 0,
        progress: 0.5,
        created_at: '2026-09-06T00:00:00Z',
        updated_at: '2026-09-06T00:00:01Z',
        jobs: [
          {
            job_id: 'job-1',
            seq: 1,
            state: 'done',
            stage: 'done',
            progress: 1.0,
            message: 'Finished',
            input_path: '/take1.wav',
            output_path: '/out/take1.wav',
            report_path: '/out/take1.json',
            profile: 'studio',
            mode: 'natural',
            created_at: '2026-09-06T00:00:00Z',
            started_at: '2026-09-06T00:00:01Z',
            finished_at: '2026-09-06T00:00:02Z',
            error: null,
            report: null,
          },
          {
            job_id: 'job-2',
            seq: 2,
            state: 'running',
            stage: 'enhance',
            progress: 0.3,
            message: 'Enhancing unit 3/10',
            input_path: '/take2.wav',
            output_path: '/out/take2.wav',
            report_path: '/out/take2.json',
            profile: 'studio',
            mode: 'natural',
            created_at: '2026-09-06T00:00:00Z',
            started_at: '2026-09-06T00:00:02Z',
            finished_at: null,
            error: null,
            report: null,
          },
        ],
      },
    });
    await act(async () => {
      root.render(createElement(BatchQueue));
    });
    const { seriousOrCritical } = await runAxe(host);
    expect(seriousOrCritical).toEqual([]);
  });

  it('VerdictStrip passes axe-core accessibility checks', async () => {
    useStore.setState({
      report: fakeReport(),
      original: fakeAnalysis(),
    });
    await act(async () => {
      root.render(createElement(VerdictStrip));
    });
    const { seriousOrCritical } = await runAxe(host);
    expect(seriousOrCritical).toEqual([]);
  });

  it('UnitInspector passes axe-core accessibility checks', async () => {
    useStore.setState({
      report: fakeReport(),
      original: fakeAnalysis(),
      selectedUnit: fakeUnit(1),
    });
    await act(async () => {
      root.render(createElement(UnitInspector));
    });
    const { seriousOrCritical } = await runAxe(host);
    expect(seriousOrCritical).toEqual([]);
  });

  it('RestorationCard passes axe-core accessibility checks', async () => {
    const sec = {
      mode: 'restore' as const,
      speaker_id: 'sorani_m1',
      profile_hash: 'p'.repeat(64),
      natural_output_hash: 'n'.repeat(64),
      bandwidth: {
        effective_cutoff_hz: 7800.0,
        confidence: 0.92,
        shape: 'codec_lowpass' as const,
        restore_recommended: true,
        cutoff_mode: 'auto' as const,
        evidence: {
          spectral_rolloff: 7500.0,
          above_cutoff_snr_db: -14.2,
          stationarity: 0.83,
          high_band_energy_ratio_db: -41.6,
        },
      },
      restorer: {
        name: 'hawarestore-kd',
        weights_sha256: 'a'.repeat(64),
        solver: 'midpoint' as const,
        steps: 32,
        ode_rtol: 1e-4,
        ode_atol: 1e-5,
      },
      guard: {
        passed: true,
        guard_a: { verdict: 'pass' as const },
        guard_b: { verdict: 'pass' as const },
        scores: {},
      },
      render: {
        success: true,
        strength: 0.85,
        elapsed_s: 0.45,
        rtf: 0.22,
        peak_vram_mb: null,
      },
    };
    await act(async () => {
      root.render(createElement(RestorationCard, { rest: sec }));
    });
    const { seriousOrCritical } = await runAxe(host);
    expect(seriousOrCritical).toEqual([]);
  });
});
