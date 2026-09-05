import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import type { AudioAnalysis, HawaVoCleanReport } from '../api/types';
import { useStore } from '../state/store';
import { SmartExplanation } from './SmartExplanation';

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

function fakeAnalysis(over: Partial<AudioAnalysis> = {}): AudioAnalysis {
  return {
    path: '/path/take1.wav',
    sample_rate: 48000,
    channels: 2,
    duration_s: 10.0,
    noise_floor_db: -58.2,
    peaks: { min: [], max: [] },
    rms_db: [],
    spectrum: { freqs_hz: [], db: [] },
    loudness: {
      integrated_lufs: -23.1,
      true_peak_dbtp: -1.0,
    },
    ...over,
  };
}

function fakeReport(over: Partial<HawaVoCleanReport['summary']> = {}): HawaVoCleanReport {
  return {
    job_id: 'job-1',
    config_hash: 'c',
    input: { path: '/a.wav', sha256: 'x', sample_rate: 48000, channels: 1, samples: 1, duration_s: 60 },
    output: { path: '/o.wav', sha256: 'y', sample_rate: 48000, channels: 1, samples: 1, duration_s: 60 },
    core: { id: 'studio', algorithm: 'x', params_hash: 'p' },
    guard: { id: 'g', probe_hash: 'p', calibration_id: 'c' },
    environment: {},
    summary: {
      units_total: 10,
      enhanced: 8,
      reverted: 2,
      continuity_crossfaded: 1,
      continuity_reverted: 0,
      ...over,
    },
    units: [],
  };
}

describe('SmartExplanation', () => {
  it('renders Step 1: Add Media when no source is loaded', async () => {
    await act(async () => {
      root.render(createElement(SmartExplanation));
    });

    const el = host.querySelector('.smart-explanation.empty');
    expect(el).not.toBeNull();
    expect(el?.getAttribute('aria-label')).toBe('Smart workflow guide');
    expect(host.textContent).toContain('Step 1: Add Media');
    expect(host.textContent).toContain('Drop any Kurdish speech audio');
  });

  it('renders analyzing state when analyzing is true', async () => {
    useStore.setState({
      source: { path: '/tmp/take1.wav', name: 'take1.wav', origin: 'file' },
      analyzing: true,
    });

    await act(async () => {
      root.render(createElement(SmartExplanation));
    });

    const el = host.querySelector('.smart-explanation.analyzing');
    expect(el).not.toBeNull();
    expect(host.textContent).toContain('Analyzing Acoustic Profile…');
  });

  it('renders armed assessment with acoustic analysis details', async () => {
    useStore.setState({
      source: { path: '/tmp/take1.wav', name: 'take1.wav', origin: 'file' },
      analyzing: false,
      original: fakeAnalysis({
        noise_floor_db: -42.0,
        loudness: { integrated_lufs: -18.4, true_peak_dbtp: -0.5 },
      }),
      profile: 'production',
      mode: 'natural',
    });

    await act(async () => {
      root.render(createElement(SmartExplanation));
    });

    const el = host.querySelector('.smart-explanation.armed');
    expect(el).not.toBeNull();
    expect(host.textContent).toContain('Smart Acoustic Assessment');
    expect(host.textContent).toContain('High noise (-42.0 dB)');
    expect(host.textContent).toContain('-18.4 LUFS');
    expect(host.textContent).toContain('48 kHz · 2ch');
    expect(host.textContent).toContain('PRODUCTION · NATURAL');
  });

  it('categorizes moderate and low noise correctly', async () => {
    useStore.setState({
      source: { path: '/tmp/take1.wav', name: 'take1.wav', origin: 'file' },
      analyzing: false,
      original: fakeAnalysis({ noise_floor_db: -50.0 }),
    });

    await act(async () => {
      root.render(createElement(SmartExplanation));
    });
    expect(host.textContent).toContain('Moderate noise (-50.0 dB)');

    await act(async () => {
      useStore.setState({
        original: fakeAnalysis({ noise_floor_db: -65.0 }),
      });
    });
    expect(host.textContent).toContain('Low noise (-65.0 dB)');
  });

  it('renders post-clean verification report when report is available', async () => {
    useStore.setState({
      source: { path: '/tmp/take1.wav', name: 'take1.wav', origin: 'file' },
      analyzing: false,
      original: fakeAnalysis(),
      report: fakeReport({ units_total: 20, enhanced: 18, reverted: 2 }),
      loudnessMatch: true,
      gainOffsetDb: -1.2,
    });

    await act(async () => {
      root.render(createElement(SmartExplanation));
    });

    const el = host.querySelector('.smart-explanation.done');
    expect(el).not.toBeNull();
    expect(host.textContent).toContain('Processing Verified');
    expect(host.textContent).toContain('Guard R: Safe');
    expect(host.textContent).toContain('18 / 20 units');
    expect(host.textContent).toContain('2 units');
    expect(host.textContent).toContain('Level-matched A/B (-1.2 dB on Cleaned)');
  });

  it('renders MP4 dialogue isolation note when input file is an MP4 video container', async () => {
    useStore.setState({
      source: { path: '/tmp/interview.mp4', name: 'interview.mp4', origin: 'file' },
      analyzing: false,
      original: fakeAnalysis(),
      profile: 'production',
      mode: 'natural',
    });

    await act(async () => {
      root.render(createElement(SmartExplanation));
    });

    expect(host.querySelector('.se-mp4-notice')).not.toBeNull();
    expect(host.textContent).toContain('MP4 Dialogue Isolated');
    expect(host.textContent).toContain('without video recoding');

    await act(async () => {
      useStore.setState({
        report: fakeReport(),
      });
    });

    expect(host.querySelector('.se-mp4-notice')).not.toBeNull();
    expect(host.textContent).toContain('MP4 Dialogue Master');
  });
});
