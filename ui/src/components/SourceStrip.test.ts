import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import axe from 'axe-core';
import type { EngineClient } from '../api/client';
import type { AudioAnalysis, TrackInfo } from '../api/types';
import { extractAndLoadTrack } from '../state/actions';
import { useStore } from '../state/store';
import { SourceStrip } from './SourceStrip';

vi.mock('../bridge', () => ({
  getBridge: () => ({
    host: 'web',
    engine: { getEndpoint: async () => ({ baseUrl: '' }) },
    files: {
      pickAudio: async () => null,
      pathForFile: () => null,
      revealInFinder: async () => undefined,
    },
  }),
}));

vi.mock('../state/actions', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../state/actions')>()),
  extractAndLoadTrack: vi.fn(async () => '/work/extracted_tracks/poly/ch01_Boom.wav'),
}));

declare global {
  var IS_REACT_ACT_ENVIRONMENT: boolean | undefined;
}
globalThis.IS_REACT_ACT_ENVIRONMENT = true;

const pristine = useStore.getState();
let host: HTMLElement;
let root: Root;

function makeAnalysis(channels: number, path = '/field/poly.wav'): AudioAnalysis {
  return {
    path,
    duration_s: 24.5,
    sample_rate: 48000,
    channels,
    peaks: { min: [-0.4], max: [0.4] },
    rms_db: [-22],
    spectrum: { freqs_hz: [200], db: [-30] },
    loudness: { integrated_lufs: -23, true_peak_dbtp: -1 },
    noise_floor_db: -52,
  };
}

const SAMPLE_TRACKS: TrackInfo[] = [
  {
    index: 0,
    name: 'Boom',
    sample_rate: 48000,
    samples: 1176000,
    duration_s: 24.5,
    rms_dbfs: -21.4,
    peak_dbfs: -4.2,
    speech_activity_ratio: 0.78,
    likely_dialogue: true,
  },
  {
    index: 1,
    name: 'Lav A',
    sample_rate: 48000,
    samples: 1176000,
    duration_s: 24.5,
    rms_dbfs: -26.8,
    peak_dbfs: -9.1,
    speech_activity_ratio: 0.64,
    likely_dialogue: true,
  },
  {
    index: 2,
    name: 'Room Ambience',
    sample_rate: 48000,
    samples: 1176000,
    duration_s: 24.5,
    rms_dbfs: -41.2,
    peak_dbfs: -18.5,
    speech_activity_ratio: 0.08,
    likely_dialogue: false,
  },
];

beforeEach(() => {
  vi.clearAllMocks();
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

async function render(): Promise<void> {
  await act(async () => {
    root.render(createElement(SourceStrip));
  });
}

describe('SourceStrip Poly-WAV Track Picker', () => {
  it('hides the Tracks button for mono and stereo clips', async () => {
    useStore.setState({
      engineStatus: 'ready',
      source: { path: '/clip.wav', name: 'clip.wav', origin: 'file' },
      original: makeAnalysis(2, '/clip.wav'),
    });
    await render();
    expect(host.querySelector('.poly-tracks-btn')).toBeNull();
  });

  it('shows Tracks (Nch) button for poly-WAV (>2ch), inspects tracks, extracts selected track, and passes axe-core', async () => {
    const inspectTracks = vi.fn(async () => ({
      path: '/field/poly.wav',
      trackCount: 6,
      tracks: SAMPLE_TRACKS,
    }));
    const client = { inspectTracks } as unknown as EngineClient;

    useStore.setState({
      engineStatus: 'ready',
      client,
      source: { path: '/field/poly.wav', name: 'poly.wav', origin: 'file' },
      original: makeAnalysis(6, '/field/poly.wav'),
    });
    await render();

    const tracksBtn = host.querySelector<HTMLButtonElement>('button.poly-tracks-btn');
    expect(tracksBtn).not.toBeNull();
    expect(tracksBtn?.textContent?.trim()).toBe('Tracks (6ch)');

    await act(async () => {
      tracksBtn?.click();
    });

    expect(inspectTracks).toHaveBeenCalledWith('/field/poly.wav');
    const picker = host.querySelector('#poly-track-picker');
    expect(picker).not.toBeNull();
    expect(picker?.textContent).toContain('Boom');
    expect(picker?.textContent).toContain('-21.4 dB RMS');
    expect(picker?.textContent).toContain('78% speech');
    expect(picker?.textContent).toContain('Lav A');
    expect(picker?.textContent).toContain('Room Ambience');

    const results = await axe.run(host, {
      rules: {
        'color-contrast': { enabled: false },
        region: { enabled: false },
      },
    });
    expect(results.violations).toEqual([]);

    const loadBtns = host.querySelectorAll<HTMLButtonElement>('button.ptp-load-btn');
    expect(loadBtns).toHaveLength(3);
    await act(async () => {
      loadBtns[1]?.click();
    });

    expect(extractAndLoadTrack).toHaveBeenCalledWith('/field/poly.wav', 1, 'Lav A');
    expect(host.querySelector('#poly-track-picker')).toBeNull();
  });
});
