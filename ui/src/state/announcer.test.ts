// state/announcer.ts — the app's single live region.
//
// This is the only surface that speaks. Everything else on screen shows
// progress silently, and the component's own comment explains why: four places
// re-render on every SSE tick, so wiring any of them up as a live region would
// announce a percentage several times a second and make the app unusable with
// a screen reader on. That makes this hook the whole accessibility story for
// the job lifecycle — and it had no test.
//
// The load-bearing cases are the ones where the sentence could be *wrong*
// rather than merely absent: saying a clip is ready when its analysis failed,
// and saying the same thing twice.

import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { act, createElement, type ReactElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import type { AudioAnalysis, HawaVoCleanReport, JobStatus } from '../api/types';
import { useJobAnnouncer } from './announcer';
import { getState, useStore } from './store';

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

function Probe(): ReactElement {
  return createElement('p', null, useJobAnnouncer());
}

async function render(): Promise<void> {
  await act(async () => {
    root.render(createElement(Probe));
  });
}

const said = (): string => host.querySelector('p')?.textContent ?? '';

function analysis(over: Partial<AudioAnalysis> = {}): AudioAnalysis {
  return {
    path: '/a.wav',
    duration_s: 60,
    sample_rate: 48000,
    channels: 1,
    peaks: { min: [-1], max: [1] },
    rms_db: [-20],
    spectrum: { freqs_hz: [100], db: [-30] },
    loudness: { integrated_lufs: -23, true_peak_dbtp: -1 },
    noise_floor_db: -48,
    ...over,
  };
}

function status(state: JobStatus['state'], over: Partial<JobStatus> = {}): JobStatus {
  return {
    job_id: 'j1',
    state,
    stage: state === 'done' ? 'done' : 'enhance',
    progress: state === 'done' ? 1 : 0.4,
    message: '',
    unit: null,
    input_path: '/a.wav',
    output_path: '/out/a.wav',
    report_path: '/out/a.json',
    profile: 'studio',
    started_at: null,
    finished_at: null,
    error: null,
    report: null,
    ...over,
  };
}

function report(enhanced: number, total: number): HawaVoCleanReport {
  return {
    job_id: 'j1',
    config_hash: 'c',
    input: { path: '/a.wav', sha256: 'x', sample_rate: 48000, channels: 1, samples: 1, duration_s: 60 },
    output: { path: '/o.wav', sha256: 'y', sample_rate: 48000, channels: 1, samples: 1, duration_s: 60 },
    core: { id: 'studio', algorithm: 'x', params_hash: 'p' },
    guard: { id: 'g', probe_hash: 'p', calibration_id: 'c' },
    environment: {},
    summary: { units_total: total, enhanced },
    units: [],
  };
}

function job(st: JobStatus | null): void {
  getState().setJob({
    id: 'j1',
    outputPath: '/out/a.wav',
    reportPath: '/out/a.json',
    status: st,
    streamConnected: true,
  });
}

describe('what the app says about a clip', () => {
  it('says nothing at all until something happens', async () => {
    await render();
    expect(said()).toBe('');
  });

  it('names the clip while it is being analysed', async () => {
    await render();
    await act(async () => {
      getState().setSource({ path: '/a.wav', name: 'take-01.wav', origin: 'file' });
      getState().setAnalyzing(true);
    });
    expect(said()).toBe('Analyzing take-01.wav');
  });

  it('calls a clip ready only once it really has an analysis', async () => {
    await render();
    await act(async () => {
      getState().setSource({ path: '/a.wav', name: 'take-01.wav', origin: 'file' });
      getState().setAnalyzing(true);
    });
    await act(async () => {
      getState().setAnalyzing(false);
      getState().setOriginal(analysis());
    });
    expect(said()).toBe('take-01.wav ready');
  });

  it('does NOT call a clip ready when its analysis failed', async () => {
    // The defect this case exists for: a file the engine refused, or an
    // analysis the user cancelled, still cleared `analyzing` with a source name
    // set. The one region a screen-reader user has announced the opposite of
    // what happened, and the run they went on to start could not work.
    await render();
    await act(async () => {
      getState().setSource({ path: '/a.wav', name: 'take-01.wav', origin: 'file' });
      getState().setAnalyzing(true);
    });
    await act(async () => {
      getState().setAnalyzing(false);
      getState().setError('sample rate 192000 Hz is not supported', 'Engine refused');
    });
    expect(said()).not.toContain('ready');
    expect(said()).toContain('take-01.wav');
    expect(said()).toContain('not analyzed');
  });
});

describe('what the app says about a run', () => {
  it('announces the start once, not the progress', async () => {
    await render();
    await act(async () => {
      getState().setSource({ path: '/a.wav', name: 'take-01.wav', origin: 'file' });
      getState().setOriginal(analysis());
      job(status('running', { progress: 0.1 }));
    });
    expect(said()).toBe('Processing started');
    // Every SSE tick re-renders this. If progress reached the region it would
    // speak several times a second and the app would be unusable.
    await act(async () => {
      job(status('running', { progress: 0.9, unit: { index: 4, total: 5 } }));
    });
    expect(said()).toBe('Processing started');
  });

  it('says how a finished run turned out, in words and not a percentage', async () => {
    await render();
    await act(async () => {
      getState().setReport(report(3, 5));
      job(status('done'));
    });
    expect(said()).toContain('Processing finished');
    expect(said()).toContain('3 of 5');
  });

  it('names the reason a run failed', async () => {
    await render();
    await act(async () => {
      job(status('failed', { error: { code: 'ENGINE_RESTARTED', message: 'gone' } }));
    });
    expect(said()).toBe('Processing failed: ENGINE_RESTARTED');
  });

  it('says a cancellation was a cancellation', async () => {
    await render();
    await act(async () => {
      job(status('cancelled'));
    });
    expect(said()).toBe('Processing cancelled');
  });
});

describe('a deck falling out of service', () => {
  it('is announced once, and not again while it stands', async () => {
    await render();
    const fault = {
      deck: 'cleaned' as const,
      headline: 'CLEANED DECK UNAVAILABLE',
      detail: 'The cleaned master is no longer on disk.',
    };
    await act(async () => {
      getState().setDeckFault(fault);
    });
    expect(said()).toBe(fault.detail);

    // Re-setting the same fault must not re-announce: it changes what the
    // transport can do once, not on every render that observes it.
    await act(async () => {
      getState().setStatus('something else changed');
      getState().setDeckFault({ ...fault });
    });
    expect(said()).toBe(fault.detail);
  });

  it('rides the same single region rather than opening a second voice', async () => {
    await render();
    await act(async () => {
      getState().setSource({ path: '/a.wav', name: 'take-01.wav', origin: 'file' });
      getState().setOriginal(analysis());
    });
    expect(said()).toBe('take-01.wav ready');
    await act(async () => {
      getState().setDeckFault({
        deck: 'cleaned',
        headline: 'CLEANED DECK UNAVAILABLE',
        detail: 'The cleaned master is no longer on disk.',
      });
    });
    // One region, one sentence at a time.
    expect(said()).toBe('The cleaned master is no longer on disk.');
  });
});
