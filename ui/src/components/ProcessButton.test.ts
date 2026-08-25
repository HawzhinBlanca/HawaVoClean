// components/ProcessButton.tsx — the plate's readout.
//
// One control carries five phases (idle, running, done, failed, cancelled) and
// translates the engine's exit-code table into words. It is the thing the user
// looks at to find out what happened, and it had no test.
//
// The exit-code mapping is the part that rots quietly: a code the engine
// renames, or one added without a mapping here, degrades into raw
// SCREAMING_SNAKE on the plate — legible to whoever wrote the engine and to
// nobody else. These cases pin every entry in the table and the fallback.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import type { AudioAnalysis, HawaVoCleanReport, JobStatus } from '../api/types';
import { getState, useStore } from '../state/store';
import { ProcessButton } from './ProcessButton';

vi.mock('../state/actions', () => ({ startJob: vi.fn(), cancelJob: vi.fn() }));

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

async function render(): Promise<void> {
  await act(async () => {
    root.render(createElement(ProcessButton));
  });
}

function analysis(): AudioAnalysis {
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
  };
}

function report(): HawaVoCleanReport {
  return {
    job_id: 'j1',
    config_hash: 'c',
    input: { path: '/a.wav', sha256: 'x', sample_rate: 48000, channels: 1, samples: 1, duration_s: 60 },
    output: { path: '/o.wav', sha256: 'y', sample_rate: 48000, channels: 1, samples: 1, duration_s: 60 },
    core: { id: 'studio', algorithm: 'x', params_hash: 'p' },
    guard: { id: 'g', probe_hash: 'p', calibration_id: 'c' },
    environment: {},
    summary: { units_total: 5, enhanced: 3 },
    units: [],
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
    report: state === 'done' ? report() : null,
    ...over,
  };
}

async function armed(): Promise<void> {
  await act(async () => {
    getState().setEngine('ready', null, '3.3.0');
    getState().setSource({ path: '/a.wav', name: 'take-01.wav', origin: 'file' });
    getState().setOriginal(analysis());
  });
}

async function withStatus(st: JobStatus | null): Promise<void> {
  await act(async () => {
    getState().setJob({
      id: 'j1',
      outputPath: '/out/a.wav',
      reportPath: '/out/a.json',
      status: st,
      streamConnected: true,
    });
  });
}

const face = (): string => (host.textContent ?? '').replace(/\s+/g, ' ');
const button = (): HTMLButtonElement | null => host.querySelector('button.process');

describe('the plate reads back the state it is in', () => {
  it('idle with no clip offers PROCESS and says why it cannot run', async () => {
    await act(async () => {
      getState().setEngine('ready', null, '3.3.0');
    });
    await render();
    expect(face()).toContain('PROCESS');
    expect(face()).toContain('No clip');
  });

  it('idle with the engine gone says Offline, not No clip', async () => {
    // Two different reasons a run cannot start. Reporting the wrong one sends
    // the user to load a file that is already loaded.
    await armed();
    await act(async () => {
      getState().setEngine('offline', null, null);
    });
    await render();
    expect(face()).toContain('Offline');
  });

  it('armed over a clip reads back the clip, not a blank face', async () => {
    await armed();
    await render();
    expect(face()).toContain('Armed');
    expect(face()).toContain('1:00');
  });

  it('running offers CANCEL', async () => {
    await armed();
    await withStatus(status('running'));
    await render();
    expect(face()).toContain('CANCEL');
  });

  it('done offers PROCESS AGAIN and the units it enhanced', async () => {
    await armed();
    await withStatus(status('done'));
    await render();
    expect(face()).toContain('PROCESS AGAIN');
    expect(face()).toContain('Complete');
    expect(face()).toContain('3 / 5');
  });

  it('cancelled offers PROCESS again and says where it stopped', async () => {
    await armed();
    await withStatus(status('cancelled', { progress: 0.62 }));
    await render();
    expect(face()).toContain('Cancelled');
    expect(face()).toContain('62%');
  });

  it('failed offers RETRY', async () => {
    await armed();
    await withStatus(status('failed', { error: { code: 'INTERNAL', message: 'boom' } }));
    await render();
    expect(face()).toContain('RETRY');
    expect(face()).toContain('Failed');
  });
});

describe('the engine’s exit codes become words', () => {
  // Every entry in FAIL_REASON. A code the engine renames, or one added
  // without a mapping, degrades into raw SCREAMING_SNAKE on the plate —
  // legible to whoever wrote the engine and to nobody else.
  const table: Array<[string, string]> = [
    ['PREFLIGHT_FAILURE', 'Preflight'],
    ['PUBLICATION_FAILURE', 'Publish'],
    ['INVALID_USER_INPUT', 'Bad input'],
    ['INTERNAL', 'Internal'],
    ['SPAWN_FAILED', 'No worker'],
    ['ENGINE_RESTARTED', 'Engine gone'],
  ];

  for (const [code, word] of table) {
    it(`${code} reads as "${word}"`, async () => {
      await armed();
      await withStatus(status('failed', { error: { code, message: `raw ${code} detail` } }));
      await render();
      expect(face()).toContain(word);
      expect(face()).not.toContain(code);
    });
  }

  it('an unmapped code still shows something rather than nothing', async () => {
    await armed();
    await withStatus(status('failed', { error: { code: 'NEW_CODE_NOBODY_MAPPED', message: 'x' } }));
    await render();
    // Ugly, but a raw code the user can quote beats a blank cell.
    expect(face()).toContain('NEW_CODE_NOBODY_MAPPED');
  });

  it('keeps the engine’s own sentence reachable behind the short word', async () => {
    await armed();
    await withStatus(status('failed', { error: { code: 'INTERNAL', message: 'worker died on unit 3' } }));
    await render();
    const carrying = [...host.querySelectorAll('[title]')].map((n) => n.getAttribute('title') ?? '');
    expect(carrying.some((t) => t.includes('worker died on unit 3'))).toBe(true);
  });
});

describe('the button itself', () => {
  it('is inert while the engine is gone, and says why', async () => {
    await armed();
    await act(async () => {
      getState().setEngine('offline', null, null);
    });
    await render();
    // `aria-disabled`, not the native attribute — the house pattern: a natively
    // disabled control takes no pointer events, so the `title` explaining why
    // it cannot run would never be shown, and it would leave the accessibility
    // tree along with the explanation.
    expect(button()?.getAttribute('aria-disabled')).toBe('true');
    expect(button()?.disabled).toBe(false);
    expect(button()?.getAttribute('title') ?? '').not.toBe('');
  });

  it('is live once there is a clip and an engine', async () => {
    await armed();
    await render();
    expect(button()?.hasAttribute('aria-disabled')).toBe(false);
  });

  it('names its action for a screen reader, not just its face', async () => {
    await armed();
    await withStatus(status('done'));
    await render();
    // The visible face is a readout; the accessible name has to be the verb.
    expect(button()?.getAttribute('aria-label')).toBe('PROCESS AGAIN');
    expect(button()?.getAttribute('aria-describedby')).toBe('process-readout');
  });
});
