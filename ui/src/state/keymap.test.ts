// state/keymap.ts — goal box B1, the whole keyboard map. Every binding, and
// just as importantly every place a binding must NOT fire: with a modifier
// held, while a text field has focus, while the shortcut dialog is up, or on
// a control that owns the key itself.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { HawaVoCleanReport, JobStatus, UnitDecisionRecord } from '../api/types';
import { handleKeyDown, isActivatable, isEditable } from './keymap';
import { getState, useStore } from './store';

const player = vi.hoisted(() => ({ time: 0, seek: vi.fn() }));
vi.mock('../audio/player', () => ({ getPlayer: () => player }));

const actions = vi.hoisted(() => ({
  cancelAnalysis: vi.fn(),
  cancelJob: vi.fn(),
  cancelUpload: vi.fn(),
  isAnalyzing: vi.fn(() => false),
  seekTo: vi.fn(),
  setAb: vi.fn(),
  startJob: vi.fn(),
  togglePlay: vi.fn(),
}));
vi.mock('./actions', () => actions);

function analysis() {
  return {
    path: '/a.wav',
    duration_s: 100,
    sample_rate: 48000,
    channels: 1,
    peaks: { min: [-1], max: [1] },
    rms_db: [-20],
    spectrum: { freqs_hz: [100], db: [-30] },
    loudness: { integrated_lufs: -24.9, true_peak_dbtp: -1.2 },
    noise_floor_db: -48.5,
  };
}

function unit(id: number, start: number, end: number): UnitDecisionRecord {
  return {
    unit_id: id,
    channel: 0,
    start_sample: 0,
    end_sample: 1,
    start_time_s: start,
    end_time_s: end,
    is_speech: true,
    input_sha256: 'i',
    output_sha256: 'o',
    guard_a_verdict: 'pass',
    final_decision: 'enhanced',
  };
}

function loadReport(units: UnitDecisionRecord[]): void {
  getState().setReport({
    job_id: 'j1',
    config_hash: 'c',
    input: { path: '/a.wav', sha256: 'x', sample_rate: 48000, channels: 1, samples: 1, duration_s: 100 },
    output: { path: '/o.wav', sha256: 'y', sample_rate: 48000, channels: 1, samples: 1, duration_s: 100 },
    core: { id: 'studio', algorithm: 'x', params_hash: 'p' },
    guard: { id: 'g', probe_hash: 'p', calibration_id: 'c' },
    environment: {},
    summary: {},
    units,
  } satisfies HawaVoCleanReport);
}

function runningJob(state: JobStatus['state'] = 'running'): void {
  getState().setJob({
    id: 'j1',
    outputPath: '/o.wav',
    reportPath: '/o.json',
    status: {
      job_id: 'j1',
      state,
      stage: 'enhance',
      progress: 0.5,
      message: 'enhancing',
      input_path: '/a.wav',
      output_path: '/o.wav',
      report_path: '/o.json',
      profile: 'studio',
      started_at: null,
      finished_at: null,
    },
    streamConnected: true,
  });
}

/** The state in which every control is armed. */
function armed(): void {
  const st = getState();
  st.setEngine('ready', null, '3.2.0');
  st.setSource({ path: '/a.wav', name: 'a.wav', origin: 'file' });
  st.setOriginal(analysis());
}

interface PressOptions {
  shiftKey?: boolean;
  metaKey?: boolean;
  ctrlKey?: boolean;
  altKey?: boolean;
  code?: string;
  target?: HTMLElement;
  preventFirst?: boolean;
}

/** Dispatch a real keydown through the real listener, on a real target. */
function press(key: string, opts: PressOptions = {}): KeyboardEvent {
  const { target = document.body, preventFirst = false, ...init } = opts;
  const ev = new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true, ...init });
  if (preventFirst) ev.preventDefault();
  target.dispatchEvent(ev);
  return ev;
}

function el(tag: string, attrs: Record<string, string> = {}): HTMLElement {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  document.body.appendChild(node);
  return node;
}

const pristine = useStore.getState();
beforeEach(() => {
  useStore.setState(pristine, true);
  player.time = 30;
  window.addEventListener('keydown', handleKeyDown);
});

afterEach(() => {
  window.removeEventListener('keydown', handleKeyDown);
  document.body.innerHTML = '';
});

describe('the ? overlay owns the keyboard while it is up', () => {
  it('? opens it and ? closes it, swallowing the key both times', () => {
    const open = press('?');
    expect(getState().shortcutsOpen).toBe(true);
    expect(open.defaultPrevented).toBe(true);
    press('?');
    expect(getState().shortcutsOpen).toBe(false);
  });

  it('no transport key fires while the dialog is open', () => {
    armed();
    press('?');
    press(' ', { code: 'Space' });
    press('ArrowRight');
    press('p');
    press('a');
    expect(actions.togglePlay).not.toHaveBeenCalled();
    expect(actions.seekTo).not.toHaveBeenCalled();
    expect(actions.startJob).not.toHaveBeenCalled();
    expect(actions.setAb).not.toHaveBeenCalled();
  });
});

describe('inertness', () => {
  it('nothing fires with Cmd, Ctrl or Alt held — those belong to the browser', () => {
    armed();
    for (const mod of ['metaKey', 'ctrlKey', 'altKey'] as const) {
      press(' ', { code: 'Space', [mod]: true });
      press('ArrowRight', { [mod]: true });
      press('a', { [mod]: true });
      press('p', { [mod]: true });
      press('?', { [mod]: true });
    }
    expect(actions.togglePlay).not.toHaveBeenCalled();
    expect(actions.seekTo).not.toHaveBeenCalled();
    expect(actions.setAb).not.toHaveBeenCalled();
    expect(actions.startJob).not.toHaveBeenCalled();
    expect(getState().shortcutsOpen).toBe(false);
  });

  it('nothing fires while a text field has focus', () => {
    armed();
    for (const tag of ['input', 'textarea', 'select']) {
      const target = el(tag);
      press(' ', { code: 'Space', target });
      press('ArrowRight', { target });
      press('p', { target });
    }
    const ce = el('div');
    Object.defineProperty(ce, 'isContentEditable', { value: true });
    press(' ', { code: 'Space', target: ce });
    expect(actions.togglePlay).not.toHaveBeenCalled();
    expect(actions.seekTo).not.toHaveBeenCalled();
    expect(actions.startJob).not.toHaveBeenCalled();
  });

  it('a key another handler has already claimed is left alone', () => {
    armed();
    press(' ', { code: 'Space', preventFirst: true });
    expect(actions.togglePlay).not.toHaveBeenCalled();
  });

  it('isEditable / isActivatable read the DOM, not a guess', () => {
    expect(isEditable(null)).toBe(false);
    expect(isEditable(el('input'))).toBe(true);
    expect(isEditable(el('div'))).toBe(false);
    expect(isActivatable(el('button'))).toBe(true);
    expect(isActivatable(el('a', { href: '#' }))).toBe(true);
    expect(isActivatable(el('div', { role: 'button' }))).toBe(true);
    expect(isActivatable(el('div'))).toBe(false);
    expect(isActivatable(null)).toBe(false);
  });
});

describe('Space — play/pause', () => {
  it('toggles the transport when there is audio, and swallows the scroll', () => {
    armed();
    const ev = press(' ', { code: 'Space' });
    expect(actions.togglePlay).toHaveBeenCalledTimes(1);
    expect(ev.defaultPrevented).toBe(true);
  });

  it('is swallowed but silent with no clip loaded', () => {
    const ev = press(' ', { code: 'Space' });
    expect(actions.togglePlay).not.toHaveBeenCalled();
    expect(ev.defaultPrevented).toBe(true);
  });

  it('belongs to a focused button, not to the transport', () => {
    armed();
    press(' ', { code: 'Space', target: el('button') });
    press(' ', { code: 'Space', target: el('div', { role: 'button' }) });
    press(' ', { code: 'Space', target: el('a', { href: '#' }) });
    expect(actions.togglePlay).not.toHaveBeenCalled();
  });

  it('answers to the physical key as well as the character', () => {
    armed();
    press('Spacebar', { code: 'Space' }); // older key name, same physical key
    expect(actions.togglePlay).toHaveBeenCalledTimes(1);
  });
});

describe('arrows — seek', () => {
  it('moves ±5 s, and ±1 s with Shift', () => {
    armed();
    press('ArrowRight');
    expect(actions.seekTo).toHaveBeenLastCalledWith(35);
    press('ArrowLeft');
    expect(actions.seekTo).toHaveBeenLastCalledWith(25);
    press('ArrowRight', { shiftKey: true });
    expect(actions.seekTo).toHaveBeenLastCalledWith(31);
    press('ArrowLeft', { shiftKey: true });
    expect(actions.seekTo).toHaveBeenLastCalledWith(29);
  });

  it('never seeks before the start of the file', () => {
    armed();
    player.time = 2;
    press('ArrowLeft');
    expect(actions.seekTo).toHaveBeenLastCalledWith(0);
  });

  it('does nothing at all with no clip, and lets the page keep the key', () => {
    const ev = press('ArrowRight');
    expect(actions.seekTo).not.toHaveBeenCalled();
    expect(ev.defaultPrevented).toBe(false);
  });

  it('swallows the arrow when it does seek, so the page does not scroll', () => {
    armed();
    expect(press('ArrowRight').defaultPrevented).toBe(true);
  });
});

describe('[ and ] — unit stepping (B4)', () => {
  beforeEach(() => {
    armed();
    loadReport([unit(0, 0, 1), unit(1, 5, 6), unit(2, 10, 11)]);
    player.time = 7; // in the gap between units 1 and 2
  });

  it('] walks forward through the units from wherever the playhead is', () => {
    press(']');
    expect(getState().selectedUnit?.unit_id).toBe(2);
    press('[');
    expect(getState().selectedUnit?.unit_id).toBe(1);
    press('[');
    expect(getState().selectedUnit?.unit_id).toBe(0);
  });

  it('accepts , and . as the same bindings (unshifted < and >)', () => {
    press('.');
    const first = getState().selectedUnit?.unit_id;
    press(',');
    expect(getState().selectedUnit?.unit_id).toBe((first ?? 1) - 1);
  });

  it('swallows the key so it never reaches a text field behind the app', () => {
    expect(press(']').defaultPrevented).toBe(true);
    expect(press('[').defaultPrevented).toBe(true);
  });
});

describe('A / B — deck switching', () => {
  it('A goes to the original deck once there is audio', () => {
    press('a');
    expect(actions.setAb).not.toHaveBeenCalled();
    armed();
    press('a');
    expect(actions.setAb).toHaveBeenCalledWith('original');
  });

  it('B only works once there is a cleaned master to switch to', () => {
    armed();
    press('b');
    expect(actions.setAb).not.toHaveBeenCalled();
    getState().setCleaned(analysis(), '/out/a.wav');
    press('b');
    expect(actions.setAb).toHaveBeenCalledWith('cleaned');
  });

  it('is case-insensitive, so Shift+A still switches decks', () => {
    armed();
    press('A', { shiftKey: true });
    expect(actions.setAb).toHaveBeenCalledWith('original');
  });
});

describe('P — process', () => {
  it('starts a run only when every precondition is met', () => {
    armed();
    press('p');
    expect(actions.startJob).toHaveBeenCalledTimes(1);
  });

  it('is silent while the engine is offline', () => {
    armed();
    getState().setEngine('offline', null, '3.2.0');
    press('p');
    expect(actions.startJob).not.toHaveBeenCalled();
  });

  it('is silent while a clip is still being analysed', () => {
    armed();
    getState().setAnalyzing(true);
    press('p');
    expect(actions.startJob).not.toHaveBeenCalled();
  });

  it('is silent while a run is already going', () => {
    armed();
    runningJob();
    press('p');
    expect(actions.startJob).not.toHaveBeenCalled();
  });

  it('arms again once the run has reached a terminal state', () => {
    armed();
    runningJob('done');
    press('p');
    expect(actions.startJob).toHaveBeenCalledTimes(1);
  });

  it('is silent with no clip at all', () => {
    getState().setEngine('ready', null, '3.2.0');
    press('p');
    expect(actions.startJob).not.toHaveBeenCalled();
  });
});

describe('Escape — stop the innermost thing that is happening', () => {
  it('cancels an upload first', () => {
    armed();
    runningJob();
    getState().setUpload({ name: 'a.wav', loaded: 1, total: 2, phase: 'sending' });
    getState().setSelectedUnit(unit(1, 5, 6));
    press('Escape');
    expect(actions.cancelUpload).toHaveBeenCalledTimes(1);
    expect(actions.cancelJob).not.toHaveBeenCalled();
    expect(getState().selectedUnit).not.toBeNull();
  });

  it('cancels an analysis before it reaches the job branch', () => {
    actions.isAnalyzing.mockReturnValue(true);
    getState().setSelectedUnit(null);
    press('Escape');
    expect(actions.cancelAnalysis).toHaveBeenCalledTimes(1);
    expect(actions.cancelJob).not.toHaveBeenCalled();
    actions.isAnalyzing.mockReturnValue(false);
  });

  it('then a running job', () => {
    armed();
    runningJob();
    getState().setSelectedUnit(unit(1, 5, 6));
    press('Escape');
    expect(actions.cancelJob).toHaveBeenCalledTimes(1);
    expect(getState().selectedUnit).not.toBeNull();
  });

  it('counts a job with no status yet as running', () => {
    armed();
    getState().setJob({
      id: 'j1',
      outputPath: '/o.wav',
      reportPath: '/o.json',
      status: null,
      streamConnected: false,
    });
    press('Escape');
    expect(actions.cancelJob).toHaveBeenCalledTimes(1);
  });

  it('then a drop refusal', () => {
    armed();
    getState().setRejection({ kind: 'type', name: 'notes.txt', detail: 'text/plain' });
    getState().setSelectedUnit(unit(1, 5, 6));
    press('Escape');
    expect(getState().rejection).toBeNull();
    expect(getState().selectedUnit).not.toBeNull();
  });

  it('and finally the selection', () => {
    armed();
    getState().setSelectedUnit(unit(1, 5, 6));
    press('Escape');
    expect(getState().selectedUnit).toBeNull();
    expect(getState().highlightRange).toBeNull();
  });

  it('is swallowed even when there is nothing to stop', () => {
    const ev = press('Escape');
    expect(ev.defaultPrevented).toBe(true);
    expect(actions.cancelJob).not.toHaveBeenCalled();
    expect(actions.cancelUpload).not.toHaveBeenCalled();
  });
});

describe('keys the app does not claim', () => {
  it('leaves ordinary typing to the page', () => {
    armed();
    for (const key of ['z', 'q', '1', 'Tab', 'Enter', 'F5']) {
      const ev = press(key);
      expect(ev.defaultPrevented, key).toBe(false);
    }
    expect(actions.togglePlay).not.toHaveBeenCalled();
    expect(actions.setAb).not.toHaveBeenCalled();
    expect(actions.startJob).not.toHaveBeenCalled();
  });
});
