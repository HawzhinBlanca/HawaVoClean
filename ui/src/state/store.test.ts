// state/store.ts — the app's state machine. Every transition here is one a
// person can see on screen (the engine LED, the run list, the selection, the
// A/B decks), so each test is written as the visible consequence.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { AudioAnalysis, HawaVoCleanReport, JobStatus, UnitDecisionRecord } from '../api/types';
import { waveView } from '../render/viewWindow';
import { HISTORY_LIMIT, useStore, getState, type AppState, type HistoryEntry } from './store';

const pristine = useStore.getState();
beforeEach(() => {
  useStore.setState(pristine, true);
  waveView.clear();
});
afterEach(() => {
  vi.useRealTimers();
});

function analysis(over: Partial<AudioAnalysis> = {}): AudioAnalysis {
  return {
    path: '/a.wav',
    duration_s: 42.5,
    sample_rate: 48000,
    channels: 1,
    peaks: { min: [-1], max: [1] },
    rms_db: [-20],
    spectrum: { freqs_hz: [100], db: [-30] },
    loudness: { integrated_lufs: -24.9, true_peak_dbtp: -1.2 },
    noise_floor_db: -48.5,
    ...over,
  };
}

function unit(id: number, start: number, end: number, channel = 0): UnitDecisionRecord {
  return {
    unit_id: id,
    channel,
    start_sample: Math.round(start * 48000),
    end_sample: Math.round(end * 48000),
    start_time_s: start,
    end_time_s: end,
    is_speech: true,
    input_sha256: 'in',
    output_sha256: 'out',
    guard_a_verdict: 'pass',
    final_decision: 'enhanced',
  };
}

function report(units: UnitDecisionRecord[]): HawaVoCleanReport {
  return {
    job_id: 'j1',
    config_hash: 'c',
    input: {
      path: '/a.wav',
      sha256: 'x',
      sample_rate: 48000,
      channels: 1,
      samples: 1,
      duration_s: 42.5,
    },
    output: {
      path: '/out/a.wav',
      sha256: 'y',
      sample_rate: 48000,
      channels: 1,
      samples: 1,
      duration_s: 42.5,
    },
    core: { id: 'studio', algorithm: 'x', params_hash: 'p' },
    guard: { id: 'g', probe_hash: 'p', calibration_id: 'c' },
    environment: {},
    summary: { units_total: units.length, enhanced: units.length },
    units,
  };
}

function entry(over: Partial<HistoryEntry> = {}): HistoryEntry {
  return {
    jobId: 'j1',
    profile: 'studio',
    inputPath: '/a.wav',
    inputName: 'a.wav',
    outcome: 'done',
    durationMs: 1000,
    at: 1,
    enhanced: 5,
    unitsTotal: 5,
    lufsIn: -24.9,
    lufsOut: -21.7,
    noiseIn: -48.5,
    noiseOut: -84.7,
    outputPath: '/out/a.wav',
    reportPath: '/out/a.hawavoclean.json',
    report: null,
    status: null,
    original: null,
    cleaned: null,
    error: null,
    ...over,
  };
}

function jobStatus(over: Partial<JobStatus> = {}): JobStatus {
  return {
    job_id: 'j1',
    state: 'running',
    stage: 'enhance',
    progress: 0.5,
    message: 'enhancing',
    input_path: '/a.wav',
    output_path: '/out/a.wav',
    report_path: '/out/a.hawavoclean.json',
    profile: 'studio',
    started_at: null,
    finished_at: null,
    ...over,
  };
}

const s = (): AppState => getState();

describe('engine status (B6)', () => {
  it('starts out connecting, with nothing marked offline', () => {
    expect(s().engineStatus).toBe('connecting');
    expect(s().engineOfflineSince).toBeNull();
    expect(s().statusLine).toBe('Connecting to engine');
  });

  it('stamps the moment the engine was first seen to be gone', () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-08-20T10:00:00Z'));
    s().setEngine('offline', null, null);
    const first = s().engineOfflineSince;
    expect(first).toBe(Date.parse('2026-08-20T10:00:00Z'));
    vi.setSystemTime(new Date('2026-08-20T10:00:30Z'));
    s().setEngine('offline', null, null);
    expect(s().engineOfflineSince).toBe(first); // the outage did not restart
  });

  it('keeps the outage stamp through a retry, so the banner does not blink', () => {
    s().setEngine('offline', null, null);
    const first = s().engineOfflineSince;
    s().setEngine('connecting', null, null);
    expect(s().engineStatus).toBe('connecting');
    expect(s().engineOfflineSince).toBe(first);
  });

  it('a first connect does not look like an outage', () => {
    s().setEngine('connecting', null, null);
    expect(s().engineOfflineSince).toBeNull();
  });

  it('a returning engine clears the outage and the probe countdown', () => {
    s().setEngine('offline', null, null);
    s().setEngineProbe(Date.now() + 5000, true);
    s().setEngine('ready', null, '3.2.0');
    expect(s().engineOfflineSince).toBeNull();
    expect(s().engineNextProbeAt).toBeNull();
    expect(s().engineProbing).toBe(false);
    expect(s().engineVersion).toBe('3.2.0');
  });

  it('losing the engine keeps everything the session has loaded', () => {
    s().setSource({ path: '/a.wav', name: 'a.wav', origin: 'file' });
    s().setOriginal(analysis());
    s().pushHistory(entry());
    s().setEngine('offline', null, '3.2.0');
    expect(s().source?.name).toBe('a.wav');
    expect(s().original).not.toBeNull();
    expect(s().history).toHaveLength(1);
  });
});

describe('source and decks', () => {
  it('the clip length comes from the analysis, and goes with it', () => {
    s().setOriginal(analysis({ duration_s: 61.25 }));
    expect(s().duration).toBe(61.25);
    s().setOriginal(null);
    expect(s().duration).toBe(0);
  });

  it('resetForNewSource clears the screen but keeps the session memory', () => {
    s().setOriginal(analysis());
    s().setCleaned(analysis(), '/out/a.wav');
    s().setReport(report([unit(0, 0, 1)]));
    s().setSelectedUnit(unit(0, 0, 1));
    s().setAbMode('cleaned');
    s().setTime(12);
    s().setError('boom');
    s().pushHistory(entry());
    s().pushHistory(entry({ jobId: 'j2' }));

    s().resetForNewSource();

    expect(s().original).toBeNull();
    expect(s().cleaned).toBeNull();
    expect(s().cleanedPath).toBeNull();
    expect(s().report).toBeNull();
    expect(s().selectedUnit).toBeNull();
    expect(s().highlightRange).toBeNull();
    expect(s().abMode).toBe('original');
    expect(s().currentTime).toBe(0);
    expect(s().duration).toBe(0);
    expect(s().error).toBeNull();
    expect(s().view).toEqual({ start: 0, end: 0 });
    // B5 · the run list is the session's memory and survives a new clip,
    // but nothing in it is on screen any more.
    expect(s().history).toHaveLength(2);
    expect(s().currentRunId).toBeNull();
  });
});

describe('job lifecycle', () => {
  it('patchJob is inert with no job, and merges when there is one', () => {
    s().patchJob({ streamConnected: true });
    expect(s().job).toBeNull();
    s().setJob({
      id: 'j1',
      outputPath: '/out/a.wav',
      reportPath: '/out/a.json',
      status: null,
      streamConnected: false,
    });
    s().patchJob({ streamConnected: true });
    expect(s().job).toMatchObject({ id: 'j1', streamConnected: true, status: null });
    s().patchJob({ status: jobStatus() });
    expect(s().job?.status?.state).toBe('running');
    expect(s().job?.streamConnected).toBe(true); // the earlier patch survived
  });
});

describe('selection (B4)', () => {
  it('selecting a unit lights its range in the waveform', () => {
    s().setSelectedUnit(unit(3, 12.5, 14.25));
    expect(s().selectedUnit?.unit_id).toBe(3);
    expect(s().highlightRange).toEqual({ start: 12.5, end: 14.25 });
  });

  it('clearing the selection puts the highlight out', () => {
    s().setSelectedUnit(unit(3, 12.5, 14.25));
    s().setSelectedUnit(null);
    expect(s().selectedUnit).toBeNull();
    expect(s().highlightRange).toBeNull();
  });

  it('a new report invalidates a selection made against the old one', () => {
    s().setReport(report([unit(0, 0, 1)]));
    s().setSelectedUnit(unit(0, 0, 1));
    s().setReport(report([unit(0, 0, 2), unit(1, 2, 4)]));
    expect(s().selectedUnit).toBeNull();
    expect(s().highlightRange).toBeNull();
  });

  it('hover is separate from selection', () => {
    const u = unit(1, 1, 2);
    s().setSelectedUnit(u);
    s().setHoverUnit({ unit: unit(2, 3, 4), x: 10, y: 20 });
    expect(s().selectedUnit?.unit_id).toBe(1);
    s().setHoverUnit(null);
    expect(s().selectedUnit?.unit_id).toBe(1);
    expect(s().highlightRange).toEqual({ start: 1, end: 2 });
  });
});

describe('session history (B5)', () => {
  it('puts the newest run first and makes it the one on screen', () => {
    s().pushHistory(entry({ jobId: 'j1' }));
    s().pushHistory(entry({ jobId: 'j2' }));
    expect(s().history.map((h) => h.jobId)).toEqual(['j2', 'j1']);
    expect(s().currentRunId).toBe('j2');
  });

  it('re-pushing a run moves it to the front instead of duplicating it', () => {
    s().pushHistory(entry({ jobId: 'j1' }));
    s().pushHistory(entry({ jobId: 'j2' }));
    s().pushHistory(entry({ jobId: 'j1', outcome: 'failed' }));
    expect(s().history.map((h) => h.jobId)).toEqual(['j1', 'j2']);
    expect(s().history[0]?.outcome).toBe('failed');
  });

  it(`keeps ${HISTORY_LIMIT} runs and drops the oldest`, () => {
    for (let i = 0; i < HISTORY_LIMIT + 4; i++) s().pushHistory(entry({ jobId: `j${i}` }));
    expect(s().history).toHaveLength(HISTORY_LIMIT);
    expect(s().history[0]?.jobId).toBe(`j${HISTORY_LIMIT + 3}`);
    expect(s().history.map((h) => h.jobId)).not.toContain('j0');
  });

  it('patchHistory touches only the run named', () => {
    s().pushHistory(entry({ jobId: 'j1' }));
    s().pushHistory(entry({ jobId: 'j2' }));
    s().patchHistory('j1', { cleaned: analysis(), lufsOut: -21.7 });
    const [j2, j1] = s().history;
    expect(j1?.cleaned).not.toBeNull();
    expect(j1?.lufsOut).toBe(-21.7);
    expect(j2?.cleaned).toBeNull();
  });

  it('patching a run that is not there changes nothing', () => {
    s().pushHistory(entry({ jobId: 'j1' }));
    const before = s().history;
    s().patchHistory('nope', { lufsOut: 0 });
    expect(s().history).toHaveLength(1);
    expect(s().history[0]).toEqual(before[0]);
  });
});

describe('the view mirror', () => {
  it('follows the imperative controller on a trailing timer, once per burst', async () => {
    waveView.setSource(100, 48000, 1200);
    // Let any mirror timer armed by an earlier test run out first — the
    // coalescing latch is module-level, exactly as it is in the app.
    await new Promise((r) => setTimeout(r, 200));
    vi.useFakeTimers();
    useStore.setState({ view: { start: 0, end: 0 } });
    // Thirty wheel events in one burst: the controller moves every time, and
    // React is told once, at the end.
    for (let i = 0; i < 30; i++) waveView.set(i, i + 10);
    expect(s().view).toEqual({ start: 0, end: 0 });
    await vi.advanceTimersByTimeAsync(119);
    expect(s().view).toEqual({ start: 0, end: 0 });
    await vi.advanceTimersByTimeAsync(2);
    expect(s().view).toEqual({ start: 29, end: 39 });
  });

  it('setView drives the controller and the mirror together', () => {
    waveView.setSource(100, 48000, 1200);
    s().setView(10, 20);
    expect(waveView.view).toEqual({ start: 10, end: 20 });
    expect(s().view).toEqual({ start: 10, end: 20 });
    s().resetView();
    expect(s().view).toEqual({ start: 0, end: 100 });
  });
});

describe('transient UI state', () => {
  it('a new refusal replaces the old one; nothing queues up', () => {
    s().setRejection({ kind: 'type', name: 'notes.txt', detail: 'text/plain' });
    s().setRejection({ kind: 'folder', name: 'Takes', detail: 'a folder' });
    expect(s().rejection?.kind).toBe('folder');
    s().setRejection(null);
    expect(s().rejection).toBeNull();
  });

  it('upload progress is a phase plus two numbers', () => {
    s().setUpload({ name: 'a.wav', loaded: 512, total: 1024, phase: 'sending' });
    expect(s().upload).toEqual({ name: 'a.wav', loaded: 512, total: 1024, phase: 'sending' });
    s().setUpload({ name: 'a.wav', loaded: 1024, total: 1024, phase: 'finishing' });
    expect(s().upload?.phase).toBe('finishing');
    s().setUpload(null);
    expect(s().upload).toBeNull();
  });

  it('the shortcut overlay is a plain latch', () => {
    expect(s().shortcutsOpen).toBe(false);
    s().setShortcutsOpen(true);
    expect(s().shortcutsOpen).toBe(true);
  });
});

describe('restore capability (contract addendum 2)', () => {
  it('starts with no capability and the natural default', () => {
    expect(s().speakers).toEqual([]);
    expect(s().restoreAvailable).toBe(false);
    expect(s().mode).toBe('natural');
    expect(s().speakerId).toBeNull();
    expect(s().cutoffHz).toBeNull();
  });

  it('offering speakers auto-selects the first, so restore is always submittable', () => {
    s().setCapabilities(['character_01', 'character_02'], true);
    expect(s().speakerId).toBe('character_01');
    // A selection already made survives a re-probe that still offers it.
    s().setSpeakerId('character_02');
    s().setCapabilities(['character_01', 'character_02'], true);
    expect(s().speakerId).toBe('character_02');
  });

  it('a speaker the engine no longer offers cannot stay selected', () => {
    s().setCapabilities(['character_01', 'character_02'], true);
    s().setSpeakerId('character_02');
    // The profile tree changed under the app: character_02 is gone.
    s().setCapabilities(['character_01'], true);
    expect(s().speakerId).toBe('character_01');
  });

  it('restore going away entirely forces the mode back to natural', () => {
    s().setCapabilities(['character_01'], true);
    s().setMode('restore');
    s().setCapabilities([], false);
    expect(s().mode).toBe('natural');
    expect(s().speakerId).toBeNull();
    // The cutoff is a typed preference, not a capability; it survives.
    s().setCutoffHz(7800);
    s().setCapabilities([], false);
    expect(s().cutoffHz).toBe(7800);
  });
});
