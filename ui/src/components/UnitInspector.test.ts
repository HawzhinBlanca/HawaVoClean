// components/UnitInspector.test.ts — Wave 2 unit action bar (Loop, Override, Undo/Redo) + a11y.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import axe from 'axe-core';
import type { EngineClient } from '../api/client';
import type { HawaVoCleanReport, UnitDecisionRecord } from '../api/types';
import { useStore } from '../state/store';
import { UnitInspector } from './UnitInspector';

const mockPlayer = vi.hoisted(() => ({
  time: 0,
  playing: false,
  duration: 10,
  load: vi.fn(),
  seek: vi.fn(),
  play: vi.fn(async () => undefined),
  pause: vi.fn(),
  setActive: vi.fn(),
  setLoopRange: vi.fn(),
  setLoopEnabled: vi.fn(),
  setLoudnessMatch: vi.fn(),
  hasDeck: vi.fn(() => true),
  gainOffsetDb: 0,
}));

vi.mock('../audio/player', () => ({ getPlayer: () => mockPlayer }));

declare global {
  var IS_REACT_ACT_ENVIRONMENT: boolean | undefined;
}
globalThis.IS_REACT_ACT_ENVIRONMENT = true;

const pristine = useStore.getState();
let host: HTMLElement;
let root: Root;

function makeUnit(id: number, over: Partial<UnitDecisionRecord> = {}): UnitDecisionRecord {
  return {
    unit_id: id,
    channel: 0,
    start_sample: id * 48000,
    end_sample: (id + 1) * 48000,
    start_time_s: id + 1,
    end_time_s: id + 2.5,
    is_speech: true,
    input_sha256: 'in',
    output_sha256: 'out',
    guard_a_verdict: 'PASS',
    final_decision: 'enhanced',
    decision_reason: 'guard pass',
    guard_a_scores: { envelope_correlation: 0.97 },
    ...over,
  };
}

function makeReport(units: UnitDecisionRecord[]): HawaVoCleanReport {
  return {
    job_id: 'job-override-1',
    config_hash: 'cfg',
    input: { path: '/a.wav', sha256: 'x', sample_rate: 48000, channels: 1, samples: 480000, duration_s: 10 },
    output: { path: '/out/a.wav', sha256: 'y', sample_rate: 48000, channels: 1, samples: 480000, duration_s: 10 },
    core: { id: 'studio_pro', algorithm: 'mossformer2', params_hash: 'p' },
    guard: { id: 'guard_a', probe_hash: 'p', calibration_id: 'c' },
    environment: {},
    summary: { units_total: units.length, enhanced: 1, reverted: 1 },
    units,
  };
}

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
    root.render(createElement(UnitInspector));
  });
}

describe('UnitInspector Detail action bar', () => {
  it('renders Loop unit, Keep original, Force cleaned, Reset auto and loops unit on click', async () => {
    const u0 = makeUnit(0);
    const u1 = makeUnit(1, { final_decision: 'original_reverted', guard_a_verdict: 'FAIL' });
    useStore.setState({
      report: makeReport([u0, u1]),
      selectedUnit: u0,
    });
    await render();

    const loopBtn = host.querySelector<HTMLButtonElement>('button[aria-label="Loop unit 0"]');
    expect(loopBtn).not.toBeNull();
    expect(host.querySelector('button[aria-label="Keep original audio for unit 0"]')).not.toBeNull();
    expect(host.querySelector('button[aria-label="Force cleaned audio for unit 0"]')).not.toBeNull();
    expect(host.querySelector('button[aria-label="Reset auto — unit 0 automatic guard decision"]')).not.toBeNull();
    // Undo/Redo hidden until history exists
    expect(host.querySelector('button[aria-label="Undo last override or rack edit"]')).toBeNull();

    await act(async () => {
      loopBtn?.click();
    });

    expect(useStore.getState().selectionRange).toEqual({ start: 1, end: 2.5 });
    expect(mockPlayer.setLoopRange).toHaveBeenCalledWith([1, 2.5]);
    expect(mockPlayer.setLoopEnabled).toHaveBeenCalledWith(true);
    expect(mockPlayer.seek).toHaveBeenCalledWith(1);
    expect(mockPlayer.play).toHaveBeenCalledTimes(1);
  });

  it('applies unit override via API, re-loads cleaned deck when restitched, and supports Undo/Redo', async () => {
    const u0 = makeUnit(0);
    const u1 = makeUnit(1, { final_decision: 'original_reverted', guard_a_verdict: 'FAIL' });
    const overrideUnit = vi.fn(async (_jobId: string, req: { unitId: number; decision: string }) => ({
      jobId: 'job-override-1',
      unitIndex: req.unitId,
      unitId: req.unitId,
      oldDecision: 'enhanced',
      newDecision: req.decision === 'force_original' ? 'original_reverted' : 'enhanced',
      override: req.decision as 'auto' | 'force_original' | 'force_enhanced',
      restitched: true,
      restitchMs: 42,
      masterSha256: 'sha_after_override',
    }));
    const client = {
      overrideUnit,
      fileUrl: (p: string) => `http://127.0.0.1:8765/api/audio?path=${encodeURIComponent(p)}`,
      analyze: vi.fn(async () => null),
    } as unknown as EngineClient;

    useStore.setState({
      engineStatus: 'ready',
      client,
      cleanedPath: '/out/a.wav',
      job: {
        id: 'job-override-1',
        outputPath: '/out/a.wav',
        reportPath: '/out/a.json',
        status: null,
        streamConnected: false,
      },
      report: makeReport([u0, u1]),
      selectedUnit: u0,
    });
    await render();

    const keepOrigBtn = host.querySelector<HTMLButtonElement>(
      'button[aria-label="Keep original audio for unit 0"]',
    );
    await act(async () => {
      keepOrigBtn?.click();
    });

    expect(overrideUnit).toHaveBeenCalledWith('job-override-1', {
      unitId: 0,
      decision: 'force_original',
    });
    expect(useStore.getState().unitOverrides[0]).toBe('force_original');
    expect(useStore.getState().selectedUnit?.final_decision).toBe('original_reverted');
    expect(mockPlayer.load).toHaveBeenCalledWith(
      'cleaned',
      expect.stringContaining('sha_after_override'),
      null,
    );

    // Undo / Redo buttons now visible
    const undoBtn = host.querySelector<HTMLButtonElement>(
      'button[aria-label="Undo last override or rack edit"]',
    );
    const redoBtn = host.querySelector<HTMLButtonElement>(
      'button[aria-label="Redo last override or rack edit"]',
    );
    expect(undoBtn).not.toBeNull();
    expect(undoBtn?.disabled).toBe(false);
    expect(redoBtn?.disabled).toBe(true);

    // Undo restores previous override state
    await act(async () => {
      undoBtn?.click();
    });
    expect(useStore.getState().unitOverrides[0]).toBeUndefined();
    expect(redoBtn?.disabled).toBe(false);

    // Redo re-applies force_original
    await act(async () => {
      redoBtn?.click();
    });
    expect(useStore.getState().unitOverrides[0]).toBe('force_original');

    // WCAG 2.2 AA axe-core check with override & history buttons rendered
    const results = await axe.run(host, {
      rules: {
        'color-contrast': { enabled: false },
        region: { enabled: false },
      },
    });
    expect(results.violations).toEqual([]);
  });

  it('fails soft cleanly when engine is offline during override', async () => {
    const u0 = makeUnit(0);
    useStore.setState({
      engineStatus: 'offline',
      client: null,
      report: makeReport([u0]),
      selectedUnit: u0,
    });
    await render();

    const forceCleanBtn = host.querySelector<HTMLButtonElement>(
      'button[aria-label="Force cleaned audio for unit 0"]',
    );
    await act(async () => {
      forceCleanBtn?.click();
    });

    expect(useStore.getState().unitOverrides[0]).toBe('force_enhanced');
    expect(useStore.getState().statusLine).toContain('offline preview');
    expect(useStore.getState().error).toBeNull();
  });
});
