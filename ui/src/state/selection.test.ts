// state/selection.ts — the one place that knows what "select a unit" means.
// The verdict strip's click and the keyboard's `[` / `]` both come through
// here, so this is where the two are proved not to have drifted apart.

import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { HawaVoCleanReport, UnitDecisionRecord } from '../api/types';
import { waveView } from '../render/viewWindow';
import { clearSelection, orderedUnits, selectedIndex, selectUnit, stepUnit, unitKey } from './selection';
import { getState, useStore } from './store';

const player = vi.hoisted(() => ({ time: 0, seek: vi.fn() }));
vi.mock('../audio/player', () => ({ getPlayer: () => player }));

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

function loadReport(units: UnitDecisionRecord[]): void {
  const report = {
    job_id: 'j1',
    config_hash: 'c',
    input: { path: '/a.wav', sha256: 'x', sample_rate: 48000, channels: 1, samples: 1, duration_s: 100 },
    output: { path: '/o.wav', sha256: 'y', sample_rate: 48000, channels: 1, samples: 1, duration_s: 100 },
    core: { id: 'studio', algorithm: 'x', params_hash: 'p' },
    guard: { id: 'g', probe_hash: 'p', calibration_id: 'c' },
    environment: {},
    summary: { units_total: units.length },
    units,
  } satisfies HawaVoCleanReport;
  getState().setReport(report);
}

const pristine = useStore.getState();
beforeEach(() => {
  useStore.setState(pristine, true);
  waveView.setSource(100, 48000, 1200);
  player.time = 0;
  player.seek.mockClear();
});

describe('unitKey', () => {
  it('keys on channel and id together, because ids repeat per channel', () => {
    expect(unitKey(unit(0, 0, 1, 0))).not.toBe(unitKey(unit(0, 0, 1, 1)));
    expect(unitKey(unit(2, 5, 6, 1))).toBe(unitKey(unit(2, 9, 9, 1)));
  });
});

describe('orderedUnits', () => {
  it('is empty with no report', () => {
    expect(orderedUnits()).toEqual([]);
  });

  it('reads in time order, not the report’s per-channel order', () => {
    loadReport([unit(0, 0, 1, 0), unit(1, 10, 11, 0), unit(0, 5, 6, 1), unit(1, 15, 16, 1)]);
    expect(orderedUnits().map((u) => [u.start_time_s, u.channel])).toEqual([
      [0, 0],
      [5, 1],
      [10, 0],
      [15, 1],
    ]);
  });

  it('breaks a tie on channel, then on unit id, so the order is total', () => {
    loadReport([unit(3, 4, 5, 1), unit(1, 4, 5, 1), unit(9, 4, 5, 0)]);
    expect(orderedUnits().map((u) => [u.channel, u.unit_id])).toEqual([
      [0, 9],
      [1, 1],
      [1, 3],
    ]);
  });

  it('does not disturb the report’s own array', () => {
    const units = [unit(1, 10, 11), unit(0, 0, 1)];
    loadReport(units);
    orderedUnits();
    expect(getState().report?.units?.[0]?.unit_id).toBe(1);
  });
});

describe('selectUnit', () => {
  it('lights the range, seeks the transport, and remembers the unit', () => {
    const u = unit(4, 12.5, 14);
    selectUnit(u);
    expect(getState().selectedUnit?.unit_id).toBe(4);
    expect(getState().highlightRange).toEqual({ start: 12.5, end: 14 });
    expect(player.seek).toHaveBeenCalledWith(12.5);
  });

  it('can select without moving the playhead (keyboard review of a running deck)', () => {
    selectUnit(unit(4, 12.5, 14), { seek: false });
    expect(getState().selectedUnit?.unit_id).toBe(4);
    expect(player.seek).not.toHaveBeenCalled();
  });

  it('clearing puts the highlight out and seeks nowhere', () => {
    selectUnit(unit(4, 12.5, 14));
    player.seek.mockClear();
    clearSelection();
    expect(getState().selectedUnit).toBeNull();
    expect(getState().highlightRange).toBeNull();
    expect(player.seek).not.toHaveBeenCalled();
  });
});

describe('bringing the unit into view (B3 + B4 together)', () => {
  it('leaves the zoom alone when the unit is already on screen', () => {
    waveView.set(10, 30);
    selectUnit(unit(1, 12, 14));
    expect(waveView.view).toEqual({ start: 10, end: 30 });
  });

  it('centres a unit that is off screen, at the same zoom', () => {
    waveView.set(10, 30);
    selectUnit(unit(1, 60, 62));
    expect(waveView.span).toBe(20);
    expect((waveView.start_s + waveView.end_s) / 2).toBeCloseTo(61, 9);
  });

  it('anchors a unit longer than the window near its start, not its middle', () => {
    waveView.set(60, 70); // 10 s window
    selectUnit(unit(1, 20, 50)); // 30 s unit
    expect(waveView.span).toBeCloseTo(10, 9);
    expect(waveView.start_s).toBeCloseTo(19.5, 9); // start − 5% of the span
  });

  it('does nothing to a view that has no clip behind it', () => {
    waveView.clear();
    expect(() => selectUnit(unit(1, 2, 3))).not.toThrow();
    expect(waveView.view).toEqual({ start: 0, end: 0 });
    expect(getState().selectedUnit?.unit_id).toBe(1);
  });
});

describe('selectedIndex', () => {
  it('is -1 with nothing selected', () => {
    loadReport([unit(0, 0, 1)]);
    expect(selectedIndex()).toBe(-1);
  });

  it('finds the selection by key, not by object identity', () => {
    loadReport([unit(0, 0, 1), unit(1, 5, 6), unit(2, 10, 11)]);
    getState().setSelectedUnit(unit(1, 5, 6)); // a structurally equal copy
    expect(selectedIndex()).toBe(1);
  });
});

describe('stepUnit', () => {
  it('does nothing with no report', () => {
    expect(() => stepUnit(1)).not.toThrow();
    expect(getState().selectedUnit).toBeNull();
  });

  it('walks forwards and backwards in time order', () => {
    loadReport([unit(0, 0, 1), unit(1, 5, 6), unit(2, 10, 11)]);
    stepUnit(1);
    expect(getState().selectedUnit?.unit_id).toBe(0);
    stepUnit(1);
    expect(getState().selectedUnit?.unit_id).toBe(1);
    stepUnit(1);
    expect(getState().selectedUnit?.unit_id).toBe(2);
    stepUnit(-1);
    expect(getState().selectedUnit?.unit_id).toBe(1);
  });

  it('stops at both ends instead of wrapping', () => {
    loadReport([unit(0, 0, 1), unit(1, 5, 6)]);
    getState().setSelectedUnit(unit(1, 5, 6));
    stepUnit(1);
    expect(getState().selectedUnit?.unit_id).toBe(1);
    getState().setSelectedUnit(unit(0, 0, 1));
    stepUnit(-1);
    expect(getState().selectedUnit?.unit_id).toBe(0);
  });

  it('starts from the unit the playhead is sitting in', () => {
    loadReport([unit(0, 0, 1), unit(1, 5, 6), unit(2, 10, 11)]);
    player.time = 5.4;
    stepUnit(1);
    expect(getState().selectedUnit?.unit_id).toBe(1);
    expect(player.seek).toHaveBeenCalledWith(5);
  });

  it('from a gap, forwards picks the next unit', () => {
    loadReport([unit(0, 0, 1), unit(1, 5, 6), unit(2, 10, 11)]);
    player.time = 7;
    stepUnit(1);
    expect(getState().selectedUnit?.unit_id).toBe(2);
  });

  it('from a gap, backwards picks the previous unit', () => {
    loadReport([unit(0, 0, 1), unit(1, 5, 6), unit(2, 10, 11)]);
    player.time = 7;
    stepUnit(-1);
    expect(getState().selectedUnit?.unit_id).toBe(1);
  });

  // With nothing selected and the playhead outside every unit, both directions
  // CLAMP. An earlier build wrapped here — `]` past the last unit came round to
  // the first — which silently threw the user back to the top of a take.
  it('past the last unit, `]` stays on the last and `[` takes the last', () => {
    loadReport([unit(0, 0, 1), unit(1, 5, 6), unit(2, 10, 11)]);
    player.time = 90;
    stepUnit(1);
    expect(getState().selectedUnit?.unit_id).toBe(2);
    getState().setSelectedUnit(null);
    stepUnit(-1);
    expect(getState().selectedUnit?.unit_id).toBe(2);
  });

  it('before the first unit, `]` takes the first and `[` stays on the first', () => {
    loadReport([unit(0, 5, 6), unit(1, 10, 11), unit(2, 15, 16)]);
    player.time = 0;
    stepUnit(1);
    expect(getState().selectedUnit?.unit_id).toBe(0);
    getState().setSelectedUnit(null);
    stepUnit(-1);
    expect(getState().selectedUnit?.unit_id).toBe(0);
  });

  it('crosses channels in time order', () => {
    loadReport([unit(0, 0, 1, 0), unit(0, 2, 3, 1), unit(1, 4, 5, 0)]);
    stepUnit(1);
    expect(unitKey(getState().selectedUnit as UnitDecisionRecord)).toBe('0-0');
    stepUnit(1);
    expect(unitKey(getState().selectedUnit as UnitDecisionRecord)).toBe('1-0');
    stepUnit(1);
    expect(unitKey(getState().selectedUnit as UnitDecisionRecord)).toBe('0-1');
  });
});
