// state/selection.ts — the one place that knows what "select a unit" means.
// The verdict strip's click and the keyboard's `[` / `]` both come through
// here, so this is where the two are proved not to have drifted apart.

import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { HawaVoCleanReport, UnitDecisionRecord } from '../api/types';
import { waveView } from '../render/viewWindow';
import {
  channelName,
  clearSelection,
  highlightFor,
  orderedUnits,
  reportChannels,
  selectedIndex,
  selectUnit,
  stepUnit,
  unitKey,
} from './selection';
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

  // CHANGED in the A7/B4 stereo pass, deliberately. This used to pin *time*
  // order across channels. On a split-speakers report the per-channel unit sets
  // overlap in time, so a time-major walk flips lane on nearly every press
  // (ch1 20.474 then ch0 20.570 is a 96 ms move and a lane change) and visits
  // two units that both start at 0.000 back to back without the playhead
  // moving at all. Channel-major is the order the lanes are drawn in: the top
  // lane left to right, then the next one.
  it('reads channel-major: each lane in time order, lane after lane', () => {
    loadReport([unit(0, 0, 1, 0), unit(1, 10, 11, 0), unit(0, 5, 6, 1), unit(1, 15, 16, 1)]);
    expect(orderedUnits().map((u) => [u.start_time_s, u.channel])).toEqual([
      [0, 0],
      [10, 0],
      [5, 1],
      [15, 1],
    ]);
  });

  it('is plain time order when there is only one channel', () => {
    loadReport([unit(2, 10, 11), unit(0, 0, 1), unit(1, 5, 6)]);
    expect(orderedUnits().map((u) => u.start_time_s)).toEqual([0, 5, 10]);
  });

  // The engine emits one channel's units after the other's, so a real stereo
  // report is never sorted by time to begin with — this is that report.
  it('sorts a report that arrives out of time order', () => {
    loadReport([
      unit(3, 0, 20.474, 1),
      unit(0, 0, 20.57, 0),
      unit(5, 40.528, 60.001, 1),
      unit(1, 20.57, 40.251, 0),
      unit(4, 20.474, 40.528, 1),
      unit(2, 40.251, 60.001, 0),
    ]);
    expect(orderedUnits().map(unitKey)).toEqual(['0-0', '0-1', '0-2', '1-3', '1-4', '1-5']);
  });

  it('breaks a tie on start time, then on unit id, so the order is total', () => {
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

  // CHANGED with orderedUnits, above: `]` walks a lane out before it steps
  // down to the next one, which is the order the strip stacks them in.
  it('walks one channel out, then steps to the next one', () => {
    loadReport([unit(0, 0, 1, 0), unit(0, 2, 3, 1), unit(1, 4, 5, 0)]);
    stepUnit(1);
    expect(unitKey(getState().selectedUnit as UnitDecisionRecord)).toBe('0-0');
    stepUnit(1);
    expect(unitKey(getState().selectedUnit as UnitDecisionRecord)).toBe('0-1');
    stepUnit(1);
    expect(unitKey(getState().selectedUnit as UnitDecisionRecord)).toBe('1-0');
    stepUnit(-1);
    expect(unitKey(getState().selectedUnit as UnitDecisionRecord)).toBe('0-1');
  });

  // With nothing selected the playhead sits inside a unit of *every* channel
  // at once, so "the unit the playhead is in" is only an answer once a lane is
  // chosen. The first lane is the one the eye starts on.
  it('bootstraps from the first channel, not from whichever unit sorts first', () => {
    loadReport([unit(0, 0, 20, 0), unit(1, 20, 40, 0), unit(2, 0, 21, 1), unit(3, 21, 40, 1)]);
    player.time = 25;
    stepUnit(1);
    expect(unitKey(getState().selectedUnit as UnitDecisionRecord)).toBe('0-1');
    expect(player.seek).toHaveBeenCalledWith(20);
  });

  it('bootstrapping backwards also stays in the first channel', () => {
    loadReport([unit(0, 0, 20, 0), unit(1, 30, 40, 0), unit(2, 0, 21, 1), unit(3, 25, 40, 1)]);
    player.time = 24; // in no unit of channel 0
    stepUnit(-1);
    expect(unitKey(getState().selectedUnit as UnitDecisionRecord)).toBe('0-0');
  });
});

// A7/B4 · a split-speakers run decides per channel, and those decisions overlap
// in time. Everything that shows a unit has to agree on how many channels there
// are and what to call them, so all of it is answered here.
describe('channels (A7/B4)', () => {
  it('has no channels with no report', () => {
    expect(reportChannels()).toEqual([]);
  });

  it('reads the channels off the units, ascending and deduplicated', () => {
    loadReport([unit(0, 0, 1, 1), unit(1, 0, 1, 0), unit(2, 1, 2, 1), unit(3, 1, 2, 0)]);
    expect(reportChannels()).toEqual([0, 1]);
  });

  // A dual-mono file has two channels but ONE set of decisions: the engine
  // processes ch0 and duplicates it. That report must keep the mono strip.
  it('a report whose units are all on one channel is one channel', () => {
    loadReport([unit(0, 0, 1, 0), unit(1, 1, 2, 0)]);
    expect(reportChannels()).toHaveLength(1);
  });

  it('names two channels L and R, and anything else by number', () => {
    expect(channelName(0, [0, 1])).toEqual({ short: 'L', long: 'left channel' });
    expect(channelName(1, [0, 1])).toEqual({ short: 'R', long: 'right channel' });
    expect(channelName(2, [0, 1, 2]).short).toBe('C2');
    expect(channelName(0, [0]).short).toBe('M');
  });

  // The channel is named by its POSITION among the report's channels, not by
  // its number: a report carrying only channels 0 and 2 still has an L and an R.
  it('names by position, so a 0/2 report still reads L and R', () => {
    expect(channelName(2, [0, 2])).toEqual({ short: 'R', long: 'right channel' });
  });

  it('a mono highlight carries no channel; a stereo one does', () => {
    expect(highlightFor(unit(0, 1, 2, 0), [0])).toEqual({ start: 1, end: 2 });
    expect(highlightFor(unit(0, 1, 2, 1), [0, 1])).toEqual({ start: 1, end: 2, channel: 1 });
    expect(highlightFor(null, [0, 1])).toBeNull();
  });

  it('selecting on a stereo report lights the range WITH its channel', () => {
    loadReport([unit(0, 0, 20.57, 0), unit(1, 0, 20.474, 1)]);
    selectUnit(unit(1, 0, 20.474, 1));
    expect(getState().highlightRange).toEqual({ start: 0, end: 20.474, channel: 1 });
  });

  // The two units below cover almost the same seconds; without the channel the
  // waveform paints the identical band for each and the selection is unreadable.
  it('two overlapping units of different channels light different bands', () => {
    loadReport([unit(0, 0, 20.57, 0), unit(1, 0, 20.474, 1)]);
    selectUnit(unit(0, 0, 20.57, 0));
    const a = getState().highlightRange;
    selectUnit(unit(1, 0, 20.474, 1));
    const b = getState().highlightRange;
    expect(a?.channel).toBe(0);
    expect(b?.channel).toBe(1);
    expect(a).not.toEqual(b);
  });

  it('a mono report still lights a plain range — no channel, no regression', () => {
    loadReport([unit(0, 0, 20.57, 0), unit(1, 20.57, 40, 0)]);
    selectUnit(unit(1, 20.57, 40, 0));
    expect(getState().highlightRange).toEqual({ start: 20.57, end: 40 });
  });
});
