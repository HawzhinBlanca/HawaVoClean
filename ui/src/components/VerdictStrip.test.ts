// components/VerdictStrip.tsx — the densest control on the screen, and the one
// carrying a documented P0 that had no regression test.
//
// A7/B4: a split-speakers run decides *per channel*, and those decisions
// overlap in time. Drawn in one lane they stack, and the measurement in the
// component's own comment is that the last channel painted was topmost on 687
// of the track's 689 px — half the report had no hover, no click and no
// selection at all. The fix is one lane per channel. Until now nothing tested
// it, so a refactor could silently put half a stereo report back out of reach.
//
// The other cases here cover the semantics a screen reader gets and the
// selection route, which is the only way to reach a unit from the keyboard.
//
// Rendering uses react-dom/client directly (no JSX, so this file rides the
// same `src/**/*.test.{ts,tsx}` include as every other unit test).

import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import type { HawaVoCleanReport, UnitDecisionRecord } from '../api/types';
import { useStore, getState } from '../state/store';
import { VerdictStrip } from './VerdictStrip';

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

function unit(id: number, over: Partial<UnitDecisionRecord> = {}): UnitDecisionRecord {
  return {
    unit_id: id,
    channel: 0,
    start_sample: 0,
    end_sample: 1,
    start_time_s: id,
    end_time_s: id + 1,
    is_speech: true,
    input_sha256: 'i',
    output_sha256: 'o',
    guard_a_verdict: 'pass',
    final_decision: 'enhanced',
    ...over,
  };
}

function report(units: UnitDecisionRecord[]): HawaVoCleanReport {
  return {
    job_id: 'j1',
    config_hash: 'c',
    input: { path: '/a.wav', sha256: 'x', sample_rate: 48000, channels: 1, samples: 1, duration_s: 10 },
    output: { path: '/o.wav', sha256: 'y', sample_rate: 48000, channels: 1, samples: 1, duration_s: 10 },
    core: { id: 'studio', algorithm: 'x', params_hash: 'p' },
    guard: { id: 'g', probe_hash: 'p', calibration_id: 'c' },
    environment: {},
    summary: { units_total: units.length, enhanced: units.length },
    units,
  };
}

async function render(units: UnitDecisionRecord[]): Promise<void> {
  await act(async () => {
    getState().setReport(report(units));
    getState().setDuration(10);
  });
  await act(async () => {
    root.render(createElement(VerdictStrip));
  });
}

/**
 * Segments inside the track only. The legend renders four decoy
 * `<i className="verdict-seg">` chips that are not units, and an unscoped
 * query silently counts them.
 */
function segs(): HTMLElement[] {
  return [...host.querySelectorAll<HTMLElement>('.verdict-track .verdict-seg')];
}

describe('VerdictStrip · per-channel lanes (A7/B4 regression)', () => {
  it('gives a two-channel report one lane per channel', async () => {
    await render([
      unit(0, { channel: 0 }),
      unit(1, { channel: 1 }),
      unit(2, { channel: 0 }),
      unit(3, { channel: 1 }),
    ]);
    const lanes = [...host.querySelectorAll<HTMLElement>('.verdict-lane')];
    expect(lanes).toHaveLength(2);
    // Each lane must declare its own index, or they draw on top of each other
    // — which is the whole defect.
    expect(lanes.map((l) => l.style.getPropertyValue('--lane'))).toEqual(['0', '1']);
  });

  it('puts every segment in its own channel’s lane and nowhere else', async () => {
    await render([
      unit(0, { channel: 0 }),
      unit(1, { channel: 1 }),
      unit(2, { channel: 0 }),
      unit(3, { channel: 1 }),
    ]);
    const lanes = [...host.querySelectorAll<HTMLElement>('.verdict-lane')];
    const names = lanes.map((l) =>
      [...l.querySelectorAll('.verdict-seg')].map((s) => s.getAttribute('aria-label') ?? ''),
    );
    // A two-channel report names them left/right rather than by index — the
    // words a listener can act on.
    expect(names[0]?.every((n) => /left channel/.test(n))).toBe(true);
    expect(names[1]?.every((n) => /right channel/.test(n))).toBe(true);
    expect(names[0]).toHaveLength(2);
    expect(names[1]).toHaveLength(2);
  });

  it('lanes and names a report with more channels than left/right', async () => {
    // Beyond two there is no left/right to claim, so the name falls back to the
    // index. No run through the web API can produce this today (it refuses
    // >2-channel files before a job starts), so this is the only place the
    // three-lane path is exercised at all.
    await render([
      unit(0, { channel: 0 }),
      unit(1, { channel: 1 }),
      unit(2, { channel: 2 }),
    ]);
    const lanes = [...host.querySelectorAll<HTMLElement>('.verdict-lane')];
    expect(lanes).toHaveLength(3);
    expect(lanes.map((l) => l.style.getPropertyValue('--lane'))).toEqual(['0', '1', '2']);
    expect(segs()[2]?.getAttribute('aria-label')).toMatch(/channel 2/);
  });

  it('keeps a mono report on the single-lane strip', async () => {
    await render([unit(0), unit(1), unit(2)]);
    // Not `data-lanes`: that derives from channels.length and would still read
    // "1" here under a mutant that broke the multi-lane branch. The load-
    // bearing signal is that no lane element is created at all.
    expect(host.querySelectorAll('.verdict-lane')).toHaveLength(0);
    expect(segs()).toHaveLength(3);
  });

  it('treats a dual-mono report (decisions on ch0 alone) as single-lane', async () => {
    await render([unit(0, { channel: 0 }), unit(1, { channel: 0 })]);
    expect(host.querySelectorAll('.verdict-lane')).toHaveLength(0);
  });
});

describe('VerdictStrip · what a screen reader gets', () => {
  it('names each segment with the fields the hover card shows', async () => {
    await render([
      unit(0, {
        final_decision: 'reverted',
        guard_a_verdict: 'revert',
        decision_reason: 'timing drift',
      }),
    ]);
    const name = segs()[0]?.getAttribute('aria-label') ?? '';
    // The pointer-only card must not be the only route to a unit's decision.
    expect(name).toMatch(/REVERTED/i);
    expect(name).toMatch(/guard A/i);
    expect(name).toMatch(/timing drift/);
  });

  it('marks the selected unit with aria-current, not aria-pressed', async () => {
    const units = [unit(0), unit(1)];
    await render(units);
    await act(async () => {
      getState().setSelectedUnit(units[1] ?? null);
    });
    const all = segs();
    // aria-pressed would describe a toggle: a screen reader would announce
    // "not pressed" on every one of a few hundred segments, for a press that
    // can never be undone.
    expect(all.some((s) => s.hasAttribute('aria-pressed'))).toBe(false);
    expect(all[1]?.getAttribute('aria-current')).toBe('true');
    expect(all[0]?.hasAttribute('aria-current')).toBe(false);
  });

  it('keeps segments out of the tab order', async () => {
    await render([unit(0), unit(1), unit(2)]);
    // A report can carry hundreds of units; tab stops on each would bury the
    // rest of the screen. `[`/`]` is the sequential keyboard route.
    expect(segs().every((s) => s.getAttribute('tabindex') === '-1')).toBe(true);
  });
});

describe('VerdictStrip · selection', () => {
  it('a click selects that unit in the store', async () => {
    const units = [unit(0), unit(1), unit(2)];
    await render(units);
    await act(async () => {
      segs()[2]?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    });
    expect(getState().selectedUnit?.unit_id).toBe(2);
  });

  it('renders nothing to select when the report goes away', async () => {
    await render([unit(0)]);
    expect(segs()).toHaveLength(1);
    await act(async () => {
      getState().setReport(null);
    });
    expect(segs()).toHaveLength(0);
  });
});
