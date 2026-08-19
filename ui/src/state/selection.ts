// Unit selection: the one place that knows what "select a unit" means —
// store state, waveform highlight, transport seek, and bringing the unit into
// the visible zoom window. Both the verdict strip (click) and the keyboard map
// (`[` / `]`) go through here so the two can never drift apart.

import type { UnitDecisionRecord } from '../api/types';
import { getPlayer } from '../audio/player';
import { waveView } from '../render/viewWindow';
import { getState } from './store';

export function unitKey(u: UnitDecisionRecord): string {
  return `${u.channel}-${u.unit_id}`;
}

/** Report order is per channel; the strip and the keyboard read time order. */
export function orderedUnits(): UnitDecisionRecord[] {
  const units = getState().report?.units ?? [];
  return [...units].sort(
    (a, b) =>
      a.start_time_s - b.start_time_s || a.channel - b.channel || a.unit_id - b.unit_id,
  );
}

/** Keep the selected range visible without changing how far the user zoomed. */
function revealInView(u: UnitDecisionRecord): void {
  const dur = waveView.duration;
  if (!(dur > 0)) return;
  const span = waveView.span;
  if (!(span > 0)) return;
  const inside = u.start_time_s >= waveView.start_s && u.end_time_s <= waveView.end_s;
  if (inside) return;
  const unitSpan = u.end_time_s - u.start_time_s;
  if (unitSpan >= span) {
    // Longer than the window: anchor on its start, a little in from the edge.
    waveView.set(u.start_time_s - span * 0.05, u.start_time_s - span * 0.05 + span);
    return;
  }
  waveView.centerOn((u.start_time_s + u.end_time_s) / 2);
}

export interface SelectOptions {
  /** Move the transport to the unit start (default true). */
  seek?: boolean;
}

export function selectUnit(u: UnitDecisionRecord | null, opts: SelectOptions = {}): void {
  const st = getState();
  st.setSelectedUnit(u);
  if (!u) return;
  if (opts.seek !== false) getPlayer().seek(u.start_time_s);
  revealInView(u);
}

export function clearSelection(): void {
  getState().setSelectedUnit(null);
}

/** Index of the selected unit in time order, or -1. */
export function selectedIndex(ordered: UnitDecisionRecord[] = orderedUnits()): number {
  const sel = getState().selectedUnit;
  if (!sel) return -1;
  const key = unitKey(sel);
  return ordered.findIndex((u) => unitKey(u) === key);
}

/**
 * `[` / `]`. With nothing selected, start from whatever the playhead is
 * sitting on so the first press lands where the user is looking.
 */
export function stepUnit(dir: 1 | -1): void {
  const ordered = orderedUnits();
  if (ordered.length === 0) return;
  const cur = selectedIndex(ordered);
  let next: number;
  if (cur >= 0) {
    next = Math.min(ordered.length - 1, Math.max(0, cur + dir));
  } else {
    const t = getPlayer().time;
    const at = ordered.findIndex((u) => t >= u.start_time_s && t < u.end_time_s);
    if (at >= 0) next = at;
    else {
      const after = ordered.findIndex((u) => u.start_time_s >= t);
      next = dir > 0 ? (after >= 0 ? after : 0) : after > 0 ? after - 1 : ordered.length - 1;
    }
  }
  const u = ordered[next];
  if (u) selectUnit(u);
}
