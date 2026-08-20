// Unit selection: the one place that knows what "select a unit" means —
// store state, waveform highlight, transport seek, and bringing the unit into
// the visible zoom window. Both the verdict strip (click) and the keyboard map
// (`[` / `]`) go through here so the two can never drift apart.
//
// It is also the one place that knows how many channels a report decided on.
// A split-speakers run emits a full set of units *per channel*, and those sets
// overlap in time — ch0 0.000–20.570 and ch1 0.000–20.474 are two different
// decisions about the same seconds. Everything on screen that shows a unit
// (the strip's lanes, the waveform's selection band, the inspector's badge)
// asks `reportChannels()` / `channelName()` here, so no two of them can end up
// naming the channels differently.

import type { UnitDecisionRecord } from '../api/types';
import { getPlayer } from '../audio/player';
import { waveView } from '../render/viewWindow';
import { getState } from './store';

export function unitKey(u: UnitDecisionRecord): string {
  return `${u.channel}-${u.unit_id}`;
}

/**
 * The channels the report actually decided on, ascending.
 *
 * Read from the units, never from `input.channels`: a dual-mono run has two
 * channels in the file but one set of decisions (the engine processes ch0 and
 * duplicates it), and that report must keep the single-lane mono look.
 */
export function reportChannels(units: UnitDecisionRecord[] = getState().report?.units ?? []): number[] {
  const seen = new Set<number>();
  for (const u of units) seen.add(u.channel);
  return [...seen].sort((a, b) => a - b);
}

export interface ChannelName {
  /** One or two glyphs for a lane tag or a badge. */
  short: string;
  /** A whole phrase for an aria-label or a tooltip. */
  long: string;
}

/**
 * What to call channel `ch` when the report carries `channels`.
 *
 * Two channels are L and R — the only naming a split-speakers stereo file can
 * have. Anything else is numbered, because a 3+ channel report (which only the
 * CLI can produce) has no agreed speaker order to name.
 */
export function channelName(ch: number, channels: number[] = reportChannels()): ChannelName {
  if (channels.length <= 1) return { short: 'M', long: 'the only channel' };
  if (channels.length === 2) {
    const i = channels.indexOf(ch);
    if (i === 0) return { short: 'L', long: 'left channel' };
    if (i === 1) return { short: 'R', long: 'right channel' };
  }
  return { short: `C${ch}`, long: `channel ${ch}` };
}

/**
 * Reading order for the strip's lanes and for `[` / `]`: channel first, then
 * time.
 *
 * With one channel this *is* time order, which is what it has always been.
 * With two it walks the top lane left to right and then the bottom one, which
 * is the order the lanes are drawn in — so a press of `]` always moves right
 * inside one lane, or steps down to the start of the next. Interleaving the
 * channels by time instead would flip lanes on nearly every press (ch1 20.474
 * then ch0 20.570 is a 96 ms move and a lane change) and would visit two units
 * that both start at 0.000 back to back without the playhead moving at all.
 */
export function orderedUnits(): UnitDecisionRecord[] {
  const units = getState().report?.units ?? [];
  return [...units].sort(
    (a, b) =>
      a.channel - b.channel || a.start_time_s - b.start_time_s || a.unit_id - b.unit_id,
  );
}

/** The highlight a unit asks for: its range, plus its channel when there is more than one. */
export function highlightFor(
  u: UnitDecisionRecord | null,
  channels: number[] = reportChannels(),
): { start: number; end: number; channel?: number } | null {
  if (!u) return null;
  const range = { start: u.start_time_s, end: u.end_time_s };
  return channels.length > 1 ? { ...range, channel: u.channel } : range;
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
  // `setSelectedUnit` lights the range; on a multi-channel report the band
  // also has to say *whose* range it is, or a ch0 unit and the ch1 unit
  // overlapping it paint the identical highlight.
  const channels = reportChannels();
  if (channels.length > 1) st.setHighlight(highlightFor(u, channels));
  if (opts.seek !== false) getPlayer().seek(u.start_time_s);
  revealInView(u);
}

export function clearSelection(): void {
  getState().setSelectedUnit(null);
}

/** Index of the selected unit in reading order, or -1. */
export function selectedIndex(ordered: UnitDecisionRecord[] = orderedUnits()): number {
  const sel = getState().selectedUnit;
  if (!sel) return -1;
  const key = unitKey(sel);
  return ordered.findIndex((u) => unitKey(u) === key);
}

/**
 * Where `[` / `]` starts when nothing is selected: from whatever the playhead
 * is sitting on, so the first press lands where the user is looking.
 *
 * The search is confined to the *first* lane. In channel-major order that lane
 * is a contiguous prefix, and it is the one the eye starts on; a playhead at
 * 30 s sits inside a unit of every channel at once, so "the unit the playhead
 * is in" is only an answer once a lane is chosen. Once something *is*
 * selected, its own lane is the one being walked and this never runs.
 */
function bootstrapIndex(ordered: UnitDecisionRecord[], dir: 1 | -1, t: number): number {
  const firstChannel = ordered[0]?.channel ?? 0;
  let lane = ordered.length;
  for (let i = 0; i < ordered.length; i++) {
    if (ordered[i]?.channel !== firstChannel) {
      lane = i;
      break;
    }
  }
  for (let i = 0; i < lane; i++) {
    const u = ordered[i];
    if (u && t >= u.start_time_s && t < u.end_time_s) return i;
  }
  // `after` = first unit of the lane starting at or after the playhead; -1
  // when the playhead is past every one of them. Both directions CLAMP at the
  // ends — a press at the last unit must not silently jump back to the first.
  let after = -1;
  for (let i = 0; i < lane; i++) {
    if ((ordered[i]?.start_time_s ?? 0) >= t) {
      after = i;
      break;
    }
  }
  if (dir > 0) return after >= 0 ? after : lane - 1;
  return after === -1 ? lane - 1 : Math.max(0, after - 1);
}

/** `[` / `]` — one step through the reading order, clamped at both ends. */
export function stepUnit(dir: 1 | -1): void {
  const ordered = orderedUnits();
  if (ordered.length === 0) return;
  const cur = selectedIndex(ordered);
  const next =
    cur >= 0
      ? Math.min(ordered.length - 1, Math.max(0, cur + dir))
      : bootstrapIndex(ordered, dir, getPlayer().time);
  const u = ordered[next];
  if (u) selectUnit(u);
}
