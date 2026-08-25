import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type MouseEvent,
} from 'react';
import { classifyDecision, decisionLabel, type UnitDecisionRecord } from '../api/types';
import { formatTime } from '../render/ticks';
import { waveView } from '../render/viewWindow';
import { unitNoun } from '../state/plural';
import { channelName, highlightFor, reportChannels, selectUnit, unitKey } from '../state/selection';
import { useStore } from '../state/store';

const SCORE_KEYS: Array<[string, string]> = [
  ['envelope_correlation', 'env corr'],
  ['timing_drift_ms', 'drift ms'],
  ['spectral_hole_score', 'spec hole'],
  ['musical_noise_score', 'mus noise'],
  ['consonant_retention', 'cons ret'],
  ['hf_preservation_ratio', 'hf keep'],
  ['mean_js_div', 'js div'],
  ['clipping_samples', 'clips'],
];

function fmtScore(v: number | string | boolean): string {
  if (typeof v === 'number') {
    if (Number.isInteger(v)) return String(v);
    return Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(3);
  }
  return String(v);
}

/** PASS / FAIL / not-run, as a class the tooltip can colour. */
function guardTone(v: string | null | undefined): 'pass' | 'fail' | 'none' {
  if (!v) return 'none';
  const s = v.toLowerCase();
  if (s.includes('pass')) return 'pass';
  if (s.includes('fail') || s.includes('revert')) return 'fail';
  return 'none';
}

/**
 * Everything the hover card says, as one sentence. Kept beside the card's own
 * markup so the two cannot drift apart.
 */
function segmentLabel(u: UnitDecisionRecord, channels: number[]): string {
  const parts = [
    `Unit ${u.unit_id}`,
    channels.length > 1 ? channelName(u.channel, channels).long : `channel ${u.channel}`,
    decisionLabel(u.final_decision),
    `${formatTime(u.start_time_s, true)} to ${formatTime(u.end_time_s, true)}`,
  ];
  if (u.guard_a_verdict) parts.push(`guard A ${u.guard_a_verdict}`);
  if (u.guard_b_verdict) parts.push(`guard B ${u.guard_b_verdict}`);
  if (typeof u.chosen_strength === 'number' && u.chosen_strength > 0) {
    parts.push(`strength ${u.chosen_strength.toFixed(2)}`);
  }
  if (u.decision_reason) parts.push(u.decision_reason);
  return parts.join(', ');
}

/**
 * Where the pointer is. Deliberately a ref, not state.
 *
 * This used to be `{ unit, x, y }` in `useState`, updated from `onMouseMove`.
 * That made every pixel of pointer travel across the strip a React render of
 * the whole component — hundreds of `<button>`s, each rebuilding its
 * `aria-label` through `segmentLabel()` — to move one card. The card is the
 * only thing that depends on the coordinates, and it is positioned by writing
 * `style.transform` directly, so the coordinates never need to reach the
 * reconciler at all. Only *which unit* is hovered is state, because that is
 * what changes the rendered tree.
 */
interface Cursor {
  x: number;
  y: number;
}

/** Per-segment custom properties; the track supplies --vs/--vd. */
type SegStyle = CSSProperties & { '--t0': number; '--t1': number };
/** Lane geometry: which sub-track this is, and how many there are. */
type LaneStyle = CSSProperties & { '--lane': number };
type TrackStyle = CSSProperties & { '--lanes': number };

export function VerdictStrip() {
  const report = useStore((s) => s.report);
  const duration = useStore((s) => s.duration);
  const setHighlight = useStore((s) => s.setHighlight);
  const selected = useStore((s) => s.selectedUnit);
  const [hover, setHover] = useState<UnitDecisionRecord | null>(null);
  const cursor = useRef<Cursor>({ x: 0, y: 0 });
  const tipRef = useRef<HTMLDivElement | null>(null);
  const trackRef = useRef<HTMLDivElement | null>(null);

  const units = report?.units ?? [];
  const total = duration > 0 ? duration : units.length ? (units[units.length - 1]?.end_time_s ?? 0) : 0;

  // A7/B4 · a split-speakers run decides *per channel*, and those decisions
  // overlap in time. Laid out in one lane they stack on top of each other:
  // measured on a real 2-channel run, the last channel painted was topmost on
  // 687 of the track's 689 px, so half the report had no hover, no click and
  // no selection at all. One lane per channel is the fix, and it is only ever
  // reached when the report really has more than one — a mono report (and a
  // dual-mono one, which decides on ch0 alone) keeps the single-lane strip
  // pixel for pixel.
  const channels = useMemo(() => reportChannels(report?.units ?? []), [report]);
  const multi = channels.length > 1;

  // The strip is a time-axis-linked element: it must show exactly the window
  // the waveform shows. Segments carry their own times as custom properties
  // and CSS places them against the track's window, so a zoom or pan is one
  // DOM write here — no React render, no per-frame work.
  const syncWindow = useCallback(() => {
    const el = trackRef.current;
    if (!el) return;
    const dur = waveView.duration;
    const start = dur > 0 ? waveView.start_s : 0;
    const span = dur > 0 ? waveView.span : total;
    el.style.setProperty('--vs', String(start));
    el.style.setProperty('--vd', String(span > 0 ? span : 1));
  }, [total]);

  useLayoutEffect(() => {
    syncWindow();
    return waveView.subscribe(syncWindow);
  }, [syncWindow]);

  // Keep the tooltip on-screen, and tell it where its arrow goes.
  //
  // The card is centred over the cursor and sits above the strip; when there
  // is no room above it flips below, and when it would run off either edge it
  // slides back in. The arrow is then placed at the cursor's offset *inside*
  // the card, clamped away from the rounded corners, so it points at the
  // segment however far the card had to slide.
  const placeTip = useCallback(() => {
    const tip = tipRef.current;
    if (!tip) return;
    const { x: cx, y: cy } = cursor.current;
    const r = tip.getBoundingClientRect();
    const M = 8; // viewport margin
    const GAP = 13; // clearance for the arrow
    let x = cx - r.width / 2;
    x = Math.max(M, Math.min(x, window.innerWidth - r.width - M));
    let y = cy - r.height - GAP;
    let place: 'top' | 'bottom' = 'top';
    if (y < M) {
      y = cy + GAP;
      place = 'bottom';
    }
    tip.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
    tip.dataset.place = place;
    const arrow = Math.max(14, Math.min(cx - x, r.width - 14));
    tip.style.setProperty('--tip-arrow', `${Math.round(arrow)}px`);
  }, []);

  const onEnter = useCallback(
    (u: UnitDecisionRecord, e: MouseEvent) => {
      cursor.current = { x: e.clientX, y: e.clientY };
      setHover(u);
      setHighlight(highlightFor(u, channels));
    },
    [setHighlight, channels],
  );
  const onMove = useCallback(
    (e: MouseEvent) => {
      cursor.current = { x: e.clientX, y: e.clientY };
      // One style write, no render. The card is pinned to the pointer rather
      // than trailing it by a frame.
      placeTip();
    },
    [placeTip],
  );
  const onLeave = useCallback(() => {
    setHover(null);
    // Hover only borrows the highlight; the selection keeps it.
    setHighlight(highlightFor(useStore.getState().selectedUnit, channels));
  }, [setHighlight, channels]);

  useEffect(() => {
    if (!report) {
      setHover(null);
      setHighlight(null);
    }
  }, [report, setHighlight]);

  // Before paint, not after. `.tooltip` rests at `position: fixed; top: 0;
  // left: 0` with no transform, so positioning it in a passive effect
  // committed one frame of a 292 px card with an --elev-3 shadow sitting in
  // the top-left corner of the window, arrow unplaced, on every single hover.
  useLayoutEffect(() => {
    if (hover) placeTip();
  }, [hover, placeTip]);

  const enhanced = units.filter((u) => u.final_decision === 'enhanced').length;

  // One segment, wherever it lives — directly in the track for a mono report,
  // inside its channel's lane for a multi-channel one. Written once so the two
  // cases cannot drift apart.
  const segment = (u: UnitDecisionRecord) => {
    const cls = classifyDecision(u.final_decision);
    const style: SegStyle = { '--t0': u.start_time_s, '--t1': u.end_time_s };
    const isSel = selected ? unitKey(selected) === unitKey(u) : false;
    return (
      <button
        type="button"
        // D1 · deliberately out of the tab order, and this is the
        // justification. A report can carry hundreds of units; making
        // each one a tab stop would put the whole rest of the screen
        // behind a few hundred presses of Tab. They keep `button`
        // semantics and a full name, so a screen reader still reaches
        // and activates them through its own element navigation, and
        // every unit has a *sequential* keyboard route through `[` and
        // `]` (and the inspector's own two buttons, which are tab
        // stops) — which is a better one anyway, because it selects,
        // seeks and pans the view in one press.
        tabIndex={-1}
        key={unitKey(u)}
        className={`verdict-seg ${cls}${hover === u ? ' hot' : ''}${isSel ? ' sel' : ''}`}
        style={style}
        // D1 · the hover tooltip must not be the only way to read a
        // unit's decision. Two routes replace it: the inspector, which
        // shows strictly more on selection (`[` / `]` or a click), and
        // this name, which carries the tooltip's own fields — range,
        // both guard verdicts, strength — so a screen reader gets the
        // card's content without a pointer ever entering the strip.
        aria-label={segmentLabel(u, channels)}
        // `aria-current`, not `aria-pressed`. A segment selects a unit; it can
        // never be un-pressed, and a screen reader announcing "not pressed" on
        // every one of a few hundred segments describes a toggle that does not
        // exist. JobHistory.tsx uses aria-current for the identical "this is
        // the one you are looking at" semantic.
        aria-current={isSel ? 'true' : undefined}
        onMouseEnter={(e) => onEnter(u, e)}
        onMouseMove={onMove}
        onClick={(e) => {
          // Keep the keyboard on the transport: a segment is a click
          // target, not a focus stop (Space must still play/pause).
          e.currentTarget.blur();
          selectUnit(u);
        }}
      />
    );
  };

  const trackStyle: TrackStyle | undefined = multi ? { '--lanes': channels.length } : undefined;

  return (
    <div className="verdict">
      <div className="label">
        <span className="caps">Verdicts</span>
        <span className="count">
          {units.length ? `${units.length} ${unitNoun(units.length)} · ${enhanced} enhanced` : '—'}
        </span>
      </div>
      <div
        className="verdict-track"
        ref={trackRef}
        // D1 · the strip is one selectable run of units over one time axis;
        // naming the group is what lets a screen-reader user understand the
        // segments inside it as one thing. (It said "a group of toggles" while
        // the segments carried `aria-pressed`; they carry `aria-current` now,
        // because selecting a unit is not a press that can be undone.)
        role="group"
        aria-label={
          multi
            ? `Per-unit guard verdicts, one lane per channel (${channels.length} channels)`
            : 'Per-unit guard verdicts'
        }
        data-lanes={channels.length || 1}
        style={trackStyle}
        onMouseLeave={onLeave}
      >
        {units.length && total > 0 ? (
          multi ? (
            channels.map((ch, i) => {
              const name = channelName(ch, channels);
              const mine = units.filter((u) => u.channel === ch);
              const laneStyle: LaneStyle = { '--lane': i };
              return (
                <div
                  className="verdict-lane"
                  key={ch}
                  style={laneStyle}
                  role="group"
                  aria-label={`${name.long}, ${mine.length} ${unitNoun(mine.length)}`}
                >
                  {/* The tag rides *inside* the track rather than in a gutter
                      beside it: the track's box is the waveform canvas's box to
                      0.000 px, and a label column would move one of them. It is
                      pointer-events: none, so it can never take a click or a
                      hover away from the segment underneath it — the scan that
                      caught this bug would otherwise find the tag topmost. */}
                  {/* No `title`: this span cannot take a pointer, so a
                      tooltip on it could never fire. The lane's own group
                      name carries the words. */}
                  <span className="lane-tag" aria-hidden="true">
                    {name.short}
                  </span>
                  {mine.map(segment)}
                </div>
              );
            })
          ) : (
            units.map(segment)
          )
        ) : (
          <div className="verdict-empty">
            {report ? 'Report has no units' : 'Per-unit guard decisions appear here after processing'}
          </div>
        )}
      </div>
      <div className="verdict-legend" aria-hidden="true">
        <span>
          <i className="verdict-seg enhanced" /> Enhanced
        </span>
        <span>
          <i className="verdict-seg reverted" /> Reverted
        </span>
        <span>
          <i className="verdict-seg passthrough" /> Bypass
        </span>
        <span>
          <i className="verdict-seg error" /> Error
        </span>
      </div>
      {hover ? (
        <div ref={tipRef} className="tooltip" role="tooltip">
          <div className="head">
            <span className="mono" style={{ color: 'var(--fg)' }}>
              Unit {hover.unit_id}
              <span style={{ color: 'var(--fg-3)' }}>
                {' · '}
                {multi ? channelName(hover.channel, channels).short : `ch ${hover.channel}`}
              </span>
            </span>
            {/* same badge recipe as the inspector and the strip itself */}
            <span className={`pill ${classifyDecision(hover.final_decision)}`}>
              {decisionLabel(hover.final_decision)}
            </span>
          </div>
          <div className="row">
            <span className="k">Range</span>
            <span className="v">
              {formatTime(hover.start_time_s, true)} → {formatTime(hover.end_time_s, true)}
            </span>
          </div>
          <div className="row">
            <span className="k">Guards</span>
            <span className="verdicts">
              <span className={`gv ${guardTone(hover.guard_a_verdict)}`}>
                A <b>{hover.guard_a_verdict ?? '—'}</b>
              </span>
              <span className={`gv ${guardTone(hover.guard_b_verdict)}`}>
                B <b>{hover.guard_b_verdict ?? '—'}</b>
              </span>
            </span>
          </div>
          {typeof hover.chosen_strength === 'number' && hover.chosen_strength > 0 ? (
            <div className="row">
              <span className="k">Strength</span>
              <span className="v">{hover.chosen_strength.toFixed(2)}</span>
            </div>
          ) : null}
          {hover.decision_reason ? <div className="reason">{hover.decision_reason}</div> : null}
          {hover.guard_a_scores && Object.keys(hover.guard_a_scores).length ? (
            <div className="scores">
              {SCORE_KEYS.filter(([k]) => hover.guard_a_scores && k in hover.guard_a_scores).map(
                ([k, label]) => (
                  <div className="row" key={k}>
                    <span className="k">{label}</span>
                    <span className="v">{fmtScore(hover.guard_a_scores?.[k] ?? '')}</span>
                  </div>
                ),
              )}
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
