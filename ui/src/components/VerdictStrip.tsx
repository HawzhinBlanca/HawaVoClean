import { useCallback, useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type MouseEvent } from 'react';
import { classifyDecision, decisionLabel, type UnitDecisionRecord, type VerdictClass } from '../api/types';
import { formatTime } from '../render/ticks';
import { waveView } from '../render/viewWindow';
import { selectUnit, unitKey } from '../state/selection';
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

const PILL_BG: Record<VerdictClass, string> = {
  enhanced: 'var(--cyan)',
  reverted: 'var(--amber)',
  passthrough: 'var(--fg-3)',
  error: 'var(--err)',
};

interface Hover {
  unit: UnitDecisionRecord;
  x: number;
  y: number;
}

/** Per-segment custom properties; the track supplies --vs/--vd. */
type SegStyle = CSSProperties & { '--t0': number; '--t1': number };

export function VerdictStrip() {
  const report = useStore((s) => s.report);
  const duration = useStore((s) => s.duration);
  const setHighlight = useStore((s) => s.setHighlight);
  const selected = useStore((s) => s.selectedUnit);
  const [hover, setHover] = useState<Hover | null>(null);
  const tipRef = useRef<HTMLDivElement | null>(null);
  const trackRef = useRef<HTMLDivElement | null>(null);

  const units = report?.units ?? [];
  const total = duration > 0 ? duration : units.length ? (units[units.length - 1]?.end_time_s ?? 0) : 0;

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

  const onEnter = useCallback(
    (u: UnitDecisionRecord, e: MouseEvent) => {
      setHover({ unit: u, x: e.clientX, y: e.clientY });
      setHighlight({ start: u.start_time_s, end: u.end_time_s });
    },
    [setHighlight],
  );
  const onMove = useCallback((e: MouseEvent) => {
    setHover((h) => (h ? { ...h, x: e.clientX, y: e.clientY } : h));
  }, []);
  const onLeave = useCallback(() => {
    setHover(null);
    // Hover only borrows the highlight; the selection keeps it.
    const sel = useStore.getState().selectedUnit;
    setHighlight(sel ? { start: sel.start_time_s, end: sel.end_time_s } : null);
  }, [setHighlight]);

  useEffect(() => {
    if (!report) {
      setHover(null);
      setHighlight(null);
    }
  }, [report, setHighlight]);

  // Keep the tooltip on-screen.
  useEffect(() => {
    const tip = tipRef.current;
    if (!tip || !hover) return;
    const r = tip.getBoundingClientRect();
    let x = hover.x + 14;
    let y = hover.y - r.height - 14;
    if (x + r.width > window.innerWidth - 8) x = hover.x - r.width - 14;
    if (y < 8) y = hover.y + 18;
    tip.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
  }, [hover]);

  const enhanced = units.filter((u) => u.final_decision === 'enhanced').length;

  return (
    <div className="verdict">
      <div className="label">
        <span className="caps">Verdicts</span>
        <span className="count">
          {units.length ? `${units.length} units · ${enhanced} enhanced` : '—'}
        </span>
      </div>
      <div className="verdict-track" ref={trackRef} onMouseLeave={onLeave}>
        {units.length && total > 0 ? (
          units.map((u) => {
            const cls = classifyDecision(u.final_decision);
            const style: SegStyle = { '--t0': u.start_time_s, '--t1': u.end_time_s };
            const isSel = selected ? unitKey(selected) === unitKey(u) : false;
            return (
              <button
                type="button"
                // Hundreds of units must not become hundreds of tab stops:
                // the keyboard path to a unit is `[` / `]` (see App).
                tabIndex={-1}
                key={`${u.channel}-${u.unit_id}`}
                className={`verdict-seg ${cls}${hover?.unit === u ? ' hot' : ''}${isSel ? ' sel' : ''}`}
                style={style}
                aria-label={`Unit ${u.unit_id}, channel ${u.channel}, ${decisionLabel(u.final_decision)}`}
                aria-pressed={isSel}
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
          })
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
              Unit {hover.unit.unit_id}
              <span style={{ color: 'var(--fg-3)' }}> · ch {hover.unit.channel}</span>
            </span>
            <span className="pill" style={{ background: PILL_BG[classifyDecision(hover.unit.final_decision)] }}>
              {decisionLabel(hover.unit.final_decision)}
            </span>
          </div>
          <div className="row">
            <span className="k">Range</span>
            <span className="v">
              {formatTime(hover.unit.start_time_s, true)} → {formatTime(hover.unit.end_time_s, true)}
            </span>
          </div>
          <div className="row">
            <span className="k">Guard A</span>
            <span className="v">
              {hover.unit.guard_a_verdict}
              {hover.unit.guard_b_verdict ? ` · B ${hover.unit.guard_b_verdict}` : ''}
            </span>
          </div>
          {typeof hover.unit.chosen_strength === 'number' && hover.unit.chosen_strength > 0 ? (
            <div className="row">
              <span className="k">Strength</span>
              <span className="v">{hover.unit.chosen_strength.toFixed(2)}</span>
            </div>
          ) : null}
          {hover.unit.decision_reason ? <div className="reason">{hover.unit.decision_reason}</div> : null}
          {hover.unit.guard_a_scores && Object.keys(hover.unit.guard_a_scores).length ? (
            <div className="scores">
              {SCORE_KEYS.filter(([k]) => hover.unit.guard_a_scores && k in hover.unit.guard_a_scores).map(
                ([k, label]) => (
                  <div className="row" key={k}>
                    <span className="k">{label}</span>
                    <span className="v">{fmtScore(hover.unit.guard_a_scores?.[k] ?? '')}</span>
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
