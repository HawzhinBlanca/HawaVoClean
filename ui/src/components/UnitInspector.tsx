// Per-unit inspection (goal box B4). The verdict strip says *what* happened;
// this panel says *why* — guard verdicts with their scores, the strength the
// core settled on, what finishing did, and the decision reason. With nothing
// selected it stays useful: a designed summary of the run.

import { useMemo } from 'react';
import {
  classifyDecision,
  decisionLabel,
  type GuardScoreValue,
  type UnitDecisionRecord,
} from '../api/types';
import { formatTime } from '../render/ticks';
import { clearSelection, orderedUnits, selectedIndex, stepUnit } from '../state/selection';
import { useStore } from '../state/store';
import { IconCancel } from './Icons';

type ScoreKind = 'ratio' | 'ms' | 'count';

interface ScoreMeta {
  label: string;
  kind: ScoreKind;
}

/** Known guard scores, in reading order; anything else is appended as-is. */
const SCORE_META: Record<string, ScoreMeta> = {
  envelope_correlation: { label: 'Envelope corr', kind: 'ratio' },
  consonant_retention: { label: 'Consonant keep', kind: 'ratio' },
  hf_preservation_ratio: { label: 'HF preserved', kind: 'ratio' },
  spectral_hole_score: { label: 'Spectral holes', kind: 'ratio' },
  musical_noise_score: { label: 'Musical noise', kind: 'ratio' },
  mean_js_div: { label: 'JS div · mean', kind: 'ratio' },
  peak_js_div: { label: 'JS div · peak', kind: 'ratio' },
  timing_drift_ms: { label: 'Timing drift', kind: 'ms' },
  anchor_drift_ms: { label: 'Anchor drift', kind: 'ms' },
  high_conf_anchors: { label: 'Anchors · high conf', kind: 'count' },
  deleted_anchors: { label: 'Anchors · deleted', kind: 'count' },
  substituted_anchors: { label: 'Anchors · swapped', kind: 'count' },
  clipping_samples: { label: 'Clipped samples', kind: 'count' },
};

const SCORE_ORDER = Object.keys(SCORE_META);

function prettify(key: string): string {
  const s = key.replace(/_/g, ' ');
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function metaFor(key: string, value: GuardScoreValue): ScoreMeta {
  const known = SCORE_META[key];
  if (known) return known;
  if (typeof value === 'number' && value >= 0 && value <= 1 && !Number.isInteger(value)) {
    return { label: prettify(key), kind: 'ratio' };
  }
  return { label: prettify(key), kind: 'count' };
}

function formatScore(value: GuardScoreValue, kind: ScoreKind): string {
  if (typeof value === 'boolean') return value ? 'YES' : 'NO';
  if (typeof value === 'string') return value;
  if (!Number.isFinite(value)) return '—';
  if (kind === 'ms') return `${value.toFixed(1)} ms`;
  if (kind === 'count') return Number.isInteger(value) ? String(value) : value.toFixed(1);
  return value.toFixed(3);
}

/** 0..1 scores get a bar; everything else keeps the column but stays empty. */
function barFraction(value: GuardScoreValue, kind: ScoreKind): number | null {
  if (kind !== 'ratio' || typeof value !== 'number' || !Number.isFinite(value)) return null;
  if (value < 0) return 0;
  return Math.min(1, value);
}

function verdictTone(verdict: string | null | undefined): 'pass' | 'fail' | 'none' {
  if (!verdict) return 'none';
  const v = verdict.toLowerCase();
  if (v.includes('pass')) return 'pass';
  if (v.includes('fail') || v.includes('revert') || v.includes('error')) return 'fail';
  return 'none';
}

interface ScoreTableProps {
  title: string;
  verdict: string | null | undefined;
  scores: Record<string, GuardScoreValue> | undefined;
}

function ScoreTable({ title, verdict, scores }: ScoreTableProps) {
  const entries = useMemo(() => {
    if (!scores) return [];
    const keys = Object.keys(scores);
    const known = SCORE_ORDER.filter((k) => k in scores);
    const rest = keys.filter((k) => !(k in SCORE_META)).sort();
    return [...known, ...rest].map((k) => {
      const value = scores[k] as GuardScoreValue;
      const meta = metaFor(k, value);
      return { key: k, meta, value };
    });
  }, [scores]);

  const tone = verdictTone(verdict);
  return (
    <div className={`insp-guard tone-${tone}`}>
      <div className="insp-guard-head">
        <span className="caps">{title}</span>
        <span className={`insp-verdict tone-${tone}`}>{verdict ?? 'NOT RUN'}</span>
      </div>
      {entries.length ? (
        <div className="insp-scores" role="table" aria-label={`${title} scores`}>
          {entries.map(({ key, meta, value }) => {
            const frac = barFraction(value, meta.kind);
            return (
              <div className="insp-score" role="row" key={key}>
                {/* D1 · the label is the row's header, not another cell —
                    a screen reader reading down the value column then has
                    something to say each value *is*. */}
                <span className="k" role="rowheader" title={key}>
                  {meta.label}
                </span>
                <span className="bar" role="cell" aria-hidden="true">
                  {frac === null ? (
                    <i className="none" />
                  ) : (
                    <i className="fill" style={{ width: `${(frac * 100).toFixed(1)}%` }} />
                  )}
                </span>
                <span className="v mono" role="cell">
                  {formatScore(value, meta.kind)}
                </span>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="insp-none">No scores recorded</div>
      )}
    </div>
  );
}

function Row({ k, v, title }: { k: string; v: string; title?: string }) {
  return (
    <div className="insp-row">
      <span className="k">{k}</span>
      <span className="v mono" title={title}>
        {v}
      </span>
    </div>
  );
}

function Summary() {
  const report = useStore((s) => s.report);
  const original = useStore((s) => s.original);
  const cleaned = useStore((s) => s.cleaned);
  const analyzing = useStore((s) => s.analyzing);
  const source = useStore((s) => s.source);

  const s = report?.summary;
  const total = s?.units_total ?? 0;
  const enhanced = s?.enhanced ?? 0;
  const reverted = (s?.reverted ?? 0) + (s?.continuity_reverted ?? 0) + (s?.unverified ?? 0);
  const bypassed = (s?.no_speech ?? 0) + (s?.error_passthrough ?? 0);

  // Prefer the analyses the metrics tiles show, so the two never disagree on
  // screen; the report's own measurements are the fallback.
  const inLufs = original?.loudness.integrated_lufs ?? report?.input.integrated_lufs ?? null;
  const outLufs = cleaned?.loudness.integrated_lufs ?? report?.output.integrated_lufs ?? null;
  const delta =
    typeof inLufs === 'number' && typeof outLufs === 'number' ? outLufs - inLufs : null;

  const hint = report
    ? 'Click a verdict segment to inspect a unit'
    : analyzing
      ? 'Analyzing the clip'
      : source
        ? 'Process the clip to get per-unit guard decisions'
        : 'Load a clip to begin';

  return (
    <div className="insp-empty">
      <div className="insp-empty-lead">
        <span className="caps">Run summary</span>
        <span className="msg">{hint}</span>
        <span className="keys">
          <kbd>[</kbd>
          <kbd>]</kbd>
          <span>step units</span>
          <kbd>?</kbd>
          <span>all shortcuts</span>
        </span>
      </div>
      <div className="insp-stats">
        <div className="stat enhanced">
          <b className="mono">{report ? enhanced : '—'}</b>
          <span className="caps">Enhanced</span>
        </div>
        <div className="stat reverted">
          <b className="mono">{report ? reverted : '—'}</b>
          <span className="caps">Reverted</span>
        </div>
        <div className="stat bypass">
          <b className="mono">{report ? bypassed : '—'}</b>
          <span className="caps">Bypassed</span>
        </div>
        <div className="stat units">
          <b className="mono">{report ? total : '—'}</b>
          <span className="caps">Units</span>
        </div>
        <div className="stat lufs">
          <b className="mono">
            {delta === null
              ? '—'
              : `${delta > 0 ? '+' : delta < 0 ? '−' : '±'}${Math.abs(delta).toFixed(1)}`}
          </b>
          <span className="caps">LUFS Δ</span>
        </div>
      </div>
    </div>
  );
}

function Detail({ unit, index, total }: { unit: UnitDecisionRecord; index: number; total: number }) {
  const cls = classifyDecision(unit.final_decision);
  const dur = unit.end_time_s - unit.start_time_s;
  const strength = typeof unit.chosen_strength === 'number' ? unit.chosen_strength : null;
  const actions = unit.finish_actions ?? [];
  return (
    <div className="insp-grid">
      <div className="insp-col insp-ident">
        <div className="insp-idhead">
          <span className="uid mono">
            UNIT {String(unit.unit_id).padStart(2, '0')}
            <span className="of"> · {index + 1}/{total}</span>
          </span>
          {/* The decision badge is the same object as a verdict-strip segment,
              so it is drawn from the same `--seg` recipe rather than from an
              inline full-saturation swatch. */}
          <span className={`pill ${cls}`}>{decisionLabel(unit.final_decision)}</span>
        </div>
        <Row k="Channel" v={String(unit.channel)} />
        <Row
          k="Range"
          v={`${formatTime(unit.start_time_s, true)} → ${formatTime(unit.end_time_s, true)}`}
          title={`${unit.start_time_s.toFixed(3)} s → ${unit.end_time_s.toFixed(3)} s`}
        />
        <Row k="Duration" v={`${dur.toFixed(2)} s`} title={`${unit.start_sample} → ${unit.end_sample} samples`} />
        <Row k="Speech" v={unit.is_speech ? 'YES' : 'NO'} />
        <Row
          k="Runtime"
          v={typeof unit.runtime_ms === 'number' ? `${Math.round(unit.runtime_ms)} ms` : '—'}
        />
        <div className="insp-row strength">
          <span className="k">Strength</span>
          <span className="strength-bar" aria-hidden="true">
            <i style={{ width: `${Math.max(0, Math.min(1, strength ?? 0)) * 100}%` }} />
          </span>
          <span className="v mono">{strength === null ? '—' : strength.toFixed(2)}</span>
        </div>
        <div className="insp-finish">
          <span className="caps">Finishing</span>
          <span className="preset">{unit.finish_preset_applied ?? 'none'}</span>
          <div className="chips">
            {actions.length ? (
              actions.map((a) => (
                <span className="chip" key={a} title={a}>
                  {a}
                </span>
              ))
            ) : (
              <span className="chip empty">no actions</span>
            )}
          </div>
        </div>
        {unit.decision_reason ? (
          <div className="insp-reason">
            <span className="caps">Reason</span>
            <p>{unit.decision_reason}</p>
          </div>
        ) : null}
      </div>
      <ScoreTable title="Guard A" verdict={unit.guard_a_verdict} scores={unit.guard_a_scores} />
      {unit.guard_b_verdict || unit.guard_b_scores ? (
        <ScoreTable title="Guard B" verdict={unit.guard_b_verdict} scores={unit.guard_b_scores} />
      ) : (
        <div className="insp-guard tone-none">
          <div className="insp-guard-head">
            <span className="caps">Guard B</span>
            <span className="insp-verdict tone-none">NOT RUN</span>
          </div>
          <div className="insp-none">
            The second guard runs only when the first one is inconclusive.
          </div>
        </div>
      )}
    </div>
  );
}

export function UnitInspector() {
  const selected = useStore((s) => s.selectedUnit);
  const report = useStore((s) => s.report);
  const setShortcutsOpen = useStore((s) => s.setShortcutsOpen);

  const ordered = useMemo(() => orderedUnits(), [report]);
  const index = selected ? selectedIndex(ordered) : -1;
  const canStep = ordered.length > 0;

  return (
    <section
      className="panel inspector"
      aria-label="Unit inspector"
      data-selected={selected ? 'true' : 'false'}
    >
      <div className="panel-head">
        <div className="panel-title">
          <span>Unit inspector</span>
          {selected ? (
            <span className="sub">
              · unit {index >= 0 ? index + 1 : '?'} of {ordered.length}
            </span>
          ) : null}
        </div>
        <div className="insp-tools">
          <button
            type="button"
            className="insp-step"
            onClick={() => stepUnit(-1)}
            disabled={!canStep}
            title="Previous unit ( [ )"
            aria-label="Previous unit"
          >
            [
          </button>
          <button
            type="button"
            className="insp-step"
            onClick={() => stepUnit(1)}
            disabled={!canStep}
            title="Next unit ( ] )"
            aria-label="Next unit"
          >
            ]
          </button>
          <button
            type="button"
            className="insp-clear"
            onClick={clearSelection}
            disabled={!selected}
            title="Clear selection (Esc)"
            aria-label="Clear selection"
          >
            <IconCancel size={12} />
          </button>
          <button
            type="button"
            className="insp-keys"
            onClick={() => setShortcutsOpen(true)}
            title="Keyboard shortcuts (?)"
          >
            <span className="qm">?</span>
            <span className="lbl">KEYS</span>
          </button>
        </div>
      </div>
      {/* D1 · the guard score tables are taller than the panel, so this box
          scrolls — and a scroll container with no focus stop cannot be
          scrolled from the keyboard at all (axe: scrollable-region-focusable).
          It is a focus stop with a name of its own; the arrows and page keys
          it takes are the browser's own, not a binding of ours. */}
      <div className="insp-body" tabIndex={0} role="group" aria-label="Unit detail">
        {selected ? (
          <Detail unit={selected} index={index >= 0 ? index : 0} total={ordered.length} />
        ) : (
          <Summary />
        )}
      </div>
    </section>
  );
}
