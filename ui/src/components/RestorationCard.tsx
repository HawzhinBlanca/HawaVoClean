// The restoration section of a done restore run's report (schema v2,
// docs/ui-contract.md Addendum 2), presented the way the .txt sidecar
// presents it (`report/summary.py::generate_human_summary`): the guard's
// verdict first, then the cutoff the run worked above, what happened to the
// segments, and the weights that did it.
//
// A FAIL here is *not* an error state. Guard R rejecting every candidate
// means the Natural master shipped — the safety net working, published in
// the reverted (amber) tone the verdict strip already uses for "the original
// audio was kept", never in the error red reserved for a run that broke.

import { classifyRestorationVerdict, type RestorationSection } from '../api/types';

// `title?: string | undefined` rather than `title?: string`: under
// exactOptionalPropertyTypes the two differ, and the Model row below passes an
// explicit `undefined` when there are no weights to name.
function Row({ k, v, title }: { k: string; v: string; title?: string | undefined }) {
  return (
    <div className="insp-row">
      <span className="k">{k}</span>
      <span className="v mono" title={title}>
        {v}
      </span>
    </div>
  );
}

export function RestorationCard({ rest }: { rest: RestorationSection }) {
  const guard = rest.guard_r ?? {};
  const verdict = guard.verdict ?? 'N/A';
  const cls = classifyRestorationVerdict(guard.verdict);
  const strength = typeof guard.accepted_strength === 'number' ? guard.accepted_strength : null;

  const bw = rest.bandwidth;
  const snr = bw?.evidence?.above_cutoff_snr_db;
  const cutoff = bw
    ? `${bw.effective_cutoff_hz.toFixed(1)} Hz · ${bw.shape} · conf ${bw.confidence.toFixed(2)}` +
      (typeof snr === 'number' ? ` · SNR above ${snr.toFixed(1)} dB` : '') +
      (bw.cutoff_mode === 'manual' ? ' · manual' : '')
    : '—';

  const segs = rest.segments;
  const segments = segs
    ? `restored ${segs.restored} · reduced ${segs.reduced} · reverted ${segs.reverted} · bypassed ${segs.bypassed} · errors ${segs.errors}`
    : '—';

  const model = rest.restorer;
  const weights = model?.weights_sha256;

  return (
    <div className="rest-card" data-verdict={cls} role="group" aria-label="Spectral restoration">
      <div className="rest-head">
        <span className="caps">Spectral restoration</span>
        {rest.speaker_id ? <span className="rest-speaker mono">{rest.speaker_id}</span> : null}
        <span className={`pill ${cls}`}>{verdict}</span>
      </div>
      {cls === 'reverted' ? (
        <p className="rest-ship">
          Guard R rejected every restoration candidate — the Natural master shipped, unchanged.
        </p>
      ) : null}
      <div className="rest-rows">
        <Row
          k="Cutoff"
          v={cutoff}
          title={bw?.cutoff_mode === 'manual' ? 'Cutoff asserted by the operator' : 'Cutoff measured from the audio'}
        />
        <Row k="Segments" v={segments} />
        <Row
          k="Strength"
          v={strength === null ? '—' : strength.toFixed(2)}
          title="Restoration strength Guard R accepted"
        />
        {model?.name || weights ? (
          <Row
            k="Model"
            v={`${model?.name ?? 'unknown'}${weights ? ` · ${weights.slice(0, 16)}…` : ''}`}
            title={weights ? `weights sha256 ${weights}` : undefined}
          />
        ) : null}
      </div>
      {guard.reason ? (
        <div className="insp-reason rest-reason">
          <span className="caps">Guard R</span>
          <p>{guard.reason}</p>
        </div>
      ) : null}
    </div>
  );
}
