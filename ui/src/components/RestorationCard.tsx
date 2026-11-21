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
  if (rest.mode === 'smart_safe') {
    const cls = rest.abstained ? 'reverted' : 'enhanced';
    const verdict = rest.abstained ? 'FALLBACK' : 'QUALIFIED';
    const selectedRoute = rest.selected_route ? rest.selected_route.toUpperCase() : 'PRESERVE';
    const conf = typeof rest.confidence === 'number' ? `${(rest.confidence * 100).toFixed(1)}%` : '—';

    return (
      <div className="rest-card" data-verdict={cls} role="group" aria-label="Smart Safe Decision">
        <div className="rest-head">
          <span className="caps">Smart Safe Decision</span>
          <span className="rest-speaker mono">{selectedRoute}</span>
          <span className={`pill ${cls}`}>{verdict}</span>
        </div>
        {rest.abstained ? (
          <p className="rest-ship">
            {rest.fallback_route
              ? `Hard guards triggered abstention — fell back to ${rest.fallback_route.toUpperCase()} (least intervention).`
              : 'Hard guards triggered abstention — fell back to least intervention route.'}
          </p>
        ) : null}
        <div className="rest-rows">
          <Row k="Selected Route" v={selectedRoute} />
          <Row k="Confidence" v={conf} />
          {rest.fallback_route ? <Row k="Fallback Route" v={rest.fallback_route.toUpperCase()} /> : null}
        </div>
        {rest.candidates && rest.candidates.length > 0 ? (
          <div className="rest-candidates" style={{ marginTop: 6 }}>
            <span className="caps" style={{ fontSize: '11px', color: 'var(--fg-3)' }}>
              Candidate Evaluation
            </span>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 3, marginTop: 4 }}>
              {rest.candidates.map((c) => (
                <div
                  key={c.route}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    fontSize: 'var(--fs-sm)',
                    alignItems: 'center',
                  }}
                >
                  <span className="mono">{c.route}</span>
                  <span
                    style={{
                      color:
                        c.status === 'accepted'
                          ? 'var(--green, #4ade80)'
                          : 'var(--amber, #facc15)',
                    }}
                  >
                    {c.status.toUpperCase()}
                    {c.rejection_reason ? ` (${c.rejection_reason})` : ''}
                  </span>
                </div>
              ))}
            </div>
          </div>
        ) : null}
        {rest.reason ? (
          <div className="insp-reason rest-reason">
            <span className="caps">Decision Rationale</span>
            <p>{rest.reason}</p>
          </div>
        ) : null}
      </div>
    );
  }

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
