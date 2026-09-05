import { useStore } from '../state/store';
import { IconBolt, IconCheck, IconClip } from './Icons';

export function SmartExplanation() {
  const source = useStore((s) => s.source);
  const analyzing = useStore((s) => s.analyzing);
  const original = useStore((s) => s.original);
  const report = useStore((s) => s.report);
  const mode = useStore((s) => s.mode);
  const profile = useStore((s) => s.profile);
  const loudnessMatch = useStore((s) => s.loudnessMatch);
  const gainOffsetDb = useStore((s) => s.gainOffsetDb);

  if (!source) {
    return (
      <div className="smart-explanation empty" role="region" aria-label="Smart workflow guide">
        <div className="se-header">
          <IconClip size={14} />
          <span className="se-title">Step 1: Add Media</span>
        </div>
        <p className="se-body">
          Drop any Kurdish speech audio or video file above to begin. HawaVoClean will inspect noise floor, speech dynamics, and recommend optimal processing.
        </p>
      </div>
    );
  }

  if (analyzing || !original) {
    return (
      <div className="smart-explanation analyzing" role="region" aria-label="Smart acoustic analysis">
        <div className="se-header">
          <span className="se-spinner" aria-hidden="true" />
          <span className="se-title">Analyzing Acoustic Profile…</span>
        </div>
        <p className="se-body">
          Measuring background noise floor, speech frequency spectrum, and dialogue loudness across the take.
        </p>
      </div>
    );
  }

  if (report) {
    const s = report.summary;
    const enhanced = s.enhanced ?? 0;
    const total = s.units_total ?? 0;
    const reverted = s.reverted ?? 0;
    const continuity = (s.continuity_crossfaded ?? 0) + (s.continuity_reverted ?? 0);

    return (
      <div className="smart-explanation done" role="region" aria-label="Smart verification report">
        <div className="se-header">
          <IconCheck size={14} />
          <span className="se-title">Processing Verified</span>
          <span className="se-badge safe">Guard R: Safe</span>
        </div>
        <div className="se-metrics">
          <div className="se-metric">
            <span className="se-metric-label">Enhanced Speech</span>
            <span className="se-metric-val mono">{enhanced} / {total} units</span>
          </div>
          <div className="se-metric">
            <span className="se-metric-label">Preserved Original</span>
            <span className="se-metric-val mono">{reverted} units</span>
          </div>
          {continuity > 0 ? (
            <div className="se-metric">
              <span className="se-metric-label">Continuity Crossfaded</span>
              <span className="se-metric-val mono">{continuity}</span>
            </div>
          ) : null}
        </div>
        <p className="se-body">
          Zero phoneme damage: all speech units verified against linguistic safety bounds.
          {loudnessMatch && Number.isFinite(gainOffsetDb) && Math.abs(gainOffsetDb) > 0.05
            ? ` Level-matched A/B (${gainOffsetDb > 0 ? '+' : ''}${gainOffsetDb.toFixed(1)} dB on Cleaned) active below to eliminate volume bias.`
            : ' Compare Original vs Cleaned below, then export your Master WAV.'}
        </p>
      </div>
    );
  }

  // Source and analysis are ready, awaiting Clean press.
  const nf = original.noise_floor_db;
  const lufs = original.loudness.integrated_lufs;
  let noiseCategory = 'Clean';
  let noiseSeverity: 'low' | 'med' | 'high' = 'low';

  if (nf !== null && Number.isFinite(nf)) {
    if (nf > -45) {
      noiseCategory = `High noise (${nf.toFixed(1)} dB)`;
      noiseSeverity = 'high';
    } else if (nf > -55) {
      noiseCategory = `Moderate noise (${nf.toFixed(1)} dB)`;
      noiseSeverity = 'med';
    } else {
      noiseCategory = `Low noise (${nf.toFixed(1)} dB)`;
      noiseSeverity = 'low';
    }
  }

  const lufsStr = lufs !== null && Number.isFinite(lufs) ? `${lufs.toFixed(1)} LUFS` : '— LUFS';

  return (
    <div className="smart-explanation armed" role="region" aria-label="Smart acoustic assessment">
      <div className="se-header">
        <IconBolt size={14} />
        <span className="se-title">Smart Acoustic Assessment</span>
        <span className={`se-badge ${noiseSeverity}`}>{noiseCategory}</span>
      </div>
      <div className="se-metrics">
        <div className="se-metric">
          <span className="se-metric-label">Loudness</span>
          <span className="se-metric-val mono">{lufsStr}</span>
        </div>
        <div className="se-metric">
          <span className="se-metric-label">Format</span>
          <span className="se-metric-val mono">
            {Math.round(original.sample_rate / 1000)} kHz · {original.channels}ch
          </span>
        </div>
        <div className="se-metric">
          <span className="se-metric-label">Strategy</span>
          <span className="se-metric-val mono">{profile.toUpperCase()} · {mode.toUpperCase()}</span>
        </div>
      </div>
      <p className="se-body">
        {mode === 'restore'
          ? 'HawaRestore-KD will rebuild missing upper harmonics above cutoff with generative reconstruction.'
          : 'Smart Safe will preserve natural Kurdish phonetics, eliminate ambient noise, and verify zero phoneme loss with Guard R.'}
      </p>
    </div>
  );
}
