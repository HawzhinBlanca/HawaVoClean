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
  const isMp4 = Boolean(source && source.name.toLowerCase().endsWith('.mp4'));

  if (!source) {
    return (
      <div className="smart-explanation empty" role="region" aria-label="Smart workflow guide">
        <div className="se-header">
          <IconClip size={14} />
          <span className="se-title">Step 1: Add Media</span>
        </div>
        <p className="se-body">
          Drop an audio or video recording above to begin. Describe your edit in the assistant, or use the dialogue cleanup controls.
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

  if (report?.separation) {
    const separation = report.separation;
    return (
      <div className="smart-explanation done" role="region" aria-label="Sound separation result">
        <div className="se-header">
          <IconCheck size={14} />
          <span className="se-title">Sound separation complete</span>
          <span className="se-badge med">Experimental</span>
        </div>
        <p className="se-body">
          {separation.operation === 'extract' ? 'Kept' : 'Removed'}: {separation.prompt}.
          {' '}Signal checks cover the output format and timeline. They do not verify word retention or which sounds the model selected.
          {' '}Compare Original vs Cleaned below and listen before exporting.
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
          <span className="se-title">Processing complete</span>
          <span className="se-badge safe">Guard decisions recorded</span>
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
          Signal checks guided which speech units were enhanced or preserved. They do not verify spoken words.
          {loudnessMatch && Number.isFinite(gainOffsetDb) && Math.abs(gainOffsetDb) > 0.05
            ? ` Level-matched A/B (${gainOffsetDb > 0 ? '+' : ''}${gainOffsetDb.toFixed(1)} dB on Cleaned) active below to eliminate volume bias.`
            : ' Compare Original vs Cleaned below, then export your Master WAV.'}
        </p>
        {isMp4 ? (
          <div className="se-mp4-notice" role="note" aria-label="MP4 dialogue extraction note">
            <span className="se-mp4-tag">MP4 Dialogue Master</span>
            <p className="se-mp4-detail">
              Master WAV contains the cleaned dialogue extracted from your MP4 container, sample-accurate and ready for timeline import.
            </p>
          </div>
        ) : null}
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
          <span className="se-metric-val mono">{profile.replace('_', ' ').toUpperCase()} · {mode.toUpperCase()}</span>
        </div>
      </div>
      <p className="se-body">
        {mode === 'restore'
          ? 'HawaRestore-KD will rebuild missing upper harmonics above cutoff with generative reconstruction.'
          : 'Dialogue cleanup reduces noise with signal checks and preserves the original where an enhancement is rejected. Listen to the result to assess speech quality.'}
      </p>
      {isMp4 ? (
        <div className="se-mp4-notice" role="note" aria-label="MP4 dialogue extraction note">
          <span className="se-mp4-tag">MP4 Dialogue Isolated</span>
          <p className="se-mp4-detail">
            Audio stream extracted directly from video container without video recoding. Original frame rate and timeline sync preserved.
          </p>
        </div>
      ) : null}
    </div>
  );
}
