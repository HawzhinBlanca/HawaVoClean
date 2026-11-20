import { useStore } from '../state/store';
import { IconBolt, IconCheck } from './Icons';
import { sourceHasVideo } from './VideoPreview';

interface HealthChip {
  label: string;
  severity: 'high' | 'good' | 'info';
  title: string;
}

function fmtMinSec(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return '0:00';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

export function AudioHealthCard() {
  const source = useStore((s) => s.source);
  const original = useStore((s) => s.original);
  const cleaned = useStore((s) => s.cleaned);
  const report = useStore((s) => s.report);
  const analyzing = useStore((s) => s.analyzing);
  const previewActive = useStore((s) => s.previewActive);

  if (!source) return null;

  if (analyzing || !original) {
    return (
      <div className="health-card" role="region" aria-label="Audio health check">
        <div className="hc-headline">
          <span className="se-spinner" aria-hidden="true" />
          <span>Listening to “{source.name}” to check background noise and clarity…</span>
        </div>
      </div>
    );
  }

  const chips: HealthChip[] = [];
  const hasVideo = sourceHasVideo(source.path, original);
  const origNoise = original.noise_floor_db;
  const cleanNoise = cleaned?.noise_floor_db ?? null;
  const origLufs = original.loudness.integrated_lufs;
  const origPeak = original.loudness.true_peak_dbtp;

  if (report && cleaned) {
    const enhanced = report.summary.enhanced ?? 0;
    const total = report.summary.units_total ?? 0;
    if (origNoise !== null && cleanNoise !== null && origNoise - cleanNoise >= 3) {
      chips.push({
        label: `Background noise reduced (${Math.round(origNoise - cleanNoise)} dB quieter)`,
        severity: 'good',
        title: `Noise floor improved from ${origNoise.toFixed(1)} dB to ${cleanNoise.toFixed(1)} dB`,
      });
    } else if (enhanced > 0) {
      chips.push({
        label: 'Voice cleaned & balanced',
        severity: 'good',
        title: `${enhanced} of ${total} speech segments enhanced`,
      });
    } else {
      chips.push({
        label: 'Original voice preserved cleanly',
        severity: 'good',
        title: 'Your recording was already clean, so original vocal fidelity was preserved.',
      });
    }
    chips.push({
      label: 'Voice safety verified',
      severity: 'info',
      title: `Guard checked ${total} speech segments automatically.`,
    });
  } else {
    if (origNoise !== null && Number.isFinite(origNoise)) {
      if (origNoise > -45) {
        chips.push({
          label: 'Heavy background noise detected',
          severity: 'high',
          title: `Measured background noise floor: ${origNoise.toFixed(1)} dB`,
        });
      } else if (origNoise > -55) {
        chips.push({
          label: 'Noticeable background noise',
          severity: 'high',
          title: `Measured background noise floor: ${origNoise.toFixed(1)} dB`,
        });
      } else {
        chips.push({
          label: 'Low background noise',
          severity: 'good',
          title: `Measured background noise floor: ${origNoise.toFixed(1)} dB`,
        });
      }
    }

    if (origLufs !== null && Number.isFinite(origLufs)) {
      if (origLufs < -26) {
        chips.push({
          label: 'Voice is quiet — will gently boost',
          severity: 'info',
          title: `Integrated loudness: ${origLufs.toFixed(1)} LUFS`,
        });
      } else if (origLufs > -13) {
        chips.push({
          label: 'Loud recording — headroom protected',
          severity: 'info',
          title: `Integrated loudness: ${origLufs.toFixed(1)} LUFS`,
        });
      }
    }

    if (origPeak !== null && Number.isFinite(origPeak) && origPeak > -0.3) {
      chips.push({
        label: 'Some distortion peaks detected',
        severity: 'high',
        title: `Peak level: ${origPeak.toFixed(1)} dBTP`,
      });
    }

    chips.push({
      label: `${hasVideo ? 'Video with audio' : 'Audio recording'} · ${fmtMinSec(original.duration_s)}`,
      severity: 'info',
      title: `${Math.round(original.sample_rate / 1000)} kHz · ${original.channels}ch`,
    });
  }

  const headline = report
    ? 'Ready to compare — toggle Original / Cleaned below'
    : previewActive
      ? 'Auditioning live preview — click Clean Audio to apply to the full recording'
      : 'Ready to clean — pick a goal or turn the Clean dial';

  return (
    <div className="health-card" role="region" aria-label="Audio health diagnosis">
      <div className="hc-headline">
        {report ? <IconCheck size={16} /> : <IconBolt size={16} />}
        <span>{headline}</span>
      </div>
      <div className="hc-chips" role="list" aria-label="Detected audio traits">
        {chips.map((c) => (
          <span key={c.label} className={`hc-chip sev-${c.severity}`} role="listitem" title={c.title}>
            <span className="hc-dot" aria-hidden="true" />
            <span>{c.label}</span>
          </span>
        ))}
      </div>
    </div>
  );
}
