import { updateRackSettings } from '../state/actions';
import { useI18n } from '../state/i18n';
import { scheduleLivePreview } from '../state/liveAudition';
import { isProfileUnavailable, jobInFlight, useStore } from '../state/store';
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
  const { t, locale } = useI18n();
  const isCkb = locale === 'ckb';
  const source = useStore((s) => s.source);
  const original = useStore((s) => s.original);
  const cleaned = useStore((s) => s.cleaned);
  const report = useStore((s) => s.report);
  const analyzing = useStore((s) => s.analyzing);
  const previewActive = useStore((s) => s.previewActive);
  const rackSettings = useStore((s) => s.rackSettings);
  const capabilities = useStore((s) => s.capabilities);
  const setProfile = useStore((s) => s.setProfile);
  const pushToast = useStore((s) => s.pushToast);
  const busy = useStore((s) => jobInFlight(s.job) || s.analyzing);

  if (!source) return null;

  if (analyzing || !original) {
    return (
      <div className="health-card" role="region" aria-label="Audio health check">
        <div className="hc-headline">
          <span className="se-spinner" aria-hidden="true" />
          <span>
            {t.healthListeningPrefix}
            {source.name}
            {t.healthListeningSuffix}
          </span>
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
        label: `${t.healthNoiseReducedPrefix}${Math.round(origNoise - cleanNoise)}${t.healthNoiseReducedSuffix}`,
        severity: 'good',
        title: `Noise floor improved from ${origNoise.toFixed(1)} dB to ${cleanNoise.toFixed(1)} dB`,
      });
    } else if (enhanced > 0) {
      chips.push({
        label: t.healthVoiceCleaned,
        severity: 'good',
        title: `${enhanced} of ${total} speech segments enhanced`,
      });
    } else {
      chips.push({
        label: t.healthVoicePreserved,
        severity: 'good',
        title: 'Your recording was already clean, so original vocal fidelity was preserved.',
      });
    }
    chips.push({
      label: t.healthSafetyVerified,
      severity: 'info',
      title: `Guard checked ${total} speech segments automatically.`,
    });
  } else {
    if (origNoise !== null && Number.isFinite(origNoise)) {
      if (origNoise > -45) {
        chips.push({
          label: t.healthHeavyNoise,
          severity: 'high',
          title: `Measured background noise floor: ${origNoise.toFixed(1)} dB`,
        });
      } else if (origNoise > -55) {
        chips.push({
          label: t.healthNoticeableNoise,
          severity: 'high',
          title: `Measured background noise floor: ${origNoise.toFixed(1)} dB`,
        });
      } else {
        chips.push({
          label: t.healthLowNoise,
          severity: 'good',
          title: `Measured background noise floor: ${origNoise.toFixed(1)} dB`,
        });
      }
    }

    if (origLufs !== null && Number.isFinite(origLufs)) {
      if (origLufs < -26) {
        chips.push({
          label: t.healthVoiceQuiet,
          severity: 'info',
          title: `Integrated loudness: ${origLufs.toFixed(1)} LUFS`,
        });
      } else if (origLufs > -13) {
        chips.push({
          label: t.healthLoudRecording,
          severity: 'info',
          title: `Integrated loudness: ${origLufs.toFixed(1)} LUFS`,
        });
      }
    }

    if (origPeak !== null && Number.isFinite(origPeak) && origPeak > -0.3) {
      chips.push({
        label: t.healthDistortionPeaks,
        severity: 'high',
        title: `Peak level: ${origPeak.toFixed(1)} dBTP`,
      });
    }

    chips.push({
      label: `${hasVideo ? t.healthVideoWithAudio : t.healthAudioRecording} · ${fmtMinSec(original.duration_s)}`,
      severity: 'info',
      title: `${Math.round(original.sample_rate / 1000)} kHz · ${original.channels}ch`,
    });
  }

  const hasFixableIssues =
    !report &&
    ((origNoise !== null && Number.isFinite(origNoise) && origNoise > -55) ||
      (origLufs !== null && Number.isFinite(origLufs) && origLufs < -26) ||
      (origPeak !== null && Number.isFinite(origPeak) && origPeak > -0.3));

  const applyAutoFix = (): void => {
    if (busy || !original) return;
    const next = { ...rackSettings };
    const tuned: string[] = [];
    if (origNoise !== null && Number.isFinite(origNoise) && origNoise > -45) {
      const targetProfile = isProfileUnavailable(capabilities ?? [], 'studio_pro')
        ? 'studio'
        : 'studio_pro';
      setProfile(targetProfile);
      next.strength = Math.max(next.strength, 0.85);
      next.hpfEnabled = true;
      next.hpfCutoffHz = Math.max(80, next.hpfCutoffHz);
      tuned.push(targetProfile === 'studio_pro' ? 'Studio Pro' : 'Studio');
    } else if (origNoise !== null && Number.isFinite(origNoise) && origNoise > -55) {
      setProfile('studio');
      next.strength = Math.max(next.strength, 0.72);
      next.hpfEnabled = true;
      tuned.push('Studio');
    }
    if (origPeak !== null && Number.isFinite(origPeak) && origPeak > -0.3) {
      next.declip = true;
      tuned.push('De-Clip');
    }
    if (origLufs !== null && Number.isFinite(origLufs) && origLufs < -26) {
      next.levelRider = true;
      next.deliveryPreset = 'podcast';
      tuned.push('Leveler + Podcast');
    }
    updateRackSettings(next);
    scheduleLivePreview();
    pushToast(
      isCkb
        ? `چارەسەری خۆکار ڕێکخرا: ${tuned.join(' · ')}`
        : `Auto-configured: ${tuned.join(' · ')}`,
      'ok',
    );
  };

  const headline = report
    ? t.healthHeadlineDone
    : previewActive
      ? t.healthHeadlinePreview
      : t.healthHeadlineReady;

  return (
    <div className="health-card" role="region" aria-label="Audio health diagnosis">
      <div className="hc-headline">
        {report ? <IconCheck size={16} /> : <IconBolt size={16} />}
        <span>{headline}</span>
        {hasFixableIssues ? (
          <button
            type="button"
            className="hc-autofix-btn"
            disabled={busy}
            onClick={applyAutoFix}
            title={
              isCkb
                ? 'ڕێکخستنی خۆکاری پڕۆفایل و چارەسەرەکان بەپێی کێشە دۆزراوەکان'
                : 'Automatically configure profile and repair modules for detected issues'
            }
          >
            {isCkb ? 'چارەسەری خۆکار' : 'Auto-Configure Fix'}
          </button>
        ) : null}
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
