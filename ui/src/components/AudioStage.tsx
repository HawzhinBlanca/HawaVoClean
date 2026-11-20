import { useState, type CSSProperties } from 'react';
import { applySpectralBoxRepair } from '../state/actions';
import { useI18n } from '../state/i18n';
import { jobInFlight, useStore } from '../state/store';
import { FloatingPopover } from './controls/FloatingPopover';
import { SpectrogramDisplay } from './SpectrogramDisplay';
import { WaveformDisplay } from './WaveformDisplay';

export function AudioStage() {
  const { t } = useI18n();
  const stageBlend = useStore((s) => s.stageBlend);
  const setStageBlend = useStore((s) => s.setStageBlend);
  const spectralSelection = useStore((s) => s.spectralSelection);
  const setSpectralSelection = useStore((s) => s.setSpectralSelection);
  const setAssistantDrawerOpen = useStore((s) => s.setAssistantDrawerOpen);
  const pushToast = useStore((s) => s.pushToast);
  const hasSource = useStore((s) => Boolean(s.source && s.original));
  const engineReady = useStore((s) => s.engineStatus === 'ready');
  const busy = useStore((s) => jobInFlight(s.job) || s.analyzing);

  const [repairing, setRepairing] = useState(false);

  const blendPct = Math.round(stageBlend * 100);
  const canRepair = hasSource && engineReady && !busy && !repairing && Boolean(spectralSelection);

  const handleQuickRepair = async (db: number): Promise<void> => {
    if (!canRepair || !spectralSelection) return;
    setRepairing(true);
    try {
      const res = await applySpectralBoxRepair(db);
      if (res) {
        pushToast(
          `Attenuated ${Math.round(res.fLowHz)}–${Math.round(res.fHighHz)} Hz by ${Math.round(res.attenuationDb)} dB`,
          'ok',
        );
      }
    } finally {
      setRepairing(false);
    }
  };

  const blendMode = stageBlend >= 0.99 ? 'full' : stageBlend > 0 ? 'blend' : 'wave';
  // Smooth RX-11 continuous alpha crossfade on one shared full-height stage
  const waveOpacity = stageBlend >= 0.99 ? 0 : Math.max(0.16, 1 - stageBlend * 0.78);
  const specOpacity = stageBlend;
  const specInteractive = stageBlend >= 0.35;

  return (
    <div
      className="as-stack"
      data-blended={blendMode}
      style={{ '--stage-blend': String(stageBlend) } as CSSProperties}
    >
      {hasSource ? (
        <div className="audio-stage-bar" role="toolbar" aria-label="Audio stage view">
          <div className="as-blend-wrap">
            <span>{t.waveformLabel}</span>
            <input
              type="range"
              className="as-blend-slider"
              min={0}
              max={100}
              step={5}
              value={blendPct}
              aria-label="Waveform to Spectrogram blend"
              aria-valuetext={`${blendPct}% spectrogram`}
              onChange={(e) => setStageBlend(Number(e.target.value) / 100)}
            />
            <span>{t.spectrogramLabel}</span>
            <span className="mono" style={{ fontSize: '11px', color: 'var(--cyan-2)', minWidth: '34px' }}>
              {blendPct}%
            </span>
          </div>
        </div>
      ) : null}

      <div className="as-stage-viewport">
        <div
          className="as-wave-layer"
          style={{
            opacity: waveOpacity,
            pointerEvents: specInteractive ? 'none' : 'auto',
          }}
        >
          <WaveformDisplay />
        </div>

        {stageBlend > 0 && hasSource ? (
          <div
            className="as-spec-layer"
            style={{
              opacity: specOpacity,
              pointerEvents: specInteractive ? 'auto' : 'none',
            }}
          >
            <SpectrogramDisplay embedded onRequestHide={() => setStageBlend(0)} />
          </div>
        ) : null}

        <FloatingPopover
          id="as-spectral-repair-popover"
          open={Boolean(hasSource && spectralSelection)}
          onClose={() => setSpectralSelection(null)}
          ariaLabel="Spectral box repair actions"
          className="as-repair-popover"
        >
          {spectralSelection ? (
            <div className="as-repair-popover-row">
              <span className="mono" style={{ fontSize: '11px', color: 'var(--coral-2)' }}>
                {Math.round(spectralSelection.fLowHz)}–{Math.round(spectralSelection.fHighHz)} Hz
              </span>
              <button
                type="button"
                className="sr-btn sr-btn-primary"
                disabled={!canRepair}
                onClick={() => void handleQuickRepair(24)}
              >
                {repairing ? t.attenuating : t.attenuate24}
              </button>
              <button
                type="button"
                className="sr-btn"
                disabled={!canRepair}
                onClick={() => void handleQuickRepair(12)}
              >
                {t.attenuate12}
              </button>
              <button
                type="button"
                className="sr-btn"
                onClick={() => setAssistantDrawerOpen(true)}
              >
                {t.askAi}
              </button>
              <button
                type="button"
                className="sr-btn"
                onClick={() => setSpectralSelection(null)}
              >
                {t.clear}
              </button>
            </div>
          ) : null}
        </FloatingPopover>
      </div>
    </div>
  );
}
