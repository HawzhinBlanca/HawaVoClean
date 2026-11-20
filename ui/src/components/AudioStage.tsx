import { useState } from 'react';
import { applySpectralBoxRepair } from '../state/actions';
import { jobInFlight, useStore } from '../state/store';
import { SpectrogramDisplay } from './SpectrogramDisplay';
import { WaveformDisplay } from './WaveformDisplay';

export function AudioStage() {
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

  const blendMode = stageBlend >= 0.95 ? 'full' : stageBlend > 0 ? 'true' : 'false';

  return (
    <div className="as-stack" data-blended={blendMode}>
      {hasSource ? (
        <div className="audio-stage-bar" role="toolbar" aria-label="Audio stage view">
          <div className="as-blend-wrap">
            <span>Waveform</span>
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
            <span>Spectrogram</span>
            <span className="mono" style={{ fontSize: '11px', color: 'var(--cyan-2)', minWidth: '34px' }}>
              {blendPct}%
            </span>
          </div>

          {spectralSelection ? (
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
              <span className="mono" style={{ fontSize: '11px', color: 'var(--coral-2)' }}>
                {Math.round(spectralSelection.fLowHz)}–{Math.round(spectralSelection.fHighHz)} Hz
              </span>
              <button
                type="button"
                className="sr-btn sr-btn-primary"
                disabled={!canRepair}
                onClick={() => void handleQuickRepair(24)}
              >
                {repairing ? 'Attenuating…' : 'Attenuate −24 dB'}
              </button>
              <button
                type="button"
                className="sr-btn"
                disabled={!canRepair}
                onClick={() => void handleQuickRepair(12)}
              >
                −12 dB
              </button>
              <button
                type="button"
                className="sr-btn"
                onClick={() => setAssistantDrawerOpen(true)}
              >
                Ask AI
              </button>
              <button
                type="button"
                className="sr-btn"
                onClick={() => setSpectralSelection(null)}
              >
                Clear
              </button>
            </div>
          ) : null}
        </div>
      ) : null}

      <div className="as-wave-layer">
        <WaveformDisplay />
      </div>

      <div className="as-spec-layer">
        <SpectrogramDisplay />
      </div>
    </div>
  );
}
