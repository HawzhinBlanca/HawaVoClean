import { useState } from 'react';
import { pinTake, removeTake, selectTake, updateRackSettings } from '../state/actions';
import { scheduleLivePreview } from '../state/liveAudition';
import { HPF_MAX_HZ, HPF_MIN_HZ, type DeliveryPresetId, type StereoModeId } from '../state/presets';
import { jobInFlight, useStore } from '../state/store';
import { Actions } from './Actions';
import { AdvancedControls } from './AdvancedControls';
import { AudioStage } from './AudioStage';
import { BatchQueue } from './BatchQueue';
import { PillToggle } from './controls/PillToggle';
import { RotaryKnob } from './controls/RotaryKnob';
import { JobHistory } from './JobHistory';
import { ProcessButton } from './ProcessButton';
import { SlideOverDrawer } from './SlideOverDrawer';
import { SpectrumDisplay } from './SpectrumDisplay';
import { TakeSwitcher } from './TakeSwitcher';
import { Transport } from './Transport';
import { UnitInspector } from './UnitInspector';

type ProDrawer = null | 'units' | 'history' | 'rack';

const DELIVERY_PILLS: readonly { id: DeliveryPresetId; label: string }[] = [
  { id: 'match_input', label: 'Match' },
  { id: 'podcast', label: 'Podcast -16' },
  { id: 'ebu_r128', label: 'EBU -23' },
  { id: 'streaming', label: 'Stream -14' },
];

const STEREO_PILLS: readonly { id: StereoModeId; label: string }[] = [
  { id: 'auto', label: 'Auto' },
  { id: 'true_stereo_ms', label: 'M/S' },
  { id: 'dual_mono_same', label: 'Mono' },
  { id: 'split_speakers', label: 'Split' },
];

export function ProWorkspace() {
  const abMode = useStore((s) => s.abMode);
  const takes = useStore((s) => s.takes);
  const activeTakeId = useStore((s) => s.activeTakeId);
  const batch = useStore((s) => s.batch);
  const report = useStore((s) => s.report);
  const history = useStore((s) => s.history);
  const rackSettings = useStore((s) => s.rackSettings);
  const playing = useStore((s) => s.playing);
  const busy = useStore((s) => jobInFlight(s.job) || s.analyzing);
  const setAdvancedOpen = useStore((s) => s.setAdvancedOpen);

  const [drawer, setDrawer] = useState<ProDrawer>(null);

  const effectiveActiveTake =
    abMode === 'original' ? 'original' : (activeTakeId ?? takes[takes.length - 1]?.id ?? 'original');
  const strengthPct = Math.round(Math.min(1, Math.max(0.1, rackSettings.strength || 0.75)) * 100);
  const unitCount = report?.units?.length ?? 0;
  const runCount = history.length;

  const patchRack = (partial: Partial<typeof rackSettings>): void => {
    if (busy) return;
    updateRackSettings({ ...rackSettings, ...partial });
    scheduleLivePreview();
  };

  return (
    <main className="main pro-main" aria-label="Pro Studio workstation">
      {/* Row 1: Full-Width Unified Audio Stage (Waveform <-> WebGL2 Spectrogram) */}
      <div className="pro-stage-row">
        <AudioStage />
        {takes.length >= 2 ? (
          <div className="app-take-strip">
            <TakeSwitcher
              takes={takes}
              activeId={effectiveActiveTake}
              onSelect={selectTake}
              onTogglePin={pinTake}
              onRemoveTake={removeTake}
            />
          </div>
        ) : null}
      </div>

      {/* Row 2: Full-Width FabFilter-Style Interactive Spectrum & Piano Scale */}
      <section className="panel pro-spectrum-row" aria-label="Spectrum">
        <SpectrumDisplay wide />
      </section>

      {/* Row 3: Full-Width Horizontal Precision Studio Console */}
      <aside className="right panel pro-console" aria-label="Analysis and controls">
        {/* Left: Full Unclipped Transport & Live Meter */}
        <div className="pc-zone pc-transport">
          <Transport />
        </div>

        {/* Center: Machined Rotary Dials, Repair Modules, Delivery & Stereo M/S */}
        <div className="pc-zone pc-dsp">
          <div className="pc-knobs">
            <RotaryKnob
              label="Clean"
              value={strengthPct}
              min={10}
              max={100}
              step={5}
              defaultValue={75}
              formatValue={(v) => `${Math.round(v)}%`}
              accent="cyan"
              size="md"
              audioReactive
              playing={playing}
              disabled={busy}
              onChange={(pct) => patchRack({ strength: Math.round(pct) / 100 })}
            />
            <RotaryKnob
              label="HPF"
              value={rackSettings.hpfCutoffHz}
              min={HPF_MIN_HZ}
              max={HPF_MAX_HZ}
              step={5}
              defaultValue={80}
              formatValue={(v) => (rackSettings.hpfEnabled === false ? 'Off' : `${Math.round(v)}Hz`)}
              accent="amber"
              size="sm"
              disabled={busy}
              onChange={(hz) => patchRack({ hpfEnabled: true, hpfCutoffHz: Math.round(hz) })}
            />
          </div>

          <div className="pc-modules">
            <div className="pc-pill-row" role="group" aria-label="Surgical repair modules">
              <PillToggle
                label="De-Plosive"
                pressed={rackSettings.deplosive}
                disabled={busy}
                onToggle={() => patchRack({ deplosive: !rackSettings.deplosive })}
              />
              <PillToggle
                label="De-Click"
                pressed={rackSettings.mouthDeclick}
                disabled={busy}
                onToggle={() => patchRack({ mouthDeclick: !rackSettings.mouthDeclick })}
              />
              <PillToggle
                label="De-Clip"
                pressed={rackSettings.declip}
                disabled={busy}
                onToggle={() => patchRack({ declip: !rackSettings.declip })}
              />
              <PillToggle
                label="De-Ess"
                pressed={rackSettings.deess}
                disabled={busy}
                onToggle={() => patchRack({ deess: !rackSettings.deess })}
              />
              <PillToggle
                label="Tonal"
                pressed={rackSettings.tonalRestoration}
                disabled={busy}
                onToggle={() => patchRack({ tonalRestoration: !rackSettings.tonalRestoration })}
              />
              <PillToggle
                label="Leveler"
                pressed={rackSettings.levelRider}
                disabled={busy}
                onToggle={() => patchRack({ levelRider: !rackSettings.levelRider })}
              />
            </div>

            <div className="pc-meta-row">
              <div className="pc-mini-group" role="group" aria-label="Delivery loudness target">
                {DELIVERY_PILLS.map((d) => (
                  <button
                    key={d.id}
                    type="button"
                    className={`pc-chip${rackSettings.deliveryPreset === d.id ? ' on' : ''}`}
                    aria-pressed={rackSettings.deliveryPreset === d.id}
                    disabled={busy}
                    onClick={() => patchRack({ deliveryPreset: d.id })}
                  >
                    {d.label}
                  </button>
                ))}
              </div>

              <div className="pc-mini-group" role="group" aria-label="Stereo routing mode">
                {STEREO_PILLS.map((s) => (
                  <button
                    key={s.id}
                    type="button"
                    className={`pc-chip${rackSettings.stereoMode === s.id ? ' on' : ''}`}
                    aria-pressed={rackSettings.stereoMode === s.id}
                    disabled={busy}
                    onClick={() => patchRack({ stereoMode: s.id })}
                  >
                    {s.label}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Right: Process Action + Export + On-Demand Drawers (Zero Empty Boxes!) */}
        <div className="pc-zone pc-action">
          <div className="pc-action-top">
            <ProcessButton />
          </div>
          <div className="pc-drawer-pills">
            <button
              type="button"
              className="pc-drawer-btn"
              onClick={() => setDrawer('units')}
              title="Inspect and override individual speech units"
            >
              <span>Units</span>
              {unitCount > 0 ? <span className="pc-badge mono">{unitCount}</span> : null}
            </button>
            <button
              type="button"
              className="pc-drawer-btn"
              onClick={() => setDrawer('history')}
              title="View session runs and batch queue"
            >
              <span>{batch ? 'Batch' : 'Runs'}</span>
              {runCount > 0 ? <span className="pc-badge mono">{runCount}</span> : null}
            </button>
            <button
              type="button"
              className="pc-drawer-btn"
              onClick={() => {
                setAdvancedOpen(true);
                setDrawer('rack');
              }}
              title="Open full Studio Rack, profiles, restore, and preset manager"
            >
              <span>Engine &amp; Presets</span>
            </button>
          </div>
          <Actions />
        </div>
      </aside>

      <SlideOverDrawer
        open={drawer === 'units'}
        title="Unit Guard Inspector"
        onClose={() => setDrawer(null)}
      >
        <UnitInspector />
      </SlideOverDrawer>

      <SlideOverDrawer
        open={drawer === 'history'}
        title={batch ? 'Batch Queue' : 'Session Runs'}
        onClose={() => setDrawer(null)}
      >
        {batch ? <BatchQueue /> : <JobHistory />}
      </SlideOverDrawer>

      <SlideOverDrawer
        open={drawer === 'rack'}
        title="Engine, Restore & Presets"
        onClose={() => setDrawer(null)}
      >
        <AdvancedControls />
      </SlideOverDrawer>
    </main>
  );
}
