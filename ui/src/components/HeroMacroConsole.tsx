import { useEffect, useState } from 'react';
import { getPlayer } from '../audio/player';
import { previewRegion, updateRackSettings } from '../state/actions';
import { scheduleLivePreview } from '../state/liveAudition';
import { jobInFlight, useStore } from '../state/store';
import { PillToggle } from './controls/PillToggle';
import { RotaryKnob } from './controls/RotaryKnob';

const ROOM_STEPS: readonly { idx: number; db: number | null; label: string }[] = [
  { idx: 0, db: null, label: 'Dry' },
  { idx: 1, db: -42, label: 'Subtle' },
  { idx: 2, db: -36, label: 'Natural' },
  { idx: 3, db: -30, label: 'Warm' },
  { idx: 4, db: -24, label: 'Full' },
];

function ambienceToStep(db: number | null): number {
  if (db === null) return 0;
  let best = 2;
  let minDiff = Infinity;
  for (const s of ROOM_STEPS) {
    if (s.db === null) continue;
    const d = Math.abs(s.db - db);
    if (d < minDiff) {
      minDiff = d;
      best = s.idx;
    }
  }
  return best;
}

function polishScoreFromRack(rack: {
  deplosive: boolean;
  mouthDeclick: boolean;
  deess: boolean;
  tonalRestoration: boolean;
  levelRider: boolean;
}): number {
  let count = 0;
  if (rack.levelRider) count += 25;
  if (rack.tonalRestoration) count += 25;
  if (rack.deess) count += 25;
  if (rack.deplosive || rack.mouthDeclick) count += 25;
  return count;
}

export function HeroMacroConsole() {
  const rackSettings = useStore((s) => s.rackSettings);
  const playing = useStore((s) => s.playing);
  const source = useStore((s) => s.source);
  const original = useStore((s) => s.original);
  const engineReady = useStore((s) => s.engineStatus === 'ready');
  const busy = useStore((s) => jobInFlight(s.job) || s.analyzing);
  const livePreviewEnabled = useStore((s) => s.livePreviewEnabled);
  const setLivePreviewEnabled = useStore((s) => s.setLivePreviewEnabled);
  const previewActive = useStore((s) => s.previewActive);
  const cleanedPath = useStore((s) => s.cleanedPath);

  const [mixPct, setMixPct] = useState(100);

  useEffect(() => {
    const unsub = getPlayer().subscribe((snap) => {
      const m = Math.round((snap.wetDryMix ?? 1) * 100);
      setMixPct((prev) => (prev === m ? prev : m));
    });
    return unsub;
  }, []);

  const disabled = busy;
  const canAudition = engineReady && Boolean(source && original) && !busy;
  const hasCleanedAudio = Boolean(cleanedPath || previewActive);

  const strengthPct = Math.round(Math.min(1, Math.max(0.1, rackSettings.strength || 0.75)) * 100);
  const roomIdx = ambienceToStep(rackSettings.ambienceFloorDb);
  const polishPct = polishScoreFromRack(rackSettings);

  const onCleanChange = (pct: number): void => {
    updateRackSettings({ ...rackSettings, strength: Math.round(pct) / 100 });
    scheduleLivePreview();
  };

  const onRoomChange = (idx: number): void => {
    const chosen = ROOM_STEPS[Math.min(ROOM_STEPS.length - 1, Math.max(0, Math.round(idx)))];
    if (!chosen) return;
    updateRackSettings({ ...rackSettings, ambienceFloorDb: chosen.db });
    scheduleLivePreview();
  };

  const onPolishChange = (pct: number): void => {
    updateRackSettings({
      ...rackSettings,
      levelRider: pct >= 25,
      tonalRestoration: pct >= 50,
      deess: pct >= 75,
      deplosive: pct >= 100,
    });
    scheduleLivePreview();
  };

  const onMixChange = (pct: number): void => {
    setMixPct(pct);
    const p = getPlayer();
    if (typeof p.setWetDryMix === 'function') {
      p.setWetDryMix(pct / 100);
    }
  };

  return (
    <section className="panel hero-console" aria-label="Quick voice controls">
      <div className="hc-side">
        <RotaryKnob
          label="Natural Room"
          value={roomIdx}
          min={0}
          max={4}
          step={1}
          defaultValue={2}
          formatValue={(v) => ROOM_STEPS[Math.round(v)]?.label ?? 'Natural'}
          hint="Keeps pauses natural"
          accent="amber"
          size="md"
          disabled={disabled}
          onChange={onRoomChange}
        />
        <RotaryKnob
          label="Voice Polish"
          value={polishPct}
          min={0}
          max={100}
          step={25}
          defaultValue={25}
          formatValue={(v) => (v === 0 ? 'Off' : `${v}%`)}
          hint="Smoothness & clarity"
          accent="emerald"
          size="md"
          disabled={disabled}
          onChange={onPolishChange}
        />
      </div>

      <div className="hc-center">
        <RotaryKnob
          label="Clean Amount"
          subLabel="CLEAN"
          value={strengthPct}
          min={10}
          max={100}
          step={1}
          defaultValue={78}
          formatValue={(v) => `${Math.round(v)}%`}
          accent="cyan"
          size="hero"
          disabled={disabled}
          audioReactive
          playing={playing}
          onChange={onCleanChange}
        />
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap', justifyContent: 'center' }}>
          <PillToggle
            label={livePreviewEnabled ? 'Auto-Preview On' : 'Auto-Preview Off'}
            pressed={livePreviewEnabled}
            onToggle={() => setLivePreviewEnabled(!livePreviewEnabled)}
            title="Automatically audition an 8-second loop when you turn a dial"
          />
          {canAudition ? (
            <button
              type="button"
              className="hc-preview-pill"
              data-live={previewActive ? 'true' : 'false'}
              onClick={() => void previewRegion()}
              title="Audition an 8-second preview right now (P)"
            >
              {previewActive ? '● Live Preview Active' : 'Audition 8s (P)'}
            </button>
          ) : null}
        </div>
      </div>

      <div className="hc-side">
        <RotaryKnob
          label="Wet / Dry Mix"
          value={mixPct}
          min={0}
          max={100}
          step={1}
          defaultValue={100}
          formatValue={(v) => `${Math.round(v)}%`}
          hint={hasCleanedAudio ? 'Real-time blend' : 'Active after clean'}
          accent="cyan"
          size="md"
          disabled={!hasCleanedAudio}
          onChange={onMixChange}
        />
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', alignItems: 'flex-start' }}>
          <PillToggle
            label="De-Click"
            pressed={rackSettings.mouthDeclick}
            disabled={disabled}
            title="Remove lip and mouth clicks"
            onToggle={() => {
              updateRackSettings({ ...rackSettings, mouthDeclick: !rackSettings.mouthDeclick });
              scheduleLivePreview();
            }}
          />
          <PillToggle
            label="De-Clip"
            pressed={rackSettings.declip}
            disabled={disabled}
            title="Fix distorted loud peaks"
            onToggle={() => {
              updateRackSettings({ ...rackSettings, declip: !rackSettings.declip });
              scheduleLivePreview();
            }}
          />
        </div>
      </div>
    </section>
  );
}
