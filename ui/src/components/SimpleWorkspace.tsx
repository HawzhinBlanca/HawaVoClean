import { useEffect, useRef, useState, type ChangeEvent } from 'react';
import { getPlayer } from '../audio/player';
import {
  ACCEPTED_EXTENSIONS,
  cancelJob,
  exportWithDelivery,
  ingestFiles,
  openFileDialog,
  saveCleanedMaster,
  setAb,
  startJob,
  togglePlay,
  updateRackSettings,
} from '../state/actions';
import { useI18n } from '../state/i18n';
import { deriveProcessingReadiness } from '../state/machines';
import type { ExportFormatId } from '../state/presets';
import { useStore, type AbMode } from '../state/store';
import { AudioHealthCard } from './AudioHealthCard';
import { MicTestButton } from './controls/MicTestButton';
import { GoalPresets } from './GoalPresets';
import { HeroMacroConsole } from './HeroMacroConsole';
import { IconBolt, IconCancel, IconCheck, IconDrop, IconFolder, IconPause, IconPlay } from './Icons';
import { Segmented } from './Segmented';
import { sourceHasVideo } from './VideoPreview';
import { WaveformDisplay } from './WaveformDisplay';

const ACCEPT_ATTR = ACCEPTED_EXTENSIONS.map((ext) => `.${ext}`).join(',');

export function SimpleWorkspace() {
  const { t } = useI18n();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const holdingOrigRef = useRef(false);
  const prevDeckRef = useRef<AbMode>('cleaned');
  const host = useStore((s) => s.host);
  const engineStatus = useStore((s) => s.engineStatus);
  const source = useStore((s) => s.source);
  const original = useStore((s) => s.original);
  const analyzing = useStore((s) => s.analyzing);
  const upload = useStore((s) => s.upload);
  const job = useStore((s) => s.job);
  const report = useStore((s) => s.report);
  const cleanedPath = useStore((s) => s.cleanedPath);
  const previewActive = useStore((s) => s.previewActive);
  const artifacts = useStore((s) => s.artifacts);
  const deckFault = useStore((s) => s.deckFault);
  const abMode = useStore((s) => s.abMode);
  const loudnessMatch = useStore((s) => s.loudnessMatch);
  const playing = useStore((s) => s.playing);
  const profile = useStore((s) => s.profile);
  const mode = useStore((s) => s.mode);
  const speakerId = useStore((s) => s.speakerId);
  const capabilities = useStore((s) => s.capabilities);
  const reconstructionConsent = useStore((s) => s.reconstructionConsent);
  const rackSettings = useStore((s) => s.rackSettings);
  const pushToast = useStore((s) => s.pushToast);

  const [deltaMode, setDeltaMode] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    const unsub = getPlayer().subscribe((snap) => {
      setDeltaMode((prev) => ((snap.deltaMode ?? false) === prev ? prev : Boolean(snap.deltaMode)));
    });
    return unsub;
  }, []);

  const readiness = deriveProcessingReadiness({
    engineStatus,
    source,
    original,
    analyzing,
    upload,
    job,
    report,
    cleanedPath,
    mode,
    profile,
    speakerId,
    capabilities,
    reconstructionConsent,
  });

  const isRunning = readiness.running;
  const hasAudio = Boolean(source && original);
  const hasVideo = Boolean(source && sourceHasVideo(source.path, original));
  const masterServed =
    ((Boolean(cleanedPath) && artifacts?.master !== false) || previewActive) &&
    deckFault?.deck !== 'cleaned';
  const canSaveFull = Boolean(cleanedPath && artifacts?.master !== false && engineStatus === 'ready' && !isRunning);

  const progress = Math.min(1, Math.max(0, job?.status?.progress ?? 0));
  const stageKey = job?.status?.stage ?? 'enhance';
  const stageMap: Record<string, string> = {
    preflight: t.stagePreflight,
    decode: t.stageDecode,
    segment: t.stageSegment,
    enhance: t.stageEnhance,
    guard: t.stageGuard,
    finish: t.stageFinish,
    publish: t.stagePublish,
    done: t.stageDone,
  };
  const stageText = stageMap[stageKey] ?? t.stageDefault;

  const handleChooseFile = (): void => {
    if (host === 'electron' || host === 'resolve') {
      void openFileDialog();
      return;
    }
    fileInputRef.current?.click();
  };

  const handleFileInput = (e: ChangeEvent<HTMLInputElement>): void => {
    const files = e.target.files ? Array.from(e.target.files) : [];
    e.target.value = '';
    if (files.length > 0) void ingestFiles(files);
  };

  const handleFormatChange = (e: ChangeEvent<HTMLSelectElement>): void => {
    const fmt = e.target.value as ExportFormatId;
    updateRackSettings({ ...rackSettings, exportFormat: fmt });
  };

  const startHoldOriginal = (): void => {
    if (!hasAudio || !masterServed || holdingOrigRef.current) return;
    holdingOrigRef.current = true;
    prevDeckRef.current = abMode;
    setAb('original');
  };

  const endHoldOriginal = (): void => {
    if (!holdingOrigRef.current) return;
    holdingOrigRef.current = false;
    if (prevDeckRef.current === 'cleaned' && masterServed) {
      setAb('cleaned');
    }
  };

  const handleSave = async (): Promise<void> => {
    if (!canSaveFull || saving) return;
    setSaving(true);
    try {
      const fmt = rackSettings.exportFormat === 'video_mux' && !hasVideo ? 'wav24' : rackSettings.exportFormat;
      if (fmt === 'wav24' && rackSettings.deliveryPreset === 'match_input' && host === 'electron') {
        await saveCleanedMaster();
        pushToast('Saved cleaned audio file', 'ok');
      } else {
        const res = await exportWithDelivery(fmt);
        if (res) {
          pushToast(
            fmt === 'video_mux' ? 'Saved cleaned video with original video quality' : `Exported ${fmt.toUpperCase()} cleaned audio`,
            'ok',
          );
        }
      }
    } finally {
      setSaving(false);
    }
  };

  const effectiveExportFormat: ExportFormatId =
    rackSettings.exportFormat === 'video_mux' && !hasVideo ? 'wav24' : rackSettings.exportFormat;

  return (
    <main className="main simple-main" aria-label="Simple Studio workspace">
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept={ACCEPT_ATTR}
        className="sr-only"
        aria-label="Choose audio or video file"
        onChange={handleFileInput}
      />

      <div className="simple-flow">
        {/* 1. Hero Audio Stage */}
        <section className="panel simple-stage-card" aria-label="Audio stage">
          {!source ? (
            <div className="simple-welcome">
              <div className="sw-crest" aria-hidden="true">
                <IconDrop size={26} />
              </div>
              <h2 className="sw-title">{t.welcomeTitle}</h2>
              <p className="sw-sub">{t.welcomeSub}</p>
              <div className="sw-actions">
                <button
                  type="button"
                  className="sw-primary-btn"
                  onClick={handleChooseFile}
                >
                  <IconFolder size={16} />
                  <span>{t.chooseFileBtn}</span>
                </button>
                <MicTestButton />
              </div>
              <span className="sw-formats">{t.welcomeFormats}</span>
            </div>
          ) : (
            <>
              <AudioHealthCard />
              <WaveformDisplay />
            </>
          )}
        </section>

        {/* 2. Goal Chips + Hero Audio-Reactive Macro Console */}
        <GoalPresets />
        <HeroMacroConsole />

        {/* 3. Effortless Action & Before/After Dock */}
        <aside className="right panel simple-dock" aria-label="Analysis and controls">
          {isRunning ? (
            <div className="sd-progress-wrap">
              <div className="sd-progress-text">
                <span>{stageText}</span>
                <span className="mono">{Math.round(progress * 100)}%</span>
              </div>
              <div
                className="sd-progress-track"
                role="progressbar"
                aria-valuenow={Math.round(progress * 100)}
                aria-valuemin={0}
                aria-valuemax={100}
              >
                <div
                  className="sd-progress-fill"
                  style={{ width: `${Math.max(4, Math.round(progress * 100))}%` }}
                />
              </div>
            </div>
          ) : null}

          <div className="sd-left">
            {isRunning ? (
              <button
                type="button"
                className="sd-clean-btn process running"
                aria-label="Cancel processing"
                onClick={() => void cancelJob()}
              >
                <IconCancel size={16} />
                <span>{t.cancelBtn}</span>
              </button>
            ) : (
              <button
                type="button"
                className="sd-clean-btn process"
                aria-label="PROCESS"
                disabled={!readiness.canStart}
                onClick={() => void startJob()}
                title={readiness.blockedReason ?? 'Clean the entire recording with voice protection'}
              >
                {report ? <IconCheck size={17} /> : <IconBolt size={17} />}
                <span>{t.processBtn}</span>
              </button>
            )}
            <button
              type="button"
              className={`playbtn${playing ? ' on' : ''}`}
              disabled={!hasAudio}
              onClick={togglePlay}
              aria-label={playing ? `${t.pause} playback` : `${t.play} audio`}
              title={playing ? `${t.pause} (Space)` : `${t.play} (Space)`}
            >
              {playing ? <IconPause size={16} /> : <IconPlay size={16} />}
            </button>
          </div>

          <div className="sd-mid">
            <Segmented<AbMode>
              ariaLabel="Compare Original or Cleaned"
              className="ab"
              thumbClassName={abMode === 'cleaned' ? 'clean' : 'orig'}
              value={abMode}
              onChange={setAb}
              options={[
                { value: 'original', label: t.original, className: 'orig', disabled: !hasAudio },
                { value: 'cleaned', label: t.cleaned, className: 'clean', disabled: !masterServed },
              ]}
            />
            <button
              type="button"
              className="btn-level-match btn-hold-orig"
              disabled={!hasAudio || !masterServed}
              onPointerDown={(e) => {
                if (e.button === 0) startHoldOriginal();
              }}
              onPointerUp={endHoldOriginal}
              onPointerCancel={endHoldOriginal}
              onPointerLeave={endHoldOriginal}
              onKeyDown={(e) => {
                if ((e.key === ' ' || e.key === 'Enter') && !e.repeat) {
                  e.preventDefault();
                  startHoldOriginal();
                }
              }}
              onKeyUp={(e) => {
                if (e.key === ' ' || e.key === 'Enter') {
                  e.preventDefault();
                  endHoldOriginal();
                }
              }}
              onBlur={endHoldOriginal}
              title={t.holdForOriginalTitle}
              style={{ minHeight: '32px', padding: '0 12px' }}
            >
              <span>{t.holdForOriginal}</span>
            </button>
            <button
              type="button"
              className={`btn-level-match btn-delta${deltaMode && masterServed ? ' on' : ''}`}
              disabled={!hasAudio || !masterServed}
              aria-pressed={deltaMode && masterServed}
              onClick={() => getPlayer().toggleDeltaMode()}
              title={t.hearRemovedTitle}
              style={{ minHeight: '32px', padding: '0 12px' }}
            >
              <span aria-hidden="true">Δ</span>
              <span>{t.hearRemoved}</span>
            </button>
          </div>

          <div className="sd-right">
            <select
              className="sd-save-select"
              aria-label="Save file format"
              value={effectiveExportFormat}
              disabled={!canSaveFull || saving}
              onChange={handleFormatChange}
            >
              <option value="wav24">{t.formatWav}</option>
              <option value="aac">{t.formatAac}</option>
              <option value="flac">{t.formatFlac}</option>
              {hasVideo ? <option value="video_mux">{t.formatVideoMux}</option> : null}
            </select>
            <button
              type="button"
              className="sd-save-btn"
              disabled={!canSaveFull || saving}
              onClick={() => void handleSave()}
            >
              <IconCheck size={15} />
              <span>
                {saving
                  ? t.saving
                  : effectiveExportFormat === 'video_mux'
                    ? t.saveCleanedVideo
                    : t.saveCleanedAudio}
              </span>
            </button>
          </div>

          <div className="sd-trust" style={{ width: '100%' }}>
            <span>
              {t.trustPrefix} · {loudnessMatch ? t.trustLevelMatchOn : t.trustLevelMatchOff}
            </span>
          </div>
        </aside>
      </div>
    </main>
  );
}
