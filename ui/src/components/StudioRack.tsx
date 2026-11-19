import { useId, useMemo, useState, type ChangeEvent, type FormEvent } from 'react';
import { previewRegion } from '../state/actions';
import { useStore } from '../state/store';
import {
  AMBIENCE_FLOOR_OPTIONS,
  BUILTIN_PRESETS,
  DEFAULT_NAMING_TEMPLATE,
  HPF_MAX_HZ,
  HPF_MIN_HZ,
  deleteUserPreset,
  exportPresetJson,
  formatBatchOutputName,
  importPresetJson,
  loadUserPresets,
  saveUserPreset,
  validateNamingTemplate,
  type DeliveryPresetId,
  type ExportFormatId,
  type StereoModeId,
  type StudioRackSettings,
  type UserPreset,
} from '../state/presets';
import '../styles/studio-rack.css';

export interface StudioRackProps {
  settings: StudioRackSettings;
  onChange: (next: StudioRackSettings) => void;
  isVideoSource?: boolean | undefined;
  disabled?: boolean | undefined;
  presets?: readonly UserPreset[] | undefined;
  activePresetId?: string | null | undefined;
  onSelectPreset?: ((preset: UserPreset) => void) | undefined;
  onSavePreset?: ((name: string, settings: StudioRackSettings) => void) | undefined;
  onDeletePreset?: ((presetId: string) => void) | undefined;
  onImportPreset?: ((preset: UserPreset) => void) | undefined;
  onExportPreset?: ((json: string, preset: UserPreset) => void) | undefined;
  namingTemplate?: string | undefined;
  onNamingTemplateChange?: ((template: string) => void) | undefined;
  sampleFileName?: string | undefined;
  activeProfile?: string | undefined;
  canPreview?: boolean | undefined;
  onPreviewRegion?: (() => void | Promise<unknown>) | undefined;
  className?: string | undefined;
}

const DELIVERY_OPTIONS: readonly { id: DeliveryPresetId; label: string; description: string }[] = [
  { id: 'match_input', label: 'Match Input', description: 'Preserve source loudness' },
  { id: 'ebu_r128', label: 'EBU -23', description: 'EBU R128 broadcast (-23 LUFS)' },
  { id: 'atsc_a85', label: 'ATSC -24', description: 'ATSC A/85 broadcast (-24 LUFS)' },
  { id: 'podcast', label: 'Podcast -16', description: 'Podcast dialogue (-16 LUFS)' },
  { id: 'streaming', label: 'Stream -14', description: 'Streaming delivery (-14 LUFS)' },
];

const EXPORT_FORMAT_OPTIONS: readonly { id: ExportFormatId; label: string; videoOnly?: boolean }[] = [
  { id: 'wav24', label: 'WAV 24-bit' },
  { id: 'wav32f', label: 'WAV 32f' },
  { id: 'flac', label: 'FLAC' },
  { id: 'aac', label: 'AAC' },
  { id: 'video_mux', label: 'Mux Video', videoOnly: true },
];

const STEREO_MODE_OPTIONS: readonly { id: StereoModeId; label: string }[] = [
  { id: 'auto', label: 'Auto' },
  { id: 'dual_mono_same', label: 'Dual Mono' },
  { id: 'split_speakers', label: 'Split Speakers' },
  { id: 'true_stereo_ms', label: 'True Stereo M/S' },
];

type FinishingBooleanKey =
  | 'deplosive'
  | 'mouthDeclick'
  | 'declip'
  | 'deess'
  | 'tonalRestoration'
  | 'levelRider';

const FINISHING_TOGGLES: readonly { key: FinishingBooleanKey; label: string; title: string }[] = [
  { key: 'deplosive', label: 'De-Plosive', title: 'Attenuate vocal P/B wind bursts' },
  { key: 'mouthDeclick', label: 'De-Click', title: 'Remove lip and mouth clicks' },
  { key: 'declip', label: 'De-Clip', title: 'Reconstruct clipped peaks' },
  { key: 'deess', label: 'De-Ess', title: 'Tame harsh sibilance' },
  { key: 'tonalRestoration', label: 'Tonal', title: 'Restore balanced vocal presence' },
  { key: 'levelRider', label: 'Leveler', title: 'Smooth dialogue loudness dynamics' },
];

function formatAmbienceLabel(db: number | null): string {
  return db === null ? 'Off' : `${db} dB`;
}

export function StudioRack({
  settings,
  onChange,
  isVideoSource = false,
  disabled = false,
  presets: controlledPresets,
  activePresetId: controlledActivePresetId,
  onSelectPreset,
  onSavePreset,
  onDeletePreset,
  onImportPreset,
  onExportPreset,
  namingTemplate: controlledNamingTemplate,
  onNamingTemplateChange,
  sampleFileName = 'interview',
  activeProfile = 'studio_pro',
  canPreview: controlledCanPreview,
  onPreviewRegion,
  className,
}: StudioRackProps) {
  const uid = useId();
  const storeEngineReady = useStore((s) => s.engineStatus === 'ready');
  const storeHasSource = useStore((s) => Boolean(s.source && s.original));
  const effectiveCanPreview =
    controlledCanPreview !== undefined
      ? controlledCanPreview
      : storeEngineReady && storeHasSource;
  const [previewing, setPreviewing] = useState(false);
  const [internalUserPresets, setInternalUserPresets] = useState<UserPreset[]>(() =>
    controlledPresets ? [] : loadUserPresets(),
  );
  const [internalActiveId, setInternalActiveId] = useState<string>('');
  const [panelMode, setPanelMode] = useState<'none' | 'save' | 'import' | 'export'>('none');
  const [saveName, setSaveName] = useState('');
  const [importText, setImportText] = useState('');
  const [exportedJson, setExportedJson] = useState('');
  const [feedback, setFeedback] = useState<{ kind: 'status' | 'error'; text: string } | null>(null);

  const allPresets = useMemo<readonly UserPreset[]>(() => {
    if (controlledPresets) return controlledPresets;
    return [...BUILTIN_PRESETS, ...internalUserPresets];
  }, [controlledPresets, internalUserPresets]);

  const builtinList = useMemo(() => allPresets.filter((p) => p.builtin), [allPresets]);
  const userList = useMemo(() => allPresets.filter((p) => !p.builtin), [allPresets]);

  const activePresetId =
    controlledActivePresetId !== undefined ? (controlledActivePresetId ?? '') : internalActiveId;
  const selectedPreset = useMemo(
    () => allPresets.find((p) => p.id === activePresetId) ?? null,
    [allPresets, activePresetId],
  );

  const strengthPct = Math.round(Math.min(1, Math.max(0.1, settings.strength)) * 100);
  const hpfActive = settings.hpfEnabled !== false;
  const effectiveNamingTemplate =
    controlledNamingTemplate ?? settings.namingTemplate ?? DEFAULT_NAMING_TEMPLATE;
  const namingCheck = validateNamingTemplate(effectiveNamingTemplate);

  const outputPreview = useMemo(() => {
    const presetSlug = selectedPreset
      ? selectedPreset.name.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '')
      : settings.deliveryPreset;
    return formatBatchOutputName(effectiveNamingTemplate, {
      name: sampleFileName,
      profile: activeProfile,
      preset: presetSlug || settings.deliveryPreset,
      exportFormat: settings.exportFormat,
    });
  }, [
    effectiveNamingTemplate,
    sampleFileName,
    activeProfile,
    selectedPreset,
    settings.deliveryPreset,
    settings.exportFormat,
  ]);

  const patchSettings = (partial: Partial<StudioRackSettings>): void => {
    if (disabled) return;
    onChange({ ...settings, ...partial });
  };

  const handlePresetSelect = (e: ChangeEvent<HTMLSelectElement>): void => {
    const id = e.target.value;
    if (!id) {
      setInternalActiveId('');
      return;
    }
    const found = allPresets.find((p) => p.id === id);
    if (!found) return;
    setInternalActiveId(found.id);
    setFeedback({ kind: 'status', text: `Loaded preset "${found.name}".` });
    if (onSelectPreset) {
      onSelectPreset(found);
    } else {
      onChange({ ...found.settings });
    }
  };

  const handleSaveSubmit = (e: FormEvent): void => {
    e.preventDefault();
    if (disabled) return;
    const trimmed = saveName.trim();
    if (!trimmed) {
      setFeedback({ kind: 'error', text: 'Enter a preset name before saving.' });
      return;
    }
    try {
      if (onSavePreset) {
        onSavePreset(trimmed, settings);
      } else {
        const saved = saveUserPreset({ name: trimmed, settings });
        setInternalUserPresets(loadUserPresets());
        setInternalActiveId(saved.id);
      }
      setSaveName('');
      setPanelMode('none');
      setFeedback({ kind: 'status', text: `Saved preset "${trimmed}".` });
    } catch (err) {
      setFeedback({
        kind: 'error',
        text: err instanceof Error ? err.message : 'Unable to save preset.',
      });
    }
  };

  const handleDeleteSelected = (): void => {
    if (disabled || !selectedPreset || selectedPreset.builtin) return;
    if (onDeletePreset) {
      onDeletePreset(selectedPreset.id);
    } else {
      deleteUserPreset(selectedPreset.id);
      setInternalUserPresets(loadUserPresets());
      setInternalActiveId('');
    }
    setFeedback({ kind: 'status', text: `Deleted preset "${selectedPreset.name}".` });
  };

  const handleExportClick = (): void => {
    if (disabled) return;
    const snapshot: UserPreset = selectedPreset
      ? { ...selectedPreset, settings }
      : {
          id: 'user:studio-rack-export',
          name: 'Custom Studio Rack',
          builtin: false,
          createdAt: new Date().toISOString(),
          settings,
        };
    try {
      const json = exportPresetJson(snapshot);
      setExportedJson(json);
      setPanelMode('export');
      onExportPreset?.(json, snapshot);
      setFeedback({ kind: 'status', text: `Exported "${snapshot.name}" as JSON.` });
    } catch (err) {
      setFeedback({
        kind: 'error',
        text: err instanceof Error ? err.message : 'Unable to export preset JSON.',
      });
    }
  };

  const handleImportSubmit = (e: FormEvent): void => {
    e.preventDefault();
    if (disabled) return;
    const res = importPresetJson(importText);
    if (!res.ok) {
      setFeedback({ kind: 'error', text: res.error });
      return;
    }
    if (onImportPreset) {
      onImportPreset(res.preset);
    } else {
      const saved = saveUserPreset({
        name: res.preset.name,
        ...(res.preset.description !== undefined ? { description: res.preset.description } : {}),
        settings: res.preset.settings,
      });
      setInternalUserPresets(loadUserPresets());
      setInternalActiveId(saved.id);
    }
    onChange({ ...res.preset.settings });
    setImportText('');
    setPanelMode('none');
    setFeedback({ kind: 'status', text: `Imported preset "${res.preset.name}".` });
  };

  const handleNamingChange = (e: ChangeEvent<HTMLInputElement>): void => {
    const nextTpl = e.target.value;
    if (onNamingTemplateChange) {
      onNamingTemplateChange(nextTpl);
    } else {
      patchSettings({ namingTemplate: nextTpl });
    }
  };

  const handlePreviewClick = (): void => {
    if (disabled || !effectiveCanPreview || previewing) return;
    setPreviewing(true);
    const run = onPreviewRegion ? onPreviewRegion() : previewRegion();
    void Promise.resolve(run).finally(() => setPreviewing(false));
  };

  return (
    <section
      className={`studio-rack${className ? ` ${className}` : ''}`}
      aria-label="Studio parameter rack"
      data-disabled={disabled ? 'true' : 'false'}
    >
      <div className="sr-header">
        <div className="sr-title-group">
          <h3 className="sr-title">Studio Rack</h3>
          <span className="sr-subtitle">
            {selectedPreset ? selectedPreset.name : 'Custom parameters'}
          </span>
        </div>

        <div className="sr-preset-bar">
          <div className="sr-preset-select-wrap">
            <label htmlFor={`${uid}-preset`} className="sr-only">
              Studio preset
            </label>
            <select
              id={`${uid}-preset`}
              className="sr-select"
              value={activePresetId}
              onChange={handlePresetSelect}
              disabled={disabled}
            >
              <option value="">Custom settings…</option>
              {builtinList.length > 0 ? (
                <optgroup label="Built-in Studio Presets">
                  {builtinList.map((preset) => (
                    <option key={preset.id} value={preset.id}>
                      {preset.name}
                    </option>
                  ))}
                </optgroup>
              ) : null}
              {userList.length > 0 ? (
                <optgroup label="User Presets">
                  {userList.map((preset) => (
                    <option key={preset.id} value={preset.id}>
                      {preset.name}
                    </option>
                  ))}
                </optgroup>
              ) : null}
            </select>
          </div>

          <div className="sr-preset-actions">
            <button
              type="button"
              className="sr-btn sr-btn-preview"
              disabled={disabled || !effectiveCanPreview || previewing}
              aria-label="Preview selected region (hotkey P)"
              title={
                effectiveCanPreview
                  ? 'Instant warm region preview on Cleaned deck (P)'
                  : 'Load a clip while the engine is ready to preview'
              }
              onClick={handlePreviewClick}
            >
              {previewing ? 'Previewing…' : 'Preview (P)'}
            </button>
            <button
              type="button"
              className="sr-btn"
              disabled={disabled}
              aria-expanded={panelMode === 'save'}
              onClick={() => {
                setFeedback(null);
                setPanelMode((m) => (m === 'save' ? 'none' : 'save'));
              }}
            >
              Save
            </button>
            <button
              type="button"
              className="sr-btn"
              disabled={disabled}
              aria-expanded={panelMode === 'import'}
              onClick={() => {
                setFeedback(null);
                setPanelMode((m) => (m === 'import' ? 'none' : 'import'));
              }}
            >
              Import
            </button>
            <button
              type="button"
              className="sr-btn"
              disabled={disabled}
              aria-expanded={panelMode === 'export'}
              onClick={handleExportClick}
            >
              Export
            </button>
            {selectedPreset && !selectedPreset.builtin ? (
              <button
                type="button"
                className="sr-btn sr-btn-danger"
                disabled={disabled}
                aria-label={`Delete preset ${selectedPreset.name}`}
                onClick={handleDeleteSelected}
              >
                Delete
              </button>
            ) : null}
          </div>
        </div>
      </div>

      {panelMode === 'save' ? (
        <form className="sr-subpanel" onSubmit={handleSaveSubmit} aria-label="Save user preset">
          <label htmlFor={`${uid}-save-name`} className="sr-label">
            Preset Name
          </label>
          <div className="sr-subpanel-row">
            <input
              id={`${uid}-save-name`}
              type="text"
              className="sr-text-input"
              value={saveName}
              onChange={(e) => setSaveName(e.target.value)}
              placeholder="e.g. Studio Boom Interview"
              maxLength={80}
              disabled={disabled}
            />
            <button type="submit" className="sr-btn sr-btn-primary" disabled={disabled}>
              Save Preset
            </button>
            <button
              type="button"
              className="sr-btn"
              onClick={() => setPanelMode('none')}
            >
              Cancel
            </button>
          </div>
        </form>
      ) : null}

      {panelMode === 'import' ? (
        <form className="sr-subpanel" onSubmit={handleImportSubmit} aria-label="Import preset JSON">
          <label htmlFor={`${uid}-import-json`} className="sr-label">
            Paste Preset JSON
          </label>
          <textarea
            id={`${uid}-import-json`}
            className="sr-textarea"
            value={importText}
            onChange={(e) => setImportText(e.target.value)}
            placeholder='{"schema":"hawavoclean.preset.v1","schemaVersion":1,"preset":{...}}'
            disabled={disabled}
          />
          <div className="sr-subpanel-row">
            <button type="submit" className="sr-btn sr-btn-primary" disabled={disabled}>
              Import Preset
            </button>
            <button
              type="button"
              className="sr-btn"
              onClick={() => setPanelMode('none')}
            >
              Cancel
            </button>
          </div>
        </form>
      ) : null}

      {panelMode === 'export' ? (
        <div className="sr-subpanel" role="region" aria-label="Exported preset JSON">
          <label htmlFor={`${uid}-export-json`} className="sr-label">
            Portable Preset JSON
          </label>
          <textarea
            id={`${uid}-export-json`}
            className="sr-textarea"
            value={exportedJson}
            readOnly
          />
          <div className="sr-subpanel-row">
            <button
              type="button"
              className="sr-btn"
              onClick={() => setPanelMode('none')}
            >
              Close
            </button>
          </div>
        </div>
      ) : null}

      {feedback ? (
        feedback.kind === 'error' ? (
          <p className="sr-error" role="alert">
            {feedback.text}
          </p>
        ) : (
          <p className="sr-status" role="status" aria-live="polite">
            {feedback.text}
          </p>
        )
      ) : null}

      {/* Amount / Strength */}
      <div className="sr-section">
        <div className="sr-row-head">
          <label htmlFor={`${uid}-strength`} className="sr-label">
            Amount / Strength
          </label>
          <span className="sr-readout" data-testid="strength-readout">
            {strengthPct}%
          </span>
        </div>
        <input
          id={`${uid}-strength`}
          type="range"
          className="sr-slider"
          min={10}
          max={100}
          step={5}
          value={strengthPct}
          disabled={disabled}
          aria-valuetext={`${strengthPct}%`}
          onChange={(e) => {
            const pct = Number(e.target.value);
            if (Number.isFinite(pct)) {
              patchSettings({ strength: Math.round(pct) / 100 });
            }
          }}
        />
      </div>

      {/* Ambience Floor */}
      <div className="sr-section">
        <div className="sr-row-head">
          <span className="sr-label" id={`${uid}-amb-label`}>
            Ambience Floor
          </span>
          <span className="sr-readout">{formatAmbienceLabel(settings.ambienceFloorDb)}</span>
        </div>
        <div
          className="sr-pill-group"
          role="group"
          aria-labelledby={`${uid}-amb-label`}
        >
          {AMBIENCE_FLOOR_OPTIONS.map((db) => {
            const selected = settings.ambienceFloorDb === db;
            return (
              <button
                key={db === null ? 'off' : String(db)}
                type="button"
                className="sr-pill"
                aria-pressed={selected}
                disabled={disabled}
                onClick={() => patchSettings({ ambienceFloorDb: db })}
              >
                {formatAmbienceLabel(db)}
              </button>
            );
          })}
        </div>
      </div>

      {/* Compact Finishing Toggles */}
      <div className="sr-section">
        <div className="sr-row-head">
          <span className="sr-label" id={`${uid}-fin-label`}>
            Finishing Chain
          </span>
          <span className="sr-readout">
            {hpfActive ? `HPF ${settings.hpfCutoffHz} Hz` : 'HPF Bypass'}
          </span>
        </div>
        <div
          className="sr-pill-group"
          role="group"
          aria-labelledby={`${uid}-fin-label`}
        >
          <button
            type="button"
            className="sr-pill"
            aria-pressed={hpfActive}
            disabled={disabled}
            title="High-pass rumble filter (20–120 Hz)"
            onClick={() => patchSettings({ hpfEnabled: !hpfActive })}
          >
            <span className="sr-led" aria-hidden="true" />
            HPF
          </button>
          {FINISHING_TOGGLES.map((toggle) => {
            const active = settings[toggle.key];
            return (
              <button
                key={toggle.key}
                type="button"
                className="sr-pill"
                aria-pressed={active}
                disabled={disabled}
                title={toggle.title}
                onClick={() => patchSettings({ [toggle.key]: !active })}
              >
                <span className="sr-led" aria-hidden="true" />
                {toggle.label}
              </button>
            );
          })}
        </div>

        {hpfActive ? (
          <div className="sr-hpf-row">
            <label htmlFor={`${uid}-hpf-hz`} className="sr-label">
              HPF Cutoff
            </label>
            <input
              id={`${uid}-hpf-hz`}
              type="range"
              className="sr-slider"
              min={HPF_MIN_HZ}
              max={HPF_MAX_HZ}
              step={5}
              value={settings.hpfCutoffHz}
              disabled={disabled}
              aria-valuetext={`${settings.hpfCutoffHz} Hz`}
              onChange={(e) => {
                const hz = Number(e.target.value);
                if (Number.isFinite(hz)) {
                  patchSettings({ hpfCutoffHz: Math.min(HPF_MAX_HZ, Math.max(HPF_MIN_HZ, Math.round(hz))) });
                }
              }}
            />
            <span className="sr-readout">{settings.hpfCutoffHz} Hz</span>
          </div>
        ) : null}
      </div>

      {/* Delivery Target Pills */}
      <div className="sr-section">
        <span className="sr-label" id={`${uid}-delivery-label`}>
          Delivery Target
        </span>
        <div
          className="sr-pill-group"
          role="group"
          aria-labelledby={`${uid}-delivery-label`}
        >
          {DELIVERY_OPTIONS.map((opt) => (
            <button
              key={opt.id}
              type="button"
              className="sr-pill"
              aria-pressed={settings.deliveryPreset === opt.id}
              disabled={disabled}
              title={opt.description}
              onClick={() => patchSettings({ deliveryPreset: opt.id })}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* Export Format & Stereo Mode */}
      <div className="sr-grid-2">
        <div className="sr-section">
          <span className="sr-label" id={`${uid}-format-label`}>
            Export Format
          </span>
          <div
            className="sr-pill-group"
            role="group"
            aria-labelledby={`${uid}-format-label`}
          >
            {EXPORT_FORMAT_OPTIONS.map((opt) => {
              const formatDisabled = disabled || (opt.videoOnly === true && !isVideoSource);
              return (
                <button
                  key={opt.id}
                  type="button"
                  className="sr-pill"
                  aria-pressed={settings.exportFormat === opt.id}
                  disabled={formatDisabled}
                  title={
                    opt.videoOnly && !isVideoSource
                      ? 'Mux Video is available only when the source is a video file'
                      : opt.label
                  }
                  onClick={() => {
                    if (!formatDisabled) {
                      patchSettings({ exportFormat: opt.id });
                    }
                  }}
                >
                  {opt.label}
                </button>
              );
            })}
          </div>
        </div>

        <div className="sr-section">
          <label htmlFor={`${uid}-stereo`} className="sr-label">
            Stereo Routing
          </label>
          <select
            id={`${uid}-stereo`}
            className="sr-select"
            value={settings.stereoMode}
            disabled={disabled}
            onChange={(e) => patchSettings({ stereoMode: e.target.value as StereoModeId })}
          >
            {STEREO_MODE_OPTIONS.map((opt) => (
              <option key={opt.id} value={opt.id}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Batch Output Naming Template */}
      <div className="sr-section">
        <label htmlFor={`${uid}-naming`} className="sr-label">
          Batch Output Naming Template
        </label>
        <input
          id={`${uid}-naming`}
          type="text"
          className="sr-text-input"
          value={effectiveNamingTemplate}
          onChange={handleNamingChange}
          placeholder={DEFAULT_NAMING_TEMPLATE}
          disabled={disabled}
          aria-invalid={!namingCheck.ok}
          aria-describedby={`${uid}-naming-preview`}
        />
        <div id={`${uid}-naming-preview`} className="sr-naming-preview">
          <span>
            {namingCheck.ok ? 'Preview:' : namingCheck.error}
          </span>
          <code>{outputPreview}</code>
        </div>
      </div>
    </section>
  );
}
