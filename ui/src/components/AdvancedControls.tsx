import type { Profile } from '../api/types';
import { jobInFlight, useStore } from '../state/store';
import { toggleAdvancedOpen, updateRackSettings } from '../state/actions';
import { ProfileControl } from './ProfileControl';
import { RestoreControl } from './RestoreControl';
import { StudioRack } from './StudioRack';
import { isVideoFilePath } from './VideoPreview';

const PROFILE_LABEL: Record<Profile, string> = {
  studio_pro: 'Studio Pro',
  dereverb: 'Dereverb',
  studio: 'Studio',
  lowband: 'Low-band',
  production: 'Production',
};

export function AdvancedControls() {
  const advancedOpen = useStore((s) => s.advancedOpen);
  const profile = useStore((s) => s.profile);
  const mode = useStore((s) => s.mode);
  const speakerId = useStore((s) => s.speakerId);
  const source = useStore((s) => s.source);
  const job = useStore((s) => s.job);
  const rackSettings = useStore((s) => s.rackSettings);
  const batchNamingTemplate = useStore((s) => s.batchNamingTemplate);
  const setBatchNamingTemplate = useStore((s) => s.setBatchNamingTemplate);

  const profileLabel = PROFILE_LABEL[profile] ?? profile;
  const modeLabel = mode === 'restore' ? `Restore${speakerId ? ` (${speakerId})` : ''}` : 'Natural';
  const summary = `${profileLabel} · ${modeLabel}`;
  const running = jobInFlight(job);
  const sampleFileName = source?.name ? source.name.replace(/\.[^.]+$/, '') : 'interview';

  return (
    <div className={`advanced-controls${advancedOpen ? ' open' : ''}`}>
      <button
        type="button"
        className="advanced-toggle"
        onClick={toggleAdvancedOpen}
        aria-expanded={advancedOpen}
        aria-controls="advanced-controls-panel"
      >
        <span className="adv-left">
          <span className="adv-icon" aria-hidden="true">
            {advancedOpen ? '▾' : '▸'}
          </span>
          <span className="adv-title caps">Advanced Controls</span>
        </span>
        <span className="adv-summary">{summary}</span>
      </button>

      {advancedOpen ? (
        <div id="advanced-controls-panel" className="advanced-body">
          <ProfileControl />
          <RestoreControl />
          <StudioRack
            settings={rackSettings}
            onChange={updateRackSettings}
            isVideoSource={isVideoFilePath(source?.path)}
            disabled={running}
            namingTemplate={batchNamingTemplate}
            onNamingTemplateChange={setBatchNamingTemplate}
            sampleFileName={sampleFileName}
            activeProfile={profile}
          />
        </div>
      ) : null}
    </div>
  );
}
