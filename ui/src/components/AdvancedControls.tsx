import type { Profile } from '../api/types';
import { useStore } from '../state/store';
import { toggleAdvancedOpen } from '../state/actions';
import { ProfileControl } from './ProfileControl';
import { RestoreControl } from './RestoreControl';

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

  const profileLabel = PROFILE_LABEL[profile] ?? profile;
  const modeLabel = mode === 'restore' ? `Restore${speakerId ? ` (${speakerId})` : ''}` : 'Natural';
  const summary = `${profileLabel} · ${modeLabel}`;

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
        </div>
      ) : null}
    </div>
  );
}
