import type { Profile } from '../api/types';
import { useStore } from '../state/store';
import { Segmented } from './Segmented';

/**
 * B8 · the sub-line ("DFN3 restoration core") used to share a row with the
 * segmented control and lost the fight at 960, where it ellipsised to
 * "DFN3 restoration …". It is the one line that says *what the profile
 * actually is*, so it now gets the panel's full width on its own row instead
 * of a share of it — at every viewport, so there is only one layout to reason
 * about.
 */
/** What each profile actually is, in the one line the control has for it. */
const CORE_LINE: Record<Profile, string> = {
  studio: 'DFN3 restoration core',
  lowband: 'DFN3 under 1 kHz, original above',
  production: 'Wiener DSP core',
};

export function ProfileControl() {
  const profile = useStore((s) => s.profile);
  const setProfile = useStore((s) => s.setProfile);
  const running = useStore(
    (s) => !!s.job?.status && (s.job.status.state === 'running' || s.job.status.state === 'queued'),
  );
  return (
    <div className="profilectl">
      <div className="pc-row">
        <span className="caps">Profile</span>
        <Segmented<Profile>
          ariaLabel="Profile"
          value={profile}
          disabled={running}
          disabledReason="The profile is part of the run — cancel it to change this."
          onChange={setProfile}
          options={[
            { value: 'studio', label: 'Studio' },
            { value: 'lowband', label: 'Low-band' },
            { value: 'production', label: 'Production' },
          ]}
        />
      </div>
      <span className="pc-sub">{CORE_LINE[profile]}</span>
    </div>
  );
}
