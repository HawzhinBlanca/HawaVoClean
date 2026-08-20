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
          onChange={setProfile}
          options={[
            { value: 'studio', label: 'Studio' },
            { value: 'production', label: 'Production' },
          ]}
        />
      </div>
      <span className="pc-sub">
        {profile === 'studio' ? 'DFN3 restoration core' : 'Wiener DSP core'}
      </span>
    </div>
  );
}
