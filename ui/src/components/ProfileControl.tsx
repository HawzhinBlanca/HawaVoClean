import type { Profile } from '../api/types';
import { jobInFlight, useStore } from '../state/store';
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
  // `jobInFlight`: a job the engine has accepted but not yet reported on has
  // `status: null`, and the hand-written check read that window as idle.
  const running = useStore((s) => jobInFlight(s.job));
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
          // Each option carries its own core line. The visible `.pc-sub` below
          // shows only the *selected* one, so arrowing through the group told a
          // listener nothing about what they were about to choose — the one
          // line that says what a profile actually is was on screen but
          // unassociated with the control that sets it.
          options={[
            { value: 'studio', label: 'Studio', description: CORE_LINE.studio },
            { value: 'lowband', label: 'Low-band', description: CORE_LINE.lowband },
            { value: 'production', label: 'Production', description: CORE_LINE.production },
          ]}
        />
      </div>
      <span className="pc-sub">{CORE_LINE[profile]}</span>
    </div>
  );
}
