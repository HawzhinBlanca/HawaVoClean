import type { Profile } from '../api/types';
import { useStore } from '../state/store';
import { Segmented } from './Segmented';

export function ProfileControl() {
  const profile = useStore((s) => s.profile);
  const setProfile = useStore((s) => s.setProfile);
  const running = useStore(
    (s) => !!s.job?.status && (s.job.status.state === 'running' || s.job.status.state === 'queued'),
  );
  return (
    <div className="row-between">
      <div style={{ display: 'flex', flexDirection: 'column', gap: 2, minWidth: 0 }}>
        <span className="caps">Profile</span>
        <span
          style={{
            fontSize: 'var(--fs-xs)',
            color: 'var(--fg-4)',
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
          }}
        >
          {profile === 'studio' ? 'DFN3 restoration core' : 'Wiener DSP core'}
        </span>
      </div>
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
  );
}
