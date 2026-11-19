import type { Profile } from '../api/types';
import { useI18n } from '../state/i18n';
import {
  getRouteBlockedReason,
  isRouteBlocked,
  isRouteUnavailable,
  jobInFlight,
  useStore,
} from '../state/store';
import { Segmented, type SegOption } from './Segmented';

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
  studio_pro: '48 kHz neural cleanup (MossFormer2)',
  dereverb: '48 kHz neural de-reverb + cleanup',
  studio: 'DFN3 restoration core',
  lowband: 'DFN3 under 1 kHz, original above',
  production: 'Wiener DSP core',
};

function acceleratedCoreLine(base: string, fallback: string, providers?: string[]): string {
  if (providers?.includes('mps')) return `${base} (MPS)`;
  if (providers?.includes('cuda')) return `${base} (CUDA)`;
  return fallback;
}

export function ProfileControl() {
  const { t } = useI18n();
  const profile = useStore((s) => s.profile);
  const setProfile = useStore((s) => s.setProfile);
  const capabilities = useStore((s) => s.capabilities);
  // `jobInFlight`: a job the engine has accepted but not yet reported on has
  // `status: null`, and the hand-written check read that window as idle.
  const running = useStore((s) => jobInFlight(s.job));

  const optionalOption = (
    value: 'studio_pro' | 'dereverb',
    label: string,
    titleName: string,
    base: string,
  ): { option: SegOption<Profile>; line: string; available: boolean; reason: string } => {
    const cap = capabilities?.find((c) => c.capability_id === value);
    const available = !isRouteUnavailable(capabilities, value);
    const line = acceleratedCoreLine(base, CORE_LINE[value], cap?.providers);
    const reason =
      cap?.reason ||
      (capabilities
        ? `${titleName} is unavailable on this machine.`
        : `Checking ${titleName} availability…`);
    return {
      available,
      line,
      reason,
      option: {
        value,
        label,
        description: available ? line : `${line} — ${reason}`,
        ...(!available ? { disabled: true, disabledReason: reason } : {}),
      },
    };
  };

  const studioPro = optionalOption(
    'studio_pro',
    t.profileStudioPro,
    'Studio Pro',
    '48 kHz neural cleanup',
  );
  const dereverb = optionalOption(
    'dereverb',
    t.profileDereverb,
    'Dereverb',
    '48 kHz neural de-reverb + cleanup',
  );

  const routeOption = (
    value: 'studio' | 'lowband' | 'production',
    label: string,
  ): SegOption<Profile> => {
    const blocked = isRouteBlocked(capabilities, value);
    const reason = blocked ? getRouteBlockedReason(capabilities, value) : null;
    const base = CORE_LINE[value];
    return {
      value,
      label,
      description: reason ? `${base} — ${reason}` : base,
      ...(blocked ? { disabled: true } : {}),
      ...(reason ? { disabledReason: reason } : {}),
    };
  };

  const options: SegOption<Profile>[] = [
    studioPro.option,
    dereverb.option,
    routeOption('studio', t.profileStudio),
    routeOption('lowband', t.profileLowband),
    routeOption('production', t.profileProduction),
  ];

  const activeLine =
    profile === 'studio_pro'
      ? studioPro.available
        ? studioPro.line
        : studioPro.reason
      : profile === 'dereverb'
        ? dereverb.available
          ? dereverb.line
          : dereverb.reason
        : (getRouteBlockedReason(capabilities, profile) ?? CORE_LINE[profile]);

  return (
    <div className="profilectl">
      <div className="pc-row">
        <span className="caps">Profile</span>
        <Segmented<Profile>
          ariaLabel="Profile"
          className="profile-seg"
          value={profile}
          disabled={running}
          disabledReason="The profile is part of the run — cancel it to change this."
          onChange={setProfile}
          // Each option carries its own core line. The visible `.pc-sub` below
          // shows only the *selected* one, so arrowing through the group told a
          // listener nothing about what they were about to choose — the one
          // line that says what a profile actually is was on screen but
          // unassociated with the control that sets it.
          options={options}
        />
      </div>
      <span className="pc-sub">{activeLine}</span>
    </div>
  );
}
