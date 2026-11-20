import type { JobMode, Profile } from '../api/types';
import { updateRackSettings } from '../state/actions';
import { scheduleLivePreview } from '../state/liveAudition';
import { DEFAULT_STUDIO_RACK_SETTINGS, type StudioRackSettings } from '../state/presets';
import { isProfileUnavailable, jobInFlight, useStore } from '../state/store';

export interface SimpleGoal {
  id: string;
  title: string;
  desc: string;
  mode: JobMode;
  profile: Profile;
  fallbackProfile?: Profile;
  settings: Partial<StudioRackSettings>;
}

export const SIMPLE_GOALS: readonly SimpleGoal[] = [
  {
    id: 'auto_smart',
    title: 'Auto Smart Clean',
    desc: 'Picks the safest engine automatically',
    mode: 'smart_safe',
    profile: 'studio',
    settings: {
      strength: 0.78,
      ambienceFloorDb: -36,
      deliveryPreset: 'match_input',
      deplosive: false,
      mouthDeclick: false,
      declip: false,
      deess: false,
      tonalRestoration: false,
      levelRider: false,
    },
  },
  {
    id: 'voice_memo',
    title: 'Voice Memo Rescue',
    desc: 'Phone recordings & noisy rooms',
    mode: 'natural',
    profile: 'studio_pro',
    fallbackProfile: 'studio',
    settings: {
      strength: 0.85,
      ambienceFloorDb: -36,
      deliveryPreset: 'podcast',
      mouthDeclick: true,
      levelRider: true,
      deplosive: false,
      declip: false,
      deess: false,
      tonalRestoration: true,
    },
  },
  {
    id: 'podcast',
    title: 'Podcast & YouTube',
    desc: 'Rich, consistent loudness (-16 LUFS)',
    mode: 'natural',
    profile: 'studio',
    settings: {
      strength: 0.75,
      ambienceFloorDb: -36,
      deliveryPreset: 'podcast',
      deplosive: true,
      deess: true,
      levelRider: true,
      tonalRestoration: true,
      mouthDeclick: false,
      declip: false,
    },
  },
  {
    id: 'remove_echo',
    title: 'Remove Room Echo',
    desc: 'Tightens hollow rooms & reverb',
    mode: 'natural',
    profile: 'dereverb',
    fallbackProfile: 'studio',
    settings: {
      strength: 0.82,
      ambienceFloorDb: -42,
      deliveryPreset: 'match_input',
      deplosive: false,
      mouthDeclick: false,
      declip: false,
      deess: false,
      tonalRestoration: false,
      levelRider: false,
    },
  },
  {
    id: 'film_interview',
    title: 'Film & Interview',
    desc: 'Natural ambience + broadcast target',
    mode: 'natural',
    profile: 'studio',
    settings: {
      strength: 0.65,
      ambienceFloorDb: -30,
      deliveryPreset: 'ebu_r128',
      deplosive: true,
      mouthDeclick: true,
      declip: false,
      deess: false,
      tonalRestoration: false,
      levelRider: false,
    },
  },
];

export function GoalPresets() {
  const activeGoalId = useStore((s) => s.activeGoalId);
  const setActiveGoalId = useStore((s) => s.setActiveGoalId);
  const setMode = useStore((s) => s.setMode);
  const setProfile = useStore((s) => s.setProfile);
  const capabilities = useStore((s) => s.capabilities);
  const rackSettings = useStore((s) => s.rackSettings);
  const pushToast = useStore((s) => s.pushToast);
  const busy = useStore((s) => jobInFlight(s.job) || s.analyzing);

  const applyGoal = (goal: SimpleGoal): void => {
    if (busy) return;
    const chosenProfile =
      goal.fallbackProfile && isProfileUnavailable(capabilities, goal.profile)
        ? goal.fallbackProfile
        : goal.profile;
    setActiveGoalId(goal.id);
    setMode(goal.mode);
    setProfile(chosenProfile);
    updateRackSettings({
      ...DEFAULT_STUDIO_RACK_SETTINGS,
      ...rackSettings,
      ...goal.settings,
    });
    pushToast(`Preset applied: ${goal.title}`, 'info', 2600);
    scheduleLivePreview(220);
  };

  return (
    <div className="goal-presets" role="group" aria-label="Cleaning goals">
      {SIMPLE_GOALS.map((goal) => (
        <button
          key={goal.id}
          type="button"
          className="goal-card"
          aria-pressed={activeGoalId === goal.id}
          disabled={busy}
          onClick={() => applyGoal(goal)}
        >
          <span className="goal-card-title">{goal.title}</span>
          <span className="goal-card-desc">{goal.desc}</span>
        </button>
      ))}
    </div>
  );
}
