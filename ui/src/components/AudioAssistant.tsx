import { useEffect, useReducer, useRef, useState, type FormEvent } from 'react';
import type {
  AcousticContext,
  AudioAnalysis,
  AudioAssistantStatus,
  AudioChatMessage,
  AudioPlan,
  BandwidthShape,
  CapabilityStatusV1,
  CleanupProfile,
  DeliveryPresetId,
  ExportFormatId,
  HawaVoCleanReport,
  ProcessingStrategyV1,
} from '../api/types';
import { getPlayer } from '../audio/player';
import { SPECTRAL_SELECTION_EVENT } from '../render/stftSpectrogram';
import * as actions from '../state/actions';
import { setLocale, useI18n } from '../state/i18n';
import {
  assistantReducer,
  deriveWorkspacePhase,
  INITIAL_ASSISTANT_STATE,
} from '../state/machines';
import { DEFAULT_STUDIO_RACK_SETTINGS, type StudioRackSettings } from '../state/presets';
import { getState, isProfileUnavailable, useStore } from '../state/store';
import '../styles/assistant.css';

const EXAMPLES = [
  'Clean the voice and reduce room noise',
  'Keep the speech, remove the music',
  'Remove the dog barking',
  'Remove room echo, fix harsh S sounds, cut rumble below 90 Hz, master for Podcast',
  'Hear what was removed',
];
const CLEANUP_PROFILES: readonly CleanupProfile[] = ['production', 'studio', 'studio_pro', 'dereverb', 'lowband'];
const BANDWIDTH_SHAPES: readonly BandwidthShape[] = ['unknown', 'silence', 'fullband', 'steep_lowpass'];
const MAX_SEPARATION_DURATION_S = 3600;
const AUTORUN_STORAGE_KEY = 'hawa.assistant.autoRun';

function readAutoRunPref(): boolean {
  try {
    return typeof localStorage !== 'undefined' && localStorage.getItem(AUTORUN_STORAGE_KEY) === '1';
  } catch {
    return false;
  }
}

function writeAutoRunPref(next: boolean): void {
  try {
    if (typeof localStorage !== 'undefined') {
      localStorage.setItem(AUTORUN_STORAGE_KEY, next ? '1' : '0');
    }
  } catch {
    // Ignore storage quota / privacy restrictions
  }
}

function formatClock(seconds: number): string {
  const rounded = Math.round(seconds * 10) / 10;
  const mins = Math.floor(rounded / 60);
  const secs = Math.round((rounded - mins * 60) * 10) / 10;
  const whole = Math.floor(secs);
  const frac = Math.round((secs - whole) * 10);
  const secStr = `${String(whole).padStart(2, '0')}${frac > 0 ? `.${frac}` : ''}`;
  return `${mins}:${secStr}`;
}

function formatBandRange(fLowHz: number, fHighHz: number): string {
  const lo = Math.max(0, Math.min(fLowHz, fHighHz));
  const hi = Math.max(fLowHz, fHighHz);
  if (hi < 1000) {
    return `${Math.round(lo)}–${Math.round(hi)} Hz`;
  }
  const fmtK = (hz: number): string => (hz % 1000 === 0 ? `${hz / 1000}` : (hz / 1000).toFixed(1));
  if (lo >= 1000) {
    return `${fmtK(lo)}–${fmtK(hi)} kHz`;
  }
  return `${Math.round(lo)} Hz–${fmtK(hi)} kHz`;
}

function isImmediateControlOp(op: AudioPlan['operation']): boolean {
  return op === 'playback' || op === 'undo' || op === 'redo' || op === 'reset' || op === 'ui_action';
}

function collectPlanChips(plan: AudioPlan): string[] {
  const chips: string[] = [];
  if (plan.hpf_enabled === false) {
    chips.push('HPF Off');
  } else if (typeof plan.hpf_cutoff_hz === 'number') {
    chips.push(`HPF ${Math.round(plan.hpf_cutoff_hz)} Hz`);
  }
  if (plan.pro_repair) {
    if (plan.pro_repair.deplosive) chips.push('De-Plosive');
    if (plan.pro_repair.declick) chips.push('De-Click');
    if (plan.pro_repair.declip) chips.push('De-Clip');
    if (plan.pro_repair.deess) chips.push('De-Ess');
    if (plan.pro_repair.tonal_restoration) chips.push('Tonal');
    if (plan.pro_repair.level_rider) chips.push('Leveler');
  }
  if (typeof plan.ambience_floor_db === 'number') {
    chips.push(`Room ${Math.round(plan.ambience_floor_db)} dB`);
  }
  if (plan.delivery_preset) {
    const labelMap: Record<string, string> = {
      match_input: 'Match Input',
      podcast: 'Podcast -16 LUFS',
      ebu_r128: 'Broadcast -23 LUFS',
      atsc_a85: 'TV -24 LKFS',
      dialogue_gated: 'Dialogue -27 LKFS',
      streaming: 'Streaming -14 LUFS',
    };
    chips.push(labelMap[plan.delivery_preset] ?? plan.delivery_preset);
  }
  if (plan.stereo_mode && plan.stereo_mode !== 'auto') {
    chips.push(plan.stereo_mode.replace(/_/g, ' '));
  }
  if (plan.spectral_repair) {
    chips.push(
      `${formatBandRange(plan.spectral_repair.f_low_hz, plan.spectral_repair.f_high_hz)} · -${Math.round(plan.spectral_repair.attenuation_db)} dB`,
    );
  }
  if (plan.export_format) {
    chips.push(plan.export_format.toUpperCase());
  }
  if (plan.unit_override) {
    chips.push(`Unit #${plan.unit_override.unit_id} · ${plan.unit_override.action}`);
  }
  return chips;
}

function buildProRepairPatch(plan: AudioPlan): Record<string, boolean> | undefined {
  const pr = plan.pro_repair;
  if (!pr) return undefined;
  const out: Record<string, boolean> = {};
  if (typeof pr.deplosive === 'boolean') out.deplosive = pr.deplosive;
  if (typeof pr.declick === 'boolean') out.declick = pr.declick;
  if (typeof pr.declip === 'boolean') out.declip = pr.declip;
  if (typeof pr.deess === 'boolean') out.deess = pr.deess;
  if (typeof pr.tonal_restoration === 'boolean') out.tonal_restoration = pr.tonal_restoration;
  if (typeof pr.level_rider === 'boolean') out.level_rider = pr.level_rider;
  return Object.keys(out).length > 0 ? out : undefined;
}

function getAction<K extends keyof typeof actions>(name: K): (typeof actions)[K] | undefined {
  try {
    const fn = actions[name];
    return typeof fn === 'function' ? fn : undefined;
  } catch {
    return undefined;
  }
}

function syncPlanToStudioRack(plan: AudioPlan): void {
  const st = getState();
  const prev = st.rackSettings;
  let changed = false;
  const next: StudioRackSettings = { ...prev };

  if (typeof plan.strength === 'number' && Number.isFinite(plan.strength) && plan.strength < 1.0) {
    next.strength = Math.max(0.1, Math.min(1.0, plan.strength));
    changed = true;
  }
  if (typeof plan.hpf_enabled === 'boolean') {
    next.hpfEnabled = plan.hpf_enabled;
    changed = true;
  }
  if (typeof plan.hpf_cutoff_hz === 'number' && Number.isFinite(plan.hpf_cutoff_hz)) {
    next.hpfEnabled = true;
    next.hpfCutoffHz = Math.max(20, Math.min(250, Math.round(plan.hpf_cutoff_hz)));
    changed = true;
  }
  if (plan.pro_repair) {
    if (typeof plan.pro_repair.deplosive === 'boolean') {
      next.deplosive = plan.pro_repair.deplosive;
      changed = true;
    }
    if (typeof plan.pro_repair.declick === 'boolean') {
      next.mouthDeclick = plan.pro_repair.declick;
      changed = true;
    }
    if (typeof plan.pro_repair.declip === 'boolean') {
      next.declip = plan.pro_repair.declip;
      changed = true;
    }
    if (typeof plan.pro_repair.deess === 'boolean') {
      next.deess = plan.pro_repair.deess;
      changed = true;
    }
    if (typeof plan.pro_repair.tonal_restoration === 'boolean') {
      next.tonalRestoration = plan.pro_repair.tonal_restoration;
      changed = true;
    }
    if (typeof plan.pro_repair.level_rider === 'boolean') {
      next.levelRider = plan.pro_repair.level_rider;
      changed = true;
    }
  }
  if (typeof plan.ambience_floor_db === 'number' && Number.isFinite(plan.ambience_floor_db)) {
    next.ambienceFloorDb = Math.max(-60, Math.min(-24, plan.ambience_floor_db));
    changed = true;
  }
  if (plan.delivery_preset && plan.delivery_preset !== 'dialogue_gated') {
    next.deliveryPreset = plan.delivery_preset as DeliveryPresetId;
    changed = true;
  }
  if (plan.stereo_mode) {
    next.stereoMode = plan.stereo_mode;
    changed = true;
  }
  if (plan.export_format) {
    next.exportFormat = plan.export_format;
    changed = true;
  }
  if (changed) {
    getAction('updateRackSettings')?.(next);
  }
  if (plan.cleanup_profile && typeof st.setProfile === 'function') {
    st.setProfile(plan.cleanup_profile);
  }
  if (plan.time_range && typeof st.setSelectionRange === 'function') {
    st.setSelectionRange({ start: plan.time_range.start_s, end: plan.time_range.end_s });
  }
}

function executeImmediateControl(plan: AudioPlan): string | null {
  if (plan.operation === 'undo') {
    getAction('undoAction')?.();
    return null;
  }
  if (plan.operation === 'redo') {
    getAction('redoAction')?.();
    return null;
  }
  if (plan.operation === 'reset') {
    getAction('updateRackSettings')?.({ ...DEFAULT_STUDIO_RACK_SETTINGS });
    getAction('clearSelectionRange')?.();
    return null;
  }
  if (plan.operation === 'ui_action' && plan.ui_action) {
    switch (plan.ui_action) {
      case 'switch_pro':
        getState().setWorkspaceMode?.('pro');
        return null;
      case 'switch_simple':
        getState().setWorkspaceMode?.('simple');
        return null;
      case 'switch_kurdish':
        setLocale('ckb');
        return null;
      case 'switch_english':
        setLocale('en');
        return null;
      case 'open_models':
        getState().setModelsDrawerOpen?.(true);
        return null;
      case 'diagnose':
        return null;
    }
  }
  if (plan.operation === 'playback' && plan.playback_action) {
    const player = typeof getPlayer === 'function' ? getPlayer() : null;
    switch (plan.playback_action) {
      case 'play':
        if (player) void player.play();
        else getAction('togglePlay')?.();
        return null;
      case 'pause':
        if (player) player.pause();
        else getAction('togglePlay')?.();
        return null;
      case 'play_original':
        player?.setDeltaMode?.(false);
        getAction('setAb')?.('original');
        if (player) void player.play();
        return null;
      case 'play_cleaned':
        player?.setDeltaMode?.(false);
        getAction('setAb')?.('cleaned');
        if (player) void player.play();
        return null;
      case 'hear_removed':
        player?.setDeltaMode?.(true);
        if (player) void player.play();
        return null;
      case 'loop_on': {
        const range = getState().selectionRange;
        const playSel = getAction('playSelection');
        if (range && playSel) {
          playSel(true);
        } else {
          player?.setLoopEnabled?.(true);
        }
        return null;
      }
      case 'loop_off':
        player?.setLoopEnabled?.(false);
        return null;
      case 'seek':
        if (typeof plan.seek_s === 'number' && Number.isFinite(plan.seek_s)) {
          getAction('seekTo')?.(plan.seek_s);
        }
        return null;
      case 'set_mix':
        if (typeof plan.wet_dry_mix === 'number' && Number.isFinite(plan.wet_dry_mix)) {
          player?.setWetDryMix?.(plan.wet_dry_mix);
        }
        return null;
    }
  }
  return null;
}

/** Compact bounded numeric/enum telemetry only — never paths, filenames, or audio bytes. */
function buildAcousticContext(
  original: AudioAnalysis | null,
  report: HawaVoCleanReport | null,
  capabilities: CapabilityStatusV1[] | null,
): AcousticContext | undefined {
  const ctx: AcousticContext = {};
  if (original) {
    if (Number.isFinite(original.duration_s) && original.duration_s >= 0 && original.duration_s <= 86400) {
      ctx.duration_s = original.duration_s;
    }
    if (Number.isInteger(original.sample_rate) && original.sample_rate >= 8000 && original.sample_rate <= 192000) {
      ctx.sample_rate = original.sample_rate;
    }
    if (Number.isInteger(original.channels) && original.channels >= 1 && original.channels <= 16) {
      ctx.channels = original.channels;
    }
    if (original.noise_floor_db !== null && Number.isFinite(original.noise_floor_db) && original.noise_floor_db >= -160 && original.noise_floor_db <= 24) {
      ctx.noise_floor_db = original.noise_floor_db;
    }
    const lufs = original.loudness?.integrated_lufs;
    if (lufs !== null && lufs !== undefined && Number.isFinite(lufs) && lufs >= -120 && lufs <= 24) {
      ctx.integrated_lufs = lufs;
    }
  }
  const detections = report?.restoration?.detections;
  if (detections) {
    if (typeof detections.music_risk === 'number' && Number.isFinite(detections.music_risk) && detections.music_risk >= 0 && detections.music_risk <= 1) {
      ctx.music_risk = detections.music_risk;
    }
    if (typeof detections.clipping_fraction === 'number' && Number.isFinite(detections.clipping_fraction) && detections.clipping_fraction >= 0 && detections.clipping_fraction <= 1) {
      ctx.clipping_risk = detections.clipping_fraction;
    }
    if (typeof detections.cutoff_hz === 'number' && Number.isFinite(detections.cutoff_hz) && detections.cutoff_hz > 0 && detections.cutoff_hz <= 96000) {
      ctx.estimated_cutoff_hz = detections.cutoff_hz;
    }
  }
  const bw = report?.restoration?.bandwidth;
  if (bw) {
    if (Number.isFinite(bw.effective_cutoff_hz) && bw.effective_cutoff_hz > 0 && bw.effective_cutoff_hz <= 96000) {
      ctx.estimated_cutoff_hz = bw.effective_cutoff_hz;
    }
    if ((BANDWIDTH_SHAPES as readonly string[]).includes(bw.shape)) {
      ctx.bandwidth_shape = bw.shape as BandwidthShape;
    }
  }
  if (capabilities) {
    const profiles = CLEANUP_PROFILES.filter((profile) => {
      const cap = capabilities.find((c) => c.capability_id === profile);
      return Boolean(cap?.available && cap.maturity !== 'blocked');
    });
    if (profiles.length > 0) ctx.available_profiles = profiles;
    const sep = capabilities.find((c) => c.capability_id === 'separate');
    if (sep) ctx.separation_available = Boolean(sep.available && sep.maturity !== 'blocked');
  }
  const st = getState();
  if (st.profile && (CLEANUP_PROFILES as readonly string[]).includes(st.profile)) {
    ctx.current_profile = st.profile as CleanupProfile;
  }
  const rack = st.rackSettings;
  if (rack) {
    if (Number.isFinite(rack.strength) && rack.strength >= 0.1 && rack.strength <= 1.0) {
      ctx.current_strength = rack.strength;
    }
    if (
      rack.hpfEnabled !== false &&
      Number.isFinite(rack.hpfCutoffHz) &&
      rack.hpfCutoffHz >= 20 &&
      rack.hpfCutoffHz <= 250
    ) {
      ctx.current_hpf_hz = rack.hpfCutoffHz;
    }
    if (rack.deliveryPreset) {
      ctx.current_delivery_preset = rack.deliveryPreset;
    }
  }
  const outLufs = report?.output?.integrated_lufs;
  if (typeof outLufs === 'number' && Number.isFinite(outLufs) && outLufs >= -120 && outLufs <= 24) {
    ctx.output_lufs = outLufs;
  }
  const outPeak = report?.output?.true_peak_dbtp;
  if (typeof outPeak === 'number' && Number.isFinite(outPeak) && outPeak >= -120 && outPeak <= 24) {
    ctx.output_true_peak_dbtp = outPeak;
  }
  if (typeof report?.summary?.enhanced === 'number' && Number.isInteger(report.summary.enhanced)) {
    ctx.units_enhanced = report.summary.enhanced;
  }
  if (typeof report?.summary?.reverted === 'number' && Number.isInteger(report.summary.reverted)) {
    ctx.units_reverted = report.summary.reverted;
  }
  return Object.keys(ctx).length > 0 ? ctx : undefined;
}

/** Universal conversational audio controller: plans & executes every studio operation via chat. */
export function AudioAssistant() {
  const { t, locale } = useI18n();
  const isCkb = locale === 'ckb';
  const source = useStore((s) => s.source);
  const original = useStore((s) => s.original);
  const cleanedPath = useStore((s) => s.cleanedPath);
  const report = useStore((s) => s.report);
  const client = useStore((s) => s.client);
  const engineReady = useStore((s) => s.engineStatus === 'ready');
  const workspacePhase = useStore(deriveWorkspacePhase);
  const analyzing = workspacePhase === 'uploading' || workspacePhase === 'analyzing';
  const running = workspacePhase === 'processing';
  const capabilities = useStore((s) => s.capabilities);
  const job = useStore((s) => s.job);
  const pendingChatPrompt = useStore((s) => s.pendingChatPrompt);
  const setPendingChatPrompt = useStore((s) => s.setPendingChatPrompt);
  const selectionRange = useStore((s) => s.selectionRange);
  const setSelectionRange = useStore((s) => s.setSelectionRange);
  const spectralSelection = useStore((s) => s.spectralSelection);
  const setSpectralSelection = useStore((s) => s.setSpectralSelection);
  const sourceKey = source?.sourceId || source?.path || '';
  const currentSource = useRef(sourceKey);
  currentSource.current = sourceKey;
  const request = useRef<AbortController | null>(null);
  const sequence = useRef(0);
  const applying = useRef(false);
  const planningRequest = useRef(false);
  const lastChatJobIdRef = useRef<string | null>(null);
  const [messages, setMessages] = useState<AudioChatMessage[]>([]);
  const [draft, setDraft] = useState('');
  const [machine, dispatch] = useReducer(assistantReducer, INITIAL_ASSISTANT_STATE);
  const [status, setStatus] = useState<AudioAssistantStatus | null>(null);
  const [lastProvider, setLastProvider] = useState<AudioPlan['provider'] | null>(null);
  const [modelSize, setModelSize] = useState<'small' | 'base' | 'large'>('large');
  const [autoRun, setAutoRun] = useState<boolean>(readAutoRunPref);
  const autoRunRef = useRef(autoRun);
  autoRunRef.current = autoRun;
  const transcript = useRef<HTMLDivElement | null>(null);

  const plan = machine.phase === 'planned' || machine.phase === 'applying' ? machine.plan : null;
  const planning = machine.phase === 'planning';
  const submitting = machine.phase === 'applying';
  const error =
    machine.phase === 'error'
      ? machine.message
      : machine.phase === 'planned'
        ? machine.applyError
        : null;

  useEffect(() => {
    sequence.current += 1;
    request.current?.abort();
    setMessages([]);
    setDraft('');
    dispatch({ type: 'RESET' });
    planningRequest.current = false;
    applying.current = false;
    lastChatJobIdRef.current = null;
    setLastProvider(null);
    return () => {
      sequence.current += 1;
      request.current?.abort();
    };
  }, [sourceKey]);

  useEffect(() => {
    setStatus(null);
    if (!client || !engineReady) return;
    const ac = new AbortController();
    void client.assistantStatus(ac.signal).then((value) => {
      if (!ac.signal.aborted) setStatus(value);
    }).catch(() => { /* Planning reports any real failure; old engines remain usable. */ });
    return () => ac.abort();
  }, [client, engineReady]);

  useEffect(() => {
    const el = transcript.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, planning]);

  const isSeparation = plan?.operation === 'extract' || plan?.operation === 'remove';
  const isRestore = plan?.operation === 'restore';
  const separationCapability = capabilities?.find((c) => c.capability_id === 'separate');
  const restoreCapability = capabilities?.find((c) => c.capability_id === 'restore_source');

  const activeRegion = (() => {
    if (
      !selectionRange ||
      !Number.isFinite(selectionRange.start) ||
      !Number.isFinite(selectionRange.end)
    ) {
      return null;
    }
    const maxDur =
      original && Number.isFinite(original.duration_s) && original.duration_s > 0
        ? original.duration_s
        : null;
    const start_s = Math.max(0, Math.round(selectionRange.start * 1000) / 1000);
    const rawEnd = Math.round(selectionRange.end * 1000) / 1000;
    const end_s = maxDur !== null ? Math.min(maxDur, rawEnd) : rawEnd;
    return end_s > start_s ? { start_s, end_s } : null;
  })();

  const activeSpectralBand = (() => {
    if (
      !spectralSelection ||
      !Number.isFinite(spectralSelection.startS) ||
      !Number.isFinite(spectralSelection.endS) ||
      !Number.isFinite(spectralSelection.fLowHz) ||
      !Number.isFinite(spectralSelection.fHighHz)
    ) {
      return null;
    }
    const maxDur =
      original && Number.isFinite(original.duration_s) && original.duration_s > 0
        ? original.duration_s
        : null;
    const startS = Math.max(0, Math.round(spectralSelection.startS * 1000) / 1000);
    const rawEnd = Math.round(spectralSelection.endS * 1000) / 1000;
    const endS = maxDur !== null ? Math.min(maxDur, rawEnd) : rawEnd;
    const fLowHz = Math.max(0, Math.min(spectralSelection.fLowHz, spectralSelection.fHighHz));
    const fHighHz = Math.max(spectralSelection.fLowHz, spectralSelection.fHighHz);
    return endS > startS && fHighHz > fLowHz ? { startS, endS, fLowHz, fHighHz } : null;
  })();

  const computeBlockReason = (candidate: AudioPlan | null): string | null => {
    if (!candidate) return null;
    if (isImmediateControlOp(candidate.operation)) return null;
    if (!source || !original) return 'Load a recording before applying an edit.';
    if (!engineReady) return 'The audio engine is offline.';
    if (analyzing) return 'Wait for the recording to finish loading.';
    if (running || submitting) return 'An audio edit is already in progress.';
    const candIsSep = candidate.operation === 'extract' || candidate.operation === 'remove';
    const candProfile: CleanupProfile = candidate.cleanup_profile ?? 'studio';
    const candCleanupCap = capabilities?.find((c) => c.capability_id === candProfile);
    if (candIsSep && (!separationCapability?.available || separationCapability.maturity === 'blocked')) {
      return separationCapability?.reason || 'Sound separation is not installed yet. Voice cleanup is still available.';
    }
    if (candIsSep && (!Number.isFinite(original.duration_s) || original.duration_s > MAX_SEPARATION_DURATION_S || original.channels > 2)) {
      return 'Sound separation currently supports recordings up to 60 minutes in mono or stereo. Open a shorter clip to try it.';
    }
    if ((candidate.operation === 'clean' || candIsSep) && candidate.cleanup_profile && isProfileUnavailable(capabilities, candidate.cleanup_profile)) {
      return (
        candCleanupCap?.reason ||
        (candidate.cleanup_profile === 'studio_pro'
          ? 'Studio Pro is unavailable on this machine.'
          : candidate.cleanup_profile === 'dereverb'
            ? 'Dereverb is unavailable on this machine.'
            : 'This cleanup option is unavailable.')
      );
    }
    if (candidate.operation === 'clean' && candCleanupCap && (!candCleanupCap.available || candCleanupCap.maturity === 'blocked')) {
      return candCleanupCap.reason || 'This cleanup option is unavailable.';
    }
    if (candidate.operation === 'restore' && (!restoreCapability?.available || restoreCapability.maturity === 'blocked')) {
      return restoreCapability?.reason || 'Bandwidth restoration is currently unavailable.';
    }
    if (candidate.operation === 'export' && candidate.export_format === 'video_mux' && original.has_video === false) {
      return 'This recording has no video track to mux. Ask to export WAV, FLAC, or AAC instead.';
    }
    return null;
  };

  const effectiveRange =
    plan?.time_range ??
    activeRegion ??
    (activeSpectralBand ? { start_s: activeSpectralBand.startS, end_s: activeSpectralBand.endS } : null);
  const blockReason = computeBlockReason(plan);

  const executePlan = async (targetPlan: AudioPlan): Promise<void> => {
    if (applying.current || targetPlan.needs_clarification || currentSource.current !== sourceKey) return;
    if (computeBlockReason(targetPlan)) return;

    if (isImmediateControlOp(targetPlan.operation)) {
      executeImmediateControl(targetPlan);
      dispatch({ type: 'APPLY_STARTED' });
      return;
    }

    applying.current = true;
    dispatch({ type: 'START_APPLY' });
    let applyFailed: string | null = null;
    try {
      syncPlanToStudioRack(targetPlan);

      if (targetPlan.operation === 'spectral_repair' && targetPlan.spectral_repair) {
        const box = targetPlan.spectral_repair;
        getState().setSpectralSelection?.({
          startS: box.start_s,
          endS: box.end_s,
          fLowHz: box.f_low_hz,
          fHighHz: box.f_high_hz,
        });
        const repairFn = getAction('applySpectralBoxRepair');
        if (repairFn) {
          const res = await repairFn(box.attenuation_db);
          if (!res) applyFailed = getState().error || 'Spectral box repair could not run.';
        }
        return;
      }

      if (targetPlan.operation === 'preview') {
        const previewFn = getAction('previewRegion');
        if (previewFn) {
          await previewFn({
            autoPlay: true,
            ...(targetPlan.cleanup_profile ? { profile: targetPlan.cleanup_profile } : {}),
            ...(typeof targetPlan.strength === 'number' ? { strength: targetPlan.strength } : {}),
            ...(targetPlan.time_range
              ? { startS: targetPlan.time_range.start_s, endS: targetPlan.time_range.end_s }
              : activeRegion
                ? { startS: activeRegion.start_s, endS: activeRegion.end_s }
                : {}),
          });
        }
        return;
      }

      if (targetPlan.operation === 'export' && targetPlan.export_format) {
        const fmt: ExportFormatId = targetPlan.export_format;
        const exportFn = getAction('exportWithDelivery');
        const saveFn = getAction('saveCleanedMaster');
        if (exportFn) {
          await exportFn(fmt);
        } else if (saveFn) {
          await saveFn();
        }
        return;
      }

      if (targetPlan.operation === 'unit_override' && targetPlan.unit_override) {
        const overrideFn = getAction('applyUnitOverride');
        if (overrideFn) {
          const decision =
            targetPlan.unit_override.action === 'force_original'
              ? 'force_original'
              : 'force_enhanced';
          await overrideFn(targetPlan.unit_override.unit_id, decision);
        }
        return;
      }

      const planRange =
        targetPlan.time_range ??
        activeRegion ??
        (activeSpectralBand ? { start_s: activeSpectralBand.startS, end_s: activeSpectralBand.endS } : null);
      const strengthPatch =
        targetPlan.strength !== undefined && targetPlan.strength < 1.0
          ? { strength: targetPlan.strength }
          : {};
      const rangePatch = planRange
        ? { time_range: [planRange.start_s, planRange.end_s] as [number, number] }
        : {};
      const proRepairPatch = buildProRepairPatch(targetPlan);
      const studioPatch = {
        ...(typeof targetPlan.hpf_enabled === 'boolean' ? { hpf_enabled: targetPlan.hpf_enabled } : {}),
        ...(typeof targetPlan.hpf_cutoff_hz === 'number' ? { hpf_cutoff_hz: targetPlan.hpf_cutoff_hz } : {}),
        ...(proRepairPatch ? { pro_repair: proRepairPatch } : {}),
        ...(typeof targetPlan.ambience_floor_db === 'number'
          ? { ambience_floor_db: targetPlan.ambience_floor_db }
          : {}),
        ...(targetPlan.delivery_preset ? { delivery_preset: targetPlan.delivery_preset } : {}),
        ...(targetPlan.stereo_mode ? { stereo_mode: targetPlan.stereo_mode } : {}),
      };

      let strategy: ProcessingStrategyV1;
      if (targetPlan.operation === 'smart_safe') {
        strategy = {
          kind: 'smart_safe',
          allow_generative_reconstruction: false,
          ...strengthPatch,
          ...rangePatch,
          ...studioPatch,
        };
      } else if (targetPlan.operation === 'clean') {
        strategy = {
          kind: 'manual',
          route: targetPlan.cleanup_profile ?? 'studio',
          allow_generative_reconstruction: false,
          ...strengthPatch,
          ...rangePatch,
          ...studioPatch,
        };
      } else if (targetPlan.operation === 'restore') {
        strategy = {
          kind: 'manual',
          route: 'restore_source',
          allow_generative_reconstruction: true,
          ...strengthPatch,
          ...rangePatch,
          ...studioPatch,
        };
      } else if ((targetPlan.operation === 'extract' || targetPlan.operation === 'remove') && targetPlan.target) {
        strategy = {
          kind: 'separate',
          operation: targetPlan.operation,
          prompt: targetPlan.target,
          model_size: modelSize,
          quality: 'high',
          seed: 0,
          cleanup_profile: targetPlan.cleanup_profile,
          device: 'auto',
          ...strengthPatch,
          ...rangePatch,
          ...studioPatch,
        };
      } else {
        return;
      }

      const priorJob = getState().job?.id;
      await actions.startJob(strategy);
      const nextJobId = getState().job?.id ?? null;
      if (nextJobId === priorJob && currentSource.current === sourceKey) {
        applyFailed = getState().error || 'The edit did not start. Check the processing message below.';
      } else if (nextJobId) {
        lastChatJobIdRef.current = nextJobId;
      }
    } finally {
      applying.current = false;
      if (currentSource.current === sourceKey) {
        dispatch(applyFailed ? { type: 'APPLY_FAILED', message: applyFailed } : { type: 'APPLY_STARTED' });
      }
    }
  };

  const submitMessage = async (rawMessage: string): Promise<void> => {
    const message = rawMessage.trim();
    if (!message || planningRequest.current || !client || !engineReady || status?.available === false || running || submitting) return;
    planningRequest.current = true;
    const sourceAtSend = sourceKey;
    const id = ++sequence.current;
    const ac = new AbortController();
    request.current?.abort();
    request.current = ac;
    const history = messages.slice(-12);
    const acousticContext = buildAcousticContext(original, report, capabilities);
    const bandContext = activeSpectralBand
      ? ` [Band: ${formatBandRange(activeSpectralBand.fLowHz, activeSpectralBand.fHighHz)} · ${formatClock(activeSpectralBand.startS)}–${formatClock(activeSpectralBand.endS)}]`
      : '';
    const plannedMessage = bandContext ? `${message}${bandContext}`.slice(0, 2000) : message;
    setMessages([...history, { role: 'user', content: message }]);
    setDraft('');
    dispatch({ type: 'START_PLAN' });
    const timeout = window.setTimeout(() => ac.abort(new DOMException('The assistant took too long to answer.', 'TimeoutError')), 120000);
    let settled = false;
    try {
      const next = await client.planAudio(
        {
          message: plannedMessage,
          history,
          allow_cloud_text: false,
          ...(acousticContext ? { acoustic_context: acousticContext } : {}),
        },
        ac.signal,
      );
      if (ac.signal.aborted || sequence.current !== id || currentSource.current !== sourceAtSend) return;
      settled = true;
      setMessages([...history, { role: 'user', content: message }, { role: 'assistant', content: next.explanation }]);
      setModelSize(next.model_size);
      setLastProvider(next.provider);

      if (next.needs_clarification) {
        dispatch({ type: 'PLAN_CLARIFIED' });
        return;
      }
      if (isImmediateControlOp(next.operation)) {
        executeImmediateControl(next);
        dispatch({ type: 'PLAN_CLARIFIED' });
        return;
      }
      dispatch({ type: 'PLAN_READY', plan: next });
      // Auto-Run executes non-generative edits immediately when the user turns Auto-Run ON
      if (autoRunRef.current && next.operation !== 'restore') {
        await executePlan(next);
      }
    } catch (e) {
      if (sequence.current !== id || currentSource.current !== sourceAtSend) return;
      if (!ac.signal.aborted || ac.signal.reason?.name === 'TimeoutError') {
        settled = true;
        dispatch({
          type: 'PLAN_FAILED',
          message: e instanceof Error ? e.message : 'The assistant could not answer. Please try again.',
        });
      }
    } finally {
      window.clearTimeout(timeout);
      if (sequence.current === id) {
        if (!settled) dispatch({ type: 'PLAN_CANCELLED' });
        planningRequest.current = false;
      }
    }
  };

  const send = async (event: FormEvent): Promise<void> => {
    event.preventDefault();
    await submitMessage(draft);
  };

  useEffect(() => {
    if (!pendingChatPrompt) return;
    if (!client || !engineReady || status?.available === false || running || submitting || planningRequest.current) {
      return;
    }
    const promptToRun = pendingChatPrompt;
    setPendingChatPrompt(null);
    void submitMessage(promptToRun);
  }, [pendingChatPrompt, client, engineReady, status?.available, running, submitting]);

  useEffect(() => {
    if (!lastChatJobIdRef.current) return;
    if (!job || job.id !== lastChatJobIdRef.current) return;
    if (job.status?.state !== 'done' || !report) return;
    lastChatJobIdRef.current = null;
    const parts: string[] = [];
    const lufs = report.output?.integrated_lufs;
    if (typeof lufs === 'number' && Number.isFinite(lufs)) {
      parts.push(`Output ${lufs.toFixed(1)} LUFS`);
    }
    const peak = report.output?.true_peak_dbtp;
    if (typeof peak === 'number' && Number.isFinite(peak)) {
      parts.push(`Peak ${peak.toFixed(1)} dBTP`);
    }
    const enhanced = report.summary?.enhanced;
    const total = report.summary?.units_total;
    if (typeof enhanced === 'number' && typeof total === 'number' && total > 0) {
      parts.push(`${enhanced}/${total} units enhanced`);
    }
    const summary = parts.length > 0 ? `${parts.join(' · ')}.` : 'Processing complete.';
    setMessages((prev) => [
      ...prev,
      {
        role: 'assistant',
        content: `Done! ${summary} Say "Hear what was removed", "Dial it back", or "Export WAV".`,
      },
    ]);
  }, [job?.id, job?.status?.state, report]);

  const apply = async (): Promise<void> => {
    if (!plan) return;
    await executePlan(plan);
  };

  const toggleAutoRun = (): void => {
    const next = !autoRun;
    setAutoRun(next);
    writeAutoRunPref(next);
  };

  const hasReducedStrength = plan?.strength !== undefined && plan.strength < 1.0;
  const planChips = plan ? collectPlanChips(plan) : [];
  const provider = lastProvider || status?.provider;
  const providerLabel =
    status?.available === false
      ? 'Setup needed'
      : status?.active_mode === 'local_llm' || provider === 'ollama'
        ? status?.local_model
          ? `Local AI · ${status.local_model}`
          : 'Local AI'
        : provider === 'rules'
          ? 'Studio Commands'
          : provider === 'auto'
            ? 'Local first'
            : 'Audio assistant';

  const planHeadline = (p: AudioPlan): string => {
    if (p.operation === 'smart_safe') return 'Auto Smart Safe (benchmark & pick safest route)';
    if (p.operation === 'clean') return `Clean the recording (${(p.cleanup_profile ?? 'studio').replace('_', ' ')})`;
    if (p.operation === 'restore') return 'Restore missing high frequencies';
    if (p.operation === 'spectral_repair' && p.spectral_repair) {
      return `Spectral box repair (${formatBandRange(p.spectral_repair.f_low_hz, p.spectral_repair.f_high_hz)})`;
    }
    if (p.operation === 'preview') {
      return `Audition fast warm preview${p.cleanup_profile ? ` (${p.cleanup_profile.replace('_', ' ')})` : ''}`;
    }
    if (p.operation === 'export' && p.export_format) {
      return `Export cleaned master (${p.export_format.toUpperCase()})`;
    }
    if (p.operation === 'unit_override' && p.unit_override) {
      return `Override unit #${p.unit_override.unit_id} (${p.unit_override.action})`;
    }
    return `${p.operation === 'extract' ? 'Keep' : 'Remove'} ${p.target}${
      (p.operation === 'extract' || p.operation === 'remove') && p.cleanup_profile
        ? `, then clean the dialogue (${p.cleanup_profile.replace('_', ' ')})`
        : ''
    }`;
  };

  const followUpSuggestions: readonly string[] = cleanedPath
    ? isCkb
      ? ['بیستنی لابراوەکان', 'نەرمتری بکە', 'پاشەکەوتکردن WAV', 'ستۆدیۆی پرۆ']
      : ['Hear what was removed', 'Dial it back', 'Export WAV 24-bit', 'Switch to Pro Studio']
    : isCkb
      ? ['کێشەی ئەم دەنگە چییە؟', 'پاککردنەوەی زیرەکی خۆکار', 'دەنگدانەوە لاببە و پۆدکاست']
      : ["What's wrong with this audio?", 'Auto Smart Safe', 'Remove room echo, master for Podcast'];

  const clearConversation = (): void => {
    sequence.current += 1;
    request.current?.abort();
    setMessages([]);
    dispatch({ type: 'RESET' });
  };

  return (
    <section className="audio-assistant" aria-labelledby="audio-assistant-title">
      <div className="aa-heading">
        <h3 id="audio-assistant-title">
          {isCkb ? 'دەتەوێت چۆن دەنگەکە ببیستیت؟' : 'What would you like to hear?'}
        </h3>
        <div className="aa-heading-meta">
          {messages.length > 0 && (
            <button
              type="button"
              className="aa-clear-btn"
              disabled={planning || running}
              onClick={clearConversation}
            >
              {isCkb ? 'سڕینەوە' : 'Clear'}
            </button>
          )}
          <button
            type="button"
            className={`aa-autorun-btn${autoRun ? ' on' : ''}`}
            aria-pressed={autoRun}
            title="When enabled, chat commands run immediately without clicking Apply"
            onClick={toggleAutoRun}
          >
            {isCkb ? `جێبەجێی خۆکار: ${autoRun ? 'کارا' : 'ناکارا'}` : `Auto-Run: ${autoRun ? 'On' : 'Off'}`}
          </button>
          <span className="aa-provider">{providerLabel}</span>
        </div>
      </div>
      <p className="aa-intro">
        {isCkb
          ? 'هەموو شتێک بە چات کۆنترۆڵ بکە — پاککردنەوە، لابردنی دەنگدانەوە، کەمکردنی س، فلتەری خوار، بەرزی دەنگ، پشکنین، لێدان، یان هەناردەکردن.'
          : 'Control everything by chat — clean, dereverb, de-ess, HPF, loudness, spectral repair, preview, playback, undo, or export.'}
      </p>
      {messages.length > 0 && (
        <div ref={transcript} className="aa-messages" role="log" aria-label="Audio edit conversation" aria-live="polite" aria-relevant="additions">
          {messages.map((message, index) => (
            <div className={`aa-message aa-${message.role}`} key={`${index}-${message.role}`}>
              <span className="aa-speaker">
                {message.role === 'user' ? (isCkb ? 'تۆ' : 'You') : isCkb ? 'هەوا' : 'Hawa'}
              </span>
              <p>{message.content}</p>
            </div>
          ))}
        </div>
      )}
      {messages.length > 0 && !planning && !running && (
        <div className="aa-followups" role="group" aria-label="Suggested next chat actions">
          {followUpSuggestions.map((suggestion) => (
            <button
              type="button"
              className="aa-followup-chip"
              key={suggestion}
              disabled={!engineReady || status?.available === false || submitting}
              onClick={() => void submitMessage(suggestion)}
            >
              {suggestion}
            </button>
          ))}
        </div>
      )}
      {!messages.length && (
        <div className="aa-examples" aria-label="Example audio requests">
          {EXAMPLES.map((example) => (
            <button
              type="button"
              key={example}
              disabled={planning || running}
              onClick={() => {
                setDraft(example);
                dispatch({ type: 'DRAFT_CHANGED' });
              }}
            >
              {example}
            </button>
          ))}
        </div>
      )}
      {activeRegion && (
        <div className="aa-region-pill" role="group" aria-label="Selected waveform region">
          <span className="aa-region-label">
            {t.region}:{' '}
            <span className="mono">
              {formatClock(activeRegion.start_s)} – {formatClock(activeRegion.end_s)}
            </span>
          </span>
          <button
            type="button"
            className="aa-region-clear"
            aria-label={`Clear region ${formatClock(activeRegion.start_s)} to ${formatClock(activeRegion.end_s)}`}
            title="Clear selected region"
            disabled={running || submitting}
            onClick={() => setSelectionRange(null)}
          >
            ×
          </button>
        </div>
      )}
      {activeSpectralBand && (
        <div
          className="aa-region-pill aa-band-pill"
          role="group"
          aria-label="Selected spectrogram frequency band"
        >
          <span className="aa-region-label">
            Band:{' '}
            <span className="mono">
              {formatBandRange(activeSpectralBand.fLowHz, activeSpectralBand.fHighHz)} ·{' '}
              {formatClock(activeSpectralBand.startS)}–{formatClock(activeSpectralBand.endS)}
            </span>
          </span>
          <button
            type="button"
            className="aa-region-clear aa-band-clear"
            aria-label={`Clear spectral band ${formatBandRange(activeSpectralBand.fLowHz, activeSpectralBand.fHighHz)} ${formatClock(activeSpectralBand.startS)} to ${formatClock(activeSpectralBand.endS)}`}
            title="Clear selected frequency band"
            disabled={running || submitting}
            onClick={() => {
              setSpectralSelection(null);
              if (typeof window !== 'undefined') {
                window.dispatchEvent(
                  new CustomEvent(SPECTRAL_SELECTION_EVENT, { detail: null, bubbles: true }),
                );
              }
            }}
          >
            ×
          </button>
        </div>
      )}
      <form onSubmit={(event) => void send(event)}>
        <label className="sr-only" htmlFor="audio-assistant-message">Describe your audio edit</label>
        <textarea
          id="audio-assistant-message"
          rows={2}
          maxLength={2000}
          value={draft}
          placeholder="Ask anything: e.g. Remove echo, fix harsh S, HPF 90 Hz, master for Podcast…"
          disabled={planning || running || submitting}
          onChange={(event) => {
            setDraft(event.target.value);
            dispatch({ type: 'DRAFT_CHANGED' });
          }}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
              event.preventDefault();
              event.currentTarget.form?.requestSubmit();
              return;
            }
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault();
              if (event.nativeEvent.isComposing || event.keyCode === 229) {
                return;
              }
              event.currentTarget.form?.requestSubmit();
            }
          }}
        />
        <div className="aa-compose-actions">
          <span className="aa-privacy">Your audio stays on this device · Enter to send</span>
          {planning ? (
            <button className="aa-send" type="button" onClick={() => request.current?.abort()}>
              Stop
            </button>
          ) : (
            <button
              className="aa-send"
              type="submit"
              disabled={!draft.trim() || !engineReady || status?.available === false || running || submitting}
            >
              {autoRun ? 'Run' : 'Make a plan'}
            </button>
          )}
        </div>
      </form>
      {planning && <p className="aa-hint" role="status">Working out your edit…</p>}
      {status?.available === false && <p className="aa-hint">{status.reason || 'The local chat service needs to be set up.'}</p>}
      {status?.ollama_reachable && status.active_mode === 'deterministic_rules' && (
        <p className="aa-hint">
          All studio commands work offline now. For open-ended free-form chat, pull a local Ollama model (e.g. <code>ollama pull qwen2.5:7b</code>).
        </p>
      )}
      {provider === 'rules' && !status?.ollama_reachable && (
        <p className="aa-hint">Full studio commands are active offline. Add a local Ollama model for open-ended conversation.</p>
      )}
      {error && <p className="aa-error" role="alert">{error}</p>}
      {plan && (
        <div className="aa-plan" aria-label="Proposed audio edit">
          <span className="caps">Your edit</span>
          <p>{planHeadline(plan)}.</p>
          {planChips.length > 0 && (
            <div className="aa-plan-chips" aria-label="Planned studio parameters">
              {planChips.map((chip) => (
                <span className="aa-plan-chip mono" key={chip}>
                  {chip}
                </span>
              ))}
            </div>
          )}
          {(hasReducedStrength || effectiveRange) && (
            <p className="aa-hint aa-plan-params">
              {hasReducedStrength ? `Strength: ${Math.round((plan.strength ?? 1) * 100)}%` : ''}
              {hasReducedStrength && effectiveRange ? ' · ' : ''}
              {effectiveRange ? `Range: ${formatClock(effectiveRange.start_s)}–${formatClock(effectiveRange.end_s)}` : ''}
            </p>
          )}
          {isSeparation && (
            <label className="aa-quality">
              Quality
              <select
                aria-label="Separation quality"
                value={modelSize}
                disabled={running || submitting}
                onChange={(event) => setModelSize(event.target.value as 'small' | 'base' | 'large')}
              >
                <option value="large">Best detail · larger model</option>
                <option value="base">Balanced · medium model</option>
                <option value="small">Faster · smaller model</option>
              </select>
            </label>
          )}
          {isSeparation && <p className="aa-hint">Separation is experimental. Listen to the result before exporting.</p>}
          {isRestore && (
            <p className="aa-hint aa-restore-disclosure">
              Guarded bandwidth restoration (experimental) — observed speech band stays locked; listen before exporting.
            </p>
          )}
          {blockReason && <p className="aa-hint" id="aa-apply-reason">{blockReason}</p>}
          <button
            type="button"
            className="aa-apply"
            disabled={!!blockReason}
            aria-describedby={blockReason ? 'aa-apply-reason' : undefined}
            onClick={() => void apply()}
          >
            {submitting ? 'Starting…' : 'Apply to recording'}
          </button>
          <p className="aa-hint">Listen with Original / Cleaned below, then save the result.</p>
        </div>
      )}
    </section>
  );
}
