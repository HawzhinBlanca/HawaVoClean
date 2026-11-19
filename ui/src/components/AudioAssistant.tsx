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
  HawaVoCleanReport,
  ProcessingStrategyV1,
} from '../api/types';
import { SPECTRAL_SELECTION_EVENT } from '../render/stftSpectrogram';
import { startJob } from '../state/actions';
import { useI18n } from '../state/i18n';
import {
  assistantReducer,
  deriveWorkspacePhase,
  INITIAL_ASSISTANT_STATE,
} from '../state/machines';
import { getState, isProfileUnavailable, useStore } from '../state/store';
import '../styles/assistant.css';

const EXAMPLES = ['Clean the voice and reduce room noise', 'Keep the speech, remove the music', 'Remove the dog barking'];
const CLEANUP_PROFILES: readonly CleanupProfile[] = ['production', 'studio', 'studio_pro', 'dereverb', 'lowband'];
const BANDWIDTH_SHAPES: readonly BandwidthShape[] = ['unknown', 'silence', 'fullband', 'steep_lowpass'];
const MAX_SEPARATION_DURATION_S = 3600;

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
  return Object.keys(ctx).length > 0 ? ctx : undefined;
}

/** The chat proposes an edit; only the separate Apply action submits audio. */
export function AudioAssistant() {
  const { t } = useI18n();
  const source = useStore((s) => s.source);
  const original = useStore((s) => s.original);
  const report = useStore((s) => s.report);
  const client = useStore((s) => s.client);
  const engineReady = useStore((s) => s.engineStatus === 'ready');
  const workspacePhase = useStore(deriveWorkspacePhase);
  const analyzing = workspacePhase === 'uploading' || workspacePhase === 'analyzing';
  const running = workspacePhase === 'processing';
  const capabilities = useStore((s) => s.capabilities);
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
  const [messages, setMessages] = useState<AudioChatMessage[]>([]);
  const [draft, setDraft] = useState('');
  const [machine, dispatch] = useReducer(assistantReducer, INITIAL_ASSISTANT_STATE);
  const [status, setStatus] = useState<AudioAssistantStatus | null>(null);
  const [lastProvider, setLastProvider] = useState<AudioPlan['provider'] | null>(null);
  const [modelSize, setModelSize] = useState<'small' | 'base' | 'large'>('large');
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
  const cleanupProfile: CleanupProfile = plan?.cleanup_profile ?? 'studio';
  const cleanupCapability = capabilities?.find((c) => c.capability_id === cleanupProfile);
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
  const effectiveRange =
    plan?.time_range ??
    activeRegion ??
    (activeSpectralBand ? { start_s: activeSpectralBand.startS, end_s: activeSpectralBand.endS } : null);
  const blockReason = !source || !original
    ? 'Load a recording before applying an edit.'
    : !engineReady ? 'The audio engine is offline.'
      : analyzing ? 'Wait for the recording to finish loading.'
        : running || submitting ? 'An audio edit is already in progress.'
          : isSeparation && (!separationCapability?.available || separationCapability.maturity === 'blocked')
            ? separationCapability?.reason || 'Sound separation is not installed yet. Voice cleanup is still available.'
            : isSeparation && (!Number.isFinite(original.duration_s) || original.duration_s > MAX_SEPARATION_DURATION_S || original.channels > 2)
              ? 'Sound separation currently supports recordings up to 60 minutes in mono or stereo. Open a shorter clip to try it.'
              : (plan?.operation === 'clean' || isSeparation) && plan?.cleanup_profile && isProfileUnavailable(capabilities, plan.cleanup_profile)
                ? cleanupCapability?.reason ||
                  (plan.cleanup_profile === 'studio_pro'
                    ? 'Studio Pro is unavailable on this machine.'
                    : plan.cleanup_profile === 'dereverb'
                      ? 'Dereverb is unavailable on this machine.'
                      : 'This cleanup option is unavailable.')
                : plan?.operation === 'clean' && cleanupCapability && (!cleanupCapability.available || cleanupCapability.maturity === 'blocked')
                  ? cleanupCapability.reason || 'This cleanup option is unavailable.'
                  : plan?.operation === 'restore' && (!restoreCapability?.available || restoreCapability.maturity === 'blocked')
                    ? restoreCapability?.reason || 'Bandwidth restoration is currently unavailable.'
                    : null;

  const send = async (event: FormEvent): Promise<void> => {
    event.preventDefault();
    const message = draft.trim();
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
      dispatch(next.needs_clarification ? { type: 'PLAN_CLARIFIED' } : { type: 'PLAN_READY', plan: next });
      setModelSize(next.model_size);
      setLastProvider(next.provider);
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

  const apply = async (): Promise<void> => {
    if (!plan || blockReason || applying.current || plan.needs_clarification || currentSource.current !== sourceKey) return;
    const strengthPatch = plan.strength !== undefined && plan.strength < 1.0 ? { strength: plan.strength } : {};
    const rangePatch = effectiveRange
      ? { time_range: [effectiveRange.start_s, effectiveRange.end_s] as [number, number] }
      : {};
    let strategy: ProcessingStrategyV1;
    if (plan.operation === 'clean') {
      strategy = {
        kind: 'manual',
        route: cleanupProfile,
        allow_generative_reconstruction: false,
        ...strengthPatch,
        ...rangePatch,
      };
    } else if (plan.operation === 'restore') {
      strategy = {
        kind: 'manual',
        route: 'restore_source',
        allow_generative_reconstruction: true,
        ...strengthPatch,
        ...rangePatch,
      };
    } else if ((plan.operation === 'extract' || plan.operation === 'remove') && plan.target) {
      strategy = {
        kind: 'separate',
        operation: plan.operation,
        prompt: plan.target,
        model_size: modelSize,
        quality: 'high',
        seed: 0,
        cleanup_profile: plan.cleanup_profile,
        device: 'auto',
        ...strengthPatch,
        ...rangePatch,
      };
    } else return;
    applying.current = true;
    dispatch({ type: 'START_APPLY' });
    const priorJob = getState().job?.id;
    let applyFailed: string | null = null;
    try {
      await startJob(strategy);
      if (getState().job?.id === priorJob && currentSource.current === sourceKey) {
        applyFailed = getState().error || 'The edit did not start. Check the processing message below.';
      }
    } finally {
      applying.current = false;
      if (currentSource.current === sourceKey) {
        dispatch(applyFailed ? { type: 'APPLY_FAILED', message: applyFailed } : { type: 'APPLY_STARTED' });
      }
    }
  };

  const hasReducedStrength = plan?.strength !== undefined && plan.strength < 1.0;
  const provider = lastProvider || status?.provider;
  const providerLabel = status?.available === false ? 'Setup needed' : provider === 'ollama' ? 'Local AI' : provider === 'rules' ? 'Basic commands' : provider === 'auto' ? 'Local first' : 'Audio assistant';
  return (
    <section className="audio-assistant" aria-labelledby="audio-assistant-title">
      <div className="aa-heading">
        <h3 id="audio-assistant-title">What would you like to hear?</h3>
        <span className="aa-provider">{providerLabel}</span>
      </div>
      <p className="aa-intro">Describe what to clean, keep, or remove.</p>
      {messages.length > 0 && (
        <div ref={transcript} className="aa-messages" role="log" aria-label="Audio edit conversation" aria-live="polite" aria-relevant="additions">
          {messages.map((message, index) => (
            <div className={`aa-message aa-${message.role}`} key={`${index}-${message.role}`}>
              <span className="aa-speaker">{message.role === 'user' ? 'You' : 'Hawa'}</span>
              <p>{message.content}</p>
            </div>
          ))}
        </div>
      )}
      {!messages.length && <div className="aa-examples" aria-label="Example audio requests">
        {EXAMPLES.map((example) => <button type="button" key={example} disabled={planning || running} onClick={() => { setDraft(example); dispatch({ type: 'DRAFT_CHANGED' }); }}>{example}</button>)}
      </div>}
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
          id="audio-assistant-message" rows={2} maxLength={2000} value={draft}
          placeholder="Keep the speech and remove background music…"
          disabled={planning || running || submitting}
          onChange={(event) => { setDraft(event.target.value); dispatch({ type: 'DRAFT_CHANGED' }); }}
          onKeyDown={(event) => { if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) { event.preventDefault(); event.currentTarget.form?.requestSubmit(); } }}
        />
        <div className="aa-compose-actions">
          <span className="aa-privacy">Your audio stays on this device.</span>
          {planning ? <button className="aa-send" type="button" onClick={() => request.current?.abort()}>Stop</button>
            : <button className="aa-send" type="submit" disabled={!draft.trim() || !engineReady || status?.available === false || running || submitting}>Make a plan</button>}
        </div>
      </form>
      {planning && <p className="aa-hint" role="status">Working out your edit…</p>}
      {status?.available === false && <p className="aa-hint">{status.reason || 'The local chat service needs to be set up.'}</p>}
      {provider === 'rules' && <p className="aa-hint">Basic commands are available. A local AI model enables more flexible conversation.</p>}
      {error && <p className="aa-error" role="alert">{error}</p>}
      {plan && (
        <div className="aa-plan" aria-label="Proposed audio edit">
          <span className="caps">Your edit</span>
          <p>
            {plan.operation === 'clean'
              ? `Clean the recording (${cleanupProfile.replace('_', ' ')})`
              : plan.operation === 'restore'
                ? 'Restore missing high frequencies'
                : `${plan.operation === 'extract' ? 'Keep' : 'Remove'} ${plan.target}`}
            {isSeparation && plan.cleanup_profile ? `, then clean the dialogue (${plan.cleanup_profile.replace('_', ' ')})` : ''}.
          </p>
          {(hasReducedStrength || effectiveRange) && (
            <p className="aa-hint aa-plan-params">
              {hasReducedStrength ? `Strength: ${Math.round((plan.strength ?? 1) * 100)}%` : ''}
              {hasReducedStrength && effectiveRange ? ' · ' : ''}
              {effectiveRange ? `Range: ${formatClock(effectiveRange.start_s)}–${formatClock(effectiveRange.end_s)}` : ''}
            </p>
          )}
          {isSeparation && <label className="aa-quality">Quality
            <select aria-label="Separation quality" value={modelSize} disabled={running || submitting} onChange={(event) => setModelSize(event.target.value as 'small' | 'base' | 'large')}>
              <option value="large">Best detail · larger model</option>
              <option value="base">Balanced · medium model</option>
              <option value="small">Faster · smaller model</option>
            </select>
          </label>}
          {isSeparation && <p className="aa-hint">Separation is experimental. Listen to the result before exporting.</p>}
          {isRestore && (
            <p className="aa-hint aa-restore-disclosure">
              Guarded bandwidth restoration (experimental) — observed speech band stays locked; listen before exporting.
            </p>
          )}
          {blockReason && <p className="aa-hint" id="aa-apply-reason">{blockReason}</p>}
          <button type="button" className="aa-apply" disabled={!!blockReason} aria-describedby={blockReason ? 'aa-apply-reason' : undefined} onClick={() => void apply()}>{submitting ? 'Starting…' : 'Apply to recording'}</button>
          <p className="aa-hint">Listen with Original / Cleaned below, then save the result.</p>
        </div>
      )}
    </section>
  );
}
