import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import axe from 'axe-core';
import type { AudioAnalysis, AudioPlan } from '../api/types';
import type { EngineClient } from '../api/client';
import { startJob } from '../state/actions';
import { getState, useStore } from '../state/store';
import { AudioAssistant } from './AudioAssistant';

vi.mock('../state/actions', () => ({ startJob: vi.fn(async () => undefined) }));
globalThis.IS_REACT_ACT_ENVIRONMENT = true;
const pristine = useStore.getState();
let host: HTMLElement;
let root: Root;
const cleanPlan: AudioPlan = {
  operation: 'clean', target: null, cleanup_profile: 'studio', model_size: 'large',
  quality: 'high', explanation: 'I will clean the voice and reduce the room noise.',
  provider: 'ollama', needs_clarification: false,
};
const separatePlan: AudioPlan = {
  ...cleanPlan, operation: 'extract', target: 'speech',
  explanation: 'I will keep the speech and clean the isolated dialogue.',
};
const planAudio = vi.fn();
const assistantStatus = vi.fn();

function armed(seconds = 30): void {
  useStore.setState({
    engineStatus: 'ready',
    client: { planAudio, assistantStatus } as unknown as EngineClient,
    source: { path: '/clip.wav', name: 'clip.wav', sourceId: 'abc', origin: 'upload' },
    original: {
      path: '/clip.wav',
      duration_s: seconds,
      sample_rate: 48000,
      channels: 2,
      noise_floor_db: -48.5,
      loudness: { integrated_lufs: -20.2, true_peak_dbtp: -1.5 },
    } as AudioAnalysis,
    capabilities: [
      { capability_id: 'production', available: true, maturity: 'qualified' },
      { capability_id: 'studio', available: true, maturity: 'qualified' },
      { capability_id: 'studio_pro', available: true, maturity: 'qualified' },
      { capability_id: 'separate', available: true, maturity: 'experimental' },
    ],
  });
}

beforeEach(() => {
  useStore.setState(pristine, true);
  vi.clearAllMocks();
  assistantStatus.mockResolvedValue({ provider: 'auto', cloud_text_available: false });
  planAudio.mockResolvedValue(cleanPlan);
  vi.mocked(startJob).mockImplementation(async () => {
    getState().setJob({ id: 'new-job', outputPath: '/out.wav', reportPath: '/out.json', status: null, streamConnected: false });
  });
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
});

async function render(): Promise<void> {
  await act(async () => root.render(createElement(AudioAssistant)));
}
async function useExample(index = 0): Promise<void> {
  await act(async () => host.querySelectorAll<HTMLButtonElement>('.aa-examples button')[index]?.click());
}
async function send(): Promise<void> {
  await act(async () => host.querySelector('form')?.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true })));
}
async function type(message: string): Promise<void> {
  const textarea = host.querySelector('textarea')!;
  const setter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(textarea), 'value')?.set;
  await act(async () => {
    setter?.call(textarea, message);
    textarea.dispatchEvent(new Event('input', { bubbles: true }));
  });
}

describe('AudioAssistant', () => {
  it('makes a plan with compact acoustic telemetry without paths, then applies the reviewed cleanup', async () => {
    armed();
    await render();
    await useExample();
    await send();
    expect(planAudio).toHaveBeenCalledWith(
      {
        message: 'Clean the voice and reduce room noise',
        history: [],
        allow_cloud_text: false,
        acoustic_context: {
          duration_s: 30,
          sample_rate: 48000,
          channels: 2,
          noise_floor_db: -48.5,
          integrated_lufs: -20.2,
          available_profiles: ['production', 'studio', 'studio_pro'],
          separation_available: true,
        },
      },
      expect.any(AbortSignal),
    );
    expect(JSON.stringify(planAudio.mock.calls[0]?.[0])).not.toContain('/clip.wav');
    expect(startJob).not.toHaveBeenCalled();
    expect(host.textContent).toContain(cleanPlan.explanation);
    await act(async () => host.querySelector<HTMLButtonElement>('.aa-apply')?.click());
    expect(startJob).toHaveBeenCalledWith({ kind: 'manual', route: 'studio', allow_generative_reconstruction: false });
  });

  it('sends an explicit best-detail separation operation with device auto using the existing job flow', async () => {
    armed();
    planAudio.mockResolvedValue(separatePlan);
    await render();
    await useExample(1);
    await send();
    await act(async () => host.querySelector<HTMLButtonElement>('.aa-apply')?.click());
    expect(startJob).toHaveBeenCalledWith({ kind: 'separate', operation: 'extract', prompt: 'speech', cleanup_profile: 'studio', model_size: 'large', quality: 'high', seed: 0, device: 'auto' });
  });

  it('displays strength and time_range in the edit card and passes them through startJob', async () => {
    armed(120);
    planAudio.mockResolvedValue({
      ...separatePlan,
      operation: 'remove',
      target: 'siren',
      cleanup_profile: null,
      strength: 0.6,
      time_range: { start_s: 15, end_s: 28 },
      explanation: 'Remove the siren between 0:15 and 0:28 at 60% strength.',
    });
    await render();
    await type('Remove the siren from 0:15 to 0:28 at 60% strength');
    await send();
    expect(host.textContent).toContain('Strength: 60%');
    expect(host.textContent).toContain('Range: 0:15–0:28');
    await act(async () => host.querySelector<HTMLButtonElement>('.aa-apply')?.click());
    expect(startJob).toHaveBeenCalledWith({
      kind: 'separate',
      operation: 'remove',
      prompt: 'siren',
      cleanup_profile: null,
      model_size: 'large',
      quality: 'high',
      seed: 0,
      device: 'auto',
      strength: 0.6,
      time_range: [15, 28],
    });
  });

  it('supports studio_pro cleanup with reduced strength and time_range', async () => {
    armed(120);
    planAudio.mockResolvedValue({
      ...cleanPlan,
      cleanup_profile: 'studio_pro',
      strength: 0.8,
      time_range: { start_s: 5, end_s: 20 },
      explanation: 'Clean dialogue with studio_pro keeping 20% room tone.',
    });
    await render();
    await type('Clean from 0:05 to 0:20 and keep 20% natural room tone');
    await send();
    expect(host.textContent).toContain('studio pro');
    expect(host.textContent).toContain('Strength: 80%');
    expect(host.textContent).toContain('Range: 0:05–0:20');
    await act(async () => host.querySelector<HTMLButtonElement>('.aa-apply')?.click());
    expect(startJob).toHaveBeenCalledWith({
      kind: 'manual',
      route: 'studio_pro',
      allow_generative_reconstruction: false,
      strength: 0.8,
      time_range: [5, 20],
    });
  });

  it('keeps missing models visible and refuses to apply an unavailable separation', async () => {
    armed();
    useStore.setState({ capabilities: [{ capability_id: 'separate', available: false, maturity: 'blocked', reason: 'Install the optional sound-separation model.' }] });
    planAudio.mockResolvedValue(separatePlan);
    await render();
    await useExample(1);
    await send();
    expect(host.textContent).toContain('Install the optional sound-separation model.');
    expect(host.querySelector<HTMLButtonElement>('.aa-apply')?.disabled).toBe(true);
    expect(startJob).not.toHaveBeenCalled();
  });

  it('allows separation up to 60 minutes and blocks clips over 60 minutes without blocking cleanup', async () => {
    armed(1800);
    planAudio.mockResolvedValue(separatePlan);
    await render();
    await useExample(1);
    await send();
    expect(host.querySelector<HTMLButtonElement>('.aa-apply')?.disabled).toBe(false);

    await act(async () => armed(3601));
    await type('Keep the speech, remove the music');
    await send();
    expect(host.querySelector<HTMLButtonElement>('.aa-apply')?.disabled).toBe(true);
    expect(host.textContent).toContain('up to 60 minutes');
    planAudio.mockResolvedValue(cleanPlan);
    await type('Clean the voice instead');
    await send();
    expect(host.querySelector<HTMLButtonElement>('.aa-apply')?.disabled).toBe(false);
  });

  it('asks a follow-up without offering an executable plan', async () => {
    armed();
    planAudio.mockResolvedValue({ ...cleanPlan, operation: 'clarify', needs_clarification: true, explanation: 'Which sound would you like to keep?' });
    await render();
    await useExample();
    await send();
    expect(host.textContent).toContain('Which sound would you like to keep?');
    expect(host.querySelector('.aa-apply')).toBeNull();
    await type('Keep just the speech');
    planAudio.mockResolvedValue(separatePlan);
    await send();
    expect(planAudio.mock.calls[1]?.[0].history).toEqual([
      { role: 'user', content: 'Clean the voice and reduce room noise' },
      { role: 'assistant', content: 'Which sound would you like to keep?' },
    ]);
  });

  it('retires a plan when the instruction changes', async () => {
    armed();
    await render();
    await useExample();
    await send();
    expect(host.querySelector('.aa-apply')).not.toBeNull();
    await type('Actually keep the music');
    expect(host.querySelector('.aa-apply')).toBeNull();
  });

  it('aborts and discards a late answer after the source changes', async () => {
    armed();
    let resolve!: (p: AudioPlan) => void;
    planAudio.mockImplementationOnce(() => new Promise<AudioPlan>((r) => { resolve = r; }));
    await render();
    await useExample();
    await send();
    const signal = planAudio.mock.calls[0]?.[1] as AbortSignal;
    await act(async () => getState().setSource({ path: '/new.wav', name: 'new.wav', sourceId: 'def', origin: 'upload' }));
    expect(signal.aborted).toBe(true);
    await act(async () => resolve(cleanPlan));
    expect(host.querySelector('.aa-apply')).toBeNull();
    expect(host.textContent).not.toContain(cleanPlan.explanation);
  });

  it('does not send a duplicate plan request while an answer is pending', async () => {
    armed();
    planAudio.mockImplementationOnce(() => new Promise(() => undefined));
    await render();
    await useExample();
    await act(async () => {
      host.querySelector('form')?.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
      host.querySelector('form')?.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    });
    expect(planAudio).toHaveBeenCalledTimes(1);
  });

  it('keeps cloud disabled even if an engine advertises a cloud option', async () => {
    armed();
    assistantStatus.mockResolvedValue({ provider: 'auto', cloud_text_available: true });
    await render();
    expect(host.querySelector('input[type="checkbox"]')).toBeNull();
    await useExample();
    await send();
    expect(planAudio.mock.calls[0]?.[0].allow_cloud_text).toBe(false);
    await type('Remove the music instead');
    await send();
    expect(planAudio.mock.calls[1]?.[0].allow_cloud_text).toBe(false);
  });

  it('wires live restore operation with guarded-restoration disclosure, strength, and time_range', async () => {
    armed(120);
    useStore.setState({
      capabilities: [
        { capability_id: 'production', available: true, maturity: 'qualified' },
        { capability_id: 'studio', available: true, maturity: 'qualified' },
        { capability_id: 'studio_pro', available: true, maturity: 'qualified', providers: ['mps'] },
        { capability_id: 'restore_source', available: true, maturity: 'experimental', providers: ['mps'] },
        { capability_id: 'separate', available: true, maturity: 'experimental' },
      ],
    });
    planAudio.mockResolvedValue({
      operation: 'restore',
      target: null,
      cleanup_profile: null,
      model_size: 'large',
      quality: 'high',
      strength: 0.75,
      time_range: { start_s: 10, end_s: 25 },
      explanation: 'Restore missing high frequencies between 0:10 and 0:25.',
      provider: 'ollama',
      needs_clarification: false,
    });
    await render();
    await type('Restore missing high frequencies from 0:10 to 0:25');
    await send();
    expect(host.textContent).toContain('Guarded bandwidth restoration (experimental)');
    expect(host.textContent).toContain('observed speech band stays locked');
    expect(host.querySelector<HTMLButtonElement>('.aa-apply')?.disabled).toBe(false);
    // AcousticContext strictly forbids unknown keys (extra="forbid")
    const sentCtx = planAudio.mock.calls[0]?.[0].acoustic_context ?? {};
    expect('restore_available' in sentCtx).toBe(false);
    expect(sentCtx.available_profiles).not.toContain('restore_source');

    await act(async () => host.querySelector<HTMLButtonElement>('.aa-apply')?.click());
    expect(startJob).toHaveBeenCalledWith({
      kind: 'manual',
      route: 'restore_source',
      allow_generative_reconstruction: true,
      strength: 0.75,
      time_range: [10, 25],
    });
  });

  it('blocks restore operation when restore_source capability is unavailable or blocked', async () => {
    armed(60);
    useStore.setState({
      capabilities: [
        {
          capability_id: 'restore_source',
          available: false,
          maturity: 'blocked',
          reason: 'Install LavaSR v2 weights to enable guarded restoration.',
        },
      ],
    });
    planAudio.mockResolvedValue({
      operation: 'restore',
      target: null,
      cleanup_profile: null,
      model_size: 'large',
      quality: 'high',
      explanation: 'Restore missing high frequencies.',
      provider: 'rules',
      needs_clarification: false,
    });
    await render();
    await type('Restore missing high frequencies');
    await send();
    expect(host.textContent).toContain('Install LavaSR v2 weights to enable guarded restoration.');
    expect(host.querySelector<HTMLButtonElement>('.aa-apply')?.disabled).toBe(true);
    await act(async () => host.querySelector<HTMLButtonElement>('.aa-apply')?.click());
    expect(startJob).not.toHaveBeenCalled();
  });

  it('shows a removable region pill for waveform selectionRange and auto-attaches time_range on apply', async () => {
    armed(120);
    useStore.setState({ selectionRange: { start: 15, end: 28 } });
    planAudio.mockResolvedValue({
      ...separatePlan,
      operation: 'remove',
      target: 'cough',
      cleanup_profile: null,
      time_range: null,
      explanation: 'Remove the cough from the selected region.',
    });
    await render();

    const pill = host.querySelector('.aa-region-pill');
    expect(pill?.textContent).toContain('Region:');
    expect(pill?.textContent).toContain('0:15 – 0:28');

    await type('Remove cough');
    await send();
    expect(host.textContent).toContain('Range: 0:15–0:28');

    await act(async () => host.querySelector<HTMLButtonElement>('.aa-apply')?.click());
    expect(startJob).toHaveBeenCalledWith({
      kind: 'separate',
      operation: 'remove',
      prompt: 'cough',
      cleanup_profile: null,
      model_size: 'large',
      quality: 'high',
      seed: 0,
      device: 'auto',
      time_range: [15, 28],
    });

    // Clearing the pill in 1 click removes selectionRange and falls back to whole-clip apply.
    vi.mocked(startJob).mockClear();
    await act(async () => {
      useStore.setState({ job: null });
    });
    await act(async () => host.querySelector<HTMLButtonElement>('.aa-region-clear')?.click());
    expect(getState().selectionRange).toBeNull();
    expect(host.querySelector('.aa-region-pill')).toBeNull();
    await act(async () => host.querySelector<HTMLButtonElement>('.aa-apply')?.click());
    expect(startJob).toHaveBeenCalledWith({
      kind: 'separate',
      operation: 'remove',
      prompt: 'cough',
      cleanup_profile: null,
      model_size: 'large',
      quality: 'high',
      seed: 0,
      device: 'auto',
    });
  });

  it('lets an explicit prompt time_range take precedence over waveform selectionRange', async () => {
    armed(120);
    useStore.setState({ selectionRange: { start: 15, end: 28 } });
    planAudio.mockResolvedValue({
      ...cleanPlan,
      cleanup_profile: 'studio_pro',
      time_range: { start_s: 2, end_s: 9 },
      explanation: 'Clean dialogue between 0:02 and 0:09.',
    });
    await render();
    await type('Clean from 0:02 to 0:09');
    await send();
    await act(async () => host.querySelector<HTMLButtonElement>('.aa-apply')?.click());
    expect(startJob).toHaveBeenCalledWith({
      kind: 'manual',
      route: 'studio_pro',
      allow_generative_reconstruction: false,
      time_range: [2, 9],
    });
  });

  it('fails closed on studio_pro clean and separation plans when capabilities omits studio_pro', async () => {
    armed(60);
    useStore.setState({
      capabilities: [
        { capability_id: 'production', available: true, maturity: 'qualified' },
        { capability_id: 'studio', available: true, maturity: 'qualified' },
        { capability_id: 'separate', available: true, maturity: 'experimental' },
      ],
    });
    planAudio.mockResolvedValue({
      ...cleanPlan,
      cleanup_profile: 'studio_pro',
      explanation: 'Clean dialogue with studio_pro.',
    });
    await render();
    await type('Clean with studio pro');
    await send();
    expect(host.querySelector<HTMLButtonElement>('.aa-apply')?.disabled).toBe(true);
    expect(host.textContent).toContain('Studio Pro is unavailable on this machine.');

    planAudio.mockResolvedValue({
      ...separatePlan,
      cleanup_profile: 'studio_pro',
      explanation: 'Keep speech and clean with studio_pro.',
    });
    await type('Keep speech and clean with studio pro');
    await send();
    expect(host.querySelector<HTMLButtonElement>('.aa-apply')?.disabled).toBe(true);
    expect(host.textContent).toContain('Studio Pro is unavailable on this machine.');
    expect(startJob).not.toHaveBeenCalled();
  });

  it('advertises dereverb in acoustic_context when available, applies it cleanly, and fails closed when absent', async () => {
    armed(90);
    useStore.setState({
      capabilities: [
        { capability_id: 'production', available: true, maturity: 'qualified' },
        { capability_id: 'studio', available: true, maturity: 'qualified' },
        { capability_id: 'studio_pro', available: true, maturity: 'qualified', providers: ['mps'] },
        { capability_id: 'dereverb', available: true, maturity: 'qualified', providers: ['mps'] },
        { capability_id: 'separate', available: true, maturity: 'experimental' },
      ],
    });
    planAudio.mockResolvedValue({
      ...cleanPlan,
      cleanup_profile: 'dereverb',
      explanation: 'Remove room echo using neural dereverb.',
    });
    await render();
    await type('Remove room echo and reverb');
    await send();

    expect(planAudio.mock.calls[0]?.[0].acoustic_context?.available_profiles).toEqual([
      'production',
      'studio',
      'studio_pro',
      'dereverb',
    ]);
    expect(host.querySelector<HTMLButtonElement>('.aa-apply')?.disabled).toBe(false);
    await act(async () => host.querySelector<HTMLButtonElement>('.aa-apply')?.click());
    expect(startJob).toHaveBeenCalledWith({
      kind: 'manual',
      route: 'dereverb',
      allow_generative_reconstruction: false,
    });

    vi.mocked(startJob).mockClear();
    await act(async () => {
      useStore.setState({
        job: null,
        capabilities: [
          { capability_id: 'production', available: true, maturity: 'qualified' },
          { capability_id: 'studio', available: true, maturity: 'qualified' },
        ],
      });
    });
    expect(host.querySelector<HTMLButtonElement>('.aa-apply')?.disabled).toBe(true);
    expect(host.textContent).toContain('Dereverb is unavailable on this machine.');
    await act(async () => host.querySelector<HTMLButtonElement>('.aa-apply')?.click());
    expect(startJob).not.toHaveBeenCalled();
  });

  it('shows a dismissible spectrogram band chip, attaches band context to planAudio without breaking acoustic_context, and passes axe-core', async () => {
    armed(60);
    useStore.setState({
      spectralSelection: { startS: 4.2, endS: 6.1, fLowHz: 140, fHighHz: 220 },
    });
    await render();

    const bandChip = host.querySelector('.aa-band-pill');
    expect(bandChip).not.toBeNull();
    expect(bandChip?.textContent).toContain('Band:');
    expect(bandChip?.textContent).toContain('140–220 Hz');
    expect(bandChip?.textContent).toContain('0:04.2–0:06.1');

    await type('Attenuate this hum');
    await send();

    const sentPayload = planAudio.mock.calls[0]?.[0];
    expect(sentPayload.message).toContain('Attenuate this hum');
    expect(sentPayload.message).toContain('140–220 Hz');
    expect(sentPayload.message).toContain('0:04.2–0:06.1');
    // AcousticContext strictly forbids unknown keys
    expect('spectralSelection' in (sentPayload.acoustic_context ?? {})).toBe(false);

    const axeResults = await axe.run(host, {
      rules: { 'color-contrast': { enabled: false }, region: { enabled: false } },
    });
    expect(axeResults.violations).toEqual([]);

    // Dismissing the chip clears spectralSelection
    await act(async () => host.querySelector<HTMLButtonElement>('.aa-band-clear')?.click());
    expect(getState().spectralSelection).toBeNull();
    expect(host.querySelector('.aa-band-pill')).toBeNull();
  });
});
