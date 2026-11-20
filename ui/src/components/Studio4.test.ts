import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import axe from 'axe-core';

vi.mock('../render/WaveformHost', () => ({
  WaveformHost: class {
    refreshTheme() {}
    setData() {}
    clear() {}
    setView() {}
    setPlayhead() {}
    setHover() {}
    setHighlight() {}
    setUnits() {}
    setFocus() {}
    requestStats() {}
    get width() {
      return 640;
    }
    dispose() {}
  },
}));

vi.mock('../render/spectrum', () => ({
  SPECTRUM_PLOT_INSETS: Object.freeze({ left: 33, right: 9, top: 11, bottom: 17 }),
  SpectrumRenderer: class {
    setCurve() {}
    setFocus() {}
    setLive() {}
    removedCoverage() {
      return 0;
    }
    dispose() {}
  },
}));
import { EngineClient } from '../api/client';
import type { AudioAnalysis, HawaVoCleanReport } from '../api/types';
import { DualPlayer, getPlayer } from '../audio/player';
import { usePlayerSync } from '../App';
import {
  applySpectralBoxRepair,
  buildStrategyFromState,
  previewRegion,
  redoAction,
  startJob,
  submitBatch,
  undoAction,
  updateRackSettings,
} from '../state/actions';
import { initLocaleFromStorage, setLocale } from '../state/i18n';
import { handleKeyDown } from '../state/keymap';
import { cancelScheduledLivePreview, scheduleLivePreview } from '../state/liveAudition';
import { DEFAULT_STUDIO_RACK_SETTINGS } from '../state/presets';
import { useStore, getState } from '../state/store';
import { AudioHealthCard } from './AudioHealthCard';
import { AudioStage } from './AudioStage';
import { encodeWavPcm16, MicTestButton } from './controls/MicTestButton';
import { RotaryKnob } from './controls/RotaryKnob';
import { PillToggle } from './controls/PillToggle';
import { Header } from './Header';
import { LiveMeter } from './LiveMeter';
import { ProWorkspace } from './ProWorkspace';
import { ShortcutOverlay } from './ShortcutOverlay';
import { SimpleWorkspace } from './SimpleWorkspace';
import { SlideOverDrawer } from './SlideOverDrawer';
import { SourceStrip } from './SourceStrip';
import { hzToMusicalNote, SpectrumDisplay } from './SpectrumDisplay';
import { ToastStack } from './ToastStack';
import { Transport } from './Transport';
import { WaveformDisplay } from './WaveformDisplay';

declare global {
  var IS_REACT_ACT_ENVIRONMENT: boolean | undefined;
}
globalThis.IS_REACT_ACT_ENVIRONMENT = true;

const pristine = useStore.getState();

let host: HTMLElement;
let root: Root;

beforeEach(() => {
  window.localStorage.clear();
  setLocale('en');
  cancelScheduledLivePreview();
  useStore.setState(pristine, true);
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});

afterEach(async () => {
  cancelScheduledLivePreview();
  vi.useRealTimers();
  vi.unstubAllGlobals();
  await act(async () => {
    root.unmount();
  });
  setLocale('en');
  host.remove();
});

async function runAxe(element: HTMLElement) {
  const results = await axe.run(element, {
    runOnly: {
      type: 'tag',
      values: ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'best-practice'],
    },
    rules: {
      'color-contrast': { enabled: false },
      region: { enabled: false },
    },
  });
  return results.violations.filter((v) => v.impact === 'serious' || v.impact === 'critical');
}

function fakeAnalysis(over: Partial<AudioAnalysis> = {}): AudioAnalysis {
  return {
    path: '/clip.m4a.mp4',
    duration_s: 94.6,
    sample_rate: 48000,
    channels: 1,
    has_video: false,
    peaks: { min: [-0.5], max: [0.5] },
    rms_db: [-22],
    spectrum: { freqs_hz: [80, 1000, 8000], db: [-40, -24, -55] },
    loudness: { integrated_lufs: -28.5, true_peak_dbtp: -0.1 },
    noise_floor_db: -41.2,
    ...over,
  };
}

describe('Studio 4.0 — RotaryKnob & PillToggle controls', () => {
  it('supports vertical pointer drag, wheel step, double-click reset, and passes axe', async () => {
    let val = 60;
    const onChange = vi.fn((next: number) => {
      val = next;
    });
    const renderKnob = async () => {
      await act(async () => {
        root.render(
          createElement(RotaryKnob, {
            label: 'Clean Amount',
            value: val,
            min: 10,
            max: 100,
            step: 5,
            defaultValue: 75,
            formatValue: (v) => `${v}%`,
            size: 'hero',
            onChange,
          }),
        );
      });
    };

    await renderKnob();
    expect(await runAxe(host)).toEqual([]);

    const dial = host.querySelector<HTMLDivElement>('.rk-dial-wrap')!;
    expect(dial).not.toBeNull();

    // Pointer drag up increases value
    await act(async () => {
      dial.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, button: 0, clientY: 200, pointerId: 1 }));
      dial.dispatchEvent(new PointerEvent('pointermove', { bubbles: true, button: 0, clientY: 160, pointerId: 1 }));
      dial.dispatchEvent(new PointerEvent('pointerup', { bubbles: true, button: 0, clientY: 160, pointerId: 1 }));
    });
    expect(val).toBeGreaterThan(60);

    // Double-click resets to defaultValue (75)
    await renderKnob();
    await act(async () => {
      dial.dispatchEvent(new MouseEvent('dblclick', { bubbles: true }));
    });
    expect(val).toBe(75);

    // Wheel up steps value up
    await renderKnob();
    await act(async () => {
      dial.dispatchEvent(new WheelEvent('wheel', { bubbles: true, deltaY: -100 }));
    });
    expect(val).toBeGreaterThan(75);
  });

  it('PillToggle toggles pressed state and passes axe', async () => {
    let pressed = false;
    const renderToggle = async () => {
      await act(async () => {
        root.render(
          createElement(PillToggle, {
            label: 'De-Click',
            pressed,
            onToggle: () => {
              pressed = !pressed;
            },
          }),
        );
      });
    };
    await renderToggle();
    expect(await runAxe(host)).toEqual([]);
    const btn = host.querySelector<HTMLButtonElement>('button')!;
    expect(btn.getAttribute('aria-pressed')).toBe('false');
    await act(async () => {
      btn.click();
    });
    expect(pressed).toBe(true);
  });
});

describe('Studio 4.0 — Simple Studio Mode, Plain-English Diagnosis & Goals', () => {
  it('renders zero-jargon welcome stage in idle and passes axe', async () => {
    await act(async () => {
      root.render(createElement(SimpleWorkspace));
    });
    expect(host.textContent).toContain('Make any voice sound studio-recorded');
    expect(await runAxe(host)).toEqual([]);
  });

  it('diagnoses noisy quiet audio-only .mp4 in plain English and gates Video export by has_video', async () => {
    useStore.setState({
      engineStatus: 'ready',
      source: { path: '/Flute 09.m4a.mp4', name: 'Flute 09.m4a.mp4', origin: 'file' },
      original: fakeAnalysis({ has_video: false }),
      cleanedPath: '/Flute 09_studio.wav',
    });

    await act(async () => {
      root.render(createElement(SimpleWorkspace));
    });

    expect(host.textContent).toContain('Heavy background noise detected');
    expect(host.textContent).toContain('Voice is quiet');
    expect(host.textContent).toContain('Some distortion peaks detected');
    expect(host.textContent).toContain('Audio recording · 1:34');

    const formatOptions = Array.from(host.querySelectorAll<HTMLOptionElement>('.sd-save-select option')).map(
      (o) => o.value,
    );
    expect(formatOptions).not.toContain('video_mux');
    expect(await runAxe(host)).toEqual([]);

    // Now simulate a real video MP4 (`has_video: true`)
    await act(async () => {
      useStore.setState({ original: fakeAnalysis({ has_video: true }) });
    });
    const videoOptions = Array.from(host.querySelectorAll<HTMLOptionElement>('.sd-save-select option')).map(
      (o) => o.value,
    );
    expect(videoOptions).toContain('video_mux');
  });

  it('GoalPresets applies Podcast & YouTube settings, emits toast, and triggers debounced live preview', async () => {
    vi.useFakeTimers();
    const renderRegionPreview = vi.fn().mockResolvedValue({
      previewPath: '/tmp/prev.wav',
      audioUrl: '/api/audio?path=%2Ftmp%2Fprev.wav',
      durationS: 8,
      renderMs: 38,
      warmHit: true,
    });

    useStore.setState({
      engineStatus: 'ready',
      livePreviewEnabled: true,
      client: {
        renderRegionPreview,
        fileUrl: (p: string) => `/api/audio?path=${encodeURIComponent(p)}`,
      } as never,
      source: { path: '/clip.wav', name: 'clip.wav', origin: 'file' },
      original: fakeAnalysis({ path: '/clip.wav' }),
    });

    await act(async () => {
      root.render(createElement(SimpleWorkspace));
    });

    const podcastBtn = Array.from(host.querySelectorAll<HTMLButtonElement>('.goal-card')).find((b) =>
      b.textContent?.includes('Podcast & YouTube'),
    );
    expect(podcastBtn).toBeDefined();

    await act(async () => {
      podcastBtn!.click();
    });

    const st = getState();
    expect(st.activeGoalId).toBe('podcast');
    expect(st.rackSettings.deliveryPreset).toBe('podcast');
    expect(st.rackSettings.deplosive).toBe(true);
    expect(st.rackSettings.deess).toBe(true);
    expect(st.toasts.some((t) => t.text.includes('Podcast & YouTube'))).toBe(true);

    // Debounced warm preview fires cleanly
    await act(async () => {
      vi.advanceTimersByTime(300);
    });
    expect(renderRegionPreview).toHaveBeenCalledTimes(1);
  });

  it('AudioHealthCard summarizes completed cleaning plainly', async () => {
    const report = {
      job_id: 'j1',
      config_hash: 'c',
      input: { path: '/a.wav', sha256: 'x', sample_rate: 48000, channels: 1, samples: 48000, duration_s: 10 },
      output: { path: '/o.wav', sha256: 'y', sample_rate: 48000, channels: 1, samples: 48000, duration_s: 10 },
      core: { id: 'studio', algorithm: 'dfn3', params_hash: 'p' },
      guard: { id: 'g', probe_hash: 'p', calibration_id: 'c' },
      environment: {},
      summary: { units_total: 6, enhanced: 6, reverted: 0, continuity_crossfaded: 0 },
      units: [],
    } satisfies HawaVoCleanReport;

    useStore.setState({
      source: { path: '/a.wav', name: 'a.wav', origin: 'file' },
      original: fakeAnalysis({ noise_floor_db: -40 }),
      cleaned: fakeAnalysis({ noise_floor_db: -58 }),
      report,
    });
    await act(async () => {
      root.render(createElement(AudioHealthCard));
    });
    expect(host.textContent).toContain('Background noise reduced (18 dB quieter)');
    expect(host.textContent).toContain('Voice safety verified');
  });
});

describe('Studio 4.0 — Header Mode Switch, Pro AudioStage, Spectrum HPF, Drawer & Toasts', () => {
  it('Header switches Simple <-> Pro Studio, persists to localStorage, and opens AI drawer via Cmd+K', async () => {
    await act(async () => {
      root.render(createElement(Header));
    });
    expect(await runAxe(host)).toEqual([]);

    const proBtn = Array.from(host.querySelectorAll<HTMLButtonElement>('.hdr-mode-btn')).find(
      (b) => b.textContent?.trim() === 'Pro Studio',
    );
    expect(proBtn).toBeDefined();
    await act(async () => {
      proBtn!.click();
    });
    expect(getState().workspaceMode).toBe('pro');
    expect(window.localStorage.getItem('hawavoclean.workspaceMode')).toBe('pro');

    // Cmd+K opens AI Assistant drawer, Escape closes it
    await act(async () => {
      handleKeyDown(new KeyboardEvent('keydown', { key: 'k', metaKey: true, cancelable: true }));
    });
    expect(getState().assistantDrawerOpen).toBe(true);
    await act(async () => {
      handleKeyDown(new KeyboardEvent('keydown', { key: 'Escape', cancelable: true }));
    });
    expect(getState().assistantDrawerOpen).toBe(false);
  });

  it('SpectrumDisplay converts Hz to musical notes and adjusts HPF via interactive handle', async () => {
    expect(hzToMusicalNote(440)).toBe('A4');
    expect(hzToMusicalNote(80)).toBe('D#2');

    await act(async () => {
      root.render(createElement(SpectrumDisplay));
    });
    const hpfNode = host.querySelector<HTMLButtonElement>('button[aria-label*="High-pass rumble filter node"]');
    expect(hpfNode).not.toBeNull();
    const initialHz = getState().rackSettings.hpfCutoffHz;

    await act(async () => {
      hpfNode!.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true, cancelable: true }));
    });
    expect(getState().rackSettings.hpfCutoffHz).toBe(initialHz + 5);

    // Double-click toggles HPF bypass
    await act(async () => {
      hpfNode!.dispatchEvent(new MouseEvent('dblclick', { bubbles: true }));
    });
    expect(getState().rackSettings.hpfEnabled).toBe(false);
  });

  it('ProWorkspace renders full-width FabFilter spectrum with piano ruler, band overlays, rotary console, and drawers', async () => {
    useStore.setState({
      engineStatus: 'ready',
      source: { path: '/a.wav', name: 'a.wav', origin: 'file' },
      original: fakeAnalysis(),
      rackSettings: {
        ...getState().rackSettings,
        deplosive: true,
        tonalRestoration: true,
        deess: true,
      },
    });
    await act(async () => {
      root.render(createElement(ProWorkspace));
    });
    expect(host.querySelector('.spec-piano-ruler')).not.toBeNull();
    expect(host.querySelector('.band-deplosive')).not.toBeNull();
    expect(host.querySelector('.band-tonal')).not.toBeNull();
    expect(host.querySelector('.band-deess')).not.toBeNull();
    expect(host.querySelector('aside.right.pro-console')).not.toBeNull();
    expect(host.querySelector('button.process[aria-label="PROCESS"]')).not.toBeNull();
    expect(await runAxe(host)).toEqual([]);

    // Opening Units drawer works cleanly
    const unitsBtn = Array.from(host.querySelectorAll<HTMLButtonElement>('.pc-drawer-btn')).find((b) =>
      b.textContent?.includes('Units'),
    );
    expect(unitsBtn).toBeDefined();
    await act(async () => {
      unitsBtn!.click();
    });
    expect(host.querySelector('[role="dialog"][aria-label="Unit Guard Inspector"]')).not.toBeNull();
  });

  it('AudioStage blend slider updates stageBlend and surfaces 2D box repair toolbar', async () => {
    useStore.setState({
      engineStatus: 'ready',
      source: { path: '/a.wav', name: 'a.wav', origin: 'file' },
      original: fakeAnalysis(),
      spectralSelection: { startS: 1, endS: 3, fLowHz: 200, fHighHz: 800 },
    });
    await act(async () => {
      root.render(createElement(AudioStage));
    });
    expect(host.textContent).toContain('Attenuate −24 dB');
    const slider = host.querySelector<HTMLInputElement>('.as-blend-slider')!;
    expect(slider).not.toBeNull();

    await act(async () => {
      useStore.getState().setStageBlend(0.6);
    });
    expect(getState().stageBlend).toBe(0.6);
  });

  it('SlideOverDrawer and ToastStack render accessibly and dismiss cleanly', async () => {
    const onClose = vi.fn();
    useStore.getState().pushToast('Saved cleaned file', 'ok', 0);

    await act(async () => {
      root.render(
        createElement(
          'div',
          null,
          createElement(SlideOverDrawer, { open: true, title: 'AI Audio Assistant', onClose }, createElement('p', null, 'Helper')),
          createElement(ToastStack),
        ),
      );
    });
    expect(await runAxe(host)).toEqual([]);
    expect(host.textContent).toContain('Saved cleaned file');

    const dismiss = host.querySelector<HTMLButtonElement>('.toast-dismiss')!;
    await act(async () => {
      dismiss.click();
    });
    expect(getState().toasts).toEqual([]);

    const closeBtn = host.querySelector<HTMLButtonElement>('.drawer-close')!;
    await act(async () => {
      closeBtn.click();
    });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('scheduleLivePreview respects livePreviewEnabled=false without firing', () => {
    vi.useFakeTimers();
    const renderRegionPreview = vi.fn();
    useStore.setState({
      engineStatus: 'ready',
      livePreviewEnabled: false,
      client: { renderRegionPreview } as never,
      source: { path: '/a.wav', name: 'a.wav', origin: 'file' },
      original: fakeAnalysis(),
    });
    scheduleLivePreview(50);
    vi.advanceTimersByTime(500);
    expect(renderRegionPreview).not.toHaveBeenCalled();
  });
});

describe('Waves C + D — Ship-Readiness Verification', () => {
  it('usePlayerSync syncs playing, abMode, gainOffsetDb, and selectionRange -> loopRange at app level', async () => {
    function Harness() {
      usePlayerSync();
      return createElement('div', null, 'sync');
    }

    await act(async () => {
      useStore.setState({ selectionRange: { start: 1.5, end: 6.0 } });
      root.render(createElement(Harness));
    });
    expect(getPlayer().snapshot().loopRange).toEqual([1.5, 6.0]);

    await act(async () => {
      getPlayer().setLoudnessMatch(true, -20, -17); // -3 dB offset
    });
    expect(getState().gainOffsetDb).toBeCloseTo(-3.0, 2);

    await act(async () => {
      useStore.setState({ selectionRange: null });
    });
    expect(getPlayer().snapshot().loopRange).toBeNull();
  });

  it('EngineClient.createV1Jobs preserves batchId / batch_id from server response', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: true,
        status: 200,
        text: async () =>
          JSON.stringify({
            schema_version: 1,
            batch_id: 'batch-snake-99',
            jobs: [
              { job_id: 'j1', source_id: 's1', output_path: '/out/1.wav', report_path: '/out/1.json' },
              { job_id: 'j2', source_id: 's2', output_path: '/out/2.wav', report_path: '/out/2.json' },
            ],
          }),
      })),
    );
    const client = new EngineClient({ baseUrl: 'http://127.0.0.1:8765' });
    const res = await client.createV1Jobs({
      schema_version: 1,
      source_ids: ['s1', 's2'],
      strategy: { kind: 'smart_safe', allow_generative_reconstruction: false },
    });
    expect(res.batchId).toBe('batch-snake-99');
    expect(res.jobs).toHaveLength(2);
  });

  it('unifies buildStrategyFromState across startJob, submitBatch, and previewRegion including smart_safe overrides & HPF', async () => {
    const createV1Jobs = vi.fn(async () => ({
      schemaVersion: 1 as const,
      batchId: 'b-unified',
      jobs: [{ jobId: 'j-u1', sourceId: 's1', outputPath: '/out/u1.wav', reportPath: '/out/u1.json' }],
    }));
    const getBatch = vi.fn(async () => ({
      batch_id: 'b-unified',
      state: 'queued' as const,
      total_items: 1,
      completed_items: 0,
      failed_items: 0,
      cancelled_items: 0,
      running_items: 0,
      queued_items: 1,
      progress: 0,
      created_at: '2026-11-20T00:00:00Z',
      updated_at: '2026-11-20T00:00:00Z',
      jobs: [],
    }));
    const renderRegionPreview = vi.fn(async () => ({
      previewPath: '/tmp/p.wav',
      audioUrl: '/api/audio?path=%2Ftmp%2Fp.wav',
      renderMs: 25,
      warmHit: true,
      metricsBefore: { lufs: -22, noiseFloorDb: -40 },
      metricsAfter: { lufs: -22, noiseFloorDb: -56 },
      durationS: 6,
    }));

    useStore.setState({
      engineStatus: 'ready',
      mode: 'smart_safe',
      profile: 'studio',
      client: {
        baseUrl: '',
        createV1Jobs,
        getBatch,
        renderRegionPreview,
        fileUrl: (p: string) => `/api/audio?path=${encodeURIComponent(p)}`,
        v1EventsUrl: (id: string) => `/api/v1/jobs/${id}/events`,
        v1BatchEventsUrl: (id: string) => `/api/v1/batches/${id}/events`,
      } as never,
      source: { path: '/clip.wav', name: 'clip.wav', sourceId: 's1', origin: 'file' },
      original: fakeAnalysis({ path: '/clip.wav' }),
    });

    updateRackSettings({
      ...DEFAULT_STUDIO_RACK_SETTINGS,
      strength: 0.9,
      ambienceFloorDb: -30,
      hpfEnabled: true,
      hpfCutoffHz: 95,
      deliveryPreset: 'podcast',
      stereoMode: 'true_stereo_ms',
      deplosive: true,
      deess: false,
    });

    const built = buildStrategyFromState();
    expect(built.ok).toBe(true);
    if (built.ok) {
      expect(built.strategy).toMatchObject({
        kind: 'smart_safe',
        strength: 0.9,
        ambienceFloorDb: -30,
        hpfEnabled: true,
        hpfCutoffHz: 95,
        deliveryPreset: 'podcast',
        stereoMode: 'true_stereo_ms',
        proRepair: expect.objectContaining({ deplosive: true, deess: false }),
      });
    }

    await startJob();
    expect(createV1Jobs).toHaveBeenCalledWith(
      expect.objectContaining({
        strategy: expect.objectContaining({
          kind: 'smart_safe',
          hpfEnabled: true,
          hpfCutoffHz: 95,
          strength: 0.9,
        }),
      }),
    );

    useStore.setState({ job: null });
    await submitBatch([{ path: '/a.wav', name: 'a.wav', sourceId: 's1' }]);
    expect(getState().batchDrawerOpen).toBe(true);
    expect(createV1Jobs).toHaveBeenLastCalledWith(
      expect.objectContaining({
        strategy: expect.objectContaining({
          kind: 'smart_safe',
          hpfEnabled: true,
          hpfCutoffHz: 95,
          deliveryPreset: 'podcast',
        }),
      }),
    );

    const playSpy = vi.spyOn(getPlayer(), 'play').mockResolvedValue(undefined);
    await previewRegion({ autoPlay: true });
    expect(renderRegionPreview).toHaveBeenCalledWith(
      expect.objectContaining({
        hpfEnabled: true,
        hpfCutoffHz: 95,
        deliveryPreset: 'podcast',
        stereoMode: 'true_stereo_ms',
        strength: 0.9,
        ambienceFloorDb: -30,
      }),
    );
    expect(playSpy).toHaveBeenCalledTimes(1);
    playSpy.mockRestore();
  });

  it('Hold for Original never flips Original->Cleaned on hover, switches on press/release & keyboard Space, and AudioHealthCard CTA says Process', async () => {
    const p = getPlayer();
    const hasDeckSpy = vi.spyOn(p, 'hasDeck').mockReturnValue(true);
    const setActiveSpy = vi.spyOn(p, 'setActive').mockImplementation(() => undefined);

    try {
      useStore.setState({
        engineStatus: 'ready',
        source: { path: '/a.wav', name: 'a.wav', origin: 'file' },
        original: fakeAnalysis(),
        cleanedPath: '/out/a.wav',
        previewActive: true,
        abMode: 'original',
        loudnessMatch: false,
      });

      await act(async () => {
        root.render(createElement(SimpleWorkspace));
      });

      expect(host.textContent).toContain('click Process');
      expect(host.textContent).toContain('level match off');

      const holdBtn = host.querySelector<HTMLButtonElement>('button.btn-hold-orig')!;
      expect(holdBtn).not.toBeNull();

      // Hovering over Hold for Original while on Original must NEVER flip to Cleaned
      await act(async () => {
        holdBtn.dispatchEvent(new PointerEvent('pointerenter', { bubbles: true }));
        holdBtn.dispatchEvent(new PointerEvent('pointerleave', { bubbles: true }));
      });
      expect(getState().abMode).toBe('original');

      // Switch to Cleaned, then press & release Hold for Original
      await act(async () => {
        useStore.setState({ abMode: 'cleaned' });
      });
      await act(async () => {
        holdBtn.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, button: 0 }));
      });
      expect(getState().abMode).toBe('original');
      await act(async () => {
        holdBtn.dispatchEvent(new PointerEvent('pointerup', { bubbles: true, button: 0 }));
      });
      expect(getState().abMode).toBe('cleaned');

      // Keyboard Space hold on button
      await act(async () => {
        holdBtn.dispatchEvent(new KeyboardEvent('keydown', { key: ' ', bubbles: true, cancelable: true }));
      });
      expect(getState().abMode).toBe('original');
      await act(async () => {
        holdBtn.dispatchEvent(new KeyboardEvent('keyup', { key: ' ', bubbles: true, cancelable: true }));
      });
      expect(getState().abMode).toBe('cleaned');
    } finally {
      hasDeckSpy.mockRestore();
      setActiveSpy.mockRestore();
    }
  });

  it('SpectrumDisplay aligns HPF curve, node, overlays, and piano ruler to SPECTRUM_PLOT_INSETS', async () => {
    useStore.setState({
      engineStatus: 'ready',
      source: { path: '/a.wav', name: 'a.wav', origin: 'file' },
      original: fakeAnalysis(),
    });
    await act(async () => {
      root.render(createElement(ProWorkspace));
    });

    const sec = host.querySelector('section.spectrum-panel.pro-spectrum-row');
    expect(sec).not.toBeNull();

    const overlay = host.querySelector<HTMLDivElement>('.spec-plot-overlay')!;
    expect(overlay).not.toBeNull();
    expect(overlay.style.left).toBe('33px');
    expect(overlay.style.right).toBe('9px');
    expect(overlay.style.top).toBe('11px');
    expect(overlay.style.bottom).toBe('17px');

    const piano = host.querySelector<HTMLDivElement>('.spec-piano-ruler')!;
    expect(piano.style.marginLeft).toBe('33px');
    expect(piano.style.marginRight).toBe('9px');

    // Pointer drag at ~20% across inset plot width (20 Hz * 2^(0.2 * log2(1000)) ≈ 80 Hz)
    vi.spyOn(overlay, 'getBoundingClientRect').mockReturnValue({
      left: 33,
      top: 11,
      width: 500,
      height: 150,
      right: 533,
      bottom: 161,
      x: 33,
      y: 11,
      toJSON: () => ({}),
    });
    const hpfBtn = overlay.querySelector<HTMLButtonElement>('button[aria-label*="High-pass"]')!;
    await act(async () => {
      hpfBtn.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, button: 0, clientX: 33 + 100, pointerId: 1 }));
      hpfBtn.dispatchEvent(new PointerEvent('pointerup', { bubbles: true, button: 0, clientX: 33 + 100, pointerId: 1 }));
    });
    expect(getState().rackSettings.hpfCutoffHz).toBe(80);
  });

  it('AudioStage blends waveform + spectrogram smoothly without blanking on HIDE and opens FloatingPopover for box repair', async () => {
    useStore.setState({
      engineStatus: 'ready',
      stageBlend: 1,
      source: { path: '/a.wav', name: 'a.wav', origin: 'file' },
      original: fakeAnalysis(),
      spectralSelection: { startS: 1, endS: 3, fLowHz: 300, fHighHz: 900 },
    });
    await act(async () => {
      root.render(createElement(AudioStage));
    });

    expect(host.querySelector('#as-spectral-repair-popover')).not.toBeNull();
    expect(host.querySelector('.as-spec-layer')).not.toBeNull();

    // Clicking HIDE at 100% blend resets stageBlend to 0 instead of leaving a blank stage
    const hideBtn = host.querySelector<HTMLButtonElement>('.as-spec-layer button.wave-fit');
    expect(hideBtn?.textContent).toBe('HIDE');
    await act(async () => {
      hideBtn?.click();
    });
    expect(getState().stageBlend).toBe(0);
    expect(host.querySelector('.as-wave-layer')).not.toBeNull();
  });

  it('DualPlayer wires ITU-R BS.1770 K-weighting, computes live NR (dB removed), and resets peak hold on new original source', async () => {
    const biquads: Array<{ type: string; frequency: { value: number }; gain: { value: number }; Q: { value: number } }> = [];
    const origSamples = new Float32Array(2048).fill(0.4);
    const cleanSamples = new Float32Array(2048).fill(0.1); // ~12 dB quieter
    const kSamples = new Float32Array(4096).fill(0.2);
    let analyserCall = 0;

    class FullMockAudioContext {
      currentTime = 2.0;
      sampleRate = 48000;
      state: AudioContextState = 'running';
      destination = {};
      resume = vi.fn(async () => undefined);
      close = vi.fn(async () => undefined);
      createMediaElementSource = vi.fn(() => ({ connect: vi.fn(), disconnect: vi.fn() }));
      createGain = vi.fn(() => ({
        gain: {
          value: 1,
          cancelScheduledValues: vi.fn(),
          setValueAtTime: vi.fn(),
          linearRampToValueAtTime: vi.fn(),
        },
        connect: vi.fn(),
        disconnect: vi.fn(),
      }));
      createBiquadFilter = vi.fn(() => {
        const node = {
          type: 'lowpass',
          frequency: { value: 0 },
          gain: { value: 0 },
          Q: { value: 1 },
          connect: vi.fn(),
          disconnect: vi.fn(),
        };
        biquads.push(node);
        return node;
      });
      createAnalyser = vi.fn(() => {
        const idx = analyserCall++;
        const sourceBuf = idx === 1 ? kSamples : idx === 2 ? origSamples : idx === 3 ? cleanSamples : origSamples;
        return {
          fftSize: sourceBuf.length,
          smoothingTimeConstant: 0.78,
          minDecibels: -100,
          maxDecibels: 0,
          connect: vi.fn(),
          disconnect: vi.fn(),
          getFloatTimeDomainData: vi.fn((out: Float32Array) => {
            const n = Math.min(out.length, sourceBuf.length);
            for (let i = 0; i < n; i++) out[i] = sourceBuf[i] ?? 0;
          }),
        };
      });
    }

    vi.stubGlobal('AudioContext', FullMockAudioContext);
    vi.stubGlobal('URL', {
      ...URL,
      createObjectURL: vi.fn(() => `blob:k-${Math.random().toString(16).slice(2)}`),
      revokeObjectURL: vi.fn(),
    });
    vi.stubGlobal(
      'fetch',
      vi.fn(async (_url: string, init?: RequestInit) => {
        const headers = new Headers();
        if (init?.headers) {
          headers.set('Content-Range', 'bytes 0-0/16');
          headers.set('Content-Length', '1');
        }
        return {
          ok: true,
          status: init?.headers ? 206 : 200,
          headers,
          arrayBuffer: async () => new ArrayBuffer(16),
          blob: async () => new Blob([new Uint8Array(16)]),
        } as unknown as Response;
      }),
    );

    const p = new DualPlayer();
    try {
      p.load('original', 'http://127.0.0.1:8765/api/audio?path=/orig1.wav');
      p.load('cleaned', 'http://127.0.0.1:8765/api/audio?path=/clean1.wav');
      for (let i = 0; i < 8; i++) await Promise.resolve();
      await p.play();

      // ITU-R BS.1770 Stage 1 (+4 dB @ ~1682 Hz highshelf) + Stage 2 (38 Hz Q=0.5 highpass)
      expect(biquads).toHaveLength(2);
      expect(biquads[0]?.type).toBe('highshelf');
      expect(biquads[0]?.gain.value).toBeCloseTo(4.0, 1);
      expect(biquads[0]?.frequency.value).toBeCloseTo(1681.97, 0);
      expect(biquads[1]?.type).toBe('highpass');
      expect(biquads[1]?.frequency.value).toBeCloseTo(38.135, 0);

      const levels = p.getLiveMeterLevels();
      expect(levels.truePeakHoldDbfs).toBeGreaterThan(-10);
      expect(levels.nrDb).not.toBeNull();
      expect(levels.nrDb!).toBeGreaterThan(2);

      // With audio silenced, peak hold still retains the previous clip's peak until reset
      origSamples.fill(0);
      kSamples.fill(0);
      cleanSamples.fill(0);
      expect(p.getLiveMeterLevels().truePeakHoldDbfs).toBeGreaterThan(-10);

      // Loading a brand new original source resets peak hold cleanly
      p.load('original', 'http://127.0.0.1:8765/api/audio?path=/orig2.wav');
      expect(p.getLiveMeterLevels().truePeakHoldDbfs).toBe(-70);
    } finally {
      p.dispose();
    }
  });

  it('LiveMeter renders NR readout with 0 React re-renders during playback', async () => {
    let renders = 0;
    const rafQueue: FrameRequestCallback[] = [];
    vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => {
      rafQueue.push(cb);
      return rafQueue.length;
    });
    vi.stubGlobal('cancelAnimationFrame', vi.fn());

    function MeterProbe() {
      renders += 1;
      return createElement(LiveMeter, {
        playing: true,
        readLevels: () => ({
          momentaryLufs: -18.5,
          shortTermLufs: -19.2,
          peakDbfs: -2.0,
          truePeakHoldDbfs: -1.8,
          nrDb: 9.6,
        }),
      });
    }

    await act(async () => {
      root.render(createElement(MeterProbe));
    });
    const afterMount = renders;
    // Pump 10 rAF frames -> DOM updates directly with 0 additional React renders!
    for (let i = 1; i <= 10; i++) {
      rafQueue.shift()?.(i * 60);
    }
    expect(renders).toBe(afterMount);
    expect(host.textContent).toContain('-18.5');
    expect(host.textContent).toContain('−9.6 dB');
  });

  it('Sorani Kurdish (ckb) persists RTL on boot and translates Studio 4.0 surfaces with 0 axe violations', async () => {
    window.localStorage.setItem('hawavoclean_locale', 'ckb');
    expect(initLocaleFromStorage()).toBe('ckb');
    expect(document.documentElement.dir).toBe('rtl');
    expect(document.documentElement.lang).toBe('ckb');

    await act(async () => {
      root.render(createElement('div', null, createElement(Header), createElement(SimpleWorkspace)));
    });
    expect(host.textContent).toContain('سادە');
    expect(host.textContent).toContain('ستۆدیۆی پرۆ');
    expect(host.textContent).toContain('هەر دەنگێک وەک ستۆدیۆ تۆمارکراو لێ بکە');
    expect(host.textContent).toContain('تاقیکردنەوە بە مایک (٥ چرکە)');
    expect(await runAxe(host)).toEqual([]);
  });

  it('MicTestButton encodes valid 16-bit PCM WAV and degrades gracefully when microphone is unavailable or denied', async () => {
    const wav = encodeWavPcm16(new Float32Array([0, 0.5, -0.5, 1.0]), 48000);
    expect(wav.byteLength).toBe(44 + 8);
    const view = new DataView(wav);
    expect(String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3))).toBe('RIFF');
    expect(String.fromCharCode(view.getUint8(8), view.getUint8(9), view.getUint8(10), view.getUint8(11))).toBe('WAVE');
    expect(view.getUint32(24, true)).toBe(48000);

    // Denied permission -> graceful notice & toast, no crash
    const origMedia = navigator.mediaDevices;
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: {
        getUserMedia: vi.fn(async () => {
          const err = new Error('Permission denied');
          err.name = 'NotAllowedError';
          throw err;
        }),
      },
    });
    try {
      useStore.setState({ engineStatus: 'ready' });
      await act(async () => {
        root.render(createElement(MicTestButton));
      });
      const btn = host.querySelector<HTMLButtonElement>('button.sw-mic-btn')!;
      expect(btn).not.toBeNull();
      await act(async () => {
        btn.click();
      });
      expect(host.textContent).toContain('Microphone permission was denied');
      expect(await runAxe(host)).toEqual([]);
    } finally {
      Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: origMedia });
    }
  });

  it('Header Undo/Redo, spectral repair undo, SourceStrip multiple+Batch, Waveform +/- zoom, Transport Hold Orig, and ShortcutOverlay host mode work cleanly', async () => {
    const repairSpectralBox = vi.fn(async () => ({
      outputPath: '/out/repaired.wav',
      audioUrl: '/api/audio?path=%2Fout%2Frepaired.wav',
      startS: 1,
      endS: 2,
      fLowHz: 200,
      fHighHz: 500,
      attenuationDb: 24,
      renderMs: 30,
    }));
    useStore.setState({
      host: 'electron',
      engineStatus: 'ready',
      shortcutsOpen: true,
      client: {
        baseUrl: '',
        repairSpectralBox,
        fileUrl: (p: string) => `/api/audio?path=${encodeURIComponent(p)}`,
      } as never,
      source: { path: '/a.wav', name: 'a.wav', origin: 'file' },
      original: fakeAnalysis(),
      spectralSelection: { startS: 1, endS: 2, fLowHz: 200, fHighHz: 500 },
    });

    await applySpectralBoxRepair(24);
    expect(getState().cleanedPath).toBe('/out/repaired.wav');
    expect(getState().spectralUndo.canUndo).toBe(true);

    undoAction();
    expect(getState().cleanedPath).toBeNull();
    expect(getState().statusLine).toBe('Undid spectral repair');

    redoAction();
    expect(getState().cleanedPath).toBe('/out/repaired.wav');
    expect(getState().statusLine).toBe('Redid spectral repair');

    await act(async () => {
      root.render(
        createElement(
          'div',
          null,
          createElement(Header),
          createElement(SourceStrip),
          createElement(WaveformDisplay),
          createElement(Transport),
          createElement(ShortcutOverlay),
        ),
      );
    });

    expect(host.querySelector('button[aria-label^="Undo"]')).not.toBeNull();
    expect(host.querySelector('button[aria-label^="Redo"]')).not.toBeNull();
    expect(host.querySelector('button.btn-batch-pick')).not.toBeNull();
    expect(host.querySelector('input[type="file"]')?.hasAttribute('multiple')).toBe(true);
    expect(host.querySelector('button[aria-label="Zoom in"]')).not.toBeNull();
    expect(host.querySelector('button[aria-label="Zoom out"]')).not.toBeNull();
    expect(host.querySelector('button.btn-hold-orig')).not.toBeNull();
    expect(host.textContent).toContain('· desktop mode');
    expect(host.textContent).toContain('AI Audio Assistant');
  });
});
