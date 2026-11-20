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
import type { AudioAnalysis, HawaVoCleanReport } from '../api/types';
import { handleKeyDown } from '../state/keymap';
import { cancelScheduledLivePreview, scheduleLivePreview } from '../state/liveAudition';
import { useStore, getState } from '../state/store';
import { AudioHealthCard } from './AudioHealthCard';
import { AudioStage } from './AudioStage';
import { RotaryKnob } from './controls/RotaryKnob';
import { PillToggle } from './controls/PillToggle';
import { Header } from './Header';
import { SimpleWorkspace } from './SimpleWorkspace';
import { SlideOverDrawer } from './SlideOverDrawer';
import { hzToMusicalNote, SpectrumDisplay } from './SpectrumDisplay';
import { ToastStack } from './ToastStack';

declare global {
  var IS_REACT_ACT_ENVIRONMENT: boolean | undefined;
}
globalThis.IS_REACT_ACT_ENVIRONMENT = true;

const pristine = useStore.getState();

let host: HTMLElement;
let root: Root;

beforeEach(() => {
  window.localStorage.clear();
  cancelScheduledLivePreview();
  useStore.setState(pristine, true);
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});

afterEach(async () => {
  cancelScheduledLivePreview();
  vi.useRealTimers();
  await act(async () => {
    root.unmount();
  });
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
    handleKeyDown(new KeyboardEvent('keydown', { key: 'k', metaKey: true, cancelable: true }));
    expect(getState().assistantDrawerOpen).toBe(true);
    handleKeyDown(new KeyboardEvent('keydown', { key: 'Escape', cancelable: true }));
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
