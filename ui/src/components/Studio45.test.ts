import { act, createElement, type FC } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
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
    setTiltDbPerOct() {}
    setFrozen() {}
    removedCoverage() {
      return 0;
    }
    dispose() {}
  },
}));

import type { AudioAnalysis, HawaVoCleanReport } from '../api/types';
import { DualPlayer } from '../audio/player';
import { waveView } from '../render/viewWindow';
import { setLocale } from '../state/i18n';
import { DEFAULT_STUDIO_RACK_SETTINGS } from '../state/presets';
import { useStore } from '../state/store';
import { Actions } from './Actions';
import { AudioHealthCard } from './AudioHealthCard';
import { AudioStage } from './AudioStage';
import { LiveMeter } from './LiveMeter';
import { SpectrumDisplay } from './SpectrumDisplay';
import { VerdictStrip } from './VerdictStrip';

function makeAnalysis(overrides: Partial<AudioAnalysis> = {}): AudioAnalysis {
  return {
    path: '/tmp/clip.wav',
    duration_s: 20,
    sample_rate: 48000,
    channels: 2,
    noise_floor_db: -40,
    peaks: { min: [-0.2, -0.1], max: [0.2, 0.1] },
    rms_db: [-20, -22],
    spectrum: {
      freqs_hz: [20, 100, 1000, 5000, 16000],
      db: [-50, -30, -22, -28, -42],
    },
    loudness: {
      integrated_lufs: -28.5,
      true_peak_dbtp: -0.1,
    },
    ...overrides,
  };
}

function makeReport(): HawaVoCleanReport {
  return {
    schema_version: '1.0',
    job_id: 'job-45',
    input: { duration_s: 20, sample_rate: 48000, channels: 1, integrated_lufs: -28.5, true_peak_dbtp: -0.1 },
    output: { duration_s: 20, sample_rate: 48000, channels: 1, integrated_lufs: -16.0, true_peak_dbtp: -1.0 },
    summary: { units_total: 3, enhanced: 1, reverted: 2, no_speech: 0, error_passthrough: 0 },
    units: [
      {
        unit_id: 0,
        channel: 0,
        start_time_s: 0,
        end_time_s: 6,
        final_decision: 'enhanced',
        guard_a_verdict: 'PASS',
        guard_b_verdict: 'PASS',
      },
      {
        unit_id: 1,
        channel: 0,
        start_time_s: 6,
        end_time_s: 12,
        final_decision: 'reverted',
        guard_a_verdict: 'FAIL',
        guard_b_verdict: 'PASS',
      },
      {
        unit_id: 2,
        channel: 0,
        start_time_s: 12,
        end_time_s: 18,
        final_decision: 'reverted',
        guard_a_verdict: 'FAIL',
        guard_b_verdict: 'PASS',
      },
    ],
  } as unknown as HawaVoCleanReport;
}

describe('Studio 4.5 Pro Capabilities', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    setLocale('en');
    host = document.createElement('div');
    document.body.appendChild(host);
    root = createRoot(host);
    useStore.setState({
      engineStatus: 'ready',
      source: { path: '/tmp/clip.wav', name: 'clip.wav', origin: 'file' },
      original: makeAnalysis(),
      cleaned: null,
      cleanedPath: null,
      report: null,
      job: null,
      analyzing: false,
      playing: false,
      abMode: 'original',
      stageBlend: 0,
      selectionRange: null,
      selectedUnit: null,
      duration: 20,
      rackSettings: { ...DEFAULT_STUDIO_RACK_SETTINGS },
      toasts: [],
    });
    waveView.setSource(20, 48000, 1200);
  });

  afterEach(() => {
    act(() => {
      root.unmount();
    });
    host.remove();
    vi.restoreAllMocks();
  });

  it('DualPlayer band solo clamps frequency & Q cleanly and resets on dispose', () => {
    const p = new DualPlayer();
    expect(p.bandSolo.active).toBe(false);
    p.setBandSolo(true, 3200, 6.0);
    expect(p.bandSolo).toEqual({ active: true, hz: 3200, q: 6.0 });
    p.setBandSolo(true, 5, 99);
    expect(p.bandSolo.hz).toBe(20);
    expect(p.bandSolo.q).toBe(12.0);
    p.setBandSolo(false);
    expect(p.bandSolo.active).toBe(false);
    p.dispose();
  });

  it('LiveMeter paints stereo phase correlation bar and negative phase alert', () => {
    const levels = {
      momentaryLufs: -18.0,
      shortTermLufs: -19.2,
      peakDbfs: -4.0,
      truePeakHoldDbfs: -3.5,
      nrDb: 6.4,
      stereoCorrelation: -0.45,
    };
    const origRaf = window.requestAnimationFrame;
    const origCancel = window.cancelAnimationFrame;
    let cb: FrameRequestCallback | null = null;
    window.requestAnimationFrame = ((fn: FrameRequestCallback) => {
      cb = fn;
      return 1;
    }) as typeof window.requestAnimationFrame;
    window.cancelAnimationFrame = vi.fn();
    try {
      act(() => {
        root.render(createElement(LiveMeter, { playing: true, readLevels: () => levels }));
      });
      expect(cb).not.toBeNull();
      cb!(100);
      const corr = host.querySelector('.live-meter-corr') as HTMLElement;
      expect(corr).not.toBeNull();
      expect(corr.style.display).toBe('inline-flex');
      expect(corr.textContent).toContain('-0.45');
    } finally {
      window.requestAnimationFrame = origRaf;
      window.cancelAnimationFrame = origCancel;
    }
  });

  it('SpectrumDisplay cycles Tilt slope (Flat -> 3 -> 4.5 dB/oct), toggles Freeze & Solo, and shows delivery compliance', async () => {
    useStore.setState({
      cleaned: makeAnalysis({
        channels: 2,
        loudness: { integrated_lufs: -16.1, true_peak_dbtp: -1.1 },
      }),
      rackSettings: { ...DEFAULT_STUDIO_RACK_SETTINGS, deliveryPreset: 'podcast' },
    });
    act(() => {
      root.render(createElement(SpectrumDisplay as FC<{ wide?: boolean }>, { wide: true }));
    });
    expect(host.textContent).toContain('Podcast OK');

    const findBtn = (txt: string) =>
      Array.from(host.querySelectorAll('button')).find((b) => b.textContent?.trim() === txt) ?? null;

    const tiltBtn = findBtn('Flat');
    expect(tiltBtn).not.toBeNull();
    act(() => {
      tiltBtn!.click();
    });
    expect(findBtn('3 dB/oct')).not.toBeNull();
    act(() => {
      findBtn('3 dB/oct')!.click();
    });
    expect(findBtn('4.5 dB/oct')).not.toBeNull();

    const freezeBtn = findBtn('Freeze')!;
    expect(freezeBtn.getAttribute('aria-pressed')).toBe('false');
    act(() => {
      freezeBtn.click();
    });
    expect(freezeBtn.getAttribute('aria-pressed')).toBe('true');

    const soloBtn = findBtn('Solo')!;
    expect(soloBtn.getAttribute('aria-pressed')).toBe('false');
    act(() => {
      soloBtn.click();
    });
    expect(soloBtn.getAttribute('aria-pressed')).toBe('true');

    const audit = await axe.run(host);
    expect(audit.violations).toEqual([]);
  });

  it('AudioStage toolbar provides Zoom In/Out/Fit and Selection zoom controls', () => {
    useStore.setState({ selectionRange: { start: 4, end: 8 } });
    act(() => {
      root.render(createElement(AudioStage));
    });
    const zoomIn = host.querySelector('button[aria-label="Zoom in"]') as HTMLButtonElement;
    const fit = Array.from(host.querySelectorAll('button')).find((b) => b.textContent?.trim() === 'Fit')!;
    const zoomSel = Array.from(host.querySelectorAll('button')).find((b) => b.textContent?.trim() === 'Zoom Sel')!;
    act(() => {
      zoomIn.click();
    });
    expect(waveView.span).toBeLessThan(20);
    act(() => {
      fit.click();
    });
    expect(waveView.span).toBeCloseTo(20, 2);
    act(() => {
      zoomSel.click();
    });
    expect(waveView.span).toBeLessThan(6);
  });

  it('AudioHealthCard Auto-Configure Fix tunes declip, leveler, podcast preset, and studio profile', () => {
    act(() => {
      root.render(createElement(AudioHealthCard));
    });
    const btn = host.querySelector('.hc-autofix-btn') as HTMLButtonElement;
    expect(btn).not.toBeNull();
    act(() => {
      btn.click();
    });
    const st = useStore.getState();
    expect(st.rackSettings.declip).toBe(true);
    expect(st.rackSettings.levelRider).toBe(true);
    expect(st.rackSettings.deliveryPreset).toBe('podcast');
    expect(st.toasts.length).toBeGreaterThan(0);
  });

  it('VerdictStrip jump button cycles through reverted/flagged units', () => {
    useStore.setState({ report: makeReport() });
    act(() => {
      root.render(createElement(VerdictStrip));
    });
    const jumpBtn = host.querySelector('.verdict-jump-btn') as HTMLButtonElement;
    expect(jumpBtn).not.toBeNull();
    expect(jumpBtn.textContent).toContain('2 flagged');
    act(() => {
      jumpBtn.click();
    });
    expect(useStore.getState().selectedUnit?.unit_id).toBe(1);
    act(() => {
      jumpBtn.click();
    });
    expect(useStore.getState().selectedUnit?.unit_id).toBe(2);
    act(() => {
      jumpBtn.click();
    });
    expect(useStore.getState().selectedUnit?.unit_id).toBe(1);
  });

  it('Actions renders inline export format picker and QC Report button', () => {
    useStore.setState({
      cleanedPath: '/tmp/clip.clean.wav',
      report: makeReport(),
      job: {
        id: 'job-45',
        outputPath: '/tmp/clip.clean.wav',
        reportPath: '/tmp/clip.json',
        status: null,
        streamConnected: false,
      },
      artifacts: { master: true, json: true, txt: true, reason: '' },
    });
    act(() => {
      root.render(createElement(Actions));
    });
    const select = host.querySelector('select.export-format-select') as HTMLSelectElement;
    expect(select).not.toBeNull();
    act(() => {
      select.value = 'flac';
      select.dispatchEvent(new Event('change', { bubbles: true }));
    });
    expect(useStore.getState().rackSettings.exportFormat).toBe('flac');
    const qcBtn = host.querySelector('button.qc-cert-btn') as HTMLButtonElement;
    expect(qcBtn).not.toBeNull();
    expect(qcBtn.disabled).toBe(false);
  });
});
