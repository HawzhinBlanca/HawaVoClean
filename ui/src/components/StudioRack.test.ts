import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import axe from 'axe-core';
import {
  BUILTIN_PRESETS,
  DEFAULT_STUDIO_RACK_SETTINGS,
  PRESET_STORAGE_KEY,
  exportPresetJson,
  type StudioRackSettings,
} from '../state/presets';
import { StudioRack, type StudioRackProps } from './StudioRack';

declare global {
  var IS_REACT_ACT_ENVIRONMENT: boolean | undefined;
}
globalThis.IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLElement;
let root: Root;

beforeEach(() => {
  window.localStorage.removeItem(PRESET_STORAGE_KEY);
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(async () => {
    root.unmount();
  });
  host.remove();
});

function setNativeValue(
  el: HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement,
  value: string,
): void {
  const proto =
    el instanceof HTMLTextAreaElement
      ? HTMLTextAreaElement.prototype
      : el instanceof HTMLSelectElement
        ? HTMLSelectElement.prototype
        : HTMLInputElement.prototype;
  const setter = Object.getOwnPropertyDescriptor(proto, 'value')?.set;
  setter?.call(el, value);
  el.dispatchEvent(new Event('input', { bubbles: true }));
  el.dispatchEvent(new Event('change', { bubbles: true }));
}

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
  return results.violations;
}

async function renderRack(props: Partial<StudioRackProps> = {}) {
  const onChange = props.onChange ?? vi.fn();
  const fullProps: StudioRackProps = {
    settings: { ...DEFAULT_STUDIO_RACK_SETTINGS },
    onChange,
    ...props,
  };
  await act(async () => {
    root.render(createElement(StudioRack, fullProps));
  });
  return { onChange };
}

describe('StudioRack', () => {
  it('renders all core controls and updates strength, ambience floor, finishing toggles, delivery, and export format', async () => {
    const changes: StudioRackSettings[] = [];
    let current: StudioRackSettings = { ...DEFAULT_STUDIO_RACK_SETTINGS };
    const rerender = async (next: StudioRackSettings) => {
      current = next;
      changes.push(next);
      await renderRack({ settings: current, onChange: rerender, isVideoSource: false });
    };

    await renderRack({ settings: current, onChange: rerender, isVideoSource: false });

    // Strength slider (10% - 100%)
    const strengthSlider = host.querySelector<HTMLInputElement>('input[type="range"][min="10"][max="100"]');
    expect(strengthSlider).not.toBeNull();
    expect(host.querySelector('[data-testid="strength-readout"]')?.textContent).toBe('75%');

    await act(async () => {
      setNativeValue(strengthSlider!, '90');
    });
    expect(current.strength).toBe(0.9);

    // Ambience floor selector
    const ambOffBtn = Array.from(host.querySelectorAll<HTMLButtonElement>('button')).find(
      (b) => b.textContent?.trim() === 'Off',
    );
    expect(ambOffBtn).toBeDefined();
    await act(async () => {
      ambOffBtn!.click();
    });
    expect(current.ambienceFloorDb).toBeNull();

    // Finishing toggles
    const tonalBtn = Array.from(host.querySelectorAll<HTMLButtonElement>('button')).find(
      (b) => b.textContent?.trim() === 'Tonal',
    );
    expect(tonalBtn?.getAttribute('aria-pressed')).toBe('false');
    await act(async () => {
      tonalBtn!.click();
    });
    expect(current.tonalRestoration).toBe(true);

    // HPF toggle hides cutoff slider when bypassed
    const hpfBtn = Array.from(host.querySelectorAll<HTMLButtonElement>('button')).find(
      (b) => b.textContent?.trim() === 'HPF',
    );
    expect(host.querySelector('input[type="range"][min="20"][max="120"]')).not.toBeNull();
    await act(async () => {
      hpfBtn!.click();
    });
    expect(current.hpfEnabled).toBe(false);
    expect(host.querySelector('input[type="range"][min="20"][max="120"]')).toBeNull();

    // Delivery target pill
    const podcastBtn = Array.from(host.querySelectorAll<HTMLButtonElement>('button')).find(
      (b) => b.textContent?.trim() === 'Podcast -16',
    );
    await act(async () => {
      podcastBtn!.click();
    });
    expect(current.deliveryPreset).toBe('podcast');

    // Mux Video disabled when isVideoSource=false
    const muxBtn = Array.from(host.querySelectorAll<HTMLButtonElement>('button')).find(
      (b) => b.textContent?.trim() === 'Mux Video',
    );
    expect(muxBtn?.disabled).toBe(true);
    expect(changes.length).toBeGreaterThanOrEqual(4);
  });

  it('enables Mux Video export format when isVideoSource is true and updates naming template preview', async () => {
    const onChange = vi.fn();
    await renderRack({
      settings: { ...DEFAULT_STUDIO_RACK_SETTINGS },
      onChange,
      isVideoSource: true,
      sampleFileName: 'keynote',
      activeProfile: 'studio_pro',
    });

    const muxBtn = Array.from(host.querySelectorAll<HTMLButtonElement>('button')).find(
      (b) => b.textContent?.trim() === 'Mux Video',
    );
    expect(muxBtn?.disabled).toBe(false);
    await act(async () => {
      muxBtn!.click();
    });
    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({ exportFormat: 'video_mux' }),
    );

    const namingInput = host.querySelector<HTMLInputElement>(
      'input[placeholder="{name}_{profile}_{preset}"]',
    );
    expect(namingInput).not.toBeNull();
    expect(host.textContent).toContain('keynote_studio_pro_ebu_r128.wav');

    await act(async () => {
      setNativeValue(namingInput!, '{name}_clean');
    });
    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({ namingTemplate: '{name}_clean' }),
    );
  });

  it('supports selecting built-in presets, saving user presets, exporting JSON, and rejecting corrupt import JSON', async () => {
    const onChange = vi.fn();
    const onExportPreset = vi.fn();
    await renderRack({
      settings: { ...DEFAULT_STUDIO_RACK_SETTINGS },
      onChange,
      onExportPreset,
    });

    // Select a built-in preset from dropdown
    const presetSelect = host.querySelector<HTMLSelectElement>('select.sr-select');
    expect(presetSelect).not.toBeNull();
    await act(async () => {
      setNativeValue(presetSelect!, BUILTIN_PRESETS[2]!.id);
    });
    expect(onChange).toHaveBeenCalledWith(BUILTIN_PRESETS[2]!.settings);

    // Save a new user preset
    const saveToggle = Array.from(host.querySelectorAll<HTMLButtonElement>('button')).find(
      (b) => b.textContent?.trim() === 'Save',
    );
    await act(async () => {
      saveToggle!.click();
    });
    const nameInput = host.querySelector<HTMLInputElement>('input[placeholder*="Studio Boom"]');
    await act(async () => {
      setNativeValue(nameInput!, 'Custom Broadcast');
    });
    const saveSubmit = Array.from(host.querySelectorAll<HTMLButtonElement>('button[type="submit"]')).find(
      (b) => b.textContent?.trim() === 'Save Preset',
    );
    await act(async () => {
      saveSubmit!.click();
    });
    expect(host.querySelector('[role="status"]')?.textContent).toContain('Custom Broadcast');

    // Export JSON
    const exportBtn = Array.from(host.querySelectorAll<HTMLButtonElement>('button')).find(
      (b) => b.textContent?.trim() === 'Export',
    );
    await act(async () => {
      exportBtn!.click();
    });
    expect(onExportPreset).toHaveBeenCalledTimes(1);
    const exportedTextarea = host.querySelector<HTMLTextAreaElement>('textarea[readonly]');
    expect(exportedTextarea?.value).toContain('hawavoclean.preset.v1');

    // Import corrupt JSON -> shows alert and does not apply
    const importBtn = Array.from(host.querySelectorAll<HTMLButtonElement>('button')).find(
      (b) => b.textContent?.trim() === 'Import',
    );
    await act(async () => {
      importBtn!.click();
    });
    const importTextarea = host.querySelector<HTMLTextAreaElement>('textarea');
    await act(async () => {
      setNativeValue(importTextarea!, '{"__proto__":{"polluted":true}}');
    });
    const importSubmit = host.querySelector<HTMLButtonElement>('button[type="submit"]');
    const beforeCalls = onChange.mock.calls.length;
    await act(async () => {
      importSubmit!.click();
    });
    expect(host.querySelector('[role="alert"]')).not.toBeNull();
    expect(onChange.mock.calls.length).toBe(beforeCalls);

    // Import valid JSON -> applies preset settings
    const validJson = exportPresetJson(BUILTIN_PRESETS[3]!);
    await act(async () => {
      setNativeValue(importTextarea!, validJson);
    });
    await act(async () => {
      importSubmit!.click();
    });
    expect(onChange).toHaveBeenLastCalledWith(BUILTIN_PRESETS[3]!.settings);
  });

  it('respects disabled state and supports stereo routing & deleting user presets', async () => {
    const onChange = vi.fn();
    const onDeletePreset = vi.fn();
    const userPreset = {
      id: 'user:my-chain',
      name: 'My Chain',
      builtin: false,
      createdAt: '2026-01-01T00:00:00.000Z',
      settings: { ...DEFAULT_STUDIO_RACK_SETTINGS },
    };

    await renderRack({
      settings: { ...DEFAULT_STUDIO_RACK_SETTINGS },
      onChange,
      disabled: true,
      presets: [...BUILTIN_PRESETS, userPreset],
      activePresetId: userPreset.id,
      onDeletePreset,
    });

    const disabledTonal = Array.from(host.querySelectorAll<HTMLButtonElement>('button')).find(
      (b) => b.textContent?.trim() === 'Tonal',
    );
    expect(disabledTonal?.disabled).toBe(true);
    await act(async () => {
      disabledTonal!.click();
    });
    expect(onChange).not.toHaveBeenCalled();

    // Re-render enabled to verify stereo mode and delete preset
    await renderRack({
      settings: { ...DEFAULT_STUDIO_RACK_SETTINGS },
      onChange,
      disabled: false,
      presets: [...BUILTIN_PRESETS, userPreset],
      activePresetId: userPreset.id,
      onDeletePreset,
    });

    const selects = host.querySelectorAll<HTMLSelectElement>('select.sr-select');
    const stereoSelect = selects[1]!;
    await act(async () => {
      setNativeValue(stereoSelect, 'split_speakers');
    });
    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({ stereoMode: 'split_speakers' }),
    );

    const deleteBtn = host.querySelector<HTMLButtonElement>('button[aria-label="Delete preset My Chain"]');
    expect(deleteBtn).not.toBeNull();
    await act(async () => {
      deleteBtn!.click();
    });
    expect(onDeletePreset).toHaveBeenCalledWith('user:my-chain');
  });

  it('passes axe-core WCAG 2.2 AA accessibility checks in default and expanded subpanel states', async () => {
    await renderRack({ isVideoSource: true });
    expect(await runAxe(host)).toEqual([]);

    // Open Save subpanel and re-audit
    const saveToggle = Array.from(host.querySelectorAll<HTMLButtonElement>('button')).find(
      (b) => b.textContent?.trim() === 'Save',
    );
    await act(async () => {
      saveToggle!.click();
    });
    expect(await runAxe(host)).toEqual([]);

    // Open Import subpanel and re-audit
    const importToggle = Array.from(host.querySelectorAll<HTMLButtonElement>('button')).find(
      (b) => b.textContent?.trim() === 'Import',
    );
    await act(async () => {
      importToggle!.click();
    });
    expect(await runAxe(host)).toEqual([]);
  });
});
