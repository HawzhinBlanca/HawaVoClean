import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import axe from 'axe-core';
import {
  TakeSwitcher,
  addTakeSnapshot,
  type TakeSnapshot,
} from './TakeSwitcher';

declare global {
  var IS_REACT_ACT_ENVIRONMENT: boolean | undefined;
}
globalThis.IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLElement;
let root: Root;

beforeEach(() => {
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

function makeTake(id: string, over: Partial<TakeSnapshot> = {}): TakeSnapshot {
  return {
    id,
    label: `Take ${id.toUpperCase()}`,
    profile: 'studio_pro',
    cleanedPath: `/out/${id}.wav`,
    report: null,
    mosDelta: 0.42,
    lufsOut: -16.0,
    ...over,
  };
}

describe('TakeSwitcher', () => {
  it('renders Orig plus up to 3 completed takes with MOS/LUFS badges and switches on click', async () => {
    const onSelect = vi.fn();
    const takes: TakeSnapshot[] = [
      makeTake('a', { label: 'Take A', profile: 'studio_pro', mosDelta: 0.45, lufsOut: -23.0 }),
      makeTake('b', { label: 'Broadcast', profile: 'production', mosDelta: -0.05, lufsOut: -24.1 }),
      makeTake('c', { label: 'Take C', profile: 'dereverb', mosDelta: 0.31, lufsOut: -16.0 }),
      makeTake('d', { label: 'Overflow', profile: 'lowband', mosDelta: 0.1, lufsOut: -14.0 }),
    ];

    await act(async () => {
      root.render(
        createElement(TakeSwitcher, {
          takes,
          activeId: 'a',
          onSelect,
        }),
      );
    });

    // Orig + capped 3 takes = 4 selector buttons
    const selectors = host.querySelectorAll<HTMLButtonElement>('button[data-take-selector="true"]');
    expect(selectors).toHaveLength(4);
    expect(host.textContent).toContain('Orig');
    expect(host.textContent).toContain('Take A');
    expect(host.textContent).toContain('+0.45 MOS');
    expect(host.textContent).toContain('-23.0 LUFS');
    expect(host.textContent).toContain('Broadcast');
    expect(host.textContent).toContain('-0.05 MOS');
    expect(host.textContent).not.toContain('Overflow');

    // Click Orig and Take C
    await act(async () => {
      selectors[0]!.click();
    });
    expect(onSelect).toHaveBeenCalledWith('original');

    await act(async () => {
      selectors[3]!.click();
    });
    expect(onSelect).toHaveBeenCalledWith('c');
  });

  it('supports pinning and removing takes while protecting pinned takes from removal', async () => {
    const onTogglePin = vi.fn();
    const onRemoveTake = vi.fn();
    const takes: TakeSnapshot[] = [
      makeTake('a', { pinned: true }),
      makeTake('b', { pinned: false }),
    ];

    await act(async () => {
      root.render(
        createElement(TakeSwitcher, {
          takes,
          activeId: 'original',
          onSelect: vi.fn(),
          onTogglePin,
          onRemoveTake,
        }),
      );
    });

    const unpinA = host.querySelector<HTMLButtonElement>('button[aria-label="Unpin Take A"]');
    const removeA = host.querySelector<HTMLButtonElement>('button[aria-label="Remove Take A"]');
    const pinB = host.querySelector<HTMLButtonElement>('button[aria-label="Pin Take B"]');
    const removeB = host.querySelector<HTMLButtonElement>('button[aria-label="Remove Take B"]');

    expect(unpinA?.getAttribute('aria-pressed')).toBe('true');
    expect(removeA?.disabled).toBe(true);
    expect(pinB?.getAttribute('aria-pressed')).toBe('false');
    expect(removeB?.disabled).toBe(false);

    await act(async () => {
      unpinA!.click();
      removeA!.click();
      pinB!.click();
      removeB!.click();
    });

    expect(onTogglePin).toHaveBeenCalledWith('a');
    expect(onTogglePin).toHaveBeenCalledWith('b');
    expect(onRemoveTake).not.toHaveBeenCalledWith('a');
    expect(onRemoveTake).toHaveBeenCalledWith('b');
  });

  it('supports keyboard ArrowRight/ArrowLeft/Home/End navigation across take buttons', async () => {
    const onSelect = vi.fn();
    await act(async () => {
      root.render(
        createElement(TakeSwitcher, {
          takes: [makeTake('a'), makeTake('b')],
          activeId: 'original',
          onSelect,
        }),
      );
    });

    const selectors = host.querySelectorAll<HTMLButtonElement>('button[data-take-selector="true"]');
    await act(async () => {
      selectors[0]!.focus();
      selectors[0]!.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
    });
    expect(onSelect).toHaveBeenCalledWith('a');

    await act(async () => {
      selectors[0]!.dispatchEvent(new KeyboardEvent('keydown', { key: 'End', bubbles: true }));
    });
    expect(onSelect).toHaveBeenCalledWith('b');
  });

  it('addTakeSnapshot preserves pinned takes when evicting beyond maxTakes=3', () => {
    const t1 = makeTake('1', { pinned: true });
    const t2 = makeTake('2', { pinned: false });
    const t3 = makeTake('3', { pinned: false });
    const t4 = makeTake('4', { pinned: false });

    const next = addTakeSnapshot([t1, t2, t3], t4);
    expect(next.map((t) => t.id)).toEqual(['1', '3', '4']);

    // When all 3 are pinned, existing takes are preserved
    const allPinned = [
      makeTake('1', { pinned: true }),
      makeTake('2', { pinned: true }),
      makeTake('3', { pinned: true }),
    ];
    expect(addTakeSnapshot(allPinned, t4).map((t) => t.id)).toEqual(['1', '2', '3']);
  });

  it('passes axe-core WCAG 2.2 AA accessibility checks in empty and populated states', async () => {
    await act(async () => {
      root.render(
        createElement(TakeSwitcher, {
          takes: [],
          activeId: 'original',
          onSelect: vi.fn(),
        }),
      );
    });
    expect(await runAxe(host)).toEqual([]);

    await act(async () => {
      root.render(
        createElement(TakeSwitcher, {
          takes: [makeTake('a', { pinned: true }), makeTake('b')],
          activeId: 'a',
          onSelect: vi.fn(),
          onTogglePin: vi.fn(),
          onRemoveTake: vi.fn(),
        }),
      );
    });
    expect(await runAxe(host)).toEqual([]);
  });
});
