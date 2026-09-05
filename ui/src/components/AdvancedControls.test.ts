import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { useStore } from '../state/store';
import { AdvancedControls } from './AdvancedControls';

declare global {
  var IS_REACT_ACT_ENVIRONMENT: boolean | undefined;
}
globalThis.IS_REACT_ACT_ENVIRONMENT = true;

const pristine = useStore.getState();

let host: HTMLElement;
let root: Root;

beforeEach(() => {
  useStore.setState(pristine, true);
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

describe('AdvancedControls', () => {
  it('renders collapsed by default with summary', async () => {
    useStore.setState({
      advancedOpen: false,
      profile: 'studio',
      mode: 'natural',
    });

    await act(async () => {
      root.render(createElement(AdvancedControls));
    });

    const rootEl = host.querySelector('.advanced-controls');
    expect(rootEl).not.toBeNull();
    expect(rootEl?.classList.contains('open')).toBe(false);

    const toggle = host.querySelector<HTMLButtonElement>('.advanced-toggle');
    expect(toggle).not.toBeNull();
    expect(toggle?.getAttribute('aria-expanded')).toBe('false');
    expect(toggle?.textContent).toContain('Advanced Controls');
    expect(toggle?.textContent).toContain('Studio · Natural');

    const panel = host.querySelector('#advanced-controls-panel');
    expect(panel).toBeNull();
  });

  it('expands panel when clicked and toggles store state', async () => {
    useStore.setState({
      advancedOpen: false,
      profile: 'production',
      mode: 'restore',
      speakerId: 'sorani_m1',
    });

    await act(async () => {
      root.render(createElement(AdvancedControls));
    });

    const toggle = host.querySelector<HTMLButtonElement>('.advanced-toggle');
    expect(toggle?.textContent).toContain('Production · Restore (sorani_m1)');

    await act(async () => {
      toggle?.click();
    });

    expect(useStore.getState().advancedOpen).toBe(true);

    const panel = host.querySelector('#advanced-controls-panel');
    expect(panel).not.toBeNull();
    expect(toggle?.getAttribute('aria-expanded')).toBe('true');
  });

  it('renders open state directly when advancedOpen is true', async () => {
    useStore.setState({
      advancedOpen: true,
      profile: 'studio',
      mode: 'natural',
    });

    await act(async () => {
      root.render(createElement(AdvancedControls));
    });

    const rootEl = host.querySelector('.advanced-controls.open');
    expect(rootEl).not.toBeNull();
    const panel = host.querySelector('#advanced-controls-panel');
    expect(panel).not.toBeNull();
  });
});
