import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  installRendererProofResponder,
  RENDERER_PROOF_CHALLENGE_EVENT,
  RENDERER_PROOF_CONTRACT,
  RENDERER_PROOF_RESPONSE_EVENT,
} from './rendererProof';

describe('packaged renderer proof responder', () => {
  const cleanups: Array<() => void> = [];

  afterEach(() => {
    for (const cleanup of cleanups.splice(0)) cleanup();
  });

  it('answers a bounded challenge only after the App-owned responder is installed', () => {
    const challenge = '7a'.repeat(32);
    const responses: unknown[] = [];
    const receive = (event: Event): void => {
      responses.push((event as CustomEvent).detail);
    };
    window.addEventListener(RENDERER_PROOF_RESPONSE_EVENT, receive);
    cleanups.push(() => window.removeEventListener(RENDERER_PROOF_RESPONSE_EVENT, receive));

    window.dispatchEvent(new CustomEvent(RENDERER_PROOF_CHALLENGE_EVENT, { detail: challenge }));
    expect(responses).toEqual([]);

    const uninstall = installRendererProofResponder('3.3.0');
    cleanups.push(uninstall);
    window.dispatchEvent(new CustomEvent(RENDERER_PROOF_CHALLENGE_EVENT, { detail: challenge }));
    expect(responses).toEqual([
      {
        challenge,
        contract: RENDERER_PROOF_CONTRACT,
        uiVersion: '3.3.0',
      },
    ]);

    uninstall();
    window.dispatchEvent(new CustomEvent(RENDERER_PROOF_CHALLENGE_EVENT, { detail: challenge }));
    expect(responses).toHaveLength(1);
  });

  it('ignores malformed or attacker-sized challenges', () => {
    const receive = vi.fn();
    window.addEventListener(RENDERER_PROOF_RESPONSE_EVENT, receive);
    cleanups.push(() => window.removeEventListener(RENDERER_PROOF_RESPONSE_EVENT, receive));
    cleanups.push(installRendererProofResponder('3.3.0'));

    for (const detail of [null, 4, 'a'.repeat(63), 'g'.repeat(64), 'a'.repeat(65)]) {
      window.dispatchEvent(new CustomEvent(RENDERER_PROOF_CHALLENGE_EVENT, { detail }));
    }
    expect(receive).not.toHaveBeenCalled();
  });
});
