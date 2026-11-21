export const RENDERER_PROOF_CONTRACT = 'hawavoclean-react-v1';
export const RENDERER_PROOF_CHALLENGE_EVENT = 'hawavoclean:renderer-proof-challenge:v1';
export const RENDERER_PROOF_RESPONSE_EVENT = 'hawavoclean:renderer-proof-response:v1';

const CHALLENGE = /^[0-9a-f]{64}$/;

/**
 * Install the package-runtime proof responder after React commits App.
 *
 * This is deliberately an unprivileged same-window handshake: it echoes only
 * a random challenge and public build metadata.  It cannot access the engine,
 * files, IPC, or credentials.  The desktop self-test combines the response
 * with structural and computed-layout assertions against the committed App
 * tree, so a loaded preload or a static/blank index page cannot pass.
 */
export function installRendererProofResponder(uiVersion: string): () => void {
  const respond = (event: Event): void => {
    if (!(event instanceof CustomEvent)) return;
    const challenge = event.detail;
    if (typeof challenge !== 'string' || !CHALLENGE.test(challenge)) return;
    window.dispatchEvent(
      new CustomEvent(RENDERER_PROOF_RESPONSE_EVENT, {
        detail: Object.freeze({
          challenge,
          contract: RENDERER_PROOF_CONTRACT,
          uiVersion,
        }),
      }),
    );
  };
  window.addEventListener(RENDERER_PROOF_CHALLENGE_EVENT, respond);
  return () => window.removeEventListener(RENDERER_PROOF_CHALLENGE_EVENT, respond);
}
