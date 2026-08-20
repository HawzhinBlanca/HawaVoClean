// audio/player.ts — the one thing this file must never do again.
//
// The adversarial audit's headline finding was a lie in the audio path: when
// the cleaned deck failed to load, `DualPlayer` fell back to the original and
// told nobody. The A/B control stayed lit on CLEANED with `aria-checked=true`,
// the ORIGINAL element was the one making sound, and there was no error
// anywhere on the screen. These tests pin the two ways a deck can fail — the
// engine answering "no" to the size probe, and the media element itself giving
// up on a source it was handed — and the one thing both of them must do:
// publish the fallback.
//
// Everything that needs a real audio pipeline (the gain cross-fade, the drift
// guard, the analyser) stays in the scripted browser verification. This file
// only holds the state machine around `state`, `active` and `faults`.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { DualPlayer, type DeckFault } from './player';

const OK = 'http://127.0.0.1:8765/api/audio?path=/a.wav';
const GONE = 'http://127.0.0.1:8765/api/audio?path=/out/gone.wav';
const REFUSED = 'http://127.0.0.1:8765/api/audio?path=/etc/passwd';

/** A one-byte ranged answer of the shape `attach()` probes for. */
function ranged(status: number, size = 16): Response {
  const headers = new Headers();
  if (status < 400) {
    headers.set('Content-Range', `bytes 0-0/${size}`);
    headers.set('Content-Length', '1');
  }
  return {
    ok: status < 400,
    status,
    headers,
    arrayBuffer: async () => new ArrayBuffer(1),
    blob: async () => new Blob([new Uint8Array(size)]),
  } as unknown as Response;
}

function whole(size = 16): Response {
  return {
    ok: true,
    status: 200,
    headers: new Headers(),
    arrayBuffer: async () => new ArrayBuffer(size),
    blob: async () => new Blob([new Uint8Array(size)]),
  } as unknown as Response;
}

/**
 * The engine, as far as the player can see it: a map of URL to status. A URL
 * that is not in the map has no engine behind it at all and the fetch throws,
 * which is the offline case, not the deleted case.
 */
function engine(status: Record<string, number>): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init?: RequestInit) => {
      const code = status[url];
      if (code === undefined) throw new TypeError('Failed to fetch');
      if (init?.headers) return ranged(code);
      if (code >= 400) return ranged(code);
      return whole();
    }),
  );
}

/** Let `attach`'s promise chain run to completion. */
async function settle(): Promise<void> {
  for (let i = 0; i < 8; i++) await Promise.resolve();
}

let player: DualPlayer;
let faults: DeckFault[];

beforeEach(() => {
  vi.stubGlobal('URL', {
    ...URL,
    createObjectURL: vi.fn(() => `blob:test-${Math.random().toString(16).slice(2)}`),
    revokeObjectURL: vi.fn(),
  });
  faults = [];
});

afterEach(() => {
  player?.dispose();
  document.querySelectorAll('audio').forEach((el) => el.remove());
});

function start(): void {
  player = new DualPlayer();
  player.onFault((f) => faults.push(f));
}

describe('a deck the engine says is not there', () => {
  it('never reports itself ready, and hands the transport back to the deck that works', async () => {
    engine({ [OK]: 206, [GONE]: 404 });
    start();
    player.load('original', OK);
    await settle();
    player.load('cleaned', GONE);
    // Arriving at the deck was already requested — the exact shape of the real
    // flow, where `setActive('cleaned')` is called while the fetch is in the air.
    player.setActive('cleaned');
    await settle();

    expect(player.isReady('cleaned')).toBe(false);
    expect(player.hasDeck('cleaned')).toBe(false);
    // The fallback, and the thing the audit found missing: it is *published*.
    expect(player.activeDeck).toBe('original');
    expect(player.snapshot().active).toBe('original');
    expect(player.snapshot().pending).toBeNull();
    expect(faults).toHaveLength(1);
    expect(faults[0]).toMatchObject({ deck: 'cleaned', kind: 'missing', status: 404, fellBackTo: 'original' });
    expect(faults[0]?.url).toBe(GONE);
    expect(player.deckFault('cleaned')?.kind).toBe('missing');
  });

  it('tells a refusal apart from a deletion', async () => {
    engine({ [OK]: 206, [REFUSED]: 403 });
    start();
    player.load('original', OK);
    await settle();
    player.load('cleaned', REFUSED);
    await settle();
    expect(player.deckFault('cleaned')?.kind).toBe('forbidden');
  });

  it('says so exactly once, however many times the element complains', async () => {
    engine({ [OK]: 206, [GONE]: 404 });
    start();
    player.load('original', OK);
    await settle();
    player.load('cleaned', GONE);
    await settle();
    player.load('cleaned', GONE); // the UI re-selecting the same dead run
    await settle();
    // Two attempts, two answers — but the *standing* fault is one fact.
    expect(faults.filter((f) => f.deck === 'cleaned')).toHaveLength(2);
    expect(player.deckFault('cleaned')?.kind).toBe('missing');
  });

  it('an engine that never answers is not a file that was deleted', async () => {
    engine({ [OK]: 206 }); // GONE is not in the map: the fetch throws
    start();
    player.load('original', OK);
    await settle();
    player.load('cleaned', GONE);
    await settle();
    // Nothing was refused, so the element is still given the URL and the
    // player waits for the element's own verdict rather than inventing one.
    expect(faults).toHaveLength(0);
    expect(player.isReady('cleaned')).toBe(true);
  });
});

describe('a deck the media element gives up on', () => {
  /** The failure mode of a streamed deck, which never goes through the probe. */
  function breakElement(index: number, code: number): void {
    const el = document.querySelectorAll('audio')[index] as HTMLAudioElement;
    Object.defineProperty(el, 'error', { configurable: true, value: { code } });
    el.dispatchEvent(new Event('error'));
  }

  it('is a fault too — the probe is not the only way a deck can die', async () => {
    engine({ [OK]: 206, [GONE]: 200 }); // the engine says yes, the bytes are junk
    start();
    player.load('original', OK);
    await settle();
    player.load('cleaned', GONE);
    await settle();
    expect(player.isReady('cleaned')).toBe(true);
    player.setActive('cleaned'); // this is the deck being listened to
    expect(player.activeDeck).toBe('cleaned');

    breakElement(1, 4); // MEDIA_ERR_SRC_NOT_SUPPORTED

    expect(player.hasDeck('cleaned')).toBe(false);
    expect(player.activeDeck).toBe('original');
    expect(faults).toHaveLength(1);
    expect(faults[0]).toMatchObject({ deck: 'cleaned', kind: 'unreadable', fellBackTo: 'original' });
  });

  it('a deck nobody was listening to reports no fallback, because there was none', async () => {
    engine({ [OK]: 206, [GONE]: 200 });
    start();
    player.load('original', OK);
    await settle();
    player.load('cleaned', GONE);
    await settle();

    breakElement(1, 4); // the ORIGINAL is live and stays live

    expect(player.activeDeck).toBe('original');
    expect(faults[0]?.fellBackTo).toBeNull();
  });

  it('ignores the abort it caused itself', async () => {
    engine({ [OK]: 206, [GONE]: 200 });
    start();
    player.load('original', OK);
    await settle();
    player.load('cleaned', GONE);
    await settle();

    breakElement(1, 1); // MEDIA_ERR_ABORTED — a src swap, ours

    expect(player.isReady('cleaned')).toBe(true);
    expect(faults).toHaveLength(0);
  });
});

describe('a deck taken out of service', () => {
  it('does not drag the surviving deck back to the top of the take', async () => {
    engine({ [OK]: 206, [GONE]: 200 });
    start();
    player.load('original', OK);
    await settle();
    player.load('cleaned', GONE);
    await settle();
    const els = document.querySelectorAll('audio');
    const orig = els[0] as HTMLAudioElement;
    const clean = els[1] as HTMLAudioElement;
    orig.currentTime = 40.25;
    clean.currentTime = 40.25;
    player.setActive('cleaned');
    expect(player.activeDeck).toBe('cleaned');
    // A deck that never played reads 0. Sample-locking the survivor to a dead
    // element's clock would throw the transport back to the start.
    Object.defineProperty(clean, 'currentTime', { configurable: true, value: 0 });

    Object.defineProperty(clean, 'error', { configurable: true, value: { code: 4 } });
    clean.dispatchEvent(new Event('error'));

    expect(player.activeDeck).toBe('original');
    expect(orig.currentTime).toBeCloseTo(40.25, 3);
  });

  it('is asked for again cleanly once the file comes back', async () => {
    engine({ [OK]: 206, [GONE]: 404 });
    start();
    player.load('original', OK);
    await settle();
    player.load('cleaned', GONE);
    await settle();
    expect(player.deckFault('cleaned')).not.toBeNull();

    engine({ [OK]: 206, [GONE]: 206 }); // the file is back
    player.load('cleaned', GONE);
    // The fault is cleared the moment the deck is asked for, not when it lands:
    // the standing answer is no longer current.
    expect(player.deckFault('cleaned')).toBeNull();
    await settle();
    expect(player.isReady('cleaned')).toBe(true);
    player.setActive('cleaned');
    expect(player.activeDeck).toBe('cleaned');
  });
});
