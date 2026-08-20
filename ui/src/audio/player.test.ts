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

describe('a deck that opens but holds no audio', () => {
  /** What a media element does with a file the decoder accepts and empties. */
  function metadata(index: number, duration: number): void {
    const el = document.querySelectorAll('audio')[index] as HTMLAudioElement;
    Object.defineProperty(el, 'duration', { configurable: true, value: duration });
    Object.defineProperty(el, 'readyState', { configurable: true, value: 4 });
    el.dispatchEvent(new Event('loadedmetadata'));
  }

  it('is a fault: a 100-byte master is readyState 4, MediaError null, duration 0.000396', async () => {
    engine({ [OK]: 206, [GONE]: 200 });
    start();
    player.load('original', OK, 8);
    await settle();
    player.load('cleaned', GONE, 8);
    await settle();
    player.setActive('cleaned');
    expect(player.activeDeck).toBe('cleaned');

    // Chrome accepts the RIFF prefix and reports a real, ready element. Every
    // other signal this class has says the deck is fine — only the length
    // knows, and only because the run said how long the file should be.
    metadata(1, 0.000396);

    expect(player.hasDeck('cleaned')).toBe(false);
    expect(player.activeDeck).toBe('original');
    expect(faults).toHaveLength(1);
    expect(faults[0]).toMatchObject({ deck: 'cleaned', kind: 'truncated', fellBackTo: 'original' });
    expect(faults[0]?.duration).toEqual({ got: 0.000396, expected: 8 });
  });

  it('a zero-length deck is a fault even when nothing said how long it should be', async () => {
    engine({ [OK]: 206, [GONE]: 200 });
    start();
    player.load('original', OK);
    await settle();
    player.load('cleaned', GONE);
    await settle();
    metadata(1, 0);
    expect(faults[0]).toMatchObject({ kind: 'truncated' });
    expect(faults[0]?.duration).toEqual({ got: 0, expected: null });
  });

  it('does not fault a deck that is merely still loading', async () => {
    engine({ [OK]: 206, [GONE]: 200 });
    start();
    player.load('original', OK, 8);
    await settle();
    player.load('cleaned', GONE, 8);
    await settle();
    metadata(1, Number.NaN); // before metadata
    metadata(1, Number.POSITIVE_INFINITY); // a stream with no declared length
    expect(faults).toHaveLength(0);
    metadata(1, 8.0); // and then the real answer
    expect(faults).toHaveLength(0);
    expect(player.isReady('cleaned')).toBe(true);
  });

  it('a deck holding a fraction of the run is a fault even when it is not empty', async () => {
    engine({ [OK]: 206, [GONE]: 200 });
    start();
    player.load('original', OK, 8);
    await settle();
    player.load('cleaned', GONE, 8);
    await settle();
    metadata(1, 2.1); // a quarter of the run: a write that stopped early
    expect(faults[0]).toMatchObject({ kind: 'truncated' });
  });

  it('a deck a little shorter than the run is not a truncated deck', async () => {
    engine({ [OK]: 206, [GONE]: 200 });
    start();
    player.load('original', OK, 94.6);
    await settle();
    player.load('cleaned', GONE, 94.6);
    await settle();
    // Container-vs-decoder timelines differ by milliseconds on this engine.
    metadata(1, 94.5985);
    expect(faults).toHaveLength(0);
  });
});

describe('a file the engine has listed and cannot read', () => {
  /**
   * chmod 000 on a master, measured against the real engine: `GET` answers
   * **200** with a full `Content-Length`, and then the body never arrives
   * (curl exit 18, zero bytes of a ranged read). Headers are not delivery.
   */
  function undeliverable(okStatus: number): void {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (u: string, init?: RequestInit) => {
        if (u === OK) return init?.headers ? ranged(206) : whole();
        const headers = new Headers();
        headers.set('Content-Length', '1152044');
        return {
          ok: okStatus < 400,
          status: okStatus,
          headers,
          arrayBuffer: async () => {
            throw new TypeError('network error');
          },
          blob: async () => {
            throw new TypeError('network error');
          },
        } as unknown as Response;
      }),
    );
  }

  it('is the file’s fault, not the engine’s — and it condemns the master', async () => {
    undeliverable(200);
    start();
    player.load('original', OK, 8);
    await settle();
    player.load('cleaned', GONE, 8);
    await settle();
    // Both probes answer and neither delivers, so the answer is about the
    // file. Reported as `network` it would have said "the engine could not be
    // reached" about a live engine — and, because an outage condemns nothing,
    // would have left `Master WAV` an enabled <a download>.
    expect(faults).toHaveLength(1);
    expect(faults[0]).toMatchObject({ deck: 'cleaned', kind: 'forbidden', status: 200 });
    expect(player.hasDeck('cleaned')).toBe(false);
  });

  it('but an engine that stops answering between the two asks is an outage', async () => {
    let asked = 0;
    vi.stubGlobal(
      'fetch',
      vi.fn(async (u: string, init?: RequestInit) => {
        if (u === OK) return init?.headers ? ranged(206) : whole();
        asked += 1;
        if (asked > 1) throw new TypeError('Failed to fetch'); // it died in between
        const headers = new Headers();
        headers.set('Content-Length', '1152044');
        return {
          ok: true,
          status: 200,
          headers,
          arrayBuffer: async () => {
            throw new TypeError('network error');
          },
        } as unknown as Response;
      }),
    );
    start();
    player.load('original', OK, 8);
    await settle();
    player.load('cleaned', GONE, 8);
    await settle();
    // No fault yet: the element still gets its chance, and the evidence that
    // the engine is gone is remembered for when it fails.
    expect(faults).toHaveLength(0);
    const el = document.querySelectorAll('audio')[1] as HTMLAudioElement;
    Object.defineProperty(el, 'error', { configurable: true, value: { code: 4 } });
    el.dispatchEvent(new Event('error'));
    expect(faults[0]).toMatchObject({ kind: 'network' });
  });
});

describe('an engine that dies while a deck is loading', () => {
  function breakElement(index: number, code: number): void {
    const el = document.querySelectorAll('audio')[index] as HTMLAudioElement;
    Object.defineProperty(el, 'error', { configurable: true, value: { code } });
    el.dispatchEvent(new Event('error'));
  }

  it('is `network`, not `unreadable` — Chromium reports both as code 4', async () => {
    engine({ [OK]: 206 }); // GONE is not in the map: nobody answers the probe
    start();
    player.load('original', OK, 8);
    await settle();
    player.load('cleaned', GONE, 8);
    await settle();
    // The element was still handed the URL, and it fails the way an
    // unreachable engine makes it fail: SRC_NOT_SUPPORTED, exactly the code a
    // PNG renamed `.wav` produces. The probe having thrown is the only
    // evidence that tells the two apart, and it is the evidence that decides.
    breakElement(1, 4);
    expect(faults).toHaveLength(1);
    expect(faults[0]).toMatchObject({ deck: 'cleaned', kind: 'network' });
    expect(player.deckFault('cleaned')?.kind).toBe('network');
  });

  it('takes the injected liveness answer when its own probe saw nothing wrong', async () => {
    engine({ [OK]: 206, [GONE]: 200 });
    start();
    player.setLivenessProbe(() => false); // the store: the engine is offline
    player.load('original', OK, 8);
    await settle();
    player.load('cleaned', GONE, 8);
    await settle();
    breakElement(1, 4);
    expect(faults[0]?.kind).toBe('network');
  });

  it('still calls junk bytes junk while the engine is answering', async () => {
    engine({ [OK]: 206, [GONE]: 200 });
    start();
    player.setLivenessProbe(() => true);
    player.load('original', OK, 8);
    await settle();
    player.load('cleaned', GONE, 8);
    await settle();
    breakElement(1, 4);
    expect(faults[0]?.kind).toBe('unreadable');
  });
});
