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

// --- the A/B contract -------------------------------------------------------
//
// Everything above is a fault path. The happy path — setActive, play, seek, and
// the sample-lock that makes A/B a comparison rather than two unrelated
// playbacks — had no test at all, which is an odd gap in a file whose headline
// bug was the A/B control lying about which deck was audible.
//
// These pin the state machine only. The gain cross-fade, the drift guard and
// the analyser need a real audio pipeline and stay in the scripted browser
// verification.

/** The `<audio>` element behind a deck, so a test can drive what happy-dom will not. */
function el(deck: 'original' | 'cleaned'): HTMLMediaElement {
  const all = [...document.querySelectorAll('audio')];
  const found = all[deck === 'original' ? 0 : 1];
  if (!found) throw new Error(`no <audio> for ${deck}`);
  return found as HTMLMediaElement;
}

/** Both decks loaded and ready, `dur` seconds long. */
async function bothReady(dur = 60): Promise<void> {
  engine({ [OK]: 206, [GONE]: 206 });
  start();
  player.load('original', OK);
  player.load('cleaned', GONE);
  await settle();
  for (const d of ['original', 'cleaned'] as const) {
    Object.defineProperty(el(d), 'duration', { configurable: true, value: dur });
  }
}

describe('A/B keeps the two decks on the same sample', () => {
  it('corrects the arriving deck when the two have drifted', async () => {
    await bothReady();
    el('original').currentTime = 12.5;
    el('cleaned').currentTime = 3;
    player.setActive('cleaned');
    // Beyond SYNC_TOLERANCE_S (0.035), so the arriving deck is pulled to where
    // the leaving one was. Without this, A/B compares two different moments —
    // which is not a comparison at all.
    expect(el('cleaned').currentTime).toBeCloseTo(12.5, 3);
    expect(player.activeDeck).toBe('cleaned');
  });

  it('leaves a deck already within tolerance alone', async () => {
    await bothReady();
    el('original').currentTime = 12.5;
    el('cleaned').currentTime = 12.52; // 20 ms, inside the 35 ms tolerance
    player.setActive('cleaned');
    // A needless `currentTime` write costs a reseek on a real element, which is
    // audible. The tolerance exists so the normal case is free.
    expect(el('cleaned').currentTime).toBeCloseTo(12.52, 3);
  });

  it('a seek moves every ready deck, not just the audible one', async () => {
    await bothReady();
    player.seek(30);
    expect(el('original').currentTime).toBeCloseTo(30, 3);
    expect(el('cleaned').currentTime).toBeCloseTo(30, 3);
    // …so switching afterwards does not jump.
    player.setActive('cleaned');
    expect(el('cleaned').currentTime).toBeCloseTo(30, 3);
  });

  it('clamps a seek to the clip', async () => {
    await bothReady(60);
    player.seek(-5);
    expect(el('original').currentTime).toBe(0);
    player.seek(999);
    expect(el('original').currentTime).toBeCloseTo(60, 3);
  });
});

describe('A/B cannot point at a deck that is not there', () => {
  it('refuses to arrive at an empty deck', async () => {
    engine({ [OK]: 206 });
    start();
    player.load('original', OK);
    await settle();
    player.setActive('cleaned'); // never loaded
    // This is the shape of the audit's headline bug: the control lit CLEANED
    // while ORIGINAL was making the sound.
    expect(player.activeDeck).toBe('original');
    expect(player.snapshot().active).toBe('original');
  });

  it('waits for a deck that is still loading, then arrives', async () => {
    engine({ [OK]: 206, [GONE]: 206 });
    start();
    player.load('original', OK);
    await settle();
    player.load('cleaned', GONE);
    player.setActive('cleaned'); // asked for while the fetch is in the air
    expect(player.activeDeck).toBe('original');
    expect(player.snapshot().pending).toBe('cleaned');
    await settle();
    expect(player.activeDeck).toBe('cleaned');
    expect(player.snapshot().pending).toBeNull();
  });

  it('is a no-op when asked for the deck already active', async () => {
    await bothReady();
    el('original').currentTime = 5;
    player.setActive('original');
    expect(player.activeDeck).toBe('original');
    expect(el('original').currentTime).toBeCloseTo(5, 3);
  });
});

describe('transport', () => {
  it('play starts both ready decks from the active one’s position', async () => {
    await bothReady();
    el('original').currentTime = 20;
    el('cleaned').currentTime = 0;
    await player.play();
    // The inactive deck is pulled into line *before* it starts, so an A/B
    // press mid-playback is instant rather than a seek.
    expect(el('cleaned').currentTime).toBeCloseTo(20, 3);
  });

  it('play on a deck that is not ready does nothing', async () => {
    engine({ [OK]: 404 });
    start();
    player.load('original', OK);
    await settle();
    await player.play();
    expect(player.playing).toBe(false);
  });

  it('toggle flips, and pause is remembered across an A/B switch', async () => {
    await bothReady();
    await player.play();
    player.toggle();
    expect(player.playing).toBe(false);
    player.setActive('cleaned');
    // `wantPlaying` is false, so arriving at a deck must not start it.
    expect(player.playing).toBe(false);
  });
});

describe('duration and time', () => {
  it('falls back to the other deck when the active one has no duration', async () => {
    await bothReady(60);
    Object.defineProperty(el('original'), 'duration', { configurable: true, value: NaN });
    // A run's length is a property of the run, not of whichever element
    // happens to have finished parsing its header first.
    expect(player.duration).toBeCloseTo(60, 3);
  });

  it('reports 0 rather than NaN for an element with no position yet', async () => {
    await bothReady();
    Object.defineProperty(el('original'), 'currentTime', { configurable: true, value: NaN });
    // NaN here reaches the timecode readout and the waveform playhead.
    expect(player.time).toBe(0);
  });
});

describe('a webview with no WebAudio', () => {
  it('still plays, losing the cross-fade rather than the sound', async () => {
    // happy-dom has no AudioContext, which is exactly the case: `ensureGraph`
    // used to throw out of `play()`, and because `toggle()` calls
    // `void this.play()` the rejection landed in the app's failure net as an
    // unexplained error while the transport silently did nothing. Element-level
    // mute is the pre-graph A/B mechanism and is still there.
    expect(window.AudioContext).toBeUndefined();
    await bothReady();
    await player.play();
    expect(player.playing).toBe(true);
    expect(player.getAnalyser()).toBeNull();

    // And A/B still switches which element is audible.
    player.setActive('cleaned');
    expect(player.activeDeck).toBe('cleaned');
    expect(el('cleaned').muted).toBe(false);
    expect(el('original').muted).toBe(true);
  });
});

describe('loudness matching (D4.2)', () => {
  it('computes level delta between original and cleaned and clamps to [-12, +12] dB', async () => {
    await bothReady();
    // Case 1: Cleaned is 2 dB louder (-18 LUFS vs -20 LUFS)
    player.setLoudnessMatch(true, -20, -18);
    expect(player.gainOffsetDb).toBeCloseTo(-2.0, 3);
    expect(player.getTargetGain('cleaned')).toBeCloseTo(Math.pow(10, -2 / 20), 3);
    expect(player.getTargetGain('original')).toBe(1.0);
    expect(player.snapshot().loudnessMatch).toBe(true);
    expect(player.snapshot().gainOffsetDb).toBeCloseTo(-2.0, 3);

    // Case 2: Cleaned is quieter by 3 dB (-21 LUFS vs -24 LUFS)
    player.setLoudnessMatch(true, -21, -24);
    expect(player.gainOffsetDb).toBeCloseTo(3.0, 3);
    expect(player.getTargetGain('cleaned')).toBeCloseTo(Math.pow(10, 3 / 20), 3);

    // Case 3: Extreme delta clamped to -12 dB
    player.setLoudnessMatch(true, -30, -10);
    expect(player.gainOffsetDb).toBe(-12);
    expect(player.getTargetGain('cleaned')).toBeCloseTo(Math.pow(10, -12 / 20), 3);

    // Case 4: Extreme delta clamped to +12 dB
    player.setLoudnessMatch(true, -10, -30);
    expect(player.gainOffsetDb).toBe(12);
    expect(player.getTargetGain('cleaned')).toBeCloseTo(Math.pow(10, 12 / 20), 3);

    // Case 5: Disabled -> unity gain (0 dB delta)
    player.setLoudnessMatch(false, -20, -18);
    expect(player.gainOffsetDb).toBe(0);
    expect(player.getTargetGain('cleaned')).toBe(1.0);
    expect(player.snapshot().loudnessMatch).toBe(false);

    // Case 6: Missing or null LUFS -> unity gain (0 dB delta)
    player.setLoudnessMatch(true, null, -18);
    expect(player.gainOffsetDb).toBe(0);
    expect(player.getTargetGain('cleaned')).toBe(1.0);
  });
});

describe('common-region time bounding (D4.2)', () => {
  it('bounds duration to the shorter deck when both are ready', async () => {
    await bothReady();
    Object.defineProperty(el('original'), 'duration', { configurable: true, value: 60 });
    Object.defineProperty(el('cleaned'), 'duration', { configurable: true, value: 45 });

    // Duration must never exceed the shorter deck during A/B comparison
    expect(player.duration).toBeCloseTo(45, 3);

    // Seeking past the shorter duration must be clamped to the common boundary
    player.seek(55);
    expect(el('original').currentTime).toBeCloseTo(45, 3);
    expect(el('cleaned').currentTime).toBeCloseTo(45, 3);

    // Switching active deck at the boundary clamps both decks
    player.setActive('cleaned');
    expect(el('cleaned').currentTime).toBeLessThanOrEqual(45);
    expect(player.activeDeck).toBe('cleaned');
  });

  it('reports the active deck duration when only one deck is ready', async () => {
    engine({ [OK]: 206 });
    start();
    player.load('original', OK);
    await settle();
    Object.defineProperty(el('original'), 'duration', { configurable: true, value: 32 });

    expect(player.duration).toBeCloseTo(32, 3);
  });
});

// --- Pro Audition Engine: Delta, Loop, Wet/Dry Mix, Live LUFS/Peak Meter ---

interface MockGainNode {
  gain: {
    value: number;
    lastTarget: number;
    cancelScheduledValues: ReturnType<typeof vi.fn>;
    setValueAtTime: ReturnType<typeof vi.fn>;
    linearRampToValueAtTime: ReturnType<typeof vi.fn>;
  };
  connect: ReturnType<typeof vi.fn>;
  disconnect: ReturnType<typeof vi.fn>;
}

function installMockWebAudio(timeDomainSamples?: Float32Array) {
  const gainNodes: MockGainNode[] = [];
  let samples = timeDomainSamples ?? new Float32Array(2048);

  const analyser = {
    fftSize: samples.length,
    smoothingTimeConstant: 0.78,
    minDecibels: -100,
    maxDecibels: 0,
    connect: vi.fn(),
    disconnect: vi.fn(),
    getFloatTimeDomainData: vi.fn((target: Float32Array) => {
      const n = Math.min(target.length, samples.length);
      for (let i = 0; i < n; i++) target[i] = samples[i] ?? 0;
      for (let i = n; i < target.length; i++) target[i] = 0;
    }),
  };

  class MockAudioContext {
    currentTime = 1.0;
    sampleRate = 48000;
    state: AudioContextState = 'running';
    destination = {};
    resume = vi.fn(async () => undefined);
    close = vi.fn(async () => undefined);
    createAnalyser = vi.fn(() => analyser);
    createMediaElementSource = vi.fn(() => ({ connect: vi.fn(), disconnect: vi.fn() }));
    createGain = vi.fn(() => {
      const param = {
        value: 1,
        lastTarget: 1,
        cancelScheduledValues: vi.fn(),
        setValueAtTime: vi.fn((v: number) => {
          param.value = v;
        }),
        linearRampToValueAtTime: vi.fn((v: number) => {
          param.value = v;
          param.lastTarget = v;
        }),
      };
      const node: MockGainNode = {
        gain: param,
        connect: vi.fn(),
        disconnect: vi.fn(),
      };
      gainNodes.push(node);
      return node;
    });
  }

  vi.stubGlobal('AudioContext', MockAudioContext);
  return {
    gainNodes,
    analyser,
    setSamples(next: Float32Array) {
      samples = next;
    },
  };
}

describe('pro audition engine — Delta ("Listen to Removed")', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('requires both decks ready and sums Original (+1) + Cleaned (-gainOffset) in WebAudio', async () => {
    const audio = installMockWebAudio();
    await bothReady(30);
    player.setLoudnessMatch(true, -20, -18); // -2 dB on cleaned
    await player.play();

    // Gain nodes created in order: original, cleaned, deltaInverter (-1), deltaGain
    expect(audio.gainNodes).toHaveLength(4);
    const [origGain, cleanGain, deltaInverter, deltaGain] = audio.gainNodes;
    expect(deltaInverter?.gain.value).toBe(-1);

    player.setDeltaMode(true);
    expect(player.snapshot().deltaMode).toBe(true);
    expect(origGain?.gain.value).toBeCloseTo(1.0, 4);
    expect(cleanGain?.gain.value).toBeCloseTo(0, 4);
    expect(deltaGain?.gain.value).toBeCloseTo(Math.pow(10, -2 / 20), 4);

    // Toggling off restores active deck ('original')
    player.toggleDeltaMode();
    expect(player.snapshot().deltaMode).toBe(false);
    expect(origGain?.gain.value).toBeCloseTo(1.0, 4);
    expect(deltaGain?.gain.value).toBeCloseTo(0, 4);
  });

  it('refuses Delta mode when only one deck is ready and auto-disables on deck retire', async () => {
    engine({ [OK]: 206 });
    start();
    player.load('original', OK);
    await settle();

    player.setDeltaMode(true);
    expect(player.snapshot().deltaMode).toBe(false);

    // Load second deck -> delta works; retire second deck -> delta disables cleanly
    engine({ [OK]: 206, [GONE]: 206 });
    player.load('cleaned', GONE);
    await settle();
    player.setDeltaMode(true);
    expect(player.snapshot().deltaMode).toBe(true);

    player.load('cleaned', null);
    expect(player.snapshot().deltaMode).toBe(false);
  });

  it('fails soft without throwing when WebAudio is unavailable', async () => {
    expect(window.AudioContext).toBeUndefined();
    await bothReady(30);
    await player.play();
    expect(() => player.setDeltaMode(true)).not.toThrow();
    expect(player.snapshot().deltaMode).toBe(true);
    expect(el('original').muted).toBe(false);
    expect(el('cleaned').muted).toBe(true);
  });
});

describe('pro audition engine — Loop Region playback', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('seeks both decks back to loopStart when playback reaches loopEnd', async () => {
    const rafQueue: FrameRequestCallback[] = [];
    vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => {
      rafQueue.push(cb);
      return rafQueue.length;
    });
    vi.stubGlobal('cancelAnimationFrame', vi.fn());

    await bothReady(30);
    player.setLoopRange([4.5, 10.0]);
    player.setLoopEnabled(true);
    expect(player.snapshot().loopEnabled).toBe(true);
    expect(player.snapshot().loopRange).toEqual([4.5, 10.0]);

    await player.play();
    // Play snaps inside loopRange if starting at 0
    expect(el('original').currentTime).toBeCloseTo(4.5, 3);
    expect(el('cleaned').currentTime).toBeCloseTo(4.5, 3);

    // Advance past loopEnd and pump one animation frame
    el('original').currentTime = 10.05;
    el('cleaned').currentTime = 10.05;
    const cb = rafQueue.shift();
    cb?.(16);

    expect(el('original').currentTime).toBeCloseTo(4.5, 3);
    expect(el('cleaned').currentTime).toBeCloseTo(4.5, 3);
    expect(player.playing).toBe(true);
  });

  it('loops full clip on ended when loopRange is null and sanitizes invalid ranges', async () => {
    await bothReady(20);
    player.setLoopRange([10, 5]); // invalid: end <= start
    expect(player.snapshot().loopRange).toBeNull();

    player.setLoopEnabled(true);
    await player.play();
    el('original').currentTime = 20;
    el('cleaned').currentTime = 20;
    el('original').dispatchEvent(new Event('ended'));

    expect(el('original').currentTime).toBe(0);
    expect(el('cleaned').currentTime).toBe(0);
  });
});

describe('pro audition engine — Live Wet/Dry Mix Preview', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('blends Original (1 - mix) and Cleaned (mix * levelMatchGain) when on cleaned deck', async () => {
    const audio = installMockWebAudio();
    await bothReady(30);
    player.setLoudnessMatch(true, -18, -24); // +6 dB on cleaned
    const expectedGain = Math.pow(10, 6 / 20);
    await player.play();

    const [origGain, cleanGain, , deltaGain] = audio.gainNodes;
    player.setActive('cleaned');
    player.setWetDryMix(0.35);

    expect(player.snapshot().wetDryMix).toBeCloseTo(0.35, 4);
    expect(origGain?.gain.value).toBeCloseTo(0.65, 4);
    expect(cleanGain?.gain.value).toBeCloseTo(0.35 * expectedGain, 4);
    expect(deltaGain?.gain.value).toBe(0);

    // Clamping outside [0, 1]
    player.setWetDryMix(2.5);
    expect(player.snapshot().wetDryMix).toBe(1);
    player.setWetDryMix(-0.5);
    expect(player.snapshot().wetDryMix).toBe(0);
  });
});

describe('pro audition engine — Live LUFS + Peak Meter', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('returns clean finite floor when no AudioContext exists', async () => {
    expect(window.AudioContext).toBeUndefined();
    await bothReady(10);
    const levels = player.getLiveMeterLevels();
    expect(Number.isFinite(levels.momentaryLufs)).toBe(true);
    expect(Number.isFinite(levels.shortTermLufs)).toBe(true);
    expect(Number.isFinite(levels.peakDbfs)).toBe(true);
    expect(Number.isFinite(levels.truePeakHoldDbfs)).toBe(true);
    expect(levels.momentaryLufs).toBe(-70);
    expect(levels.peakDbfs).toBe(-70);
  });

  it('computes LUFS, peak dBFS, and peak hold from AnalyserNode and resets peak hold', async () => {
    // 1 kHz sine at 0.5 amplitude (-6.02 dBFS peak, mean square = 0.125 -> ~-9.72 LUFS)
    const buf = new Float32Array(8192);
    for (let i = 0; i < buf.length; i++) {
      buf[i] = 0.5 * Math.sin((2 * Math.PI * 1000 * i) / 48000);
    }
    const audio = installMockWebAudio(buf);
    await bothReady(10);
    await player.play();

    const loud = player.getLiveMeterLevels();
    expect(loud.peakDbfs).toBeCloseTo(20 * Math.log10(0.5), 1);
    expect(loud.momentaryLufs).toBeCloseTo(-0.691 + 10 * Math.log10(0.125), 1);
    expect(loud.shortTermLufs).toBeCloseTo(-0.691 + 10 * Math.log10(0.125), 1);
    expect(loud.truePeakHoldDbfs).toBeGreaterThanOrEqual(loud.peakDbfs - 0.05);

    // Silence after loud frame: peak drops, peak hold stays latched
    audio.setSamples(new Float32Array(8192));
    const quiet = player.getLiveMeterLevels();
    expect(quiet.peakDbfs).toBe(-70);
    expect(quiet.truePeakHoldDbfs).toBeCloseTo(loud.truePeakHoldDbfs, 1);

    // Resetting peak hold clears latch back to floor
    player.resetPeakHold();
    const reset = player.getLiveMeterLevels();
    expect(reset.truePeakHoldDbfs).toBe(-70);
  });
});

describe('active-deck reload continuity & region preview offset alignment', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('retains wantPlaying and seeks to pending position when reloading active cleaned deck', async () => {
    await bothReady(30);
    player.setActive('cleaned');
    player.seek(5.5);
    await player.play();
    expect(player.playing).toBe(true);
    expect(el('cleaned').currentTime).toBeCloseTo(5.5, 2);

    // Reload active 'cleaned' deck with cache-busted re-stitched master URL
    const RESTITCHED = 'http://127.0.0.1:8765/api/audio?path=/out/master.wav&v=abc';
    engine({ [OK]: 206, [RESTITCHED]: 206 });
    const cleanEl = el('cleaned');
    const playSpy = vi.spyOn(cleanEl, 'play');

    player.load('cleaned', RESTITCHED, 30);
    // Immediate seek + play while deck is still 'loading' (as actions.ts does)
    player.seek(5.5);
    await player.play();

    // Simulate browser resetting src to 0 & paused when setSrc runs, then wait for attach
    await settle();

    expect(player.isReady('cleaned')).toBe(true);
    expect(player.activeDeck).toBe('cleaned');
    expect(cleanEl.currentTime).toBeCloseTo(5.5, 2);
    expect(playSpy).toHaveBeenCalled();
    expect(player.playing).toBe(true);
  });

  it('keeps full duration, loop bounds, and A/B alignment when cleaned deck holds an offset region preview', async () => {
    const rafQueue: FrameRequestCallback[] = [];
    vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => {
      rafQueue.push(cb);
      return rafQueue.length;
    });
    vi.stubGlobal('cancelAnimationFrame', vi.fn());

    const PREVIEW = 'http://127.0.0.1:8765/api/audio?path=/tmp/preview_9_12.wav';
    engine({ [OK]: 206, [PREVIEW]: 206 });
    start();
    player.load('original', OK, 12);
    await settle();
    Object.defineProperty(el('original'), 'duration', { configurable: true, value: 12 });

    // Load 3s preview of [9s, 12s] onto cleaned deck with regionOffsetS = 9
    player.load('cleaned', PREVIEW, 3, 9);
    player.setLoopRange([9, 12]);
    player.setLoopEnabled(true);
    player.seek(9);
    player.setActive('cleaned');
    await settle();
    Object.defineProperty(el('cleaned'), 'duration', { configurable: true, value: 3 });

    // Full timeline duration must NOT collapse to 3s
    expect(player.duration).toBeCloseTo(12, 3);
    expect(player.activeDeck).toBe('cleaned');
    // Timeline t=9.0s maps to original=9.0s, cleaned=0.0s
    expect(el('original').currentTime).toBeCloseTo(9.0, 2);
    expect(el('cleaned').currentTime).toBeCloseTo(0.0, 2);
    expect(player.time).toBeCloseTo(9.0, 2);

    // Advance 1.5s into preview (cleaned=1.5 -> timeline=10.5) and switch to original
    el('cleaned').currentTime = 1.5;
    player.setActive('original');
    expect(el('original').currentTime).toBeCloseTo(10.5, 2);

    // Toggling Delta during an offset preview must preserve region-offset phase alignment
    player.setActive('cleaned');
    el('cleaned').currentTime = 1.5;
    el('original').currentTime = 0; // force drift
    player.setDeltaMode(true);
    expect(el('original').currentTime).toBeCloseTo(10.5, 2);
    expect(el('cleaned').currentTime).toBeCloseTo(1.5, 2);
    player.setDeltaMode(false);

    // Play and advance past loopEnd (12.0s) -> loops both decks back to 9.0 / 0.0
    player.setActive('cleaned');
    await player.play();
    el('cleaned').currentTime = 3.05; // timeline 12.05 >= loopEnd 12.0
    const cb = rafQueue.shift();
    cb?.(16);
    expect(el('original').currentTime).toBeCloseTo(9.0, 2);
    expect(el('cleaned').currentTime).toBeCloseTo(0.0, 2);
  });

  it('applies real-time 24 dB/oct Butterworth HPF on Original audition and bypasses on Cleaned/Delta', async () => {
    const biquads: { type: string; frequency: { value: number; setTargetAtTime: ReturnType<typeof vi.fn> }; Q: { value: number }; connect: ReturnType<typeof vi.fn>; disconnect: ReturnType<typeof vi.fn> }[] = [];
    const gains: { gain: { value: number; cancelScheduledValues: ReturnType<typeof vi.fn>; setValueAtTime: ReturnType<typeof vi.fn>; linearRampToValueAtTime: ReturnType<typeof vi.fn> }; connect: ReturnType<typeof vi.fn>; disconnect: ReturnType<typeof vi.fn> }[] = [];
    class MockCtxWithBiquad {
      state = 'running';
      currentTime = 0;
      sampleRate = 48000;
      destination = {};
      resume = vi.fn(async () => undefined);
      close = vi.fn(async () => { this.state = 'closed'; });
      createGain() {
        const g = {
          gain: {
            value: 1,
            cancelScheduledValues: vi.fn(),
            setValueAtTime: vi.fn(),
            linearRampToValueAtTime: vi.fn((v: number) => { g.gain.value = v; }),
          },
          connect: vi.fn(),
          disconnect: vi.fn(),
        };
        gains.push(g);
        return g;
      }
      createBiquadFilter() {
        const f = {
          type: 'lowpass',
          frequency: { value: 350, setTargetAtTime: vi.fn((v: number) => { f.frequency.value = v; }) },
          gain: { value: 0 },
          Q: { value: 1 },
          connect: vi.fn(),
          disconnect: vi.fn(),
        };
        biquads.push(f);
        return f;
      }
      createAnalyser() {
        return { fftSize: 2048, smoothingTimeConstant: 0.8, minDecibels: -100, maxDecibels: 0, connect: vi.fn(), disconnect: vi.fn(), getFloatTimeDomainData: vi.fn() };
      }
      createMediaElementSource() {
        return { connect: vi.fn(), disconnect: vi.fn() };
      }
    }
    vi.stubGlobal('AudioContext', MockCtxWithBiquad);
    await bothReady(10);
    await player.play();

    // First two biquads are the cascaded 24 dB/oct Butterworth HPF stages (Q=0.5412, Q=1.3066)
    expect(biquads.length).toBeGreaterThanOrEqual(2);
    expect(biquads[0]?.type).toBe('highpass');
    expect(biquads[1]?.type).toBe('highpass');

    player.setLiveHpf(true, 120);
    expect(biquads[0]?.frequency.value).toBe(120);
    expect(biquads[1]?.frequency.value).toBe(120);

    // Switching to Cleaned or Delta must automatically bypass the live HPF (dry=1, wet=0)
    // gains order: [0]=original, [1]=cleaned, [2]=deltaInverter, [3]=deltaGain, [4]=hpfDry, [5]=hpfWet
    expect(gains[4]?.gain.value).toBe(0);
    expect(gains[5]?.gain.value).toBe(1);

    player.setActive('cleaned');
    expect(gains[4]?.gain.value).toBe(1);
    expect(gains[5]?.gain.value).toBe(0);

    player.setActive('original');
    expect(gains[4]?.gain.value).toBe(0);
    expect(gains[5]?.gain.value).toBe(1);

    player.setDeltaMode(true);
    expect(gains[4]?.gain.value).toBe(1);
    expect(gains[5]?.gain.value).toBe(0);
  });
});
