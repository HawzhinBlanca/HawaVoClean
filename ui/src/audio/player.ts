// Dual-deck player: ORIGINAL and CLEANED <audio> elements run in lock-step so
// the A/B switch is instantaneous (a 6 ms gain cross-fade, no restart). Both
// feed one AnalyserNode for the live spectrum overlay.
//
// ---------------------------------------------------------------- C4 note
// A media element that streams `http(s)` audio is a network client we do not
// control, and every time Chromium's media stack stops reading a response it
// *aborts* the connection — which the network panel records as
// `net::ERR_ABORTED`. Measured on this engine (13.6 MB master, `preload=auto`,
// element paused and never touched by script): Chromium buffers 21.8 s, goes
// `NETWORK_IDLE`, and the open range request turns into ERR_ABORTED a few
// seconds later. A seek outside the buffered range opens a second request that
// meets the same end. None of that is ours, and none of it can be prevented
// from script while the element is fetching over HTTP.
//
// So a deck does not stream unless it has to. `load()` asks the engine for the
// size (HEAD, no body), and anything at or below MEMORY_DECK_MAX_BYTES is
// pulled down with one ordinary `fetch` that always runs to completion and
// handed to the element as a `blob:` URL. From then on the deck does zero
// network I/O: seeks are free, A/B switching is free, and there is nothing left
// for the media stack to abort. Files above the cap keep streaming (pulling a
// 1 GB master into memory is not a trade worth making) and their aborts are
// browser-internal and documented, not ours.
//
// The rules that remain ours, and that this file keeps:
//   * a deck's `src` is never *removed*. Retiring a deck pauses and mutes it
//     and marks it unavailable; the attribute stays exactly as it was, so no
//     in-flight request is torn down by us.
//   * a deck's `src` is only ever *replaced* when a genuinely different file
//     arrives for that deck.
//   * a superseded in-flight fetch is dropped by epoch check, never aborted.

export type Deck = 'original' | 'cleaned';

export interface PlayerSnapshot {
  playing: boolean;
  time: number;
  duration: number;
  /** The deck that is actually audible right now. */
  active: Deck;
  /** A deck that has been asked for but is still fetching, or null. */
  pending: Deck | null;
  /** Whether level-matching gain adjustment is active. */
  loudnessMatch: boolean;
  /** Gain offset in dB applied to Cleaned deck during level matching. */
  gainOffsetDb: number;
}

/**
 * Why a deck is out of service.
 *
 * `missing`    the engine answered 404 — the file is not there any more
 * `forbidden`  the engine refused to serve it (403, path policy)
 * `unreadable` the bytes arrived but no decoder would take them
 * `truncated`  a decoder took them and found no audio worth the name — a
 *              length of zero, or a small fraction of the length the run says
 *              this file has. Chrome accepts a RIFF header with nothing behind
 *              it: `readyState` 4, `MediaError` null, `duration` 0.000396 s.
 *              Left alone that is the worst deck of all, because every other
 *              signal says the deck is fine and pressing Play does nothing.
 * `network`    the engine could not be reached at all
 */
export type DeckFaultKind = 'missing' | 'forbidden' | 'unreadable' | 'truncated' | 'network';

/**
 * A deck that was asked for and could not be played.
 *
 * This is the one thing the player cannot keep to itself. Falling back to the
 * other deck silently is what made the A/B control state something untrue —
 * the switch stayed lit on CLEANED while the ORIGINAL element was the one
 * making sound — so every fallback is published here and the store mirrors it.
 */
export interface DeckFault {
  deck: Deck;
  /** The engine URL this deck was asked to hold when it failed. */
  url: string;
  kind: DeckFaultKind;
  /** The HTTP status when the engine answered, else null. */
  status: number | null;
  /** The deck that took over, or null when there was nothing to fall back to. */
  fellBackTo: Deck | null;
  /** For `truncated`: what the element read, and what the run says it should be. */
  duration?: { got: number; expected: number | null };
}

type Listener = (snap: PlayerSnapshot) => void;
type FaultListener = (fault: DeckFault) => void;

/**
 * `empty`   nothing playable on this deck (it may still hold a retired src)
 * `loading` a source has been requested and is being fetched
 * `ready`   the element has a src it can play
 */
type DeckState = 'empty' | 'loading' | 'ready';

const FADE_S = 0.006;
const SYNC_TOLERANCE_S = 0.035;

/**
 * Ceiling for pulling a deck into memory instead of streaming it. 128 MB is
 * about 25 minutes of a 48 kHz/24-bit stereo master, or several hours of AAC —
 * i.e. every clip this tool is actually pointed at during a session — while
 * still refusing to buffer a feature-length file into the tab.
 */
const MEMORY_DECK_MAX_BYTES = 128 * 1024 * 1024;

/** `MediaError.code` values, named (the spec's constants live on the class). */
const MEDIA_ERR_ABORTED = 1;
const MEDIA_ERR_NETWORK = 2;

/**
 * How short a deck may be before it is a fault rather than a file.
 *
 * Two rules, and both are needed. A deck of *zero* length can make no sound
 * whatever the run says about it. A deck that is a small fraction of the
 * length the run reports is not the file that run wrote: the container
 * timelines this app deals with differ by milliseconds at most (measured: 1.46
 * ms over a 95 s lossy clip), so half is orders of magnitude outside anything
 * legitimate and cannot fire on a healthy master.
 */
const DECK_MIN_DURATION_S = 0.05;
const DECK_MIN_DURATION_RATIO = 0.5;

export class DualPlayer {
  private readonly els: Record<Deck, HTMLAudioElement>;
  private ctx: AudioContext | null = null;
  private gains: Record<Deck, GainNode> | null = null;
  private analyser: AnalyserNode | null = null;
  private active: Deck = 'original';
  private state: Record<Deck, DeckState> = { original: 'empty', cleaned: 'empty' };
  /** The engine URL the deck has been asked to hold (null = retired). */
  private wanted: Record<Deck, string | null> = { original: null, cleaned: null };
  /** The engine URL currently attached to the element, blob or not. */
  private attached: Record<Deck, string | null> = { original: null, cleaned: null };
  /** The `blob:` URL handed to the element, so it can be revoked exactly once. */
  private objectUrl: Record<Deck, string | null> = { original: null, cleaned: null };
  /** Bumped on every load/retire so a slow fetch can be dropped, not aborted. */
  private epoch: Record<Deck, number> = { original: 0, cleaned: 0 };
  /**
   * How long the run says this deck's file is, when the caller knows. The
   * element's own `duration` is checked against it — see {@link DeckFaultKind}
   * `truncated`.
   */
  private expected: Record<Deck, number | null> = { original: null, cleaned: null };
  /** One duration verdict per load; `durationchange` fires many times. */
  private durationJudged: Record<Deck, boolean> = { original: false, cleaned: false };
  /**
   * The size probe threw rather than answering. That is the engine not being
   * reachable at all — evidence the element's own `MediaError` does not carry,
   * because Chromium reports a src it could not fetch as code 4
   * (`SRC_NOT_SUPPORTED`), which is indistinguishable from junk bytes.
   */
  private probeFailed: Record<Deck, boolean> = { original: false, cleaned: false };
  /** Injected: is the engine answering at all right now? (see setLivenessProbe) */
  private engineLive: (() => boolean) | null = null;
  /** A `setActive` that arrived while the deck was still fetching. */
  private pendingActive: Deck | null = null;
  /** The last published fault per deck, cleared when the deck is asked for again. */
  private faults: Record<Deck, DeckFault | null> = { original: null, cleaned: null };
  private listeners = new Set<Listener>();
  private faultListeners = new Set<FaultListener>();
  private raf = 0;
  private wantPlaying = false;
  private loudnessMatchEnabled = true;
  private lufsIn: number | null = null;
  private lufsOut: number | null = null;

  constructor() {
    this.els = {
      original: this.makeElement(),
      cleaned: this.makeElement(),
    };
    for (const deck of ['original', 'cleaned'] as const) {
      const el = this.els[deck];
      el.addEventListener('ended', () => {
        if (deck === this.active) {
          this.wantPlaying = false;
          this.pauseAll();
          this.emit();
        }
      });
      el.addEventListener('loadedmetadata', () => {
        this.judgeDuration(deck);
        this.emit();
      });
      // A deck can also arrive at its real length later than its metadata.
      el.addEventListener('durationchange', () => this.judgeDuration(deck));
      // A media element that cannot play what it was given is a deck that is
      // out of service, not a line in a log. The size probe in `attach` cuts
      // off the common case (a 404 never reaches the element at all), but a
      // deck above MEMORY_DECK_MAX_BYTES streams, and a stream that dies
      // arrives here — so this is the path for *any* load failure the probe
      // did not already own.
      el.addEventListener('error', () => {
        const code = el.error?.code ?? 0;
        // MEDIA_ERR_ABORTED is ours: a src swap or a dispose. Everything else
        // means this deck cannot make sound.
        if (this.state[deck] !== 'empty' && code !== MEDIA_ERR_ABORTED) {
          this.fail(deck, this.classifyElementError(deck, code), null);
          return;
        }
        this.emit();
      });
      el.addEventListener('pause', () => {
        // The browser can pause an element on its own (media pipeline hiccup,
        // source swap). If we still intend to play, recover rather than
        // leaving the transport in a half-playing state.
        if (this.wantPlaying && deck === this.active && !el.ended && this.state[deck] === 'ready') {
          window.setTimeout(() => {
            if (this.wantPlaying && this.active === deck && el.paused && !el.ended) {
              void el.play().catch(() => {
                this.wantPlaying = false;
                this.pauseAll();
                this.emit();
              });
            }
          }, 0);
        }
        this.emit();
      });
    }
  }

  private makeElement(): HTMLAudioElement {
    const el = document.createElement('audio');
    el.preload = 'auto';
    el.crossOrigin = 'anonymous';
    el.style.display = 'none';
    document.body.appendChild(el);
    return el;
  }

  /** Before the Web Audio graph exists, A/B is done with element mute. */
  private applyMuteFallback(): void {
    if (this.ctx) return;
    for (const deck of ['original', 'cleaned'] as const) {
      this.els[deck].muted = deck !== this.active || this.state[deck] === 'empty';
    }
  }

  /**
   * Must be called from a user gesture the first time (AudioContext policy).
   *
   * Fails soft. Every caller already null-guards `ctx`/`gains` — `setActive`
   * branches to `applyMuteFallback()` and `play` skips the resume — so a
   * missing graph costs the cross-fade and the analyser, not playback. It used
   * to throw instead: `window.AudioContext` absent, or
   * `createMediaElementSource` refusing (a media element can only ever back one
   * source node, and tainted media throws), took `play()` down with it, and
   * since `toggle()` calls `void this.play()` the rejection landed in the app's
   * failure net as an unexplained error while the transport simply did nothing.
   * Element-level mute is the pre-graph A/B mechanism and still works; a
   * webview without WebAudio should lose the fade, not the sound.
   */
  private ensureGraph(): void {
    if (this.ctx) return;
    try {
      this.buildGraph();
    } catch {
      // Left null on purpose: `applyMuteFallback()` owns A/B from here.
      this.ctx = null;
      this.gains = null;
      this.analyser = null;
      this.applyMuteFallback();
    }
  }

  private buildGraph(): void {
    const Ctor = window.AudioContext;
    const ctx = new Ctor({ latencyHint: 'interactive' });
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 8192;
    analyser.smoothingTimeConstant = 0.78;
    analyser.minDecibels = -100;
    analyser.maxDecibels = 0;
    analyser.connect(ctx.destination);
    const gains = {
      original: ctx.createGain(),
      cleaned: ctx.createGain(),
    };
    for (const deck of ['original', 'cleaned'] as const) {
      const src = ctx.createMediaElementSource(this.els[deck]);
      src.connect(gains[deck]);
      gains[deck].connect(analyser);
      gains[deck].gain.value = deck === this.active ? this.getTargetGain(deck) : 0;
      // Element-level mute was the pre-graph A/B mechanism; from here on the
      // gain nodes own it (a muted element also mutes its source node).
      this.els[deck].muted = false;
    }
    this.ctx = ctx;
    this.gains = gains;
    this.analyser = analyser;
  }

  setLoudnessMatch(enabled: boolean, lufsIn: number | null, lufsOut: number | null): void {
    this.loudnessMatchEnabled = enabled;
    this.lufsIn = lufsIn;
    this.lufsOut = lufsOut;
    this.applyTargetGain();
    this.emit();
  }

  get isLoudnessMatchEnabled(): boolean {
    return this.loudnessMatchEnabled;
  }

  get gainOffsetDb(): number {
    if (
      !this.loudnessMatchEnabled ||
      this.lufsIn === null ||
      this.lufsOut === null ||
      !Number.isFinite(this.lufsIn) ||
      !Number.isFinite(this.lufsOut)
    ) {
      return 0;
    }
    const delta = this.lufsIn - this.lufsOut;
    return Math.max(-12, Math.min(12, delta));
  }

  getTargetGain(deck: Deck): number {
    if (deck === 'original') return 1.0;
    const offset = this.gainOffsetDb;
    if (offset === 0) return 1.0;
    return Math.pow(10, offset / 20);
  }

  private applyTargetGain(): void {
    const targetCleaned = this.getTargetGain('cleaned');
    if (this.gains && this.ctx) {
      const now = this.ctx.currentTime;
      for (const deck of ['original', 'cleaned'] as const) {
        const g = this.gains[deck].gain;
        const target = deck === this.active ? this.getTargetGain(deck) : 0;
        g.cancelScheduledValues(now);
        g.setValueAtTime(g.value, now);
        g.linearRampToValueAtTime(target, now + FADE_S);
      }
    } else {
      this.els.cleaned.volume = Math.min(1, Math.max(0, targetCleaned));
      this.applyMuteFallback();
    }
  }

  getAnalyser(): AnalyserNode | null {
    return this.analyser;
  }

  get sampleRate(): number {
    return this.ctx?.sampleRate ?? 48000;
  }

  get activeDeck(): Deck {
    return this.active;
  }

  hasDeck(deck: Deck): boolean {
    return this.state[deck] !== 'empty';
  }

  /** True once the deck can actually be played (its element has a source). */
  isReady(deck: Deck): boolean {
    return this.state[deck] === 'ready';
  }

  /** The standing fault on a deck, or null when it has none. */
  deckFault(deck: Deck): DeckFault | null {
    return this.faults[deck];
  }

  /**
   * How the player asks whether the engine is answering at all.
   *
   * It needs an answer because `MediaError.code` cannot give it one. An engine
   * that dies while a deck is loading makes Chromium report code **4**
   * (`SRC_NOT_SUPPORTED`) — the same code as a PNG renamed `.wav` — so a
   * healthy master went out of service as "nothing here could decode it" and,
   * because only a `network` fault is ever retried, stayed dead for the rest
   * of the session after the engine came back. Liveness is what tells the two
   * apart, and this is the seam the store hangs it on.
   */
  setLivenessProbe(fn: (() => boolean) | null): void {
    this.engineLive = fn;
  }

  /**
   * Which kind of fault an element error really is.
   *
   * Order matters: the element's own `MEDIA_ERR_NETWORK` is the clearest
   * answer; the size probe having thrown is the next, because a probe that
   * never got a reply is the engine being unreachable regardless of what the
   * media stack made of the src afterwards; the injected liveness check is the
   * last, for the case where nothing on this deck's own path failed visibly.
   */
  private classifyElementError(deck: Deck, code: number): DeckFaultKind {
    if (code === MEDIA_ERR_NETWORK) return 'network';
    if (this.probeFailed[deck]) return 'network';
    if (this.engineLive && !this.engineLive()) return 'network';
    return 'unreadable';
  }

  /**
   * A deck that decodes to nothing is out of service, however happy the
   * element looks.
   *
   * The truncated master is the quietest failure this player can have: a RIFF
   * header with 60 bytes behind it gives `readyState` 4, `MediaError` null and
   * `duration` 0.000396 s, so every check the rest of this class makes passes,
   * the A/B stays lit on CLEANED, the transport reads 00:00.0 / 00:00.0 and
   * Play does nothing at all. The run's own report knows how long that file is
   * meant to be, so the length is checkable — and being checkable, it is
   * checked.
   */
  private judgeDuration(deck: Deck): void {
    if (this.state[deck] !== 'ready') return;
    if (this.durationJudged[deck]) return;
    const d = this.els[deck].duration;
    // Not knowing yet is not a verdict: `NaN` before metadata and `Infinity`
    // on a stream with no declared length both mean "ask again later".
    if (!Number.isFinite(d)) return;
    const expected = this.expected[deck];
    const tooShort =
      d <= DECK_MIN_DURATION_S ||
      (expected !== null && expected > 0 && d < expected * DECK_MIN_DURATION_RATIO);
    this.durationJudged[deck] = true;
    if (!tooShort) return;
    this.fail(deck, 'truncated', null, { got: d, expected });
  }

  /** Publishes every deck that goes out of service. See {@link DeckFault}. */
  onFault(fn: FaultListener): () => void {
    this.faultListeners.add(fn);
    return () => {
      this.faultListeners.delete(fn);
    };
  }

  /**
   * @param expectedDuration seconds this file is known to hold — the run's own
   * report, or the analysis of the file itself. Passing it is what lets a deck
   * that decodes to nothing be caught (see {@link judgeDuration}); `null` keeps
   * only the "zero length is never playable" half of that check.
   */
  load(deck: Deck, url: string | null, expectedDuration: number | null = null): void {
    if (url === null) {
      this.retire(deck);
      return;
    }
    // Asking for a deck again is a fresh attempt: whatever it failed at last
    // time is no longer the current answer.
    this.faults[deck] = null;
    this.expected[deck] = expectedDuration;
    if (this.wanted[deck] === url && this.state[deck] !== 'empty') return;
    this.wanted[deck] = url;
    this.durationJudged[deck] = false;
    this.probeFailed[deck] = false;
    // Re-selecting a run whose master is still on the element: revive it
    // instead of re-fetching, so no request is made at all.
    if (this.attached[deck] === url) {
      this.state[deck] = 'ready';
      this.epoch[deck] += 1;
      this.afterAttach(deck);
      this.judgeDuration(deck);
      return;
    }
    this.state[deck] = 'loading';
    const gen = ++this.epoch[deck];
    void this.attach(deck, url, gen);
    this.emit();
  }

  /**
   * A7 · the identity above this deck is switching NOW; whatever the deck is
   * holding may stay claimed only if it is already the named file.
   *
   * `selectRun` switches the run on screen synchronously, but the deck loads
   * that follow sit behind engine round-trips — and a hung engine mid-restore
   * used to leave both decks holding the PREVIOUS run's audio under the new
   * run's name for as long as the hang lasted (measured: transport
   * `00:00.0 / 00:12.0` and two 12 s blobs under a 94.6 s run, indefinitely,
   * with the A/B lit CLEANED). A deck must never claim audio it does not
   * hold, so the mismatch is retired before the first await, not after the
   * last one. A deck already holding `url` is left exactly as it is — the
   * revive path in {@link load} still costs zero requests — so a healthy
   * same-file restore is unchanged. `retire` keeps the element's `src`, so
   * the audio itself also revives without a fetch if the same file is asked
   * for again.
   */
  claimOnly(deck: Deck, url: string | null): void {
    if (url !== null && this.attached[deck] === url) return;
    this.retire(deck);
  }

  /**
   * One byte of the deck's file, and what happened when we asked for it.
   *
   * A one-byte ranged GET rather than a HEAD: a HEAD response carries a body
   * stream that is never consumed, and Chromium records the unread stream as
   * `net::ERR_ABORTED` — the exact entry this path exists to avoid. One byte
   * can be read to completion, and `Content-Range` carries the full length.
   *
   * `null` means nobody answered. `delivered: false` means the engine answered
   * and then could not produce the bytes, which is a different fact and needs
   * to stay a different fact.
   */
  private async probeOnce(
    url: string,
  ): Promise<{ status: number; ok: boolean; delivered: boolean; headers: Headers } | null> {
    let res: Response;
    try {
      res = await fetch(url, { headers: { Range: 'bytes=0-0' }, cache: 'no-store' });
    } catch {
      return null;
    }
    try {
      await res.arrayBuffer();
      return { status: res.status, ok: res.ok, delivered: true, headers: res.headers };
    } catch {
      return { status: res.status, ok: res.ok, delivered: false, headers: res.headers };
    }
  }

  /**
   * Fetch the deck's audio into memory when it is small enough, then attach it.
   * A fetch that has been superseded is *dropped*, never aborted — an
   * AbortController here would put the very ERR_ABORTED back in the log that
   * this whole path exists to remove.
   */
  private async attach(deck: Deck, url: string, gen: number): Promise<void> {
    let src = url;
    /** The engine's own answer to the probe, or null when it never answered. */
    let answered: number | null = null;
    let probe = await this.probeOnce(url);
    // Answered, then handed over nothing. That is two different worlds — a file
    // the engine has listed and cannot read (a chmod 000 master, a volume that
    // went away under it), or the engine dying between the headers and the
    // body — and one more byte-sized question separates them: a second answer
    // means the file, no answer at all means the engine.
    if (probe && probe.ok && !probe.delivered) {
      const second = await this.probeOnce(url);
      if (this.epoch[deck] !== gen) return;
      if (second === null) {
        this.probeFailed[deck] = true;
      } else if (second.ok && !second.delivered) {
        this.fail(deck, 'forbidden', second.status);
        return;
      }
      probe = second;
    }
    if (probe === null) {
      // Nobody answered. That is not the engine saying no, so the element is
      // still given the URL and its own error — if there is one — comes back
      // through the `error` listener as a fault. It is recorded, though: the
      // element reports that failure as code 4, which reads exactly like junk
      // bytes, and this is the only evidence that says otherwise.
      this.probeFailed[deck] = true;
    } else {
      answered = probe.status;
      if (probe.ok) {
        const total = probe.headers.get('Content-Range')?.split('/')[1];
        const len = Number(total ?? probe.headers.get('Content-Length') ?? Number.NaN);
        if (Number.isFinite(len) && len > 0 && len <= MEMORY_DECK_MAX_BYTES) {
          try {
            const res = await fetch(url, { cache: 'no-store' });
            if (res.ok) {
              const blob = await res.blob();
              if (this.epoch[deck] === gen) src = URL.createObjectURL(blob);
            }
          } catch {
            // The pull died where the probe had succeeded: the engine went
            // away in between. Same evidence, same conclusion.
            this.probeFailed[deck] = true;
          }
        }
      }
    }
    if (this.epoch[deck] !== gen) {
      if (src.startsWith('blob:')) URL.revokeObjectURL(src);
      return;
    }
    // The engine answered and the answer was no. Handing the URL to the element
    // anyway would give us a deck that reports itself ready and makes no sound
    // — the whole bug this path exists to refuse.
    if (answered !== null && answered >= 400) {
      this.fail(
        deck,
        answered === 404 ? 'missing' : answered === 403 ? 'forbidden' : 'unreadable',
        answered,
      );
      return;
    }
    this.setSrc(deck, url, src);
  }

  private setSrc(deck: Deck, wantedUrl: string, src: string): void {
    const el = this.els[deck];
    const stale = this.objectUrl[deck];
    el.src = src;
    this.attached[deck] = wantedUrl;
    this.objectUrl[deck] = src.startsWith('blob:') ? src : null;
    // The element has already moved off the old resource; the URL handle can go.
    if (stale && stale !== src) URL.revokeObjectURL(stale);
    this.state[deck] = 'ready';
    this.afterAttach(deck);
  }

  private afterAttach(deck: Deck): void {
    const el = this.els[deck];
    this.applyMuteFallback();
    // Keep a freshly loaded deck aligned with the current transport position.
    const t = this.els[this.active].currentTime;
    if (deck !== this.active && Number.isFinite(t) && t > 0) {
      el.currentTime = t;
    }
    if (this.pendingActive === deck) {
      this.pendingActive = null;
      this.setActive(deck);
    } else if (this.wantPlaying && deck !== this.active) {
      void el.play().catch(() => undefined);
    }
    this.emit();
  }

  /**
   * Take a deck out of service without touching its `src`. Removing the
   * attribute (or calling `load()` on an empty element) is what used to abort
   * the deck's open range request; the element keeps the source it has, stays
   * silent, and reports itself unavailable until something is loaded onto it.
   */
  private retire(deck: Deck): void {
    this.faults[deck] = null;
    if (this.state[deck] === 'empty' && this.wanted[deck] === null) return;
    this.takeOutOfService(deck, false);
    this.emit();
  }

  /**
   * The shared body of "this deck stops making sound". Returns the deck that
   * took over, or null when there was nothing to hand to.
   *
   * `keepPlaying` is false for a deliberate retire — a new clip is arriving and
   * playback ends with the old one — and true for a fault, where the run
   * carries on over whatever deck still works. That fallback is exactly the
   * event the user has to be told about, so `fail` publishes it.
   */
  private takeOutOfService(deck: Deck, keepPlaying: boolean): Deck | null {
    this.epoch[deck] += 1; // supersede any in-flight attach
    this.wanted[deck] = null;
    this.state[deck] = 'empty';
    if (this.pendingActive === deck) this.pendingActive = null;
    const el = this.els[deck];
    el.pause();
    if (deck === this.active && this.wantPlaying && !keepPlaying) {
      this.wantPlaying = false;
      this.pauseAll();
    }
    if (this.gains && this.ctx) {
      const g = this.gains[deck].gain;
      g.cancelScheduledValues(this.ctx.currentTime);
      g.setValueAtTime(0, this.ctx.currentTime);
    } else {
      el.muted = true;
    }
    // Never leave the transport pointing at a deck that is out of service:
    // `time` and `duration` read the active element.
    if (deck === this.active) {
      const other: Deck = deck === 'original' ? 'cleaned' : 'original';
      if (this.state[other] === 'ready') {
        // `setActive` sample-locks the arriving deck to the leaving one. A deck
        // that never loaded reads 0, and locking to that would throw the
        // transport back to the top of the take — so the survivor's own
        // position is taken before the switch and put back after it.
        const keep = this.els[other].currentTime;
        this.setActive(other);
        if (Number.isFinite(keep) && Math.abs(this.els[other].currentTime - keep) > 0.001) {
          this.els[other].currentTime = keep;
        }
        return other;
      }
      if (this.wantPlaying) {
        this.wantPlaying = false;
        this.pauseAll();
      }
    }
    return null;
  }

  /**
   * A deck could not be played.
   *
   * Take it out of service, fall back to the deck that still works, and *say
   * so*. Before this existed the fallback happened silently: the A/B control
   * stayed lit on CLEANED, `aria-checked` stayed true, and the ORIGINAL element
   * was the one making sound. A player that quietly swaps decks is a player
   * that makes the screen lie, so the swap is now an event.
   */
  private fail(
    deck: Deck,
    kind: DeckFaultKind,
    status: number | null,
    duration?: { got: number; expected: number | null },
  ): void {
    const already = this.faults[deck];
    if (already && already.kind === kind && this.state[deck] === 'empty') return;
    // Read before `takeOutOfService` clears it: the fault has to name the file
    // it is about, so a listener can tell a stale answer from a live one.
    const url = this.wanted[deck] ?? this.attached[deck] ?? '';
    // Whatever the element is holding, it is not something this deck can play.
    this.attached[deck] = null;
    // "Heading here" covers both shapes of the same request: the deck is live,
    // or a `setActive` is waiting on its fetch. Either way the transport was on
    // its way to this deck, so whatever it ends up on instead is a fallback the
    // user has to be told about.
    const wasHeadingHere = this.active === deck || this.pendingActive === deck;
    const other: Deck = deck === 'original' ? 'cleaned' : 'original';
    const switched = this.takeOutOfService(deck, true);
    const fellBackTo =
      switched ?? (wasHeadingHere && this.state[other] === 'ready' ? other : null);
    const fault: DeckFault = { deck, url, kind, status, fellBackTo, ...(duration ? { duration } : {}) };
    this.faults[deck] = fault;
    this.emit();
    for (const fn of this.faultListeners) fn(fault);
  }

  setActive(deck: Deck): void {
    if (deck === this.active) return;
    if (this.state[deck] === 'empty') return;
    if (this.state[deck] === 'loading') {
      // Arrive at the deck as soon as it has a source.
      this.pendingActive = deck;
      return;
    }
    const prev = this.active;
    this.active = deck;
    const from = this.els[prev];
    const to = this.els[deck];
    // Common-region bounds: ensure neither deck compares past the end of the shorter take.
    const maxTime = this.duration;
    if (maxTime > 0 && from.currentTime > maxTime) {
      from.currentTime = maxTime;
    }
    if (maxTime > 0 && to.currentTime > maxTime) {
      to.currentTime = maxTime;
    }
    // A/B has to stay sample-locked; when both decks are in memory (the normal
    // case) this correction costs nothing at all.
    if (Math.abs(to.currentTime - from.currentTime) > SYNC_TOLERANCE_S) {
      to.currentTime = from.currentTime;
    }
    if (this.gains && this.ctx) {
      const now = this.ctx.currentTime;
      const gFrom = this.gains[prev].gain;
      const gTo = this.gains[deck].gain;
      gFrom.cancelScheduledValues(now);
      gTo.cancelScheduledValues(now);
      gFrom.setValueAtTime(gFrom.value, now);
      gTo.setValueAtTime(gTo.value, now);
      gFrom.linearRampToValueAtTime(0, now + FADE_S);
      gTo.linearRampToValueAtTime(this.getTargetGain(deck), now + FADE_S);
    } else {
      this.applyMuteFallback();
      this.els.cleaned.volume = Math.min(1, Math.max(0, this.getTargetGain('cleaned')));
    }
    if (this.wantPlaying && to.paused) {
      void to.play().catch(() => undefined);
    }
    this.emit();
  }

  async play(): Promise<void> {
    if (this.state[this.active] !== 'ready') return;
    this.ensureGraph();
    if (this.ctx && this.ctx.state === 'suspended') {
      await this.ctx.resume();
    }
    this.wantPlaying = true;
    if (this.duration > 0 && this.time >= this.duration) {
      this.seek(0);
    }
    const t = this.els[this.active].currentTime;
    const plays: Promise<void>[] = [];
    for (const deck of ['original', 'cleaned'] as const) {
      if (this.state[deck] !== 'ready') continue;
      const el = this.els[deck];
      if (deck !== this.active && Math.abs(el.currentTime - t) > SYNC_TOLERANCE_S) {
        el.currentTime = t;
      }
      plays.push(el.play().catch(() => undefined));
    }
    await Promise.all(plays);
    this.startTicker();
    this.emit();
  }

  pause(): void {
    this.wantPlaying = false;
    this.pauseAll();
    this.emit();
  }

  toggle(): void {
    if (this.wantPlaying) this.pause();
    else void this.play();
  }

  seek(time: number): void {
    const d = this.duration;
    const t = Math.max(0, Math.min(d > 0 ? d : time, time));
    for (const deck of ['original', 'cleaned'] as const) {
      if (this.state[deck] === 'ready') this.els[deck].currentTime = t;
    }
    this.emit();
  }

  get time(): number {
    const t = this.els[this.active].currentTime;
    return Number.isFinite(t) ? t : 0;
  }

  get duration(): number {
    const origReady = this.state.original === 'ready';
    const cleanReady = this.state.cleaned === 'ready';
    const dOrig =
      origReady && Number.isFinite(this.els.original.duration) && this.els.original.duration > 0
        ? this.els.original.duration
        : null;
    const dClean =
      cleanReady && Number.isFinite(this.els.cleaned.duration) && this.els.cleaned.duration > 0
        ? this.els.cleaned.duration
        : null;

    if (dOrig !== null && dClean !== null) {
      return Math.min(dOrig, dClean);
    }
    if (dOrig !== null) return dOrig;
    if (dClean !== null) return dClean;
    return 0;
  }

  get playing(): boolean {
    return this.wantPlaying && !this.els[this.active].paused;
  }

  snapshot(): PlayerSnapshot {
    return {
      playing: this.playing,
      time: this.time,
      duration: this.duration,
      // The A/B control renders these two, not its own idea of what is on:
      // `active` is the deck making sound and `pending` is the deck it has
      // been asked to arrive at. Neither can drift from the player.
      active: this.active,
      pending: this.pendingActive,
      loudnessMatch: this.loudnessMatchEnabled,
      gainOffsetDb: this.gainOffsetDb,
    };
  }

  subscribe(fn: Listener): () => void {
    this.listeners.add(fn);
    fn(this.snapshot());
    return () => {
      this.listeners.delete(fn);
    };
  }

  private pauseAll(): void {
    for (const deck of ['original', 'cleaned'] as const) {
      this.els[deck].pause();
    }
    this.stopTicker();
  }

  private startTicker(): void {
    if (this.raf) return;
    const tick = (): void => {
      this.raf = 0;
      if (!this.wantPlaying) return;
      const a = this.els[this.active];
      const other: Deck = this.active === 'original' ? 'cleaned' : 'original';
      const b = this.els[other];
      const maxTime = this.duration;
      const bothReady = this.state.original === 'ready' && this.state.cleaned === 'ready';
      if (bothReady && maxTime > 0 && (a.currentTime >= maxTime || b.currentTime >= maxTime)) {
        this.wantPlaying = false;
        this.pauseAll();
        if (a.currentTime > maxTime) a.currentTime = maxTime;
        if (b.currentTime > maxTime) b.currentTime = maxTime;
        this.emit();
        return;
      }
      // Drift guard: keep the inactive deck within tolerance of the active one.
      if (
        this.state[other] === 'ready' &&
        !b.paused &&
        Math.abs(a.currentTime - b.currentTime) > 0.08
      ) {
        b.currentTime = a.currentTime;
      }
      this.emit();
      this.raf = window.requestAnimationFrame(tick);
    };
    this.raf = window.requestAnimationFrame(tick);
  }

  private stopTicker(): void {
    if (this.raf) {
      window.cancelAnimationFrame(this.raf);
      this.raf = 0;
    }
  }

  private emit(): void {
    const snap = this.snapshot();
    for (const fn of this.listeners) fn(snap);
  }

  dispose(): void {
    this.stopTicker();
    this.listeners.clear();
    this.faultListeners.clear();
    this.faults = { original: null, cleaned: null };
    for (const deck of ['original', 'cleaned'] as const) {
      const el = this.els[deck];
      this.epoch[deck] += 1;
      el.pause();
      el.removeAttribute('src');
      el.load();
      el.remove();
      const blob = this.objectUrl[deck];
      if (blob) URL.revokeObjectURL(blob);
      this.objectUrl[deck] = null;
      this.attached[deck] = null;
      this.state[deck] = 'empty';
    }
    void this.ctx?.close();
  }
}

let singleton: DualPlayer | null = null;
export function getPlayer(): DualPlayer {
  if (!singleton) singleton = new DualPlayer();
  return singleton;
}
