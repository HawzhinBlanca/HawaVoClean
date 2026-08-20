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
}

type Listener = (snap: PlayerSnapshot) => void;

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
  /** A `setActive` that arrived while the deck was still fetching. */
  private pendingActive: Deck | null = null;
  private listeners = new Set<Listener>();
  private raf = 0;
  private wantPlaying = false;

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
      el.addEventListener('loadedmetadata', () => this.emit());
      el.addEventListener('error', () => this.emit());
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

  /** Must be called from a user gesture the first time (AudioContext policy). */
  private ensureGraph(): void {
    if (this.ctx) return;
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
      gains[deck].gain.value = deck === this.active ? 1 : 0;
      // Element-level mute was the pre-graph A/B mechanism; from here on the
      // gain nodes own it (a muted element also mutes its source node).
      this.els[deck].muted = false;
    }
    this.ctx = ctx;
    this.gains = gains;
    this.analyser = analyser;
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

  load(deck: Deck, url: string | null): void {
    if (url === null) {
      this.retire(deck);
      return;
    }
    if (this.wanted[deck] === url && this.state[deck] !== 'empty') return;
    this.wanted[deck] = url;
    // Re-selecting a run whose master is still on the element: revive it
    // instead of re-fetching, so no request is made at all.
    if (this.attached[deck] === url) {
      this.state[deck] = 'ready';
      this.epoch[deck] += 1;
      this.afterAttach(deck);
      return;
    }
    this.state[deck] = 'loading';
    const gen = ++this.epoch[deck];
    void this.attach(deck, url, gen);
    this.emit();
  }

  /**
   * Fetch the deck's audio into memory when it is small enough, then attach it.
   * A fetch that has been superseded is *dropped*, never aborted — an
   * AbortController here would put the very ERR_ABORTED back in the log that
   * this whole path exists to remove.
   */
  private async attach(deck: Deck, url: string, gen: number): Promise<void> {
    let src = url;
    try {
      // The size probe is a one-byte ranged GET rather than a HEAD: a HEAD
      // response carries a body stream that is never consumed, and Chromium
      // records the unread stream as `net::ERR_ABORTED` — the exact entry this
      // path exists to avoid. One byte can be read to completion, and
      // `Content-Range` carries the full length.
      const probe = await fetch(url, { headers: { Range: 'bytes=0-0' }, cache: 'no-store' });
      await probe.arrayBuffer();
      const total = probe.headers.get('Content-Range')?.split('/')[1];
      const len = Number(total ?? probe.headers.get('Content-Length') ?? Number.NaN);
      if (probe.ok && Number.isFinite(len) && len > 0 && len <= MEMORY_DECK_MAX_BYTES) {
        const res = await fetch(url, { cache: 'no-store' });
        if (res.ok) {
          const blob = await res.blob();
          if (this.epoch[deck] === gen) src = URL.createObjectURL(blob);
        }
      }
    } catch {
      // Offline, refused, or a body we could not read: fall back to streaming.
      // The element will surface its own error if the URL is truly bad.
    }
    if (this.epoch[deck] !== gen) {
      if (src.startsWith('blob:')) URL.revokeObjectURL(src);
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
    if (this.state[deck] === 'empty' && this.wanted[deck] === null) return;
    this.epoch[deck] += 1; // supersede any in-flight attach
    this.wanted[deck] = null;
    this.state[deck] = 'empty';
    if (this.pendingActive === deck) this.pendingActive = null;
    const el = this.els[deck];
    el.pause();
    if (deck === this.active && this.wantPlaying) {
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
      if (this.state[other] === 'ready') this.setActive(other);
    }
    this.emit();
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
      gTo.linearRampToValueAtTime(1, now + FADE_S);
    } else {
      this.applyMuteFallback();
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
    if (this.state[this.active] === 'ready') {
      const d = this.els[this.active].duration;
      if (Number.isFinite(d) && d > 0) return d;
    }
    const other: Deck = this.active === 'original' ? 'cleaned' : 'original';
    if (this.state[other] === 'ready') {
      const d2 = this.els[other].duration;
      if (Number.isFinite(d2) && d2 > 0) return d2;
    }
    return 0;
  }

  get playing(): boolean {
    return this.wantPlaying && !this.els[this.active].paused;
  }

  snapshot(): PlayerSnapshot {
    return { playing: this.playing, time: this.time, duration: this.duration };
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
      // Drift guard: keep the inactive deck within tolerance of the active one.
      const a = this.els[this.active];
      const other: Deck = this.active === 'original' ? 'cleaned' : 'original';
      const b = this.els[other];
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
