// Dual-deck player: ORIGINAL and CLEANED <audio> elements run in lock-step so
// the A/B switch is instantaneous (a 6 ms gain cross-fade, no restart). Both
// feed one AnalyserNode for the live spectrum overlay.

export type Deck = 'original' | 'cleaned';

export interface PlayerSnapshot {
  playing: boolean;
  time: number;
  duration: number;
}

type Listener = (snap: PlayerSnapshot) => void;

const FADE_S = 0.006;
const SYNC_TOLERANCE_S = 0.035;

export class DualPlayer {
  private readonly els: Record<Deck, HTMLAudioElement>;
  private ctx: AudioContext | null = null;
  private gains: Record<Deck, GainNode> | null = null;
  private analyser: AnalyserNode | null = null;
  private active: Deck = 'original';
  private loaded: Record<Deck, boolean> = { original: false, cleaned: false };
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
        if (this.wantPlaying && deck === this.active && !el.ended && this.loaded[deck]) {
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
      this.els[deck].muted = deck !== this.active;
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
    return this.loaded[deck];
  }

  load(deck: Deck, url: string | null): void {
    const el = this.els[deck];
    const wasPlaying = this.wantPlaying;
    if (url === null) {
      el.removeAttribute('src');
      el.load();
      this.loaded[deck] = false;
      if (deck === this.active && wasPlaying) {
        this.wantPlaying = false;
        this.pauseAll();
      }
      this.emit();
      return;
    }
    if (el.src === url) return;
    el.src = url;
    el.load();
    this.loaded[deck] = true;
    this.applyMuteFallback();
    // Keep a freshly loaded deck aligned with the current transport position.
    const t = this.els[this.active].currentTime;
    if (deck !== this.active && Number.isFinite(t) && t > 0) {
      el.currentTime = t;
    }
    if (wasPlaying && deck !== this.active) {
      void el.play().catch(() => undefined);
    }
    this.emit();
  }

  setActive(deck: Deck): void {
    if (deck === this.active) return;
    if (!this.loaded[deck]) return;
    const prev = this.active;
    this.active = deck;
    const from = this.els[prev];
    const to = this.els[deck];
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
    if (!this.loaded[this.active]) return;
    this.ensureGraph();
    if (this.ctx && this.ctx.state === 'suspended') {
      await this.ctx.resume();
    }
    this.wantPlaying = true;
    const t = this.els[this.active].currentTime;
    const plays: Promise<void>[] = [];
    for (const deck of ['original', 'cleaned'] as const) {
      if (!this.loaded[deck]) continue;
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
      if (this.loaded[deck]) this.els[deck].currentTime = t;
    }
    this.emit();
  }

  get time(): number {
    const t = this.els[this.active].currentTime;
    return Number.isFinite(t) ? t : 0;
  }

  get duration(): number {
    const d = this.els[this.active].duration;
    if (Number.isFinite(d) && d > 0) return d;
    const other: Deck = this.active === 'original' ? 'cleaned' : 'original';
    const d2 = this.els[other].duration;
    return Number.isFinite(d2) && d2 > 0 ? d2 : 0;
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
      if (this.loaded[other] && !b.paused && Math.abs(a.currentTime - b.currentTime) > 0.08) {
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
      el.pause();
      el.removeAttribute('src');
      el.load();
      el.remove();
    }
    void this.ctx?.close();
  }
}

let singleton: DualPlayer | null = null;
export function getPlayer(): DualPlayer {
  if (!singleton) singleton = new DualPlayer();
  return singleton;
}
