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

import { deriveAuditionMode, type AuditionMode } from '../state/machines';

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
  /** Whether Delta ("Listen to Removed": Original - Cleaned) audition is active. */
  deltaMode: boolean;
  /** Whether loop playback is enabled. */
  loopEnabled: boolean;
  /** Active loop region [startS, endS], or null to loop the whole clip. */
  loopRange: [number, number] | null;
  /** Wet/dry mix in [0, 1] when active deck is cleaned (1.0 = 100% cleaned). */
  wetDryMix: number;
  /** Single derived audition mode across A/B, Δ Removed, and Wet/Dry Blend. */
  auditionMode: AuditionMode;
}

export interface LiveMeterLevels {
  momentaryLufs: number;
  shortTermLufs: number;
  peakDbfs: number;
  truePeakHoldDbfs: number;
  /** Instantaneous noise reduction in dB (Original RMS - Cleaned RMS) when both decks play; null otherwise. */
  nrDb?: number | null;
  /** Real-time stereo phase correlation in [-1, +1] (+1 = mono/in-phase, <0 = out of phase); null when silent. */
  stereoCorrelation?: number | null;
}

export interface BandSoloState {
  active: boolean;
  hz: number;
  q: number;
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
const SEEK_THROTTLE_MS = 32;
const SEEK_DUCK_S = 0.01;
const SYNC_TOLERANCE_S = 0.035;
const METER_FLOOR_DB = -70;
const MOMENTARY_WINDOW_MS = 400;
const SHORT_TERM_WINDOW_MS = 3000;

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
  private deltaInverter: GainNode | null = null;
  private deltaGain: GainNode | null = null;
  private analyser: AnalyserNode | null = null;
  /** ITU-R BS.1770 K-weighting filter chain feeding a dedicated metering analyser (unconnected to destination). */
  private kStage1: BiquadFilterNode | null = null;
  private kStage2: BiquadFilterNode | null = null;
  private kAnalyser: AnalyserNode | null = null;
  /** Per-deck pre-fader analysers for real-time Noise Reduction (NR dB Removed) measurement. */
  private origAnalyser: AnalyserNode | null = null;
  private cleanAnalyser: AnalyserNode | null = null;
  /** Parallel 24 dB/oct 4th-order Butterworth live HPF audition on Original playback. */
  private hpfStage1: BiquadFilterNode | null = null;
  private hpfStage2: BiquadFilterNode | null = null;
  private hpfDryGain: GainNode | null = null;
  private hpfWetGain: GainNode | null = null;
  private liveHpfEnabled = false;
  private liveHpfCutoffHz = 80;
  /** Parallel FabFilter-style real-time bandpass frequency solo audition (post-analyser, pre-destination). */
  private soloFilter: BiquadFilterNode | null = null;
  private soloDryGain: GainNode | null = null;
  private soloWetGain: GainNode | null = null;
  private soloActive = false;
  private soloCenterHz = 1000;
  private soloQ = 4.5;
  /** Stereo phase correlation splitter & per-channel analysers (meter-only, never routed to destination). */
  private corrSplitter: ChannelSplitterNode | null = null;
  private corrAnalyserL: AnalyserNode | null = null;
  private corrAnalyserR: AnalyserNode | null = null;
  private corrBufL: Float32Array<ArrayBuffer> | null = null;
  private corrBufR: Float32Array<ArrayBuffer> | null = null;
  private smoothCorr: number | null = null;
  /** Raw bytes and decoded PCM AudioBuffers for 0-sample-skew AudioBufferSourceNode playback. */
  private rawBytes: Record<Deck, ArrayBuffer | null> = { original: null, cleaned: null };
  private buffers: Record<Deck, AudioBuffer | null> = { original: null, cleaned: null };
  private bufferSources: Record<Deck, AudioBufferSourceNode | null> = { original: null, cleaned: null };
  private decodeInFlight: Record<Deck, Promise<void> | null> = { original: null, cleaned: null };
  private bufferSourceToken = 0;
  private bufferPlaying = false;
  private bufferStartCtxTime = 0;
  private bufferStartOffset = 0;
  private bufferPosition = 0;
  /** Scrubbing seek throttle state (32 ms cadence with 10 ms momentary gain ducking). */
  private lastSeekAppliedMs = 0;
  private pendingScrubTime: number | null = null;
  private seekThrottleTimer: ReturnType<typeof setTimeout> | null = null;
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
  /** Timeline start offset (seconds) when a deck holds a region preview clip rather than t=0. */
  private offset: Record<Deck, number> = { original: 0, cleaned: 0 };
  /** True when a deck holds an ephemeral region preview slice so `duration` keeps the full clip length. */
  private isPreviewDeck: Record<Deck, boolean> = { original: false, cleaned: false };
  /** Timeline position (seconds) to apply as soon as a loading/reloading deck attaches. */
  private pendingSeekTime: Record<Deck, number | null> = { original: null, cleaned: null };
  /** Deferred deck-local seek applied on `loadedmetadata` if `readyState` was 0 during `afterAttach`. */
  private seekOnMeta: Record<Deck, number | null> = { original: null, cleaned: null };
  /** The last published fault per deck, cleared when the deck is asked for again. */
  private faults: Record<Deck, DeckFault | null> = { original: null, cleaned: null };
  private listeners = new Set<Listener>();
  private faultListeners = new Set<FaultListener>();
  private raf = 0;
  private wantPlaying = false;
  private playToken = 0;
  private disposed = false;
  private loudnessMatchEnabled = true;
  private lufsIn: number | null = null;
  private lufsOut: number | null = null;
  private deltaModeEnabled = false;
  private loopEnabledFlag = false;
  private loopRangeValue: [number, number] | null = null;
  private wetDryMixValue = 1.0;
  private meterBuf: Float32Array<ArrayBuffer> | null = null;
  private kMeterBuf: Float32Array<ArrayBuffer> | null = null;
  private origMeterBuf: Float32Array<ArrayBuffer> | null = null;
  private cleanMeterBuf: Float32Array<ArrayBuffer> | null = null;
  private lufsHistory: Array<{ t: number; ms: number; weight: number }> = [];
  private lastMeterSampleTime = 0;
  private smoothNrDb = 0;
  private peakHoldDbfs = METER_FLOOR_DB;

  constructor() {
    this.els = {
      original: this.makeElement(),
      cleaned: this.makeElement(),
    };
    for (const deck of ['original', 'cleaned'] as const) {
      const el = this.els[deck];
      el.addEventListener('ended', () => {
        if (deck === this.active) {
          if (this.loopEnabledFlag) {
            const bounds = this.effectiveLoopBounds();
            const restartAt = bounds ? bounds[0] : 0;
            this.restartLoopAt(restartAt);
            return;
          }
          this.wantPlaying = false;
          this.pauseAll();
          this.emit();
        }
      });
      el.addEventListener('loadedmetadata', () => {
        const deferred = this.seekOnMeta[deck];
        if (deferred !== null) {
          this.seekOnMeta[deck] = null;
          if (this.state[deck] === 'ready' && Math.abs(el.currentTime - deferred) > SYNC_TOLERANCE_S) {
            el.currentTime = deferred;
          }
        }
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
            if (
              !this.disposed &&
              this.wantPlaying &&
              this.active === deck &&
              el.paused &&
              !el.ended &&
              this.state[deck] === 'ready'
            ) {
              void el.play().catch(() => {
                if (this.disposed) return;
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

  private bothDecksReady(): boolean {
    return this.state.original === 'ready' && this.state.cleaned === 'ready';
  }

  /** Before the Web Audio graph exists, A/B is done with element mute. */
  private applyMuteFallback(): void {
    if (this.ctx) return;
    if (this.deltaModeEnabled && this.bothDecksReady()) {
      // HTMLAudioElement cannot invert phase; fail soft by keeping Original audible.
      this.els.original.muted = false;
      this.els.cleaned.muted = true;
      return;
    }
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
  private stopBufferSources(): void {
    this.bufferSourceToken += 1;
    this.bufferPlaying = false;
    for (const deck of ['original', 'cleaned'] as const) {
      const src = this.bufferSources[deck];
      this.bufferSources[deck] = null;
      if (src) {
        src.onended = null;
        try {
          src.stop();
        } catch {
          // Ignore if already stopped.
        }
        try {
          src.disconnect();
        } catch {
          // Ignore disconnect errors.
        }
      }
    }
  }

  private clearSeekThrottle(): void {
    if (this.seekThrottleTimer !== null) {
      clearTimeout(this.seekThrottleTimer);
      this.seekThrottleTimer = null;
    }
    this.pendingScrubTime = null;
  }

  private teardownGraph(closeContext = false): void {
    this.stopBufferSources();
    this.clearSeekThrottle();
    try {
      this.gains?.original.disconnect();
      this.gains?.cleaned.disconnect();
      this.deltaInverter?.disconnect();
      this.deltaGain?.disconnect();
      this.analyser?.disconnect();
      this.kStage1?.disconnect();
      this.kStage2?.disconnect();
      this.kAnalyser?.disconnect();
      this.origAnalyser?.disconnect();
      this.cleanAnalyser?.disconnect();
      this.hpfStage1?.disconnect();
      this.hpfStage2?.disconnect();
      this.hpfDryGain?.disconnect();
      this.hpfWetGain?.disconnect();
      this.soloFilter?.disconnect();
      this.soloDryGain?.disconnect();
      this.soloWetGain?.disconnect();
      this.corrSplitter?.disconnect();
      this.corrAnalyserL?.disconnect();
      this.corrAnalyserR?.disconnect();
    } catch {
      // Ignore disconnect errors on torn-down or closed contexts.
    }
    const ctx = this.ctx;
    this.ctx = null;
    this.gains = null;
    this.deltaInverter = null;
    this.deltaGain = null;
    this.analyser = null;
    this.kStage1 = null;
    this.kStage2 = null;
    this.kAnalyser = null;
    this.origAnalyser = null;
    this.cleanAnalyser = null;
    this.hpfStage1 = null;
    this.hpfStage2 = null;
    this.hpfDryGain = null;
    this.hpfWetGain = null;
    this.soloFilter = null;
    this.soloDryGain = null;
    this.soloWetGain = null;
    this.corrSplitter = null;
    this.corrAnalyserL = null;
    this.corrAnalyserR = null;
    if (closeContext && ctx && ctx.state !== 'closed') {
      try {
        void ctx.close().catch(() => undefined);
      } catch {
        // Ignore synchronous close errors.
      }
    }
  }

  private ensureGraph(): void {
    if (this.disposed) return;
    if (this.ctx && this.ctx.state !== 'closed') return;
    if (this.ctx && this.ctx.state === 'closed') {
      this.teardownGraph(false);
    }
    try {
      this.buildGraph();
    } catch {
      // Left null on purpose: `applyMuteFallback()` owns A/B from here.
      this.teardownGraph(false);
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

    // Meter-only stereo phase correlation branch (ChannelSplitter -> L/R analysers).
    let corrSplitter: ChannelSplitterNode | null = null;
    let corrAnalyserL: AnalyserNode | null = null;
    let corrAnalyserR: AnalyserNode | null = null;
    if (typeof ctx.createChannelSplitter === 'function') {
      try {
        const splitter = ctx.createChannelSplitter(2);
        const aL = ctx.createAnalyser();
        const aR = ctx.createAnalyser();
        if (aL && aR && aL !== analyser && aR !== analyser && aL !== aR) {
          aL.fftSize = 1024;
          aR.fftSize = 1024;
          analyser.connect(splitter);
          splitter.connect(aL, 0);
          splitter.connect(aR, 1);
          corrSplitter = splitter;
          corrAnalyserL = aL;
          corrAnalyserR = aR;
        }
      } catch {
        // Fail soft on mock or channel-limited AudioContexts.
      }
    }
    const gains = {
      original: ctx.createGain(),
      cleaned: ctx.createGain(),
    };
    const deltaInverter = ctx.createGain();
    deltaInverter.gain.value = -1;
    const deltaGain = ctx.createGain();
    deltaGain.gain.value = 0;

    const origSrc = ctx.createMediaElementSource(this.els.original);
    origSrc.connect(gains.original);

    // Parallel bit-exact dry bypass + real-time 24 dB/oct 4th-order Butterworth HPF
    // on Original audition so dragging the FabFilter HPF node sweeps rumble out at 0 ms
    // latency while leaving Cleaned masters and Δ Removed nulling 100% phase-untouched.
    let hpfStage1: BiquadFilterNode | null = null;
    let hpfStage2: BiquadFilterNode | null = null;
    let hpfDryGain: GainNode | null = null;
    let hpfWetGain: GainNode | null = null;
    if (typeof ctx.createBiquadFilter === 'function') {
      try {
        const dry = ctx.createGain();
        const wet = ctx.createGain();
        const hp1 = ctx.createBiquadFilter();
        const hp2 = ctx.createBiquadFilter();
        dry.gain.value = 1;
        wet.gain.value = 0;
        hp1.type = 'highpass';
        hp1.frequency.value = this.liveHpfCutoffHz;
        hp1.Q.value = 0.5411961;
        hp2.type = 'highpass';
        hp2.frequency.value = this.liveHpfCutoffHz;
        hp2.Q.value = 1.306563;
        gains.original.connect(dry);
        dry.connect(analyser);
        gains.original.connect(hp1);
        hp1.connect(hp2);
        hp2.connect(wet);
        wet.connect(analyser);
        hpfStage1 = hp1;
        hpfStage2 = hp2;
        hpfDryGain = dry;
        hpfWetGain = wet;
      } catch {
        gains.original.connect(analyser);
      }
    } else {
      gains.original.connect(analyser);
    }

    const cleanSrc = ctx.createMediaElementSource(this.els.cleaned);
    cleanSrc.connect(gains.cleaned);
    gains.cleaned.connect(analyser);
    // Inverted cleaned branch for phase-cancelled Delta ("Listen to Removed") audition:
    // Original (+1) + Cleaned (-gainOffset) sums cleanly to Original - Cleaned.
    cleanSrc.connect(deltaInverter);
    deltaInverter.connect(deltaGain);
    deltaGain.connect(analyser);

    // ITU-R BS.1770 K-weighting chain ahead of dedicated LUFS metering analyser
    // (Stage 1: high-shelf +4.0 dB @ 1681 Hz, Stage 2: high-pass 38 Hz Q=0.5).
    // Never connected to ctx.destination so audition audio remains completely uncolored.
    let kStage1: BiquadFilterNode | null = null;
    let kStage2: BiquadFilterNode | null = null;
    let kAnalyser: AnalyserNode | null = null;
    let origAnalyser: AnalyserNode | null = null;
    let cleanAnalyser: AnalyserNode | null = null;

    if (typeof ctx.createBiquadFilter === 'function') {
      try {
        const s1 = ctx.createBiquadFilter();
        s1.type = 'highshelf';
        s1.frequency.value = 1681.97;
        s1.gain.value = 4.0;
        s1.Q.value = 0.7071;

        const s2 = ctx.createBiquadFilter();
        s2.type = 'highpass';
        s2.frequency.value = 38.135;
        s2.Q.value = 0.5003;

        const ka = ctx.createAnalyser();
        if (ka && ka !== analyser) {
          ka.fftSize = 4096;
          ka.smoothingTimeConstant = 0;
          analyser.connect(s1);
          s1.connect(s2);
          s2.connect(ka);
          kStage1 = s1;
          kStage2 = s2;
          kAnalyser = ka;
        }

        const oa = ctx.createAnalyser();
        const ca = ctx.createAnalyser();
        if (oa && ca && oa !== analyser && ca !== analyser && oa !== ca) {
          oa.fftSize = 2048;
          ca.fftSize = 2048;
          origSrc.connect(oa);
          cleanSrc.connect(ca);
          origAnalyser = oa;
          cleanAnalyser = ca;
        }
      } catch {
        // Fail soft if biquad filters or secondary analysers are unsupported.
      }
    }

    for (const deck of ['original', 'cleaned'] as const) {
      // Element-level mute was the pre-graph A/B mechanism; from here on the
      // gain nodes own it (a muted element also mutes its source node).
      this.els[deck].muted = false;
    }
    this.ctx = ctx;
    this.gains = gains;
    this.deltaInverter = deltaInverter;
    this.deltaGain = deltaGain;
    this.analyser = analyser;
    this.kStage1 = kStage1;
    this.kStage2 = kStage2;
    this.kAnalyser = kAnalyser;
    this.origAnalyser = origAnalyser;
    this.cleanAnalyser = cleanAnalyser;
    this.hpfStage1 = hpfStage1;
    this.hpfStage2 = hpfStage2;
    this.hpfDryGain = hpfDryGain;
    this.hpfWetGain = hpfWetGain;
    this.soloFilter = null;
    this.soloDryGain = null;
    this.soloWetGain = null;
    this.corrSplitter = corrSplitter;
    this.corrAnalyserL = corrAnalyserL;
    this.corrAnalyserR = corrAnalyserR;
    this.applyTargetGain(true);
    if (this.soloActive) {
      this.applyBandSolo(true);
    }

    // Trigger background decode for any in-memory decks already fetched.
    for (const deck of ['original', 'cleaned'] as const) {
      if (this.rawBytes[deck] && !this.buffers[deck] && this.state[deck] === 'ready') {
        void this.decodeDeckBuffer(deck, this.epoch[deck]);
      }
    }
  }

  /**
   * Decode in-memory audio bytes into an AudioBuffer for sample-locked
   * AudioBufferSourceNode playback (0-sample skew for Delta & Wet/Dry Mix).
   */
  private decodeDeckBuffer(deck: Deck, gen: number): Promise<void> {
    if (!this.ctx || typeof this.ctx.decodeAudioData !== 'function') {
      return Promise.resolve();
    }
    const raw = this.rawBytes[deck];
    if (!raw || raw.byteLength === 0) return Promise.resolve();
    if (this.buffers[deck]) return Promise.resolve();
    if (this.decodeInFlight[deck]) return this.decodeInFlight[deck]!;

    const ctx = this.ctx;
    let p: Promise<void> | null = null;
    p = (async () => {
      try {
        // Slice a fresh copy because decodeAudioData detaches its ArrayBuffer.
        const copy = raw.slice(0);
        const decoded = await ctx.decodeAudioData(copy);
        if (this.disposed || this.epoch[deck] !== gen || this.ctx !== ctx) return;
        if (decoded && Number.isFinite(decoded.duration) && decoded.duration > 0) {
          this.buffers[deck] = decoded;
        }
      } catch {
        // Seamless fallback to <audio> element streaming when decodeAudioData fails.
      } finally {
        if (this.decodeInFlight[deck] === p) {
          this.decodeInFlight[deck] = null;
        }
      }
    })();
    this.decodeInFlight[deck] = p;
    return p;
  }

  private canUseBufferMode(): boolean {
    if (!this.ctx || this.ctx.state === 'closed' || !this.gains) return false;
    if (typeof this.ctx.createBufferSource !== 'function') return false;
    if (!this.buffers[this.active]) return false;
    const other: Deck = this.active === 'original' ? 'cleaned' : 'original';
    if (this.state[other] === 'ready' && !this.buffers[other]) return false;
    return true;
  }

  private startBufferPlaybackAt(timelinePos: number): boolean {
    if (!this.canUseBufferMode() || !this.ctx || !this.gains) return false;
    this.stopBufferSources();
    for (const deck of ['original', 'cleaned'] as const) {
      this.els[deck].pause();
    }
    const ctx = this.ctx;
    const t0 = ctx.currentTime + 0.005;
    const token = ++this.bufferSourceToken;
    let startedActive = false;

    for (const deck of ['original', 'cleaned'] as const) {
      if (this.state[deck] !== 'ready') continue;
      const buf = this.buffers[deck];
      if (!buf) continue;
      const deckOffset = this.timelineToDeck(deck, timelinePos);
      if (deckOffset >= buf.duration) continue;

      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(this.gains[deck]);
      if (deck === 'original' && this.origAnalyser) {
        src.connect(this.origAnalyser);
      } else if (deck === 'cleaned') {
        if (this.deltaInverter) src.connect(this.deltaInverter);
        if (this.cleanAnalyser) src.connect(this.cleanAnalyser);
      }

      src.onended = () => {
        if (this.disposed || token !== this.bufferSourceToken || !this.bufferPlaying) return;
        if (deck !== this.active) return;
        if (this.loopEnabledFlag) {
          const bounds = this.effectiveLoopBounds();
          const restartAt = bounds ? bounds[0] : 0;
          this.restartLoopAt(restartAt);
          return;
        }
        this.wantPlaying = false;
        this.bufferPosition = this.duration;
        this.pauseAll();
        this.emit();
      };

      // Schedule both decks on the exact same AudioContext.currentTime frame (0-sample skew).
      src.start(t0, deckOffset);
      this.bufferSources[deck] = src;
      if (deck === this.active) startedActive = true;
    }

    if (!startedActive) {
      this.stopBufferSources();
      return false;
    }
    this.bufferStartCtxTime = t0;
    this.bufferStartOffset = timelinePos;
    this.bufferPosition = timelinePos;
    this.bufferPlaying = true;
    return true;
  }

  /** Momentary 10 ms gain ducking during rapid scrub seeks to eliminate click/stutter. */
  private duckForSeek(): void {
    if (!this.gains || !this.ctx || this.ctx.state === 'closed') return;
    try {
      const now = this.ctx.currentTime;
      const branch = this.computeBranchGains();
      const half = SEEK_DUCK_S * 0.4;
      const duckNode = (param: AudioParam, target: number): void => {
        param.cancelScheduledValues(now);
        param.setValueAtTime(param.value, now);
        param.linearRampToValueAtTime(0, now + half);
        param.linearRampToValueAtTime(target, now + SEEK_DUCK_S);
      };
      duckNode(this.gains.original.gain, branch.original);
      duckNode(this.gains.cleaned.gain, branch.cleaned);
      if (this.deltaGain) {
        duckNode(this.deltaGain.gain, branch.delta);
      }
    } catch {
      // Ignore if context closed mid-duck.
    }
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

  /**
   * Compute target linear gains for the 3 WebAudio branches:
   *  - `original`: positive Original deck gain
   *  - `cleaned`: positive Cleaned deck gain
   *  - `delta`: gain feeding the -1 inverter on Cleaned (Original - Cleaned)
   */
  get auditionMode(): AuditionMode {
    return deriveAuditionMode({
      active: this.active,
      bothDecksReady: this.bothDecksReady(),
      deltaMode: this.deltaModeEnabled,
      wetDryMix: this.wetDryMixValue,
    });
  }

  private computeBranchGains(): { original: number; cleaned: number; delta: number } {
    const cleanTarget = this.getTargetGain('cleaned');
    switch (this.auditionMode) {
      case 'delta_removed':
        return { original: 1.0, cleaned: 0, delta: cleanTarget };
      case 'wet_dry_blend': {
        const mix = this.wetDryMixValue;
        return { original: 1 - mix, cleaned: mix * cleanTarget, delta: 0 };
      }
      case 'cleaned':
        return { original: 0, cleaned: cleanTarget, delta: 0 };
      case 'original':
        return { original: 1.0, cleaned: 0, delta: 0 };
    }
  }

  private rampParam(param: AudioParam, target: number, now: number, immediate: boolean): void {
    if (immediate) {
      param.value = target;
      return;
    }
    param.cancelScheduledValues(now);
    param.setValueAtTime(param.value, now);
    param.linearRampToValueAtTime(target, now + FADE_S);
  }

  private applyTargetGain(immediate = false): void {
    const branch = this.computeBranchGains();
    if (this.gains && this.ctx && this.ctx.state !== 'closed') {
      try {
        const now = this.ctx.currentTime;
        this.rampParam(this.gains.original.gain, branch.original, now, immediate);
        this.rampParam(this.gains.cleaned.gain, branch.cleaned, now, immediate);
        if (this.deltaGain) {
          this.rampParam(this.deltaGain.gain, branch.delta, now, immediate);
        }
        this.applyLiveHpf(immediate);
        return;
      } catch {
        // Fall through to mute fallback if context closed mid-ramp.
      }
    }
    if (this.ctx && this.ctx.state === 'closed') {
      this.teardownGraph(false);
    }
    const targetCleaned = this.getTargetGain('cleaned');
    this.els.cleaned.volume = Math.min(1, Math.max(0, targetCleaned));
    this.applyMuteFallback();
  }

  /**
   * Real-time 24 dB/oct Butterworth HPF audition on Original playback.
   * Automatically bypassed on Cleaned masters, Wet/Dry blend, and Δ Removed nulling.
   */
  setLiveHpf(enabled: boolean, cutoffHz: number): void {
    const clamped = Number.isFinite(cutoffHz) ? Math.max(20, Math.min(250, cutoffHz)) : 80;
    const nextEnabled = Boolean(enabled);
    if (nextEnabled === this.liveHpfEnabled && Math.abs(clamped - this.liveHpfCutoffHz) < 0.01) {
      return;
    }
    this.liveHpfEnabled = nextEnabled;
    this.liveHpfCutoffHz = clamped;
    this.applyLiveHpf(false);
  }

  private applyLiveHpf(immediate = false): void {
    if (!this.ctx || this.ctx.state === 'closed') return;
    if (!this.hpfStage1 || !this.hpfStage2 || !this.hpfDryGain || !this.hpfWetGain) return;
    const activeHpf = this.liveHpfEnabled && this.auditionMode === 'original';
    try {
      const now = this.ctx.currentTime;
      for (const stage of [this.hpfStage1, this.hpfStage2]) {
        if (!immediate && typeof stage.frequency.setTargetAtTime === 'function') {
          stage.frequency.setTargetAtTime(this.liveHpfCutoffHz, now, 0.015);
        } else {
          stage.frequency.value = this.liveHpfCutoffHz;
        }
      }
      this.rampParam(this.hpfDryGain.gain, activeHpf ? 0 : 1, now, immediate);
      this.rampParam(this.hpfWetGain.gain, activeHpf ? 1 : 0, now, immediate);
    } catch {
      // Fail soft if AudioContext closed mid-ramp.
    }
  }

  /**
   * FabFilter-style real-time bandpass frequency solo audition.
   * Isolates a narrow band around `centerHz` (20–18000 Hz) with quality factor `q` (1.0–12.0)
   * at 0 ms latency without altering the spectrum analyser or LUFS meters.
   */
  setBandSolo(active: boolean, centerHz?: number, q?: number): void {
    const nextActive = Boolean(active);
    const maxHz = Math.min(18000, Math.max(4000, this.sampleRate * 0.45));
    const nextHz =
      typeof centerHz === 'number' && Number.isFinite(centerHz)
        ? Math.max(20, Math.min(maxHz, centerHz))
        : this.soloCenterHz;
    const nextQ =
      typeof q === 'number' && Number.isFinite(q) ? Math.max(1.0, Math.min(12.0, q)) : this.soloQ;
    if (
      nextActive === this.soloActive &&
      Math.abs(nextHz - this.soloCenterHz) < 0.1 &&
      Math.abs(nextQ - this.soloQ) < 0.01
    ) {
      return;
    }
    this.soloActive = nextActive;
    this.soloCenterHz = nextHz;
    this.soloQ = nextQ;
    this.applyBandSolo(false);
  }

  get bandSolo(): BandSoloState {
    return { active: this.soloActive, hz: this.soloCenterHz, q: this.soloQ };
  }

  private ensureSoloBranch(): void {
    if (!this.ctx || this.ctx.state === 'closed' || !this.analyser) return;
    if (this.soloFilter && this.soloDryGain && this.soloWetGain) return;
    if (typeof this.ctx.createBiquadFilter !== 'function') return;
    try {
      const sDry = this.ctx.createGain();
      const sWet = this.ctx.createGain();
      const sBp = this.ctx.createBiquadFilter();
      sDry.gain.value = 1;
      sWet.gain.value = 0;
      sBp.type = 'bandpass';
      sBp.frequency.value = this.soloCenterHz;
      sBp.Q.value = this.soloQ;
      try {
        this.analyser.disconnect(this.ctx.destination);
      } catch {
        // Ignore if mock AudioNode does not support selective disconnect.
      }
      this.analyser.connect(sDry);
      sDry.connect(this.ctx.destination);
      this.analyser.connect(sBp);
      sBp.connect(sWet);
      sWet.connect(this.ctx.destination);
      this.soloFilter = sBp;
      this.soloDryGain = sDry;
      this.soloWetGain = sWet;
    } catch {
      // Fail soft if biquad creation fails.
    }
  }

  private applyBandSolo(immediate = false): void {
    if (!this.ctx || this.ctx.state === 'closed') return;
    if (this.soloActive) {
      this.ensureSoloBranch();
    }
    if (!this.soloFilter || !this.soloDryGain || !this.soloWetGain) return;
    try {
      const now = this.ctx.currentTime;
      if (!immediate && typeof this.soloFilter.frequency.setTargetAtTime === 'function') {
        this.soloFilter.frequency.setTargetAtTime(this.soloCenterHz, now, 0.012);
        this.soloFilter.Q.setTargetAtTime(this.soloQ, now, 0.012);
      } else {
        this.soloFilter.frequency.value = this.soloCenterHz;
        this.soloFilter.Q.value = this.soloQ;
      }
      // Subtle makeup gain (+2.5 dB when narrow Q) keeps soloed bands easily audible without clipping.
      const wetGain = this.soloActive ? Math.min(1.8, 1.0 + Math.max(0, this.soloQ - 1) * 0.08) : 0;
      this.rampParam(this.soloDryGain.gain, this.soloActive ? 0 : 1, now, immediate);
      this.rampParam(this.soloWetGain.gain, wetGain, now, immediate);
    } catch {
      // Fail soft if AudioContext closed mid-ramp.
    }
  }

  /**
   * Enable or disable Delta ("Listen to Removed": Original - Cleaned) audition.
   * Requires both decks to be ready; fails soft cleanly when WebAudio is unavailable.
   */
  setDeltaMode(enabled: boolean): void {
    const next = Boolean(enabled) && this.bothDecksReady();
    if (next === this.deltaModeEnabled) return;
    this.deltaModeEnabled = next;
    if (next) {
      // Keep both decks sample-locked (accounting for any region-preview offset)
      // before summing phase-inverted Cleaned.
      const t = this.time;
      const other: Deck = this.active === 'original' ? 'cleaned' : 'original';
      const expectedOther = this.timelineToDeck(other, t);
      if (
        Number.isFinite(expectedOther) &&
        Math.abs(this.els[other].currentTime - expectedOther) > SYNC_TOLERANCE_S
      ) {
        this.els[other].currentTime = expectedOther;
      }
      if (this.wantPlaying && this.els[other].paused) {
        void this.els[other].play().catch(() => undefined);
      }
    }
    this.applyTargetGain();
    this.emit();
  }

  toggleDeltaMode(): void {
    this.setDeltaMode(!this.deltaModeEnabled);
  }

  get isDeltaMode(): boolean {
    return this.deltaModeEnabled && this.bothDecksReady();
  }

  /** Enable or disable loop playback (loops loopRange if set, else full clip). */
  setLoopEnabled(enabled: boolean): void {
    const next = Boolean(enabled);
    if (next === this.loopEnabledFlag) return;
    this.loopEnabledFlag = next;
    this.emit();
  }

  toggleLoop(): void {
    this.setLoopEnabled(!this.loopEnabledFlag);
  }

  get isLoopEnabled(): boolean {
    return this.loopEnabledFlag;
  }

  /** Set active loop region [startS, endS], or null to loop the full clip. */
  setLoopRange(range: [number, number] | null): void {
    let next: [number, number] | null = null;
    if (range !== null) {
      const [rawStart, rawEnd] = range;
      if (Number.isFinite(rawStart) && Number.isFinite(rawEnd)) {
        const start = Math.max(0, rawStart);
        const end = Math.max(0, rawEnd);
        if (end > start) {
          next = [start, end];
        }
      }
    }
    const same =
      (this.loopRangeValue === null && next === null) ||
      (this.loopRangeValue !== null &&
        next !== null &&
        this.loopRangeValue[0] === next[0] &&
        this.loopRangeValue[1] === next[1]);
    if (same) return;
    this.loopRangeValue = next;
    this.emit();
  }

  get loopRange(): [number, number] | null {
    return this.loopRangeValue ? [this.loopRangeValue[0], this.loopRangeValue[1]] : null;
  }

  /**
   * Set Live Wet/Dry Mix in [0, 1].
   * When active deck is 'cleaned' and not in deltaMode, blends Original (1 - mix)
   * with Cleaned (mix * levelMatchGain) in WebAudio.
   */
  setWetDryMix(mix: number): void {
    if (!Number.isFinite(mix)) return;
    const clamped = Math.max(0, Math.min(1, mix));
    if (clamped === this.wetDryMixValue) return;
    this.wetDryMixValue = clamped;
    this.applyTargetGain();
    this.emit();
  }

  get wetDryMix(): number {
    return this.wetDryMixValue;
  }

  private effectiveLoopBounds(): [number, number] | null {
    const maxTime = this.duration;
    if (this.loopRangeValue) {
      const [rawStart, rawEnd] = this.loopRangeValue;
      const start = Math.max(0, maxTime > 0 ? Math.min(rawStart, maxTime) : rawStart);
      const end = maxTime > 0 ? Math.min(rawEnd, maxTime) : rawEnd;
      if (end > start) return [start, end];
    }
    if (maxTime > 0) return [0, maxTime];
    return null;
  }

  private deckToTimeline(deck: Deck, deckTime: number): number {
    return deckTime + this.offset[deck];
  }

  private timelineToDeck(deck: Deck, timelineTime: number): number {
    const raw = timelineTime - this.offset[deck];
    const bufDur = this.buffers[deck]?.duration;
    const elDur = this.els[deck].duration;
    const d =
      typeof bufDur === 'number' && Number.isFinite(bufDur) && bufDur > 0
        ? bufDur
        : elDur;
    if (Number.isFinite(d) && d > 0) {
      return Math.max(0, Math.min(d, raw));
    }
    return Math.max(0, raw);
  }

  private restartLoopAt(loopStart: number): void {
    this.bufferPosition = loopStart;
    if (this.wantPlaying && this.canUseBufferMode()) {
      this.startBufferPlaybackAt(loopStart);
      this.emit();
      return;
    }
    for (const deck of ['original', 'cleaned'] as const) {
      if (this.state[deck] !== 'ready') continue;
      const el = this.els[deck];
      el.currentTime = this.timelineToDeck(deck, loopStart);
      if (this.wantPlaying && el.paused) {
        void el.play().catch(() => undefined);
      }
    }
    this.emit();
  }

  /**
   * Live broadcast-style ITU-R BS.1770 K-weighted LUFS (Momentary 400ms, Short-Term 3s),
   * Peak / True-Peak Hold, and real-time Noise Reduction (NR dB Removed) comparing
   * Original vs Cleaned energy.
   * Returns clean finite floor (-70) when silent or when no AudioContext exists.
   */
  getLiveMeterLevels(): LiveMeterLevels {
    const analyser = this.analyser;
    if (!analyser || (this.ctx && this.ctx.state === 'closed')) {
      return {
        momentaryLufs: METER_FLOOR_DB,
        shortTermLufs: METER_FLOOR_DB,
        peakDbfs: METER_FLOOR_DB,
        truePeakHoldDbfs: this.peakHoldDbfs,
        nrDb: null,
      };
    }

    const size = analyser.fftSize > 0 ? analyser.fftSize : 2048;
    if (!this.meterBuf || this.meterBuf.length !== size) {
      this.meterBuf = new Float32Array(size);
    }
    try {
      analyser.getFloatTimeDomainData(this.meterBuf);
    } catch {
      return {
        momentaryLufs: METER_FLOOR_DB,
        shortTermLufs: METER_FLOOR_DB,
        peakDbfs: METER_FLOOR_DB,
        truePeakHoldDbfs: this.peakHoldDbfs,
        nrDb: null,
      };
    }

    // Unweighted sample peak + 4-point Catmull-Rom cubic inter-sample true peak.
    const buf = this.meterBuf;
    let samplePeak = 0;
    for (let i = 0; i < buf.length; i++) {
      const raw = buf[i] ?? 0;
      const s = Number.isFinite(raw) ? raw : 0;
      const a = Math.abs(s);
      if (a > samplePeak) samplePeak = a;
    }

    let truePeak = samplePeak;
    if (samplePeak > 1e-4 && buf.length >= 4) {
      const thresh = samplePeak * 0.65;
      for (let i = 1; i < buf.length - 2; i++) {
        const y1 = buf[i] ?? 0;
        const y2 = buf[i + 1] ?? 0;
        if (Math.abs(y1) < thresh && Math.abs(y2) < thresh) continue;
        const y0 = buf[i - 1] ?? 0;
        const y3 = buf[i + 2] ?? 0;
        for (const t of [0.25, 0.5, 0.75]) {
          const t2 = t * t;
          const t3 = t2 * t;
          const interp =
            0.5 *
            (2 * y1 +
              (-y0 + y2) * t +
              (2 * y0 - 5 * y1 + 4 * y2 - y3) * t2 +
              (-y0 + 3 * y1 - 3 * y2 + y3) * t3);
          const absI = Math.abs(interp);
          if (absI > truePeak) truePeak = absI;
        }
      }
    }

    // K-weighted signal for ITU-R BS.1770 LUFS integration (fallback to main analyser when biquad unsupported).
    let kBuf: Float32Array<ArrayBuffer> = buf;
    if (this.kAnalyser) {
      const kSize = this.kAnalyser.fftSize > 0 ? this.kAnalyser.fftSize : 2048;
      if (!this.kMeterBuf || this.kMeterBuf.length !== kSize) {
        this.kMeterBuf = new Float32Array(kSize);
      }
      try {
        this.kAnalyser.getFloatTimeDomainData(this.kMeterBuf);
        kBuf = this.kMeterBuf;
      } catch {
        kBuf = buf;
      }
    }

    // Deduplicate overlapping AnalyserNode time-domain samples based on elapsed wall time.
    const nowMs = typeof performance !== 'undefined' ? performance.now() : Date.now();
    const sr = this.sampleRate;
    const bufDurationMs = (kBuf.length / Math.max(8000, sr)) * 1000;
    const dtMs = this.lastMeterSampleTime > 0 ? nowMs - this.lastMeterSampleTime : 0;
    const freshCount =
      dtMs > 0 && dtMs < bufDurationMs
        ? Math.min(kBuf.length, Math.max(1, Math.round((dtMs / 1000) * sr)))
        : kBuf.length;
    this.lastMeterSampleTime = nowMs;

    const startIdx = Math.max(0, kBuf.length - freshCount);
    let sumSq = 0;
    for (let i = startIdx; i < kBuf.length; i++) {
      const raw = kBuf[i] ?? 0;
      const s = Number.isFinite(raw) ? raw : 0;
      sumSq += s * s;
    }
    const ms = freshCount > 0 ? sumSq / freshCount : 0;

    const peakDbfs =
      samplePeak > 1e-7 ? Math.max(METER_FLOOR_DB, 20 * Math.log10(samplePeak)) : METER_FLOOR_DB;
    const truePeakDbfs =
      truePeak > 1e-7 ? Math.max(METER_FLOOR_DB, 20 * Math.log10(truePeak)) : METER_FLOOR_DB;
    if (truePeakDbfs > this.peakHoldDbfs) {
      this.peakHoldDbfs = truePeakDbfs;
    }

    this.lufsHistory.push({ t: nowMs, ms, weight: freshCount });
    // Prune entries older than the 3-second Short-Term window.
    const cutoffShort = nowMs - SHORT_TERM_WINDOW_MS;
    while (this.lufsHistory.length > 1 && (this.lufsHistory[0]?.t ?? nowMs) < cutoffShort) {
      this.lufsHistory.shift();
    }

    const cutoffMomentary = nowMs - MOMENTARY_WINDOW_MS;
    let momWeightedSum = 0;
    let momWeight = 0;
    let shortWeightedSum = 0;
    let shortWeight = 0;
    for (const entry of this.lufsHistory) {
      shortWeightedSum += entry.ms * entry.weight;
      shortWeight += entry.weight;
      if (entry.t >= cutoffMomentary) {
        momWeightedSum += entry.ms * entry.weight;
        momWeight += entry.weight;
      }
    }
    const momentaryMs = momWeight > 0 ? momWeightedSum / momWeight : ms;
    const shortTermMs = shortWeight > 0 ? shortWeightedSum / shortWeight : ms;

    const momentaryLufs =
      momentaryMs > 1e-7
        ? Math.max(METER_FLOOR_DB, -0.691 + 10 * Math.log10(momentaryMs))
        : METER_FLOOR_DB;
    const shortTermLufs =
      shortTermMs > 1e-7
        ? Math.max(METER_FLOOR_DB, -0.691 + 10 * Math.log10(shortTermMs))
        : METER_FLOOR_DB;

    // Instantaneous Noise Reduction (NR dB Removed): compare Original vs level-matched Cleaned RMS.
    let nrDb: number | null = null;
    if (this.bothDecksReady() && this.origAnalyser && this.cleanAnalyser) {
      const oSize = this.origAnalyser.fftSize > 0 ? this.origAnalyser.fftSize : 2048;
      const cSize = this.cleanAnalyser.fftSize > 0 ? this.cleanAnalyser.fftSize : 2048;
      if (!this.origMeterBuf || this.origMeterBuf.length !== oSize) {
        this.origMeterBuf = new Float32Array(oSize);
      }
      if (!this.cleanMeterBuf || this.cleanMeterBuf.length !== cSize) {
        this.cleanMeterBuf = new Float32Array(cSize);
      }
      try {
        this.origAnalyser.getFloatTimeDomainData(this.origMeterBuf);
        this.cleanAnalyser.getFloatTimeDomainData(this.cleanMeterBuf);
        let oSq = 0;
        let cSq = 0;
        for (let i = 0; i < this.origMeterBuf.length; i++) {
          const v = this.origMeterBuf[i] ?? 0;
          if (Number.isFinite(v)) oSq += v * v;
        }
        for (let i = 0; i < this.cleanMeterBuf.length; i++) {
          const v = this.cleanMeterBuf[i] ?? 0;
          if (Number.isFinite(v)) cSq += v * v;
        }
        const rmsOrig = Math.sqrt(oSq / Math.max(1, this.origMeterBuf.length));
        const rmsClean =
          Math.sqrt(cSq / Math.max(1, this.cleanMeterBuf.length)) * this.getTargetGain('cleaned');
        if (rmsOrig > 1e-5) {
          const rawNr = Math.max(0, Math.min(48, 20 * Math.log10(rmsOrig / Math.max(1e-6, rmsClean))));
          const alpha = rawNr > this.smoothNrDb ? 0.35 : 0.14;
          this.smoothNrDb = this.smoothNrDb * (1 - alpha) + rawNr * alpha;
          nrDb = this.smoothNrDb;
        } else {
          this.smoothNrDb *= 0.85;
          nrDb = this.smoothNrDb > 0.05 ? this.smoothNrDb : 0;
        }
      } catch {
        nrDb = null;
      }
    }

    // Real-time stereo phase correlation r = sum(L*R) / sqrt(sum(L^2) * sum(R^2)).
    let stereoCorrelation: number | null = null;
    if (this.corrAnalyserL && this.corrAnalyserR) {
      const lSize = this.corrAnalyserL.fftSize > 0 ? this.corrAnalyserL.fftSize : 1024;
      const rSize = this.corrAnalyserR.fftSize > 0 ? this.corrAnalyserR.fftSize : 1024;
      if (!this.corrBufL || this.corrBufL.length !== lSize) {
        this.corrBufL = new Float32Array(lSize);
      }
      if (!this.corrBufR || this.corrBufR.length !== rSize) {
        this.corrBufR = new Float32Array(rSize);
      }
      try {
        this.corrAnalyserL.getFloatTimeDomainData(this.corrBufL);
        this.corrAnalyserR.getFloatTimeDomainData(this.corrBufR);
        const n = Math.min(this.corrBufL.length, this.corrBufR.length);
        let dot = 0;
        let eL = 0;
        let eR = 0;
        for (let i = 0; i < n; i++) {
          const lv = this.corrBufL[i] ?? 0;
          const rv = this.corrBufR[i] ?? 0;
          if (Number.isFinite(lv) && Number.isFinite(rv)) {
            dot += lv * rv;
            eL += lv * lv;
            eR += rv * rv;
          }
        }
        const denom = Math.sqrt(eL * eR);
        if (denom > 1e-8) {
          const raw = Math.max(-1, Math.min(1, dot / denom));
          this.smoothCorr =
            this.smoothCorr === null ? raw : this.smoothCorr * 0.72 + raw * 0.28;
          stereoCorrelation = this.smoothCorr;
        }
      } catch {
        stereoCorrelation = null;
      }
    }

    return {
      momentaryLufs,
      shortTermLufs,
      peakDbfs,
      truePeakHoldDbfs: this.peakHoldDbfs,
      nrDb,
      stereoCorrelation,
    };
  }

  resetPeakHold(): void {
    this.peakHoldDbfs = METER_FLOOR_DB;
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
  load(
    deck: Deck,
    url: string | null,
    expectedDuration: number | null = null,
    regionOffsetS?: number,
  ): void {
    if (this.disposed) return;
    if (url === null) {
      this.retire(deck);
      return;
    }
    const isPreview = regionOffsetS !== undefined;
    const safeOffset =
      isPreview && Number.isFinite(regionOffsetS) && regionOffsetS > 0 ? regionOffsetS : 0;
    // Asking for a deck again is a fresh attempt: whatever it failed at last
    // time is no longer the current answer.
    this.faults[deck] = null;
    this.expected[deck] = expectedDuration;
    if (
      this.wanted[deck] === url &&
      this.offset[deck] === safeOffset &&
      this.isPreviewDeck[deck] === isPreview &&
      this.state[deck] !== 'empty'
    ) {
      return;
    }
    const isNewOriginalSource =
      deck === 'original' && this.attached.original !== url && this.wanted.original !== url;
    if (isNewOriginalSource) {
      this.deltaModeEnabled = false;
      this.loopRangeValue = null;
      this.loopEnabledFlag = false;
      this.resetPeakHold();
      this.lufsHistory = [];
      this.lastMeterSampleTime = 0;
      this.smoothNrDb = 0;
      this.smoothCorr = null;
      this.soloActive = false;
      this.applyBandSolo(true);
      this.bufferPosition = 0;
    }
    this.rawBytes[deck] = null;
    this.buffers[deck] = null;
    this.decodeInFlight[deck] = null;
    if (this.bufferPlaying) {
      this.bufferPosition = this.time;
      this.stopBufferSources();
    }
    // Preserve timeline position when reloading the currently active cleaned deck
    // (e.g. live unit re-stitch or spectral box repair while playing/scrubbed),
    // never when switching `original` to a brand-new source.
    const prevActiveTime =
      !isNewOriginalSource && deck === this.active && this.state[deck] === 'ready'
        ? this.time
        : null;
    this.wanted[deck] = url;
    this.offset[deck] = safeOffset;
    this.isPreviewDeck[deck] = isPreview;
    this.durationJudged[deck] = false;
    this.probeFailed[deck] = false;
    this.seekOnMeta[deck] = null;
    this.pendingSeekTime[deck] =
      prevActiveTime !== null && Number.isFinite(prevActiveTime) && prevActiveTime > 0
        ? prevActiveTime
        : null;
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
              if (this.epoch[deck] === gen) {
                src = URL.createObjectURL(blob);
                if (typeof blob.arrayBuffer === 'function') {
                  try {
                    const ab = await blob.arrayBuffer();
                    if (this.epoch[deck] === gen && ab && ab.byteLength > 0) {
                      this.rawBytes[deck] = ab;
                    }
                  } catch {
                    // Fail soft to <audio> blob playback.
                  }
                }
              }
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
    if (this.rawBytes[deck] && this.ctx && typeof this.ctx.decodeAudioData === 'function') {
      void this.decodeDeckBuffer(deck, this.epoch[deck]);
    }
    this.afterAttach(deck);
  }

  private afterAttach(deck: Deck): void {
    const el = this.els[deck];
    this.applyMuteFallback();
    // Apply any pending seek recorded across an active-deck reload, or keep a
    // freshly loaded inactive deck aligned with the current transport position.
    const pendingSeek = this.pendingSeekTime[deck];
    this.pendingSeekTime[deck] = null;
    if (pendingSeek !== null && Number.isFinite(pendingSeek)) {
      const deckT = this.timelineToDeck(deck, pendingSeek);
      el.currentTime = deckT;
      if (el.readyState < 1 && deckT > 0) this.seekOnMeta[deck] = deckT;
    } else if (deck !== this.active) {
      const t = this.time;
      if (Number.isFinite(t) && t > 0) {
        const deckT = this.timelineToDeck(deck, t);
        el.currentTime = deckT;
        if (el.readyState < 1 && deckT > 0) this.seekOnMeta[deck] = deckT;
      }
    }
    if (this.pendingActive === deck) {
      this.pendingActive = null;
      this.setActive(deck);
    } else {
      if (this.ctx && this.gains) {
        this.applyTargetGain();
      }
      if (this.wantPlaying) {
        void el.play().catch(() => undefined);
        if (deck === this.active) {
          this.startTicker();
        }
      }
    }
    this.emit();
  }

  private revokeBlob(deck: Deck): void {
    const blob = this.objectUrl[deck];
    if (blob) {
      URL.revokeObjectURL(blob);
      this.objectUrl[deck] = null;
      // A revoked blob URL cannot be revived on the element without a fresh fetch.
      this.attached[deck] = null;
    }
  }

  /**
   * Take a deck out of service without touching its `src`. Removing the
   * attribute (or calling `load()` on an empty element) is what used to abort
   * the deck's open range request; the element keeps the source it has, stays
   * silent, and reports itself unavailable until something is loaded onto it.
   * Any `blob:` handle owned by this deck is revoked immediately to prevent
   * retaining large in-memory audio buffers across source switches.
   */
  private retire(deck: Deck): void {
    this.faults[deck] = null;
    if (this.state[deck] === 'empty' && this.wanted[deck] === null) {
      this.revokeBlob(deck);
      if (deck === 'original') {
        this.loopRangeValue = null;
        this.loopEnabledFlag = false;
      }
      return;
    }
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
    this.offset[deck] = 0;
    this.isPreviewDeck[deck] = false;
    this.pendingSeekTime[deck] = null;
    this.seekOnMeta[deck] = null;
    this.state[deck] = 'empty';
    this.rawBytes[deck] = null;
    this.buffers[deck] = null;
    this.decodeInFlight[deck] = null;
    if (this.bufferPlaying) {
      this.bufferPosition = this.time;
      this.stopBufferSources();
    }
    this.deltaModeEnabled = false;
    if (deck === 'original') {
      this.loopRangeValue = null;
      this.loopEnabledFlag = false;
    }
    if (this.pendingActive === deck) this.pendingActive = null;
    const el = this.els[deck];
    el.pause();
    this.revokeBlob(deck);
    if (deck === this.active && this.wantPlaying && !keepPlaying) {
      this.wantPlaying = false;
      this.playToken += 1;
      this.pauseAll();
    }
    if (this.gains && this.ctx && this.ctx.state !== 'closed') {
      try {
        const g = this.gains[deck].gain;
        g.cancelScheduledValues(this.ctx.currentTime);
        g.setValueAtTime(0, this.ctx.currentTime);
        if (this.deltaGain) {
          const dg = this.deltaGain.gain;
          dg.cancelScheduledValues(this.ctx.currentTime);
          dg.setValueAtTime(0, this.ctx.currentTime);
        }
      } catch {
        el.muted = true;
      }
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
    } else if (this.state[this.active] === 'ready') {
      // Ensure the surviving active deck holds full unity/target gain (not wet/dry blended).
      this.applyTargetGain();
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
    if (this.disposed) return;
    if (deck === this.active && !this.deltaModeEnabled) {
      if (this.pendingActive !== null) {
        this.pendingActive = null;
        this.emit();
      }
      return;
    }
    if (this.state[deck] === 'empty') return;
    if (this.state[deck] === 'loading') {
      // Arrive at the deck as soon as it has a source.
      if (this.pendingActive !== deck) {
        this.pendingActive = deck;
        this.emit();
      }
      return;
    }
    const prev = this.active;
    this.active = deck;
    this.pendingActive = null;
    this.deltaModeEnabled = false;
    // In sample-locked AudioBufferSourceNode mode both decks are already running
    // on the exact same AudioContext frame — switching A/B is a pure 6 ms gain crossfade.
    if (this.bufferPlaying) {
      this.applyTargetGain();
      this.emit();
      return;
    }
    const from = this.els[prev];
    const to = this.els[deck];
    const maxTime = this.duration;
    let timelineTime = this.deckToTimeline(prev, from.currentTime);
    if (!Number.isFinite(timelineTime)) timelineTime = 0;
    if (maxTime > 0 && timelineTime > maxTime) {
      timelineTime = maxTime;
    }
    const expectedFrom = this.timelineToDeck(prev, timelineTime);
    if (Number.isFinite(from.currentTime) && Math.abs(from.currentTime - expectedFrom) > SYNC_TOLERANCE_S) {
      from.currentTime = expectedFrom;
    }
    const expectedTo = this.timelineToDeck(deck, timelineTime);
    // A/B has to stay sample-locked; when both decks are in memory (the normal
    // case) this correction costs nothing at all.
    if (Math.abs(to.currentTime - expectedTo) > SYNC_TOLERANCE_S) {
      to.currentTime = expectedTo;
    }
    this.applyTargetGain();
    if (this.wantPlaying && to.paused) {
      void to.play().catch(() => undefined);
      this.startTicker();
    }
    this.emit();
  }

  async play(): Promise<void> {
    if (this.disposed || this.state[this.active] === 'empty') return;
    this.wantPlaying = true;
    const token = ++this.playToken;
    this.ensureGraph();
    if (this.ctx && this.ctx.state === 'suspended') {
      try {
        await this.ctx.resume();
      } catch {
        // Autoplay or closed-context resume rejections fail soft cleanly.
      }
    }
    if (this.disposed || !this.wantPlaying || this.playToken !== token) return;
    if (this.state[this.active] === 'empty') {
      this.wantPlaying = false;
      return;
    }
    // Active deck is reloading (e.g. live unit re-stitch or spectral repair):
    // retain wantPlaying so `afterAttach` resumes playback seamlessly.
    if (this.state[this.active] !== 'ready') {
      this.emit();
      return;
    }
    // Decode any in-memory decks so AudioBufferSourceNode playback starts sample-locked.
    if (this.ctx && typeof this.ctx.decodeAudioData === 'function') {
      const decodes: Promise<void>[] = [];
      for (const deck of ['original', 'cleaned'] as const) {
        if (this.state[deck] === 'ready' && this.rawBytes[deck] && !this.buffers[deck]) {
          decodes.push(this.decodeDeckBuffer(deck, this.epoch[deck]));
        }
      }
      if (decodes.length > 0) {
        await Promise.all(decodes);
        if (this.disposed || !this.wantPlaying || this.playToken !== token) return;
      }
    }
    if (this.loopEnabledFlag && this.loopRangeValue) {
      const bounds = this.effectiveLoopBounds();
      if (bounds) {
        const [loopStart, loopEnd] = bounds;
        const cur = this.time;
        if (cur < loopStart || cur >= loopEnd) {
          this.applySeekImmediate(loopStart, false);
        }
      }
    } else if (this.duration > 0 && this.time >= this.duration) {
      this.applySeekImmediate(0, false);
    }
    const t = this.time;
    if (this.canUseBufferMode() && this.startBufferPlaybackAt(t)) {
      this.applyTargetGain(true);
      this.startTicker();
      this.emit();
      return;
    }
    const plays: Promise<void>[] = [];
    for (const deck of ['original', 'cleaned'] as const) {
      if (this.state[deck] !== 'ready') continue;
      const el = this.els[deck];
      const expected = this.timelineToDeck(deck, t);
      if (deck !== this.active && Math.abs(el.currentTime - expected) > SYNC_TOLERANCE_S) {
        el.currentTime = expected;
      }
      plays.push(el.play().catch(() => undefined));
    }
    await Promise.all(plays);
    if (this.disposed || !this.wantPlaying || this.playToken !== token) return;
    this.startTicker();
    this.emit();
  }

  pause(): void {
    this.wantPlaying = false;
    this.playToken += 1;
    this.pauseAll();
    this.emit();
  }

  toggle(): void {
    if (this.wantPlaying) this.pause();
    else void this.play();
  }

  private applySeekImmediate(t: number, duck: boolean): void {
    this.bufferPosition = t;
    if (duck) {
      this.duckForSeek();
    }
    if (this.bufferPlaying && this.wantPlaying && this.canUseBufferMode()) {
      this.startBufferPlaybackAt(t);
    }
    for (const deck of ['original', 'cleaned'] as const) {
      this.seekOnMeta[deck] = null;
      if (this.state[deck] === 'ready') {
        this.els[deck].currentTime = this.timelineToDeck(deck, t);
        this.pendingSeekTime[deck] = null;
      } else if (this.state[deck] === 'loading') {
        this.pendingSeekTime[deck] = t;
      }
    }
  }

  seek(time: number): void {
    if (this.disposed || !Number.isFinite(time)) return;
    const d = this.duration;
    const t = Math.max(0, Math.min(d > 0 ? d : time, time));
    if (!this.wantPlaying) {
      this.clearSeekThrottle();
      this.applySeekImmediate(t, false);
      this.emit();
      return;
    }
    const nowMs = typeof performance !== 'undefined' ? performance.now() : Date.now();
    const elapsed = nowMs - this.lastSeekAppliedMs;
    if (elapsed >= SEEK_THROTTLE_MS || this.lastSeekAppliedMs === 0) {
      this.clearSeekThrottle();
      this.lastSeekAppliedMs = nowMs;
      this.applySeekImmediate(t, true);
      this.emit();
      return;
    }
    // Rapid scrubbing while playing: keep UI playhead responsive immediately and
    // throttle audio seek + 10 ms ducking to a clean 32 ms cadence.
    this.pendingScrubTime = t;
    if (this.seekThrottleTimer === null) {
      const waitMs = Math.max(1, SEEK_THROTTLE_MS - elapsed);
      this.seekThrottleTimer = setTimeout(() => {
        this.seekThrottleTimer = null;
        if (this.disposed) return;
        const target = this.pendingScrubTime;
        this.pendingScrubTime = null;
        if (target === null) return;
        this.lastSeekAppliedMs = typeof performance !== 'undefined' ? performance.now() : Date.now();
        this.applySeekImmediate(target, this.wantPlaying);
        this.emit();
      }, waitMs);
    }
    this.emit();
  }

  get time(): number {
    if (this.pendingScrubTime !== null && Number.isFinite(this.pendingScrubTime)) {
      return this.pendingScrubTime;
    }
    if (this.state[this.active] === 'loading') {
      const pending = this.pendingSeekTime[this.active];
      if (pending !== null && Number.isFinite(pending)) return pending;
    }
    if (this.bufferPlaying && this.ctx) {
      const elapsed = Math.max(0, this.ctx.currentTime - this.bufferStartCtxTime);
      const raw = this.bufferStartOffset + elapsed;
      const d = this.duration;
      return d > 0 ? Math.min(d, raw) : raw;
    }
    const t = this.els[this.active].currentTime;
    return Number.isFinite(t) ? this.deckToTimeline(this.active, t) : this.bufferPosition;
  }

  get duration(): number {
    const origReady = this.state.original === 'ready';
    const cleanReady = this.state.cleaned === 'ready';
    const origBufDur = this.buffers.original?.duration;
    const cleanBufDur = this.buffers.cleaned?.duration;
    const dOrig = origReady
      ? Number.isFinite(this.els.original.duration) && this.els.original.duration > 0
        ? this.els.original.duration
        : typeof origBufDur === 'number' && Number.isFinite(origBufDur) && origBufDur > 0
          ? origBufDur
          : null
      : null;
    const dClean = cleanReady
      ? Number.isFinite(this.els.cleaned.duration) && this.els.cleaned.duration > 0
        ? this.els.cleaned.duration
        : typeof cleanBufDur === 'number' && Number.isFinite(cleanBufDur) && cleanBufDur > 0
          ? cleanBufDur
          : null
      : null;

    if (dOrig !== null && dClean !== null) {
      if (this.isPreviewDeck.cleaned || this.isPreviewDeck.original) {
        return dOrig;
      }
      return Math.min(dOrig, dClean);
    }
    if (dOrig !== null) return dOrig;
    if (dClean !== null) return dClean + this.offset.cleaned;
    return 0;
  }

  get playing(): boolean {
    return this.wantPlaying && (this.bufferPlaying || !this.els[this.active].paused);
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
      deltaMode: this.isDeltaMode,
      loopEnabled: this.loopEnabledFlag,
      loopRange: this.loopRange,
      wetDryMix: this.wetDryMixValue,
      auditionMode: this.auditionMode,
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
    this.clearSeekThrottle();
    if (this.bufferPlaying) {
      const pos = this.time;
      this.bufferPosition = pos;
      this.stopBufferSources();
      for (const deck of ['original', 'cleaned'] as const) {
        if (this.state[deck] === 'ready') {
          this.els[deck].currentTime = this.timelineToDeck(deck, pos);
        }
      }
    }
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
      const maxTime = this.duration;
      const bothReady = this.bothDecksReady();

      if (this.bufferPlaying) {
        const t = this.time;
        if (this.loopEnabledFlag) {
          const bounds = this.effectiveLoopBounds();
          if (bounds) {
            const [loopStart, loopEnd] = bounds;
            if (t >= loopEnd) {
              this.restartLoopAt(loopStart);
              this.raf = window.requestAnimationFrame(tick);
              return;
            }
          }
        }
        if (maxTime > 0 && t >= maxTime) {
          this.wantPlaying = false;
          this.bufferPosition = maxTime;
          this.pauseAll();
          this.emit();
          return;
        }
        this.emit();
        this.raf = window.requestAnimationFrame(tick);
        return;
      }

      const a = this.els[this.active];
      const other: Deck = this.active === 'original' ? 'cleaned' : 'original';
      const b = this.els[other];
      const tA = this.deckToTimeline(this.active, a.currentTime);
      const tB = this.deckToTimeline(other, b.currentTime);
      if (this.loopEnabledFlag) {
        const bounds = this.effectiveLoopBounds();
        if (bounds) {
          const [loopStart, loopEnd] = bounds;
          if (tA >= loopEnd || (bothReady && tB >= loopEnd)) {
            this.restartLoopAt(loopStart);
            this.raf = window.requestAnimationFrame(tick);
            return;
          }
        }
      }
      if (bothReady && maxTime > 0 && (tA >= maxTime || tB >= maxTime)) {
        this.wantPlaying = false;
        this.pauseAll();
        if (tA > maxTime) a.currentTime = this.timelineToDeck(this.active, maxTime);
        if (tB > maxTime) b.currentTime = this.timelineToDeck(other, maxTime);
        this.emit();
        return;
      }
      // Drift guard: keep the inactive deck within tolerance of the active one.
      if (this.state[other] === 'ready' && !b.paused) {
        const expectedB = this.timelineToDeck(other, tA);
        if (Math.abs(b.currentTime - expectedB) > 0.08) {
          b.currentTime = expectedB;
        }
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
    this.disposed = true;
    this.wantPlaying = false;
    this.playToken += 1;
    this.stopTicker();
    this.clearSeekThrottle();
    this.stopBufferSources();
    this.listeners.clear();
    this.faultListeners.clear();
    this.faults = { original: null, cleaned: null };
    this.pendingActive = null;
    this.deltaModeEnabled = false;
    this.loopEnabledFlag = false;
    this.loopRangeValue = null;
    this.offset = { original: 0, cleaned: 0 };
    this.isPreviewDeck = { original: false, cleaned: false };
    this.pendingSeekTime = { original: null, cleaned: null };
    this.seekOnMeta = { original: null, cleaned: null };
    this.rawBytes = { original: null, cleaned: null };
    this.buffers = { original: null, cleaned: null };
    this.decodeInFlight = { original: null, cleaned: null };
    this.bufferPosition = 0;
    this.lufsHistory = [];
    this.lastMeterSampleTime = 0;
    this.smoothNrDb = 0;
    this.smoothCorr = null;
    this.soloActive = false;
    this.meterBuf = null;
    this.kMeterBuf = null;
    this.origMeterBuf = null;
    this.cleanMeterBuf = null;
    this.corrBufL = null;
    this.corrBufR = null;
    this.peakHoldDbfs = METER_FLOOR_DB;
    for (const deck of ['original', 'cleaned'] as const) {
      const el = this.els[deck];
      this.epoch[deck] += 1;
      el.pause();
      el.removeAttribute('src');
      el.load();
      el.remove();
      this.revokeBlob(deck);
      this.wanted[deck] = null;
      this.attached[deck] = null;
      this.state[deck] = 'empty';
    }
    this.teardownGraph(true);
  }
}

let singleton: DualPlayer | null = null;
export function getPlayer(): DualPlayer {
  if (!singleton) singleton = new DualPlayer();
  return singleton;
}
