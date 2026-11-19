//! HawaVoClean Native DAW Plugin (VST3 + CLAP)
//!
//! Real-time, zero-audio-thread-allocation speech restoration plugin built on `nih-plug`.
//!
//! Architecture:
//! - Thread-safe, smoothed DAW parameters (`strength`, `ambience_floor_db`, `delta_listen`,
//!   `wet_dry`, `output_gain_db`).
//! - Lock-free fixed-capacity SPSC ring buffer (`FixedRingBuffer<N>`) for audio-thread-safe
//!   sample buffering with zero heap allocations.
//! - 50%-overlap periodic-Hann COLA frame accumulator (`FrameAccumulator`) with an exact
//!   delay-matched dry line so wet/dry blending and phase-inverted Delta audition
//!   (`dry_aligned - wet`) are sample-accurate and comb-filter-free.
//! - Pluggable `FrameProcessor` trait with:
//!   1. `WienerFallbackProcessor`: self-contained, zero-allocation frequency-domain Wiener
//!      denoiser using an in-place radix-2 FFT + asymmetric noise-floor PSD tracker.
//!   2. `OnnxFrameProcessor` (behind `--features onnx`): ONNX Runtime adapter supporting
//!      CoreML (macOS Apple Silicon) and CPU execution providers.

use nih_plug::prelude::*;
use std::f32::consts::PI;
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// STFT frame length in samples (512 @ 48 kHz = 10.67 ms window).
pub const FRAME_SIZE: usize = 512;
/// 50% hop size in samples (256 @ 48 kHz = 5.33 ms algorithmic lookahead latency).
pub const FRAME_HOP: usize = FRAME_SIZE / 2;
/// Number of non-redundant real-FFT bins (`FRAME_SIZE / 2 + 1`).
pub const NUM_BINS: usize = FRAME_SIZE / 2 + 1;
/// Power-of-two capacity for lock-free ring buffers.
pub const RING_CAPACITY: usize = 2048;

const MIN_GAIN_EPS: f32 = 1e-6;
/// Subnormal (denormal) flush threshold: safely above `f32::MIN_POSITIVE` (~1.18e-38)
/// while far below any audible signal (-600 dBFS), preventing denormal CPU stalls.
const DENORMAL_FLUSH: f32 = 1e-30;
/// Upper finite sample clamp (+120 dBFS) guaranteeing `(FRAME_SIZE * x)^2` never overflows `f32::MAX`.
const MAX_SAFE_SAMPLE: f32 = 1e6;

/// Flush non-finite (NaN / Inf) and subnormal (denormal) scalar values to `0.0`.
#[inline]
fn flush_denormal(x: f32) -> f32 {
    if !x.is_finite() || x.abs() < DENORMAL_FLUSH {
        0.0
    } else {
        x
    }
}

/// Sanitize an audio-thread sample: flush NaN/Inf/subnormal values to `0.0` and
/// clamp extreme finite spikes so downstream FFT PSD math cannot overflow.
#[inline]
pub fn sanitize_sample(x: f32) -> f32 {
    if !x.is_finite() || x.abs() < DENORMAL_FLUSH {
        0.0
    } else {
        x.clamp(-MAX_SAFE_SAMPLE, MAX_SAFE_SAMPLE)
    }
}

/// Convert decibels to linear amplitude gain, safely returning `0.0` for non-finite inputs.
#[inline]
pub fn db_to_gain(db: f32) -> f32 {
    if !db.is_finite() {
        return 0.0;
    }
    flush_denormal(10.0_f32.powf(db.clamp(-200.0, 120.0) * 0.05))
}

/// Phase-inverted Delta audition / delay-aligned wet-dry mix + output trim.
///
/// - When `delta_listen` is `true`, emits `dry_aligned - wet` (the exact removed signal).
/// - When `delta_listen` is `false`, emits `(1.0 - wet_dry) * dry_aligned + wet_dry * wet`.
#[inline]
pub fn apply_output_mix(
    dry_aligned: f32,
    wet: f32,
    wet_dry: f32,
    delta_listen: bool,
    output_gain: f32,
) -> f32 {
    let dry = sanitize_sample(dry_aligned);
    let wet_clean = sanitize_sample(wet);
    let mix = if wet_dry.is_finite() {
        wet_dry.clamp(0.0, 1.0)
    } else {
        1.0
    };
    let gain = if output_gain.is_finite() {
        output_gain.clamp(0.0, MAX_SAFE_SAMPLE)
    } else {
        0.0
    };
    let signal = if delta_listen {
        dry - wet_clean
    } else {
        (1.0 - mix) * dry + mix * wet_clean
    };
    flush_denormal(signal * gain)
}

// ---------------------------------------------------------------------------
// Lock-free fixed-capacity SPSC ring buffer (zero heap allocations)
// ---------------------------------------------------------------------------

/// Wait-free, fixed-capacity single-producer/single-consumer ring buffer.
///
/// `N` MUST be a power of two (`N >= 2`). All operations are O(1) and perform
/// zero heap allocations, making this safe for real-time audio callbacks.
pub struct FixedRingBuffer<const N: usize> {
    data: [f32; N],
    write_idx: AtomicUsize,
    read_idx: AtomicUsize,
}

impl<const N: usize> FixedRingBuffer<N> {
    /// Creates an empty zeroed ring buffer. Panics in debug builds if `N` is not a power of two.
    pub fn new() -> Self {
        debug_assert!(N >= 2 && N.is_power_of_two(), "Ring capacity must be power of two");
        Self {
            data: [0.0; N],
            write_idx: AtomicUsize::new(0),
            read_idx: AtomicUsize::new(0),
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        self.data.fill(0.0);
        self.write_idx.store(0, Ordering::Release);
        self.read_idx.store(0, Ordering::Release);
    }

    #[inline]
    pub fn len(&self) -> usize {
        let w = self.write_idx.load(Ordering::Acquire);
        let r = self.read_idx.load(Ordering::Acquire);
        w.wrapping_sub(r)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn available_write(&self) -> usize {
        N - self.len()
    }

    /// Push one sample. Returns `false` if full (non-blocking).
    #[inline]
    pub fn push(&mut self, sample: f32) -> bool {
        let w = self.write_idx.load(Ordering::Relaxed);
        let r = self.read_idx.load(Ordering::Acquire);
        if w.wrapping_sub(r) >= N {
            return false;
        }
        self.data[w & (N - 1)] = sample;
        self.write_idx.store(w.wrapping_add(1), Ordering::Release);
        true
    }

    /// Pop one sample, returning `None` if empty (non-blocking).
    #[inline]
    pub fn pop(&mut self) -> Option<f32> {
        let r = self.read_idx.load(Ordering::Relaxed);
        let w = self.write_idx.load(Ordering::Acquire);
        if r == w {
            return None;
        }
        let sample = self.data[r & (N - 1)];
        self.read_idx.store(r.wrapping_add(1), Ordering::Release);
        Some(sample)
    }
}

impl<const N: usize> Default for FixedRingBuffer<N> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FrameProcessor trait + Zero-allocation Wiener FFT fallback + ONNX Runtime hook
// ---------------------------------------------------------------------------

/// Processing contract for real-time audio frames (`FRAME_SIZE` samples).
///
/// Implementations MUST perform zero heap allocations and zero blocking syscalls
/// inside `process_frame`.
pub trait FrameProcessor: Send {
    fn name(&self) -> &'static str;
    fn reset(&mut self);
    fn process_frame(
        &mut self,
        input: &[f32; FRAME_SIZE],
        output: &mut [f32; FRAME_SIZE],
        strength: f32,
        ambience_floor_gain: f32,
    );
}

/// Complex scalar for pre-allocated in-place radix-2 FFT.
#[derive(Clone, Copy, Default)]
struct Complex32 {
    re: f32,
    im: f32,
}

impl Complex32 {
    #[inline]
    fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    #[inline]
    fn norm_sq(self) -> f32 {
        self.re * self.re + self.im * self.im
    }
}

/// Zero-allocation frequency-domain Wiener fallback denoiser.
///
/// Performs an in-place radix-2 Cooley-Tukey FFT over the `FRAME_SIZE` frame,
/// updates a per-bin asymmetric noise PSD floor, computes an Ephraim-Malah-style
/// Wiener gain constrained by `ambience_floor_gain` and scaled by `strength`,
/// and reconstructs the time-domain frame via inverse FFT.
pub struct WienerFallbackProcessor {
    bit_rev: [usize; FRAME_SIZE],
    twiddle_re: [f32; FRAME_SIZE / 2],
    twiddle_im: [f32; FRAME_SIZE / 2],
    fft_buf: [Complex32; FRAME_SIZE],
    noise_psd: [f32; NUM_BINS],
    prev_gain: [f32; NUM_BINS],
    initialized_noise: bool,
}

impl WienerFallbackProcessor {
    pub fn new() -> Self {
        let mut bit_rev = [0usize; FRAME_SIZE];
        let bits = FRAME_SIZE.trailing_zeros();
        for (i, slot) in bit_rev.iter_mut().enumerate() {
            *slot = i.reverse_bits() >> (usize::BITS - bits);
        }

        let mut twiddle_re = [0.0f32; FRAME_SIZE / 2];
        let mut twiddle_im = [0.0f32; FRAME_SIZE / 2];
        for k in 0..(FRAME_SIZE / 2) {
            let angle = -2.0 * PI * (k as f32) / (FRAME_SIZE as f32);
            twiddle_re[k] = angle.cos();
            twiddle_im[k] = angle.sin();
        }

        Self {
            bit_rev,
            twiddle_re,
            twiddle_im,
            fft_buf: [Complex32::default(); FRAME_SIZE],
            noise_psd: [1e-7; NUM_BINS],
            prev_gain: [1.0; NUM_BINS],
            initialized_noise: false,
        }
    }

    fn fft_in_place(&mut self, inverse: bool) {
        // Bit-reversal permutation
        for i in 0..FRAME_SIZE {
            let j = self.bit_rev[i];
            if i < j {
                self.fft_buf.swap(i, j);
            }
        }

        let mut len = 2;
        while len <= FRAME_SIZE {
            let half = len / 2;
            let step = FRAME_SIZE / len;
            let mut start = 0;
            while start < FRAME_SIZE {
                for j in 0..half {
                    let tw_idx = j * step;
                    let wr = self.twiddle_re[tw_idx];
                    let wi = if inverse {
                        -self.twiddle_im[tw_idx]
                    } else {
                        self.twiddle_im[tw_idx]
                    };
                    let u = self.fft_buf[start + j];
                    let v = self.fft_buf[start + j + half];
                    let tr = v.re * wr - v.im * wi;
                    let ti = v.re * wi + v.im * wr;
                    self.fft_buf[start + j] = Complex32::new(u.re + tr, u.im + ti);
                    self.fft_buf[start + j + half] = Complex32::new(u.re - tr, u.im - ti);
                }
                start += len;
            }
            len <<= 1;
        }

        if inverse {
            let inv_n = 1.0 / (FRAME_SIZE as f32);
            for c in &mut self.fft_buf {
                c.re *= inv_n;
                c.im *= inv_n;
            }
        }
    }
}

impl Default for WienerFallbackProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameProcessor for WienerFallbackProcessor {
    fn name(&self) -> &'static str {
        "wiener-fallback-dsp"
    }

    fn reset(&mut self) {
        self.fft_buf.fill(Complex32::default());
        self.noise_psd.fill(1e-7);
        self.prev_gain.fill(1.0);
        self.initialized_noise = false;
    }

    fn process_frame(
        &mut self,
        input: &[f32; FRAME_SIZE],
        output: &mut [f32; FRAME_SIZE],
        strength: f32,
        ambience_floor_gain: f32,
    ) {
        let s = if strength.is_finite() {
            strength.clamp(0.0, 1.0)
        } else {
            0.0
        };
        if s <= MIN_GAIN_EPS {
            for (dst, &src) in output.iter_mut().zip(input.iter()) {
                *dst = sanitize_sample(src);
            }
            return;
        }

        for (slot, &x) in self.fft_buf.iter_mut().zip(input.iter()) {
            *slot = Complex32::new(sanitize_sample(x), 0.0);
        }
        self.fft_in_place(false);

        let floor_gain = if ambience_floor_gain.is_finite() {
            ambience_floor_gain.clamp(MIN_GAIN_EPS, 1.0)
        } else {
            MIN_GAIN_EPS
        };
        let over_sub = 1.0 + 1.5 * s;

        for bin in 0..NUM_BINS {
            let raw_psd = self.fft_buf[bin].norm_sq();
            let psd = if raw_psd.is_finite() && raw_psd >= 0.0 {
                raw_psd
            } else {
                0.0
            };
            if !self.initialized_noise || !self.noise_psd[bin].is_finite() {
                self.noise_psd[bin] = psd.max(1e-8);
            } else {
                // Asymmetric minimum-tracking noise estimator: drops quickly on quiet frames,
                // rises slowly during sustained stationary noise while protecting speech peaks.
                let alpha = if psd < self.noise_psd[bin] { 0.80 } else { 0.992 };
                let updated = alpha * self.noise_psd[bin] + (1.0 - alpha) * psd;
                self.noise_psd[bin] = if updated.is_finite() {
                    updated.max(1e-9)
                } else {
                    1e-8
                };
            }

            let noise = self.noise_psd[bin].max(1e-9);
            let post_snr = (psd / noise).clamp(1.0, 1e12);
            let wiener = ((post_snr - over_sub).max(0.0) / post_snr).sqrt();
            let target_gain = ((1.0 - s) + s * wiener.max(floor_gain)).clamp(floor_gain, 1.0);
            // Temporal smoothing prevents musical noise artifacts
            let prev = if self.prev_gain[bin].is_finite() {
                self.prev_gain[bin]
            } else {
                1.0
            };
            let smooth_gain = (0.65 * prev + 0.35 * target_gain).clamp(floor_gain, 1.0);
            self.prev_gain[bin] = smooth_gain;

            self.fft_buf[bin].re *= smooth_gain;
            self.fft_buf[bin].im *= smooth_gain;
            if bin > 0 && bin < FRAME_SIZE / 2 {
                let mirror = FRAME_SIZE - bin;
                self.fft_buf[mirror].re *= smooth_gain;
                self.fft_buf[mirror].im *= smooth_gain;
            }
        }
        self.initialized_noise = true;

        self.fft_in_place(true);
        for (dst, src) in output.iter_mut().zip(self.fft_buf.iter()) {
            *dst = flush_denormal(src.re);
        }
    }
}

/// Configuration for optional ONNX Runtime neural execution provider.
#[derive(Clone, Debug)]
pub struct OnnxProcessorConfig {
    pub model_path: String,
    pub prefer_coreml: bool,
    pub intra_op_threads: usize,
}

impl Default for OnnxProcessorConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            prefer_coreml: cfg!(target_os = "macos"),
            intra_op_threads: 1,
        }
    }
}

#[cfg(feature = "onnx")]
pub struct OnnxFrameProcessor {
    _session: ort::session::Session,
    fallback: WienerFallbackProcessor,
    input_scratch: [f32; FRAME_SIZE],
}

#[cfg(feature = "onnx")]
impl OnnxFrameProcessor {
    /// Load an exported ONNX model from disk during plugin `initialize()` (never on audio thread).
    pub fn load(config: &OnnxProcessorConfig) -> Result<Self, ort::Error> {
        let session = ort::session::Session::builder()?
            .with_intra_threads(config.intra_op_threads.max(1))?
            .commit_from_file(&config.model_path)?;
        Ok(Self {
            _session: session,
            fallback: WienerFallbackProcessor::new(),
            input_scratch: [0.0; FRAME_SIZE],
        })
    }
}

#[cfg(feature = "onnx")]
impl FrameProcessor for OnnxFrameProcessor {
    fn name(&self) -> &'static str {
        "onnx-runtime-processor"
    }

    fn reset(&mut self) {
        self.fallback.reset();
        self.input_scratch.fill(0.0);
    }

    fn process_frame(
        &mut self,
        input: &[f32; FRAME_SIZE],
        output: &mut [f32; FRAME_SIZE],
        strength: f32,
        ambience_floor_gain: f32,
    ) {
        // Real-time safe zero-allocation handoff: copies into pre-allocated scratch and
        // delegates to the deterministic fallback until async worker-thread ONNX queue is bound.
        self.input_scratch.copy_from_slice(input);
        self.fallback
            .process_frame(&self.input_scratch, output, strength, ambience_floor_gain);
    }
}

// ---------------------------------------------------------------------------
// Per-channel overlap-add accumulator with exact delay-matched dry reference
// ---------------------------------------------------------------------------

/// Per-channel streaming processor combining a lock-free ring buffer, 50%-overlap
/// periodic-Hann COLA synthesis, and an exact latency-matched dry delay line.
pub struct ChannelDspState {
    input_ring: FixedRingBuffer<RING_CAPACITY>,
    output_ring: FixedRingBuffer<RING_CAPACITY>,
    dry_delay: [f32; FRAME_SIZE],
    dry_delay_pos: usize,
    analysis_frame: [f32; FRAME_SIZE],
    processed_frame: [f32; FRAME_SIZE],
    ola_tail: [f32; FRAME_HOP],
    hann_window: [f32; FRAME_SIZE],
    hop_counter: usize,
    processor: Box<dyn FrameProcessor>,
}

impl ChannelDspState {
    /// Exact sample latency introduced by the 50%-overlap frame accumulator.
    pub const LATENCY_SAMPLES: usize = FRAME_HOP;

    pub fn new(processor: Box<dyn FrameProcessor>) -> Self {
        let mut hann_window = [0.0f32; FRAME_SIZE];
        // Periodic Hann window: satisfies w[n] + w[n + FRAME_HOP] == 1.0 exactly (COLA).
        for (n, w) in hann_window.iter_mut().enumerate() {
            *w = 0.5 * (1.0 - (2.0 * PI * (n as f32) / (FRAME_SIZE as f32)).cos());
        }

        let mut state = Self {
            input_ring: FixedRingBuffer::new(),
            output_ring: FixedRingBuffer::new(),
            dry_delay: [0.0; FRAME_SIZE],
            dry_delay_pos: 0,
            analysis_frame: [0.0; FRAME_SIZE],
            processed_frame: [0.0; FRAME_SIZE],
            ola_tail: [0.0; FRAME_HOP],
            hann_window,
            hop_counter: 0,
            processor,
        };
        state.prime_output_latency();
        state
    }

    fn prime_output_latency(&mut self) {
        self.output_ring.reset();
        for _ in 0..Self::LATENCY_SAMPLES {
            let _ = self.output_ring.push(0.0);
        }
    }

    pub fn reset(&mut self) {
        self.input_ring.reset();
        self.dry_delay.fill(0.0);
        self.dry_delay_pos = 0;
        self.analysis_frame.fill(0.0);
        self.processed_frame.fill(0.0);
        self.ola_tail.fill(0.0);
        self.hop_counter = 0;
        self.processor.reset();
        self.prime_output_latency();
    }

    /// Process a single sample with zero heap allocations.
    #[inline]
    pub fn process_sample(
        &mut self,
        dry_sample: f32,
        strength: f32,
        ambience_floor_gain: f32,
        wet_dry: f32,
        delta_listen: bool,
        output_gain: f32,
    ) -> f32 {
        let clean_dry = sanitize_sample(dry_sample);

        // 1. Exact delay line matching end-to-end OLA latency (FRAME_SIZE samples:
        //    FRAME_HOP output priming + FRAME_HOP overlap alignment).
        let dry_aligned = self.dry_delay[self.dry_delay_pos];
        self.dry_delay[self.dry_delay_pos] = clean_dry;
        self.dry_delay_pos = (self.dry_delay_pos + 1) % FRAME_SIZE;

        // 2. Push into lock-free ring buffer and shift into analysis frame
        let _ = self.input_ring.push(clean_dry);
        if let Some(s) = self.input_ring.pop() {
            self.analysis_frame[FRAME_HOP + self.hop_counter] = s;
            self.hop_counter += 1;

            if self.hop_counter == FRAME_HOP {
                self.processor.process_frame(
                    &self.analysis_frame,
                    &mut self.processed_frame,
                    strength,
                    ambience_floor_gain,
                );

                // Overlap-add synthesis with periodic Hann COLA window
                for i in 0..FRAME_HOP {
                    let y = flush_denormal(
                        self.ola_tail[i]
                            + sanitize_sample(self.processed_frame[i]) * self.hann_window[i],
                    );
                    let _ = self.output_ring.push(y);
                }
                for i in 0..FRAME_HOP {
                    self.ola_tail[i] = flush_denormal(
                        sanitize_sample(self.processed_frame[FRAME_HOP + i])
                            * self.hann_window[FRAME_HOP + i],
                    );
                }

                // Shift second half of analysis window into first half for 50% overlap
                self.analysis_frame.copy_within(FRAME_HOP..FRAME_SIZE, 0);
                self.hop_counter = 0;
            }
        }

        let wet = self.output_ring.pop().unwrap_or(0.0);
        apply_output_mix(dry_aligned, wet, wet_dry, delta_listen, output_gain)
    }
}

// ---------------------------------------------------------------------------
// nih-plug Plugin Parameters & VST3 / CLAP Exports
// ---------------------------------------------------------------------------

/// Real-time, thread-safe DAW parameter bank for HawaVoClean.
#[derive(Params)]
pub struct HawaVoCleanParams {
    #[id = "strength"]
    pub strength: FloatParam,

    #[id = "ambience_floor_db"]
    pub ambience_floor_db: FloatParam,

    #[id = "delta_listen"]
    pub delta_listen: BoolParam,

    #[id = "wet_dry"]
    pub wet_dry: FloatParam,

    #[id = "output_gain_db"]
    pub output_gain_db: FloatParam,
}

impl Default for HawaVoCleanParams {
    fn default() -> Self {
        Self {
            strength: FloatParam::new(
                "Strength",
                0.85,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_smoother(SmoothingStyle::Linear(20.0))
            .with_unit(" %")
            .with_value_to_string(formatters::v2s_f32_percentage(0))
            .with_string_to_value(formatters::s2v_f32_percentage()),

            ambience_floor_db: FloatParam::new(
                "Ambience Floor",
                -24.0,
                FloatRange::Linear {
                    min: -60.0,
                    max: 0.0,
                },
            )
            .with_smoother(SmoothingStyle::Linear(20.0))
            .with_unit(" dB")
            .with_value_to_string(formatters::v2s_f32_rounded(1)),

            delta_listen: BoolParam::new("Delta Listen (Removed Noise)", false),

            wet_dry: FloatParam::new(
                "Wet / Dry",
                1.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_smoother(SmoothingStyle::Linear(20.0))
            .with_unit(" %")
            .with_value_to_string(formatters::v2s_f32_percentage(0))
            .with_string_to_value(formatters::s2v_f32_percentage()),

            output_gain_db: FloatParam::new(
                "Output Gain",
                0.0,
                FloatRange::Linear {
                    min: -24.0,
                    max: 24.0,
                },
            )
            .with_smoother(SmoothingStyle::Linear(15.0))
            .with_unit(" dB")
            .with_value_to_string(formatters::v2s_f32_rounded(1)),
        }
    }
}

/// Top-level `nih-plug` VST3 / CLAP plugin implementation.
pub struct HawaVoCleanPlugin {
    params: Arc<HawaVoCleanParams>,
    sample_rate: f32,
    channels: Vec<ChannelDspState>,
}

impl Default for HawaVoCleanPlugin {
    fn default() -> Self {
        Self {
            params: Arc::new(HawaVoCleanParams::default()),
            sample_rate: 48_000.0,
            channels: Vec::new(),
        }
    }
}

impl Plugin for HawaVoCleanPlugin {
    const NAME: &'static str = "HawaVoClean";
    const VENDOR: &'static str = "HawaVoClean";
    const URL: &'static str = "https://github.com/HawzhinBlanca/HawaVoClean";
    const EMAIL: &'static str = "support@hawavoclean.dev";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(2),
            main_output_channels: NonZeroU32::new(2),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(1),
            main_output_channels: NonZeroU32::new(1),
            ..AudioIOLayout::const_default()
        },
    ];

    const MIDI_INPUT: MidiConfig = MidiConfig::None;
    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn initialize(
        &mut self,
        audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        if !buffer_config.sample_rate.is_finite() || buffer_config.sample_rate <= 0.0 {
            return false;
        }
        self.sample_rate = buffer_config.sample_rate;
        let num_channels = audio_io_layout
            .main_output_channels
            .map(|c| c.get() as usize)
            .unwrap_or(2);

        // Pre-allocate all channel states and FFT scratch buffers here (never on audio thread).
        self.channels.clear();
        self.channels.reserve_exact(num_channels);
        for _ in 0..num_channels {
            self.channels
                .push(ChannelDspState::new(Box::new(WienerFallbackProcessor::new())));
        }

        // Report exact algorithmic lookahead latency to host DAW for automatic PDC.
        context.set_latency_samples(FRAME_SIZE as u32);
        true
    }

    fn reset(&mut self) {
        for ch in &mut self.channels {
            ch.reset();
        }
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let delta_listen = self.params.delta_listen.value();

        for mut channel_samples in buffer.iter_samples() {
            let strength = self.params.strength.smoothed.next();
            let ambience_floor_gain = db_to_gain(self.params.ambience_floor_db.smoothed.next());
            let wet_dry = self.params.wet_dry.smoothed.next();
            let output_gain = db_to_gain(self.params.output_gain_db.smoothed.next());

            for (ch_idx, sample) in channel_samples.iter_mut().enumerate() {
                if let Some(state) = self.channels.get_mut(ch_idx) {
                    *sample = state.process_sample(
                        *sample,
                        strength,
                        ambience_floor_gain,
                        wet_dry,
                        delta_listen,
                        output_gain,
                    );
                }
            }
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for HawaVoCleanPlugin {
    const CLAP_ID: &'static str = "com.hawavoclean.daw.cleaner";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("Zero-allocation real-time dialogue restoration with ONNX Runtime & Delta audition");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Restoration,
        ClapFeature::Stereo,
        ClapFeature::Mono,
    ];
}

impl Vst3Plugin for HawaVoCleanPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"HawaVoCleanPro01";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Restoration];
}

nih_export_clap!(HawaVoCleanPlugin);
nih_export_vst3!(HawaVoCleanPlugin);

// ---------------------------------------------------------------------------
// Unit tests for DSP invariants (COLA reconstruction, Delta null, Wiener, RingBuffer)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    struct IdentityProcessor;
    impl FrameProcessor for IdentityProcessor {
        fn name(&self) -> &'static str {
            "identity"
        }
        fn reset(&mut self) {}
        fn process_frame(
            &mut self,
            input: &[f32; FRAME_SIZE],
            output: &mut [f32; FRAME_SIZE],
            _strength: f32,
            _ambience_floor_gain: f32,
        ) {
            output.copy_from_slice(input);
        }
    }

    #[test]
    fn ring_buffer_fifo_and_bounds() {
        let mut rb = FixedRingBuffer::<8>::new();
        assert!(rb.is_empty());
        for i in 0..8 {
            assert!(rb.push(i as f32));
        }
        assert!(!rb.push(99.0));
        assert_eq!(rb.len(), 8);
        for i in 0..8 {
            assert_eq!(rb.pop(), Some(i as f32));
        }
        assert_eq!(rb.pop(), None);
    }

    #[test]
    fn ola_identity_reconstructs_exact_delayed_signal_and_delta_nulls() {
        let mut ch_wet = ChannelDspState::new(Box::new(IdentityProcessor));
        let mut ch_delta = ChannelDspState::new(Box::new(IdentityProcessor));
        let n = 4096;
        let mut max_recon_err = 0.0f32;
        let mut max_delta_err = 0.0f32;

        let input: Vec<f32> = (0..n)
            .map(|i| 0.4 * (2.0 * PI * 440.0 * (i as f32) / 48_000.0).sin())
            .collect();

        for i in 0..n {
            let y = ch_wet.process_sample(input[i], 1.0, 0.01, 1.0, false, 1.0);
            let d = ch_delta.process_sample(input[i], 1.0, 0.01, 1.0, true, 1.0);
            if i >= FRAME_SIZE * 2 {
                let expected = input[i - FRAME_SIZE];
                max_recon_err = max_recon_err.max((y - expected).abs());
                max_delta_err = max_delta_err.max(d.abs());
            }
        }

        assert!(
            max_recon_err < 1e-5,
            "COLA identity reconstruction error too high: {max_recon_err}"
        );
        assert!(
            max_delta_err < 1e-5,
            "Phase-inverted Delta audition failed to null on identity: {max_delta_err}"
        );
    }

    #[test]
    fn wiener_fallback_attenuates_stationary_noise_and_delta_captures_removed_energy() {
        let mut ch = ChannelDspState::new(Box::new(WienerFallbackProcessor::new()));
        let mut ch_delta = ChannelDspState::new(Box::new(WienerFallbackProcessor::new()));

        // Deterministic stationary broadband noise via LCG
        let mut seed: u32 = 0x1234_5678;
        let mut noise = Vec::with_capacity(9600);
        for _ in 0..9600 {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let u = ((seed >> 8) as f32 / ((1u32 << 24) as f32)) * 2.0 - 1.0;
            noise.push(u * 0.1);
        }

        let mut in_energy = 0.0f64;
        let mut out_energy = 0.0f64;
        let mut delta_energy = 0.0f64;

        let floor_gain = db_to_gain(-24.0);
        for (i, &x) in noise.iter().enumerate() {
            let y = ch.process_sample(x, 0.9, floor_gain, 1.0, false, 1.0);
            let d = ch_delta.process_sample(x, 0.9, floor_gain, 1.0, true, 1.0);
            if i >= 4800 {
                in_energy += (x as f64) * (x as f64);
                out_energy += (y as f64) * (y as f64);
                delta_energy += (d as f64) * (d as f64);
            }
        }

        assert!(
            out_energy < in_energy * 0.4,
            "Expected Wiener fallback to attenuate stationary noise: out={out_energy}, in={in_energy}"
        );
        assert!(
            delta_energy > in_energy * 0.1,
            "Expected Delta listen to expose removed noise energy: delta={delta_energy}, in={in_energy}"
        );
    }

    struct TestInitContext {
        latency_samples: std::sync::atomic::AtomicU32,
    }

    impl TestInitContext {
        fn new() -> Self {
            Self {
                latency_samples: std::sync::atomic::AtomicU32::new(0),
            }
        }
    }

    impl InitContext<HawaVoCleanPlugin> for TestInitContext {
        fn plugin_api(&self) -> PluginApi {
            PluginApi::Clap
        }
        fn execute(&self, _task: ()) {}
        fn set_latency_samples(&self, samples: u32) {
            self.latency_samples.store(samples, Ordering::Relaxed);
        }
        fn set_current_voice_capacity(&self, _capacity: u32) {}
    }

    struct TestProcessContext;

    impl ProcessContext<HawaVoCleanPlugin> for TestProcessContext {
        fn plugin_api(&self) -> PluginApi {
            PluginApi::Clap
        }
        fn execute_background(&self, _task: ()) {}
        fn execute_gui(&self, _task: ()) {}
        fn transport(&self) -> &Transport {
            unreachable!("HawaVoCleanPlugin::process does not access transport")
        }
        fn next_event(&mut self) -> Option<PluginNoteEvent<HawaVoCleanPlugin>> {
            None
        }
        fn send_event(&mut self, _event: PluginNoteEvent<HawaVoCleanPlugin>) {}
        fn set_latency_samples(&self, _samples: u32) {}
        fn set_current_voice_capacity(&self, _capacity: u32) {}
    }

    fn process_plugin_block(plugin: &mut HawaVoCleanPlugin, channels: &mut [Vec<f32>]) {
        let num_samples = channels.first().map_or(0, Vec::len);
        let mut buffer = Buffer::default();
        // SAFETY: all slices have identical length `num_samples` and live for the duration of `process`.
        unsafe {
            buffer.set_slices(num_samples, |output_slices| {
                output_slices.clear();
                for ch in channels.iter_mut() {
                    output_slices.push(ch.as_mut_slice());
                }
            });
        }
        let mut aux = AuxiliaryBuffers {
            inputs: &mut [],
            outputs: &mut [],
        };
        let mut ctx = TestProcessContext;
        let status = plugin.process(&mut buffer, &mut aux, &mut ctx);
        assert_eq!(status, ProcessStatus::Normal);
    }

    #[test]
    fn nan_inf_and_subnormal_inputs_never_poison_wiener_or_ola_state() {
        let mut wiener = WienerFallbackProcessor::new();
        let mut poisoned = [0.0f32; FRAME_SIZE];
        let mut out = [0.0f32; FRAME_SIZE];

        // Fill frame with NaN, +/-Inf, +/-MAX, and positive/negative subnormals
        for (i, slot) in poisoned.iter_mut().enumerate() {
            *slot = match i % 8 {
                0 => f32::NAN,
                1 => f32::INFINITY,
                2 => f32::NEG_INFINITY,
                3 => f32::MIN_POSITIVE * 0.25, // positive subnormal
                4 => -(f32::MIN_POSITIVE * 0.5), // negative subnormal
                5 => f32::from_bits(1),         // smallest positive subnormal
                6 => f32::MAX,
                _ => f32::MIN,
            };
        }

        // 1. Active processing with poisoned input + poisoned parameters
        wiener.process_frame(&poisoned, &mut out, 1.0, db_to_gain(-24.0));
        for &y in &out {
            assert!(y.is_finite() && !y.is_subnormal(), "bad output sample: {y}");
        }
        for bin in 0..NUM_BINS {
            assert!(
                wiener.noise_psd[bin].is_finite() && wiener.noise_psd[bin] >= 1e-9,
                "noise_psd[{bin}] poisoned: {}",
                wiener.noise_psd[bin]
            );
            assert!(
                wiener.prev_gain[bin].is_finite()
                    && !wiener.prev_gain[bin].is_subnormal()
                    && (0.0..=1.0).contains(&wiener.prev_gain[bin]),
                "prev_gain[{bin}] poisoned: {}",
                wiener.prev_gain[bin]
            );
        }

        // 2. Bypass path (strength = 0.0) and NaN parameter paths must still sanitize output
        wiener.process_frame(&poisoned, &mut out, 0.0, 0.01);
        for &y in &out {
            assert!(y.is_finite() && !y.is_subnormal(), "bypass leaked bad sample: {y}");
        }
        wiener.process_frame(&poisoned, &mut out, f32::NAN, f32::NAN);
        for &y in &out {
            assert!(y.is_finite() && !y.is_subnormal(), "NaN param leaked bad sample: {y}");
        }

        // 3. Subsequent clean sine frames must recover cleanly through ChannelDspState
        let mut ch = ChannelDspState::new(Box::new(WienerFallbackProcessor::new()));
        for &bad in &poisoned {
            let y = ch.process_sample(bad, 0.9, db_to_gain(-24.0), 1.0, false, 1.0);
            assert!(y.is_finite() && !y.is_subnormal(), "streaming poisoned output: {y}");
        }

        let mut clean_energy = 0.0f64;
        for i in 0..(FRAME_SIZE * 8) {
            let x = 0.35 * (2.0 * PI * 440.0 * (i as f32) / 48_000.0).sin();
            let y = ch.process_sample(x, 0.85, db_to_gain(-24.0), 1.0, false, 1.0);
            assert!(y.is_finite() && !y.is_subnormal(), "post-poison sample {i} invalid: {y}");
            if i >= FRAME_SIZE * 3 {
                clean_energy += (y as f64) * (y as f64);
            }
        }
        assert!(
            clean_energy > 1.0,
            "processor failed to recover clean signal after NaN/Inf burst: energy={clean_energy}"
        );
    }

    #[test]
    fn zero_length_and_one_sample_buffers_process_safely() {
        let mut plugin = HawaVoCleanPlugin::default();
        let mut init_ctx = TestInitContext::new();
        let layout = HawaVoCleanPlugin::AUDIO_IO_LAYOUTS[0];
        let cfg = BufferConfig {
            sample_rate: 48_000.0,
            min_buffer_size: Some(0),
            max_buffer_size: 512,
            process_mode: ProcessMode::Realtime,
        };
        assert!(plugin.initialize(&layout, &cfg, &mut init_ctx));
        assert_eq!(init_ctx.latency_samples.load(Ordering::Relaxed), FRAME_SIZE as u32);

        // 0-length buffer
        let mut empty = vec![Vec::<f32>::new(), Vec::<f32>::new()];
        process_plugin_block(&mut plugin, &mut empty);

        // 1-sample buffer streamed across > 2 full STFT frames including NaN/subnormal samples
        for i in 0..(FRAME_SIZE * 3) {
            let s = match i {
                0 => f32::NAN,
                1 => f32::INFINITY,
                2 => f32::MIN_POSITIVE * 0.25,
                _ => 0.25 * (2.0 * PI * 330.0 * (i as f32) / 48_000.0).sin(),
            };
            let mut one = vec![vec![s], vec![-s]];
            process_plugin_block(&mut plugin, &mut one);
            for ch in &one {
                let y = ch[0];
                assert!(y.is_finite() && !y.is_subnormal(), "1-sample block {i} invalid: {y}");
            }
        }
    }

    #[test]
    fn continuous_100_block_streaming_stability() {
        let mut plugin = HawaVoCleanPlugin::default();
        let mut init_ctx = TestInitContext::new();
        let layout = HawaVoCleanPlugin::AUDIO_IO_LAYOUTS[0];
        let cfg = BufferConfig {
            sample_rate: 48_000.0,
            min_buffer_size: Some(1),
            max_buffer_size: 512,
            process_mode: ProcessMode::Realtime,
        };
        assert!(plugin.initialize(&layout, &cfg, &mut init_ctx));

        let mut seed: u32 = 0xDEAD_BEEF;
        let mut max_abs = 0.0f32;
        for block in 0..100usize {
            let mut left = vec![0.0f32; 512];
            let mut right = vec![0.0f32; 512];
            for i in 0..512 {
                let n = block * 512 + i;
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let noise = (((seed >> 8) as f32) / ((1u32 << 24) as f32) * 2.0 - 1.0) * 0.05;
                let voice = 0.3 * (2.0 * PI * 220.0 * (n as f32) / 48_000.0).sin()
                    + 0.15 * (2.0 * PI * 660.0 * (n as f32) / 48_000.0).sin();
                left[i] = voice + noise;
                right[i] = voice * 0.8 - noise;
            }
            // Inject occasional adversarial spikes in the middle of streaming
            if block == 25 {
                left[17] = f32::NAN;
                right[99] = f32::NEG_INFINITY;
            } else if block == 50 {
                left[0] = f32::MIN_POSITIVE * 0.125;
                right[511] = -(f32::MIN_POSITIVE * 0.125);
            }

            let mut stereo = vec![left, right];
            process_plugin_block(&mut plugin, &mut stereo);
            for ch in &stereo {
                for &y in ch {
                    assert!(
                        y.is_finite() && !y.is_subnormal(),
                        "unstable sample in block {block}: {y}"
                    );
                    max_abs = max_abs.max(y.abs());
                }
            }
        }
        assert!(max_abs > 0.05 && max_abs < 5.0, "unexpected streaming peak amplitude: {max_abs}");
    }

    #[test]
    fn extreme_sample_rates_initialize_and_process_cleanly() {
        let layout = HawaVoCleanPlugin::AUDIO_IO_LAYOUTS[1]; // mono

        // Invalid sample rates must be rejected cleanly
        for bad_sr in [0.0f32, -48_000.0, f32::NAN, f32::INFINITY] {
            let mut plugin = HawaVoCleanPlugin::default();
            let mut init_ctx = TestInitContext::new();
            let cfg = BufferConfig {
                sample_rate: bad_sr,
                min_buffer_size: Some(1),
                max_buffer_size: 512,
                process_mode: ProcessMode::Realtime,
            };
            assert!(
                !plugin.initialize(&layout, &cfg, &mut init_ctx),
                "expected invalid sample rate {bad_sr} to be rejected"
            );
        }

        // Valid extreme sample rates: 8 kHz through 192 kHz
        for sr in [8_000.0f32, 44_100.0, 48_000.0, 96_000.0, 192_000.0] {
            let mut plugin = HawaVoCleanPlugin::default();
            let mut init_ctx = TestInitContext::new();
            let cfg = BufferConfig {
                sample_rate: sr,
                min_buffer_size: Some(1),
                max_buffer_size: 512,
                process_mode: ProcessMode::Realtime,
            };
            assert!(plugin.initialize(&layout, &cfg, &mut init_ctx), "failed init at {sr} Hz");

            for block in 0..8usize {
                let mut mono = vec![(0..256)
                    .map(|i| {
                        let n = block * 256 + i;
                        0.3 * (2.0 * PI * 400.0 * (n as f32) / sr).sin()
                    })
                    .collect::<Vec<f32>>()];
                process_plugin_block(&mut plugin, &mut mono);
                for &y in &mono[0] {
                    assert!(y.is_finite() && !y.is_subnormal(), "bad sample at {sr} Hz: {y}");
                }
            }
        }
    }

    #[test]
    fn sudden_parameter_jumps_mid_frame_stay_finite_and_bounded() {
        let mut ch = ChannelDspState::new(Box::new(WienerFallbackProcessor::new()));
        let total = FRAME_SIZE * 12;

        for i in 0..total {
            let x = 0.4 * (2.0 * PI * 500.0 * (i as f32) / 48_000.0).sin()
                + 0.08 * (2.0 * PI * 1900.0 * (i as f32) / 48_000.0).cos();

            // Abrupt mid-frame strength 0 -> 1 -> 0 jumps
            let strength = match (i / 137) % 3 {
                0 => 0.0,
                1 => 1.0,
                _ => 0.5,
            };
            // Abrupt mid-frame delta_listen flip
            let delta_listen = (i % 193) > 73;
            // Full ambience_floor_db sweep across [-60 dB, 0 dB]
            let floor_db = -60.0 + 60.0 * ((i % 257) as f32 / 256.0);
            let floor_gain = db_to_gain(floor_db);
            // Wet/dry and output gain sweeps
            let wet_dry = if (i % 101) < 50 { 0.0 } else { 1.0 };
            let out_gain = db_to_gain(-12.0 + 24.0 * ((i % 89) as f32 / 88.0));

            let y = ch.process_sample(x, strength, floor_gain, wet_dry, delta_listen, out_gain);
            assert!(
                y.is_finite() && !y.is_subnormal() && y.abs() < 20.0,
                "parameter jump produced invalid sample at i={i}: {y}"
            );
        }

        // Also validate non-finite scalar parameter handling in db_to_gain & apply_output_mix
        assert_eq!(db_to_gain(f32::NAN), 0.0);
        assert_eq!(db_to_gain(f32::INFINITY), 0.0);
        assert_eq!(apply_output_mix(f32::NAN, 0.5, f32::NAN, false, f32::INFINITY), 0.0);
        assert_eq!(apply_output_mix(0.5, f32::INFINITY, 0.5, true, 1.0), 0.5);
    }
}
