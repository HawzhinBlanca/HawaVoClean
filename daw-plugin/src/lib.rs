//! HawaVoClean Native DAW Plugin (VST3 + CLAP)
//!
//! Real-time, zero-audio-thread-allocation speech restoration plugin built on `nih-plug`.
//!
//! Architecture:
//! - Thread-safe, smoothed DAW parameters (`strength`, `ambience_floor_db`, `delta_listen`,
//!   `wet_dry`, `output_gain_db`).
//! - Lock-free fixed-capacity SPSC ring buffer (`FixedRingBuffer<N>`) for audio-thread-safe
//!   sample buffering with zero heap allocations.
//! - Causal 960-sample / 480-hop (20 ms frame / 10 ms hop @ 48 kHz, 481 FFT bins) 50%-overlap
//!   `sqrt(Hann)` analysis + `sqrt(Hann)` synthesis COLA frame engine (`w_a[n] * w_s[n] = Hann[n]`)
//!   with an exact delay-matched dry line so wet/dry blending and phase-inverted Delta audition
//!   (`dry_aligned - wet`) are sample-accurate, leakage-suppressed, and comb-filter-free.
//! - Zero-allocation mixed-radix-2/3/5 Stockham FFT (`Fft960`, `960 = 2^6 * 3 * 5`) with
//!   precomputed twiddles and zero trigonometric calls on the audio thread.
//! - Pluggable `FrameProcessor` trait with:
//!   1. `WienerFallbackProcessor`: self-contained, zero-allocation frequency-domain Wiener
//!      denoiser over the `sqrt(Hann)`-windowed 960-point STFT + asymmetric noise-floor PSD tracker.
//!   2. `OnnxFrameProcessor` (behind `--features onnx`): real ONNX Runtime streaming mask
//!      inference (`[1, 1, 481]` magnitude -> `[1, 1, 481]` gain mask) with CoreML EP (macOS)
//!      and safe CPU / Wiener fallback.

use nih_plug::prelude::*;
use std::f32::consts::PI;
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Causal streaming STFT frame length in samples (960 @ 48 kHz = 20.0 ms, DeepFilterNet3 geometry).
pub const FRAME_SIZE: usize = 960;
/// 50% hop size in samples (480 @ 48 kHz = 10.0 ms hop).
pub const FRAME_HOP: usize = FRAME_SIZE / 2;
/// Number of non-redundant real-FFT bins (`FRAME_SIZE / 2 + 1 = 481`).
pub const NUM_BINS: usize = FRAME_SIZE / 2 + 1;
/// Power-of-two capacity for lock-free ring buffers (>= 4x FRAME_SIZE).
pub const RING_CAPACITY: usize = 4096;

/// Mixed-radix stage decomposition of `FRAME_SIZE = 960 = 2^6 * 3 * 5` (8 stages, even count).
const FFT_RADICES: [usize; 8] = [2, 2, 2, 2, 2, 2, 3, 5];
/// Total precomputed stage twiddle count: `1 + 2 + 4 + 8 + 16 + 32 + 2*64 + 4*192 = 959`.
const FFT_TWIDDLE_LEN: usize = 959;

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

/// Compute the periodic `sqrt(Hann)` analysis/synthesis window of length `FRAME_SIZE`.
///
/// Satisfies the exact 50%-overlap COLA identity:
/// `w_a[n] * w_s[n] + w_a[n + FRAME_HOP] * w_s[n + FRAME_HOP] == Hann[n] + Hann[n + FRAME_HOP] == 1.0`.
pub fn make_sqrt_hann_window() -> [f32; FRAME_SIZE] {
    let mut win = [0.0f32; FRAME_SIZE];
    for (n, w) in win.iter_mut().enumerate() {
        let hann = 0.5 * (1.0 - (2.0 * PI * (n as f32) / (FRAME_SIZE as f32)).cos());
        *w = hann.max(0.0).sqrt();
    }
    win
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
        debug_assert!(
            N >= 2 && N.is_power_of_two(),
            "Ring capacity must be power of two"
        );
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
// Zero-allocation mixed-radix Stockham FFT for FRAME_SIZE = 960 (2^6 * 3 * 5)
// ---------------------------------------------------------------------------

/// Complex scalar for pre-allocated mixed-radix FFT.
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

    #[inline]
    fn mul_twiddle(self, tw: Complex32, inverse: bool) -> Self {
        let wi = if inverse { -tw.im } else { tw.im };
        Self {
            re: self.re * tw.re - self.im * wi,
            im: self.re * wi + self.im * tw.re,
        }
    }
}

/// Exact, zero-allocation 960-point mixed-radix Stockham FFT (`960 = 2^6 * 3 * 5`).
///
/// Precomputes all stage twiddle factors at construction time so forward and inverse
/// transforms perform zero heap allocations and zero trigonometric calls on the audio thread.
struct Fft960 {
    twiddles: [Complex32; FFT_TWIDDLE_LEN],
    scratch: [Complex32; FRAME_SIZE],
}

impl Fft960 {
    fn new() -> Self {
        let mut twiddles = [Complex32::default(); FFT_TWIDDLE_LEN];
        let mut offset = 0usize;
        let mut m = 1usize;
        for &r in &FFT_RADICES {
            let rm = (r * m) as f32;
            for p in 1..r {
                for b in 0..m {
                    let angle = -2.0 * PI * ((p * b) as f32) / rm;
                    twiddles[offset] = Complex32::new(angle.cos(), angle.sin());
                    offset += 1;
                }
            }
            m *= r;
        }
        debug_assert_eq!(offset, FFT_TWIDDLE_LEN);
        debug_assert_eq!(m, FRAME_SIZE);

        Self {
            twiddles,
            scratch: [Complex32::default(); FRAME_SIZE],
        }
    }

    fn reset(&mut self) {
        self.scratch.fill(Complex32::default());
    }

    fn transform(&mut self, buf: &mut [Complex32; FRAME_SIZE], inverse: bool) {
        // Radix-3 constants: cos(2*pi/3) = -0.5, sin(2*pi/3) = sqrt(3)/2
        const C3: f32 = -0.5;
        const S3_MAG: f32 = 0.866_025_4;
        // Radix-5 constants: cos/sin(2*pi/5) and cos/sin(4*pi/5)
        const C5_1: f32 = 0.309_017;
        const S5_1_MAG: f32 = 0.951_056_54;
        const C5_2: f32 = -0.809_017;
        const S5_2_MAG: f32 = 0.587_785_24;

        let dir = if inverse { 1.0f32 } else { -1.0f32 };
        let s3 = dir * S3_MAG;
        let s5_1 = dir * S5_1_MAG;
        let s5_2 = dir * S5_2_MAG;

        let mut m = 1usize;
        let mut tw_offset = 0usize;
        let mut src_is_buf = true;

        for &r in &FFT_RADICES {
            let rm = r * m;
            let s = FRAME_SIZE / rm;
            let (src, dst): (&[Complex32; FRAME_SIZE], &mut [Complex32; FRAME_SIZE]) = if src_is_buf
            {
                (&*buf, &mut self.scratch)
            } else {
                (&self.scratch, &mut *buf)
            };

            match r {
                2 => {
                    for u in 0..s {
                        let out_base = u * rm;
                        let in0_base = u * m;
                        let in1_base = (s + u) * m;
                        for b in 0..m {
                            let y0 = src[in0_base + b];
                            let y1 = src[in1_base + b].mul_twiddle(self.twiddles[tw_offset + b], inverse);
                            dst[out_base + b] = Complex32::new(y0.re + y1.re, y0.im + y1.im);
                            dst[out_base + m + b] = Complex32::new(y0.re - y1.re, y0.im - y1.im);
                        }
                    }
                    tw_offset += m;
                }
                3 => {
                    let tw1 = tw_offset;
                    let tw2 = tw_offset + m;
                    for u in 0..s {
                        let out_base = u * rm;
                        let in0_base = u * m;
                        let in1_base = (s + u) * m;
                        let in2_base = (2 * s + u) * m;
                        for b in 0..m {
                            let y0 = src[in0_base + b];
                            let y1 = src[in1_base + b].mul_twiddle(self.twiddles[tw1 + b], inverse);
                            let y2 = src[in2_base + b].mul_twiddle(self.twiddles[tw2 + b], inverse);

                            let t1_re = y1.re + y2.re;
                            let t1_im = y1.im + y2.im;
                            let t2_re = y1.re - y2.re;
                            let t2_im = y1.im - y2.im;

                            let re_mid = y0.re + C3 * t1_re;
                            let im_mid = y0.im + C3 * t1_im;
                            let dre = -s3 * t2_im;
                            let dim = s3 * t2_re;

                            dst[out_base + b] = Complex32::new(y0.re + t1_re, y0.im + t1_im);
                            dst[out_base + m + b] = Complex32::new(re_mid + dre, im_mid + dim);
                            dst[out_base + 2 * m + b] = Complex32::new(re_mid - dre, im_mid - dim);
                        }
                    }
                    tw_offset += 2 * m;
                }
                5 => {
                    let tw1 = tw_offset;
                    let tw2 = tw_offset + m;
                    let tw3 = tw_offset + 2 * m;
                    let tw4 = tw_offset + 3 * m;
                    for u in 0..s {
                        let out_base = u * rm;
                        let in0_base = u * m;
                        let in1_base = (s + u) * m;
                        let in2_base = (2 * s + u) * m;
                        let in3_base = (3 * s + u) * m;
                        let in4_base = (4 * s + u) * m;
                        for b in 0..m {
                            let y0 = src[in0_base + b];
                            let y1 = src[in1_base + b].mul_twiddle(self.twiddles[tw1 + b], inverse);
                            let y2 = src[in2_base + b].mul_twiddle(self.twiddles[tw2 + b], inverse);
                            let y3 = src[in3_base + b].mul_twiddle(self.twiddles[tw3 + b], inverse);
                            let y4 = src[in4_base + b].mul_twiddle(self.twiddles[tw4 + b], inverse);

                            let t1_re = y1.re + y4.re;
                            let t1_im = y1.im + y4.im;
                            let t2_re = y2.re + y3.re;
                            let t2_im = y2.im + y3.im;
                            let t3_re = y1.re - y4.re;
                            let t3_im = y1.im - y4.im;
                            let t4_re = y2.re - y3.re;
                            let t4_im = y2.im - y3.im;

                            let r1 = y0.re + C5_1 * t1_re + C5_2 * t2_re;
                            let i1 = y0.im + C5_1 * t1_im + C5_2 * t2_im;
                            let r2 = y0.re + C5_2 * t1_re + C5_1 * t2_re;
                            let i2 = y0.im + C5_2 * t1_im + C5_1 * t2_im;

                            let dr1 = -(s5_1 * t3_im + s5_2 * t4_im);
                            let di1 = s5_1 * t3_re + s5_2 * t4_re;
                            let dr2 = -(s5_2 * t3_im - s5_1 * t4_im);
                            let di2 = s5_2 * t3_re - s5_1 * t4_re;

                            dst[out_base + b] =
                                Complex32::new(y0.re + t1_re + t2_re, y0.im + t1_im + t2_im);
                            dst[out_base + m + b] = Complex32::new(r1 + dr1, i1 + di1);
                            dst[out_base + 2 * m + b] = Complex32::new(r2 + dr2, i2 + di2);
                            dst[out_base + 3 * m + b] = Complex32::new(r2 - dr2, i2 - di2);
                            dst[out_base + 4 * m + b] = Complex32::new(r1 - dr1, i1 - di1);
                        }
                    }
                    tw_offset += 4 * m;
                }
                _ => unreachable!("FFT_RADICES contains only 2, 3, and 5"),
            }

            m = rm;
            src_is_buf = !src_is_buf;
        }

        // 8 stages (even) guarantees the final output always lands back in `buf`.
        debug_assert!(src_is_buf);

        if inverse {
            let inv_n = 1.0 / (FRAME_SIZE as f32);
            for c in buf.iter_mut() {
                c.re *= inv_n;
                c.im *= inv_n;
            }
        }
    }

    #[inline]
    fn forward(&mut self, buf: &mut [Complex32; FRAME_SIZE]) {
        self.transform(buf, false);
    }

    #[inline]
    fn inverse(&mut self, buf: &mut [Complex32; FRAME_SIZE]) {
        self.transform(buf, true);
    }
}

// ---------------------------------------------------------------------------
// FrameProcessor trait + Zero-allocation Wiener FFT fallback + ONNX Runtime hook
// ---------------------------------------------------------------------------

/// Processing contract for real-time audio frames (`FRAME_SIZE = 960` samples).
///
/// Frames passed into `process_frame` are already `sqrt(Hann)`-windowed by `ChannelDspState`
/// prior to forward FFT, and the output frame is multiplied by the matching `sqrt(Hann)`
/// synthesis window during 50% overlap-add (`w_a[n] * w_s[n] = Hann[n]`).
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

/// Zero-allocation frequency-domain Wiener fallback denoiser.
///
/// Performs an exact 960-point mixed-radix FFT over the `sqrt(Hann)`-windowed frame,
/// updates a per-bin asymmetric noise PSD floor across all `481` bins, computes an
/// Ephraim-Malah-style Wiener gain constrained by `ambience_floor_gain` and scaled by
/// `strength`, and reconstructs the time-domain frame via inverse 960-point FFT.
pub struct WienerFallbackProcessor {
    fft: Fft960,
    fft_buf: [Complex32; FRAME_SIZE],
    noise_psd: [f32; NUM_BINS],
    prev_gain: [f32; NUM_BINS],
    initialized_noise: bool,
}

impl WienerFallbackProcessor {
    pub fn new() -> Self {
        Self {
            fft: Fft960::new(),
            fft_buf: [Complex32::default(); FRAME_SIZE],
            noise_psd: [1e-7; NUM_BINS],
            prev_gain: [1.0; NUM_BINS],
            initialized_noise: false,
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
        self.fft.reset();
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
        self.fft.forward(&mut self.fft_buf);

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
                let alpha = if psd < self.noise_psd[bin] {
                    0.80
                } else {
                    0.992
                };
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

        self.fft.inverse(&mut self.fft_buf);
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

/// Locate an exported streaming mask ONNX model from `HAWAVOCLEAN_DAW_ONNX_MODEL` or
/// standard macOS `.vst3`/`.clap` bundle `Contents/Resources/model.onnx` locations.
pub fn discover_onnx_model_path() -> Option<String> {
    if let Ok(env_path) = std::env::var("HAWAVOCLEAN_DAW_ONNX_MODEL") {
        let trimmed = env_path.trim();
        if !trimmed.is_empty() && Path::new(trimmed).is_file() {
            return Some(trimmed.to_string());
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            let bundled = parent.join("../Resources/model.onnx");
            if bundled.is_file() {
                return Some(bundled.to_string_lossy().into_owned());
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        let mut candidates = Vec::new();
        if let Ok(home) = std::env::var("HOME") {
            for (kind, ext) in [("VST3", "vst3"), ("CLAP", "clap")] {
                candidates.push(format!(
                    "{home}/Library/Audio/Plug-Ins/{kind}/HawaVoClean.{ext}/Contents/Resources/model.onnx"
                ));
            }
        }
        for (kind, ext) in [("VST3", "vst3"), ("CLAP", "clap")] {
            candidates.push(format!(
                "/Library/Audio/Plug-Ins/{kind}/HawaVoClean.{ext}/Contents/Resources/model.onnx"
            ));
        }
        for cand in candidates {
            if Path::new(&cand).is_file() {
                return Some(cand);
            }
        }
    }

    None
}

#[cfg(feature = "onnx")]
fn discover_ort_dylib_path(model_path: &str) -> Option<std::path::PathBuf> {
    if let Ok(env_dylib) = std::env::var("ORT_DYLIB_PATH") {
        let trimmed = env_dylib.trim();
        if !trimmed.is_empty() && Path::new(trimmed).is_file() {
            return Some(std::path::PathBuf::from(trimmed));
        }
    }

    if let Some(model_dir) = Path::new(model_path).parent() {
        let sibling = model_dir.join("libonnxruntime.dylib");
        if sibling.is_file() {
            return Some(sibling);
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            let bundled = parent.join("../Resources/libonnxruntime.dylib");
            if bundled.is_file() {
                return Some(bundled);
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        for path_str in [
            "/opt/homebrew/lib/libonnxruntime.dylib",
            "/usr/local/lib/libonnxruntime.dylib",
        ] {
            if Path::new(path_str).is_file() {
                return Some(std::path::PathBuf::from(path_str));
            }
        }
    }

    None
}

#[cfg(feature = "onnx")]
pub struct OnnxFrameProcessor {
    _env: ort::environment::Environment,
    session: ort::session::Session,
    fft: Fft960,
    fallback: WienerFallbackProcessor,
    fft_buf: [Complex32; FRAME_SIZE],
    mag_input: [f32; NUM_BINS],
    prev_gain: [f32; NUM_BINS],
    using_coreml: bool,
}

#[cfg(feature = "onnx")]
impl OnnxFrameProcessor {
    /// Load an exported streaming mask ONNX model (`[1, 1, 481]` -> `[1, 1, 481]`)
    /// during plugin `initialize()` (never on the real-time audio thread).
    ///
    /// Configures CoreML EP when `config.prefer_coreml` is true with safe CPU EP fallback.
    /// Returns `Err(String)` without invoking ORT FFI when the model or runtime dylib is absent.
    pub fn load(config: &OnnxProcessorConfig) -> Result<Self, String> {
        if config.model_path.trim().is_empty() || !Path::new(&config.model_path).is_file() {
            return Err(format!("ONNX model file not found: {}", config.model_path));
        }

        let dylib_path = discover_ort_dylib_path(&config.model_path).ok_or_else(|| {
            "libonnxruntime dynamic library not found (set ORT_DYLIB_PATH or bundle Contents/Resources/libonnxruntime.dylib)".to_string()
        })?;

        let env = ort::init_from(&dylib_path)
            .map_err(|e| format!("Failed to load ONNX Runtime dylib: {e}"))?
            .with_name("hawavoclean-daw")
            .build()
            .map_err(|e| format!("Failed to build ONNX Runtime environment: {e}"))?;

        let mut builder = ort::session::Session::builder(&env)
            .map_err(|e| format!("Failed to create ONNX session builder: {e}"))?
            .with_intra_threads(config.intra_op_threads.max(1))
            .map_err(|e| format!("Failed to configure ONNX intra-op threads: {e}"))?;

        let mut using_coreml = false;
        if config.prefer_coreml {
            if let Ok(b) = builder.with_execution_providers([
                ort::ep::CoreML::default().build(),
                ort::ep::CPU::default().build(),
            ]) {
                builder = b;
                using_coreml = true;
            } else {
                // Rebuild clean CPU-only session builder if CoreML registration errored.
                builder = ort::session::Session::builder(&env)
                    .map_err(|e| format!("Failed to create fallback ONNX session builder: {e}"))?
                    .with_intra_threads(config.intra_op_threads.max(1))
                    .map_err(|e| format!("Failed to configure fallback ONNX threads: {e}"))?
                    .with_execution_providers([ort::ep::CPU::default().build()])
                    .map_err(|e| format!("Failed to register CPU EP: {e}"))?;
            }
        } else {
            builder = builder
                .with_execution_providers([ort::ep::CPU::default().build()])
                .map_err(|e| format!("Failed to register CPU EP: {e}"))?;
        }

        let session = builder
            .commit_from_file(&config.model_path)
            .map_err(|e| format!("Failed to load ONNX session from {}: {e}", config.model_path))?;
        Ok(Self {
            _env: env,
            session,
            fft: Fft960::new(),
            fallback: WienerFallbackProcessor::new(),
            fft_buf: [Complex32::default(); FRAME_SIZE],
            mag_input: [0.0; NUM_BINS],
            prev_gain: [1.0; NUM_BINS],
            using_coreml,
        })
    }

    pub fn is_coreml_configured(&self) -> bool {
        self.using_coreml
    }

    fn run_onnx_gains(&mut self, s: f32, floor_gain: f32) -> bool {
        let Ok(input_tensor) =
            ort::value::TensorRef::from_array_view(([1usize, 1, NUM_BINS], &self.mag_input[..]))
        else {
            return false;
        };

        let Ok(outputs) = self.session.run(ort::inputs![input_tensor]) else {
            return false;
        };
        if outputs.len() == 0 {
            return false;
        }
        let Ok((_shape, mask_slice)) = outputs[0].try_extract_tensor::<f32>() else {
            return false;
        };
        if mask_slice.len() != NUM_BINS {
            return false;
        }

        for (bin, &raw_mask) in mask_slice.iter().enumerate() {
            let mask = if raw_mask.is_finite() {
                raw_mask.clamp(0.0, 1.0)
            } else {
                1.0
            };
            let target_gain = ((1.0 - s) + s * mask.max(floor_gain)).clamp(floor_gain, 1.0);
            let prev = if self.prev_gain[bin].is_finite() {
                self.prev_gain[bin]
            } else {
                1.0
            };
            let smooth_gain = (0.5 * prev + 0.5 * target_gain).clamp(floor_gain, 1.0);
            self.prev_gain[bin] = smooth_gain;

            self.fft_buf[bin].re *= smooth_gain;
            self.fft_buf[bin].im *= smooth_gain;
            if bin > 0 && bin < FRAME_SIZE / 2 {
                let mirror = FRAME_SIZE - bin;
                self.fft_buf[mirror].re *= smooth_gain;
                self.fft_buf[mirror].im *= smooth_gain;
            }
        }
        true
    }
}

#[cfg(feature = "onnx")]
impl FrameProcessor for OnnxFrameProcessor {
    fn name(&self) -> &'static str {
        if self.using_coreml {
            "onnx-runtime-coreml"
        } else {
            "onnx-runtime-cpu"
        }
    }

    fn reset(&mut self) {
        self.fft.reset();
        self.fallback.reset();
        self.fft_buf.fill(Complex32::default());
        self.mag_input.fill(0.0);
        self.prev_gain.fill(1.0);
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

        let floor_gain = if ambience_floor_gain.is_finite() {
            ambience_floor_gain.clamp(MIN_GAIN_EPS, 1.0)
        } else {
            MIN_GAIN_EPS
        };

        for (slot, &x) in self.fft_buf.iter_mut().zip(input.iter()) {
            *slot = Complex32::new(sanitize_sample(x), 0.0);
        }
        self.fft.forward(&mut self.fft_buf);

        for (bin, mag_slot) in self.mag_input.iter_mut().enumerate() {
            let sq = self.fft_buf[bin].norm_sq();
            *mag_slot = if sq.is_finite() && sq >= 0.0 {
                flush_denormal(sq.sqrt())
            } else {
                0.0
            };
        }

        if !self.run_onnx_gains(s, floor_gain) {
            // Safe fallback on runtime error or shape mismatch without ever panicking.
            self.fallback
                .process_frame(input, output, strength, ambience_floor_gain);
            return;
        }

        self.fft.inverse(&mut self.fft_buf);
        for (dst, src) in output.iter_mut().zip(self.fft_buf.iter()) {
            *dst = flush_denormal(src.re);
        }
    }
}

// ---------------------------------------------------------------------------
// Per-channel overlap-add accumulator with sqrt(Hann) COLA & delay-matched dry
// ---------------------------------------------------------------------------

/// Per-channel streaming processor combining a lock-free ring buffer, 50%-overlap
/// `sqrt(Hann)` analysis + `sqrt(Hann)` synthesis COLA framing (`960/480`),
/// and an exact latency-matched dry delay line.
pub struct ChannelDspState {
    input_ring: FixedRingBuffer<RING_CAPACITY>,
    output_ring: FixedRingBuffer<RING_CAPACITY>,
    dry_delay: [f32; FRAME_SIZE],
    dry_delay_pos: usize,
    analysis_frame: [f32; FRAME_SIZE],
    windowed_frame: [f32; FRAME_SIZE],
    processed_frame: [f32; FRAME_SIZE],
    ola_tail: [f32; FRAME_HOP],
    sqrt_hann_window: [f32; FRAME_SIZE],
    hop_counter: usize,
    processor: Box<dyn FrameProcessor>,
}

impl ChannelDspState {
    /// Exact sample latency introduced by the 50%-overlap frame accumulator.
    pub const LATENCY_SAMPLES: usize = FRAME_HOP;

    pub fn new(processor: Box<dyn FrameProcessor>) -> Self {
        let sqrt_hann_window = make_sqrt_hann_window();

        let mut state = Self {
            input_ring: FixedRingBuffer::new(),
            output_ring: FixedRingBuffer::new(),
            dry_delay: [0.0; FRAME_SIZE],
            dry_delay_pos: 0,
            analysis_frame: [0.0; FRAME_SIZE],
            windowed_frame: [0.0; FRAME_SIZE],
            processed_frame: [0.0; FRAME_SIZE],
            ola_tail: [0.0; FRAME_HOP],
            sqrt_hann_window,
            hop_counter: 0,
            processor,
        };
        state.prime_output_latency();
        state
    }

    pub fn processor_name(&self) -> &'static str {
        self.processor.name()
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
        self.windowed_frame.fill(0.0);
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

        // 1. Exact delay line matching end-to-end OLA latency (FRAME_SIZE = 960 samples:
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
                // Apply sqrt(Hann) analysis window BEFORE forward FFT to prevent spectral leakage.
                for i in 0..FRAME_SIZE {
                    self.windowed_frame[i] =
                        flush_denormal(self.analysis_frame[i] * self.sqrt_hann_window[i]);
                }

                self.processor.process_frame(
                    &self.windowed_frame,
                    &mut self.processed_frame,
                    strength,
                    ambience_floor_gain,
                );

                // Overlap-add synthesis with matching sqrt(Hann) window:
                // w_a[n] * w_s[n] = Hann[n], satisfying exact 50% COLA reconstruction.
                for i in 0..FRAME_HOP {
                    let y = flush_denormal(
                        self.ola_tail[i]
                            + sanitize_sample(self.processed_frame[i]) * self.sqrt_hann_window[i],
                    );
                    let _ = self.output_ring.push(y);
                }
                for i in 0..FRAME_HOP {
                    self.ola_tail[i] = flush_denormal(
                        sanitize_sample(self.processed_frame[FRAME_HOP + i])
                            * self.sqrt_hann_window[FRAME_HOP + i],
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

/// Build the best available `FrameProcessor` for one channel during `Plugin::initialize()`.
fn create_channel_processor() -> Box<dyn FrameProcessor> {
    #[cfg(feature = "onnx")]
    {
        if let Some(model_path) = discover_onnx_model_path() {
            let cfg = OnnxProcessorConfig {
                model_path,
                prefer_coreml: cfg!(target_os = "macos"),
                intra_op_threads: 1,
            };
            if let Ok(proc) = OnnxFrameProcessor::load(&cfg) {
                return Box::new(proc);
            }
        }
    }
    Box::new(WienerFallbackProcessor::new())
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

        // Pre-allocate all channel states, FFT scratch buffers, and optional ONNX sessions here
        // (never on the real-time audio thread).
        self.channels.clear();
        self.channels.reserve_exact(num_channels);
        for _ in 0..num_channels {
            self.channels
                .push(ChannelDspState::new(create_channel_processor()));
        }

        // Report exact algorithmic lookahead latency (960 samples = 20 ms @ 48 kHz) to host DAW for PDC.
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

#[cfg(feature = "onnx")]
const CLAP_DESC: &str = "Real-time dialogue restoration with causal 960/480 STFT, ONNX Runtime (CoreML/CPU), Wiener fallback & Delta audition";
#[cfg(not(feature = "onnx"))]
const CLAP_DESC: &str = "Zero-allocation real-time dialogue restoration with causal 960/480 sqrt(Hann) Wiener DSP & Delta audition";

impl ClapPlugin for HawaVoCleanPlugin {
    const CLAP_ID: &'static str = "com.hawavoclean.daw.cleaner";
    const CLAP_DESCRIPTION: Option<&'static str> = Some(CLAP_DESC);
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
// Unit tests for DSP invariants (960/480 FFT, sqrt(Hann) COLA, Delta null, Wiener, RingBuffer)
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
    fn stft_geometry_and_package_version_match_ship_contract() {
        assert_eq!(FRAME_SIZE, 960);
        assert_eq!(FRAME_HOP, 480);
        assert_eq!(NUM_BINS, 481);
        assert_eq!(HawaVoCleanPlugin::VERSION, "3.3.0");
    }

    #[test]
    fn sqrt_hann_window_satisfies_exact_cola_identity() {
        let w = make_sqrt_hann_window();
        let mut max_err = 0.0f32;
        for n in 0..FRAME_HOP {
            let sum = w[n] * w[n] + w[n + FRAME_HOP] * w[n + FRAME_HOP];
            max_err = max_err.max((sum - 1.0).abs());
        }
        assert!(
            max_err < 1e-6,
            "sqrt(Hann) analysis*synthesis COLA sum deviated from 1.0 by {max_err}"
        );
    }

    #[test]
    fn fft960_matches_naive_dft_and_roundtrips_exactly() {
        let mut fft = Fft960::new();
        let mut signal = [Complex32::default(); FRAME_SIZE];
        let mut seed: u32 = 0x5A5A_1234;
        for slot in &mut signal {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let x = (((seed >> 8) as f32) / ((1u32 << 24) as f32)) * 2.0 - 1.0;
            *slot = Complex32::new(x, 0.0);
        }

        let mut buf = signal;
        fft.forward(&mut buf);

        // Spot-check selected frequency bins against exact double-precision naive DFT
        for &k in &[0usize, 1, 7, 64, 192, 480, 959] {
            let mut ref_re = 0.0f64;
            let mut ref_im = 0.0f64;
            for (n, s) in signal.iter().enumerate() {
                let angle = -2.0 * std::f64::consts::PI * ((k * n) as f64) / (FRAME_SIZE as f64);
                ref_re += (s.re as f64) * angle.cos();
                ref_im += (s.re as f64) * angle.sin();
            }
            let err_re = (buf[k].re as f64 - ref_re).abs();
            let err_im = (buf[k].im as f64 - ref_im).abs();
            assert!(
                err_re < 1e-3 && err_im < 1e-3,
                "Fft960 bin {k} mismatch: got ({}, {}), ref ({ref_re}, {ref_im})",
                buf[k].re,
                buf[k].im
            );
        }

        // Exact inverse roundtrip back to original time-domain samples
        fft.inverse(&mut buf);
        let mut max_rt_err = 0.0f32;
        for (got, orig) in buf.iter().zip(signal.iter()) {
            max_rt_err = max_rt_err.max((got.re - orig.re).abs());
            max_rt_err = max_rt_err.max(got.im.abs());
        }
        assert!(
            max_rt_err < 1e-5,
            "Fft960 forward->inverse roundtrip error too high: {max_rt_err}"
        );
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
    fn ola_sqrt_hann_reconstructs_exact_delayed_signal_and_delta_nulls_at_960_480() {
        let mut ch_wet = ChannelDspState::new(Box::new(IdentityProcessor));
        let mut ch_delta = ChannelDspState::new(Box::new(IdentityProcessor));
        let mut ch_wiener_bypass = ChannelDspState::new(Box::new(WienerFallbackProcessor::new()));
        let n = FRAME_SIZE * 8;
        let mut max_recon_err = 0.0f32;
        let mut max_delta_err = 0.0f32;
        let mut max_bypass_err = 0.0f32;

        // Multi-tone + sharp transient impulse to verify zero COLA ringing or leakage distortion
        let input: Vec<f32> = (0..n)
            .map(|i| {
                let tone = 0.35 * (2.0 * PI * 440.0 * (i as f32) / 48_000.0).sin()
                    + 0.15 * (2.0 * PI * 1320.0 * (i as f32) / 48_000.0).cos();
                let impulse = if i == FRAME_SIZE * 3 + 123 { 0.45 } else { 0.0 };
                tone + impulse
            })
            .collect();

        for i in 0..n {
            let y = ch_wet.process_sample(input[i], 1.0, 0.01, 1.0, false, 1.0);
            let d = ch_delta.process_sample(input[i], 1.0, 0.01, 1.0, true, 1.0);
            let b = ch_wiener_bypass.process_sample(input[i], 0.0, 0.01, 1.0, false, 1.0);
            if i >= FRAME_SIZE * 2 {
                let expected = input[i - FRAME_SIZE];
                max_recon_err = max_recon_err.max((y - expected).abs());
                max_delta_err = max_delta_err.max(d.abs());
                max_bypass_err = max_bypass_err.max((b - expected).abs());
            }
        }

        assert!(
            max_recon_err < 1e-5,
            "sqrt(Hann) 960/480 COLA identity reconstruction error too high: {max_recon_err}"
        );
        assert!(
            max_delta_err < 1e-5,
            "Phase-inverted Delta audition failed to null on 960/480 identity: {max_delta_err}"
        );
        assert!(
            max_bypass_err < 1e-5,
            "Wiener bypass (strength=0) failed exact 960/480 COLA reconstruction: {max_bypass_err}"
        );
    }

    #[test]
    fn wiener_fallback_attenuates_stationary_noise_and_delta_captures_removed_energy() {
        let mut ch = ChannelDspState::new(Box::new(WienerFallbackProcessor::new()));
        let mut ch_delta = ChannelDspState::new(Box::new(WienerFallbackProcessor::new()));

        // Deterministic stationary broadband noise via LCG (14400 samples = 300 ms @ 48 kHz)
        let mut seed: u32 = 0x1234_5678;
        let mut noise = Vec::with_capacity(14_400);
        for _ in 0..14_400 {
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
            if i >= 7200 {
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
                3 => f32::MIN_POSITIVE * 0.25,
                4 => -(f32::MIN_POSITIVE * 0.5),
                5 => f32::from_bits(1),
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
            assert!(
                y.is_finite() && !y.is_subnormal(),
                "bypass leaked bad sample: {y}"
            );
        }
        wiener.process_frame(&poisoned, &mut out, f32::NAN, f32::NAN);
        for &y in &out {
            assert!(
                y.is_finite() && !y.is_subnormal(),
                "NaN param leaked bad sample: {y}"
            );
        }

        // 3. Subsequent clean sine frames must recover cleanly through ChannelDspState
        let mut ch = ChannelDspState::new(Box::new(WienerFallbackProcessor::new()));
        for &bad in &poisoned {
            let y = ch.process_sample(bad, 0.9, db_to_gain(-24.0), 1.0, false, 1.0);
            assert!(
                y.is_finite() && !y.is_subnormal(),
                "streaming poisoned output: {y}"
            );
        }

        let mut clean_energy = 0.0f64;
        for i in 0..(FRAME_SIZE * 8) {
            let x = 0.35 * (2.0 * PI * 440.0 * (i as f32) / 48_000.0).sin();
            let y = ch.process_sample(x, 0.85, db_to_gain(-24.0), 1.0, false, 1.0);
            assert!(
                y.is_finite() && !y.is_subnormal(),
                "post-poison sample {i} invalid: {y}"
            );
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
        assert_eq!(
            init_ctx.latency_samples.load(Ordering::Relaxed),
            FRAME_SIZE as u32
        );

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
                assert!(
                    y.is_finite() && !y.is_subnormal(),
                    "1-sample block {i} invalid: {y}"
                );
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
        assert!(
            max_abs > 0.05 && max_abs < 5.0,
            "unexpected streaming peak amplitude: {max_abs}"
        );
    }

    #[test]
    fn extreme_sample_rates_initialize_and_process_cleanly() {
        let layout = HawaVoCleanPlugin::AUDIO_IO_LAYOUTS[1]; // mono

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

        for sr in [8_000.0f32, 44_100.0, 48_000.0, 96_000.0, 192_000.0] {
            let mut plugin = HawaVoCleanPlugin::default();
            let mut init_ctx = TestInitContext::new();
            let cfg = BufferConfig {
                sample_rate: sr,
                min_buffer_size: Some(1),
                max_buffer_size: 512,
                process_mode: ProcessMode::Realtime,
            };
            assert!(
                plugin.initialize(&layout, &cfg, &mut init_ctx),
                "failed init at {sr} Hz"
            );

            for block in 0..8usize {
                let mut mono = vec![(0..480)
                    .map(|i| {
                        let n = block * 480 + i;
                        0.3 * (2.0 * PI * 400.0 * (n as f32) / sr).sin()
                    })
                    .collect::<Vec<f32>>()];
                process_plugin_block(&mut plugin, &mut mono);
                for &y in &mono[0] {
                    assert!(
                        y.is_finite() && !y.is_subnormal(),
                        "bad sample at {sr} Hz: {y}"
                    );
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

            let strength = match (i / 137) % 3 {
                0 => 0.0,
                1 => 1.0,
                _ => 0.5,
            };
            let delta_listen = (i % 193) > 73;
            let floor_db = -60.0 + 60.0 * ((i % 257) as f32 / 256.0);
            let floor_gain = db_to_gain(floor_db);
            let wet_dry = if (i % 101) < 50 { 0.0 } else { 1.0 };
            let out_gain = db_to_gain(-12.0 + 24.0 * ((i % 89) as f32 / 88.0));

            let y = ch.process_sample(x, strength, floor_gain, wet_dry, delta_listen, out_gain);
            assert!(
                y.is_finite() && !y.is_subnormal() && y.abs() < 20.0,
                "parameter jump produced invalid sample at i={i}: {y}"
            );
        }

        assert_eq!(db_to_gain(f32::NAN), 0.0);
        assert_eq!(db_to_gain(f32::INFINITY), 0.0);
        assert_eq!(
            apply_output_mix(f32::NAN, 0.5, f32::NAN, false, f32::INFINITY),
            0.0
        );
        assert_eq!(apply_output_mix(0.5, f32::INFINITY, 0.5, true, 1.0), 0.5);
    }

    #[test]
    fn clap_description_matches_compiled_feature_set() {
        let desc = HawaVoCleanPlugin::CLAP_DESCRIPTION.expect("CLAP_DESCRIPTION must be set");
        assert!(
            desc.contains("960/480"),
            "description should document 960/480 STFT: {desc}"
        );
        #[cfg(feature = "onnx")]
        assert!(desc.contains("ONNX Runtime"));
        #[cfg(not(feature = "onnx"))]
        assert!(!desc.contains("ONNX Runtime"));
    }

    #[test]
    fn sqrt_hann_analysis_window_reduces_off_bin_spectral_leakage() {
        // Off-bin sinusoid at bin 10.5 (525 Hz @ 48 kHz, N=960)
        let mut fft = Fft960::new();
        let win = make_sqrt_hann_window();
        let mut rect_buf = [Complex32::default(); FRAME_SIZE];
        let mut win_buf = [Complex32::default(); FRAME_SIZE];

        for n in 0..FRAME_SIZE {
            let x = (2.0 * PI * 10.5 * (n as f32) / (FRAME_SIZE as f32)).sin();
            rect_buf[n] = Complex32::new(x, 0.0);
            win_buf[n] = Complex32::new(x * win[n], 0.0);
        }

        fft.forward(&mut rect_buf);
        fft.forward(&mut win_buf);

        // Measure far-sidelobe energy (>= 50 bins away from bin 10.5) normalized by total energy
        let far_ratio = |buf: &[Complex32; FRAME_SIZE]| -> f64 {
            let mut total = 0.0f64;
            let mut far = 0.0f64;
            for (k, c) in buf.iter().enumerate().take(NUM_BINS) {
                let p = c.norm_sq() as f64;
                total += p;
                if k >= 60 {
                    far += p;
                }
            }
            far / total.max(1e-12)
        };

        let rect_leak = far_ratio(&rect_buf);
        let win_leak = far_ratio(&win_buf);
        assert!(
            win_leak < rect_leak * 0.2,
            "Expected sqrt(Hann) analysis window to dramatically reduce spectral leakage: win={win_leak}, rect={rect_leak}"
        );
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn onnx_loader_returns_clean_error_when_model_missing() {
        let cfg = OnnxProcessorConfig {
            model_path: "/nonexistent/hawavoclean/model.onnx".to_string(),
            prefer_coreml: true,
            intra_op_threads: 1,
        };
        assert!(OnnxFrameProcessor::load(&cfg).is_err());
    }
}
