#!/usr/bin/env node
// Mock HawaVoClean engine for UI development (Node 22, no dependencies).
// Implements docs/ui-contract.md §1 (with addenda 1 and 2) with synthetic
// data: a fake 60 s clip, real peaks/spectrum computed from the synthesized
// signal, windowed /api/peaks over the same signal, a job that walks through
// the stages over ~6 s and returns a plausible schema-v2 HawaVoCleanReport
// (restore mode included), and /api/audio serving a generated WAV with Range
// support.
//
//   node dev/mock-engine.mjs [--port 8765] [--token dev] [--ui-dir dist]
//                            [--speakers character_01,character_02] [--speed 1]
//
// --speakers "" simulates an engine with no speaker profiles (the UI must
// hide the restore control); --speed N divides every job-stage delay by N so
// a scripted round-trip does not have to sit through the six real seconds.

import { createServer } from 'node:http';
import { createHash, randomUUID } from 'node:crypto';
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { basename, extname, join, normalize, resolve as resolvePath, sep } from 'node:path';

// ---------------------------------------------------------------------------
// CLI

const argv = process.argv.slice(2);
function arg(name, dflt) {
  const i = argv.indexOf(name);
  return i >= 0 && argv[i + 1] !== undefined ? argv[i + 1] : dflt;
}
const PORT = Number(arg('--port', '8765'));
const HOST = arg('--host', '127.0.0.1');
const TOKEN = arg('--token', 'dev');
const UI_DIR = arg('--ui-dir', null);
// Speaker profiles the fake engine claims to have (contract addendum 2).
// The real engine recomputes this per health request from profiles/ on disk;
// here it is fixed for the process, which is enough to drive the UI.
const SPEAKERS = arg('--speakers', 'character_01,character_02')
  .split(',')
  .map((s) => s.trim())
  .filter(Boolean)
  .sort();
// Delay divisor for the staged job walk (tests pass e.g. --speed 25).
const SPEED = Math.max(1, Number(arg('--speed', '1')) || 1);
const WORK_DIR = join(tmpdir(), 'hawavoclean-mock');
const RELEASE_IDENTITY_BYTES = await readFile(
  new URL('../../src/hawavoclean/release.json', import.meta.url),
);
const RELEASE_IDENTITY = JSON.parse(RELEASE_IDENTITY_BYTES.toString('utf8'));
const VERSION = RELEASE_IDENTITY.version;
const REPORT_RELEASE = {
  product: RELEASE_IDENTITY.product,
  version: RELEASE_IDENTITY.version,
  report_schema_version: RELEASE_IDENTITY.report_schema_version,
  identity_sha256: createHash('sha256').update(RELEASE_IDENTITY_BYTES).digest('hex'),
};
// Schema-v2 build provenance; field names mirror BuildMetadata in
// src/hawavoclean/report/schema.py (values are honest fakes, not claims).
const REPORT_BUILD = {
  provenance_schema_version: 1,
  artifact_type: 'source-tree',
  source_revision: 'mock-engine',
  source_date_epoch: 0,
  source_dirty: false,
  dependency_lock_sha256: createHash('sha256').update('mock-lock').digest('hex'),
  release_identity_sha256: REPORT_RELEASE.identity_sha256,
  build_id: 'mock-build',
  distribution_record_sha256: null,
};

// ---------------------------------------------------------------------------
// Deterministic synthesis

const SR = 48000;
const DURATION_S = 60;
const N = SR * DURATION_S;

function seedFrom(str) {
  const h = createHash('sha256').update(str).digest();
  return h.readUInt32LE(0);
}
function mulberry32(a) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Unit plan shared by the signal, the report and the verdict strip.
const UNITS = [
  { id: 0, start: 0.0, end: 11.0, speech: true, decision: 'enhanced' },
  { id: 1, start: 11.0, end: 14.0, speech: false, decision: 'original_no_speech' },
  { id: 2, start: 14.0, end: 30.0, speech: true, decision: 'enhanced' },
  { id: 3, start: 30.0, end: 44.0, speech: true, decision: 'original_reverted' },
  { id: 4, start: 44.0, end: 60.0, speech: true, decision: 'enhanced' },
];

/** Engine-side default output suffix per profile (server/app.py `_OUTPUT_SUFFIX`). */
const OUTPUT_SUFFIX = { studio: 'studio', lowband: 'lowband', production: 'clean' };

function isCleanedPath(p) {
  const b = basename(p).toLowerCase();
  return Object.values(OUTPUT_SUFFIX).some((s) => b.includes(`_${s}`));
}

/** Synthesize a speech-like mono signal. Cleaned variant: lower noise floor. */
function synthesize(path) {
  const cleaned = isCleanedPath(path);
  const stemSeed = seedFrom(basename(path).replace(/_(studio|lowband|clean)\.wav$/i, ''));
  const rnd = mulberry32(stemSeed);
  const out = new Float32Array(N);

  const noiseLin = cleaned ? 10 ** (-68 / 20) : 10 ** (-46 / 20);
  const hum = cleaned ? 0 : 10 ** (-54 / 20);

  // Phrase structure: within speech units, phrases of 1.2–3 s separated by
  // 0.25–0.6 s gaps; syllable modulation at ~4.2 Hz.
  const phrases = [];
  for (const u of UNITS) {
    if (!u.speech) continue;
    let t = u.start + 0.15 + rnd() * 0.3;
    while (t < u.end - 0.3) {
      const len = 1.2 + rnd() * 1.8;
      const end = Math.min(u.end - 0.1, t + len);
      phrases.push({ start: t, end, pitch: 105 + rnd() * 75, gain: 0.55 + rnd() * 0.4 });
      t = end + 0.25 + rnd() * 0.4;
    }
  }

  // pink-ish noise via 3 one-pole filters
  let b0 = 0, b1 = 0, b2 = 0;
  let phase = 0;
  let pIdx = 0;
  let lp = 0; // for fricative band
  for (let i = 0; i < N; i++) {
    const t = i / SR;
    const white = rnd() * 2 - 1;
    b0 = 0.99765 * b0 + white * 0.099046;
    b1 = 0.963 * b1 + white * 0.2965164;
    b2 = 0.57 * b2 + white * 1.0526913;
    const pink = (b0 + b1 + b2 + white * 0.1848) * 0.12;

    let s = pink * noiseLin * 3 + white * noiseLin * 0.9 + hum * Math.sin(2 * Math.PI * 50 * t) * 0.7 + hum * 0.4 * Math.sin(2 * Math.PI * 150 * t);

    while (pIdx < phrases.length && t > phrases[pIdx].end) pIdx++;
    const ph = phrases[pIdx];
    if (ph && t >= ph.start) {
      const rel = (t - ph.start) / (ph.end - ph.start);
      const env = Math.sin(Math.PI * Math.min(1, Math.max(0, rel))) ** 0.6;
      const syll = 0.55 + 0.45 * Math.max(0, Math.sin(2 * Math.PI * 4.2 * t + ph.pitch));
      const amp = ph.gain * env * syll;
      const f0 = ph.pitch * (1 + 0.06 * Math.sin(2 * Math.PI * 0.7 * t) + 0.015 * Math.sin(2 * Math.PI * 5.5 * t));
      phase += (2 * Math.PI * f0) / SR;
      // harmonic stack with formant-ish weighting
      let v = 0;
      for (let h = 1; h <= 14; h++) {
        const fh = f0 * h;
        const formant = Math.exp(-((fh - 600) ** 2) / (2 * 300 ** 2)) + 0.6 * Math.exp(-((fh - 1700) ** 2) / (2 * 450 ** 2)) + 0.35 * Math.exp(-((fh - 2700) ** 2) / (2 * 500 ** 2));
        v += (Math.sin(phase * h) * (0.12 + formant)) / h ** 0.7;
      }
      // fricative bursts (high band noise) on syllable onsets
      const fric = Math.max(0, Math.sin(2 * Math.PI * 4.2 * t + ph.pitch + 1.3)) ** 8;
      lp = 0.82 * lp + 0.18 * white;
      const hiNoise = (white - lp) * 0.45 * fric;
      s += amp * (v * 0.26 + hiNoise);
    }
    out[i] = Math.max(-0.98, Math.min(0.98, s));
  }
  if (cleaned) {
    // cleaned master: normalised a touch hotter (finishing stage)
    for (let i = 0; i < N; i++) out[i] = Math.max(-0.95, Math.min(0.95, out[i] * 1.45));
  }
  return out;
}

function wavFromFloat(samples) {
  const bytes = 44 + samples.length * 2;
  const buf = Buffer.alloc(bytes);
  buf.write('RIFF', 0);
  buf.writeUInt32LE(bytes - 8, 4);
  buf.write('WAVE', 8);
  buf.write('fmt ', 12);
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20);
  buf.writeUInt16LE(1, 22);
  buf.writeUInt32LE(SR, 24);
  buf.writeUInt32LE(SR * 2, 28);
  buf.writeUInt16LE(2, 32);
  buf.writeUInt16LE(16, 34);
  buf.write('data', 36);
  buf.writeUInt32LE(samples.length * 2, 40);
  let o = 44;
  for (let i = 0; i < samples.length; i++) {
    const v = Math.max(-1, Math.min(1, samples[i]));
    buf.writeInt16LE(Math.round(v * 32767), o);
    o += 2;
  }
  return buf;
}

// --- tiny radix-2 FFT for the long-term spectrum -----------------------------
function fftMag(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = i + k + len / 2;
        const tr = re[b] * cr - im[b] * ci;
        const ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr; im[a] += ti;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr; cr = ncr;
      }
    }
  }
}

function longTermSpectrum(samples) {
  const FFT = 8192;
  const frames = 48;
  const acc = new Float64Array(FFT / 2);
  const re = new Float64Array(FFT), im = new Float64Array(FFT);
  const win = new Float64Array(FFT);
  for (let i = 0; i < FFT; i++) win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (FFT - 1));
  let winSum = 0;
  for (let i = 0; i < FFT; i++) winSum += win[i];
  for (let f = 0; f < frames; f++) {
    const start = Math.floor(((f + 0.5) / frames) * (samples.length - FFT));
    for (let i = 0; i < FFT; i++) { re[i] = samples[start + i] * win[i]; im[i] = 0; }
    fftMag(re, im);
    for (let k = 0; k < FFT / 2; k++) {
      const mag = Math.hypot(re[k], im[k]) * (2 / winSum); // sine of amplitude 1 → 1.0
      acc[k] += mag * mag;
    }
  }
  const binHz = SR / FFT;
  const freqs = [], db = [];
  const nBands = Math.floor(12 * Math.log2(20000 / 40)) + 1;
  for (let b = 0; b < nBands; b++) {
    const fc = 40 * 2 ** (b / 12);
    if (fc > SR / 2) break;
    const lo = fc * 2 ** (-1 / 24), hi = fc * 2 ** (1 / 24);
    const k0 = Math.max(1, Math.floor(lo / binHz)), k1 = Math.min(FFT / 2 - 1, Math.ceil(hi / binHz));
    let m = 0;
    for (let k = k0; k <= k1; k++) m = Math.max(m, acc[k] / frames);
    const v = 10 * Math.log10(Math.max(1e-12, m));
    freqs.push(Number(fc.toFixed(2)));
    db.push(Number(Math.max(-120, v).toFixed(2)));
  }
  return { freqs_hz: freqs, db };
}

function analyze(path, buckets) {
  const s = getSignal(path);
  const cleaned = isCleanedPath(path);
  const min = new Array(buckets), max = new Array(buckets), rms = new Array(buckets);
  for (let b = 0; b < buckets; b++) {
    const i0 = Math.floor((b / buckets) * N), i1 = Math.floor(((b + 1) / buckets) * N);
    let lo = Infinity, hi = -Infinity, sq = 0;
    for (let i = i0; i < i1; i++) { const v = s[i]; if (v < lo) lo = v; if (v > hi) hi = v; sq += v * v; }
    const n = Math.max(1, i1 - i0);
    min[b] = Number(lo.toFixed(4)); max[b] = Number(hi.toFixed(4));
    const r = Math.sqrt(sq / n);
    rms[b] = r > 1e-6 ? Number((20 * Math.log10(r)).toFixed(2)) : -120;
  }
  const above = rms.filter((v) => v > -120).sort((a, b) => a - b);
  const nf = above.length ? above[Math.floor(above.length * 0.1)] : -120;
  return {
    path,
    duration_s: DURATION_S,
    sample_rate: SR,
    channels: 1,
    peaks: { min, max },
    rms_db: rms,
    spectrum: longTermSpectrum(s),
    loudness: cleaned ? { integrated_lufs: -19.0, true_peak_dbtp: -1.0 } : { integrated_lufs: -23.4, true_peak_dbtp: -3.1 },
    noise_floor_db: Number(nf.toFixed(1)),
  };
}

/**
 * `PeaksWindow` over `[startS, endS)` — the shape and rounding of the real
 * `compute_peaks_window` (server/analysis.py): the span snaps to the sample
 * grid, `buckets` is clamped down so every bucket covers >= 1 sample, and
 * `samples_per_bucket` is the ceiling, so 1 means raw samples on screen.
 */
function peaksWindow(path, startS, endS, buckets) {
  const s = getSignal(path);
  const startSample = Math.floor(startS * SR);
  const endSample = Math.min(N, Math.round(Math.min(endS, DURATION_S) * SR));
  const n = endSample - startSample;
  const b = Math.min(buckets, n);
  const min = new Array(b), max = new Array(b), rms = new Array(b);
  for (let i = 0; i < b; i++) {
    const i0 = startSample + Math.floor((i / b) * n), i1 = startSample + Math.floor(((i + 1) / b) * n);
    let lo = Infinity, hi = -Infinity, sq = 0;
    for (let j = i0; j < i1; j++) { const v = s[j]; if (v < lo) lo = v; if (v > hi) hi = v; sq += v * v; }
    const cnt = Math.max(1, i1 - i0);
    min[i] = Number(lo.toFixed(6)); max[i] = Number(hi.toFixed(6));
    const r = Math.sqrt(sq / cnt);
    rms[i] = r > 1e-6 ? Number((20 * Math.log10(r)).toFixed(2)) : -120;
  }
  return {
    path,
    start_s: Number((startSample / SR).toFixed(6)),
    end_s: Number((endSample / SR).toFixed(6)),
    sample_rate: SR,
    channels: 1,
    duration_s: Number(DURATION_S.toFixed(4)),
    samples_per_bucket: Math.ceil(n / b),
    peaks: { min, max },
    rms_db: rms,
  };
}

const signalCache = new Map();
function getSignal(path) {
  let s = signalCache.get(path);
  if (!s) { s = synthesize(path); signalCache.set(path, s); }
  return s;
}
const wavCache = new Map();
function getWav(path) {
  let w = wavCache.get(path);
  if (!w) { w = wavFromFloat(getSignal(path)); wavCache.set(path, w); }
  return w;
}

// ---------------------------------------------------------------------------
// Jobs

const jobs = new Map();
const sourceMap = new Map();
const queue = [];
let runningJob = null;

function cleanStem(p) {
  let b = basename(p);
  for (;;) {
    const e = extname(b);
    if (!e || !/^\.(wav|m4a|mp4|mov|mp3|flac|aac|aif|aiff|ogg|opus|webm|mkv)$/i.test(e)) break;
    b = b.slice(0, -e.length);
  }
  return b;
}

/**
 * Restoration section of a restore-mode report. Field names are copied from
 * `hawavoclean.restoration.report.RestorationReport` (the engine serialises
 * the dataclass verbatim), so the UI reads the same wire shape it will get
 * from a real run. Dev hook: an input whose name contains "revert" makes
 * Guard R reject everything, so the honest-FAIL presentation (the Natural
 * master shipped) can be exercised.
 */
function makeRestoration(job) {
  const failed = /revert/i.test(basename(job.input_path));
  const manual = job.cutoff_hz !== null && job.cutoff_hz !== undefined;
  const cutoff = manual ? job.cutoff_hz : 7800.0;
  return {
    mode: 'restore',
    speaker_id: job.speaker_id,
    profile_hash: createHash('sha256').update(`profile:${job.speaker_id}`).digest('hex'),
    natural_output_hash: createHash('sha256').update(`natural:${job.id}`).digest('hex'),
    bandwidth: {
      effective_cutoff_hz: cutoff,
      confidence: manual ? 1.0 : 0.92,
      shape: manual ? 'manual_override' : 'codec_lowpass',
      restore_recommended: true,
      evidence: {
        spectral_rolloff: cutoff * 0.96,
        above_cutoff_snr_db: -14.2,
        stationarity: 0.83,
        high_band_energy_ratio_db: -41.6,
      },
      cutoff_mode: manual ? 'manual' : 'auto',
    },
    restorer: {
      name: 'hawarestore-kd',
      commit: '26dc21c44e11f9f19e823f02b0d4641dd5ea5af2',
      weights_sha256: createHash('sha256').update('mock-weights').digest('hex'),
      checkpoint_path: '/mock/models/hawarestore-kd/checkpoint.pt',
      device: 'cpu',
      seed_policy: 'deterministic_job_id',
      solver: 'midpoint',
      steps: 4,
      guidance_scale: 0.0,
    },
    segments: failed
      ? { restored: 0, reduced: 0, reverted: 1, bypassed: 0, errors: 0 }
      : { restored: 1, reduced: 0, reverted: 0, bypassed: 0, errors: 0 },
    guard_r: failed
      ? {
          verdict: 'FAIL',
          accepted_strength: 0.0,
          reason:
            'All candidate strengths rejected; reverted to Natural-safe audio. ' +
            'Last rejection — strength 0.35: protected band deviation 1.9 dB exceeds 1.0 dB',
          protected_band: {},
          ctc: {},
          highband_events: {},
          harmonic: {},
          speaker: {},
        }
      : {
          verdict: 'PASS',
          accepted_strength: 0.7,
          reason: 'Accepted strength 0.70: all guard layers passed',
          protected_band: {},
          ctc: {},
          highband_events: {},
          harmonic: {},
          speaker: {},
        },
    review_timecodes: [],
  };
}

function makeSmartSafeRestoration(job) {
  return {
    cutoff_hz: 8000,
    speaker_id: null,
    speaker_enrolled: false,
    selected_route: 'production',
    confidence: 0.94,
    abstained: false,
    reason: 'Production Wiener filter selected with high confidence',
    fallback_route: 'production',
    candidates: [
      {
        route: 'production',
        score: 0.94,
        confidence: 0.94,
        rank: 1,
        selected: true,
        rejection_reason: null,
      },
      {
        route: 'studio',
        score: 0.72,
        confidence: 0.72,
        rank: 2,
        selected: false,
        rejection_reason: 'Phase coherence margin lower than production',
      },
      {
        route: 'lowband',
        score: 0.61,
        confidence: 0.61,
        rank: 3,
        selected: false,
        rejection_reason: 'High band bandwidth available, crossover unnecessary',
      },
    ],
    detections: {
      clipping_ratio: 0.0,
      snr_estimate_db: 18.5,
      bandwidth_hz: 16000,
      reverberant: false,
    },
    passes: [],
    segments: { restored: 0, reduced: 0, reverted: 0, bypassed: 0, errors: 0 },
    guard_r: {
      verdict: 'PASS',
      accepted_strength: 0,
      reason: 'Smart Safe routing selected production',
      protected_band: {},
      ctc: {},
      highband_events: {},
      harmonic: {},
      speaker: {},
    },
    review_timecodes: [],
  };
}

function makeReport(job) {
  const units = UNITS.map((u) => {
    const base = {
      unit_id: u.id, channel: 0,
      start_sample: Math.round(u.start * SR), end_sample: Math.round(u.end * SR),
      start_time_s: u.start, end_time_s: u.end, is_speech: u.speech,
      input_sha256: createHash('sha256').update(`in${u.id}${job.id}`).digest('hex'),
      candidate_sha256: u.speech ? createHash('sha256').update(`cand${u.id}${job.id}`).digest('hex') : null,
      output_sha256: createHash('sha256').update(`out${u.id}${job.id}`).digest('hex'),
      guard_b_verdict: null, guard_b_scores: {}, finish_preset_applied: 'bypass', finish_actions: [],
      runtime_ms: 180 + u.id * 37.5,
    };
    if (!u.speech) {
      return { ...base, guard_a_verdict: 'NO_SPEECH', guard_a_scores: {}, chosen_strength: 0,
        final_decision: 'original_no_speech', decision_reason: 'Non-speech unit: neural enhancement bypassed.' };
    }
    if (u.decision === 'enhanced') {
      return { ...base, guard_a_verdict: 'PASS',
        guard_a_scores: { envelope_correlation: 0.982 - u.id * 0.004, timing_drift_ms: 1.2 + u.id * 0.3, spectral_hole_score: 0.031, musical_noise_score: 0.044, consonant_retention: 0.93, clipping_samples: 0 },
        chosen_strength: 1.0, guard_b_verdict: 'PASS', guard_b_scores: { envelope_correlation: 0.995, timing_drift_ms: 0.4 },
        finish_preset_applied: 'gentle', finish_actions: ['dc_subsonic', 'dehum', 'dynamic_eq', 'deess', 'level_rider'],
        final_decision: 'enhanced', decision_reason: 'Passed Guard A with strength s=1.00' };
    }
    return { ...base, guard_a_verdict: 'REVERT',
      guard_a_scores: { envelope_correlation: 0.871, timing_drift_ms: 46.3, spectral_hole_score: 0.19, musical_noise_score: 0.27, consonant_retention: 0.71, clipping_samples: 0 },
      chosen_strength: 0, final_decision: 'original_reverted',
      decision_reason: "Reverted to original audio: ['timing drift 46.3 ms exceeds 40.0 ms', 'musical noise 0.27 exceeds 0.20']" };
  });
  const enhanced = units.filter((u) => u.final_decision === 'enhanced').length;
  const reverted = units.filter((u) => u.final_decision === 'original_reverted').length;
  const noSpeech = units.filter((u) => u.final_decision === 'original_no_speech').length;
  const media = (p, lufs, tp) => ({ path: p, sha256: createHash('sha256').update(p).digest('hex'), sample_rate: SR, channels: 1, samples: N, duration_s: DURATION_S, integrated_lufs: lufs, true_peak_dbtp: tp });
  return {
    schema_version: RELEASE_IDENTITY.report_schema_version, release: REPORT_RELEASE, build: REPORT_BUILD,
    job_id: job.id, config_hash: createHash('sha256').update('mock-config').digest('hex'),
    input: media(job.input_path, -23.4, -3.1), output: media(job.output_path, -19.0, -1.0),
    core: {
      studio: { id: 'studio-dfn3-48k-v1', algorithm: 'WPE + DeepFilterNet3', params_hash: 'b1f7' + '0'.repeat(60), phase_coherent: true },
      lowband: { id: 'studio-dfn3-lowband-48k-v1', algorithm: 'DeepFilterNet3 crossed over with the original at 1000 Hz', params_hash: 'd5e8' + '0'.repeat(60), phase_coherent: false },
      production: { id: 'wiener-dd-48k-v1', algorithm: 'Wiener decision-directed', params_hash: 'a9c2' + '0'.repeat(60), phase_coherent: true },
    }[job.profile],
    guard: { id: 'spectral-guard-v1', probe_hash: 'c3d4' + '0'.repeat(60), calibration_id: job.profile === 'production' ? 'guard-calibration' : 'guard-calibration-studio' },
    environment: { platform: 'mock', os_version: process.version, python_version: 'n/a', numpy_version: 'n/a', scipy_version: 'n/a', soundfile_version: 'n/a', cpu_model: null },
    summary: { units_total: units.length, enhanced, reverted, unverified: 0, error_passthrough: 0, continuity_reverted: 0, no_speech: noSpeech, finish_applied: enhanced, finish_bypassed: units.length - enhanced },
    review_timecodes: units.filter((u) => u.final_decision === 'original_reverted').map((u) => ({ unit_id: u.unit_id, start_time_s: u.start_time_s, end_time_s: u.end_time_s, channel: 0, verdict: 'REVERT', reason: u.decision_reason })),
    units,
    passes: [],
    ...(job.mode === 'restore'
      ? { restoration: makeRestoration(job) }
      : job.mode === 'smart_safe'
        ? { restoration: makeSmartSafeRestoration(job) }
        : {}),
  };
}

function publicStatus(job) {
  const o = {
    job_id: job.id, state: job.state, stage: job.stage, progress: job.progress, message: job.message,
    unit: job.unit, input_path: job.input_path, output_path: job.output_path, report_path: job.report_path,
    profile: job.profile, mode: job.mode, started_at: job.started_at, finished_at: job.finished_at,
  };
  // Contract addendum 2: `mode` is always present; the restore parameters
  // appear only on a restore snapshot (natural stays revision-1 compatible).
  if (job.mode === 'restore') {
    o.speaker_id = job.speaker_id;
    o.cutoff_hz = job.cutoff_hz;
  }
  if (job.state === 'failed') o.error = job.error;
  if (job.state === 'done') o.report = job.report;
  return o;
}

function emit(job) {
  job.version += 1;
  for (const fn of job.listeners) fn();
}

function schedule() {
  if (runningJob || queue.length === 0) return;
  const job = queue.shift();
  runningJob = job;
  runJob(job);
}

function runJob(job) {
  job.state = 'running';
  job.started_at = new Date().toISOString();
  const speechUnits = UNITS.length;
  const steps = [
    [250, 'preflight', 0.02, 'Preflight checks passed', null],
    [350, 'decode', 0.05, `Decoded ${DURATION_S.toFixed(1)} s @ 48 kHz, 1 ch`, null],
    [300, 'segment', 0.08, `${speechUnits} units`, null],
  ];
  for (let i = 1; i <= speechUnits; i++) {
    const p0 = 0.08 + ((i - 0.5) / speechUnits) * 0.72;
    const p1 = 0.08 + (i / speechUnits) * 0.72;
    steps.push([520, 'enhance', p0, `Enhancing unit ${i}/${speechUnits}`, { index: i, total: speechUnits }]);
    const dec = UNITS[i - 1].decision;
    const label = dec === 'enhanced' ? 'ENHANCED' : dec === 'original_no_speech' ? 'NO_SPEECH' : 'REVERTED';
    steps.push([260, 'guard', p1, `Unit ${i}/${speechUnits}: ${label}`, { index: i, total: speechUnits }]);
  }
  steps.push([700, 'finish', 0.88, 'Finishing: EQ/limiter/loudness', null]);
  steps.push([450, 'publish', 0.98, 'Publishing master', null]);

  // Dev hook: an input whose name contains "fail" dies during enhancement so
  // the UI's failure path can be exercised.
  const failAt = /fail/i.test(basename(job.input_path)) ? 5 : -1;

  let idx = 0;
  const tick = () => {
    if (job.state !== 'running') return;
    if (idx === failAt) {
      job.stage = 'error'; job.message = 'Enhancement core failed'; job.unit = null;
      job.state = 'failed'; job.finished_at = new Date().toISOString();
      job.error = { code: 'ENHANCEMENT_FAILURE', message: 'Enhancement worker exited with code 137 (out of memory) while processing unit 2/5. stderr: torch.cuda.OutOfMemoryError: CUDA out of memory' };
      emit(job);
      runningJob = null; schedule();
      return;
    }
    if (idx >= steps.length) {
      job.stage = 'done'; job.progress = 1; job.message = 'Done'; job.unit = null;
      job.state = 'done'; job.finished_at = new Date().toISOString();
      job.report = makeReport(job);
      wavCache.delete(job.output_path); signalCache.delete(job.output_path);
      // Persist the report sidecar ONLY inside the mock work dir — never next
      // to real user files (the default report_path points at the input's dir).
      if (job.report_path.startsWith(WORK_DIR)) {
        writeFile(job.report_path, JSON.stringify(job.report, null, 2)).catch(() => {});
      }
      emit(job);
      runningJob = null; schedule();
      return;
    }
    const [rawDelay, stage, progress, message, unit] = steps[idx++];
    const delay = rawDelay / SPEED;
    job.timer = setTimeout(() => {
      if (job.state !== 'running') return;
      job.stage = stage; job.progress = progress; job.message = message; job.unit = unit;
      emit(job);
      tick();
    }, delay);
  };
  emit(job);
  tick();
}

function cancelJob(job) {
  if (job.state === 'done' || job.state === 'failed' || job.state === 'cancelled') return;
  clearTimeout(job.timer);
  const qi = queue.indexOf(job);
  if (qi >= 0) queue.splice(qi, 1);
  job.state = 'cancelled'; job.stage = 'error'; job.message = 'Cancelled by user'; job.finished_at = new Date().toISOString();
  emit(job);
  if (runningJob === job) { runningJob = null; schedule(); }
}

// ---------------------------------------------------------------------------
// HTTP helpers

function cors(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'X-Hawa-Token, Content-Type, Range');
  res.setHeader('Access-Control-Expose-Headers', 'Content-Length, Content-Range, Accept-Ranges');
}
function json(res, code, body) {
  cors(res);
  const data = Buffer.from(JSON.stringify(body));
  res.writeHead(code, { 'Content-Type': 'application/json', 'Content-Length': data.length });
  res.end(data);
}
function err(res, code, error, message) { json(res, code, { error, message }); }
function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}
function authorized(req, url) {
  const h = req.headers['x-hawa-token'];
  return h === TOKEN || url.searchParams.get('token') === TOKEN;
}
function underRoot(n, root) {
  return n === root || n.startsWith(root.endsWith(sep) ? root : root + sep);
}
function pathAllowed(p) {
  if (typeof p !== 'string' || !p.startsWith('/')) return false;
  const n = normalize(p);
  const home = process.env.HOME || '/';
  // Contract path policy: under $HOME, /Volumes, or the work dir — nothing else.
  return underRoot(n, home) || underRoot(n, '/Volumes') || underRoot(n, WORK_DIR);
}

// The real engine pins every request model `extra="forbid"`: a misspelled
// field is refused with 422, never silently dropped (contract addendum 2).
// The mock refuses the same way so the UI's error path is exercisable.
function unknownField(body, allowed) {
  return Object.keys(body).find((k) => !allowed.includes(k)) ?? null;
}

const MIME = { '.wav': 'audio/wav', '.m4a': 'audio/mp4', '.mp4': 'audio/mp4', '.mp3': 'audio/mpeg', '.flac': 'audio/flac', '.aac': 'audio/aac', '.mov': 'video/quicktime' };
const STATIC_MIME = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.json': 'application/json', '.map': 'application/json', '.png': 'image/png', '.ico': 'image/x-icon', '.woff2': 'font/woff2' };

function serveBuffer(req, res, buf, mime) {
  cors(res);
  res.setHeader('Accept-Ranges', 'bytes');
  const range = req.headers.range;
  const m = range && /^bytes=(\d*)-(\d*)$/.exec(range);
  if (m && (m[1] || m[2])) {
    let start = m[1] ? Number(m[1]) : Math.max(0, buf.length - Number(m[2]));
    let end = m[1] && m[2] ? Math.min(buf.length - 1, Number(m[2])) : buf.length - 1;
    if (start > end || start >= buf.length) {
      res.writeHead(416, { 'Content-Range': `bytes */${buf.length}` });
      return res.end();
    }
    res.writeHead(206, { 'Content-Type': mime, 'Content-Length': end - start + 1, 'Content-Range': `bytes ${start}-${end}/${buf.length}` });
    return res.end(buf.subarray(start, end + 1));
  }
  res.writeHead(200, { 'Content-Type': mime, 'Content-Length': buf.length });
  res.end(buf);
}

async function serveStatic(req, res, pathname) {
  if (!UI_DIR) return err(res, 404, 'not_found', 'No UI dir configured');
  const root = resolvePath(UI_DIR);
  let rel = decodeURIComponent(pathname);
  if (rel === '/' || rel === '') rel = '/index.html';
  const file = resolvePath(join(root, rel));
  if (file !== root && !file.startsWith(root + sep)) return err(res, 403, 'forbidden', 'Path outside ui dir');
  try {
    const st = await stat(file);
    if (!st.isFile()) throw new Error('not a file');
    const buf = await readFile(file);
    serveBuffer(req, res, buf, STATIC_MIME[extname(file).toLowerCase()] || 'application/octet-stream');
  } catch {
    err(res, 404, 'not_found', `No such file: ${rel}`);
  }
}

function parseMultipart(buf, contentType) {
  const m = /boundary=(?:"([^"]+)"|([^;]+))/i.exec(contentType || '');
  if (!m) return null;
  const boundary = Buffer.from(`--${m[1] || m[2]}`);
  let pos = buf.indexOf(boundary);
  while (pos >= 0) {
    const start = pos + boundary.length;
    if (buf.subarray(start, start + 2).toString() === '--') break;
    const headEnd = buf.indexOf('\r\n\r\n', start);
    if (headEnd < 0) break;
    const head = buf.subarray(start + 2, headEnd).toString();
    const next = buf.indexOf(boundary, headEnd + 4);
    const body = buf.subarray(headEnd + 4, next >= 0 ? next - 2 : buf.length);
    const name = /name="([^"]*)"/.exec(head)?.[1];
    const filename = /filename="([^"]*)"/.exec(head)?.[1];
    if (name === 'file') return { filename: filename || 'upload.bin', body };
    pos = next;
  }
  return null;
}

// ---------------------------------------------------------------------------
// Server

const server = createServer(async (req, res) => {
  const url = new URL(req.url || '/', `http://${req.headers.host || 'localhost'}`);
  const p = url.pathname;
  if (req.method === 'OPTIONS') { cors(res); res.writeHead(204); return res.end(); }

  if (!p.startsWith('/api/')) {
    if (req.method !== 'GET') return err(res, 405, 'method_not_allowed', 'GET only');
    return serveStatic(req, res, p);
  }
  if (!authorized(req, url)) return json(res, 401, { error: 'unauthorized' });

  try {
    if (p === '/api/health' && req.method === 'GET') {
      return json(res, 200, {
        ok: true, version: VERSION, profiles: ['studio', 'lowband', 'production'],
        speakers: SPEAKERS, restore_available: SPEAKERS.length > 0,
        engine_pid: process.pid,
      });
    }
    if (p === '/api/analyze' && req.method === 'POST') {
      const body = JSON.parse((await readBody(req)).toString() || '{}');
      const unknown = unknownField(body, ['path', 'buckets']);
      if (unknown !== null) return err(res, 422, 'bad_request', `unknown field: ${unknown}`);
      if (!pathAllowed(body.path)) return err(res, 403, 'forbidden', 'Path outside allowed roots');
      const buckets = Math.max(16, Math.min(8000, Number(body.buckets) || 1200));
      await new Promise((r) => setTimeout(r, 350)); // feel like real decode
      return json(res, 200, analyze(body.path, buckets));
    }
    if (p === '/api/peaks' && req.method === 'POST') {
      // Contract addendum 1: same semantics as /api/analyze's waveform
      // fields, computed over [start_s, end_s) of the synthesized signal.
      const body = JSON.parse((await readBody(req)).toString() || '{}');
      const unknown = unknownField(body, ['path', 'start_s', 'end_s', 'buckets']);
      if (unknown !== null) return err(res, 422, 'bad_request', `unknown field: ${unknown}`);
      if (!pathAllowed(body.path)) return err(res, 403, 'forbidden', 'Path outside allowed roots');
      const start = Number(body.start_s), end = Number(body.end_s);
      // Real engine: only unknown fields earn 422 — malformed values and an
      // out-of-range window keep the historical 400.
      if (!Number.isFinite(start) || start < 0 || !Number.isFinite(end) || end <= start) {
        return err(res, 400, 'bad_request', `bad window: start_s=${body.start_s}, end_s=${body.end_s}`);
      }
      if (start >= DURATION_S) {
        return err(res, 400, 'bad_request', `start_s ${start} is at or past the end of the file (${DURATION_S.toFixed(4)} s)`);
      }
      const buckets = Math.max(1, Math.min(8000, Number(body.buckets) || 1200));
      return json(res, 200, peaksWindow(body.path, start, end, buckets));
    }
    if (p === '/api/jobs' && req.method === 'POST') {
      const body = JSON.parse((await readBody(req)).toString() || '{}');
      const unknown = unknownField(body, ['input_path', 'profile', 'output_path', 'overwrite', 'mode', 'speaker_id', 'cutoff_hz']);
      if (unknown !== null) return err(res, 422, 'bad_request', `unknown field: ${unknown}`);
      // Restore-mode cross-field rules, exactly the real engine's 422s
      // (contract addendum 2). A JSON null reads as "not sent", as pydantic's
      // `str | None = None` does.
      const mode = body.mode === 'restore' ? 'restore' : 'natural';
      if (mode === 'restore') {
        if (body.speaker_id == null || body.speaker_id === '') {
          return err(res, 422, 'bad_request', 'mode "restore" requires speaker_id (see /api/health)');
        }
      } else {
        if (body.speaker_id != null) return err(res, 422, 'bad_request', 'speaker_id is only valid when mode is "restore"');
        if (body.cutoff_hz != null) return err(res, 422, 'bad_request', 'cutoff_hz is only valid when mode is "restore"');
      }
      if (body.speaker_id != null && !/^[a-z0-9_]{1,64}$/.test(body.speaker_id)) {
        return err(res, 422, 'bad_request', 'speaker_id must match ^[a-z0-9_]{1,64}$');
      }
      if (body.cutoff_hz != null && !(Number.isFinite(body.cutoff_hz) && body.cutoff_hz > 0)) {
        return err(res, 400, 'bad_request', 'cutoff_hz must be a positive finite number');
      }
      if (!pathAllowed(body.input_path)) return err(res, 403, 'forbidden', 'Path outside allowed roots');
      if (body.output_path !== undefined && !pathAllowed(body.output_path)) {
        return err(res, 403, 'forbidden', 'Output path outside allowed roots');
      }
      const profile = ['production', 'lowband'].includes(body.profile) ? body.profile : 'studio';
      const dir = body.input_path.slice(0, body.input_path.lastIndexOf('/'));
      const stem = cleanStem(body.input_path);
      const output_path = body.output_path || `${dir}/${stem}_${OUTPUT_SUFFIX[profile]}.wav`;
      const report_path = output_path.replace(/\.wav$/i, '') + '.hawavoclean.json';
      const job = {
        id: `j_${randomUUID().replace(/-/g, '').slice(0, 12)}`, state: 'queued', stage: 'preflight', progress: 0,
        message: 'Queued', unit: null, input_path: body.input_path, output_path, report_path, profile,
        mode, speaker_id: mode === 'restore' ? body.speaker_id : null,
        cutoff_hz: mode === 'restore' ? (body.cutoff_hz ?? null) : null,
        started_at: null, finished_at: null, error: null, report: null, listeners: new Set(), version: 0, timer: null,
      };
      jobs.set(job.id, job);
      queue.push(job);
      schedule();
      return json(res, 202, { job_id: job.id, output_path, report_path });
    }
    const jm = /^\/api\/jobs\/([^/]+)(\/events|\/cancel)?$/.exec(p);
    if (jm) {
      const job = jobs.get(jm[1]);
      if (!job) return err(res, 404, 'not_found', `Unknown job ${jm[1]}`);
      if (!jm[2] && req.method === 'GET') return json(res, 200, publicStatus(job));
      if (jm[2] === '/cancel' && req.method === 'POST') { cancelJob(job); return json(res, 200, { ok: true }); }
      if (jm[2] === '/events' && req.method === 'GET') {
        cors(res);
        res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive', 'X-Accel-Buffering': 'no' });
        const send = (event, data) => res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
        let lastSent = 0, pending = null;
        const push = () => {
          const now = Date.now();
          const terminal = job.state === 'done' || job.state === 'failed' || job.state === 'cancelled';
          const go = () => { pending = null; lastSent = Date.now(); send('status', publicStatus(job)); if (terminal) { send('end', {}); cleanup(); res.end(); } };
          if (terminal || now - lastSent >= 50) { clearTimeout(pending); go(); }
          else if (!pending) pending = setTimeout(go, 50 - (now - lastSent));
        };
        const ping = setInterval(() => res.write(': ping\n\n'), 15000);
        const cleanup = () => { clearInterval(ping); clearTimeout(pending); job.listeners.delete(push); };
        job.listeners.add(push);
        req.on('close', cleanup);
        push();
        return;
      }
    }
    if (p === '/api/audio' && req.method === 'GET') {
      const path = url.searchParams.get('path') || '';
      if (!pathAllowed(path)) return err(res, 403, 'forbidden', 'Path outside allowed roots');
      const ext = extname(path).toLowerCase();
      // Real uploaded files are streamed as-is; anything else is synthesized.
      if (path.startsWith(WORK_DIR)) {
        try { const buf = await readFile(path); return serveBuffer(req, res, buf, MIME[ext] || 'application/octet-stream'); } catch { /* fall through to synth */ }
      }
      return serveBuffer(req, res, getWav(path), 'audio/wav');
    }
    if (p === '/api/v1/capabilities' && req.method === 'GET') {
      return json(res, 200, {
        schemaVersion: 1,
        capabilities: [
          {
            capabilityId: 'preserve',
            available: false,
            maturity: 'blocked',
            reason: 'Preserve is a Smart Safe candidate, but qualified Smart Safe routing is not yet available through the versioned job API',
          },
          { capabilityId: 'production', available: true, maturity: 'qualified', providers: ['cpu'] },
          { capabilityId: 'studio', available: true, maturity: 'qualified', providers: ['cpu'] },
          { capabilityId: 'lowband', available: true, maturity: 'qualified', providers: ['cpu'] },
          {
            capabilityId: 'lowband_then_production',
            available: false,
            maturity: 'experimental',
            reason: 'The route is not yet wired through the versioned job API',
          },
          {
            capabilityId: 'smart_analysis',
            available: true,
            maturity: 'experimental',
            reason: 'Bounded streaming acoustic proxies are available, but they are not a calibrated Sorani classifier and cannot qualify Smart Safe routing',
            providers: ['cpu'],
          },
          {
            capabilityId: 'smart_safe',
            available: true,
            maturity: 'qualified',
            providers: ['cpu'],
          },
          {
            capabilityId: 'restore_source',
            available: false,
            maturity: 'blocked',
            reason: 'No qualified signed source-conditioned Sorani Restore pack is installed',
          },
          {
            capabilityId: 'restore_enrolled',
            available: false,
            maturity: 'blocked',
            reason: 'No qualified signed enrolled-speaker Sorani Restore pack is installed',
          },
          {
            capabilityId: 'cloud',
            available: false,
            maturity: 'blocked',
            reason: 'Invite-only UAE cloud execution is not deployed',
          },
        ],
      });
    }
    if (p === '/api/v1/jobs' && req.method === 'POST') {
      const body = JSON.parse((await readBody(req)).toString() || '{}');
      const sourceIds = body.source_ids || body.sourceIds;
      if (!Array.isArray(sourceIds) || sourceIds.length === 0) {
        return err(res, 422, 'bad_request', 'source_ids must be a non-empty array');
      }
      const strategy = body.strategy || {};
      const stratType = strategy.strategy || strategy.type;
      if (stratType === 'manual') {
        const route = strategy.route;
        if (route === 'restore_enrolled' || route === 'restore_source' || route === 'preserve') {
          return err(res, 503, 'capability_blocked', `Route "${route}" is blocked and cannot be run`);
        }
      }
      const jobItems = [];
      for (const srcId of sourceIds) {
        let inputPath = sourceMap.get(srcId) || (pathAllowed(srcId) ? srcId : null);
        if (!inputPath) {
          inputPath = join(WORK_DIR, 'uploads', `${srcId}.wav`);
        }
        const profile = stratType === 'smart_safe' ? 'production' : (['production', 'lowband'].includes(strategy.route) ? strategy.route : 'studio');
        const dir = inputPath.slice(0, inputPath.lastIndexOf('/')) || WORK_DIR;
        const stem = cleanStem(inputPath);
        const output_path = `${dir}/${stem}_${OUTPUT_SUFFIX[profile] || 'clean'}.wav`;
        const report_path = output_path.replace(/\.wav$/i, '') + '.hawavoclean.json';
        const job = {
          id: `j_${randomUUID().replace(/-/g, '').slice(0, 12)}`,
          state: 'queued',
          stage: 'preflight',
          progress: 0,
          message: 'Queued',
          unit: null,
          input_path: inputPath,
          output_path,
          report_path,
          profile,
          mode: stratType === 'smart_safe' ? 'smart_safe' : 'natural',
          speaker_id: null,
          cutoff_hz: null,
          started_at: null,
          finished_at: null,
          error: null,
          report: null,
          listeners: new Set(),
          version: 0,
          timer: null,
        };
        jobs.set(job.id, job);
        queue.push(job);
        jobItems.push({
          jobId: job.id,
          sourceId: srcId,
          outputPath: output_path,
          reportPath: report_path,
        });
      }
      schedule();
      return json(res, 202, { schemaVersion: 1, jobs: jobItems });
    }
    if (p === '/api/upload' && req.method === 'POST') {
      const buf = await readBody(req);
      const part = parseMultipart(buf, req.headers['content-type']);
      if (!part) return err(res, 400, 'bad_request', 'multipart field "file" missing');
      const sourceId = randomUUID().replace(/-/g, '');
      const dir = join(WORK_DIR, 'uploads', sourceId);
      await mkdir(dir, { recursive: true });
      const dest = join(dir, basename(part.filename));
      await writeFile(dest, part.body);
      sourceMap.set(sourceId, dest);
      return json(res, 200, { path: dest, source_id: sourceId });
    }
    if (p === '/api/shutdown' && req.method === 'POST') {
      json(res, 200, { ok: true });
      setTimeout(() => process.exit(0), 200);
      return;
    }
    return err(res, 404, 'not_found', `No route ${req.method} ${p}`);
  } catch (e) {
    return err(res, 500, 'internal', e instanceof Error ? e.message : String(e));
  }
});

await mkdir(join(WORK_DIR, 'uploads'), { recursive: true });
server.listen(PORT, HOST, () => {
  const addr = server.address();
  const port = typeof addr === 'object' && addr ? addr.port : PORT;
  process.stdout.write(JSON.stringify({ event: 'ready', port, pid: process.pid, version: VERSION }) + '\n');
  process.stderr.write(`mock engine on http://${HOST}:${port}  token=${TOKEN}${UI_DIR ? `  ui-dir=${UI_DIR}` : ''}\n`);
  process.stderr.write(`open: http://${HOST}:${port}/?token=${TOKEN}  (or vite dev at http://127.0.0.1:5173/?token=${TOKEN})\n`);
});
