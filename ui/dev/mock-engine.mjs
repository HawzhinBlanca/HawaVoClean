#!/usr/bin/env node
// Mock HawaVoClean engine for UI development (Node 22, no dependencies).
// Implements docs/ui-contract.md §1 with synthetic data: a fake 60 s clip,
// real peaks/spectrum computed from the synthesized signal, a job that walks
// through the stages over ~6 s and returns a plausible HawaVoCleanReport,
// and /api/audio serving a generated WAV with Range support.
//
//   node dev/mock-engine.mjs [--port 8765] [--token dev] [--ui-dir dist]

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
const WORK_DIR = join(tmpdir(), 'hawavoclean-mock');
const VERSION = '3.2.0';

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

function isCleanedPath(p) {
  const b = basename(p).toLowerCase();
  return b.includes('_studio') || b.includes('_clean');
}

/** Synthesize a speech-like mono signal. Cleaned variant: lower noise floor. */
function synthesize(path) {
  const cleaned = isCleanedPath(path);
  const stemSeed = seedFrom(basename(path).replace(/_(studio|clean)\.wav$/i, ''));
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
    schema_version: 1, job_id: job.id, config_hash: createHash('sha256').update('mock-config').digest('hex'),
    input: media(job.input_path, -23.4, -3.1), output: media(job.output_path, -19.0, -1.0),
    core: job.profile === 'studio'
      ? { id: 'studio-dfn3-48k-v1', algorithm: 'WPE + DeepFilterNet3', params_hash: 'b1f7' + '0'.repeat(60), phase_coherent: true }
      : { id: 'wiener-dd-48k-v1', algorithm: 'Wiener decision-directed', params_hash: 'a9c2' + '0'.repeat(60), phase_coherent: true },
    guard: { id: 'spectral-guard-v1', probe_hash: 'c3d4' + '0'.repeat(60), calibration_id: job.profile === 'studio' ? 'guard-calibration-studio' : 'guard-calibration' },
    environment: { platform: 'mock', os_version: process.version, python_version: 'n/a', numpy_version: 'n/a', scipy_version: 'n/a', soundfile_version: 'n/a', cpu_model: null },
    summary: { units_total: units.length, enhanced, reverted, unverified: 0, error_passthrough: 0, continuity_reverted: 0, no_speech: noSpeech, finish_applied: enhanced, finish_bypassed: units.length - enhanced },
    review_timecodes: units.filter((u) => u.final_decision === 'original_reverted').map((u) => ({ unit_id: u.unit_id, start_time_s: u.start_time_s, end_time_s: u.end_time_s, channel: 0, verdict: 'REVERT', reason: u.decision_reason })),
    units,
  };
}

function publicStatus(job) {
  const o = {
    job_id: job.id, state: job.state, stage: job.stage, progress: job.progress, message: job.message,
    unit: job.unit, input_path: job.input_path, output_path: job.output_path, report_path: job.report_path,
    profile: job.profile, started_at: job.started_at, finished_at: job.finished_at,
  };
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
    const [delay, stage, progress, message, unit] = steps[idx++];
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
      return json(res, 200, { ok: true, version: VERSION, profiles: ['studio', 'production'], engine_pid: process.pid });
    }
    if (p === '/api/analyze' && req.method === 'POST') {
      const body = JSON.parse((await readBody(req)).toString() || '{}');
      if (!pathAllowed(body.path)) return err(res, 403, 'forbidden', 'Path outside allowed roots');
      const buckets = Math.max(16, Math.min(8000, Number(body.buckets) || 1200));
      await new Promise((r) => setTimeout(r, 350)); // feel like real decode
      return json(res, 200, analyze(body.path, buckets));
    }
    if (p === '/api/jobs' && req.method === 'POST') {
      const body = JSON.parse((await readBody(req)).toString() || '{}');
      if (!pathAllowed(body.input_path)) return err(res, 403, 'forbidden', 'Path outside allowed roots');
      if (body.output_path !== undefined && !pathAllowed(body.output_path)) {
        return err(res, 403, 'forbidden', 'Output path outside allowed roots');
      }
      const profile = body.profile === 'production' ? 'production' : 'studio';
      const dir = body.input_path.slice(0, body.input_path.lastIndexOf('/'));
      const stem = cleanStem(body.input_path);
      const output_path = body.output_path || `${dir}/${stem}_${profile === 'studio' ? 'studio' : 'clean'}.wav`;
      const report_path = output_path.replace(/\.wav$/i, '') + '.hawavoclean.json';
      const job = {
        id: `j_${randomUUID().replace(/-/g, '').slice(0, 12)}`, state: 'queued', stage: 'preflight', progress: 0,
        message: 'Queued', unit: null, input_path: body.input_path, output_path, report_path, profile,
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
    if (p === '/api/upload' && req.method === 'POST') {
      const buf = await readBody(req);
      const part = parseMultipart(buf, req.headers['content-type']);
      if (!part) return err(res, 400, 'bad_request', 'multipart field "file" missing');
      const dir = join(WORK_DIR, 'uploads', randomUUID());
      await mkdir(dir, { recursive: true });
      const dest = join(dir, basename(part.filename));
      await writeFile(dest, part.body);
      return json(res, 200, { path: dest });
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
