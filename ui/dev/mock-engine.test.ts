// @vitest-environment node
//
// dev/mock-engine.mjs against the shapes this client is typed for. The mock
// is the engine the UI is developed against, so the contract facts the real
// server pins (docs/ui-contract.md, addenda 1 and 2) must hold here too:
// the health capabilities, the /api/peaks window shape, the restore job's
// snapshot echo and the restoration section of its done report. Everything
// is asserted through the types in ./types — a drift in either the mock or
// the types fails here, at compile time or at runtime.
//
// One real HTTP server per file (a child process, OS-assigned port,
// --speed 40 so the staged job walk takes ~150 ms instead of ~6 s).

import { spawn, type ChildProcess } from 'node:child_process';
import { homedir } from 'node:os';
// Explicit import: the app tsconfig deliberately keeps node's globals out of
// scope (`types: ["vite/client"]`), and this one node-environment file should
// not be the reason browser code gets them.
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import type { HawaVoCleanReport, HealthResponse, JobStatus, PeaksWindow } from '../src/api/types.ts';

const SCRIPT = fileURLToPath(new URL('./mock-engine.mjs', import.meta.url));
const TOKEN = 'dev';
// A path under $HOME (the mock's path policy) that never touches disk: the
// mock synthesizes audio for any allowed path and writes report sidecars
// only inside its own work dir.
const CLIP = `${homedir()}/hawa-mock-contract-test.wav`;

let child: ChildProcess;
let base = '';

beforeAll(async () => {
  child = spawn(process.execPath, [SCRIPT, '--port', '0', '--speed', '40'], {
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  base = await new Promise<string>((resolve, reject) => {
    let out = '';
    child.stdout?.on('data', (d: unknown) => {
      out += String(d);
      const line = out.split('\n')[0];
      if (line) resolve(`http://127.0.0.1:${(JSON.parse(line) as { port: number }).port}`);
    });
    child.on('error', reject);
    child.on('exit', (code: number | null) =>
      reject(new Error(`mock engine exited early (${code})`)),
    );
  });
});

afterAll(() => {
  child.kill();
});

const H = { 'X-Hawa-Token': TOKEN, 'Content-Type': 'application/json' };

async function post(path: string, body: unknown): Promise<Response> {
  return fetch(`${base}${path}`, { method: 'POST', headers: H, body: JSON.stringify(body) });
}

async function submit(body: Record<string, unknown>): Promise<{ job_id: string }> {
  const res = await post('/api/jobs', { input_path: CLIP, profile: 'studio', ...body });
  expect(res.status).toBe(202);
  return (await res.json()) as { job_id: string };
}

/** Poll the snapshot until the job leaves the queue/run states. */
async function finished(jobId: string): Promise<JobStatus> {
  for (let i = 0; i < 200; i++) {
    const res = await fetch(`${base}/api/jobs/${jobId}`, { headers: H });
    const snap = (await res.json()) as JobStatus;
    if (snap.state !== 'queued' && snap.state !== 'running') return snap;
    await new Promise((r) => setTimeout(r, 25));
  }
  throw new Error(`job ${jobId} never finished`);
}

describe('capabilities (GET /api/health)', () => {
  it('offers speakers and the restore flag the UI keys on', async () => {
    const res = await fetch(`${base}/api/health`, { headers: H });
    expect(res.status).toBe(200);
    const health = (await res.json()) as HealthResponse;
    expect(health.ok).toBe(true);
    expect(health.speakers).toEqual(['character_01', 'character_02']);
    expect(health.restore_available).toBe(true);
  });
});

describe('POST /api/peaks (contract addendum 1)', () => {
  it('answers the PeaksWindow shape over the requested span', async () => {
    const res = await post('/api/peaks', { path: CLIP, start_s: 12, end_s: 18.5, buckets: 64 });
    expect(res.status).toBe(200);
    const win = (await res.json()) as PeaksWindow;
    expect(win.path).toBe(CLIP);
    expect(win.start_s).toBe(12);
    expect(win.end_s).toBe(18.5);
    expect(win.sample_rate).toBe(48000);
    expect(win.channels).toBe(1);
    expect(win.duration_s).toBe(60);
    expect(win.peaks.min).toHaveLength(64);
    expect(win.peaks.max).toHaveLength(64);
    expect(win.rms_db).toHaveLength(64);
    // 6.5 s at 48 kHz over 64 buckets: every bucket covers many samples.
    expect(win.samples_per_bucket).toBe(Math.ceil((6.5 * 48000) / 64));
    for (let i = 0; i < 64; i++) {
      expect(win.peaks.min[i]!).toBeLessThanOrEqual(win.peaks.max[i]!);
    }
  }, 20000);

  it('clamps buckets to the sample count so 1 sample/bucket is reachable', async () => {
    const res = await post('/api/peaks', {
      path: CLIP,
      start_s: 0,
      end_s: 32 / 48000,
      buckets: 8000,
    });
    const win = (await res.json()) as PeaksWindow;
    expect(win.peaks.min).toHaveLength(32);
    expect(win.samples_per_bucket).toBe(1);
  });

  it('a window past the end of the file is a 400, an unknown field a 422', async () => {
    const past = await post('/api/peaks', { path: CLIP, start_s: 61, end_s: 62 });
    expect(past.status).toBe(400);
    const typo = await post('/api/peaks', { path: CLIP, start_s: 0, end_s: 1, bucket: 100 });
    expect(typo.status).toBe(422);
  });
});

describe('restore jobs (contract addendum 2)', () => {
  it('refuses the combinations the real engine refuses, with the same codes', async () => {
    const noSpeaker = await post('/api/jobs', { input_path: CLIP, profile: 'studio', mode: 'restore' });
    expect(noSpeaker.status).toBe(422);
    const naturalSpeaker = await post('/api/jobs', {
      input_path: CLIP,
      profile: 'studio',
      speaker_id: 'character_01',
    });
    expect(naturalSpeaker.status).toBe(422);
    const naturalCutoff = await post('/api/jobs', { input_path: CLIP, profile: 'studio', cutoff_hz: 7800 });
    expect(naturalCutoff.status).toBe(422);
    const badId = await post('/api/jobs', {
      input_path: CLIP,
      profile: 'studio',
      mode: 'restore',
      speaker_id: 'Not-A-Speaker!',
    });
    expect(badId.status).toBe(422);
    const typo = await post('/api/jobs', { input_path: CLIP, profile: 'studio', speaker: 'x' });
    expect(typo.status).toBe(422);
  });

  it('a natural job stays revision-1 shaped: mode, no restore fields', async () => {
    const { job_id } = await submit({});
    const snap = await finished(job_id);
    expect(snap.state).toBe('done');
    expect(snap.mode).toBe('natural');
    expect('speaker_id' in snap).toBe(false);
    expect('cutoff_hz' in snap).toBe(false);
    expect(snap.report?.restoration).toBeUndefined();
  });

  it('a restore job echoes its fields and reports its restoration section', async () => {
    const { job_id } = await submit({ mode: 'restore', speaker_id: 'character_01', cutoff_hz: 8000 });
    const snap = await finished(job_id);
    expect(snap.state).toBe('done');
    expect(snap.mode).toBe('restore');
    expect(snap.speaker_id).toBe('character_01');
    expect(snap.cutoff_hz).toBe(8000);

    const report = snap.report as HawaVoCleanReport;
    expect(report.schema_version).toBe(2);
    expect(report.release?.version).toBeTypeOf('string');
    expect(report.build).toBeTruthy();
    expect(report.passes).toEqual([]);

    const rest = report.restoration;
    expect(rest?.mode).toBe('restore');
    expect(rest?.speaker_id).toBe('character_01');
    // A manual cutoff is recorded as asserted, at the asserted frequency.
    expect(rest?.bandwidth?.effective_cutoff_hz).toBe(8000);
    expect(rest?.bandwidth?.cutoff_mode).toBe('manual');
    expect(rest?.bandwidth?.evidence?.above_cutoff_snr_db).toBeTypeOf('number');
    expect(rest?.restorer?.weights_sha256).toMatch(/^[0-9a-f]{64}$/);
    expect(rest?.segments).toEqual({ restored: 1, reduced: 0, reverted: 0, bypassed: 0, errors: 0 });
    expect(rest?.guard_r?.verdict).toBe('PASS');
    expect(rest?.guard_r?.accepted_strength).toBe(0.7);
  });

  it('an auto-cutoff restore snapshot carries cutoff_hz null, not absent', async () => {
    const { job_id } = await submit({ mode: 'restore', speaker_id: 'character_02' });
    const snap = await finished(job_id);
    expect('cutoff_hz' in snap).toBe(true);
    expect(snap.cutoff_hz).toBeNull();
    expect(snap.report?.restoration?.bandwidth?.cutoff_mode).toBe('auto');
  });

  it('the "revert" dev hook exercises the honest-FAIL path end to end', async () => {
    const revertClip = `${homedir()}/hawa-mock-revert-test.wav`;
    const res = await post('/api/jobs', {
      input_path: revertClip,
      profile: 'studio',
      mode: 'restore',
      speaker_id: 'character_01',
    });
    const { job_id } = (await res.json()) as { job_id: string };
    const snap = await finished(job_id);
    const rest = snap.report?.restoration;
    // The job is done — a rejected restoration is a shipped Natural master,
    // not a failed run.
    expect(snap.state).toBe('done');
    expect(rest?.guard_r?.verdict).toBe('FAIL');
    expect(rest?.segments?.reverted).toBe(1);
    expect(rest?.guard_r?.reason).toContain('reverted to Natural-safe audio');
  });
});
