// The two halves of one honesty fix, pinned.
//
// C5 · an 8-channel file used to analyse cleanly, arm PROCESS on a plain
// CHANNELS cell, and then fail with the engine's own words — "Multi-channel
// audio with 8 channels is not supported without explicit split_speakers
// declaration." `split_speakers` is a `channel_mode` value in the engine's
// config file; the web API's JobRequest carries no `channel_mode` field at all
// (input_path, profile, output_path, overwrite, and the restore trio mode /
// speaker_id / cutoff_hz), so there is no control on the screen, and no
// request this page could send, that would satisfy that sentence. It was also
// long enough to be cut mid-word in the status line and in the run list.
//
// This file pins both ends: the warning that arrives ten seconds before the
// refusal, and the refusal itself. The engine strings are the real ones, from
// `src/hawavoclean/audio/channels.py`.

import { describe, expect, it, vi } from 'vitest';
import { EngineError } from '../api/client';
import { classifyFailure, failureSource } from './errors';

vi.mock('../audio/player', () => ({ getPlayer: () => ({}) }));
vi.mock('../bridge', () => ({
  getBridge: () => ({
    host: 'web',
    engine: { getEndpoint: async () => ({ baseUrl: 'http://127.0.0.1:8765' }) },
  }),
}));

const { channelWarning, PIPELINE_MAX_CHANNELS } = await import('./actions');

const MULTI =
  'Multi-channel audio with 8 channels is not supported without explicit split_speakers declaration.';
const AMBIGUOUS =
  "Input stereo channels exhibit correlation=0.812 and level_ratio=1.01. Auto-classification returned 'ambiguous_stereo'. To prevent phase/spatial corruption, declare channel_mode in config explicitly: 'dual_mono_same' (channels carry the same signal) or 'split_speakers' (one speaker per channel).";

describe('channelWarning — the pre-flight on the CHANNELS cell', () => {
  it('says nothing about mono or stereo, which the pipeline takes', () => {
    expect(channelWarning(1)).toBeNull();
    expect(channelWarning(2)).toBeNull();
    expect(PIPELINE_MAX_CHANNELS).toBe(2);
  });

  it('flags a 7.1 file before PROCESS is ever pressed, and says what to do', () => {
    const w = channelWarning(8);
    expect(w).toContain('8 channels');
    expect(w).toMatch(/mono or stereo/);
    // The rate warning next to it makes the same promise in the same words:
    // the file reads, the *run* is what will be refused.
    expect(w).toMatch(/a run will be refused/);
  });

  it('never names a knob this UI does not have', () => {
    expect(channelWarning(8)).not.toMatch(/split_speakers|channel_mode/);
  });

  it('is not confused by a missing or nonsense channel count', () => {
    expect(channelWarning(Number.NaN)).toBeNull();
    expect(channelWarning(0)).toBeNull();
  });
});

describe('the refusal, when the run does happen', () => {
  const multi = classifyFailure(new EngineError(400, 'INVALID_USER_INPUT', MULTI), 'mix.wav');

  it('is a designed kind, not the generic branch that cuts at 89 characters', () => {
    expect(multi.kind).toBe('channels');
    expect(multi.headline).toBe('8-channel audio is more than this tool takes');
    // A headline that ends in an ellipsis is one that was cut mid-thought.
    expect(multi.headline).not.toMatch(/…$/);
  });

  it('says what the user can actually do, and names their file', () => {
    expect(multi.detail).toContain('“mix.wav”');
    expect(multi.detail).toContain('8 channels');
    expect(multi.detail).toMatch(/fold the file down to mono or stereo/);
  });

  it('keeps the engine knob out of every string the screen shows', () => {
    expect(multi.headline).not.toMatch(/split_speakers|channel_mode/);
    expect(multi.detail).not.toMatch(/split_speakers|channel_mode/);
    // …while the engine's own words stay available for a bug report.
    expect(multi.raw).toContain('split_speakers');
  });

  it('answers the ambiguous-stereo refusal the same way', () => {
    const f = classifyFailure(new EngineError(400, 'AMBIGUOUS_STEREO', AMBIGUOUS), 'music.wav');
    expect(f.kind).toBe('stereo-ambiguous');
    expect(f.detail).toMatch(/Fold it to mono/);
    expect(f.detail).not.toMatch(/dual_mono_same|channel_mode|ambiguous_stereo/);
  });

  it('gives the interrupted run a whole thought instead of a cut sentence', () => {
    const f = classifyFailure(
      new EngineError(
        400,
        'ENGINE_RESTARTED',
        'The engine stopped while this run was in flight, so it never finished. Nothing was written; press PROCESS to run it again.',
      ),
      'take.wav',
    );
    expect(f.kind).toBe('interrupted');
    expect(f.headline).toBe('The engine stopped mid-run');
    expect(f.detail).toMatch(/press PROCESS to run it again/);
  });
});

describe('failureSource — the error bar names the source, not "Engine"', () => {
  it('calls a refused clip a refused clip', () => {
    for (const message of [MULTI, AMBIGUOUS]) {
      expect(failureSource(classifyFailure(new EngineError(400, 'x', message)))).toBe(
        'Clip refused',
      );
    }
    expect(
      failureSource(classifyFailure(new EngineError(404, 'not_found', 'missing'), 'a.wav')),
    ).toBe('Clip refused');
  });

  it('keeps "Engine error" for the engine actually breaking', () => {
    expect(failureSource(classifyFailure(new EngineError(500, 'internal', 'boom')))).toBe(
      'Engine error',
    );
  });

  it('separates a dead engine from a broken one', () => {
    expect(failureSource(classifyFailure(new TypeError('Failed to fetch')))).toBe('Engine offline');
  });

  it('does not blame the engine for something this page threw', () => {
    expect(failureSource(classifyFailure(new Error('undefined is not a function')))).toBe(
      'App error',
    );
  });
});
