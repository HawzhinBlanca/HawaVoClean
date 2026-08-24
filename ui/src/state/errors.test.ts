// state/errors.ts — C5's designed-failure layer. Iteration 4 of the web
// perfection log claims four adversarial inputs stopped showing raw exceptions
// and started showing one plain sentence each. Those sentences shipped with no
// test: this file pins them, and pins the two reductions
// (`Command '[…]'` argv noise, absolute work-dir paths) that make them possible.
//
// The engine strings below are the real ones, copied from
// `src/hawavoclean/audio/probe.py` — a test built on invented wording would
// pass while the UI printed a stack trace.

import { describe, expect, it, vi } from 'vitest';
import { EngineError } from '../api/client';
import {
  classifyFailure,
  failureDetail,
  failureHeadline,
  installFailureNet,
  isCancellation,
  sanitizeEngineMessage,
} from './errors';

const WORK = '/Users/someone/Library/Application Support/hawavoclean/work/uploads/9f2c/take.wav';

describe('sanitizeEngineMessage', () => {
  it('drops the subprocess argv tail instead of ellipsising it', () => {
    const raw =
      `ffprobe failed to probe ${WORK}: Command '['/opt/homebrew/bin/ffprobe', ` +
      `'-v', 'error', '-show_streams', '-of', 'json', '${WORK}']' returned non-zero exit status 1.`;
    const out = sanitizeEngineMessage(raw);
    expect(out).not.toContain('Command');
    expect(out).not.toContain('ffprobe failed to probe /Users');
    expect(out.length).toBeLessThan(raw.length / 2);
  });

  it('collapses every absolute path to the name the user would recognise', () => {
    expect(
      sanitizeEngineMessage('No audio stream found in /var/work/uploads/9f2c/take.wav (1 stream(s))'),
    ).toBe('No audio stream found in "take.wav" (1 stream(s))');
    // The work directory must never survive into anything rendered.
    expect(sanitizeEngineMessage(`cannot open ${WORK}`)).not.toContain('/');
    expect(sanitizeEngineMessage(`cannot open ${WORK}`)).not.toContain('uploads');
  });

  // KNOWN LIMIT, pinned so a change to it is a decision and not a surprise: the
  // path pattern stops at whitespace, so a directory with a space in it
  // ("Application Support") is reduced in pieces. No path component escapes and
  // the basename still reads, but the sentence is not one clean thought. Only
  // the `unknown` fallback ever renders this string; every classified kind
  // writes its own sentence from the file name instead.
  it('a path containing spaces is still fully reduced, component by component', () => {
    const out = sanitizeEngineMessage(`cannot open ${WORK}`);
    expect(out).toContain('"take.wav"');
    expect(out).not.toContain('/');
    expect(out).not.toContain('hawavoclean/work');
  });

  it('never grows past the raw cap', () => {
    const out = sanitizeEngineMessage('x'.repeat(5000));
    expect(out.length).toBeLessThanOrEqual(600);
    expect(out.endsWith('…')).toBe(true);
  });
});

describe('C5 · the four inputs iteration 4 redesigned', () => {
  it('a corrupt container reads as one sentence, not 358 characters of argv', () => {
    const raw =
      `ffprobe failed to probe ${WORK}: Command '['/opt/homebrew/bin/ffprobe', '-v', 'error']' ` +
      `returned non-zero exit status 1.`;
    const f = classifyFailure(new EngineError(400, 'invalid_user_input', raw), 'random.wav');
    expect(f.kind).toBe('unreadable');
    expect(f.detail).toBe(
      '“random.wav” is not readable audio — the container is empty, truncated or corrupt. Re-export it, or try the original file.',
    );
    expect(f.headline).toBe('Cannot read “random.wav”');
    expect(f.detail).not.toContain('/Users');
    expect(f.detail).not.toContain('Command');
    expect(f.raw).toBe(raw); // the engine's own words survive for the title attribute
    expect(f.retryable).toBe(false);
  });

  it('a truncated container is the same class as a corrupt one', () => {
    for (const raw of [
      'Audio stream in /w/x/t.m4a has no decodable samples',
      'Invalid audio stream in /w/x/t.m4a: rate=0, channels=0',
      'neither ffprobe nor soundfile could read /w/x/t.m4a',
    ]) {
      expect(classifyFailure(new EngineError(400, 'invalid_user_input', raw), 't.m4a').kind).toBe(
        'unreadable',
      );
    }
  });

  it('a video with no audio track says there is nothing to clean', () => {
    const raw = `No audio stream found in ${WORK} (1 non-audio stream(s) present)`;
    const f = classifyFailure(new EngineError(400, 'invalid_user_input', raw), 'noaudio.mp4');
    expect(f.kind).toBe('no-audio');
    expect(f.detail).toBe(
      '“noaudio.mp4” carries no audio track — it is a video-only (or data-only) container. There is nothing here to clean.',
    );
  });

  it('192 kHz names the rate and the limit in kHz, not a bare error code', () => {
    const raw =
      'Input sample rate 192000 Hz exceeds maximum supported 48000 Hz. Ultrasonic rates are rejected in V1.';
    const f = classifyFailure(new EngineError(400, 'invalid_user_input', raw), 'hi192k.wav');
    expect(f.kind).toBe('rate-high');
    expect(f.headline).toBe('192 kHz is above this tool’s range');
    expect(f.detail).toBe(
      '“hi192k.wav” is 192 kHz. This tool works up to 48 kHz — resample it down and load it again.',
    );
    // The bare `INVALID_USER_INPUT:` the log records as the old behaviour.
    expect(f.detail).not.toMatch(/INVALID_USER_INPUT/i);
  });

  it('a non-round rate keeps one decimal rather than lying', () => {
    const f = classifyFailure(
      new EngineError(400, 'invalid_user_input', 'Input sample rate 7350 Hz is below the minimum supported 8000 Hz.'),
      'tiny.wav',
    );
    expect(f.kind).toBe('rate-low');
    expect(f.detail).toContain('7.3 kHz'); // 7350 Hz, one decimal
    expect(f.detail).toContain('from 8 kHz upwards');
  });
});

describe('the file the sentence is about', () => {
  it('uses the name the user dropped, never the upload path', () => {
    const f = classifyFailure(new EngineError(404, 'not_found', `path not found: ${WORK}`), 'my take.wav');
    expect(f.detail.startsWith('“my take.wav”')).toBe(true);
    expect(f.detail).not.toContain('uploads');
  });

  it('falls back to the basename the engine mentioned, not the whole path', () => {
    const f = classifyFailure(new EngineError(404, 'not_found', `path not found: ${WORK}`));
    expect(f.detail.startsWith('“take.wav”')).toBe(true);
    expect(f.detail).not.toContain('/');
  });

  it('says "This file" when there is no name anywhere', () => {
    const f = classifyFailure(new EngineError(403, 'forbidden', 'outside allowed roots'));
    expect(f.detail.startsWith('This file')).toBe(true);
  });
});

describe('the kinds that change what the UI may do', () => {
  it('an abort is a cancellation and is never an error bar', () => {
    const f = classifyFailure(new DOMException('Aborted', 'AbortError'));
    expect(f.kind).toBe('cancelled');
    expect(isCancellation(f)).toBe(true);
  });

  it('a dead socket is offline and retryable, whichever shape it arrives in', () => {
    for (const e of [new TypeError('Failed to fetch'), new EngineError(0, 'engine_offline', 'refused')]) {
      const f = classifyFailure(e);
      expect(f.kind).toBe('offline');
      expect(f.retryable).toBe(true);
      expect(isCancellation(f)).toBe(false);
    }
  });

  it('401 tells the user to reload rather than to retry', () => {
    const f = classifyFailure(new EngineError(401, 'unauthorized', 'bad token'));
    expect(f.kind).toBe('unauthorized');
    expect(f.retryable).toBe(false);
    expect(f.detail).toContain('Reload the page');
  });

  it('a 5xx is the engine’s fault and says so', () => {
    const f = classifyFailure(new EngineError(500, 'internal', 'boom'), 'a.wav');
    expect(f.kind).toBe('engine-fault');
    expect(f.retryable).toBe(true);
    expect(f.detail).toContain('not your file');
  });

  it('413 points at the on-disk path instead of the upload', () => {
    const f = classifyFailure(new EngineError(413, 'too_large', 'body too large'), 'big.wav');
    expect(f.kind).toBe('too-large');
    expect(f.detail).toContain('on disk');
  });

  it('an unclassified refusal still arrives cleaned, never as a raw path', () => {
    const f = classifyFailure(
      new EngineError(422, 'invalid_user_input', `weird refusal about ${WORK}: Command '['x']' failed`),
      'a.wav',
    );
    expect(f.kind).toBe('unknown');
    expect(f.detail).not.toContain('/Users');
    expect(f.detail).not.toContain('Command');
  });

  it('headline and detail helpers agree with the classification', () => {
    const e = new EngineError(401, 'unauthorized', 'bad token');
    expect(failureHeadline(e)).toBe(classifyFailure(e).headline);
    expect(failureDetail(e)).toBe(classifyFailure(e).detail);
  });
});

describe('installFailureNet', () => {
  it('turns an unhandled rejection into the same designed failure, and swallows aborts', () => {
    const seen: string[] = [];
    installFailureNet((f) => seen.push(f.kind));

    const rejection = new Event('unhandledrejection') as Event & {
      reason: unknown;
      preventDefault: () => void;
    };
    rejection.reason = new EngineError(401, 'unauthorized', 'bad token');
    rejection.preventDefault = vi.fn();
    window.dispatchEvent(rejection);
    expect(seen).toEqual(['unauthorized']);

    const aborted = new Event('unhandledrejection') as Event & {
      reason: unknown;
      preventDefault: () => void;
    };
    aborted.reason = new DOMException('Aborted', 'AbortError');
    const prevented = vi.fn();
    aborted.preventDefault = prevented;
    window.dispatchEvent(aborted);
    expect(seen).toEqual(['unauthorized']); // no second report
    expect(prevented).toHaveBeenCalled();
  });
});

describe('an API route is not a work-directory path', () => {
  // The sanitizer collapses absolute paths to a basename because in web mode
  // they point at the upload work directory the user never sees. It did the
  // same to the routes the engine deliberately sends the reader to, so
  // "see /api/health" reached the UI as 'see "health"' and
  // 'no such endpoint: /api/x' as 'no such endpoint: "x"' — deleting the only
  // actionable part of the sentence.
  it.each([
    ['mode "restore" requires speaker_id (see /api/health)', '/api/health'],
    ["unknown speaker_id 'character_99' (see /api/health for installed speakers)", '/api/health'],
    ['no such endpoint: /api/nope', '/api/nope'],
  ])('keeps %s intact', (message, route) => {
    expect(sanitizeEngineMessage(message)).toContain(route);
  });

  it('still collapses a real work-directory path', () => {
    const out = sanitizeEngineMessage('cannot read /Users/x/.cache/hawavoclean/work/ab12/take.wav');
    expect(out).not.toContain('/Users/x/.cache');
    expect(out).toContain('take.wav');
  });
});
