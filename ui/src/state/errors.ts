// C5 · one place that turns any failure into a *designed* state.
//
// The engine's contract is honest — every refusal is `{error, message}` with a
// correct status — but a `message` is written for whoever reads the log, not
// for whoever dropped the file. `ffprobe failed to probe
// /Users/…/work/uploads/<uuid>/take.wav: Command '['/opt/homebrew/bin/ffprobe',
// '-v', 'error', …]' returned non-zero exit status 1.` is 358 characters of
// subprocess repr; put in the error bar it ellipsises mid-path and tells the
// user nothing. Worse, it names a work-directory path they have never seen
// instead of the file they dropped.
//
// So: nothing in this UI renders an exception. Every failure is classified
// here into a kind, given a headline for the status line and one plain
// sentence for the error bar, and only then shown. The engine's own words are
// kept verbatim in `raw` so they can ride in a `title` attribute and be copied
// into a bug report — legible when wanted, never in the way.

import { EngineError } from '../api/client';

export type FailureKind =
  | 'cancelled'
  | 'offline'
  | 'unauthorized'
  | 'missing'
  | 'forbidden'
  | 'unreadable'
  | 'no-audio'
  | 'rate-high'
  | 'rate-low'
  | 'channels'
  | 'stereo-ambiguous'
  | 'interrupted'
  | 'too-large'
  | 'bad-response'
  | 'engine-fault'
  | 'unknown';

export interface UiFailure {
  kind: FailureKind;
  /** Machine code (the engine's `error` field where there is one). */
  code: string;
  /** Short form for the one-line status bar. */
  headline: string;
  /** One plain sentence for the error bar: what happened and what to do. */
  detail: string;
  /** The engine's own words, unedited, for a `title` attribute. */
  raw: string;
  /** Whether pressing the same button again could plausibly work. */
  retryable: boolean;
}

/** A cancellation is not a failure and must never reach the error bar. */
export function isCancellation(f: UiFailure): boolean {
  return f.kind === 'cancelled';
}

const MAX_RAW = 600;

/**
 * The engine's message, made safe to show when we have nothing better.
 *
 * Two things make an engine message unfit for a UI: the `Command '[…]'` tail
 * that `subprocess.CalledProcessError` appends (pure noise — an argv the user
 * never typed), and absolute paths, which in web mode point at the upload
 * work directory rather than anywhere the user recognises. Both are reduced,
 * never simply truncated, so whatever survives is a whole thought.
 */
export function sanitizeEngineMessage(message: string): string {
  let out = message.replace(/:?\s*Command\s+'\[[\s\S]*$/, '');
  // Any absolute POSIX path collapses to its last component. Paths are taken
  // up to whitespace or a quote; a trailing ':' or ',' is punctuation, not name.
  out = out.replace(/\/(?:[^\s'"<>|]+\/)*[^\s'"<>|]*/g, (p) => {
    // An API route is not a work-directory path: it is where the message is
    // sending the reader. Collapsing it turned "see /api/health" into
    // 'see "health"' and 'no such endpoint: /api/x' into 'no such endpoint:
    // "x"', deleting the only actionable part of the sentence.
    if (p.startsWith('/api/')) return p;
    const trimmed = p.replace(/[.,;:]+$/, '');
    const base = trimmed.slice(trimmed.lastIndexOf('/') + 1);
    return base ? `"${base}"` : p;
  });
  out = out.trim().replace(/\s+/g, ' ');
  if (out.length > MAX_RAW) out = `${out.slice(0, MAX_RAW - 1)}…`;
  return out;
}

function quoted(name: string | undefined): string {
  const n = (name ?? '').trim();
  return n ? `“${n}”` : 'This file';
}

/** The label the sentence should use: the name the user knows, if we have it. */
function subject(name: string | undefined, message: string): string {
  if (name && name.trim()) return quoted(name);
  // Fall back to the basename the engine mentioned, never the whole path.
  const m = /\/(?:[^\s'"<>|]+\/)*([^\s'"<>|/]+\.[A-Za-z0-9]{1,8})/.exec(message);
  return m?.[1] ? quoted(m[1]) : 'This file';
}

function rate(message: string, which: 'max' | 'min'): { got: string; limit: string } | null {
  const re =
    which === 'max'
      ? /sample rate (\d+) Hz exceeds maximum supported (\d+) Hz/i
      : /sample rate (\d+) Hz is below the minimum supported (\d+) Hz/i;
  const m = re.exec(message);
  if (!m?.[1] || !m[2]) return null;
  const khz = (v: string): string => {
    const n = Number(v);
    return Number.isFinite(n) ? `${(n / 1000).toFixed(n % 1000 === 0 ? 0 : 1)} kHz` : `${v} Hz`;
  };
  return { got: khz(m[1]), limit: khz(m[2]) };
}

/**
 * Classify anything that can be thrown at this UI.
 *
 * `name` is the name the user knows the file by — the one they dropped, not
 * the path it was uploaded to. Pass it wherever there is one; the sentence
 * reads much better and never leaks the work directory.
 */
export function classifyFailure(e: unknown, name?: string): UiFailure {
  if (e instanceof DOMException && e.name === 'AbortError') {
    return {
      kind: 'cancelled',
      code: 'cancelled',
      headline: 'Cancelled',
      detail: 'Cancelled.',
      raw: e.message,
      retryable: true,
    };
  }

  // A dead socket surfaces as a TypeError from `fetch`, with no status. A
  // socket that opened and then went silent surfaces as a DOMException named
  // 'TimeoutError' — that is what `AbortSignal.timeout` rejects with, and it
  // is deliberately NOT 'AbortError', which above means the *user* cancelled.
  // Both mean the same thing to the reader: the engine is not answering.
  if (
    e instanceof TypeError ||
    (e instanceof DOMException && e.name === 'TimeoutError') ||
    (e instanceof EngineError && e.status === 0)
  ) {
    const raw = e instanceof Error ? e.message : String(e);
    return {
      kind: 'offline',
      code: 'engine_offline',
      headline: 'Engine unreachable',
      detail:
        'The engine is not answering. Nothing already on screen is lost — this comes back on its own when it reconnects.',
      raw,
      retryable: true,
    };
  }

  if (e instanceof EngineError) {
    const raw = e.message;
    const who = subject(name, raw);
    const clean = sanitizeEngineMessage(raw);

    if (e.status === 401) {
      return {
        kind: 'unauthorized',
        code: e.code,
        headline: 'Session token rejected',
        detail:
          'The engine refused this page’s token. Reload the page to pick up the current one.',
        raw,
        retryable: false,
      };
    }
    if (e.status === 404 && e.code === 'not_found') {
      return {
        kind: 'missing',
        code: e.code,
        headline: `${who} is not there`,
        // Covers both halves of the same 404: a clip that vanished between
        // being analysed and being processed, and a path that was never right.
        detail: `${who} is not at that path — it has been moved, renamed or deleted, or the path was wrong to begin with. Load it again from where it is now.`,
        raw,
        retryable: false,
      };
    }
    if (e.status === 403) {
      return {
        kind: 'forbidden',
        code: e.code,
        headline: `${who} is out of bounds`,
        detail: `${who} is outside the places this tool may read — your home folder, /Volumes and its own work folder. Copy it into one of those and load it again.`,
        raw,
        retryable: false,
      };
    }
    if (e.status === 413) {
      return {
        kind: 'too-large',
        code: e.code,
        headline: `${who} is too large`,
        detail: `${who} is larger than this engine will accept in one upload. Point the tool at the file on disk instead, or raise the engine’s upload limit.`,
        raw,
        retryable: false,
      };
    }

    const hi = rate(raw, 'max');
    if (hi) {
      return {
        kind: 'rate-high',
        code: e.code,
        headline: `${hi.got} is above this tool’s range`,
        detail: `${who} is ${hi.got}. This tool works up to ${hi.limit} — resample it down and load it again.`,
        raw,
        retryable: false,
      };
    }
    const lo = rate(raw, 'min');
    if (lo) {
      return {
        kind: 'rate-low',
        code: e.code,
        headline: `${lo.got} is below this tool’s range`,
        detail: `${who} is ${lo.got}. This tool works from ${lo.limit} upwards.`,
        raw,
        retryable: false,
      };
    }
    /*
     * C5 · the layouts this pipeline cannot take, said without naming a knob
     * that does not exist here.
     *
     * The engine refuses an 8-channel file with "Multi-channel audio with 8
     * channels is not supported without explicit split_speakers declaration."
     * `split_speakers` is a `channel_mode` value in the engine's config file;
     * the web API's JobRequest carries no `channel_mode` field at all — its
     * fields are input_path, profile, output_path, overwrite and the restore
     * trio (mode, speaker_id, cutoff_hz) — so there is no control on this
     * screen, and no request this page could send, that would satisfy it. Printing it put
     * the one word that sounds like the answer in front of the user and then
     * offered no way to type it, and it was long enough to be cut mid-word in
     * both the status line and the run list.
     *
     * What is actually true: this tool cleans one voice at a time, and it will
     * not guess which of eight channels carry it. That is a fold-down, and a
     * fold-down is something the user can do.
     */
    /*
     * The run whose engine died under it. The status is synthesised in
     * actions.ts with a sentence already written for a person, so the detail
     * is that sentence — but with no `kind` of its own it fell through to the
     * generic branch, which builds a headline by cutting the detail at 89
     * characters. Both the status line and the run list therefore ended on
     * "Nothing was writte…". A designed kind gives them a whole thought.
     */
    if (e.code === 'ENGINE_RESTARTED') {
      return {
        kind: 'interrupted',
        code: e.code,
        headline: 'The engine stopped mid-run',
        detail: clean,
        raw,
        retryable: true,
      };
    }

    const multi = /Multi-channel audio with (\d+) channels/i.exec(raw);
    if (multi?.[1]) {
      const n = multi[1];
      return {
        kind: 'channels',
        code: e.code,
        headline: `${n}-channel audio is more than this tool takes`,
        detail: `${who} carries ${n} channels. This tool cleans one voice at a time — fold the file down to mono or stereo and load that.`,
        raw,
        retryable: false,
      };
    }
    if (/ambiguous_stereo/i.test(raw)) {
      return {
        kind: 'stereo-ambiguous',
        code: e.code,
        headline: `${who}’s two channels are not one voice each`,
        detail: `${who} is stereo, but its channels are neither the same signal nor one voice each — cleaning that as speech would smear the stereo image. Fold it to mono and load that.`,
        raw,
        retryable: false,
      };
    }
    if (/no audio stream found/i.test(raw)) {
      return {
        kind: 'no-audio',
        code: e.code,
        headline: `${who} has no audio`,
        detail: `${who} carries no audio track — it is a video-only (or data-only) container. There is nothing here to clean.`,
        raw,
        retryable: false,
      };
    }
    if (
      /ffprobe failed to probe|neither ffprobe nor soundfile|no decodable samples|invalid audio stream|could not (?:be )?decode/i.test(
        raw,
      )
    ) {
      return {
        kind: 'unreadable',
        code: e.code,
        headline: `Cannot read ${who}`,
        detail: `${who} is not readable audio — the container is empty, truncated or corrupt. Re-export it, or try the original file.`,
        raw,
        retryable: false,
      };
    }
    if (e.status >= 500) {
      return {
        kind: 'engine-fault',
        code: e.code,
        headline: 'The engine hit an internal error',
        detail: `The engine failed while working on ${who}. This one is a bug, not your file — the engine log has the detail.`,
        raw,
        retryable: true,
      };
    }
    if (e.code === 'bad_response') {
      return {
        kind: 'bad-response',
        code: e.code,
        headline: 'Unreadable reply from the engine',
        detail:
          'The engine answered with something this page could not read. It may be a different version than this UI expects.',
        raw,
        retryable: true,
      };
    }
    // Anything else the engine refuses: its own words, cleaned of argv noise
    // and of paths, are still the most accurate thing we can say.
    return {
      kind: 'unknown',
      code: e.code,
      headline: clean.length > 90 ? `${clean.slice(0, 89)}…` : clean,
      detail: clean,
      raw,
      retryable: e.status >= 500,
    };
  }

  const raw = e instanceof Error ? e.message : String(e);
  const clean = sanitizeEngineMessage(raw) || 'Something went wrong.';
  return {
    kind: 'unknown',
    code: 'unexpected',
    headline: clean.length > 90 ? `${clean.slice(0, 89)}…` : clean,
    detail: clean,
    raw,
    retryable: false,
  };
}

/**
 * Where a failure came from, in two or three words — the error bar's label.
 *
 * The bar used to print `Engine error` over every sentence it was ever given,
 * which was true for most of them and a lie for the rest: a clipboard
 * permission the *browser* refused was published to the user as ENGINE ERROR,
 * and so was a clip the engine read perfectly well and simply would not take.
 * The classification already knows which is which; this is that knowledge in
 * label form. Callers that do not go through `classifyFailure` pass their own.
 */
export function failureSource(f: UiFailure): string {
  switch (f.kind) {
    case 'offline':
      return 'Engine offline';
    case 'unauthorized':
      return 'Access refused';
    case 'missing':
    case 'forbidden':
    case 'unreadable':
    case 'no-audio':
    case 'rate-high':
    case 'rate-low':
    case 'channels':
    case 'stereo-ambiguous':
    case 'too-large':
      // The engine is healthy and said no to this file. The clip is the news.
      return 'Clip refused';
    case 'interrupted':
      // Not a refusal and not a fault: the process that owned the run left.
      return 'Engine gone';
    case 'engine-fault':
      return 'Engine error';
    case 'bad-response':
      return 'Version mismatch';
    case 'cancelled':
      return 'Cancelled';
    default:
      // 'unknown' with an engine code came off the wire; without one it was
      // thrown in this page and calling it an engine error would be guessing.
      return f.code === 'unexpected' ? 'App error' : 'Engine refused';
  }
}

/** The one line the error bar shows. */
export function failureDetail(e: unknown, name?: string): string {
  return classifyFailure(e, name).detail;
}

/** The short form for the status line. */
export function failureHeadline(e: unknown, name?: string): string {
  return classifyFailure(e, name).headline;
}

// ---------------------------------------------------------------------------
// The net under everything else

let netInstalled = false;

/**
 * C5 · nothing escapes silently.
 *
 * Every async flow in this app catches its own failure, and that is where the
 * designed states come from. This is the net *under* that: a promise nobody
 * awaited, or an error thrown out of a React event handler, would otherwise be
 * one grey line in a console the user never opens — the "silent nothing" C5
 * forbids. Here it becomes the same designed error bar as everything else.
 *
 * Cancellations are excluded on purpose: an aborted fetch is the normal way
 * this UI abandons work, and it is not something to tell anyone about.
 */
export function installFailureNet(report: (f: UiFailure) => void): void {
  if (netInstalled) return;
  netInstalled = true;
  window.addEventListener('unhandledrejection', (ev: PromiseRejectionEvent) => {
    const f = classifyFailure(ev.reason);
    if (isCancellation(f)) {
      // A rejected abort is expected traffic; keep it out of the console too.
      ev.preventDefault();
      return;
    }
    report(f);
    // The console line is left in place deliberately: the user gets the
    // designed bar, and whoever is debugging still gets the stack.
  });
  window.addEventListener('error', (ev: ErrorEvent) => {
    // Resource load errors (`<img>`, `<audio>`) arrive here with no `error`
    // object and are handled by the elements themselves.
    if (!ev.error) return;
    const f = classifyFailure(ev.error);
    if (isCancellation(f)) return;
    report(f);
  });
}
