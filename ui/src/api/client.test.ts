// api/client.ts — every call the UI makes to the engine. The things that go
// wrong here are invisible in a happy browser run: a token on the wrong
// carrier, an error body that parses to the wrong sentence, an abort that
// leaks, an upload whose progress never reaches the bar.

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { EngineClient, EngineError } from './client';

const BASE = 'http://127.0.0.1:8765';

function client(baseUrl = `${BASE}/`): EngineClient {
  return new EngineClient({ baseUrl });
}

/** A Response stand-in: only the four members the client actually touches. */
function res(status: number, body: string, statusText = ''): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText,
    text: async () => body,
    json: async () => JSON.parse(body) as unknown,
  } as unknown as Response;
}

function mockFetch(...responses: Response[]): ReturnType<typeof vi.fn> {
  const fn = vi.fn();
  for (const r of responses) fn.mockResolvedValueOnce(r);
  vi.stubGlobal('fetch', fn);
  return fn;
}

function lastInit(fn: ReturnType<typeof vi.fn>): RequestInit {
  return (fn.mock.calls[fn.mock.calls.length - 1]?.[1] ?? {}) as RequestInit;
}

function lastHeaders(fn: ReturnType<typeof vi.fn>): Record<string, string> {
  return (lastInit(fn).headers ?? {}) as Record<string, string>;
}

describe('endpoint handling', () => {
  it('strips trailing slashes so no URL ever doubles up', () => {
    expect(new EngineClient({ baseUrl: `${BASE}///` }).baseUrl).toBe(BASE);
    expect(client().eventsUrl('j1')).toBe(`${BASE}/api/jobs/j1/events`);
  });
});

describe('renderer requests contain no authentication secret', () => {
  it('leaves authentication to the shell and sends only JSON content type', async () => {
    const fetchFn = mockFetch(res(200, '{"path":"/a.wav"}'));
    await client().analyze('/a.wav', 1200);
    expect(fetchFn).toHaveBeenCalledWith(`${BASE}/api/analyze`, expect.anything());
    expect(lastHeaders(fetchFn)['X-Hawa-Token']).toBeUndefined();
    expect(lastHeaders(fetchFn).Authorization).toBeUndefined();
    expect(lastHeaders(fetchFn)['Content-Type']).toBe('application/json');
    expect(lastInit(fetchFn).credentials).toBe('same-origin');
    expect(JSON.parse(String(lastInit(fetchFn).body))).toEqual({ path: '/a.wav', buckets: 1200 });
  });

  it('sends no content type on a GET, so no preflight is provoked', async () => {
    const fetchFn = mockFetch(res(200, '{"ok":true}'));
    await client().health();
    expect(lastHeaders(fetchFn)['X-Hawa-Token']).toBeUndefined();
    expect(lastHeaders(fetchFn).Authorization).toBeUndefined();
    expect(lastHeaders(fetchFn)['Content-Type']).toBeUndefined();
  });

  it('leaves multipart alone so the browser can set the boundary', async () => {
    const fetchFn = mockFetch(res(200, '{"path":"/up/a.wav"}'));
    await client().upload(new File(['abc'], 'a.wav', { type: 'audio/wav' }));
    expect(lastHeaders(fetchFn)['Content-Type']).toBeUndefined();
    expect(lastInit(fetchFn).body).toBeInstanceOf(FormData);
  });
});

describe('media and SSE URLs are credential-free', () => {
  it('puts only the path in the query for <audio src> and download anchors', () => {
    const url = new URL(client().fileUrl('/Users/me/My Clip #1.wav'));
    expect(url.origin + url.pathname).toBe(`${BASE}/api/audio`);
    expect(url.searchParams.get('path')).toBe('/Users/me/My Clip #1.wav');
    expect([...url.searchParams.keys()]).toEqual(['path']);
  });

  it('encodes the job id in the SSE path and has no query', () => {
    const url = new URL(client().eventsUrl('job/with space'));
    expect(url.pathname).toBe('/api/jobs/job%2Fwith%20space/events');
    expect(url.search).toBe('');
  });
});

describe('error shape parsing', () => {
  it('lifts the engine’s {error, message} pair into code and message', async () => {
    mockFetch(res(400, '{"error":"INVALID_USER_INPUT","message":"sample rate 192000 Hz"}'));
    const err = await client().analyze('/a.wav').catch((e: unknown) => e);
    expect(err).toBeInstanceOf(EngineError);
    expect((err as EngineError).status).toBe(400);
    expect((err as EngineError).code).toBe('INVALID_USER_INPUT');
    expect((err as EngineError).message).toBe('sample rate 192000 Hz');
  });

  it('uses the error code as the message when there is no message', async () => {
    mockFetch(res(403, '{"error":"forbidden_path"}'));
    const err = (await client().analyze('/a.wav').catch((e: unknown) => e)) as EngineError;
    expect(err.code).toBe('forbidden_path');
    expect(err.message).toBe('forbidden_path');
  });

  it('falls back to the status line when the error body is not JSON', async () => {
    mockFetch(res(502, '<html>bad gateway</html>', 'Bad Gateway'));
    const err = (await client().health().catch((e: unknown) => e)) as EngineError;
    expect(err.code).toBe('http_502');
    expect(err.message).toBe('502 Bad Gateway');
  });

  it('names a 404 so the SSE client can tell "gone" from "unreachable"', async () => {
    mockFetch(res(404, '{"error":"not_found","message":"no such job"}'));
    const err = (await client().getJob('j1').catch((e: unknown) => e)) as EngineError;
    expect(err.status).toBe(404);
  });

  it('turns an unparseable 200 body into a designed error, not a SyntaxError', async () => {
    mockFetch(res(200, '{"duration_s": 12'));
    const err = (await client().analyze('/a.wav').catch((e: unknown) => e)) as EngineError;
    expect(err).toBeInstanceOf(EngineError);
    expect(err.code).toBe('bad_response');
    expect(err.status).toBe(200);
    expect(err.message).toContain('17 bytes');
  });

  it('turns a transport failure into an EngineError with status 0', async () => {
    // It used to re-raise the raw TypeError. That put the burden of
    // recognising "the socket is dead" on the consumer, whose only handle is
    // `e instanceof TypeError` — which also matches every null-dereference in
    // a React handler that reaches the app's failure net, so a genuine app bug
    // was published as "The engine is not answering… this comes back on its
    // own when it reconnects" while the header lamp said READY.
    //
    // status 0 is the shape errors.ts and probeSoon already understand, and it
    // is what the XHR upload path in this file has always produced.
    const boom = new TypeError('Failed to fetch');
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.reject(boom)),
    );
    const err = (await client().health().catch((e: unknown) => e)) as EngineError;
    expect(err).toBeInstanceOf(EngineError);
    expect(err.status).toBe(0);
    expect(err.code).toBe('engine_offline');
    // The original wording survives, because "Failed to fetch" vs "network
    // error" vs "Load failed" is the only clue to which layer gave up.
    expect(err.message).toBe('Failed to fetch');
  });

  it('leaves an abort alone — that is the user, not the network', async () => {
    const abort = new DOMException('aborted', 'AbortError');
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.reject(abort)),
    );
    await expect(client().health()).rejects.toBe(abort);
  });

  it('leaves a timeout alone, and it is not an abort', async () => {
    // `AbortSignal.timeout` and the app's own timeoutSignal reject with
    // TimeoutError. Wrapping it as engine_offline would be defensible, but it
    // must not become AbortError, which this app reads as "the user cancelled".
    const t = new DOMException('No answer in 4000 ms', 'TimeoutError');
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.reject(t)),
    );
    await expect(client().health()).rejects.toBe(t);
  });
});

describe('abort plumbing', () => {
  it('forwards the caller’s signal on every abortable call', async () => {
    const ac = new AbortController();
    const fetchFn = mockFetch(
      res(200, '{}'),
      res(200, '{}'),
      res(200, '{}'),
      res(200, '{}'),
      res(200, '{}'),
    );
    const c = client();
    await c.health(ac.signal);
    expect(lastInit(fetchFn).signal).toBe(ac.signal);
    await c.analyze('/a.wav', 1200, ac.signal);
    expect(lastInit(fetchFn).signal).toBe(ac.signal);
    await c.peaks('/a.wav', 0, 1, 800, ac.signal);
    expect(lastInit(fetchFn).signal).toBe(ac.signal);
    await c.getJob('j1', ac.signal);
    expect(lastInit(fetchFn).signal).toBe(ac.signal);
    await c.createJob({ input_path: '/a.wav', profile: 'studio' }, ac.signal);
    expect(lastInit(fetchFn).signal).toBe(ac.signal);
  });

  it('passes null rather than undefined when nobody is watching', async () => {
    const fetchFn = mockFetch(res(200, '{}'));
    await client().health();
    expect(lastInit(fetchFn).signal).toBeNull();
  });

  it('rejects with the AbortError the caller can recognise as a cancellation', async () => {
    const abort = new DOMException('The user aborted a request.', 'AbortError');
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.reject(abort)),
    );
    const err = (await client().analyze('/a.wav').catch((e: unknown) => e)) as DOMException;
    expect(err.name).toBe('AbortError');
  });
});

describe('request bodies and paths', () => {
  it('peaks asks for the window the zoom is looking at', async () => {
    const fetchFn = mockFetch(res(200, '{}'));
    await client().peaks('/a.wav', 1.5, 2.5, 900);
    expect(fetchFn.mock.calls[0]?.[0]).toBe(`${BASE}/api/peaks`);
    expect(JSON.parse(String(lastInit(fetchFn).body))).toEqual({
      path: '/a.wav',
      start_s: 1.5,
      end_s: 2.5,
      buckets: 900,
    });
  });

  it('createJob posts the profile and overwrite flag verbatim', async () => {
    const fetchFn = mockFetch(res(200, '{"job_id":"j1"}'));
    await client().createJob({ input_path: '/a.wav', profile: 'production', overwrite: true });
    expect(JSON.parse(String(lastInit(fetchFn).body))).toEqual({
      input_path: '/a.wav',
      profile: 'production',
      overwrite: true,
    });
  });

  it('escapes the job id in every job route', async () => {
    const fetchFn = mockFetch(res(200, '{}'), res(200, '{"ok":true}'));
    await client().getJob('a b/c');
    expect(fetchFn.mock.calls[0]?.[0]).toBe(`${BASE}/api/jobs/a%20b%2Fc`);
    await client().cancelJob('a b/c');
    expect(fetchFn.mock.calls[1]?.[0]).toBe(`${BASE}/api/jobs/a%20b%2Fc/cancel`);
  });

  it('fetchText reads a served sidecar through a credential-free URL', async () => {
    const fetchFn = mockFetch(res(200, 'HAWAVOCLEAN REPORT'));
    await expect(client().fetchText('/out/a.hawavoclean.txt')).resolves.toBe('HAWAVOCLEAN REPORT');
    const url = new URL(String(fetchFn.mock.calls[0]?.[0]));
    expect(url.searchParams.get('path')).toBe('/out/a.hawavoclean.txt');
    expect(url.searchParams.has('token')).toBe(false);
    expect(lastInit(fetchFn).credentials).toBe('same-origin');
  });

  it('fetchText reports a refused file as an EngineError', async () => {
    mockFetch(res(403, '{"error":"forbidden_path","message":"outside the allowed roots"}'));
    const err = (await client().fetchText('/etc/passwd').catch((e: unknown) => e)) as EngineError;
    expect(err.code).toBe('forbidden_path');
  });
});

// ---------------------------------------------------------------------------
// uploadWithProgress — the one XHR in the client (fetch cannot report request
// progress on loopback HTTP/1.1).

interface FakeUploadTarget {
  onprogress: ((e: ProgressEvent) => void) | null;
}

class FakeXhr {
  static last: FakeXhr | null = null;
  upload: FakeUploadTarget = { onprogress: null };
  onerror: (() => void) | null = null;
  ontimeout: (() => void) | null = null;
  onabort: (() => void) | null = null;
  onload: (() => void) | null = null;
  status = 0;
  statusText = '';
  response: unknown = '';
  responseType = '';
  headers: Record<string, string> = {};
  method = '';
  url = '';
  body: unknown = null;
  sent = false;

  constructor() {
    FakeXhr.last = this;
  }
  open(method: string, url: string): void {
    this.method = method;
    this.url = url;
  }
  setRequestHeader(k: string, v: string): void {
    this.headers[k] = v;
  }
  send(body: unknown): void {
    this.body = body;
    this.sent = true;
  }
  abort(): void {
    this.onabort?.();
  }
  // test drivers
  progress(loaded: number, total: number, lengthComputable = true): void {
    this.upload.onprogress?.({ loaded, total, lengthComputable } as ProgressEvent);
  }
  finish(status: number, body: string, statusText = ''): void {
    this.status = status;
    this.statusText = statusText;
    this.response = body;
    this.onload?.();
  }
}

function bigFile(bytes: number, name = 'take.wav'): File {
  return new File([new Uint8Array(bytes)], name, { type: 'audio/wav' });
}

describe('uploadWithProgress', () => {
  beforeEach(() => {
    FakeXhr.last = null;
    vi.stubGlobal('XMLHttpRequest', FakeXhr);
  });

  it('opens a POST to /api/upload without renderer-owned auth and with a multipart body', () => {
    void client().uploadWithProgress(bigFile(8));
    const xhr = FakeXhr.last as FakeXhr;
    expect(xhr.method).toBe('POST');
    expect(xhr.url).toBe(`${BASE}/api/upload`);
    expect(xhr.headers['X-Hawa-Token']).toBeUndefined();
    expect(xhr.headers.Authorization).toBeUndefined();
    expect(xhr.body).toBeInstanceOf(FormData);
    expect(xhr.sent).toBe(true);
  });

  it('reports byte progress and pins the bar at 100% when the engine answers', async () => {
    const seen: Array<[number, number]> = [];
    const p = client().uploadWithProgress(bigFile(1000), {
      onProgress: (loaded, total) => seen.push([loaded, total]),
    });
    const xhr = FakeXhr.last as FakeXhr;
    xhr.progress(250, 1000);
    xhr.progress(750, 1000);
    xhr.finish(200, '{"path":"/work/uploads/take.wav"}');
    await expect(p).resolves.toEqual({ path: '/work/uploads/take.wav' });
    expect(seen).toEqual([
      [250, 1000],
      [750, 1000],
      [1000, 1000],
    ]);
  });

  it('falls back to the file size when the transfer is not length-computable', async () => {
    const seen: Array<[number, number]> = [];
    const p = client().uploadWithProgress(bigFile(512), {
      onProgress: (loaded, total) => seen.push([loaded, total]),
    });
    const xhr = FakeXhr.last as FakeXhr;
    xhr.progress(128, 0, false);
    xhr.finish(200, '{"path":"/work/a.wav"}');
    await p;
    expect(seen[0]).toEqual([128, 512]);
  });

  it('hands back a cancel that aborts the transfer and reads as a cancellation', async () => {
    let cancel: (() => void) | null = null;
    const p = client().uploadWithProgress(bigFile(1000), {
      onCancelHandle: (fn) => {
        cancel = fn;
      },
    });
    expect(cancel).toBeTypeOf('function');
    (cancel as unknown as () => void)();
    const err = (await p.catch((e: unknown) => e)) as DOMException;
    expect(err).toBeInstanceOf(DOMException);
    expect(err.name).toBe('AbortError');
    expect(err.message).toBe('Upload cancelled');
  });

  it('distinguishes an abort we did not ask for', async () => {
    const p = client().uploadWithProgress(bigFile(10));
    (FakeXhr.last as FakeXhr).abort();
    const err = (await p.catch((e: unknown) => e)) as DOMException;
    expect(err.name).toBe('AbortError');
    expect(err.message).toBe('Aborted');
  });

  it('maps a dead socket and a timeout to zero-status EngineErrors', async () => {
    const p1 = client().uploadWithProgress(bigFile(10));
    (FakeXhr.last as FakeXhr).onerror?.();
    const e1 = (await p1.catch((e: unknown) => e)) as EngineError;
    expect(e1).toBeInstanceOf(EngineError);
    expect([e1.status, e1.code]).toEqual([0, 'network']);

    const p2 = client().uploadWithProgress(bigFile(10));
    (FakeXhr.last as FakeXhr).ontimeout?.();
    const e2 = (await p2.catch((e: unknown) => e)) as EngineError;
    expect([e2.status, e2.code]).toEqual([0, 'timeout']);
  });

  it('reports the engine’s own refusal for a rejected upload', async () => {
    const p = client().uploadWithProgress(bigFile(10));
    (FakeXhr.last as FakeXhr).finish(413, '{"error":"too_large","message":"file exceeds the cap"}');
    const err = (await p.catch((e: unknown) => e)) as EngineError;
    expect([err.status, err.code, err.message]).toEqual([413, 'too_large', 'file exceeds the cap']);
  });

  it('treats a 200 with no path as a failure, not a success with undefined', async () => {
    const p = client().uploadWithProgress(bigFile(10));
    (FakeXhr.last as FakeXhr).finish(200, '{"ok":true}', 'OK');
    const err = (await p.catch((e: unknown) => e)) as EngineError;
    expect(err).toBeInstanceOf(EngineError);
    expect(err.code).toBe('http_200');
  });

  it('survives a non-JSON body on a failure', async () => {
    const p = client().uploadWithProgress(bigFile(10));
    (FakeXhr.last as FakeXhr).finish(500, '<html>', 'Internal Server Error');
    const err = (await p.catch((e: unknown) => e)) as EngineError;
    expect(err.code).toBe('http_500');
    expect(err.message).toBe('500 Internal Server Error');
  });
});

// ---------------------------------------------------------------------------
// verify — the one-byte ranged read behind artefact re-verification. A HEAD
// answers 200 for a chmod-000 file and for a truncated one, which is exactly
// why this method exists; these tests pin the three answers it can give.

/** A ranged-GET Response stand-in: status, headers, and a body that can die. */
function byteRes(over: {
  status?: number;
  headers?: Record<string, string>;
  bodyFails?: boolean;
}): Response {
  const status = over.status ?? 206;
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: '',
    headers: new Headers(over.headers ?? {}),
    arrayBuffer: async () => {
      if (over.bodyFails) throw new TypeError('stream died');
      return new ArrayBuffer(1);
    },
  } as unknown as Response;
}

describe('verify', () => {
  it('asks for exactly one byte and reads the full length off Content-Range', async () => {
    const fn = mockFetch(
      byteRes({ status: 206, headers: { 'Content-Range': 'bytes 0-0/13624364' } }),
    );
    const p = await client().verify('/out/a.wav');
    expect(lastHeaders(fn)['Range']).toBe('bytes=0-0');
    expect(p).toEqual({ status: 206, delivered: true, size: 13624364 });
  });

  it('a 2xx whose body dies before the byte is not delivered — the chmod-000 shape', async () => {
    mockFetch(byteRes({ status: 200, bodyFails: true }));
    const p = await client().verify('/out/a.wav');
    expect(p.delivered).toBe(false);
    expect(p.status).toBe(200);
  });

  it('a 404 is an answer: not delivered, no size, status carried', async () => {
    mockFetch(byteRes({ status: 404 }));
    const p = await client().verify('/out/a.wav');
    expect(p).toEqual({ status: 404, delivered: false, size: null });
  });

  it('a 200 without Content-Range still yields a size — the empty-file shape', async () => {
    mockFetch(byteRes({ status: 200, headers: { 'Content-Length': '0' } }));
    const p = await client().verify('/out/a.wav');
    expect(p).toEqual({ status: 200, delivered: true, size: 0 });
  });

  it('an engine that never answers re-raises — offline is not deleted', async () => {
    const fn = vi.fn().mockRejectedValue(new TypeError('Failed to fetch'));
    vi.stubGlobal('fetch', fn);
    // Still re-raised rather than answered as "the file is gone" — "the engine
    // is offline" and "your master was deleted" are two different things and
    // the UI says two different things about them. It now arrives typed, as an
    // EngineError with status 0, instead of a raw TypeError.
    const err = (await client().verify('/out/a.wav').catch((e: unknown) => e)) as EngineError;
    expect(err).toBeInstanceOf(EngineError);
    expect(err.status).toBe(0);
  });
});

describe('v1 capabilities and jobs (True-10 D4.11)', () => {
  it('fetches and normalizes /api/v1/capabilities', async () => {
    const fetchFn = mockFetch(
      res(
        200,
        JSON.stringify({
          schemaVersion: 1,
          capabilities: [
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
              reason: 'No qualified model pack',
            },
          ],
        }),
      ),
    );
    const caps = await client().capabilities();
    expect(fetchFn.mock.calls[0]?.[0]).toBe(`${BASE}/api/v1/capabilities`);
    expect(caps.capabilities).toEqual([
      {
        capability_id: 'smart_safe',
        available: true,
        maturity: 'qualified',
        reason: null,
        manifest_sha256: null,
        providers: ['cpu'],
      },
      {
        capability_id: 'restore_source',
        available: false,
        maturity: 'blocked',
        reason: 'No qualified model pack',
        manifest_sha256: null,
        providers: [],
      },
    ]);
  });

  it('posts /api/v1/jobs and normalizes response', async () => {
    const fetchFn = mockFetch(
      res(
        202,
        JSON.stringify({
          schemaVersion: 1,
          jobs: [
            {
              jobId: 'j123',
              sourceId: 'src456',
              outputPath: '/out/a.wav',
              reportPath: '/out/a.json',
            },
          ],
        }),
      ),
    );
    const result = await client().createV1Jobs({
      schema_version: 1,
      source_ids: ['src456'],
      strategy: {
        kind: 'smart_safe',
        restore_policy: 'disabled',
        allow_generative_reconstruction: false,
      },
      execution_policy: 'offline_only',
      conflict_policy: 'unique',
      record_bundle: false,
      idempotency_key: 'test-key',
    });
    expect(fetchFn.mock.calls[0]?.[0]).toBe(`${BASE}/api/v1/jobs`);
    expect(JSON.parse(String(lastInit(fetchFn).body))).toEqual(
      expect.objectContaining({
        schema_version: 1,
        source_ids: ['src456'],
        idempotency_key: 'test-key',
      }),
    );
    expect(result.jobs).toEqual([
      {
        jobId: 'j123',
        sourceId: 'src456',
        outputPath: '/out/a.wav',
        reportPath: '/out/a.json',
      },
    ]);
  });
});
