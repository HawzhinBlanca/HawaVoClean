import type { HawaBridge } from './types';

export type { HawaBridge, HawaHost, ResolveClip } from './types';

const TOKEN_KEY = 'hawa.token';

function readWebToken(): string {
  try {
    const q = new URLSearchParams(window.location.search).get('token');
    if (q) {
      window.localStorage.setItem(TOKEN_KEY, q);
      return q;
    }
    return window.localStorage.getItem(TOKEN_KEY) ?? '';
  } catch {
    return '';
  }
}

/**
 * Hosts a `?engine=` override may name.
 *
 * Pinned to the one literal spelling, because this set and index.html's
 * `connect-src`/`media-src` are two lists that have to agree and did not. CSP
 * host matching is literal: `http://127.0.0.1:*` does not match the host
 * `localhost`, and CSP's host grammar has no IPv6-literal form at all — there
 * is no source expression that can cover `[::1]`, so adding one would be
 * silently dropped by the browser. Accepting those three here meant the
 * override produced a base URL the page's own policy forbade it from
 * contacting, and every `fetch`, `EventSource` and `<audio src>` against it
 * failed and was reported to the user as "Engine unreachable" — a true
 * sentence about a cause the user could not possibly guess.
 *
 * `https:` is deliberately still accepted by the protocol test below: CSP3
 * matches `http://127.0.0.1:*` against an https origin on the same host, so it
 * was never part of this problem.
 *
 * `bridge.test.ts` asserts every member of this set appears in both CSP
 * directives, so widening either side alone fails loudly.
 */
const LOOPBACK_HOSTS = new Set(['127.0.0.1']);

function webBaseUrl(): string {
  // Served by `hawavoclean serve --ui-dir` → same origin. Under the Vite dev
  // server (port 5173) or a static preview there is no engine on this origin,
  // so fall back to ?engine= or the mock engine's default port. The override
  // is restricted to loopback so a crafted link cannot point the UI (and its
  // stored token) at a remote host.
  const q = new URLSearchParams(window.location.search).get('engine');
  if (q) {
    try {
      const u = new URL(q);
      if ((u.protocol === 'http:' || u.protocol === 'https:') && LOOPBACK_HOSTS.has(u.hostname)) {
        return u.origin;
      }
    } catch {
      /* malformed ?engine= — ignore */
    }
  }
  // Under the dev server there is never an engine on this origin, whatever
  // port Vite ended up on. Deciding by build mode rather than by matching a
  // port literal means a moved dev port cannot make the UI aim its API client
  // (and its token) at the Vite server and then report the engine unreachable.
  if (import.meta.env.DEV) return 'http://127.0.0.1:8765';
  const { protocol, port } = window.location;
  if (protocol === 'http:' || protocol === 'https:') {
    if (port === '5173' || port === '4173' || port === '8080' || port === '3000') {
      return 'http://127.0.0.1:8765';
    }
    return window.location.origin;
  }
  return 'http://127.0.0.1:8765';
}

function buildWebBridge(): HawaBridge {
  return {
    host: 'web',
    engine: {
      async getEndpoint() {
        return { baseUrl: webBaseUrl(), token: readWebToken() };
      },
    },
    files: {
      async pickAudio() {
        // Browsers cannot hand us a path; the SourceStrip uses <input type=file>
        // + POST /api/upload in web mode instead.
        return null;
      },
      pathForFile() {
        return null;
      },
      async revealInFinder() {
        /* not available in a browser */
      },
    },
  };
}

let cached: HawaBridge | null = null;

export function getBridge(): HawaBridge {
  if (cached) return cached;
  cached = window.hawa ?? buildWebBridge();
  return cached;
}
