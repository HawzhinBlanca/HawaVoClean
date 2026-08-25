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

const LOOPBACK_HOSTS = new Set(['127.0.0.1', 'localhost', '[::1]', '::1']);

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
