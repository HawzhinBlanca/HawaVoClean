import type { HawaBridge } from './types';

export type { HawaBridge, HawaHost, ResolveClip } from './types';

const LOOPBACK_HOSTS = new Set(['127.0.0.1', 'localhost']);
const UNSUPPORTED_PREVIEW_PORTS = new Set(['3000', '4173', '5173', '8080']);

function webBaseUrl(): string {
  // Web mode is intentionally same-origin only: its authentication is an
  // HttpOnly session cookie. There is no token query, localStorage fallback,
  // or cross-origin development secret. The desktop/Resolve shells provide a
  // trusted bridge and therefore never take this branch.
  const location = new URL(window.location.href);
  if (import.meta.env.DEV || UNSUPPORTED_PREVIEW_PORTS.has(location.port)) {
    throw new Error(
      'Web development authentication is unavailable. Use the desktop shell or serve the built UI from the engine origin.',
    );
  }
  if (
    (location.protocol === 'http:' || location.protocol === 'https:') &&
    LOOPBACK_HOSTS.has(location.hostname)
  ) {
    return location.origin;
  }
  throw new Error('Web mode requires a same-origin loopback HawaVoClean engine session.');
}

function buildWebBridge(): HawaBridge {
  return {
    host: 'web',
    engine: {
      async getEndpoint() {
        return { baseUrl: webBaseUrl() };
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
      async registerDroppedFile() {
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
