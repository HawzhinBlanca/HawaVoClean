import fs from 'node:fs';
import path from 'node:path';

export const APP_SCHEME = 'hawa';
export const APP_ORIGIN = `${APP_SCHEME}://app`;
export const APP_ENTRY_URL = `${APP_ORIGIN}/index.html`;

export function pathInside(root: string, candidate: string): boolean {
  const relative = path.relative(path.resolve(root), path.resolve(candidate));
  return (
    relative === '' ||
    (relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative))
  );
}

export function isTrustedRendererUrl(value: string): boolean {
  try {
    const url = new URL(value);
    return url.protocol === `${APP_SCHEME}:` && url.hostname === 'app' && url.pathname === '/index.html';
  } catch {
    return false;
  }
}

export function isEngineApiRequest(value: string, engineOrigin: string | null): boolean {
  if (engineOrigin === null) return false;
  try {
    const url = new URL(value);
    return (
      url.protocol === 'http:' &&
      url.hostname === '127.0.0.1' &&
      url.username === '' &&
      url.password === '' &&
      url.origin === engineOrigin &&
      (url.pathname === '/api' || url.pathname.startsWith('/api/'))
    );
  } catch {
    return false;
  }
}

export function withoutRendererCredentials(
  headers: Readonly<Record<string, string>>,
): Record<string, string> {
  const secured: Record<string, string> = {};
  for (const [name, value] of Object.entries(headers)) {
    const lowered = name.toLowerCase();
    if (lowered === 'authorization' || lowered === 'x-hawa-token' || lowered === 'cookie') continue;
    secured[name] = value;
  }
  return secured;
}

/** Replace every renderer-supplied credential with the main-owned session. */
export function withEngineAuthorization(
  headers: Readonly<Record<string, string>>,
  authorization: string,
): Record<string, string> {
  if (!/^Bearer [A-Za-z0-9_-]{32,256}$/.test(authorization)) {
    throw new Error('Engine session authorization is invalid.');
  }
  const secured = withoutRendererCredentials(headers);
  secured.Authorization = authorization;
  return secured;
}

export function isAllowedRendererRequest(value: string, engineOrigin: string | null): boolean {
  try {
    const url = new URL(value);
    if (url.protocol === `${APP_SCHEME}:`) return url.hostname === 'app';
    if (url.protocol === 'blob:' || url.protocol === 'data:') return true;
    // Session minting is main-only even though it is an API route.
    return (
      isEngineApiRequest(value, engineOrigin) &&
      url.pathname !== '/api/session' &&
      !url.pathname.startsWith('/api/session/')
    );
  } catch {
    return false;
  }
}

export function resolveAppAsset(root: string, rawUrl: string): string | null {
  try {
    const url = new URL(rawUrl);
    if (url.protocol !== `${APP_SCHEME}:` || url.hostname !== 'app' || url.search || url.hash) return null;
    const decoded = decodeURIComponent(url.pathname);
    if (decoded.includes('\u0000') || decoded.includes('\\')) return null;
    const rootReal = fs.realpathSync(root);
    const candidate = path.resolve(rootReal, `.${decoded}`);
    if (!pathInside(rootReal, candidate)) return null;
    const candidateReal = fs.realpathSync(candidate);
    if (!pathInside(rootReal, candidateReal) || !fs.statSync(candidateReal).isFile()) return null;
    return candidateReal;
  } catch {
    return null;
  }
}
