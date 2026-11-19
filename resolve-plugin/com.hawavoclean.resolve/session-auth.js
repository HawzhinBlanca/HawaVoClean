'use strict';

const http = require('node:http');
const path = require('node:path');

const SESSION_HTTP_TIMEOUT_MS = 3_000;
const SESSION_RESPONSE_LIMIT = 8 * 1024;
const SESSION_MAX_TTL_S = 60 * 60;
const SESSION_RENEW_SKEW_MS = 60_000;
const NATIVE_SOURCE_HTTP_TIMEOUT_MS = 3_000;
const NATIVE_SOURCE_RESPONSE_LIMIT = 8 * 1024;

function rendererEngineEndpoint(port) {
  if (!Number.isInteger(port) || port < 1 || port > 65_535) {
    throw new Error('Engine ready port is invalid.');
  }
  return Object.freeze({ baseUrl: `http://127.0.0.1:${port}` });
}

function exactEngineOrigin(value) {
  if (typeof value !== 'string') return null;
  try {
    const url = new URL(value);
    if (
      url.protocol !== 'http:' ||
      url.hostname !== '127.0.0.1' ||
      url.username !== '' ||
      url.password !== '' ||
      url.pathname !== '/' ||
      url.search !== '' ||
      url.hash !== '' ||
      !/^\d+$/.test(url.port) ||
      Number(url.port) < 1 ||
      Number(url.port) > 65_535
    ) {
      return null;
    }
    return url.origin;
  } catch {
    return null;
  }
}

function isEngineApiRequest(value, engineOrigin) {
  if (exactEngineOrigin(engineOrigin) === null || typeof value !== 'string') return false;
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

function withoutRendererCredentials(headers) {
  const secured = {};
  for (const [name, value] of Object.entries(headers || {})) {
    const lowered = name.toLowerCase();
    if (lowered === 'authorization' || lowered === 'x-hawa-token' || lowered === 'cookie') continue;
    secured[name] = value;
  }
  return secured;
}

function withEngineAuthorization(headers, authorization) {
  if (!/^Bearer [A-Za-z0-9_-]{32,256}$/.test(authorization)) {
    throw new Error('Engine session authorization is invalid.');
  }
  return { ...withoutRendererCredentials(headers), Authorization: authorization };
}

function sessionNeedsRenewal(capability, nowMs) {
  return (
    capability === null ||
    typeof capability !== 'object' ||
    !Number.isFinite(nowMs) ||
    nowMs >= capability.refreshAtMs ||
    nowMs >= capability.expiresAtMs
  );
}

function postLoopbackJson(
  endpoint,
  rootToken,
  routePath,
  label,
  timeoutMs,
  maxBytes,
  refusedStatusMessage,
  payload,
) {
  const origin = endpoint && exactEngineOrigin(endpoint.baseUrl);
  if (origin === null) {
    return Promise.reject(new Error('Engine endpoint is not an exact 127.0.0.1 HTTP origin.'));
  }
  if (typeof rootToken !== 'string' || rootToken.length < 1 || rootToken.length > 256) {
    return Promise.reject(new Error('Engine bootstrap secret is invalid.'));
  }
  const body = payload === undefined ? null : Buffer.from(JSON.stringify(payload), 'utf8');
  const url = new URL(origin);
  return new Promise((resolve, reject) => {
    let settled = false;
    const finish = (error, value) => {
      if (settled) return;
      settled = true;
      if (error) reject(error);
      else if (value) resolve(value);
      else reject(new Error(`${label} produced no result.`));
    };
    const headers = {
      'X-Hawa-Token': rootToken,
      Accept: 'application/json',
      'Content-Length': body ? String(body.length) : '0',
    };
    if (body) headers['Content-Type'] = 'application/json';
    const request = http.request(
      { hostname: '127.0.0.1', port: Number(url.port), method: 'POST', path: routePath, headers, timeout: timeoutMs },
      (response) => {
        const chunks = [];
        let bytes = 0;
        response.on('data', (chunk) => {
          bytes += chunk.length;
          if (bytes > maxBytes) {
            response.destroy();
            finish(new Error(`${label} response exceeded its size limit.`));
            return;
          }
          chunks.push(chunk);
        });
        response.once('aborted', () => finish(new Error(`${label} response was interrupted.`)));
        response.once('error', () => finish(new Error(`${label} response failed.`)));
        response.once('end', () => {
          if (response.statusCode !== 200) {
            finish(new Error(refusedStatusMessage(response.statusCode || 0)));
            return;
          }
          const cacheControl = String(response.headers['cache-control'] || '').toLowerCase();
          if (!cacheControl.includes('no-store')) {
            finish(new Error(`${label} response was not marked no-store.`));
            return;
          }
          let parsed;
          try {
            parsed = JSON.parse(Buffer.concat(chunks).toString('utf8'));
          } catch {
            finish(new Error(`${label} response was not valid JSON.`));
            return;
          }
          if (!parsed || Array.isArray(parsed) || typeof parsed !== 'object') {
            finish(new Error(`${label} response had the wrong shape.`));
            return;
          }
          finish(null, parsed);
        });
      },
    );
    request.once('timeout', () => {
      request.destroy();
      finish(new Error(`${label} timed out.`));
    });
    request.once('error', () => finish(new Error(`${label} failed.`)));
    request.end(body || undefined);
  });
}

async function requestEngineSession(endpoint, rootToken, now = () => Date.now()) {
  const body = await postLoopbackJson(
    endpoint,
    rootToken,
    '/api/session',
    'Engine session bootstrap',
    SESSION_HTTP_TIMEOUT_MS,
    SESSION_RESPONSE_LIMIT,
    (code) => `Engine refused session bootstrap (HTTP ${String(code)}).`,
  );
  const token = body.sessionToken;
  const ttlS = body.expiresInSeconds;
  if (
    body.tokenType !== 'Bearer' ||
    typeof token !== 'string' ||
    !/^[A-Za-z0-9_-]{32,256}$/.test(token) ||
    !Number.isInteger(ttlS) ||
    ttlS < 1 ||
    ttlS > SESSION_MAX_TTL_S
  ) {
    throw new Error('Engine session response had invalid capability metadata.');
  }
  const issuedAt = now();
  if (!Number.isFinite(issuedAt)) {
    throw new Error('Engine session clock is invalid.');
  }
  const ttlMs = ttlS * 1_000;
  const skewMs = Math.min(SESSION_RENEW_SKEW_MS, Math.max(1_000, ttlMs / 4));
  return Object.freeze({
    authorization: `Bearer ${token}`,
    expiresAtMs: issuedAt + ttlMs,
    refreshAtMs: issuedAt + Math.max(1, ttlMs - skewMs),
  });
}

async function requestNativeSourceRegistration(endpoint, rootToken, filePath) {
  if (
    typeof filePath !== 'string' ||
    filePath.length < 1 ||
    filePath.length > 32_768 ||
    filePath.includes('\0') ||
    !path.isAbsolute(filePath)
  ) {
    throw new Error('Native source path is invalid.');
  }
  const parsed = await postLoopbackJson(
    endpoint,
    rootToken,
    '/api/v1/native-sources',
    'Native source registration',
    NATIVE_SOURCE_HTTP_TIMEOUT_MS,
    NATIVE_SOURCE_RESPONSE_LIMIT,
    (code) => `Engine refused the selected file (HTTP ${String(code)}).`,
    { path: filePath },
  );
  if (
    typeof parsed.sourceId !== 'string' ||
    !/^[0-9a-f]{32}$/.test(parsed.sourceId) ||
    typeof parsed.path !== 'string' ||
    parsed.path.length < 1 ||
    parsed.path.length > 32_768 ||
    parsed.path.includes('\0') ||
    !path.isAbsolute(parsed.path)
  ) {
    throw new Error('Native source registration response had invalid capability metadata.');
  }
  return Object.freeze({ sourceId: parsed.sourceId, path: parsed.path });
}

module.exports = {
  exactEngineOrigin,
  isEngineApiRequest,
  rendererEngineEndpoint,
  requestEngineSession,
  requestNativeSourceRegistration,
  sessionNeedsRenewal,
  withEngineAuthorization,
  withoutRendererCredentials,
};
