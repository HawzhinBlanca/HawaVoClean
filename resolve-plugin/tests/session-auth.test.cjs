'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const test = require('node:test');

const {
  exactEngineOrigin,
  isEngineApiRequest,
  rendererEngineEndpoint,
  requestEngineSession,
  requestNativeSourceRegistration,
  sessionNeedsRenewal,
  withEngineAuthorization,
  withoutRendererCredentials,
} = require('../com.hawavoclean.resolve/session-auth.js');

const ENGINE = 'http://127.0.0.1:48123';

test('renderer endpoint is immutable and contains no credential material', () => {
  const endpoint = rendererEngineEndpoint(48123);
  assert.deepEqual(endpoint, { baseUrl: ENGINE });
  assert.deepEqual(Object.keys(endpoint), ['baseUrl']);
  assert.equal(Object.isFrozen(endpoint), true);
  assert.throws(() => rendererEngineEndpoint(0), /port/);

  const preload = fs.readFileSync(
    path.join(__dirname, '..', 'com.hawavoclean.resolve', 'preload.js'),
    'utf8',
  );
  assert.doesNotMatch(
    preload,
    /X-Hawa-Token|Authorization|sessionToken|hawa_session|localStorage|sessionStorage/,
    'preload must neither receive nor persist an engine secret',
  );
});

test('engine origin and API matching reject origin confusion and non-API paths', () => {
  assert.equal(exactEngineOrigin(ENGINE), ENGINE);
  assert.equal(exactEngineOrigin('http://localhost:48123'), null);
  assert.equal(exactEngineOrigin('http://user@127.0.0.1:48123'), null);
  assert.equal(exactEngineOrigin('http://127.0.0.1:48123/path'), null);
  assert.equal(exactEngineOrigin('https://127.0.0.1:48123'), null);

  for (const value of [
    `${ENGINE}/api`,
    `${ENGINE}/api/health`,
    `${ENGINE}/api/jobs/1/events`,
    `${ENGINE}/api/audio?path=%2Fa.wav`,
  ]) {
    assert.equal(isEngineApiRequest(value, ENGINE), true, value);
  }
  for (const value of [
    `${ENGINE}/`,
    `${ENGINE}/apiary`,
    'http://127.0.0.1:48124/api/health',
    'http://localhost:48123/api/health',
    'http://127.0.0.1.evil.invalid:48123/api/health',
    'http://127.0.0.1:48123.evil.invalid/api/health',
    'http://user@127.0.0.1:48123/api/health',
    'https://127.0.0.1:48123/api/health',
  ]) {
    assert.equal(isEngineApiRequest(value, ENGINE), false, value);
  }
});

test('main-owned Authorization replaces every renderer-controlled credential', () => {
  const rendererHeaders = {
    Range: 'bytes=0-1023',
    authorization: 'Bearer renderer-controlled',
    'X-Hawa-Token': 'renderer-controlled-root',
    Cookie: 'hawa_session=renderer-controlled',
  };
  assert.deepEqual(withoutRendererCredentials(rendererHeaders), { Range: 'bytes=0-1023' });
  const authorization = `Bearer ${'a'.repeat(43)}`;
  assert.deepEqual(withEngineAuthorization(rendererHeaders, authorization), {
    Range: 'bytes=0-1023',
    Authorization: authorization,
  });
  assert.throws(() => withEngineAuthorization({}, 'Bearer short'), /invalid/);
});

test('session bootstrap sends only the root header and validates bounded no-store metadata', async () => {
  let observed = null;
  const server = http.createServer((request, response) => {
    observed = { method: request.method, url: request.url, headers: request.headers };
    response.writeHead(200, {
      'Cache-Control': 'private, no-store',
      'Content-Type': 'application/json',
      'Set-Cookie': 'hawa_session=must-not-be-used-by-node',
    });
    response.end(JSON.stringify({
      sessionToken: 's'.repeat(43),
      expiresInSeconds: 900,
      tokenType: 'Bearer',
    }));
  });
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  try {
    const address = server.address();
    assert.ok(address && typeof address === 'object');
    const capability = await requestEngineSession(
      { baseUrl: `http://127.0.0.1:${address.port}` },
      'native-root-secret',
      () => 1_000,
    );
    assert.deepEqual(capability, {
      authorization: `Bearer ${'s'.repeat(43)}`,
      expiresAtMs: 901_000,
      refreshAtMs: 841_000,
    });
    assert.equal(Object.isFrozen(capability), true);
    assert.equal(observed.method, 'POST');
    assert.equal(observed.url, '/api/session');
    assert.equal(observed.headers['x-hawa-token'], 'native-root-secret');
    assert.equal(observed.headers.authorization, undefined);
    assert.equal(observed.headers.cookie, undefined);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test('session bootstrap fails closed on a foreign endpoint and malformed capability', async () => {
  await assert.rejects(
    requestEngineSession({ baseUrl: 'http://localhost:48123' }, 'root'),
    /exact 127\.0\.0\.1/,
  );

  const server = http.createServer((_request, response) => {
    response.writeHead(200, { 'Cache-Control': 'no-store' });
    response.end(JSON.stringify({
      sessionToken: 'too-short',
      expiresInSeconds: 900,
      tokenType: 'Bearer',
    }));
  });
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  try {
    const address = server.address();
    assert.ok(address && typeof address === 'object');
    await assert.rejects(
      requestEngineSession({ baseUrl: `http://127.0.0.1:${address.port}` }, 'root'),
      /invalid capability metadata/,
    );
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test('Resolve registers selected files through the root-owned native channel', async () => {
  let observed = null;
  const selected = path.resolve('/tmp/resolve-selected.wav');
  const server = http.createServer((request, response) => {
    const chunks = [];
    request.on('data', (chunk) => chunks.push(chunk));
    request.on('end', () => {
      observed = {
        method: request.method,
        url: request.url,
        headers: request.headers,
        body: JSON.parse(Buffer.concat(chunks).toString('utf8')),
      };
      response.writeHead(200, { 'Cache-Control': 'no-store', 'Content-Type': 'application/json' });
      response.end(JSON.stringify({ sourceId: 'b'.repeat(32), path: selected }));
    });
  });
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  try {
    const address = server.address();
    assert.ok(address && typeof address === 'object');
    const result = await requestNativeSourceRegistration(
      { baseUrl: `http://127.0.0.1:${address.port}` },
      'resolve-root-secret',
      selected,
    );
    assert.deepEqual(result, { sourceId: 'b'.repeat(32), path: selected });
    assert.equal(observed.url, '/api/v1/native-sources');
    assert.equal(observed.headers['x-hawa-token'], 'resolve-root-secret');
    assert.equal(observed.headers.authorization, undefined);
    assert.deepEqual(observed.body, { path: selected });
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test('session renewal is early and fails safe for invalid clocks', () => {
  const capability = { authorization: `Bearer ${'a'.repeat(43)}`, refreshAtMs: 840, expiresAtMs: 900 };
  assert.equal(sessionNeedsRenewal(capability, 839), false);
  assert.equal(sessionNeedsRenewal(capability, 840), true);
  assert.equal(sessionNeedsRenewal(capability, 900), true);
  assert.equal(sessionNeedsRenewal(capability, Number.NaN), true);
  assert.equal(sessionNeedsRenewal(null, 1), true);
});

test('Resolve main intercepts only loopback HTTP and blocks renderer session minting', () => {
  const main = fs.readFileSync(
    path.join(__dirname, '..', 'com.hawavoclean.resolve', 'main.js'),
    'utf8',
  );
  assert.match(main, /onBeforeSendHeaders\(\s*\{ urls: \['http:\/\/127\.0\.0\.1:\*\/\*'\] \}/);
  assert.doesNotMatch(main, /onBeforeSendHeaders\(\s*\{ urls: \['<all_urls>'\] \}/);
  assert.match(main, /url\.pathname !== '\/api\/session'/);
  assert.match(main, /detached: process\.platform !== 'win32'/);
  assert.match(main, /env\.HAWAVOCLEAN_PARENT_PID = String\(process\.pid\)/);
  assert.match(main, /process\.kill\(-child\.pid, signal\)/);
  assert.doesNotMatch(main, /engine\.child\.kill\('SIGTERM'\)/);
  assert.match(main, /'--token-stdin'/);
  assert.doesNotMatch(main, /'--token', engine\.token/);
  assert.match(main, /child\.stdin\.end\(`\$\{engine\.token\}\\n`\)/);
  assert.match(main, /requestNativeSourceRegistration\(endpoint, engine\.token, filePath\)/);
  assert.match(main, /filePath: await registerNativePath\(clip\.filePath\)/);
  assert.match(main, /hawa:files:register-dropped/);
  assert.match(main, /\['wav', 'wave', 'aif', 'aiff', 'aifc', 'flac', 'mp3', 'm4a', 'mp4'\]/);
  assert.match(main, /AUDIO_EXTENSIONS\.includes\(extension\)/);
  assert.doesNotMatch(main, /'mov'|'aac'|'ogg'|'opus'|'caf'|'mkv'|'webm'/);
});
