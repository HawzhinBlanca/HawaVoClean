'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const {
  AUDIO_EXTENSIONS,
  IPC,
  parseExportRequest,
  requireAbsolutePath,
  requireSupportedMediaPath,
  safeClearSessionStorage,
  safeSuggestedExportName,
  SESSION_PARTITION,
} = require('../dist/contracts.js');
const {
  engineSessionNeedsRenewal,
  parseDevelopmentCommand,
  rendererEngineEndpoint,
  requestEngineSession,
  requestNativeSourceRegistration,
  resolveEngineSpec,
} = require('../dist/engine.js');
const {
  APP_ENTRY_URL,
  isAllowedRendererRequest,
  isEngineApiRequest,
  isTrustedRendererUrl,
  pathInside,
  resolveAppAsset,
  withoutRendererCredentials,
  withEngineAuthorization,
} = require('../dist/security.js');
const {
  PACKAGE_PROVENANCE_NAME,
  packagedProofSelfTestAllowed,
} = require('../dist/proof.js');

test('session renewal starts before expiry and never trusts a non-finite clock', () => {
  const capability = { authorization: `Bearer ${'a'.repeat(43)}`, refreshAtMs: 840, expiresAtMs: 900 };
  assert.equal(engineSessionNeedsRenewal(capability, 839), false);
  assert.equal(engineSessionNeedsRenewal(capability, 840), true);
  assert.equal(engineSessionNeedsRenewal(capability, 900), true);
  assert.equal(engineSessionNeedsRenewal(capability, Number.NaN), true);
  assert.equal(engineSessionNeedsRenewal(null, 1), true);
});

test('renderer engine endpoint contains a base URL and no credential material', () => {
  const endpoint = rendererEngineEndpoint(48123);
  assert.deepEqual(endpoint, { baseUrl: 'http://127.0.0.1:48123' });
  assert.deepEqual(Object.keys(endpoint), ['baseUrl']);
  assert.equal(Object.isFrozen(endpoint), true);
  assert.throws(() => rendererEngineEndpoint(0), /port/);
});

test('IPC surface is fixed, unique, and contains no generic transport', () => {
  const values = Object.values(IPC);
  assert.equal(new Set(values).size, values.length);
  assert.deepEqual(values.sort(), [
    'hawa:app:info',
    'hawa:diagnostics:state',
    'hawa:engine:endpoint',
    'hawa:files:choose-export',
    'hawa:files:pick-audio',
    'hawa:files:pick-folder',
    'hawa:files:register-dropped',
    'hawa:files:reveal',
    'hawa:session:clear-local-data',
    'hawa:updates:state',
  ]);
  assert.ok(values.every((value) => !value.includes('*') && !value.includes('eval')));
  const compiledPreload = fs.readFileSync(path.join(__dirname, '..', 'dist', 'preload.js'), 'utf8');
  assert.doesNotMatch(compiledPreload, /require\(["']\.\.?\//, 'sandboxed preload must be self-contained');
  assert.doesNotMatch(
    compiledPreload,
    /X-Hawa-Token|Authorization|localStorage|sessionStorage|hawa_session/,
    'preload must not expose or persist engine credentials',
  );
  for (const channel of values) assert.match(compiledPreload, new RegExp(channel.replaceAll(':', '\\:')));
});

test('packaged media chooser advertises only the production decode contract', () => {
  assert.deepEqual([...AUDIO_EXTENSIONS], [
    'wav', 'wave', 'aif', 'aiff', 'aifc', 'flac', 'mp3', 'm4a', 'mp4',
  ]);
  assert.equal(requireSupportedMediaPath('/tmp/take.MP4', 'source'), '/tmp/take.MP4');
  for (const extension of ['mov', 'aac', 'ogg', 'opus', 'caf', 'mkv', 'webm']) {
    assert.throws(
      () => requireSupportedMediaPath(`/tmp/take.${extension}`, 'source'),
      /supported audio or video extension/,
    );
  }
});

test('renderer trust is exact and does not extend to foreign paths or origins', () => {
  assert.equal(isTrustedRendererUrl(APP_ENTRY_URL), true);
  assert.equal(isTrustedRendererUrl('hawa://app/assets/index.js'), false);
  assert.equal(isTrustedRendererUrl('hawa://evil/index.html'), false);
  assert.equal(isTrustedRendererUrl('https://example.com/index.html'), false);
});

test('renderer network allowlist admits only app assets, blobs, data, and exact engine origin', () => {
  const engine = 'http://127.0.0.1:48123';
  assert.equal(isAllowedRendererRequest('hawa://app/assets/index.js', engine), true);
  assert.equal(isAllowedRendererRequest('blob:hawa://app/id', engine), true);
  assert.equal(isAllowedRendererRequest('data:image/png;base64,AA==', engine), true);
  assert.equal(isAllowedRendererRequest(`${engine}/api/health`, engine), true);
  assert.equal(isAllowedRendererRequest(`${engine}/not-api`, engine), false);
  assert.equal(isAllowedRendererRequest(`${engine}/api/session`, engine), false);
  assert.equal(isAllowedRendererRequest(`${engine}/api/session/`, engine), false);
  assert.equal(isAllowedRendererRequest('http://127.0.0.1:48124/api/health', engine), false);
  assert.equal(isAllowedRendererRequest('http://localhost:48123/api/health', engine), false);
  assert.equal(isAllowedRendererRequest('https://example.com/', engine), false);
  assert.equal(isAllowedRendererRequest(`${engine}/api/health`, null), false);
});

test('engine API matching and auth injection are exact and replace renderer credentials', () => {
  const engine = 'http://127.0.0.1:48123';
  assert.equal(isEngineApiRequest(`${engine}/api/audio?path=%2Fa.wav`, engine), true);
  assert.equal(isEngineApiRequest(`${engine}/apiary`, engine), false);
  assert.equal(isEngineApiRequest(`http://user@127.0.0.1:48123/api/health`, engine), false);
  assert.equal(isEngineApiRequest('http://127.0.0.1:48124/api/health', engine), false);

  const original = {
    Range: 'bytes=0-0',
    authorization: 'Bearer renderer-controlled',
    'X-Hawa-Token': 'root-must-not-cross-renderer',
    Cookie: 'hawa_session=stale',
  };
  const authorization = `Bearer ${'a'.repeat(43)}`;
  assert.deepEqual(withoutRendererCredentials(original), { Range: 'bytes=0-0' });
  const secured = withEngineAuthorization(original, authorization);
  assert.deepEqual(secured, { Range: 'bytes=0-0', Authorization: authorization });
  assert.equal(original['X-Hawa-Token'], 'root-must-not-cross-renderer');
  assert.throws(() => withEngineAuthorization({}, 'Bearer short'), /invalid/);
});

test('session bootstrap uses only a root header and validates bounded no-store metadata', async () => {
  let observed = null;
  const server = http.createServer((request, response) => {
    observed = { method: request.method, url: request.url, headers: request.headers };
    response.writeHead(200, {
      'Cache-Control': 'no-store',
      'Content-Type': 'application/json',
      'Set-Cookie': 'hawa_session=must-not-be-persisted-by-node',
    });
    response.end(
      JSON.stringify({
        sessionToken: 's'.repeat(43),
        expiresInSeconds: 900,
        tokenType: 'Bearer',
      }),
    );
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
    assert.equal(observed.method, 'POST');
    assert.equal(observed.url, '/api/session');
    assert.equal(observed.headers['x-hawa-token'], 'native-root-secret');
    assert.equal(observed.headers.authorization, undefined);
    assert.equal(observed.headers.cookie, undefined);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test('session bootstrap fails closed on a foreign endpoint or malformed capability', async () => {
  await assert.rejects(
    requestEngineSession({ baseUrl: 'http://localhost:48123' }, 'root'),
    /exact 127\.0\.0\.1/,
  );

  const server = http.createServer((_request, response) => {
    response.writeHead(200, { 'Cache-Control': 'no-store' });
    response.end(
      JSON.stringify({ sessionToken: 'too-short', expiresInSeconds: 900, tokenType: 'Bearer' }),
    );
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

test('native source registration stays on the root-owned channel and validates the capability', async () => {
  let observed = null;
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
      response.end(JSON.stringify({ sourceId: 'a'.repeat(32), path: path.resolve('/tmp/selected.wav') }));
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
      'native-root-secret',
      path.resolve('/tmp/selected.wav'),
    );
    assert.deepEqual(result, { sourceId: 'a'.repeat(32), path: path.resolve('/tmp/selected.wav') });
    assert.equal(Object.isFrozen(result), true);
    assert.equal(observed.method, 'POST');
    assert.equal(observed.url, '/api/v1/native-sources');
    assert.equal(observed.headers['x-hawa-token'], 'native-root-secret');
    assert.equal(observed.headers.authorization, undefined);
    assert.deepEqual(observed.body, { path: path.resolve('/tmp/selected.wav') });
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }

  await assert.rejects(
    requestNativeSourceRegistration({ baseUrl: 'http://localhost:48123' }, 'root', '/tmp/a.wav'),
    /exact 127\.0\.0\.1/,
  );
  await assert.rejects(
    requestNativeSourceRegistration({ baseUrl: 'http://127.0.0.1:48123' }, 'root', '../a.wav'),
    /path is invalid/,
  );
});

test('custom protocol resolver blocks traversal, encoded traversal, queries, and symlink escape', () => {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'hawa-desktop-test-'));
  const root = path.join(temp, 'ui');
  fs.mkdirSync(path.join(root, 'assets'), { recursive: true });
  fs.writeFileSync(path.join(root, 'index.html'), 'ok');
  fs.writeFileSync(path.join(root, 'assets', 'app.js'), 'ok');
  const outside = path.join(temp, 'secret.txt');
  fs.writeFileSync(outside, 'secret');
  fs.symlinkSync(outside, path.join(root, 'assets', 'escape.txt'));
  try {
    assert.equal(resolveAppAsset(root, APP_ENTRY_URL), fs.realpathSync(path.join(root, 'index.html')));
    assert.equal(resolveAppAsset(root, 'hawa://app/assets/app.js'), fs.realpathSync(path.join(root, 'assets', 'app.js')));
    assert.equal(resolveAppAsset(root, 'hawa://app/../secret.txt'), null);
    assert.equal(resolveAppAsset(root, 'hawa://app/%2e%2e/secret.txt'), null);
    assert.equal(resolveAppAsset(root, 'hawa://app/index.html?debug=1'), null);
    assert.equal(resolveAppAsset(root, 'hawa://app/assets/escape.txt'), null);
    assert.equal(pathInside(root, outside), false);
  } finally {
    fs.rmSync(temp, { recursive: true, force: true });
  }
});

test('export requests are schema checked and filenames cannot escape the dialog folder', () => {
  assert.deepEqual(parseExportRequest({ kind: 'master' }), { kind: 'master' });
  assert.equal(safeSuggestedExportName({ kind: 'master', suggestedName: '../../voice.zip' }), 'voice.wav');
  assert.equal(
    safeSuggestedExportName({ kind: 'record_bundle', suggestedName: '..\\..\\record.wav' }),
    'record.zip',
  );
  assert.equal(safeSuggestedExportName({ kind: 'master', suggestedName: 'CON.wav' }), 'HawaVoClean-CON.wav');
  assert.throws(() => parseExportRequest({ kind: 'raw_directory' }), /Export kind/);
  assert.throws(() => parseExportRequest({ kind: 'master', suggestedName: 4 }), /must be a string/);
});

test('absolute-path guard rejects relative and NUL-bearing paths', () => {
  assert.equal(requireAbsolutePath(path.resolve('/tmp', 'voice.wav'), 'reveal'), path.resolve('/tmp', 'voice.wav'));
  assert.throws(() => requireAbsolutePath('../voice.wav', 'reveal'), /absolute path/);
  assert.throws(() => requireAbsolutePath(`${path.resolve('/tmp', 'voice')}\u0000.wav`, 'reveal'), /NUL/);
});

test('development engine override is structured JSON, never shell text', () => {
  assert.deepEqual(parseDevelopmentCommand('["uv","run","hawavoclean"]'), ['uv', 'run', 'hawavoclean']);
  assert.equal(parseDevelopmentCommand(undefined), null);
  assert.throws(() => parseDevelopmentCommand('uv run hawavoclean'), /JSON array/);
  assert.throws(() => parseDevelopmentCommand('["uv",4]'), /bounded strings/);
});

test('packaged engine ignores development overrides and stays under resources', () => {
  const spec = resolveEngineSpec(
    {
      packaged: true,
      resourcesPath: '/Applications/HawaVoClean.app/Contents/Resources',
      repositoryRoot: '/repo',
      userData: '/user-data',
      temp: '/tmp',
      platform: 'darwin',
    },
    '["evil"]',
  );
  assert.equal(spec.executable, '/Applications/HawaVoClean.app/Contents/Resources/engine/hawavoclean-engine');
  assert.deepEqual(spec.prefixArgs, []);
  assert.equal(spec.cwd, '/user-data');
});

test('packaged self-test is limited to an exact non-distributable full-engine proof marker', () => {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'hawa-desktop-proof-marker-'));
  const marker = path.join(temp, PACKAGE_PROVENANCE_NAME);
  const proof = {
    schema_version: 1,
    artifact_type: 'unsigned-macos-app-proof',
    distribution_eligible: false,
    product: 'hawavoclean',
    product_version: '3.3.0',
    source_revision: 'a'.repeat(40),
    target: 'macos-arm64',
    engine_mode: 'full',
    packaged_selftest_allowed: true,
    signing: { developer_id: false, notarized: false, stapled: false },
  };
  try {
    assert.equal(packagedProofSelfTestAllowed(temp, true), false);
    fs.writeFileSync(marker, JSON.stringify(proof));
    assert.equal(packagedProofSelfTestAllowed(temp, false), false);
    assert.equal(packagedProofSelfTestAllowed(temp, true), true);

    fs.writeFileSync(marker, JSON.stringify({ ...proof, distribution_eligible: true }));
    assert.equal(packagedProofSelfTestAllowed(temp, true), false);
    fs.writeFileSync(marker, JSON.stringify({ ...proof, signing: { developer_id: true } }));
    assert.equal(packagedProofSelfTestAllowed(temp, true), false);
    fs.writeFileSync(marker, JSON.stringify({ ...proof, engine_mode: 'shell-only' }));
    assert.equal(packagedProofSelfTestAllowed(temp, true), false);

    fs.rmSync(marker);
    const realMarker = path.join(temp, 'real-marker.json');
    fs.writeFileSync(realMarker, JSON.stringify(proof));
    fs.symlinkSync(realMarker, marker);
    assert.equal(packagedProofSelfTestAllowed(temp, true), false);
  } finally {
    fs.rmSync(temp, { recursive: true, force: true });
  }
});

test('desktop bootstrap secret uses a one-shot stdin pipe, never argv', () => {
  const engine = fs.readFileSync(path.join(__dirname, '..', 'dist', 'engine.js'), 'utf8');
  assert.match(engine, /'--token-stdin'/);
  assert.doesNotMatch(engine, /'--token',\s*this\.#token/);
  assert.match(engine, /stdio:\s*\['pipe', 'pipe', 'pipe'\]/);
  assert.match(engine, /stdin\?\.end\(`\$\{this\.#token\}\\n`\)/);
  assert.match(engine, /HAWAVOCLEAN_PARENT_PID:\s*String\(process\.pid\)/);
});

test('desktop fuses require ASAR integrity without an unavailable custom V8 snapshot', () => {
  const afterPack = fs.readFileSync(path.join(__dirname, '..', 'scripts', 'after-pack.cjs'), 'utf8');
  assert.match(afterPack, /EnableEmbeddedAsarIntegrityValidation\]: true/);
  assert.match(afterPack, /OnlyLoadAppFromAsar\]: true/);
  assert.match(afterPack, /RunAsNode\]: false/);
  assert.match(afterPack, /LoadBrowserProcessSpecificV8Snapshot\]: false/);
});

test('session partition is in-memory and non-persistent to eliminate disk caching', () => {
  assert.equal(typeof SESSION_PARTITION, 'string');
  assert.equal(SESSION_PARTITION, 'hawavoclean-desktop');
  assert.ok(!SESSION_PARTITION.startsWith('persist:'), 'partition must not use persist: prefix');
  const mainCode = fs.readFileSync(path.join(__dirname, '..', 'dist', 'main.js'), 'utf8');
  assert.doesNotMatch(mainCode, /persist:hawavoclean/, 'main process must not use persistent session partition');
});

test('safe local data clearing purges session and partition caches while strictly preserving user-exported masters and engine jobs database', async () => {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'hawa-clear-test-'));
  const userData = path.join(temp, 'user-data');
  const documents = path.join(temp, 'documents');
  const partitions = path.join(userData, 'Partitions', 'hawavoclean-desktop');
  fs.mkdirSync(partitions, { recursive: true });
  fs.mkdirSync(documents, { recursive: true });

  // Simulate cached session files
  const cacheFile = path.join(partitions, 'Cache_Data_0');
  fs.writeFileSync(cacheFile, 'transient-cache-data');

  // Simulate persistent engine database that MUST be retained
  const jobsDb = path.join(userData, 'jobs.db');
  fs.writeFileSync(jobsDb, 'sqlite-header-jobs-db');

  // Simulate exported master WAV and Processing Record that MUST be retained
  const exportedMaster = path.join(documents, 'Master-Output.wav');
  fs.writeFileSync(exportedMaster, 'RIFF-master-audio-bytes');
  const exportedRecord = path.join(documents, 'Processing-Record.zip');
  fs.writeFileSync(exportedRecord, 'PK-zip-record-bytes');

  let cacheCleared = false;
  let storageDataCleared = false;
  let codeCacheCleared = false;

  const sessionClearers = {
    clearCache: async () => { cacheCleared = true; },
    clearStorageData: async () => { storageDataCleared = true; },
    clearCodeCaches: async () => { codeCacheCleared = true; },
  };

  try {
    const result = await safeClearSessionStorage(userData, sessionClearers);
    assert.equal(result.ok, true);
    assert.equal(cacheCleared, true);
    assert.equal(storageDataCleared, true);
    assert.equal(codeCacheCleared, true);

    // Legacy partition cache must be deleted
    assert.equal(fs.existsSync(cacheFile), false);
    assert.equal(fs.existsSync(partitions), false);

    // CRITICAL: User-exported masters and engine jobs database must survive completely intact
    assert.equal(fs.existsSync(jobsDb), true);
    assert.equal(fs.readFileSync(jobsDb, 'utf8'), 'sqlite-header-jobs-db');
    assert.equal(fs.existsSync(exportedMaster), true);
    assert.equal(fs.readFileSync(exportedMaster, 'utf8'), 'RIFF-master-audio-bytes');
    assert.equal(fs.existsSync(exportedRecord), true);
    assert.equal(fs.readFileSync(exportedRecord, 'utf8'), 'PK-zip-record-bytes');

    // Retained and cleared item contracts must match privacy documentation
    assert.deepEqual([...result.retainedItems], [
      'exported_wav_masters',
      'exported_processing_records',
      'user_source_media',
      'engine_jobs_database',
    ]);
    assert.ok(result.clearedItems.includes('session_http_cache'));
    assert.ok(result.clearedItems.includes('session_storage_data'));
    assert.ok(result.clearedItems.includes('session_code_cache'));
    assert.ok(result.clearedItems.includes('legacy_partition_hawavoclean-desktop'));
  } finally {
    fs.rmSync(temp, { recursive: true, force: true });
  }
});
