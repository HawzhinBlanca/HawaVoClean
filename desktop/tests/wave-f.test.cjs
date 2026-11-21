'use strict';

const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const http = require('node:http');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const {
  AUDIO_EXTENSIONS,
  MASTER_EXPORT_FILTERS,
  PREFERENCES_FILENAME,
  RECORD_BUNDLE_EXPORT_FILTERS,
  extractExternalMediaPaths,
  loadUserPreferences,
  parseExportRequest,
  parseUserPreferences,
  redactDiagnosticText,
  requireSupportedMediaPath,
  safeClearSessionStorage,
  saveUserPreferences,
} = require('../dist/contracts.js');
const {
  ALLOWED_ENGINE_BROKER_TRANSITIONS,
  ENGINE_LOG_MAX_BYTES,
  ENGINE_LOG_MAX_ROTATIONS,
  EngineBroker,
  RotatingFileLogger,
  verifyBundledEngineIntegrity,
} = require('../dist/engine.js');

function sha256File(filePath) {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

function buildValidEngineFixture(dir, platform = 'darwin') {
  const launcherName = platform === 'win32' ? 'hawavoclean-engine.exe' : 'hawavoclean-engine';
  const manifestPlatform = platform === 'win32' ? 'win32' : 'macos-arm64';
  const launcherPath = path.join(dir, launcherName);
  const workerPath = path.join(dir, 'launcher.py');
  const manifestPath = path.join(dir, 'ENGINE-MANIFEST.json');
  const checksumsPath = path.join(dir, 'ENGINE-SHA256SUMS');

  fs.writeFileSync(launcherPath, platform === 'win32' ? 'MZ-fake-pe' : '#!/bin/sh\nexit 0\n', { mode: 0o755 });
  fs.writeFileSync(workerPath, 'print("ok")\n', 'utf8');
  fs.writeFileSync(
    manifestPath,
    JSON.stringify({
      bundle_schema_version: 1,
      artifact_type: 'resolve-engine-directory',
      version: '3.3.0',
      product_version: '3.3.0',
      launcher: launcherName,
      platform: manifestPlatform,
      arch: platform === 'win32' ? 'x64' : 'arm64',
    }, null, 2) + '\n',
    'utf8',
  );

  const lines = [
    `${sha256File(launcherPath)}  ./${launcherName}`,
    `${sha256File(workerPath)}  ./launcher.py`,
    `${sha256File(manifestPath)}  ./ENGINE-MANIFEST.json`,
  ];
  fs.writeFileSync(checksumsPath, lines.join('\n') + '\n', 'utf8');
  return { launcherPath, workerPath, manifestPath, checksumsPath, launcherName };
}

test('verifyBundledEngineIntegrity validates real manifest + SHA-256 sums for darwin and win32 and fails closed on tampering', () => {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'hawa-wavef-integrity-'));
  try {
    // 1. Valid macOS bundle passes
    const macDir = path.join(temp, 'mac');
    fs.mkdirSync(macDir);
    const mac = buildValidEngineFixture(macDir, 'darwin');
    const okMac = verifyBundledEngineIntegrity(macDir, 'darwin');
    assert.equal(okMac.valid, true);
    assert.equal(okMac.verifiedFiles, 3);
    assert.equal(okMac.manifestSha256, sha256File(mac.manifestPath));

    // 2. Valid Windows bundle passes
    const winDir = path.join(temp, 'win');
    fs.mkdirSync(winDir);
    buildValidEngineFixture(winDir, 'win32');
    const okWin = verifyBundledEngineIntegrity(winDir, 'win32');
    assert.equal(okWin.valid, true);
    assert.equal(okWin.verifiedFiles, 3);

    // 3. Wrong platform fails closed
    const badPlatform = verifyBundledEngineIntegrity(macDir, 'win32');
    assert.equal(badPlatform.valid, false);

    // 4. Tampered worker file fails with checksum mismatch
    fs.appendFileSync(mac.workerPath, '# tampered\n');
    const tampered = verifyBundledEngineIntegrity(macDir, 'darwin');
    assert.equal(tampered.valid, false);
    assert.match(tampered.error, /checksum mismatch/i);

    // 5. Missing listed file fails closed
    const missingDir = path.join(temp, 'missing');
    fs.mkdirSync(missingDir);
    const missing = buildValidEngineFixture(missingDir, 'darwin');
    fs.unlinkSync(missing.workerPath);
    const missingRes = verifyBundledEngineIntegrity(missingDir, 'darwin');
    assert.equal(missingRes.valid, false);
    assert.match(missingRes.error, /missing/i);

    // 6. Path traversal in ENGINE-SHA256SUMS fails closed
    const travDir = path.join(temp, 'traversal');
    fs.mkdirSync(travDir);
    const trav = buildValidEngineFixture(travDir, 'darwin');
    const outsideFile = path.join(temp, 'outside.txt');
    fs.writeFileSync(outsideFile, 'outside', 'utf8');
    fs.appendFileSync(trav.checksumsPath, `${sha256File(outsideFile)}  ./../outside.txt\n`);
    const travRes = verifyBundledEngineIntegrity(travDir, 'darwin');
    assert.equal(travRes.valid, false);
    assert.match(travRes.error, /traversal|unsafe/i);

    // 7. Missing launcher in ENGINE-SHA256SUMS fails closed
    const noLauncherDir = path.join(temp, 'no-launcher-sum');
    fs.mkdirSync(noLauncherDir);
    const noLauncher = buildValidEngineFixture(noLauncherDir, 'darwin');
    fs.writeFileSync(noLauncher.checksumsPath, `${sha256File(noLauncher.manifestPath)}  ./ENGINE-MANIFEST.json\n`);
    const noLauncherRes = verifyBundledEngineIntegrity(noLauncherDir, 'darwin');
    assert.equal(noLauncherRes.valid, false);
    assert.match(noLauncherRes.error, /does not cover launcher/i);

    // 8. Invalid manifest launcher or version fails closed
    const badManifestDir = path.join(temp, 'bad-manifest');
    fs.mkdirSync(badManifestDir);
    const badM = buildValidEngineFixture(badManifestDir, 'darwin');
    fs.writeFileSync(
      badM.manifestPath,
      JSON.stringify({ version: '3.3.0', launcher: 'wrong-bin', platform: 'macos-arm64' }),
    );
    const badManifestRes = verifyBundledEngineIntegrity(badManifestDir, 'darwin');
    assert.equal(badManifestRes.valid, false);
    assert.match(badManifestRes.error, /launcher mismatch/i);
  } finally {
    fs.rmSync(temp, { recursive: true, force: true });
  }
});

test('entitlements.mac.plist files are byte-identical and electron-builder declares audio/video fileAssociations', () => {
  const root = path.resolve(__dirname, '..');
  const rootPlist = fs.readFileSync(path.join(root, 'entitlements.mac.plist'), 'utf8');
  const buildPlist = fs.readFileSync(path.join(root, 'build', 'entitlements.mac.plist'), 'utf8');
  assert.equal(rootPlist, buildPlist, 'desktop/entitlements.mac.plist and desktop/build/entitlements.mac.plist must stay in exact sync');

  const config = require(path.join(root, 'electron-builder.config.cjs'));
  assert.ok(Array.isArray(config.fileAssociations), 'electron-builder must declare fileAssociations');
  const exts = config.fileAssociations.map((entry) => entry.ext);
  for (const required of ['wav', 'm4a', 'mp3', 'flac', 'aiff', 'mp4', 'mov']) {
    assert.ok(exts.includes(required), `fileAssociations must include .${required}`);
  }
});

test('EngineBroker auto-recovers from unexpected child crash and transitions to failed after backoff cap', async () => {
  assert.ok(ALLOWED_ENGINE_BROKER_TRANSITIONS.ready.includes('restarting'));
  assert.ok(ALLOWED_ENGINE_BROKER_TRANSITIONS.restarting.includes('starting'));
  assert.deepEqual([...ALLOWED_ENGINE_BROKER_TRANSITIONS.stopped], []);

  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'hawa-wavef-recovery-'));
  const prevOverride = process.env.HAWAVOCLEAN_DESKTOP_ENGINE_COMMAND;
  try {
    // 1. Engine crashes once after becoming ready, then stays healthy on restart.
    const counterFile = path.join(temp, 'runs.txt');
    const crashOnceScript = path.join(temp, 'crash-once-engine.js');
    fs.writeFileSync(
      crashOnceScript,
      `
const fs = require('node:fs');
const http = require('node:http');
let token = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (c) => { token += c; });
const count = fs.existsSync(${JSON.stringify(counterFile)})
  ? Number(fs.readFileSync(${JSON.stringify(counterFile)}, 'utf8')) + 1
  : 1;
fs.writeFileSync(${JSON.stringify(counterFile)}, String(count));
const server = http.createServer((req, res) => {
  if (req.method === 'POST' && req.url === '/api/session') {
    res.writeHead(200, { 'Cache-Control': 'no-store', 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ tokenType: 'Bearer', sessionToken: 's'.repeat(43), expiresInSeconds: 600 }));
    return;
  }
  if (req.method === 'POST' && req.url === '/api/shutdown') {
    res.writeHead(200);
    res.end('ok');
    server.close(() => process.exit(0));
    return;
  }
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ ok: true, run: count }));
});
server.listen(0, '127.0.0.1', () => {
  const { port } = server.address();
  process.stdout.write(JSON.stringify({ event: 'ready', port }) + '\\n');
  if (count === 1) {
    setTimeout(() => process.exit(139), 60);
  }
});
`,
      'utf8',
    );

    const phases = [];
    process.env.HAWAVOCLEAN_DESKTOP_ENGINE_COMMAND = JSON.stringify([process.execPath, crashOnceScript]);
    const broker = new EngineBroker(
      {
        packaged: false,
        platform: process.platform,
        resourcesPath: temp,
        repositoryRoot: temp,
        userData: path.join(temp, 'ud-recover'),
        temp,
      },
      {
        maxRestartAttempts: 3,
        restartBaseDelayMs: 25,
        healthyResetMs: 60_000,
        onPhaseChange: (phase) => phases.push(phase),
      },
    );

    const firstEndpoint = await broker.endpoint();
    assert.equal(broker.phase, 'ready');

    // Wait for crash -> restarting -> starting -> ready
    const recoverDeadline = Date.now() + 5000;
    while (Date.now() < recoverDeadline) {
      if (broker.phase === 'ready' && broker.restartCount === 1 && broker.origin !== firstEndpoint.baseUrl) {
        break;
      }
      await new Promise((r) => setTimeout(r, 25));
    }
    assert.equal(broker.phase, 'ready');
    assert.equal(broker.restartCount, 1);
    assert.ok(phases.includes('restarting'), `expected restarting phase in ${JSON.stringify(phases)}`);
    const recoveredEndpoint = await broker.endpoint();
    assert.notEqual(recoveredEndpoint.baseUrl, firstEndpoint.baseUrl);

    // Explicit stop still transitions cleanly to stopped.
    await broker.stop();
    assert.equal(broker.phase, 'stopped');
    assert.ok(fs.existsSync(broker.logPath), 'engine rotating log file must be written');

    // 2. Repeated crashes hit the 3-attempt cap and transition cleanly to failed (never stopped).
    const alwaysCrashScript = path.join(temp, 'always-crash-engine.js');
    fs.writeFileSync(
      alwaysCrashScript,
      `
const http = require('node:http');
process.stdin.resume();
const server = http.createServer((_req, res) => { res.writeHead(200); res.end('ok'); });
server.listen(0, '127.0.0.1', () => {
  const { port } = server.address();
  process.stdout.write(JSON.stringify({ event: 'ready', port }) + '\\n');
  setTimeout(() => process.exit(42), 30);
});
`,
      'utf8',
    );
    process.env.HAWAVOCLEAN_DESKTOP_ENGINE_COMMAND = JSON.stringify([process.execPath, alwaysCrashScript]);
    const failingBroker = new EngineBroker(
      {
        packaged: false,
        platform: process.platform,
        resourcesPath: temp,
        repositoryRoot: temp,
        userData: path.join(temp, 'ud-exhaust'),
        temp,
      },
      {
        maxRestartAttempts: 3,
        restartBaseDelayMs: 20,
        healthyResetMs: 60_000,
      },
    );
    failingBroker.start();
    const exhaustDeadline = Date.now() + 5000;
    while (failingBroker.phase !== 'failed' && Date.now() < exhaustDeadline) {
      await new Promise((r) => setTimeout(r, 25));
    }
    assert.equal(failingBroker.phase, 'failed');
    assert.equal(failingBroker.restartCount, 3);
    await assert.rejects(() => failingBroker.endpoint(), /maximum restart attempts/);
    await failingBroker.stop();
    assert.equal(failingBroker.phase, 'stopped');
  } finally {
    if (prevOverride === undefined) delete process.env.HAWAVOCLEAN_DESKTOP_ENGINE_COMMAND;
    else process.env.HAWAVOCLEAN_DESKTOP_ENGINE_COMMAND = prevOverride;
    fs.rmSync(temp, { recursive: true, force: true });
  }
});

test('RotatingFileLogger enforces size bounds, max 3 rotations, 0600 permissions, and secret redaction', () => {
  assert.equal(ENGINE_LOG_MAX_BYTES, 5 * 1024 * 1024);
  assert.equal(ENGINE_LOG_MAX_ROTATIONS, 3);

  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'hawa-wavef-log-'));
  try {
    const logPath = path.join(temp, 'logs', 'engine.log');
    const logger = new RotatingFileLogger(logPath, 512, 3);
    for (let i = 0; i < 40; i += 1) {
      logger.append(
        `line-${String(i).padStart(2, '0')} Authorization: Bearer ${'A'.repeat(40)} X-Hawa-Token: ${'B'.repeat(32)} /Users/alice/audio.wav`,
      );
    }
    assert.equal(fs.existsSync(logPath), true);
    assert.equal(fs.existsSync(`${logPath}.1`), true);
    assert.equal(fs.existsSync(`${logPath}.2`), true);
    assert.equal(fs.existsSync(`${logPath}.3`), true);
    assert.equal(fs.existsSync(`${logPath}.4`), false, 'rotations must be bounded at 3');

    if (process.platform !== 'win32') {
      assert.equal(fs.statSync(logPath).mode & 0o777, 0o600);
    }
    const recent = logger.readRecentLines(50).join('\n');
    assert.doesNotMatch(recent, /AAAAAAAAAAAAAAAA/);
    assert.doesNotMatch(recent, /BBBBBBBBBBBBBBBB/);
    assert.doesNotMatch(recent, /\/Users\/alice\//);
    assert.match(recent, /Bearer \[REDACTED\]/);
    assert.match(recent, /X-Hawa-Token: \[REDACTED\]/);
  } finally {
    fs.rmSync(temp, { recursive: true, force: true });
  }
});

test('extended formats, export filters, sourceId validation, and open-with argv parsing behave cleanly', () => {
  for (const ext of ['mov', 'aac', 'ogg', 'opus', 'webm', 'mxf']) {
    assert.ok(AUDIO_EXTENSIONS.includes(ext), `AUDIO_EXTENSIONS must include ${ext}`);
    assert.equal(requireSupportedMediaPath(`/tmp/clip.${ext}`, 'test'), `/tmp/clip.${ext}`);
  }
  assert.deepEqual(
    MASTER_EXPORT_FILTERS.map((f) => [...f.extensions]),
    [['wav'], ['flac'], ['aac', 'm4a'], ['mp4']],
  );
  assert.deepEqual(
    RECORD_BUNDLE_EXPORT_FILTERS.map((f) => [...f.extensions]),
    [['zip']],
  );

  const validId = '0123456789abcdef0123456789abcdef';
  assert.deepEqual(
    parseExportRequest({ kind: 'master', suggestedName: 'clean.wav', sourceId: validId }),
    { kind: 'master', suggestedName: 'clean.wav', sourceId: validId },
  );
  assert.throws(
    () => parseExportRequest({ kind: 'master', sourceId: 'not-a-valid-hex-id' }),
    /sourceId/,
  );

  const extracted = extractExternalMediaPaths([
    '--user-data-dir=/tmp/profile',
    '.',
    '/tmp/interview.mov',
    '/tmp/podcast.opus',
    '/tmp/notes.txt',
    '/tmp/interview.mov',
  ]);
  assert.deepEqual(extracted, ['/tmp/interview.mov', '/tmp/podcast.opus']);

  const scrubbed = redactDiagnosticText('Bearer secret_token_value_123456 at /Users/bob/Desktop/clip.wav');
  assert.match(scrubbed, /Bearer \[REDACTED\]/);
  assert.doesNotMatch(scrubbed, /secret_token_value/);
  assert.doesNotMatch(scrubbed, /\/Users\/bob\//);
});

test('user preferences persist atomically at 0600, reject invalid input, and survive safeClearSessionStorage', async () => {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'hawa-wavef-prefs-'));
  try {
    assert.deepEqual(loadUserPreferences(temp), {});
    const saved1 = saveUserPreferences(temp, {
      workspaceMode: 'pro',
      locale: 'ckb',
      livePreviewEnabled: true,
      windowBounds: { x: 100, y: 120, width: 1440, height: 900 },
      ignoredUnknownSecret: 'must-not-be-saved',
    });
    assert.deepEqual(saved1, {
      workspaceMode: 'pro',
      locale: 'ckb',
      livePreviewEnabled: true,
      windowBounds: { x: 100, y: 120, width: 1440, height: 900 },
    });

    const prefsFile = path.join(temp, PREFERENCES_FILENAME);
    assert.equal(fs.existsSync(prefsFile), true);
    if (process.platform !== 'win32') {
      assert.equal(fs.statSync(prefsFile).mode & 0o777, 0o600);
    }
    const rawDisk = fs.readFileSync(prefsFile, 'utf8');
    assert.doesNotMatch(rawDisk, /ignoredUnknownSecret|must-not-be-saved/);

    // Partial update merges cleanly
    const saved2 = saveUserPreferences(temp, { livePreviewEnabled: false });
    assert.equal(saved2.workspaceMode, 'pro');
    assert.equal(saved2.locale, 'ckb');
    assert.equal(saved2.livePreviewEnabled, false);

    // Survives session storage clearing
    await safeClearSessionStorage(temp);
    assert.deepEqual(loadUserPreferences(temp), saved2);

    // Invalid inputs throw
    assert.throws(() => parseUserPreferences({ workspaceMode: 'invalid' }), /workspaceMode/);
    assert.throws(() => parseUserPreferences({ locale: '../../../etc' }), /locale/);
    assert.throws(() => parseUserPreferences({ livePreviewEnabled: 'yes' }), /boolean/);
  } finally {
    fs.rmSync(temp, { recursive: true, force: true });
  }
});

test('main process awaits session clearing, registers renderer crash/unresponsive handlers, native menu, and open-file integration', () => {
  const compiledMain = fs.readFileSync(path.join(__dirname, '..', 'dist', 'main.js'), 'utf8');
  assert.match(compiledMain, /await\s+(?:\(0,\s*contracts_js_1\.)?safeClearSessionStorage/);
  assert.doesNotMatch(compiledMain, /void\s+(?:\(0,\s*contracts_js_1\.)?safeClearSessionStorage/);
  assert.match(compiledMain, /await\s+configureSession\(\)/);
  assert.match(compiledMain, /render-process-gone/);
  assert.match(compiledMain, /unresponsive/);
  assert.match(compiledMain, /Open Audio\/Video/);
  assert.match(compiledMain, /Export Redacted Diagnostics/);
  assert.match(compiledMain, /open-file/);
  assert.match(compiledMain, /second-instance/);
});
