'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const {
  verifyFileSecurity,
  readActiveRendezvous,
  discoverDesktopEngine,
  verifyEngineExecutable,
} = require('../com.hawavoclean.resolve/engine-discovery.js');

test('verifyFileSecurity admits user-owned non-world-writable files and rejects insecure permissions', () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hawa-sec-test-'));
  const testFile = path.join(tempDir, 'secure.json');
  fs.writeFileSync(testFile, '{"test":true}', { mode: 0o600 });

  try {
    const sec1 = verifyFileSecurity(testFile);
    assert.equal(sec1.ok, true);

    if (process.platform !== 'win32') {
      // Insecure: world-writable
      fs.chmodSync(testFile, 0o666);
      const sec2 = verifyFileSecurity(testFile);
      assert.equal(sec2.ok, false);
      assert.equal(sec2.reason, 'insecure_permissions_group_or_world_writable');

      // Insecure: group-writable
      fs.chmodSync(testFile, 0o660);
      const sec3 = verifyFileSecurity(testFile);
      assert.equal(sec3.ok, false);
      assert.equal(sec3.reason, 'insecure_permissions_group_or_world_writable');
    }
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
});

test('readActiveRendezvous parses valid rendezvous and rejects untrusted origins or corrupted files', () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hawa-rendezvous-test-'));
  const rFile = path.join(tempDir, 'engine.json');

  try {
    // 1. Missing file returns null
    assert.equal(readActiveRendezvous(rFile), null);

    // 2. Valid rendezvous with current PID (which is running)
    const validData = {
      schemaVersion: 1,
      pid: process.pid,
      origin: 'http://127.0.0.1:48123',
      token: 'abcdef0123456789abcdef0123456789',
      appVersion: '3.3.0',
    };
    fs.writeFileSync(rFile, JSON.stringify(validData), { mode: 0o600 });
    const active = readActiveRendezvous(rFile);
    assert.ok(active);
    assert.equal(active.origin, 'http://127.0.0.1:48123');
    assert.equal(active.token, validData.token);
    assert.equal(active.pid, process.pid);

    // 3. Untrusted / foreign origin fails closed
    const badOrigin = { ...validData, origin: 'http://evil.domain.com:48123' };
    fs.writeFileSync(rFile, JSON.stringify(badOrigin), { mode: 0o600 });
    assert.throws(() => readActiveRendezvous(rFile), /invalid or non-loopback origin/);

    // 4. Localhost string (must be exact loopback 127.0.0.1) fails closed
    const localhostOrigin = { ...validData, origin: 'http://localhost:48123' };
    fs.writeFileSync(rFile, JSON.stringify(localhostOrigin), { mode: 0o600 });
    assert.throws(() => readActiveRendezvous(rFile), /invalid or non-loopback origin/);

    // 5. Stale PID returns null
    const deadPid = { ...validData, pid: 9999999 };
    fs.writeFileSync(rFile, JSON.stringify(deadPid), { mode: 0o600 });
    assert.equal(readActiveRendezvous(rFile), null);
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
});

test('discoverDesktopEngine prefers active rendezvous and falls back to desktop app binary', () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hawa-discovery-test-'));
  const rFile = path.join(tempDir, 'rendezvous.json');
  const fakeEngine = path.join(tempDir, 'hawavoclean-engine');

  fs.writeFileSync(fakeEngine, '#!/bin/sh\necho ok\n', { mode: 0o700 });

  try {
    // 1. Missing both rendezvous and desktop app
    const missing = discoverDesktopEngine({
      rendezvousPath: rFile,
      engineExecutable: path.join(tempDir, 'nonexistent'),
    });
    assert.equal(missing.type, 'missing');
    assert.equal(missing.error, 'missing_desktop_app');

    // 2. Discovers installed desktop app when no active rendezvous
    const installed = discoverDesktopEngine({
      rendezvousPath: rFile,
      engineExecutable: fakeEngine,
    });
    assert.equal(installed.type, 'installed_desktop_engine');
    assert.equal(installed.executable, fakeEngine);
    assert.deepEqual(installed.command, [fakeEngine, 'serve']);

    // 3. When active rendezvous exists, it takes precedence!
    fs.writeFileSync(
      rFile,
      JSON.stringify({
        schemaVersion: 1,
        pid: process.pid,
        origin: 'http://127.0.0.1:49999',
        token: '0123456789abcdef0123456789abcdef',
      }),
      { mode: 0o600 },
    );

    const active = discoverDesktopEngine({
      rendezvousPath: rFile,
      engineExecutable: fakeEngine,
    });
    assert.equal(active.type, 'active_rendezvous');
    assert.equal(active.origin, 'http://127.0.0.1:49999');
    assert.equal(active.token, '0123456789abcdef0123456789abcdef');
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
});
