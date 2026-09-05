'use strict';

const assert = require('node:assert/strict');
const os = require('node:os');
const test = require('node:test');

const {
  DiagnosticsManager,
  redactPath,
  redactSecrets,
  sanitizeMessage,
  sanitizeStack,
} = require('../dist/diagnostics.js');

test('diagnostics manager starts opted out by default with no telemetry egress', () => {
  const manager = new DiagnosticsManager();
  const state = manager.getState();

  assert.equal(state.optIn, false);
  assert.equal(state.status, 'ready');
  assert.equal(state.telemetryEgress, 'none');
  assert.equal(state.pendingErrorCount, 0);
  assert.equal(Object.isFrozen(state), true);
});

test('diagnostics manager strictly drops errors when user has not opted in', () => {
  const manager = new DiagnosticsManager();
  manager.recordError('TEST_ERR', new Error('Something failed at /Users/hawzhin/secret/file.ts'));

  const state = manager.getState();
  assert.equal(state.pendingErrorCount, 0);
  assert.throws(() => manager.generateReport(), /opt-in required/i);
});

test('diagnostics manager records and sanitizes errors when opted in', () => {
  const manager = new DiagnosticsManager({ initialOptIn: true });
  assert.equal(manager.getState().optIn, true);

  manager.recordError(
    'NETWORK_FAILURE',
    new Error(`Failed request with Bearer secret-token-12345 at /Users/hawzhin/dev/app.ts`),
  );

  const state = manager.getState();
  assert.equal(state.pendingErrorCount, 1);

  const report = manager.generateReport({
    appName: 'HawaVoClean',
    appVersion: '3.3.0',
    packaged: true,
    engineStatus: 'running',
    enginePid: 12345,
  });

  assert.equal(report.schemaVersion, '1.0.0');
  assert.equal(report.optInConfirmed, true);
  assert.equal(report.app.name, 'HawaVoClean');
  assert.equal(report.app.version, '3.3.0');
  assert.equal(report.engine.status, 'running');
  assert.equal(report.engine.pid, 12345);
  assert.equal(report.recentErrors.length, 1);

  const error = report.recentErrors[0];
  assert.equal(error.code, 'NETWORK_FAILURE');
  assert.ok(error.message.includes('[REDACTED_SECRET]'));
  assert.ok(!error.message.includes('secret-token-12345'));
  assert.ok(!error.message.includes('/Users/hawzhin'));
  assert.ok(error.message.includes('[USER_HOME]'));
});

test('withdrawing diagnostics consent immediately wipes all recorded errors', () => {
  const manager = new DiagnosticsManager({ initialOptIn: true });
  manager.recordError('ERR1', 'Some error');
  manager.recordError('ERR2', 'Another error');
  assert.equal(manager.getState().pendingErrorCount, 2);

  // User revokes consent
  const state = manager.setOptIn(false);
  assert.equal(state.optIn, false);
  assert.equal(state.pendingErrorCount, 0);

  // Even re-checking state verifies complete wipe
  assert.equal(manager.getState().pendingErrorCount, 0);
  assert.throws(() => manager.generateReport(), /opt-in required/i);
});

test('redactPath removes current user homedir and Unix/Windows user paths', () => {
  const homedir = os.homedir();
  const sample1 = `${homedir}/Library/Application Support/HawaVoClean/models/checkpoint.pt`;
  const redacted1 = redactPath(sample1);
  assert.ok(!redacted1.includes(homedir));
  assert.ok(redacted1.startsWith('[USER_HOME]'));

  const sampleUnix = 'file loaded from /Users/alice/Music/recording.wav';
  assert.equal(redactPath(sampleUnix), 'file loaded from [USER_HOME]/Music/recording.wav');

  const sampleWin = 'failed reading C:\\Users\\Bob\\Desktop\\audio.wav';
  assert.equal(redactPath(sampleWin), 'failed reading [USER_HOME]\\Desktop\\audio.wav');
});

test('redactSecrets strips authorization headers and tokens', () => {
  const secretString = 'Authorization: Bearer secret_jwt_token_here and api_key="secret-12345"';
  const redacted = redactSecrets(secretString);
  assert.ok(!redacted.includes('secret_jwt_token_here'));
  assert.ok(!redacted.includes('secret-12345'));
  assert.ok(redacted.includes('[REDACTED_SECRET]'));
});

test('bounded error collection caps at 50 errors without unbounded memory leak', () => {
  const manager = new DiagnosticsManager({ initialOptIn: true });
  for (let i = 0; i < 60; i++) {
    manager.recordError(`ERR_${i}`, `Error number ${i}`);
  }
  const state = manager.getState();
  assert.equal(state.pendingErrorCount, 50);

  const report = manager.generateReport();
  assert.equal(report.recentErrors.length, 50);
  // Oldest errors 0..9 should have been shifted out
  assert.equal(report.recentErrors[0].code, 'ERR_10');
  assert.equal(report.recentErrors[49].code, 'ERR_59');
});
