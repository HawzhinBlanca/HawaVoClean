'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawn } = require('node:child_process');

const RESULT_PREFIX = 'HAWA_DESKTOP_SELFTEST_RESULT ';

function assertSelfTestResult(result, expectedPackaged) {
  assert.equal(result.host, 'electron');
  assert.equal(result.healthStatus, 200);
  assert.ok(Number.isInteger(result.enginePid) && result.enginePid > 0, 'engine PID is missing');
  assert.equal(result.authenticatedPostStatus, 400);
  assert.equal(result.authenticatedMediaStatus, 400);
  assert.equal(result.mediaNoStore, true);
  assert.ok(result.sessionClearing && result.sessionClearing.ok === true);
  assert.ok(result.sessionClearing.clearedItems.includes('session_http_cache'));
  assert.ok(result.sessionClearing.retainedItems.includes('exported_wav_masters'));
  assert.ok(result.sessionClearing.retainedItems.includes('engine_jobs_database'));
  assert.deepEqual(result.endpointKeys, ['baseUrl']);
  assert.equal(result.endpointHasSecret, false);
  assert.equal(result.endpointUrlHasAuth, false);
  assert.equal(result.remoteBlocked, true);
  assert.equal(result.popupBlocked, true);
  assert.equal(result.nodeHidden, true);
  assert.deepEqual(result.topLevelKeys, ['app', 'diagnostics', 'engine', 'files', 'host', 'session', 'updates']);
  assert.equal(result.app.name, 'HawaVoClean');
  assert.equal(result.app.version, '3.3.0');
  assert.equal(result.app.packaged, expectedPackaged);
  assert.deepEqual(result.renderer, {
    contract: 'hawavoclean-react-v1',
    challengeVerified: true,
    uiVersion: result.app.version,
    domContract: 'hawavoclean-react-v1',
    domUiVersion: result.app.version,
    rootChildCount: 1,
    shellIsOnlyRootChild: true,
    shellTag: 'DIV',
    shellClass: 'app',
    missingSelectors: [],
    brandText: `HAWAVOCLEANv${result.app.version}`,
    openFileText: 'Open file…',
    processLabel: 'PROCESS',
    display: 'grid',
    width: result.renderer.width,
    height: result.renderer.height,
    stylesheetCount: result.renderer.stylesheetCount,
    moduleScriptCount: result.renderer.moduleScriptCount,
    documentTitle: 'HawaVoClean',
  });
  assert.ok(Number.isFinite(result.renderer.width) && result.renderer.width >= 960);
  assert.ok(Number.isFinite(result.renderer.height) && result.renderer.height >= 640);
  assert.ok(
    Number.isInteger(result.renderer.stylesheetCount) && result.renderer.stylesheetCount >= 2,
    'production renderer stylesheets did not load',
  );
  assert.ok(
    Number.isInteger(result.renderer.moduleScriptCount) && result.renderer.moduleScriptCount >= 1,
    'production renderer module did not load',
  );
  assert.deepEqual(result.diagnostics, { status: 'unavailable', reason: 'not_implemented' });
  assert.deepEqual(result.updates, {
    status: 'disabled',
    reason: 'release_feed_not_configured',
    canCheck: false,
  });
}

async function runDesktopSelfTest({ executable, args, cwd, expectedPackaged, label }) {
  const userData = fs.mkdtempSync(path.join(os.tmpdir(), 'hawavoclean-desktop-selftest-'));
  try {
    await new Promise((resolve, reject) => {
      const child = spawn(executable, [...args, `--user-data-dir=${userData}`], {
        cwd,
        env: { ...process.env, HAWAVOCLEAN_DESKTOP_SELFTEST: '1' },
        stdio: ['ignore', 'pipe', 'pipe'],
        windowsHide: true,
      });
      let stdout = '';
      let stderr = '';
      let timedOut = false;
      const appendBounded = (current, chunk) => (current + chunk).slice(-1_000_000);
      child.stdout.setEncoding('utf8');
      child.stderr.setEncoding('utf8');
      child.stdout.on('data', (chunk) => {
        stdout = appendBounded(stdout, chunk);
      });
      child.stderr.on('data', (chunk) => {
        stderr = appendBounded(stderr, chunk);
      });
      const deadline = setTimeout(() => {
        timedOut = true;
        child.kill('SIGKILL');
      }, 90_000);
      child.once('error', (error) => {
        clearTimeout(deadline);
        reject(error);
      });
      child.once('exit', (code, signal) => {
        clearTimeout(deadline);
        try {
          if (timedOut) throw new Error(`${label} timed out after 90 seconds.`);
          const line = stdout.split(/\r?\n/).find((value) => value.startsWith(RESULT_PREFIX));
          if (!line) {
            throw new Error(
              `${label} produced no result (code ${String(code)}, signal ${String(signal)}).\n${stderr}`,
            );
          }
          assert.equal(code, 0, stderr);
          assertSelfTestResult(JSON.parse(line.slice(RESULT_PREFIX.length)), expectedPackaged);
          resolve();
        } catch (error) {
          reject(error);
        }
      });
    });
    console.log(`${label} passed`);
  } finally {
    fs.rmSync(userData, { recursive: true, force: true });
  }
}

module.exports = { assertSelfTestResult, runDesktopSelfTest };
