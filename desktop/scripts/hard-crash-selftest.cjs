'use strict';

const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const electron = require('electron');

const RESULT_PREFIX = 'HAWA_DESKTOP_SELFTEST_RESULT ';
const DEADLINE_MS = 90_000;
const ORPHAN_DEADLINE_MS = 10_000;

function pidIsAlive(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    if (error && error.code === 'ESRCH') return false;
    throw error;
  }
}

function waitForExit(child) {
  if (child.exitCode !== null || child.signalCode !== null) return Promise.resolve();
  return new Promise((resolve) => child.once('exit', resolve));
}

async function waitForPidToDisappear(pid) {
  const deadline = Date.now() + ORPHAN_DEADLINE_MS;
  while (Date.now() < deadline) {
    if (!pidIsAlive(pid)) return;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  assert.fail(`engine ${pid} survived its Electron parent by more than ${ORPHAN_DEADLINE_MS} ms`);
}

async function main() {
  const child = spawn(electron, ['.'], {
    cwd: __dirname + '/..',
    env: {
      ...process.env,
      HAWAVOCLEAN_DESKTOP_SELFTEST: '1',
      HAWAVOCLEAN_DESKTOP_SELFTEST_HARD_CRASH: '1',
    },
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true,
  });
  let stdout = '';
  let stderr = '';
  let enginePid = null;
  let settled = false;
  const deadline = setTimeout(() => {
    if (!settled) child.kill('SIGKILL');
  }, DEADLINE_MS);

  try {
    const result = await new Promise((resolve, reject) => {
      child.once('error', reject);
      child.once('exit', (code, signal) => {
        if (!settled) {
          reject(
            new Error(
              `desktop exited before hard-crash evidence (code ${String(code)}, signal ${String(signal)}): ${stderr}`,
            ),
          );
        }
      });
      child.stderr.setEncoding('utf8');
      child.stderr.on('data', (chunk) => {
        stderr = (stderr + chunk).slice(-1_000_000);
      });
      child.stdout.setEncoding('utf8');
      child.stdout.on('data', (chunk) => {
        stdout = (stdout + chunk).slice(-1_000_000);
        const line = stdout.split(/\r?\n/).find((value) => value.startsWith(RESULT_PREFIX));
        if (!line || settled) return;
        settled = true;
        try {
          resolve(JSON.parse(line.slice(RESULT_PREFIX.length)));
        } catch (error) {
          reject(error);
        }
      });
    });
    assert.equal(result.healthStatus, 200);
    assert.ok(Number.isInteger(result.enginePid) && result.enginePid > 1, 'self-test omitted engine pid');
    enginePid = result.enginePid;

    // SIGKILL runs no Electron shutdown hook.  The engine must notice that
    // its declared direct parent disappeared and tear itself down.
    assert.equal(child.kill('SIGKILL'), true, 'could not hard-kill desktop test host');
    await waitForExit(child);
    await waitForPidToDisappear(enginePid);
    console.log(`desktop hard-crash self-test passed (engine ${enginePid} stopped)`);
  } finally {
    settled = true;
    clearTimeout(deadline);
    if (child.exitCode === null && child.signalCode === null) child.kill('SIGKILL');
    if (enginePid !== null && pidIsAlive(enginePid)) {
      // Test-owned emergency cleanup only; reaching this path is still a
      // failed watchdog assertion above.
      try { process.kill(enginePid, 'SIGKILL'); } catch {}
    }
  }
}

void main().catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.stack : String(error)}\n`);
  process.exitCode = 1;
});
