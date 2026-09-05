'use strict';

const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const { exactEngineOrigin } = require('./session-auth.js');

const DEFAULT_MACOS_APP_PATH = '/Applications/HawaVoClean.app';
const DEFAULT_ENGINE_SUBPATH = path.join('Contents', 'Resources', 'engine', 'hawavoclean-engine');

/**
 * Resolves the OS-protected rendezvous directory path.
 * On macOS: ~/Library/Application Support/HawaVoClean/rendezvous/
 */
function getRendezvousDir() {
  if (process.env.HAWAVOCLEAN_RENDEZVOUS_DIR) {
    return path.resolve(process.env.HAWAVOCLEAN_RENDEZVOUS_DIR);
  }
  if (process.platform === 'darwin') {
    return path.join(os.homedir(), 'Library', 'Application Support', 'HawaVoClean', 'rendezvous');
  }
  if (process.platform === 'win32') {
    const appData = process.env.APPDATA || path.join(os.homedir(), 'AppData', 'Roaming');
    return path.join(appData, 'HawaVoClean', 'rendezvous');
  }
  return path.join(os.homedir(), '.config', 'hawavoclean', 'rendezvous');
}

function getRendezvousFilePath() {
  if (process.env.HAWAVOCLEAN_RENDEZVOUS_PATH) {
    return path.resolve(process.env.HAWAVOCLEAN_RENDEZVOUS_PATH);
  }
  return path.join(getRendezvousDir(), 'engine.json');
}

/**
 * Checks if a process with the given PID is currently active.
 */
function isPidRunning(pid) {
  if (typeof pid !== 'number' || !Number.isInteger(pid) || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch (err) {
    return err.code === 'EPERM';
  }
}

/**
 * Verifies that a file has secure OS permissions (user-only read/write on POSIX).
 * Fails closed if the file or parent directory is group-writable or world-writable.
 */
function verifyFileSecurity(filePath) {
  let stat;
  try {
    stat = fs.statSync(filePath);
  } catch {
    return { ok: false, reason: 'file_not_found' };
  }

  if (process.platform !== 'win32') {
    // Must be owned by current user or root
    if (stat.uid !== process.getuid() && stat.uid !== 0) {
      return { ok: false, reason: 'unauthorized_file_owner' };
    }
    // Must not be world-writable or group-writable (mode & 0o022 === 0)
    if ((stat.mode & 0o022) !== 0) {
      return { ok: false, reason: 'insecure_permissions_group_or_world_writable' };
    }
  }

  return { ok: true, stat };
}

/**
 * Reads and validates an active OS-protected rendezvous record.
 */
function readActiveRendezvous(rendezvousFile = getRendezvousFilePath()) {
  if (!fs.existsSync(rendezvousFile)) {
    return null;
  }

  const security = verifyFileSecurity(rendezvousFile);
  if (!security.ok) {
    throw new Error(`Rendezvous file security violation (${security.reason}): ${rendezvousFile}`);
  }

  let record;
  try {
    record = JSON.parse(fs.readFileSync(rendezvousFile, 'utf8'));
  } catch (err) {
    throw new Error(`Rendezvous file is invalid JSON: ${err.message}`);
  }

  if (record.schemaVersion !== 1) {
    throw new Error(`Unsupported rendezvous schemaVersion: ${String(record.schemaVersion)}`);
  }

  // Validate exact loopback IPv4 origin
  const origin = exactEngineOrigin(record.origin);
  if (!origin) {
    throw new Error(`Rendezvous specifies invalid or non-loopback origin: ${String(record.origin)}`);
  }

  if (typeof record.token !== 'string' || record.token.length < 16) {
    throw new Error('Rendezvous token is missing or too short.');
  }

  // Verify PID is still alive
  if (!isPidRunning(record.pid)) {
    return null; // Stale rendezvous file
  }

  return {
    origin,
    token: record.token,
    pid: record.pid,
    executable: record.executable || null,
    version: record.appVersion || null,
  };
}

/**
 * Locates the desktop application's installed engine executable.
 */
function findDesktopEngineExecutable() {
  // 1. Explicit environment override for testing or custom install locations
  if (process.env.HAWAVOCLEAN_DESKTOP_ENGINE) {
    const candidate = path.resolve(process.env.HAWAVOCLEAN_DESKTOP_ENGINE);
    if (fs.existsSync(candidate)) return candidate;
  }

  // 2. Standard macOS /Applications installation
  const defaultMacEngine = path.join(DEFAULT_MACOS_APP_PATH, DEFAULT_ENGINE_SUBPATH);
  if (fs.existsSync(defaultMacEngine)) {
    return defaultMacEngine;
  }

  // 3. User Applications directory (~/Applications)
  const userMacEngine = path.join(os.homedir(), 'Applications', 'HawaVoClean.app', DEFAULT_ENGINE_SUBPATH);
  if (fs.existsSync(userMacEngine)) {
    return userMacEngine;
  }

  return null;
}

/**
 * Validates that an engine binary is safe and trusted to execute.
 */
function verifyEngineExecutable(executablePath) {
  const security = verifyFileSecurity(executablePath);
  if (!security.ok) {
    throw new Error(`Engine executable security verification failed: ${security.reason}`);
  }

  if (process.platform !== 'win32') {
    try {
      fs.accessSync(executablePath, fs.constants.X_OK);
    } catch {
      throw new Error(`Engine binary is not executable: ${executablePath}`);
    }
  }

  return true;
}

/**
 * Main discovery entrypoint: discovers an active desktop broker via OS-protected
 * rendezvous or locates the installed desktop engine binary.
 */
function discoverDesktopEngine(options = {}) {
  const rendezvousPath = options.rendezvousPath || getRendezvousFilePath();

  // 1. Try active OS-protected rendezvous
  try {
    const active = readActiveRendezvous(rendezvousPath);
    if (active) {
      return {
        type: 'active_rendezvous',
        origin: active.origin,
        token: active.token,
        pid: active.pid,
        executable: active.executable,
        version: active.version,
      };
    }
  } catch (err) {
    // If rendezvous file exists but failed security checks, fail closed immediately!
    throw new Error(`Refusing untrusted broker: ${err.message}`);
  }

  // 2. Locate installed Desktop App engine
  const executable = options.engineExecutable || findDesktopEngineExecutable();
  if (executable && fs.existsSync(executable)) {
    verifyEngineExecutable(executable);
    return {
      type: 'installed_desktop_engine',
      executable,
      command: [executable, 'serve'],
      cwd: os.homedir(),
      env: {
        PYTHONNOUSERSITE: '1',
        PYTHONDONTWRITEBYTECODE: '1',
      },
    };
  }

  // 3. Neither active broker nor installed desktop app was found
  return {
    type: 'missing',
    error: 'missing_desktop_app',
    message: 'HawaVoClean Desktop installation not found. Please install HawaVoClean.app in /Applications.',
  };
}

module.exports = {
  getRendezvousDir,
  getRendezvousFilePath,
  verifyFileSecurity,
  readActiveRendezvous,
  findDesktopEngineExecutable,
  verifyEngineExecutable,
  discoverDesktopEngine,
};
