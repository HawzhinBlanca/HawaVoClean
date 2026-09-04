'use strict';

const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');

function sha256(filePath) {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

function validateEnginePayload(engineDir, platform) {
  if (!fs.existsSync(engineDir) || !fs.statSync(engineDir).isDirectory()) {
    throw new Error(`Engine directory does not exist or is not a directory: ${engineDir}`);
  }

  // 1. Reject placeholders
  const placeholder = path.join(engineDir, 'README.txt');
  if (fs.existsSync(placeholder)) {
    throw new Error(`Archive validation rejected placeholder engine resource: ${placeholder}`);
  }

  // 2. Required launcher executable
  const executableName = platform === 'win32' ? 'hawavoclean-engine.exe' : 'hawavoclean-engine';
  const launcher = path.join(engineDir, executableName);
  if (!fs.existsSync(launcher) || !fs.statSync(launcher).isFile()) {
    throw new Error(`Engine launcher executable is missing: ${launcher}`);
  }
  if (platform === 'darwin') {
    try {
      fs.accessSync(launcher, fs.constants.X_OK);
    } catch {
      throw new Error(`Engine launcher is not executable: ${launcher}`);
    }
  }

  // 3. Required manifests
  const manifestPath = path.join(engineDir, 'ENGINE-MANIFEST.json');
  const checksumsPath = path.join(engineDir, 'ENGINE-SHA256SUMS');
  const symlinksPath = path.join(engineDir, 'ENGINE-SYMLINKS');
  for (const [file, label] of [
    [manifestPath, 'ENGINE-MANIFEST.json'],
    [checksumsPath, 'ENGINE-SHA256SUMS'],
    [symlinksPath, 'ENGINE-SYMLINKS'],
  ]) {
    if (!fs.existsSync(file) || !fs.statSync(file).isFile()) {
      throw new Error(`Engine manifest is missing: ${label}`);
    }
  }

  // 4. Validate manifest metadata
  let manifest;
  try {
    manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  } catch (err) {
    throw new Error(`ENGINE-MANIFEST.json is not valid JSON: ${err.message}`);
  }
  if (
    manifest.bundle_schema_version !== 1 ||
    manifest.artifact_type !== 'resolve-engine-directory' ||
    manifest.product_version !== '3.3.0'
  ) {
    throw new Error('ENGINE-MANIFEST.json does not conform to the HawaVoClean engine contract.');
  }

  // 5. Verify checksum inventory
  const lines = fs.readFileSync(checksumsPath, 'utf8').split('\n').filter(Boolean);
  const checkedFiles = new Set();
  for (const line of lines) {
    const parts = line.split(/\s+/);
    if (parts.length < 2) continue;
    const expectedHash = parts[0].toLowerCase();
    const relativePath = parts[1].replace(/^\.\//, '');
    const fullPath = path.join(engineDir, relativePath);
    if (!fs.existsSync(fullPath) || !fs.statSync(fullPath).isFile()) {
      throw new Error(`Engine payload file missing from checksum inventory: ${relativePath}`);
    }
    const actualHash = sha256(fullPath);
    if (actualHash !== expectedHash) {
      throw new Error(
        `Engine payload checksum mismatch for ${relativePath}: expected ${expectedHash}, got ${actualHash}`,
      );
    }
    checkedFiles.add(relativePath);
  }

  return {
    verifiedFiles: checkedFiles.size,
    launcherHash: sha256(launcher),
    manifestHash: sha256(manifestPath),
  };
}

module.exports = { validateEnginePayload };

if (require.main === module) {
  const args = process.argv.slice(2);
  const targetDir = args[0] || path.join(__dirname, '..', 'resources', 'engine', `${process.platform}-${process.arch}`);
  try {
    const result = validateEnginePayload(targetDir, process.platform);
    console.log(`Engine payload at ${targetDir} verified successfully: ${result.verifiedFiles} files intact.`);
  } catch (err) {
    console.error(`Engine validation error: ${err.message}`);
    process.exit(1);
  }
}
