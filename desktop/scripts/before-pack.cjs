'use strict';

const fs = require('node:fs');
const path = require('node:path');

function requireEnv(names, message) {
  const missing = names.filter((name) => !process.env[name]);
  if (missing.length) throw new Error(`${message}: missing ${missing.join(', ')}`);
}

module.exports = async function beforePack(context) {
  const desktopRoot = context.packager.appDir || context.packager.projectDir;
  const repositoryRoot = path.resolve(desktopRoot, '..');
  const uiIndex = path.join(repositoryRoot, 'ui', 'dist', 'index.html');
  if (!fs.existsSync(uiIndex)) throw new Error(`UI production bundle is missing: ${uiIndex}`);

  const packageJson = JSON.parse(fs.readFileSync(path.join(desktopRoot, 'package.json'), 'utf8'));
  const release = JSON.parse(fs.readFileSync(path.join(repositoryRoot, 'src', 'hawavoclean', 'release.json'), 'utf8'));
  if (packageJson.version !== release.version) {
    throw new Error(`Desktop version ${packageJson.version} does not match release identity ${release.version}.`);
  }

  const releaseBuild = process.env.HAWAVOCLEAN_RELEASE_BUILD === '1';
  const proofBuild = process.env.HAWAVOCLEAN_DESKTOP_PROOF_BUILD === '1';
  if (releaseBuild && proofBuild) {
    throw new Error('A desktop package cannot be both a release build and an unsigned proof build.');
  }
  if (!releaseBuild && !proofBuild) return;
  requireEnv(['HAWAVOCLEAN_RELEASE_SOURCE_SHA'], 'Release provenance is incomplete');
  if (!/^[0-9a-f]{40}$/i.test(process.env.HAWAVOCLEAN_RELEASE_SOURCE_SHA)) {
    throw new Error('HAWAVOCLEAN_RELEASE_SOURCE_SHA must be a full 40-character Git SHA.');
  }

  const platform = process.env.HAWAVOCLEAN_TARGET_PLATFORM;
  const arch = process.env.HAWAVOCLEAN_TARGET_ARCH;
  if (platform !== 'darwin' && platform !== 'win32') throw new Error('Release target platform must be darwin or win32.');
  if ((platform === 'darwin' && arch !== 'arm64') || (platform === 'win32' && arch !== 'x64')) {
    throw new Error(`Unsupported release target: ${platform}-${arch}.`);
  }
  const engineMode = process.env.HAWAVOCLEAN_DESKTOP_ENGINE_MODE || 'full';
  if (!['full', 'shell-only'].includes(engineMode)) {
    throw new Error(`Unsupported desktop proof engine mode: ${engineMode}.`);
  }
  if (releaseBuild && engineMode !== 'full') {
    throw new Error('A release desktop package must contain the full engine payload.');
  }
  const engineRoot = process.env.HAWAVOCLEAN_DESKTOP_ENGINE_SOURCE
    ? path.resolve(process.env.HAWAVOCLEAN_DESKTOP_ENGINE_SOURCE)
    : path.join(desktopRoot, 'resources', 'engine', `${platform}-${arch}`);
  const executable = platform === 'win32' ? 'hawavoclean-engine.exe' : 'hawavoclean-engine';
  const enginePath = path.join(engineRoot, executable);
  if (engineMode === 'full' && (!fs.existsSync(enginePath) || !fs.statSync(enginePath).isFile())) {
    throw new Error(`Release engine is missing: ${enginePath}`);
  }
  if (engineMode === 'shell-only') {
    const placeholder = path.join(engineRoot, 'README.txt');
    if (!fs.existsSync(placeholder) || !fs.statSync(placeholder).isFile()) {
      throw new Error(`Shell-only proof placeholder is missing: ${placeholder}`);
    }
    if (fs.existsSync(enginePath)) {
      throw new Error('A shell-only proof must not be mistaken for a full engine package.');
    }
  }
  if (platform === 'darwin' && engineMode === 'full') {
    fs.accessSync(enginePath, fs.constants.X_OK);
  }
  if (!releaseBuild) return;
  if (platform === 'darwin') {
    requireEnv(
      ['CSC_LINK', 'CSC_KEY_PASSWORD', 'APPLE_ID', 'APPLE_APP_SPECIFIC_PASSWORD', 'APPLE_TEAM_ID'],
      'macOS signing/notarization is incomplete',
    );
  } else {
    if (!(process.env.WIN_CSC_LINK || process.env.CSC_LINK)) {
      throw new Error('Windows signing is incomplete: missing WIN_CSC_LINK or CSC_LINK.');
    }
    if (!(process.env.WIN_CSC_KEY_PASSWORD || process.env.CSC_KEY_PASSWORD)) {
      throw new Error('Windows signing is incomplete: missing WIN_CSC_KEY_PASSWORD or CSC_KEY_PASSWORD.');
    }
  }
};
