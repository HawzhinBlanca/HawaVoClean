'use strict';

const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const { flipFuses, FuseVersion, FuseV1Options } = require('@electron/fuses');

function sha256(file) {
  return crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
}

function writeProofProvenance(context) {
  if (process.env.HAWAVOCLEAN_DESKTOP_PROOF_BUILD !== '1') return;
  const sourceRevision = process.env.HAWAVOCLEAN_RELEASE_SOURCE_SHA;
  if (!/^[0-9a-f]{40}$/i.test(sourceRevision || '')) {
    throw new Error('Unsigned desktop proof has no exact source revision.');
  }
  if (context.electronPlatformName !== 'darwin' || context.arch !== 3) {
    // electron-builder's Arch.arm64 enum value is 3. Keeping this check here
    // prevents a proof metadata file from mislabelling another target.
    throw new Error('Unsigned desktop proof must target macOS arm64.');
  }
  const app = path.join(context.appOutDir, 'HawaVoClean.app');
  const resources = path.join(app, 'Contents', 'Resources');
  const engineMode = process.env.HAWAVOCLEAN_DESKTOP_ENGINE_MODE || 'full';
  const engineManifest = path.join(resources, 'engine', 'ENGINE-MANIFEST.json');
  const provenance = {
    schema_version: 1,
    artifact_type: 'unsigned-macos-app-proof',
    distribution_eligible: false,
    product: 'hawavoclean',
    product_version: '3.3.0',
    source_revision: sourceRevision.toLowerCase(),
    target: 'macos-arm64',
    engine_mode: engineMode,
    engine_manifest_sha256:
      engineMode === 'full' && fs.existsSync(engineManifest) ? sha256(engineManifest) : null,
    packaged_selftest_allowed: engineMode === 'full',
    signing: {
      developer_id: false,
      notarized: false,
      stapled: false,
    },
  };
  fs.writeFileSync(
    path.join(resources, 'HAWAVOCLEAN-PACKAGE-PROVENANCE.json'),
    `${JSON.stringify(provenance, null, 2)}\n`,
    { encoding: 'utf8', mode: 0o644 },
  );
}

module.exports = async function afterPack(context) {
  const platform = context.electronPlatformName;
  let executable;
  if (platform === 'darwin') {
    executable = path.join(context.appOutDir, 'HawaVoClean.app', 'Contents', 'MacOS', 'HawaVoClean');
  } else if (platform === 'win32') {
    executable = path.join(context.appOutDir, 'HawaVoClean.exe');
  } else {
    throw new Error(`Desktop packaging is unsupported on ${platform}.`);
  }
  await flipFuses(executable, {
    version: FuseVersion.V1,
    [FuseV1Options.RunAsNode]: false,
    [FuseV1Options.EnableCookieEncryption]: true,
    [FuseV1Options.EnableNodeOptionsEnvironmentVariable]: false,
    [FuseV1Options.EnableNodeCliInspectArguments]: false,
    [FuseV1Options.EnableEmbeddedAsarIntegrityValidation]: true,
    [FuseV1Options.OnlyLoadAppFromAsar]: true,
    [FuseV1Options.LoadBrowserProcessSpecificV8Snapshot]: false,
    [FuseV1Options.GrantFileProtocolExtraPrivileges]: false,
  });
  writeProofProvenance(context);
};
