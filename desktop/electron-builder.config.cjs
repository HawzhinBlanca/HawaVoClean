'use strict';

const path = require('node:path');

const releaseBuild = process.env.HAWAVOCLEAN_RELEASE_BUILD === '1';
const proofBuild = process.env.HAWAVOCLEAN_DESKTOP_PROOF_BUILD === '1';
if (releaseBuild && proofBuild) {
  throw new Error('A desktop package cannot be both a release build and an unsigned proof build.');
}
const targetPlatform = process.env.HAWAVOCLEAN_TARGET_PLATFORM || process.platform;
const targetArch = process.env.HAWAVOCLEAN_TARGET_ARCH || process.arch;
const engineResource =
  process.env.HAWAVOCLEAN_DESKTOP_ENGINE_SOURCE ||
  path.join('resources', 'engine', `${targetPlatform}-${targetArch}`);
const outputDirectory = process.env.HAWAVOCLEAN_DESKTOP_OUTPUT || '../build/desktop';

/** @type {import('electron-builder').Configuration} */
module.exports = {
  appId: 'com.hawavoclean.desktop',
  productName: 'HawaVoClean',
  copyright: 'Copyright © Hawzhin',
  electronVersion: '43.4.1',
  artifactName: '${productName}-${version}-${os}-${arch}.${ext}',
  extraMetadata: {
    hawavocleanSourceSha: releaseBuild
      ? process.env.HAWAVOCLEAN_RELEASE_SOURCE_SHA
      : proofBuild
        ? process.env.HAWAVOCLEAN_RELEASE_SOURCE_SHA
        : 'development-unbound',
  },
  directories: {
    buildResources: 'build',
    output: outputDirectory,
  },
  icon: 'build/icon.png',
  files: [
    'dist/**/*',
    'package.json',
    '!**/*.map',
    '!**/node_modules/.cache/**/*',
  ],
  extraResources: [
    {
      from: '../ui/dist',
      to: 'ui',
      filter: ['**/*'],
    },
    {
      from: 'resources/branding',
      to: 'branding',
      filter: ['**/*'],
    },
    {
      from: engineResource,
      to: 'engine',
      filter: ['**/*'],
    },
  ],
  asar: true,
  compression: 'maximum',
  npmRebuild: false,
  buildDependenciesFromSource: false,
  removePackageScripts: true,
  forceCodeSigning: releaseBuild,
  publish: null,
  beforePack: './scripts/before-pack.cjs',
  afterPack: './scripts/after-pack.cjs',
  mac: {
    icon: 'build/icon.icns',
    target: [
      { target: 'dmg', arch: ['arm64'] },
      { target: 'zip', arch: ['arm64'] },
    ],
    category: 'public.app-category.music',
    minimumSystemVersion: '14.0.0',
    hardenedRuntime: true,
    entitlements: './entitlements.mac.plist',
    entitlementsInherit: './entitlements.mac.plist',
    identity: releaseBuild ? undefined : null,
    notarize: releaseBuild,
  },
  dmg: {
    sign: releaseBuild,
    writeUpdateInfo: true,
  },
  win: {
    icon: 'build/icon.ico',
    target: [{ target: 'nsis', arch: ['x64'] }],
    requestedExecutionLevel: 'asInvoker',
    signAndEditExecutable: true,
    signExts: ['.exe', '.dll'],
    verifyUpdateCodeSignature: true,
    signtoolOptions: {
      signingHashAlgorithms: ['sha256'],
      rfc3161TimeStampServer: 'http://timestamp.digicert.com',
    },
  },
  nsis: {
    oneClick: false,
    perMachine: false,
    allowElevation: true,
    allowToChangeInstallationDirectory: false,
    createDesktopShortcut: true,
    createStartMenuShortcut: true,
    deleteAppDataOnUninstall: false,
    differentialPackage: true,
  },
};
