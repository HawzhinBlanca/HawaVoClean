'use strict';

const { spawnSync } = require('node:child_process');

const requested = process.argv[2];
if (requested !== 'mac' && requested !== 'win') throw new Error('Usage: package-release.cjs mac|win');
const platform = requested === 'mac' ? 'darwin' : 'win32';
const arch = requested === 'mac' ? 'arm64' : 'x64';
if (process.platform !== platform || process.arch !== arch) {
  throw new Error(`Release ${platform}-${arch} must run on a native ${platform}-${arch} host.`);
}
const command = process.platform === 'win32' ? 'pnpm.cmd' : 'pnpm';
const environment = {
  ...process.env,
  HAWAVOCLEAN_RELEASE_BUILD: '1',
  HAWAVOCLEAN_TARGET_PLATFORM: platform,
  HAWAVOCLEAN_TARGET_ARCH: arch,
};
for (const args of [
  ['run', 'build:all'],
  ['exec', 'electron-builder', `--${requested}`, `--${arch}`, '--config', 'electron-builder.config.cjs'],
]) {
  const result = spawnSync(command, args, { cwd: __dirname + '/..', env: environment, stdio: 'inherit' });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
}
