'use strict';

const { spawnSync } = require('node:child_process');

const env = {
  ...process.env,
  CSC_IDENTITY_AUTO_DISCOVERY: 'false',
  HAWAVOCLEAN_RELEASE_BUILD: '0',
  HAWAVOCLEAN_TARGET_PLATFORM: process.platform,
  HAWAVOCLEAN_TARGET_ARCH: process.arch,
};
const result = spawnSync(
  process.platform === 'win32' ? 'pnpm.cmd' : 'pnpm',
  ['exec', 'electron-builder', '--dir', '--config', 'electron-builder.config.cjs'],
  { cwd: __dirname + '/..', env, stdio: 'inherit' },
);
if (result.error) throw result.error;
process.exit(result.status ?? 1);

