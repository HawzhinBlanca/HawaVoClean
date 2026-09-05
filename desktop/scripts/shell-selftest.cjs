'use strict';

const electron = require('electron');
const path = require('node:path');

const { runDesktopSelfTest } = require('./selftest-harness.cjs');

runDesktopSelfTest({
  executable: electron,
  args: ['.'],
  cwd: path.resolve(__dirname, '..'),
  expectedPackaged: false,
  label: 'desktop shell self-test',
}).catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.stack : String(error)}\n`);
  process.exitCode = 1;
});
