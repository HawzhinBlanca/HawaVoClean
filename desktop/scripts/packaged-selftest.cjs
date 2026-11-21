'use strict';

const fs = require('node:fs');
const path = require('node:path');

const { runDesktopSelfTest } = require('./selftest-harness.cjs');

function fail(message) {
  process.stderr.write(`desktop packaged self-test failed: ${message}\n`);
  process.exit(2);
}

if (process.platform !== 'darwin') fail('the packaged macOS proof requires a macOS host');
if (process.argv.length !== 3) fail('usage: packaged-selftest.cjs /path/to/HawaVoClean.app');
const app = path.resolve(process.argv[2]);
let details;
try {
  details = fs.lstatSync(app);
} catch (error) {
  fail(error instanceof Error ? error.message : String(error));
}
if (path.extname(app) !== '.app' || details.isSymbolicLink() || !details.isDirectory()) {
  fail('target must be a real macOS application directory');
}
const executable = path.join(app, 'Contents', 'MacOS', 'HawaVoClean');
try {
  const executableDetails = fs.lstatSync(executable);
  if (executableDetails.isSymbolicLink() || !executableDetails.isFile()) {
    fail('packaged executable must be a real file');
  }
  fs.accessSync(executable, fs.constants.X_OK);
} catch (error) {
  fail(error instanceof Error ? error.message : String(error));
}

runDesktopSelfTest({
  executable,
  args: [],
  cwd: path.dirname(app),
  expectedPackaged: true,
  label: 'desktop packaged app self-test',
}).catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.stack : String(error)}\n`);
  process.exitCode = 1;
});
