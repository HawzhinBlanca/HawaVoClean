'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const root = path.resolve(__dirname, '..');
const packageJson = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8'));
const release = JSON.parse(fs.readFileSync(path.join(root, '..', 'src', 'hawavoclean', 'release.json'), 'utf8'));
const config = require(path.join(root, 'electron-builder.config.cjs'));

assert.equal(packageJson.version, release.version, 'desktop and product versions must agree');
assert.equal(packageJson.main, 'dist/main.js');
assert.equal(packageJson.packageManager, 'pnpm@11.22.0');
assert.equal(packageJson.devDependencies.electron, config.electronVersion);
for (const [name, version] of Object.entries(packageJson.devDependencies)) {
  assert.match(version, /^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/, `${name} must be exactly pinned`);
}
assert.equal(config.asar, true);
assert.equal(config.extraMetadata.hawavocleanSourceSha, 'development-unbound');
assert.equal(config.mac.hardenedRuntime, true);
assert.deepEqual(config.mac.target, [
  { target: 'dmg', arch: ['arm64'] },
  { target: 'zip', arch: ['arm64'] },
]);
assert.deepEqual(config.win.target, [{ target: 'nsis', arch: ['x64'] }]);
assert.deepEqual(config.win.signExts, ['.exe', '.dll']);
assert.equal(config.nsis.deleteAppDataOnUninstall, false);
assert.equal(config.publish, null);
assert.equal(config.directories.buildResources, 'build');
assert.equal(config.icon, 'build/icon.png');
assert.equal(config.mac.icon, 'build/icon.icns');
assert.equal(config.win.icon, 'build/icon.ico');
assert.ok(fs.existsSync(path.join(root, 'build', 'icon.icns')), 'branded mac icon must exist');
assert.ok(fs.statSync(path.join(root, 'build', 'icon.icns')).size > 10000, 'branded mac icon must be valid');
assert.ok(fs.existsSync(path.join(root, 'build', 'icon.ico')), 'branded win icon must exist');
assert.ok(fs.statSync(path.join(root, 'build', 'icon.ico')).size > 5000, 'branded win icon must be valid');
assert.ok(fs.existsSync(path.join(root, 'build', 'icon.png')), 'branded master icon must exist');
assert.ok(fs.existsSync(path.join(root, 'resources', 'branding', 'icon.png')), 'branded runtime icon must exist');
assert.ok(fs.existsSync(path.join(root, '..', 'ui', 'dist', 'index.html')), 'existing UI bundle must be built');

console.log('desktop config valid');
