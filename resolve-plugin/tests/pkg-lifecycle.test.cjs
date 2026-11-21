// @ts-check
'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const { execFileSync } = require('node:child_process');
const crypto = require('node:crypto');

const UNINSTALL_SH = path.join(__dirname, '..', 'uninstall.sh');
const BUILD_PKG_SH = path.join(__dirname, '..', 'build-pkg.sh');

test('uninstall.sh safely uninstalls owned plugin and refuses foreign directory', (t) => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'uninstall-test-'));
  t.after(() => fs.rmSync(tmpDir, { recursive: true, force: true }));

  const pluginDir = path.join(tmpDir, 'com.hawavoclean.resolve');
  fs.mkdirSync(pluginDir, { recursive: true });
  fs.writeFileSync(path.join(pluginDir, 'PLUGIN_ID'), 'com.hawavoclean.resolve\n');
  fs.writeFileSync(path.join(pluginDir, 'main.js'), 'console.log("hello");\n');

  // 1. Dry run should leave plugin intact
  execFileSync(UNINSTALL_SH, ['--dest', tmpDir, '--dry-run'], {
    env: { ...process.env, HAWA_UNINSTALL_ALLOW_RUNNING: '1' },
  });
  assert.ok(fs.existsSync(pluginDir), 'Dry-run must not delete plugin');

  // 2. Real uninstall with backup
  execFileSync(UNINSTALL_SH, ['--dest', tmpDir, '--backup'], {
    env: { ...process.env, HAWA_UNINSTALL_ALLOW_RUNNING: '1' },
  });
  assert.ok(!fs.existsSync(pluginDir), 'Plugin should be removed');

  // Verify backup was created
  const files = fs.readdirSync(tmpDir);
  const backup = files.find((f) => f.startsWith('.com.hawavoclean.resolve.backup.'));
  assert.ok(backup, 'Backup directory must exist');

  // 3. Uninstalling when already removed exits cleanly (code 0)
  execFileSync(UNINSTALL_SH, ['--dest', tmpDir], {
    env: { ...process.env, HAWA_UNINSTALL_ALLOW_RUNNING: '1' },
  });

  // 4. Refuse foreign directory
  fs.mkdirSync(pluginDir, { recursive: true });
  fs.writeFileSync(path.join(pluginDir, 'PLUGIN_ID'), 'foreign.rogue.plugin\n');
  assert.throws(
    () => {
      execFileSync(UNINSTALL_SH, ['--dest', tmpDir], {
        env: { ...process.env, HAWA_UNINSTALL_ALLOW_RUNNING: '1' },
      });
    },
    /not recognizably owned/
  );
});

test('build-pkg.sh generates package and packages postinstall/preinstall lifecycle', (t) => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'pkg-test-'));
  t.after(() => fs.rmSync(tmpDir, { recursive: true, force: true }));

  const stageDir = path.join(tmpDir, 'stage');
  fs.mkdirSync(stageDir, { recursive: true });
  fs.writeFileSync(path.join(stageDir, 'PLUGIN_ID'), 'com.hawavoclean.resolve\n');
  fs.writeFileSync(path.join(stageDir, 'VERSION'), '3.3.0\n');
  fs.writeFileSync(path.join(stageDir, 'main.js'), '// main script\n');

  const mainHash = crypto.createHash('sha256').update(fs.readFileSync(path.join(stageDir, 'main.js'))).digest('hex');
  fs.writeFileSync(path.join(stageDir, 'SHA256SUMS'), `${mainHash}  ./main.js\n`);

  const outputPkg = path.join(tmpDir, 'com.hawavoclean.resolve.pkg');
  execFileSync(BUILD_PKG_SH, ['--stage', stageDir, '--output', outputPkg], {
    env: { ...process.env },
  });

  assert.ok(fs.existsSync(outputPkg), 'Package artifact must be generated');
  assert.ok(fs.statSync(outputPkg).size > 0, 'Package must not be empty');

  if (process.platform === 'darwin') {
    const payload = execFileSync('pkgutil', ['--payload-files', outputPkg], { encoding: 'utf8' });
    const appleDouble = payload.split('\n').filter((line) => /(^|\/)\._/.test(line.trim()));
    assert.deepEqual(appleDouble, [], 'PKG payload must not contain macOS AppleDouble ._* files');
  }
});

test('install.sh and activate.sh stage and verify every sibling JS module required by main.js', () => {
  const mainSource = fs.readFileSync(
    path.join(__dirname, '..', 'com.hawavoclean.resolve', 'main.js'),
    'utf8'
  );
  const installSource = fs.readFileSync(path.join(__dirname, '..', 'install.sh'), 'utf8');
  const activateSource = fs.readFileSync(path.join(__dirname, '..', 'activate.sh'), 'utf8');
  const requiredModules = [...mainSource.matchAll(/require\(['"]\.\/([A-Za-z0-9._-]+\.js)['"]\)/g)].map(
    (m) => m[1]
  );
  assert.ok(requiredModules.includes('engine-discovery.js'), 'main.js must require engine-discovery.js');
  assert.ok(requiredModules.includes('timeline-transactions.js'), 'main.js must require timeline-transactions.js');
  for (const mod of requiredModules) {
    assert.ok(installSource.includes(mod), `install.sh must stage required module ${mod}`);
    assert.ok(activateSource.includes(mod), `activate.sh must verify required module ${mod}`);
  }
});
