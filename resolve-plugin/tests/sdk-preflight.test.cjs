// @ts-check
'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const crypto = require('node:crypto');

const {
  inspectMachOBinary,
  discoverWorkflowIntegrationSdk,
  acquireWorkflowIntegrationNode,
  CPU_TYPE_ARM64,
  CPU_TYPE_X86_64,
} = require('../com.hawavoclean.resolve/sdk-preflight.js');

test('inspectMachOBinary correctly parses Mach-O headers and rejects non-Mach-O', (t) => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'macho-test-'));
  t.after(() => fs.rmSync(tmpDir, { recursive: true, force: true }));

  // 1. Non-Mach-O file
  const textFile = path.join(tmpDir, 'plain.txt');
  fs.writeFileSync(textFile, 'This is a plain text file, not Mach-O.\n');
  const resText = inspectMachOBinary(textFile);
  assert.equal(resText.valid, false);
  assert.equal(resText.isMachO, false);

  // 2. Truncated file (< 32 bytes)
  const shortFile = path.join(tmpDir, 'short.bin');
  fs.writeFileSync(shortFile, Buffer.from([0xca, 0xfe, 0xba, 0xbe]));
  const resShort = inspectMachOBinary(shortFile);
  assert.equal(resShort.valid, false);

  // 3. Synthetic single 64-bit Mach-O arm64 binary
  const arm64File = path.join(tmpDir, 'arm64.bin');
  const arm64Buf = Buffer.alloc(64);
  arm64Buf.writeUInt32BE(0xfeedfacf, 0); // MH_MAGIC_64
  arm64Buf.writeUInt32BE(CPU_TYPE_ARM64, 4); // CPU_TYPE_ARM64
  fs.writeFileSync(arm64File, arm64Buf);
  const resArm64 = inspectMachOBinary(arm64File);
  assert.equal(resArm64.valid, true);
  assert.equal(resArm64.isMachO, true);
  assert.deepEqual(resArm64.architectures, ['arm64']);

  // 4. Synthetic universal/fat binary with both arm64 and x86_64
  const fatFile = path.join(tmpDir, 'universal.bin');
  const fatBuf = Buffer.alloc(128);
  fatBuf.writeUInt32BE(0xcafebabe, 0); // FAT_MAGIC
  fatBuf.writeUInt32BE(2, 4); // nfat = 2
  // arch 0: x86_64
  fatBuf.writeUInt32BE(CPU_TYPE_X86_64, 8);
  // arch 1: arm64
  fatBuf.writeUInt32BE(CPU_TYPE_ARM64, 28);
  fs.writeFileSync(fatFile, fatBuf);
  const resFat = inspectMachOBinary(fatFile);
  assert.equal(resFat.valid, true);
  assert.equal(resFat.isMachO, true);
  assert.deepEqual(resFat.architectures.sort(), ['arm64', 'x86_64'].sort());
});

test('discoverWorkflowIntegrationSdk discovers real host SDK when present on macOS', () => {
  if (process.platform !== 'darwin') return;

  const result = discoverWorkflowIntegrationSdk();
  // If host has Resolve Studio installed, it should find it cleanly
  const defaultSdk = '/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Workflow Integrations/Examples/SamplePlugin/WorkflowIntegration.node';
  if (fs.existsSync(defaultSdk)) {
    assert.equal(result.ok, true);
    assert.equal(result.status, 'ready');
    assert.equal(result.code, 'SDK_READY');
    assert.equal(result.path, defaultSdk);
    assert.ok(result.diagnostics.architectures.includes('arm64'));
  }
});

test('discoverWorkflowIntegrationSdk fails gracefully on non-Darwin platforms', () => {
  const result = discoverWorkflowIntegrationSdk({ platform: 'win32' });
  assert.equal(result.ok, false);
  assert.equal(result.status, 'unsupported_platform');
  assert.equal(result.code, 'ERR_UNSUPPORTED_PLATFORM');
  assert.ok(result.repairGuidance.includes('macOS'));
});

test('discoverWorkflowIntegrationSdk detects missing Resolve and gives repair guidance', (t) => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'sdk-missing-'));
  t.after(() => fs.rmSync(tmpDir, { recursive: true, force: true }));

  const nonExistentApp = path.join(tmpDir, 'DaVinci Resolve.app');
  const nonExistentSdk = path.join(tmpDir, 'SDK');

  const result = discoverWorkflowIntegrationSdk({
    platform: 'darwin',
    resolveAppPath: nonExistentApp,
    sdkDir: nonExistentSdk,
  });

  assert.equal(result.ok, false);
  assert.equal(result.status, 'resolve_not_installed');
  assert.equal(result.code, 'ERR_RESOLVE_NOT_INSTALLED');
  assert.ok(result.repairGuidance.includes('Please install DaVinci Resolve Studio'));
});

test('discoverWorkflowIntegrationSdk detects DaVinci Resolve Free and warns Workflow Integrations unsupported', (t) => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'sdk-free-'));
  t.after(() => fs.rmSync(tmpDir, { recursive: true, force: true }));

  const fakeApp = path.join(tmpDir, 'DaVinci Resolve.app');
  fs.mkdirSync(path.join(fakeApp, 'Contents'), { recursive: true });
  fs.writeFileSync(
    path.join(fakeApp, 'Contents', 'Info.plist'),
    '<?xml version="1.0" encoding="UTF-8"?><plist version="1.0"><dict><key>CFBundleIdentifier</key><string>com.blackmagic-design.DaVinciResolve</string></dict></plist>'
  );
  const nonExistentSdk = path.join(tmpDir, 'SDK');

  const result = discoverWorkflowIntegrationSdk({
    platform: 'darwin',
    resolveAppPath: fakeApp,
    sdkDir: nonExistentSdk,
  });

  assert.equal(result.ok, false);
  assert.equal(result.status, 'free_edition_unsupported');
  assert.equal(result.code, 'ERR_RESOLVE_FREE_EDITION');
  assert.ok(result.repairGuidance.includes('Free Edition'));
  assert.ok(result.repairGuidance.includes('HawaVoClean desktop application'));
});

test('acquireWorkflowIntegrationNode copies valid node and verifies checksum', (t) => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'sdk-acquire-'));
  t.after(() => fs.rmSync(tmpDir, { recursive: true, force: true }));

  // Create a synthetic universal node
  const sourceNode = path.join(tmpDir, 'source', 'WorkflowIntegration.node');
  fs.mkdirSync(path.dirname(sourceNode), { recursive: true });
  const fatBuf = Buffer.alloc(128);
  fatBuf.writeUInt32BE(0xcafebabe, 0); // FAT_MAGIC
  fatBuf.writeUInt32BE(2, 4); // 2 archs
  fatBuf.writeUInt32BE(CPU_TYPE_ARM64, 8);
  fatBuf.writeUInt32BE(CPU_TYPE_X86_64, 28);
  fs.writeFileSync(sourceNode, fatBuf);

  const targetDir = path.join(tmpDir, 'target');
  const result = acquireWorkflowIntegrationNode(targetDir, {
    platform: 'darwin',
    targetArch: 'arm64',
    overrideNodePath: sourceNode,
  });

  assert.equal(result.ok, true);
  assert.equal(result.code, 'ACQUIRED_OK');
  const destPath = path.join(targetDir, 'WorkflowIntegration.node');
  assert.ok(fs.existsSync(destPath));

  const sourceHash = crypto.createHash('sha256').update(fs.readFileSync(sourceNode)).digest('hex');
  const destHash = crypto.createHash('sha256').update(fs.readFileSync(destPath)).digest('hex');
  assert.equal(destHash, sourceHash);
});
