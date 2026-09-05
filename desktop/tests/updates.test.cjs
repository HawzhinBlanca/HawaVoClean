'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const {
  compareSemver,
  canonicalizeManifest,
  generateSigningKeyPair,
  signManifest,
  verifyManifestSignature,
  createDatabaseBackup,
  restoreDatabaseBackup,
  getDatabaseVersion,
  verifyDatabaseIntegrity,
  migrateDatabase,
  UpdateManager,
} = require('../dist/updates/index.js');

test('semver comparator correctly handles major, minor, patch and prerelease', () => {
  assert.equal(compareSemver('3.4.0', '3.3.0'), 1);
  assert.equal(compareSemver('3.3.0', '3.4.0'), -1);
  assert.equal(compareSemver('3.3.0', '3.3.0'), 0);
  assert.equal(compareSemver('v3.3.1', '3.3.0'), 1);
  assert.equal(compareSemver('4.0.0', '3.99.99'), 1);
  assert.equal(compareSemver('3.3.0', '3.3.0-beta.1'), 1);
  assert.equal(compareSemver('3.3.0-beta.1', '3.3.0'), -1);
});

test('manifest canonicalization produces deterministic key-ordered JSON', () => {
  const m1 = {
    version: '3.4.0',
    product: 'hawavoclean',
    notes: 'A note',
    signature: 'to-be-ignored',
  };
  const m2 = {
    signature: 'different-sig',
    notes: 'A note',
    product: 'hawavoclean',
    version: '3.4.0',
  };
  assert.equal(canonicalizeManifest(m1), canonicalizeManifest(m2));
  assert.equal(canonicalizeManifest(m1), JSON.stringify({
    notes: 'A note',
    product: 'hawavoclean',
    version: '3.4.0',
  }));
});

test('Ed25519 signing and verification validates legitimate manifests and rejects corrupt signatures', () => {
  const keys = generateSigningKeyPair();
  const manifest = {
    schemaVersion: 1,
    product: 'hawavoclean',
    version: '3.4.0',
    minSupportedVersion: '3.0.0',
    releaseDate: '2026-09-06T00:00:00Z',
    channel: 'stable',
    target: 'darwin-arm64',
    sha256: crypto.createHash('sha256').update('binary-payload').digest('hex'),
    notes: 'Feature update',
    downloadUrl: 'https://updates.hawavoclean.com/v3.4.0.bin',
  };

  const sig = signManifest(manifest, keys.privateKey);
  const signedManifest = { ...manifest, signature: sig };

  // 1. Valid signature passes
  const validCheck = verifyManifestSignature(signedManifest, keys.publicKey);
  assert.equal(validCheck.valid, true);

  // 2. Modified payload fails closed
  const tamperedManifest = { ...signedManifest, notes: 'Tampered notes' };
  const tamperedCheck = verifyManifestSignature(tamperedManifest, keys.publicKey);
  assert.equal(tamperedCheck.valid, false);
  assert.ok(tamperedCheck.reason.includes('corrupt_signature'));

  // 3. Corrupted signature string fails closed
  const corruptSigManifest = { ...signedManifest, signature: Buffer.from('invalid-sig').toString('base64') };
  const corruptCheck = verifyManifestSignature(corruptSigManifest, keys.publicKey);
  assert.equal(corruptCheck.valid, false);
  assert.ok(corruptCheck.reason.includes('corrupt_signature'));

  // 4. Foreign key fails closed
  const foreignKeys = generateSigningKeyPair();
  const foreignCheck = verifyManifestSignature(signedManifest, foreignKeys.publicKey);
  assert.equal(foreignCheck.valid, false);
});

test('UpdateManager checks updates, rejects corrupt signatures, and enforces downgrade protection', async () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hawa-updates-test-'));
  try {
    const keys = generateSigningKeyPair();
    const manager = new UpdateManager({
      currentVersion: '3.3.0',
      publicKey: keys.publicKey,
      updateDir: tempDir,
    });

    assert.equal(manager.getState().status, 'idle');
    assert.equal(manager.getState().canCheck, true);

    const validPayload = Buffer.from('payload-v3.4.0');
    const validSha = crypto.createHash('sha256').update(validPayload).digest('hex');

    // Scenario A: Corrupt signature rejected
    const corruptManifest = {
      schemaVersion: 1,
      product: 'hawavoclean',
      version: '3.4.0',
      minSupportedVersion: '3.0.0',
      releaseDate: '2026-09-06T00:00:00Z',
      channel: 'stable',
      target: 'darwin-arm64',
      sha256: validSha,
      notes: 'New version',
      downloadUrl: 'https://updates.hawavoclean.com/v3.4.0.bin',
      signature: Buffer.from('corrupt-ed25519-sig').toString('base64'),
    };
    const corruptResult = await manager.checkForUpdates({ manifest: corruptManifest });
    assert.equal(corruptResult.ok, false);
    assert.equal(corruptResult.error, 'corrupt_signature');
    assert.equal(manager.getState().status, 'error');
    assert.equal(manager.getState().reason, 'corrupt_signature');

    // Scenario B: Downgrade rejected
    const olderManifestBase = {
      schemaVersion: 1,
      product: 'hawavoclean',
      version: '3.2.0',
      minSupportedVersion: '3.0.0',
      releaseDate: '2026-08-01T00:00:00Z',
      channel: 'stable',
      target: 'darwin-arm64',
      sha256: validSha,
      notes: 'Older version',
      downloadUrl: 'https://updates.hawavoclean.com/v3.2.0.bin',
    };
    const olderSig = signManifest(olderManifestBase, keys.privateKey);
    const olderManifest = { ...olderManifestBase, signature: olderSig };

    const downgradeResult = await manager.checkForUpdates({ manifest: olderManifest });
    assert.equal(downgradeResult.ok, false);
    assert.equal(downgradeResult.error, 'downgrade_rejected');
    assert.equal(manager.getState().status, 'error');
    assert.equal(manager.getState().reason, 'downgrade_rejected');

    // Scenario C: Offline network failure handled gracefully
    const offlineResult = await manager.checkForUpdates({ offline: true });
    assert.equal(offlineResult.ok, false);
    assert.equal(offlineResult.error, 'offline');
    assert.equal(manager.getState().status, 'error');
    assert.equal(manager.getState().reason, 'offline');

    // Scenario D: Valid newer version succeeds
    const newerManifestBase = {
      schemaVersion: 1,
      product: 'hawavoclean',
      version: '3.4.0',
      minSupportedVersion: '3.0.0',
      releaseDate: '2026-09-06T00:00:00Z',
      channel: 'stable',
      target: 'darwin-arm64',
      sha256: validSha,
      notes: 'Clean audio v3.4.0',
      downloadUrl: 'https://updates.hawavoclean.com/v3.4.0.bin',
      requiredFuses: {
        runAsNode: false,
        onlyLoadAppFromAsar: true,
        enableEmbeddedAsarIntegrityValidation: true,
      },
    };
    const newerSig = signManifest(newerManifestBase, keys.privateKey);
    const newerManifest = { ...newerManifestBase, signature: newerSig };

    const validCheck = await manager.checkForUpdates({ manifest: newerManifest });
    assert.equal(validCheck.ok, true);
    assert.equal(manager.getState().status, 'available');
    assert.equal(manager.getState().stagedVersion, '3.4.0');
    assert.equal(manager.getState().releaseNotes, 'Clean audio v3.4.0');
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
});

test('UpdateManager verifies SHA-256 and Electron fuses during staging', async () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hawa-fuses-test-'));
  try {
    const keys = generateSigningKeyPair();
    const manager = new UpdateManager({
      currentVersion: '3.3.0',
      publicKey: keys.publicKey,
      updateDir: tempDir,
    });

    const payload = Buffer.from('update-binary-data');
    const correctSha = crypto.createHash('sha256').update(payload).digest('hex');

    const manifestBase = {
      schemaVersion: 1,
      product: 'hawavoclean',
      version: '3.4.0',
      minSupportedVersion: '3.0.0',
      releaseDate: '2026-09-06T00:00:00Z',
      channel: 'stable',
      target: 'darwin-arm64',
      sha256: correctSha,
      notes: 'Fuses test',
      downloadUrl: 'https://updates.hawavoclean.com/v3.4.0.bin',
      requiredFuses: {
        runAsNode: true, // INSECURE FUSE VIOLATION
        onlyLoadAppFromAsar: true,
        enableEmbeddedAsarIntegrityValidation: true,
      },
    };
    const sig = signManifest(manifestBase, keys.privateKey);
    const manifest = { ...manifestBase, signature: sig };

    // Staging with invalid fuse fails closed
    const stageFail = await manager.stageUpdate(manifest, payload);
    assert.equal(stageFail.ok, false);
    assert.equal(stageFail.error, 'fuse_integrity_violation');

    // Staging with checksum mismatch fails closed
    const badShaManifest = { ...manifest, requiredFuses: { runAsNode: false, onlyLoadAppFromAsar: true, enableEmbeddedAsarIntegrityValidation: true }, sha256: 'deadbeef' };
    const badShaFail = await manager.stageUpdate(badShaManifest, payload);
    assert.equal(badShaFail.ok, false);
    assert.equal(badShaFail.error, 'checksum_mismatch');

    // Staging with valid fuses and matching checksum succeeds
    const validManifest = { ...badShaManifest, sha256: correctSha };
    const stageOk = await manager.stageUpdate(validManifest, payload);
    assert.equal(stageOk.ok, true);
    assert.equal(manager.getState().status, 'staged');
    assert.equal(manager.getState().canApply, true);
    assert.equal(fs.existsSync(manager.getStagedArtifactPath()), true);
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
});

test('Active-job protection: updates never interrupt active audio processing work', async () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hawa-active-job-test-'));
  try {
    let jobRunning = true;
    const keys = generateSigningKeyPair();
    const manager = new UpdateManager({
      currentVersion: '3.3.0',
      publicKey: keys.publicKey,
      updateDir: tempDir,
      isJobActive: () => jobRunning,
    });

    const payload = Buffer.from('update-binary');
    const sha = crypto.createHash('sha256').update(payload).digest('hex');
    const manifestBase = {
      schemaVersion: 1,
      product: 'hawavoclean',
      version: '3.4.0',
      minSupportedVersion: '3.0.0',
      releaseDate: '2026-09-06T00:00:00Z',
      channel: 'stable',
      target: 'darwin-arm64',
      sha256: sha,
      notes: 'Active job test',
      downloadUrl: 'https://updates.hawavoclean.com/v3.4.0.bin',
    };
    const manifest = { ...manifestBase, signature: signManifest(manifestBase, keys.privateKey) };

    await manager.stageUpdate(manifest, payload);
    assert.equal(manager.getState().status, 'staged');

    // ATTEMPT TO APPLY WHILE JOB IS ACTIVE: STRICTLY BLOCKED!
    const blocked = await manager.applyUpdate();
    assert.equal(blocked.ok, false);
    assert.equal(blocked.error, 'cannot_apply_during_active_job');
    assert.equal(manager.getState().status, 'staged');
    assert.equal(manager.getState().activeJobBlocking, true);

    // WHEN JOB COMPLETES (IDLE): APPLY SUCCEEDS CLEANLY!
    jobRunning = false;
    const applyOk = await manager.applyUpdate();
    assert.equal(applyOk.ok, true);
    assert.equal(manager.getState().status, 'idle');
    assert.equal(manager.getState().currentVersion, '3.4.0');
    assert.equal(manager.getState().activeJobBlocking, false);
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
});

test('N-1 -> N Database Migration executes cleanly and preserves historical jobs and master outputs', async () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hawa-migration-n1-test-'));
  const dbPath = path.join(tempDir, 'jobs.sqlite3');
  const userDocs = path.join(tempDir, 'Documents');
  fs.mkdirSync(userDocs, { recursive: true });
  const exportedMaster = path.join(userDocs, 'Master-Output.wav');
  fs.writeFileSync(exportedMaster, 'RIFF-audio-data-v1');
  const exportedRecord = path.join(userDocs, 'Processing-Record.zip');
  fs.writeFileSync(exportedRecord, 'PK-archive-data-v1');

  try {
    const { DatabaseSync } = require('node:sqlite');
    const db = new DatabaseSync(dbPath);
    db.exec(`
      PRAGMA user_version = 1;
      CREATE TABLE jobs (
        job_id TEXT PRIMARY KEY,
        input_path TEXT NOT NULL,
        state TEXT NOT NULL
      );
      INSERT INTO jobs VALUES ('job_1', '/media/1.wav', 'completed');
      INSERT INTO jobs VALUES ('job_2', '/media/2.wav', 'completed');
    `);
    db.close();

    assert.equal(getDatabaseVersion(dbPath), 1);

    // Define N-1 -> N migration (version 1 -> 2: adds columns and index)
    const migrations = [
      {
        version: 2,
        name: 'add_reconstruction_consent_and_metadata',
        up: (conn) => {
          conn.exec('ALTER TABLE jobs ADD COLUMN reconstruction_consent INTEGER DEFAULT 0;');
          conn.exec('CREATE INDEX idx_jobs_state ON jobs(state);');
        },
      },
    ];

    const result = await migrateDatabase({
      dbPath,
      targetVersion: 2,
      migrations,
    });

    assert.equal(result.ok, true);
    assert.equal(result.fromVersion, 1);
    assert.equal(result.toVersion, 2);
    assert.equal(result.appliedCount, 1);
    assert.equal(result.rolledBack, false);

    // CRITICAL: Historical jobs from N-1 must survive with 100% fidelity
    const verifyDb = new DatabaseSync(dbPath);
    const rows = verifyDb.prepare('SELECT * FROM jobs ORDER BY job_id;').all();
    assert.equal(rows.length, 2);
    assert.equal(rows[0].job_id, 'job_1');
    assert.equal(rows[0].reconstruction_consent, 0);
    assert.equal(rows[1].job_id, 'job_2');
    verifyDb.close();

    // User files survive
    assert.equal(fs.existsSync(exportedMaster), true);
    assert.equal(fs.readFileSync(exportedMaster, 'utf8'), 'RIFF-audio-data-v1');
    assert.equal(fs.existsSync(exportedRecord), true);
    assert.equal(fs.readFileSync(exportedRecord, 'utf8'), 'PK-archive-data-v1');
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
});

test('Failed database migration automatically rolls back to pre-migration snapshot without data loss', async () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hawa-failed-migration-test-'));
  const dbPath = path.join(tempDir, 'jobs.sqlite3');
  const userDocs = path.join(tempDir, 'Documents');
  fs.mkdirSync(userDocs, { recursive: true });
  const exportedMaster = path.join(userDocs, 'Master-Output.wav');
  fs.writeFileSync(exportedMaster, 'RIFF-audio-data-unbroken');

  try {
    const { DatabaseSync } = require('node:sqlite');
    const db = new DatabaseSync(dbPath);
    db.exec(`
      PRAGMA user_version = 1;
      CREATE TABLE jobs (
        job_id TEXT PRIMARY KEY,
        state TEXT NOT NULL
      );
      INSERT INTO jobs VALUES ('historical_job_1', 'completed');
    `);
    db.close();

    assert.equal(getDatabaseVersion(dbPath), 1);

    // Migration that deliberately throws an error
    const faultyMigrations = [
      {
        version: 2,
        name: 'faulty_migration',
        up: (conn) => {
          conn.exec('CREATE TABLE temp_scratch (id TEXT);');
          throw new Error('Simulated IO or constraint failure during migration');
        },
      },
    ];

    const result = await migrateDatabase({
      dbPath,
      targetVersion: 2,
      migrations: faultyMigrations,
    });

    // Fails closed and rolls back!
    assert.equal(result.ok, false);
    assert.equal(result.rolledBack, true);
    assert.ok(result.error.includes('failed_migration_rolled_back'));

    // Database version remains N-1
    assert.equal(getDatabaseVersion(dbPath), 1);

    // Historical data survived intact and the temp table was not retained
    const verifyDb = new DatabaseSync(dbPath);
    const rows = verifyDb.prepare('SELECT * FROM jobs;').all();
    assert.equal(rows.length, 1);
    assert.equal(rows[0].job_id, 'historical_job_1');

    // Scratch table was rolled back
    const tables = verifyDb.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='temp_scratch';").all();
    assert.equal(tables.length, 0);
    verifyDb.close();

    // User master audio survives
    assert.equal(fs.existsSync(exportedMaster), true);
    assert.equal(fs.readFileSync(exportedMaster, 'utf8'), 'RIFF-audio-data-unbroken');
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
});

test('Authorized rollback to N-1 restores executable state and database compatibility', async () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hawa-rollback-test-'));
  const dbPath = path.join(tempDir, 'jobs.sqlite3');

  try {
    const { DatabaseSync } = require('node:sqlite');
    const db = new DatabaseSync(dbPath);
    db.exec(`
      PRAGMA user_version = 1;
      CREATE TABLE jobs (id TEXT PRIMARY KEY, name TEXT);
      INSERT INTO jobs VALUES ('1', 'Initial Record');
    `);
    db.close();

    const keys = generateSigningKeyPair();
    let rollbackCallbackCalled = false;
    const manager = new UpdateManager({
      currentVersion: '3.3.0',
      publicKey: keys.publicKey,
      updateDir: tempDir,
      dbPath,
      onRollback: (ver) => {
        if (ver === '3.3.0') rollbackCallbackCalled = true;
      },
    });

    // Stage and apply update to 3.4.0 with migration
    const payload = Buffer.from('v3.4.0-payload');
    const sha = crypto.createHash('sha256').update(payload).digest('hex');
    const manifestBase = {
      schemaVersion: 1,
      product: 'hawavoclean',
      version: '3.4.0',
      minSupportedVersion: '3.0.0',
      releaseDate: '2026-09-06T00:00:00Z',
      channel: 'stable',
      target: 'darwin-arm64',
      sha256: sha,
      notes: 'v3.4.0 release',
      downloadUrl: 'https://updates.hawavoclean.com/v3.4.0.bin',
    };
    const manifest = { ...manifestBase, signature: signManifest(manifestBase, keys.privateKey) };

    await manager.stageUpdate(manifest, payload);
    const applyOk = await manager.applyUpdate({
      targetSchemaVersion: 2,
      migrations: [
        {
          version: 2,
          name: 'v2',
          up: (conn) => {
            conn.exec('ALTER TABLE jobs ADD COLUMN extra TEXT;');
          },
        },
      ],
    });
    assert.equal(applyOk.ok, true);
    assert.equal(manager.getState().currentVersion, '3.4.0');
    assert.equal(getDatabaseVersion(dbPath), 2);

    // NOW EXECUTE ROLLBACK TO 3.3.0:
    const rollbackResult = await manager.rollback('3.3.0');
    assert.equal(rollbackResult.ok, true);
    assert.equal(manager.getState().currentVersion, '3.3.0');
    assert.equal(rollbackCallbackCalled, true);

    // Database state restored from pre-migration backup
    assert.equal(getDatabaseVersion(dbPath), 1);
    const checkDb = new DatabaseSync(dbPath);
    const rows = checkDb.prepare('SELECT * FROM jobs;').all();
    assert.equal(rows.length, 1);
    assert.equal(rows[0].name, 'Initial Record');
    checkDb.close();
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
});
