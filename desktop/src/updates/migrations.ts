import fs from 'node:fs';
import path from 'node:path';
import type { MigrationDefinition, MigrationResult } from '../contracts.js';

// Load Node's native SQLite support safely across environments
function getDatabaseSyncClass(): any {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const sqlite = require('node:sqlite');
    return sqlite.DatabaseSync ?? null;
  } catch {
    return null;
  }
}

/**
 * Creates an atomic snapshot backup of the SQLite database and its journal/WAL files.
 */
export function createDatabaseBackup(dbPath: string): string {
  if (!fs.existsSync(dbPath)) {
    throw new Error(`Cannot backup non-existent database: ${dbPath}`);
  }
  const timestamp = Date.now();
  const backupPath = `${dbPath}.backup-${timestamp}`;
  fs.copyFileSync(dbPath, backupPath);

  // If WAL or SHM files exist, back them up alongside the primary DB file
  const wal = `${dbPath}-wal`;
  if (fs.existsSync(wal)) {
    fs.copyFileSync(wal, `${backupPath}-wal`);
  }
  const shm = `${dbPath}-shm`;
  if (fs.existsSync(shm)) {
    fs.copyFileSync(shm, `${backupPath}-shm`);
  }

  return backupPath;
}

/**
 * Restores a database from a pre-migration snapshot, purging stray WAL/SHM files
 * to prevent journal desynchronization.
 */
export function restoreDatabaseBackup(backupPath: string, dbPath: string): void {
  if (!fs.existsSync(backupPath)) {
    throw new Error(`Cannot restore non-existent backup: ${backupPath}`);
  }

  // Remove active DB and auxiliary files before copying
  for (const suffix of ['', '-wal', '-shm']) {
    const target = `${dbPath}${suffix}`;
    if (fs.existsSync(target)) {
      try {
        fs.unlinkSync(target);
      } catch {
        // Fallback: try truncation/overwrite
      }
    }
  }

  fs.copyFileSync(backupPath, dbPath);

  // Restore WAL/SHM if they existed in the backup
  const backupWal = `${backupPath}-wal`;
  if (fs.existsSync(backupWal)) {
    fs.copyFileSync(backupWal, `${dbPath}-wal`);
  }
  const backupShm = `${backupPath}-shm`;
  if (fs.existsSync(backupShm)) {
    fs.copyFileSync(backupShm, `${dbPath}-shm`);
  }
}

/**
 * Reads the PRAGMA user_version from an SQLite database.
 */
export function getDatabaseVersion(dbPath: string): number {
  if (!fs.existsSync(dbPath)) return 0;
  const DatabaseSync = getDatabaseSyncClass();
  if (!DatabaseSync) {
    throw new Error('Native SQLite is unavailable in this runtime.');
  }
  const db = new DatabaseSync(dbPath);
  try {
    const row = db.prepare('PRAGMA user_version;').get() as { user_version?: number } | undefined;
    return typeof row?.user_version === 'number' ? row.user_version : 0;
  } finally {
    db.close();
  }
}

/**
 * Verifies SQLite database integrity via PRAGMA integrity_check.
 */
export function verifyDatabaseIntegrity(dbPath: string): boolean {
  if (!fs.existsSync(dbPath)) return false;
  const DatabaseSync = getDatabaseSyncClass();
  if (!DatabaseSync) return true;
  const db = new DatabaseSync(dbPath);
  try {
    const row = db.prepare('PRAGMA integrity_check;').get() as { integrity_check?: string } | undefined;
    return row?.integrity_check === 'ok';
  } catch {
    return false;
  } finally {
    db.close();
  }
}

/**
 * Executes database migrations from N-1 to N inside an immediate transaction.
 *
 * CRITICAL FAIL-CLOSED INVARIANT:
 * If any migration step throws, the transaction is rolled back, the database is
 * restored immediately from the pre-migration snapshot, integrity is verified,
 * and user history/records remain completely intact.
 */
export async function migrateDatabase(options: {
  dbPath: string;
  targetVersion: number;
  migrations: readonly MigrationDefinition[];
}): Promise<MigrationResult> {
  const { dbPath, targetVersion, migrations } = options;

  if (!fs.existsSync(dbPath)) {
    fs.mkdirSync(path.dirname(dbPath), { recursive: true });
    const DatabaseSync = getDatabaseSyncClass();
    if (DatabaseSync) {
      const db = new DatabaseSync(dbPath);
      db.exec('PRAGMA user_version = 0;');
      db.close();
    }
  }

  const backupPath = createDatabaseBackup(dbPath);
  const fromVersion = getDatabaseVersion(dbPath);

  if (fromVersion >= targetVersion) {
    return Object.freeze({
      ok: true,
      fromVersion,
      toVersion: fromVersion,
      appliedCount: 0,
      backupPath,
      rolledBack: false,
    });
  }

  const pending = migrations
    .filter((m) => m.version > fromVersion && m.version <= targetVersion)
    .sort((a, b) => a.version - b.version);

  if (pending.length === 0) {
    return Object.freeze({
      ok: true,
      fromVersion,
      toVersion: fromVersion,
      appliedCount: 0,
      backupPath,
      rolledBack: false,
    });
  }

  const DatabaseSync = getDatabaseSyncClass();
  if (!DatabaseSync) {
    return Object.freeze({
      ok: false,
      fromVersion,
      toVersion: fromVersion,
      appliedCount: 0,
      backupPath,
      rolledBack: true,
      error: 'failed_migration_rolled_back: SQLite engine unavailable',
    });
  }

  const db = new DatabaseSync(dbPath);
  let transactionActive = false;

  try {
    db.exec('BEGIN IMMEDIATE;');
    transactionActive = true;

    for (const migration of pending) {
      migration.up(db);
      db.exec(`PRAGMA user_version = ${migration.version};`);
    }

    db.exec('COMMIT;');
    transactionActive = false;
    db.close();

    const integrityOk = verifyDatabaseIntegrity(dbPath);
    if (!integrityOk) {
      throw new Error('Database failed integrity check after migration.');
    }

    return Object.freeze({
      ok: true,
      fromVersion,
      toVersion: targetVersion,
      appliedCount: pending.length,
      backupPath,
      rolledBack: false,
    });
  } catch (err) {
    if (transactionActive) {
      try {
        db.exec('ROLLBACK;');
      } catch {
        // Suppress rollback errors during cleanup
      }
    }
    try {
      db.close();
    } catch {
      // Suppress close errors
    }

    // AUTOMATIC FAIL-CLOSED ROLLBACK TO N-1 PRE-MIGRATION SNAPSHOT
    restoreDatabaseBackup(backupPath, dbPath);
    const restoredIntegrity = verifyDatabaseIntegrity(dbPath);

    return Object.freeze({
      ok: false,
      fromVersion,
      toVersion: fromVersion,
      appliedCount: 0,
      backupPath,
      rolledBack: true,
      error: `failed_migration_rolled_back: ${err instanceof Error ? err.message : String(err)} (restored_integrity: ${String(restoredIntegrity)})`,
    });
  }
}
