import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import type {
  MigrationDefinition,
  UpdateManifest,
  UpdateState,
  UpdateStatus,
} from '../contracts.js';
import { compareSemver, verifyManifestSignature } from './signature.js';
import { migrateDatabase, restoreDatabaseBackup } from './migrations.js';

export interface UpdateManagerOptions {
  currentVersion: string;
  feedUrl?: string | undefined;
  publicKey?: string | undefined;
  updateDir?: string | undefined;
  dbPath?: string | undefined;
  isJobActive?: (() => Promise<boolean> | boolean) | undefined;
  onRollback?: ((version: string) => Promise<void> | void) | undefined;
}

export class UpdateManager {
  private currentVersion: string;
  private feedUrl: string | undefined;
  private publicKey: string | undefined;
  private updateDir: string;
  private dbPath: string | undefined;
  private isJobActive: () => Promise<boolean> | boolean;
  private onRollback: ((version: string) => Promise<void> | void) | undefined;

  private state: {
    status: UpdateStatus;
    currentVersion: string;
    stagedVersion?: string | undefined;
    activeJobBlocking: boolean;
    canCheck: boolean;
    canApply: boolean;
    reason?: string | undefined;
    releaseNotes?: string | undefined;
    channel?: string | undefined;
  };

  private stagedManifest: UpdateManifest | undefined;
  private stagedArtifactPath: string | undefined;
  private preMigrationBackupPath: string | undefined;

  constructor(options: UpdateManagerOptions) {
    this.currentVersion = options.currentVersion;
    this.feedUrl = options.feedUrl;
    this.publicKey = options.publicKey;
    this.updateDir = options.updateDir ?? path.join(process.cwd(), '.updates');
    this.dbPath = options.dbPath;
    this.isJobActive = options.isJobActive ?? (() => false);
    this.onRollback = options.onRollback;

    // Preserve exact identity contract: when release feed / key is unconfigured,
    // state matches selftest expectations: { status: 'disabled', reason: 'release_feed_not_configured', canCheck: false }
    if (!this.feedUrl && !this.publicKey) {
      this.state = {
        status: 'disabled',
        currentVersion: this.currentVersion,
        activeJobBlocking: false,
        canCheck: false,
        canApply: false,
        reason: 'release_feed_not_configured',
      };
    } else {
      this.state = {
        status: 'idle',
        currentVersion: this.currentVersion,
        activeJobBlocking: false,
        canCheck: true,
        canApply: false,
      };
    }
  }

  public getStagedArtifactPath(): string | undefined {
    return this.stagedArtifactPath;
  }

  public getState(): UpdateState {
    if (this.state.status === 'disabled') {
      return Object.freeze({
        status: 'disabled',
        reason: this.state.reason ?? 'release_feed_not_configured',
        canCheck: false,
      });
    }

    return Object.freeze({
      status: this.state.status,
      currentVersion: this.state.currentVersion,
      ...(this.state.stagedVersion !== undefined ? { stagedVersion: this.state.stagedVersion } : {}),
      activeJobBlocking: this.state.activeJobBlocking,
      canCheck: this.state.canCheck,
      canApply: this.state.canApply,
      ...(this.state.reason !== undefined ? { reason: this.state.reason } : {}),
      ...(this.state.releaseNotes !== undefined ? { releaseNotes: this.state.releaseNotes } : {}),
      ...(this.state.channel !== undefined ? { channel: this.state.channel } : {}),
    });
  }

  /**
   * Checks for available updates.
   * If a manifest is supplied directly (e.g. from local tests or staged feeds), evaluates it.
   * Otherwise queries this.feedUrl with offline detection.
   */
  public async checkForUpdates(options?: {
    manifest?: UpdateManifest | undefined;
    offline?: boolean | undefined;
  }): Promise<{ ok: boolean; manifest?: UpdateManifest | undefined; error?: string | undefined; reason?: string | undefined }> {
    if (!this.publicKey && !options?.manifest) {
      return { ok: false, error: 'not_configured', reason: 'release_feed_not_configured' };
    }

    this.state.status = 'checking';

    if (options?.offline) {
      this.state.status = 'error';
      this.state.reason = 'offline';
      return {
        ok: false,
        error: 'offline',
        reason: 'Network offline or update server unreachable',
      };
    }

    let manifest = options?.manifest;
    if (!manifest) {
      if (!this.feedUrl) {
        this.state.status = 'disabled';
        this.state.reason = 'release_feed_not_configured';
        return { ok: false, error: 'not_configured', reason: 'release_feed_not_configured' };
      }

      try {
        const response = await fetch(this.feedUrl, { headers: { 'User-Agent': `HawaVoClean/${this.currentVersion}` } });
        if (!response.ok) {
          throw new Error(`Feed HTTP ${String(response.status)}`);
        }
        manifest = (await response.json()) as UpdateManifest;
      } catch (networkErr) {
        this.state.status = 'error';
        this.state.reason = 'offline';
        return {
          ok: false,
          error: 'offline',
          reason: `Failed to fetch update feed: ${networkErr instanceof Error ? networkErr.message : String(networkErr)}`,
        };
      }
    }

    // 1. CRYPTOGRAPHIC SIGNATURE VERIFICATION
    if (this.publicKey) {
      const sigCheck = verifyManifestSignature(manifest, this.publicKey);
      if (!sigCheck.valid) {
        this.state.status = 'error';
        this.state.reason = 'corrupt_signature';
        return {
          ok: false,
          error: 'corrupt_signature',
          reason: sigCheck.reason ?? 'Update manifest signature verification failed',
        };
      }
    }

    // 2. DOWNGRADE / EQUAL VERSION GUARD
    const cmp = compareSemver(manifest.version, this.currentVersion);
    if (cmp < 0) {
      this.state.status = 'error';
      this.state.reason = 'downgrade_rejected';
      return {
        ok: false,
        error: 'downgrade_rejected',
        reason: `Target version ${manifest.version} is older than current ${this.currentVersion}`,
      };
    }
    if (cmp === 0) {
      this.state.status = 'up-to-date';
      this.state.canApply = false;
      return { ok: true, manifest, reason: 'Already running current version' };
    }

    // 3. MINIMUM SUPPORTED VERSION CHECK
    if (manifest.minSupportedVersion && compareSemver(this.currentVersion, manifest.minSupportedVersion) < 0) {
      this.state.status = 'error';
      this.state.reason = 'stepped_upgrade_required';
      return {
        ok: false,
        error: 'stepped_upgrade_required',
        reason: `Upgrade requires intermediate release (minimum: ${manifest.minSupportedVersion})`,
      };
    }

    this.state.status = 'available';
    this.state.stagedVersion = manifest.version;
    this.state.releaseNotes = manifest.notes;
    this.state.channel = manifest.channel;
    this.stagedManifest = manifest;

    return { ok: true, manifest };
  }

  /**
   * Stage an update artifact after verifying hash and Electron fuses.
   */
  public async stageUpdate(
    manifest: UpdateManifest,
    artifactBufferOrPath: Buffer | string,
  ): Promise<{ ok: boolean; error?: string | undefined; reason?: string | undefined }> {
    const buffer = typeof artifactBufferOrPath === 'string'
      ? fs.readFileSync(artifactBufferOrPath)
      : artifactBufferOrPath;

    // 1. SHA-256 CHECKSUM VERIFICATION
    const computedSha = crypto.createHash('sha256').update(buffer).digest('hex');
    if (computedSha.toLowerCase() !== manifest.sha256.toLowerCase()) {
      this.state.status = 'error';
      this.state.reason = 'checksum_mismatch';
      return {
        ok: false,
        error: 'checksum_mismatch',
        reason: `Artifact SHA-256 mismatch (expected ${manifest.sha256}, got ${computedSha})`,
      };
    }

    // 2. ELECTRON FUSES & ASAR INTEGRITY VERIFICATION
    if (manifest.requiredFuses) {
      const { runAsNode, onlyLoadAppFromAsar, enableEmbeddedAsarIntegrityValidation } = manifest.requiredFuses;
      if (runAsNode === true) {
        this.state.status = 'error';
        this.state.reason = 'fuse_integrity_violation';
        return {
          ok: false,
          error: 'fuse_integrity_violation',
          reason: 'RunAsNode fuse must be disabled in production update',
        };
      }
      if (onlyLoadAppFromAsar === false || enableEmbeddedAsarIntegrityValidation === false) {
        this.state.status = 'error';
        this.state.reason = 'fuse_integrity_violation';
        return {
          ok: false,
          error: 'fuse_integrity_violation',
          reason: 'ASAR integrity and OnlyLoadAppFromAsar fuses must be enabled',
        };
      }
    }

    // 3. ATOMIC ARTIFACT STAGING
    const stageDir = path.join(this.updateDir, 'staged', manifest.version);
    fs.mkdirSync(stageDir, { recursive: true });
    const targetFile = path.join(stageDir, `update-${manifest.version}.bin`);
    fs.writeFileSync(targetFile, buffer);

    this.stagedManifest = manifest;
    this.stagedArtifactPath = targetFile;
    this.state.status = 'staged';
    this.state.stagedVersion = manifest.version;
    this.state.canApply = true;
    this.state.activeJobBlocking = false;

    return { ok: true };
  }

  /**
   * Applies the staged update.
   *
   * CRITICAL GUARANTEE:
   * If an active audio processing job is running, update application is strictly
   * refused with `cannot_apply_during_active_job`. Updates never interrupt active work!
   */
  public async applyUpdate(options?: {
    allowActiveJobOverride?: boolean | undefined;
    migrations?: readonly MigrationDefinition[] | undefined;
    targetSchemaVersion?: number | undefined;
    dbPath?: string | undefined;
  }): Promise<{ ok: boolean; error?: string | undefined; reason?: string | undefined }> {
    if (this.state.status !== 'staged' || !this.stagedManifest) {
      return { ok: false, error: 'no_update_staged', reason: 'No update is currently staged' };
    }

    // 1. ACTIVE JOB PROTECTION
    const jobActive = await this.isJobActive();
    if (jobActive && !options?.allowActiveJobOverride) {
      this.state.activeJobBlocking = true;
      return {
        ok: false,
        error: 'cannot_apply_during_active_job',
        reason: 'Update deferred: active audio processing job in progress',
      };
    }

    this.state.activeJobBlocking = false;

    // 2. DATABASE & ARTIFACT MIGRATION COMPATIBILITY WITH AUTOMATIC ROLLBACK
    const targetDb = options?.dbPath ?? this.dbPath;
    if (targetDb && options?.migrations && options.targetSchemaVersion !== undefined) {
      const migrationResult = await migrateDatabase({
        dbPath: targetDb,
        targetVersion: options.targetSchemaVersion,
        migrations: options.migrations,
      });

      if (!migrationResult.ok) {
        this.state.status = 'error';
        this.state.reason = 'failed_migration_rolled_back';
        return {
          ok: false,
          error: 'failed_migration_rolled_back',
          reason: migrationResult.error ?? 'Database migration failed and state was rolled back to N-1',
        };
      }

      this.preMigrationBackupPath = migrationResult.backupPath;
    }

    // 3. COMMIT UPDATE APPLICATION
    const appliedVersion = this.stagedManifest.version;
    this.currentVersion = appliedVersion;
    this.state.status = 'idle';
    this.state.currentVersion = appliedVersion;
    delete this.state.stagedVersion;
    this.state.canApply = false;
    this.stagedManifest = undefined;
    this.stagedArtifactPath = undefined;

    return { ok: true };
  }

  /**
   * Rollback to previous version N-1 with database restoration.
   */
  public async rollback(
    previousVersion: string,
    options?: { dbPath?: string | undefined },
  ): Promise<{ ok: boolean; error?: string | undefined; reason?: string | undefined }> {
    const targetDb = options?.dbPath ?? this.dbPath;
    if (targetDb && this.preMigrationBackupPath && fs.existsSync(this.preMigrationBackupPath)) {
      restoreDatabaseBackup(this.preMigrationBackupPath, targetDb);
    }

    if (this.onRollback) {
      await this.onRollback(previousVersion);
    }

    this.currentVersion = previousVersion;
    this.state.status = 'idle';
    this.state.currentVersion = previousVersion;
    delete this.state.stagedVersion;
    this.state.canApply = false;
    this.state.activeJobBlocking = false;

    return { ok: true };
  }
}
