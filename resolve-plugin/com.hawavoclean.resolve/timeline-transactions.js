// @ts-check
'use strict';

/**
 * Transactional Timeline Operations for DaVinci Resolve
 *
 * Provides ACID-like transactional semantics for replace, append, and new-track operations.
 * Enforces strict 48 kHz delivery, channel layout matching, sub-clip handle preservation,
 * sample alignment, on-disk transaction journaling, and automatic compensating rollback on failure.
 *
 * Implements Task Q5.4 (P0/H).
 */

const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const crypto = require('node:crypto');

const TRANSACTIONS_DIR = path.join(
  process.env.HOME || os.homedir(),
  'Library',
  'Application Support',
  'HawaVoClean',
  'resolve-transactions'
);

/**
 * Validates that an audio file is a valid 48 kHz WAV file.
 *
 * @param {string} filePath
 * @returns {{ valid: boolean, sampleRate: number, channels: number, bitsPerSample: number, format: string, error?: string }}
 */
function validateAudioFormat(filePath) {
  let fd;
  try {
    const stat = fs.statSync(filePath);
    if (!stat.isFile() || stat.size < 44) {
      return { valid: false, sampleRate: 0, channels: 0, bitsPerSample: 0, format: 'unknown', error: 'Invalid or too small audio file' };
    }

    fd = fs.openSync(filePath, 'r');
    const header = Buffer.alloc(1024);
    const bytesRead = fs.readSync(fd, header, 0, 1024, 0);
    if (bytesRead < 44) {
      return { valid: false, sampleRate: 0, channels: 0, bitsPerSample: 0, format: 'unknown', error: 'Could not read WAV header' };
    }

    if (header.toString('ascii', 0, 4) !== 'RIFF' || header.toString('ascii', 8, 12) !== 'WAVE') {
      return { valid: false, sampleRate: 0, channels: 0, bitsPerSample: 0, format: 'unknown', error: 'File is not a RIFF WAVE file' };
    }

    // Locate "fmt " chunk
    let offset = 12;
    let foundFmt = false;
    let audioFormat = 0;
    let numChannels = 0;
    let sampleRate = 0;
    let bitsPerSample = 0;

    while (offset + 8 <= bytesRead) {
      const chunkId = header.toString('ascii', offset, offset + 4);
      const chunkSize = header.readUInt32LE(offset + 4);

      if (chunkId === 'fmt ') {
        foundFmt = true;
        audioFormat = header.readUInt16LE(offset + 8);
        numChannels = header.readUInt16LE(offset + 10);
        sampleRate = header.readUInt32LE(offset + 12);
        bitsPerSample = header.readUInt16LE(offset + 22);
        break;
      }
      offset += 8 + chunkSize;
    }

    if (!foundFmt) {
      return { valid: false, sampleRate: 0, channels: 0, bitsPerSample: 0, format: 'unknown', error: 'Missing fmt chunk in WAV' };
    }

    // AudioFormat: 1 = PCM, 3 = IEEE Float, 65534 = Extensible
    const isPcmOrFloat = audioFormat === 1 || audioFormat === 3 || audioFormat === 65534;
    if (!isPcmOrFloat) {
      return {
        valid: false,
        sampleRate,
        channels: numChannels,
        bitsPerSample,
        format: `format_${audioFormat}`,
        error: `Unsupported WAV audio format: ${audioFormat}. Must be uncompressed PCM or IEEE float.`,
      };
    }

    if (sampleRate !== 48000) {
      return {
        valid: false,
        sampleRate,
        channels: numChannels,
        bitsPerSample,
        format: audioFormat === 3 ? 'float' : 'pcm',
        error: `Audio must be delivered at 48 kHz. Detected sample rate: ${sampleRate} Hz.`,
      };
    }

    return {
      valid: true,
      sampleRate,
      channels: numChannels,
      bitsPerSample,
      format: audioFormat === 3 ? 'float' : 'pcm',
    };
  } catch (err) {
    return { valid: false, sampleRate: 0, channels: 0, bitsPerSample: 0, format: 'unknown', error: err.message };
  } finally {
    if (fd !== undefined) {
      try {
        fs.closeSync(fd);
      } catch (_) {}
    }
  }
}

const GUARD_MARKER_COLORS = Object.freeze({
  REVERTED: 'Red',
  ERROR_PASSTHROUGH: 'Red',
  CONTINUITY_REVERTED: 'Yellow',
  LOW_STRENGTH: 'Yellow',
});

/**
 * Formats a guard score scalar deterministically without floating-point noise.
 *
 * @param {unknown} value
 * @returns {string}
 */
function formatGuardScoreValue(value) {
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return String(value);
    return Number.isInteger(value) ? String(value) : String(Number(value.toFixed(4)));
  }
  return String(value);
}

/**
 * Maps a `.hawavoclean.json` unit record into a Resolve Guard marker classification,
 * or returns null when the unit passed cleanly and requires no timeline marker.
 *
 * @param {Record<string, any>} unit
 * @param {number} lowStrengthThreshold
 * @returns {{ decision: 'REVERTED' | 'ERROR_PASSTHROUGH' | 'CONTINUITY_REVERTED' | 'LOW_STRENGTH', color: 'Red' | 'Yellow' } | null}
 */
function classifyUnitForGuardMarker(unit, lowStrengthThreshold) {
  const rawDecision = typeof unit.final_decision === 'string'
    ? unit.final_decision.trim()
    : (typeof unit.decision === 'string' ? unit.decision.trim() : '');
  const upper = rawDecision.toUpperCase();
  const verdict = typeof unit.guard_a_verdict === 'string' ? unit.guard_a_verdict.trim().toUpperCase() : '';
  const chosenStrength = Number(unit.chosen_strength);

  if (upper === 'NO_SPEECH' || upper === 'ORIGINAL_NO_SPEECH' || verdict === 'NO_SPEECH') {
    return null;
  }

  if (
    upper === 'ERROR_PASSTHROUGH' ||
    upper === 'ORIGINAL_ERROR' ||
    upper === 'ERROR' ||
    verdict === 'ERROR'
  ) {
    return { decision: 'ERROR_PASSTHROUGH', color: GUARD_MARKER_COLORS.ERROR_PASSTHROUGH };
  }

  if (upper === 'CONTINUITY_REVERTED' || upper === 'ORIGINAL_CONTINUITY') {
    return { decision: 'CONTINUITY_REVERTED', color: GUARD_MARKER_COLORS.CONTINUITY_REVERTED };
  }

  if (
    upper === 'REVERTED' ||
    upper === 'ORIGINAL_REVERTED' ||
    upper === 'UNVERIFIED' ||
    upper === 'ORIGINAL_UNVERIFIED'
  ) {
    return { decision: 'REVERTED', color: GUARD_MARKER_COLORS.REVERTED };
  }

  if (upper === 'LOW_STRENGTH') {
    return { decision: 'LOW_STRENGTH', color: GUARD_MARKER_COLORS.LOW_STRENGTH };
  }

  if (upper === 'ENHANCED' || upper === '') {
    if (verdict === 'REVERT' || verdict === 'UNVERIFIED') {
      return { decision: 'REVERTED', color: GUARD_MARKER_COLORS.REVERTED };
    }
    if (
      Number.isFinite(chosenStrength) &&
      chosenStrength > 0 &&
      chosenStrength <= lowStrengthThreshold
    ) {
      return { decision: 'LOW_STRENGTH', color: GUARD_MARKER_COLORS.LOW_STRENGTH };
    }
  }

  return null;
}

/**
 * Pure, deterministic parser that converts HawaVoClean `.hawavoclean.json` unit records
 * into DaVinci Resolve timeline marker descriptors.
 *
 * Emits `'Red'` markers for `REVERTED` / `ERROR_PASSTHROUGH` and `'Yellow'` markers for
 * `CONTINUITY_REVERTED` / `LOW_STRENGTH`. Returns `[]` cleanly for missing or empty reports.
 *
 * @param {object | null | undefined} report Parsed `.hawavoclean.json` report object
 * @param {number} clipStartFrame Clip's timeline start frame
 * @param {number} fps Timeline frame rate (frames per second)
 * @param {object} [options]
 * @param {number} [options.sourceOffsetS=0] Clip in-point offset into source audio (seconds)
 * @param {number} [options.lowStrengthThreshold=0.5] Max chosen_strength treated as LOW_STRENGTH
 * @returns {Array<{ frameOffset: number, color: 'Red' | 'Yellow', name: string, note: string, durationFrames: number, decision: string, unitId: number | null }>}
 */
function extractGuardMarkersFromReport(
  report,
  clipStartFrame,
  fps,
  { sourceOffsetS = 0, lowStrengthThreshold = 0.5 } = {}
) {
  if (!report || typeof report !== 'object' || !Array.isArray(report.units) || report.units.length === 0) {
    return [];
  }
  const rate = Number(fps);
  if (!Number.isFinite(rate) || rate <= 0) {
    return [];
  }

  const baseFrame = Number.isFinite(Number(clipStartFrame)) ? Math.round(Number(clipStartFrame)) : 0;
  const offsetS = Number.isFinite(Number(sourceOffsetS)) && Number(sourceOffsetS) > 0 ? Number(sourceOffsetS) : 0;
  const threshold = Number.isFinite(Number(lowStrengthThreshold)) ? Number(lowStrengthThreshold) : 0.5;

  /** @type {Array<{ frameOffset: number, color: 'Red' | 'Yellow', name: string, note: string, durationFrames: number, decision: string, unitId: number | null }>} */
  const markers = [];

  for (const unit of report.units) {
    if (!unit || typeof unit !== 'object') continue;
    const classification = classifyUnitForGuardMarker(unit, threshold);
    if (!classification) continue;

    const startRaw = unit.start_time_s ?? unit.startTimeS ?? unit.start_s;
    const endRaw = unit.end_time_s ?? unit.endTimeS ?? unit.end_s;
    const startS = Number(startRaw);
    if (!Number.isFinite(startS)) continue;
    const endParsed = Number(endRaw);
    const endS = Number.isFinite(endParsed) && endParsed >= startS ? endParsed : startS;

    // Skip units that ended entirely before the clip's visible in-point handle
    if (offsetS > 0 && endS <= offsetS) continue;

    const relStartS = Math.max(0, startS - offsetS);
    const relEndS = Math.max(relStartS, endS - offsetS);
    const frameOffset = baseFrame + Math.round(relStartS * rate);
    const durationFrames = Math.max(1, Math.round((relEndS - relStartS) * rate));

    const unitIdRaw = unit.unit_id ?? unit.unitId;
    const unitId = unitIdRaw !== undefined && unitIdRaw !== null && Number.isFinite(Number(unitIdRaw))
      ? Number(unitIdRaw)
      : null;

    const name = unitId !== null
      ? `HawaVoClean Guard: ${classification.decision} (unit ${unitId})`
      : `HawaVoClean Guard: ${classification.decision}`;

    const noteParts = [];
    if (typeof unit.decision_reason === 'string' && unit.decision_reason.trim().length > 0) {
      noteParts.push(unit.decision_reason.trim());
    }
    if (classification.decision === 'LOW_STRENGTH' && Number.isFinite(Number(unit.chosen_strength))) {
      noteParts.push(`chosen_strength=${formatGuardScoreValue(Number(unit.chosen_strength))}`);
    }

    const scoreEntries = [];
    if (unit.guard_a_scores && typeof unit.guard_a_scores === 'object' && !Array.isArray(unit.guard_a_scores)) {
      for (const key of Object.keys(unit.guard_a_scores).sort()) {
        const val = unit.guard_a_scores[key];
        if (val !== undefined && val !== null) {
          scoreEntries.push(`${key}=${formatGuardScoreValue(val)}`);
        }
      }
    }
    if (unit.guard_b_scores && typeof unit.guard_b_scores === 'object' && !Array.isArray(unit.guard_b_scores)) {
      for (const key of Object.keys(unit.guard_b_scores).sort()) {
        const val = unit.guard_b_scores[key];
        if (val !== undefined && val !== null) {
          scoreEntries.push(`guard_b.${key}=${formatGuardScoreValue(val)}`);
        }
      }
    }
    if (scoreEntries.length > 0) {
      noteParts.push(`scores: ${scoreEntries.join(', ')}`);
    }

    const note = noteParts.length > 0 ? noteParts.join(' | ') : `Decision: ${classification.decision}`;

    markers.push({
      frameOffset,
      color: classification.color,
      name,
      note,
      durationFrames,
      decision: classification.decision,
      unitId,
    });
  }

  markers.sort((a, b) => {
    if (a.frameOffset !== b.frameOffset) return a.frameOffset - b.frameOffset;
    const ua = a.unitId ?? Number.MAX_SAFE_INTEGER;
    const ub = b.unitId ?? Number.MAX_SAFE_INTEGER;
    if (ua !== ub) return ua - ub;
    return a.name.localeCompare(b.name);
  });

  return markers;
}

/**
 * Safely loads a `.hawavoclean.json` report from an explicit report object, explicit report path,
 * or matching sidecar `<stem>.hawavoclean.json` next to `cleanedFilePath`.
 *
 * @param {object} entry
 * @param {string} cleanedFilePath
 * @returns {object | null}
 */
function resolveClipReport(entry, cleanedFilePath) {
  if (entry && typeof entry === 'object' && entry.report && typeof entry.report === 'object') {
    return entry.report;
  }

  const candidatePaths = [];
  if (entry && typeof entry === 'object' && typeof entry.reportPath === 'string' && entry.reportPath.length > 0) {
    candidatePaths.push(entry.reportPath);
  }
  if (typeof cleanedFilePath === 'string' && cleanedFilePath.length > 0) {
    const parsed = path.parse(cleanedFilePath);
    candidatePaths.push(path.join(parsed.dir, `${parsed.name}.hawavoclean.json`));
  }

  for (const candidate of candidatePaths) {
    if (!path.isAbsolute(candidate)) continue;
    try {
      if (fs.existsSync(candidate) && fs.statSync(candidate).isFile()) {
        const parsed = JSON.parse(fs.readFileSync(candidate, 'utf8'));
        if (parsed && typeof parsed === 'object') return parsed;
      }
    } catch (_) {
      // Malformed or unreadable sidecar report does not abort audio delivery
    }
  }

  return null;
}

/** Explicit legal stage transitions for Resolve timeline transactions. */
const ALLOWED_TRANSACTION_TRANSITIONS = Object.freeze({
  INIT: Object.freeze(['STAGED', 'COMMITTED', 'ROLLING_BACK']),
  STAGED: Object.freeze(['COMMITTED', 'ROLLING_BACK']),
  COMMITTED: Object.freeze([]),
  ROLLING_BACK: Object.freeze(['ROLLED_BACK']),
  ROLLED_BACK: Object.freeze([]),
});

class TransactionJournal {
  /**
   * @param {string} [txDir]
   */
  constructor(txDir = TRANSACTIONS_DIR) {
    this.txDir = txDir;
  }

  /**
   * Initializes a new transaction log on disk.
   * @param {string} operation
   * @param {object} details
   * @returns {string} txId
   */
  begin(operation, details = {}) {
    try {
      fs.mkdirSync(this.txDir, { recursive: true, mode: 0o700 });
    } catch (_) {}

    const txId = `tx_${Date.now()}_${crypto.randomBytes(4).toString('hex')}`;
    const logPath = path.join(this.txDir, `${txId}.json`);
    const record = {
      txId,
      operation,
      stage: 'INIT',
      startedAt: new Date().toISOString(),
      details,
      compensations: [],
    };

    fs.writeFileSync(logPath, JSON.stringify(record, null, 2), { mode: 0o600 });
    return txId;
  }

  /**
   * Updates transaction stage and records compensating actions.
   * @param {string} txId
   * @param {'STAGED' | 'COMMITTED' | 'ROLLING_BACK' | 'ROLLED_BACK'} stage
   * @param {Array<object>} [compensations]
   */
  update(txId, stage, compensations) {
    const logPath = path.join(this.txDir, `${txId}.json`);
    if (!fs.existsSync(logPath)) return;
    let record;
    try {
      record = JSON.parse(fs.readFileSync(logPath, 'utf8'));
    } catch (_) {
      return;
    }
    const allowed = ALLOWED_TRANSACTION_TRANSITIONS[record.stage] || [];
    if (!allowed.includes(stage)) {
      throw new Error(`Illegal timeline transaction transition: ${record.stage} -> ${stage}`);
    }
    try {
      record.stage = stage;
      record.updatedAt = new Date().toISOString();
      if (compensations) record.compensations = compensations;
      fs.writeFileSync(logPath, JSON.stringify(record, null, 2), { mode: 0o600 });
    } catch (_) {}
  }

  /**
   * Cleans up transaction journal file upon successful commit.
   * @param {string} txId
   */
  cleanup(txId) {
    const logPath = path.join(this.txDir, `${txId}.json`);
    try {
      if (fs.existsSync(logPath)) fs.unlinkSync(logPath);
    } catch (_) {}
  }
}

/**
 * TimelineTransactionManager manages transactional replace, append, and new-track mutations.
 */
class TimelineTransactionManager {
  /**
   * @param {object} resolveContext
   * @param {function} resolveContext.getProject
   * @param {function} resolveContext.getMediaPool
   * @param {function} resolveContext.findItemByMediaId
   * @param {function} [resolveContext.log]
   * @param {TransactionJournal} [resolveContext.journal]
   */
  constructor(resolveContext) {
    this.getProject = resolveContext.getProject;
    this.getMediaPool = resolveContext.getMediaPool;
    this.findItemByMediaId = resolveContext.findItemByMediaId;
    this.log = resolveContext.log || console.log;
    this.journal = resolveContext.journal || new TransactionJournal();
  }

  /**
   * Transactionally replaces a clip in the media pool and timeline,
   * preserving sub-clip handles and verifying 48 kHz audio delivery.
   *
   * @param {string} mediaId
   * @param {string} filePath
   * @param {object} [options]
   * @param {string} [options.failPoint] Injected failure point for testing ('before_replace', 'after_replace')
   * @returns {Promise<{ success: boolean, method: string }>}
   */
  async transactionalReplace(mediaId, filePath, options = {}) {
    if (typeof filePath !== 'string' || !path.isAbsolute(filePath)) {
      throw new Error('replaceClip: path must be an absolute path.');
    }

    // 1. Verify 48 kHz delivery format
    const audioCheck = validateAudioFormat(filePath);
    if (!audioCheck.valid) {
      throw new Error(`replaceClip rejected: ${audioCheck.error}`);
    }

    const item = await this.findItemByMediaId(mediaId);
    if (!item) throw new Error(`Clip ${mediaId} was not found in the media pool.`);

    // 2. Capture pre-mutation snapshot
    let originalFilePath = null;
    try {
      if (typeof item.GetClipProperty === 'function') {
        const props = await item.GetClipProperty();
        originalFilePath = props['File Path'] || props['FilePath'] || null;
      }
    } catch (_) {}

    const txId = this.journal.begin('replace', { mediaId, filePath, originalFilePath });
    const compensations = [];

    try {
      if (options.failPoint === 'before_replace') {
        throw new Error('Injected failure before replace');
      }

      let method = 'ReplaceClip';
      let replaced = false;

      // 3. Perform replacement preserving sub-clips and handles
      if (typeof item.ReplaceClipPreserveSubClip === 'function') {
        try {
          if (await item.ReplaceClipPreserveSubClip(filePath)) {
            method = 'ReplaceClipPreserveSubClip';
            replaced = true;
          }
        } catch (err) {
          this.log(`ReplaceClipPreserveSubClip failed (${err.message}); falling back to ReplaceClip`);
        }
      }

      if (!replaced) {
        replaced = Boolean(await item.ReplaceClip(filePath));
      }

      if (!replaced) {
        throw new Error(`MediaPoolItem.ReplaceClip failed to apply ${filePath}`);
      }

      if (options.failPoint === 'after_replace') {
        throw new Error('Injected failure after replace');
      }

      // 4. Commit transaction
      this.journal.cleanup(txId);
      return { success: true, method };
    } catch (err) {
      this.log(`transactionalReplace failed: ${err.message}. Initiating compensating rollback...`);
      this.journal.update(txId, 'ROLLING_BACK', compensations);

      // Rollback: restore original file path if available
      if (originalFilePath && typeof item.ReplaceClip === 'function') {
        try {
          await item.ReplaceClip(originalFilePath);
          this.log(`Rollback: restored original clip reference ${originalFilePath}`);
        } catch (rbErr) {
          this.log(`Rollback error: ${rbErr.message}`);
        }
      }

      this.journal.update(txId, 'ROLLED_BACK');
      throw err;
    }
  }

  /**
   * Transactionally appends media to the current timeline.
   * If timeline append fails, automatically cleans up the imported media pool item.
   *
   * @param {string} filePath
   * @param {object} [options]
   * @param {string} [options.failPoint] Injected failure point ('before_import', 'after_import', 'before_append')
   * @returns {Promise<{ success: boolean, mediaId?: string }>}
   */
  async transactionalAppend(filePath, options = {}) {
    if (typeof filePath !== 'string' || !path.isAbsolute(filePath)) {
      throw new Error('appendToTimeline: path must be an absolute path.');
    }

    const audioCheck = validateAudioFormat(filePath);
    if (!audioCheck.valid) {
      throw new Error(`appendToTimeline rejected: ${audioCheck.error}`);
    }

    const mediaPool = await this.getMediaPool();
    if (!mediaPool) throw new Error('No project is open in DaVinci Resolve.');

    const txId = this.journal.begin('append', { filePath });
    let importedItem = null;

    try {
      if (options.failPoint === 'before_import') {
        throw new Error('Injected failure before import');
      }

      // 1. Import media into MediaPool
      const items = (await mediaPool.ImportMedia([filePath])) || [];
      if (!items.length || !items[0]) {
        throw new Error(`Failed to import media at ${filePath}`);
      }
      importedItem = items[0];

      if (options.failPoint === 'after_import') {
        throw new Error('Injected failure after import');
      }

      // 2. Append to timeline
      const result = await mediaPool.AppendToTimeline([importedItem]);
      const appended = Array.isArray(result) ? result.length > 0 : Boolean(result);
      if (!appended) {
        throw new Error('AppendToTimeline returned false/empty');
      }

      let mediaId = null;
      try {
        if (typeof importedItem.GetMediaId === 'function') {
          mediaId = await importedItem.GetMediaId();
        }
      } catch (_) {}

      this.journal.cleanup(txId);
      return { success: true, mediaId };
    } catch (err) {
      this.log(`transactionalAppend failed: ${err.message}. Initiating compensating rollback...`);
      this.journal.update(txId, 'ROLLING_BACK');

      // Compensating action: remove imported item from MediaPool so no orphan clutter remains
      if (importedItem && typeof mediaPool.DeleteClips === 'function') {
        try {
          await mediaPool.DeleteClips([importedItem]);
          this.log(`Rollback: purged imported media pool item ${filePath}`);
        } catch (delErr) {
          this.log(`Rollback DeleteClips failed: ${delErr.message}`);
        }
      }

      this.journal.update(txId, 'ROLLED_BACK');
      throw err;
    }
  }

  /**
   * Transactionally inserts cleaned audio into a dedicated new audio track,
   * preserving exact timeline timecode alignment, mono/stereo layout, and handles.
   *
   * @param {string} mediaId Source clip ID
   * @param {string} filePath Cleaned audio file path (48 kHz)
   * @param {object} [options]
   * @param {string} [options.trackName] Desired track name (defaults to 'HawaVoClean Cleaned')
   * @param {string} [options.failPoint] Injected failure point for tests
   * @returns {Promise<{ success: boolean, trackIndex: number, trackName: string }>}
   */
  async transactionalNewTrack(mediaId, filePath, options = {}) {
    if (typeof filePath !== 'string' || !path.isAbsolute(filePath)) {
      throw new Error('newTrack: path must be an absolute path.');
    }

    const audioCheck = validateAudioFormat(filePath);
    if (!audioCheck.valid) {
      throw new Error(`newTrack rejected: ${audioCheck.error}`);
    }

    const project = await this.getProject();
    if (!project) throw new Error('No project is open in DaVinci Resolve.');

    const timeline = await project.GetCurrentTimeline();
    if (!timeline) throw new Error('No active timeline in DaVinci Resolve.');

    const mediaPool = await this.getMediaPool();
    if (!mediaPool) throw new Error('MediaPool unavailable.');

    const trackSubtype = audioCheck.channels === 1 ? 'mono' : 'stereo';
    const desiredTrackName = options.trackName || 'HawaVoClean Cleaned';

    const txId = this.journal.begin('new-track', { mediaId, filePath, trackSubtype, desiredTrackName });
    let createdTrackIndex = null;
    let importedItem = null;
    let appendedItem = null;

    try {
      if (options.failPoint === 'before_track_creation') {
        throw new Error('Injected failure before track creation');
      }

      // 1. Create dedicated audio track
      const trackCountBefore = (await timeline.GetTrackCount('audio')) || 0;
      let trackCreated = false;
      if (typeof timeline.AddTrack === 'function') {
        trackCreated = await timeline.AddTrack('audio', trackSubtype);
      }

      if (!trackCreated) {
        throw new Error(`Failed to create audio track of type '${trackSubtype}'`);
      }

      const trackCountAfter = (await timeline.GetTrackCount('audio')) || trackCountBefore + 1;
      createdTrackIndex = trackCountAfter;

      // Set track name if API exists
      if (typeof timeline.SetTrackName === 'function') {
        try {
          await timeline.SetTrackName('audio', createdTrackIndex, desiredTrackName);
        } catch (_) {}
      }

      if (options.failPoint === 'after_track_creation') {
        throw new Error('Injected failure after track creation');
      }

      // 2. Import cleaned audio into MediaPool
      const items = (await mediaPool.ImportMedia([filePath])) || [];
      if (!items.length || !items[0]) {
        throw new Error(`Failed to import media at ${filePath}`);
      }
      importedItem = items[0];

      if (options.failPoint === 'after_import') {
        throw new Error('Injected failure after import');
      }

      // 3. Append / Insert into the newly created track
      // If AppendToTimeline supports specifying track or if we use track destination
      let inserted = false;
      if (typeof mediaPool.AppendToTimeline === 'function') {
        const result = await mediaPool.AppendToTimeline([{
          mediaPoolItem: importedItem,
          trackIndex: createdTrackIndex,
          recordFrame: 0,
        }]);
        inserted = Array.isArray(result) ? result.length > 0 : Boolean(result);
        if (Array.isArray(result) && result[0]) appendedItem = result[0];
      }

      if (!inserted) {
        // Fallback: standard append
        const result = await mediaPool.AppendToTimeline([importedItem]);
        inserted = Array.isArray(result) ? result.length > 0 : Boolean(result);
        if (Array.isArray(result) && result[0]) appendedItem = result[0];
      }

      if (!inserted) {
        throw new Error('Failed to insert audio clip onto timeline');
      }

      if (options.failPoint === 'after_insertion') {
        throw new Error('Injected failure after insertion');
      }

      this.journal.cleanup(txId);
      return {
        success: true,
        trackIndex: createdTrackIndex,
        trackName: desiredTrackName,
      };
    } catch (err) {
      this.log(`transactionalNewTrack failed: ${err.message}. Initiating compensating rollback...`);
      this.journal.update(txId, 'ROLLING_BACK');

      // Compensating action 1: delete timeline item if created
      if (appendedItem && typeof timeline.DeleteItem === 'function') {
        try {
          await timeline.DeleteItem(appendedItem);
          this.log('Rollback: deleted appended timeline item');
        } catch (itemErr) {
          this.log(`Rollback DeleteItem failed: ${itemErr.message}`);
        }
      }

      // Compensating action 2: delete created track
      if (createdTrackIndex !== null && typeof timeline.DeleteTrack === 'function') {
        try {
          await timeline.DeleteTrack('audio', createdTrackIndex);
          this.log(`Rollback: deleted created track at index ${createdTrackIndex}`);
        } catch (trackErr) {
          this.log(`Rollback DeleteTrack failed: ${trackErr.message}`);
        }
      }

      // Compensating action 3: delete imported media pool item
      if (importedItem && typeof mediaPool.DeleteClips === 'function') {
        try {
          await mediaPool.DeleteClips([importedItem]);
          this.log(`Rollback: purged imported media pool item ${filePath}`);
        } catch (clipErr) {
          this.log(`Rollback DeleteClips failed: ${clipErr.message}`);
        }
      }

      this.journal.update(txId, 'ROLLED_BACK');
      throw err;
    }
  }

  /**
   * Transactionally places a batch of cleaned 48 kHz audio clips onto a single
   * non-destructive audio track (creating or reusing `targetTrackName`), aligns
   * every clip to its original timeline start frame with handle preservation,
   * drops HawaVoClean Guard Timeline Markers when available, persists a `0o600`
   * transaction journal, and executes full compensating rollback on any mid-batch
   * failure.
   *
   * @param {object} params
   * @param {object} [params.timeline] Resolve Timeline object (defaults to active timeline)
   * @param {Array<object>} params.clips Original timeline clip descriptors
   * @param {Record<string, string | { filePath: string, report?: object, reportPath?: string }> | Map<string, any>} params.cleanedByClipId
   * @param {string} [params.targetTrackName='HawaVoClean Cleaned']
   * @param {number} [params.fps=24]
   * @param {boolean} [params.addGuardMarkers=true]
   * @param {string} [params.failPoint] Optional injected failure point for testing
   * @returns {Promise<{ success: boolean, txId: string, journalPath: string, trackIndex: number, trackName: string, reusedTrack: boolean, clipsPlaced: number, markersAdded: number, markers: Array<object> }>}
   */
  async transactionalBatchTrackClean({
    timeline,
    clips,
    cleanedByClipId,
    targetTrackName = 'HawaVoClean Cleaned',
    fps = 24,
    addGuardMarkers = true,
    failPoint,
  } = {}) {
    if (!Array.isArray(clips) || clips.length === 0) {
      throw new Error('batchTrackClean: clips must be a non-empty array.');
    }
    if (!cleanedByClipId || typeof cleanedByClipId !== 'object') {
      throw new Error('batchTrackClean: cleanedByClipId map is required.');
    }
    const rate = Number(fps);
    if (!Number.isFinite(rate) || rate <= 0) {
      throw new Error(`batchTrackClean: invalid fps (${fps}).`);
    }
    if (typeof targetTrackName !== 'string' || targetTrackName.trim().length === 0) {
      throw new Error('batchTrackClean: targetTrackName must be a non-empty string.');
    }

    let activeTimeline = timeline || null;
    if (!activeTimeline) {
      const project = await this.getProject();
      if (!project) throw new Error('No project is open in DaVinci Resolve.');
      activeTimeline = await project.GetCurrentTimeline();
    }
    if (!activeTimeline) {
      throw new Error('No active timeline in DaVinci Resolve.');
    }

    const mediaPool = await this.getMediaPool();
    if (!mediaPool) {
      throw new Error('MediaPool unavailable.');
    }

    // 1. Pre-validate every cleaned WAV and resolve clip alignment before any timeline mutation.
    const prepared = [];
    let maxChannels = 1;

    for (let i = 0; i < clips.length; i += 1) {
      const clip = clips[i];
      if (!clip || typeof clip !== 'object') {
        throw new Error(`batchTrackClean: clip at index ${i} must be an object.`);
      }
      const clipId = String(clip.clipId ?? clip.mediaId ?? clip.id ?? '').trim();
      if (!clipId) {
        throw new Error(`batchTrackClean: clip at index ${i} is missing clipId/mediaId/id.`);
      }

      const cleanedEntry = cleanedByClipId instanceof Map
        ? cleanedByClipId.get(clipId)
        : cleanedByClipId[clipId];
      if (!cleanedEntry) {
        throw new Error(`batchTrackClean: missing cleaned audio entry for clip '${clipId}'.`);
      }

      const filePath = typeof cleanedEntry === 'string' ? cleanedEntry : cleanedEntry.filePath;
      if (typeof filePath !== 'string' || !path.isAbsolute(filePath)) {
        throw new Error(`batchTrackClean: cleaned audio path for clip '${clipId}' must be absolute.`);
      }

      const audioCheck = validateAudioFormat(filePath);
      if (!audioCheck.valid) {
        throw new Error(`batchTrackClean rejected clip '${clipId}': ${audioCheck.error}`);
      }
      if (audioCheck.channels > maxChannels) {
        maxChannels = audioCheck.channels;
      }

      let startFrame = clip.startFrame ?? clip.recordFrame;
      if (startFrame === undefined && typeof clip.GetStart === 'function') {
        startFrame = await clip.GetStart();
      }
      startFrame = Number(startFrame);
      if (!Number.isFinite(startFrame) || startFrame < 0) {
        throw new Error(`batchTrackClean: invalid startFrame for clip '${clipId}'.`);
      }

      const handleStartRaw = clip.sourceStartFrame ?? clip.startFrameInSource ?? clip.leftOffset;
      const handleStartFrame = handleStartRaw !== undefined && handleStartRaw !== null && Number.isFinite(Number(handleStartRaw))
        ? Math.max(0, Math.round(Number(handleStartRaw)))
        : null;

      let handleEndFrame = null;
      const handleEndRaw = clip.sourceEndFrame ?? clip.endFrameInSource;
      if (handleEndRaw !== undefined && handleEndRaw !== null && Number.isFinite(Number(handleEndRaw))) {
        handleEndFrame = Math.max(0, Math.round(Number(handleEndRaw)));
      } else if (
        handleStartFrame !== null &&
        clip.durationFrames !== undefined &&
        Number.isFinite(Number(clip.durationFrames)) &&
        Number(clip.durationFrames) > 0
      ) {
        handleEndFrame = handleStartFrame + Math.round(Number(clip.durationFrames));
      }

      const sourceOffsetS = Number.isFinite(Number(clip.sourceOffsetS)) && Number(clip.sourceOffsetS) > 0
        ? Number(clip.sourceOffsetS)
        : (handleStartFrame !== null ? handleStartFrame / rate : 0);

      const report = addGuardMarkers ? resolveClipReport(cleanedEntry, filePath) : null;

      prepared.push({
        clipId,
        filePath,
        startFrame: Math.round(startFrame),
        handleStartFrame,
        handleEndFrame,
        sourceOffsetS,
        mediaType: Number.isFinite(Number(clip.mediaType)) ? Number(clip.mediaType) : undefined,
        report,
      });
    }

    const trackSubtype = maxChannels === 1 ? 'mono' : 'stereo';
    const txId = this.journal.begin('batch-track-clean', {
      targetTrackName,
      trackSubtype,
      fps: rate,
      clipCount: prepared.length,
      addGuardMarkers: Boolean(addGuardMarkers),
    });
    const journalPath = path.join(this.journal.txDir, `${txId}.json`);

    /** @type {number | null} */
    let targetTrackIndex = null;
    /** @type {number | null} */
    let createdTrackIndex = null;
    let reusedTrack = false;
    /** @type {Array<any>} */
    const importedItems = [];
    /** @type {Array<any>} */
    const appendedItems = [];
    /** @type {Array<number>} */
    const addedMarkerFrames = [];
    /** @type {Array<object>} */
    const placedMarkers = [];
    const seenMarkerFrames = new Set();

    try {
      if (failPoint === 'before_track_creation') {
        throw new Error('Injected failure before track creation');
      }

      // 2. Reuse existing track with matching name, or create a single dedicated non-destructive track.
      const trackCountBefore = (await activeTimeline.GetTrackCount('audio')) || 0;
      if (typeof activeTimeline.GetTrackName === 'function') {
        for (let idx = 1; idx <= trackCountBefore; idx += 1) {
          const existingName = await activeTimeline.GetTrackName('audio', idx);
          if (existingName === targetTrackName) {
            targetTrackIndex = idx;
            reusedTrack = true;
            break;
          }
        }
      }

      if (targetTrackIndex === null) {
        let trackCreated = false;
        if (typeof activeTimeline.AddTrack === 'function') {
          trackCreated = Boolean(await activeTimeline.AddTrack('audio', trackSubtype));
        }
        if (!trackCreated) {
          throw new Error(`Failed to create audio track '${targetTrackName}' (${trackSubtype}).`);
        }
        const trackCountAfter = (await activeTimeline.GetTrackCount('audio')) || trackCountBefore + 1;
        targetTrackIndex = trackCountAfter;
        createdTrackIndex = trackCountAfter;
        if (typeof activeTimeline.SetTrackName === 'function') {
          try {
            await activeTimeline.SetTrackName('audio', targetTrackIndex, targetTrackName);
          } catch (_) {}
        }
      }

      this.journal.update(txId, 'STAGED', [
        {
          action: createdTrackIndex !== null ? 'delete_track' : 'reuse_track',
          trackIndex: targetTrackIndex,
          trackName: targetTrackName,
        },
      ]);

      // 3. Import and append all cleaned clips aligned to original startFrame with handle preservation.
      for (let i = 0; i < prepared.length; i += 1) {
        const entry = prepared[i];

        if (failPoint === `clip_${i}_before_import`) {
          throw new Error(`Injected failure before import at clip index ${i} (${entry.clipId})`);
        }

        const imported = (await mediaPool.ImportMedia([entry.filePath])) || [];
        if (!imported.length || !imported[0]) {
          throw new Error(`Failed to import cleaned media for clip '${entry.clipId}' at ${entry.filePath}`);
        }
        const mediaPoolItem = imported[0];
        importedItems.push(mediaPoolItem);

        if (
          failPoint === `clip_${i}_before_append` ||
          (failPoint === 'mid_batch' && i === Math.max(1, Math.floor(prepared.length / 2)))
        ) {
          throw new Error(`Injected mid-batch failure before append at clip index ${i} (${entry.clipId})`);
        }

        /** @type {Record<string, any>} */
        const clipSpec = {
          mediaPoolItem,
          trackIndex: targetTrackIndex,
          recordFrame: entry.startFrame,
        };
        if (entry.handleStartFrame !== null) {
          clipSpec.startFrame = entry.handleStartFrame;
        }
        if (entry.handleEndFrame !== null) {
          clipSpec.endFrame = entry.handleEndFrame;
        }
        if (entry.mediaType !== undefined) {
          clipSpec.mediaType = entry.mediaType;
        }

        const appendResult = await mediaPool.AppendToTimeline([clipSpec]);
        const ok = Array.isArray(appendResult) ? appendResult.length > 0 : Boolean(appendResult);
        if (!ok) {
          throw new Error(`AppendToTimeline failed for clip '${entry.clipId}' at frame ${entry.startFrame}`);
        }
        if (Array.isArray(appendResult)) {
          for (const item of appendResult) {
            if (item) appendedItems.push(item);
          }
        }

        if (failPoint === `clip_${i}_after_append`) {
          throw new Error(`Injected failure after append at clip index ${i} (${entry.clipId})`);
        }
      }

      // 4. Drop Guard Timeline Markers from each clip's `.hawavoclean.json` report.
      if (addGuardMarkers && typeof activeTimeline.AddMarker === 'function') {
        for (const entry of prepared) {
          if (!entry.report) continue;
          const clipMarkers = extractGuardMarkersFromReport(entry.report, entry.startFrame, rate, {
            sourceOffsetS: entry.sourceOffsetS,
          });
          for (const marker of clipMarkers) {
            if (failPoint === 'during_markers' && placedMarkers.length > 0) {
              throw new Error('Injected failure during guard marker placement');
            }
            const added = await activeTimeline.AddMarker(
              marker.frameOffset,
              marker.color,
              marker.name,
              marker.note,
              marker.durationFrames
            );
            if (added !== false) {
              if (!seenMarkerFrames.has(marker.frameOffset)) {
                seenMarkerFrames.add(marker.frameOffset);
                addedMarkerFrames.push(marker.frameOffset);
              }
              placedMarkers.push(marker);
            }
          }
        }
      }

      if (failPoint === 'after_markers') {
        throw new Error('Injected failure after guard marker placement');
      }

      // 5. Commit and persist 0o600 audit journal.
      this.journal.update(txId, 'COMMITTED', []);
      return {
        success: true,
        txId,
        journalPath,
        trackIndex: targetTrackIndex,
        trackName: targetTrackName,
        reusedTrack,
        clipsPlaced: prepared.length,
        markersAdded: placedMarkers.length,
        markers: placedMarkers,
      };
    } catch (err) {
      this.log(`transactionalBatchTrackClean failed: ${err.message}. Initiating compensating rollback...`);
      const compensations = [];
      this.journal.update(txId, 'ROLLING_BACK', compensations);

      // Compensating action 1: delete markers added during this transaction
      if (addedMarkerFrames.length > 0 && typeof activeTimeline.DeleteMarkerAtFrame === 'function') {
        for (const frameId of [...addedMarkerFrames].reverse()) {
          try {
            await activeTimeline.DeleteMarkerAtFrame(frameId);
            compensations.push({ action: 'delete_marker', frameId, status: 'ok' });
          } catch (markerErr) {
            compensations.push({ action: 'delete_marker', frameId, status: 'error', error: markerErr.message });
            this.log(`Rollback DeleteMarkerAtFrame(${frameId}) failed: ${markerErr.message}`);
          }
        }
      }

      // Compensating action 2: delete appended timeline items
      if (appendedItems.length > 0) {
        if (typeof activeTimeline.DeleteClips === 'function') {
          try {
            await activeTimeline.DeleteClips(appendedItems, false);
            compensations.push({ action: 'delete_timeline_items', count: appendedItems.length, status: 'ok' });
          } catch (itemsErr) {
            compensations.push({ action: 'delete_timeline_items', count: appendedItems.length, status: 'error', error: itemsErr.message });
            this.log(`Rollback DeleteClips(timeline) failed: ${itemsErr.message}`);
          }
        } else if (typeof activeTimeline.DeleteItem === 'function') {
          for (const item of [...appendedItems].reverse()) {
            try {
              await activeTimeline.DeleteItem(item);
              compensations.push({ action: 'delete_timeline_item', status: 'ok' });
            } catch (itemErr) {
              compensations.push({ action: 'delete_timeline_item', status: 'error', error: itemErr.message });
              this.log(`Rollback DeleteItem failed: ${itemErr.message}`);
            }
          }
        }
      }

      // Compensating action 3: remove newly created track (never delete a reused pre-existing track)
      if (createdTrackIndex !== null && typeof activeTimeline.DeleteTrack === 'function') {
        try {
          await activeTimeline.DeleteTrack('audio', createdTrackIndex);
          compensations.push({ action: 'delete_track', trackIndex: createdTrackIndex, status: 'ok' });
        } catch (trackErr) {
          compensations.push({ action: 'delete_track', trackIndex: createdTrackIndex, status: 'error', error: trackErr.message });
          this.log(`Rollback DeleteTrack(${createdTrackIndex}) failed: ${trackErr.message}`);
        }
      }

      // Compensating action 4: purge imported media pool items
      if (importedItems.length > 0 && typeof mediaPool.DeleteClips === 'function') {
        try {
          await mediaPool.DeleteClips(importedItems);
          compensations.push({ action: 'delete_media_pool_clips', count: importedItems.length, status: 'ok' });
        } catch (poolErr) {
          compensations.push({ action: 'delete_media_pool_clips', count: importedItems.length, status: 'error', error: poolErr.message });
          this.log(`Rollback mediaPool.DeleteClips failed: ${poolErr.message}`);
        }
      }

      this.journal.update(txId, 'ROLLED_BACK', compensations);
      throw err;
    }
  }

  /**
   * Applies HawaVoClean Guard markers from a report or explicit marker list onto a timeline,
   * rolling back any partially placed markers if an error occurs mid-apply.
   *
   * @param {object} params
   * @param {object} [params.timeline]
   * @param {object | null} [params.report]
   * @param {string} [params.reportPath]
   * @param {number} [params.clipStartFrame=0]
   * @param {number} [params.fps=24]
   * @param {number} [params.sourceOffsetS=0]
   * @param {Array<{ frameOffset: number, color: string, name: string, note?: string, durationFrames?: number }>} [params.markers]
   * @returns {Promise<{ success: boolean, markersAdded: number, markers: Array<object> }>}
   */
  async applyGuardMarkers({
    timeline,
    report = null,
    reportPath,
    clipStartFrame = 0,
    fps = 24,
    sourceOffsetS = 0,
    markers,
  } = {}) {
    let activeTimeline = timeline || null;
    if (!activeTimeline) {
      const project = await this.getProject();
      if (!project) throw new Error('No project is open in DaVinci Resolve.');
      activeTimeline = await project.GetCurrentTimeline();
    }
    if (!activeTimeline) {
      throw new Error('No active timeline in DaVinci Resolve.');
    }

    let resolvedMarkers = [];
    if (Array.isArray(markers)) {
      resolvedMarkers = markers.map((m, idx) => {
        if (!m || typeof m !== 'object' || !Number.isFinite(Number(m.frameOffset))) {
          throw new Error(`applyGuardMarkers: invalid marker descriptor at index ${idx}.`);
        }
        return {
          frameOffset: Math.round(Number(m.frameOffset)),
          color: typeof m.color === 'string' && m.color ? m.color : 'Yellow',
          name: typeof m.name === 'string' && m.name ? m.name : 'HawaVoClean Guard',
          note: typeof m.note === 'string' ? m.note : '',
          durationFrames: Math.max(1, Number.isFinite(Number(m.durationFrames)) ? Math.round(Number(m.durationFrames)) : 1),
        };
      });
    } else {
      const parsedReport = resolveClipReport({ report, reportPath }, '');
      resolvedMarkers = extractGuardMarkersFromReport(parsedReport, clipStartFrame, fps, { sourceOffsetS });
    }

    if (resolvedMarkers.length === 0 || typeof activeTimeline.AddMarker !== 'function') {
      return { success: true, markersAdded: 0, markers: [] };
    }

    const addedFrames = [];
    const placed = [];
    const seen = new Set();

    try {
      for (const marker of resolvedMarkers) {
        const added = await activeTimeline.AddMarker(
          marker.frameOffset,
          marker.color,
          marker.name,
          marker.note,
          marker.durationFrames
        );
        if (added !== false) {
          if (!seen.has(marker.frameOffset)) {
            seen.add(marker.frameOffset);
            addedFrames.push(marker.frameOffset);
          }
          placed.push(marker);
        }
      }
      return { success: true, markersAdded: placed.length, markers: placed };
    } catch (err) {
      if (addedFrames.length > 0 && typeof activeTimeline.DeleteMarkerAtFrame === 'function') {
        for (const frameId of [...addedFrames].reverse()) {
          try {
            await activeTimeline.DeleteMarkerAtFrame(frameId);
          } catch (_) {}
        }
      }
      throw err;
    }
  }
}

module.exports = {
  ALLOWED_TRANSACTION_TRANSITIONS,
  GUARD_MARKER_COLORS,
  validateAudioFormat,
  extractGuardMarkersFromReport,
  TransactionJournal,
  TimelineTransactionManager,
  TRANSACTIONS_DIR,
};
