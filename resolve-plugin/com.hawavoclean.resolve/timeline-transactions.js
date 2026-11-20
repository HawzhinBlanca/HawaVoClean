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

const MAX_MARKER_NAME_LEN = 256;
const MAX_MARKER_NOTE_LEN = 2048;
const MAX_IPC_MARKER_NOTE_LEN = 10_240;
const MAX_MARKER_COLOR_LEN = 64;
const MAX_SCORE_VALUE_LEN = 128;
const MAX_REPORT_FILE_BYTES = 16 * 1024 * 1024;
const MAX_BATCH_CLIPS = 1000;
const MAX_GUARD_MARKERS = 5000;
const MAX_GUARD_REPORT_UNITS = 10_000;
const MAX_PATH_LEN = 4096;
const MAX_TRACK_NAME_LEN = 256;
const MAX_CLIP_ID_LEN = 512;
const MAX_FPS = 1000;

/**
 * Safely extracts an error message from any thrown value (including non-Error/null throws).
 *
 * @param {unknown} err
 * @returns {string}
 */
function errorMessage(err) {
  if (err && typeof err === 'object' && 'message' in err) {
    return String(/** @type {{ message: unknown }} */ (err).message);
  }
  return String(err);
}

/**
 * @param {unknown} value
 * @returns {value is Record<string, any>}
 */
function isPlainObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

/**
 * Parses a finite numeric value while rejecting null, empty strings, and booleans.
 *
 * @param {unknown} value
 * @returns {number | null}
 */
function parseFiniteNumber(value) {
  if (value === null || value === undefined || value === '' || typeof value === 'boolean' || typeof value === 'symbol') {
    return null;
  }
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

/**
 * Resolves the active timeline's frame rate via `timeline.GetSetting('timelineFrameRate')`
 * when no explicit FPS was provided, falling back to 24 fps only when unavailable.
 *
 * @param {any} timeline
 * @param {unknown} [explicitFps]
 * @returns {Promise<number>}
 */
async function resolveTimelineFps(timeline, explicitFps) {
  if (explicitFps !== undefined) {
    const parsed = parseFiniteNumber(explicitFps);
    if (parsed !== null && parsed > 0 && parsed <= MAX_FPS) {
      return parsed;
    }
  }
  if (timeline && typeof timeline.GetSetting === 'function') {
    try {
      const raw = await timeline.GetSetting('timelineFrameRate');
      const parsed = parseFiniteNumber(raw);
      if (parsed !== null && parsed > 0 && parsed <= MAX_FPS) {
        return parsed;
      }
    } catch (_) {
      /* fall through to default */
    }
  }
  return 24;
}

/**
 * Normalizes a clip mediaType value into DaVinci Resolve's numeric AppendToTimeline mediaType
 * (`1` = video only, `2` = audio only) when supplied as a number or `'video'`/`'audio'`.
 *
 * @param {unknown} value
 * @returns {number | undefined}
 */
function normalizeResolveMediaType(value) {
  if (typeof value === 'string') {
    const lower = value.trim().toLowerCase();
    if (lower === 'audio') return 2;
    if (lower === 'video') return 1;
  }
  const parsed = parseFiniteNumber(value);
  return parsed !== null ? parsed : undefined;
}

/**
 * Searches audio and video tracks on a timeline to find a timeline item matching `mediaId`
 * so single-clip operations can recover exact `startFrame`, `leftOffset`, and `durationFrames`.
 *
 * @param {any} timeline
 * @param {string} mediaId
 * @returns {Promise<{ startFrame: number, leftOffset: number | null, durationFrames: number | null, trackIndex: number, mediaType: number } | null>}
 */
async function findTimelineClipTiming(timeline, mediaId) {
  if (!timeline || typeof mediaId !== 'string' || !mediaId) return null;

  const inspectItem = async (tItem, trackIndex, mediaType) => {
    if (!tItem || typeof tItem.GetMediaPoolItem !== 'function') return null;
    try {
      const mpItem = await tItem.GetMediaPoolItem();
      if (!mpItem || typeof mpItem.GetMediaId !== 'function') return null;
      const itemId = String((await mpItem.GetMediaId()) ?? '');
      if (itemId !== mediaId) return null;
      const startParsed = typeof tItem.GetStart === 'function' ? parseFiniteNumber(await tItem.GetStart()) : null;
      if (startParsed === null || startParsed < 0) return null;
      const endParsed = typeof tItem.GetEnd === 'function' ? parseFiniteNumber(await tItem.GetEnd()) : null;
      const durParsed = typeof tItem.GetDuration === 'function' ? parseFiniteNumber(await tItem.GetDuration()) : null;
      const leftParsed = typeof tItem.GetLeftOffset === 'function' ? parseFiniteNumber(await tItem.GetLeftOffset()) : null;
      const durationFrames = durParsed !== null && durParsed > 0
        ? Math.round(durParsed)
        : (endParsed !== null && endParsed > startParsed ? Math.round(endParsed - startParsed) : null);
      return {
        startFrame: Math.round(startParsed),
        leftOffset: leftParsed !== null && leftParsed >= 0 ? Math.round(leftParsed) : null,
        durationFrames,
        trackIndex,
        mediaType,
      };
    } catch (_) {
      return null;
    }
  };

  for (const [trackType, mediaType] of [['audio', 2], ['video', 1]]) {
    if (typeof timeline.GetTrackCount !== 'function' || typeof timeline.GetItemListInTrack !== 'function') continue;
    let trackCount = 0;
    try {
      trackCount = Number(await timeline.GetTrackCount(trackType)) || 0;
    } catch (_) {
      trackCount = 0;
    }
    for (let trackIdx = 1; trackIdx <= trackCount; trackIdx += 1) {
      let items = [];
      try {
        items = (await timeline.GetItemListInTrack(trackType, trackIdx)) || [];
      } catch (_) {
        items = [];
      }
      for (const tItem of items) {
        const matched = await inspectItem(tItem, trackIdx, mediaType);
        if (matched) return matched;
      }
    }
  }

  if (typeof timeline.GetCurrentVideoItem === 'function') {
    try {
      const cur = await timeline.GetCurrentVideoItem();
      const matched = await inspectItem(cur, 1, 1);
      if (matched) return matched;
    } catch (_) {}
  }

  return null;
}

/**
 * Strips NUL bytes and caps string length deterministically so oversized text cannot
 * wedge DaVinci Resolve's IPC or marker database.
 *
 * @param {unknown} value
 * @param {number} maxLen
 * @returns {string}
 */
function sanitizeBoundedText(value, maxLen) {
  const cleaned = String(value).replace(/\0/g, '');
  if (cleaned.length <= maxLen) return cleaned;
  return `${cleaned.slice(0, Math.max(0, maxLen - 1))}…`;
}

/**
 * Validates that an audio file is a valid 48 kHz WAV file.
 *
 * @param {string} filePath
 * @returns {{ valid: boolean, sampleRate: number, channels: number, bitsPerSample: number, format: string, error?: string }}
 */
function validateAudioFormat(filePath) {
  if (typeof filePath !== 'string' || filePath.length === 0 || filePath.includes('\0')) {
    return { valid: false, sampleRate: 0, channels: 0, bitsPerSample: 0, format: 'unknown', error: 'Invalid audio file path' };
  }
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
        if (chunkSize < 16 || offset + 24 > bytesRead) {
          return { valid: false, sampleRate: 0, channels: 0, bitsPerSample: 0, format: 'unknown', error: 'Truncated fmt chunk in WAV' };
        }
        foundFmt = true;
        audioFormat = header.readUInt16LE(offset + 8);
        numChannels = header.readUInt16LE(offset + 10);
        sampleRate = header.readUInt32LE(offset + 12);
        bitsPerSample = header.readUInt16LE(offset + 22);
        break;
      }
      offset += 8 + chunkSize + (chunkSize % 2);
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
    return { valid: false, sampleRate: 0, channels: 0, bitsPerSample: 0, format: 'unknown', error: errorMessage(err) };
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
  return sanitizeBoundedText(value, MAX_SCORE_VALUE_LEN);
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
  options = {}
) {
  if (!isPlainObject(report) || !Array.isArray(report.units) || report.units.length === 0) {
    return [];
  }
  const rate = parseFiniteNumber(fps);
  if (rate === null || rate <= 0) {
    return [];
  }

  const opts = isPlainObject(options) ? options : {};
  const parsedClipStart = parseFiniteNumber(clipStartFrame);
  const baseFrame = parsedClipStart !== null ? Math.max(0, Math.round(parsedClipStart)) : 0;
  const parsedOffset = parseFiniteNumber(opts.sourceOffsetS);
  const offsetS = parsedOffset !== null && parsedOffset > 0 ? parsedOffset : 0;
  const parsedThreshold = parseFiniteNumber(opts.lowStrengthThreshold);
  const threshold = parsedThreshold !== null ? parsedThreshold : 0.5;

  /** @type {Array<{ frameOffset: number, color: 'Red' | 'Yellow', name: string, note: string, durationFrames: number, decision: string, unitId: number | null }>} */
  const markers = [];
  const unitLimit = Math.min(report.units.length, MAX_GUARD_REPORT_UNITS);

  for (let i = 0; i < unitLimit; i += 1) {
    const unit = report.units[i];
    if (!isPlainObject(unit)) continue;
    const classification = classifyUnitForGuardMarker(unit, threshold);
    if (!classification) continue;

    const startRaw = unit.start_time_s ?? unit.startTimeS ?? unit.start_s;
    const endRaw = unit.end_time_s ?? unit.endTimeS ?? unit.end_s;
    const startS = parseFiniteNumber(startRaw);
    if (startS === null) continue;
    const endParsed = parseFiniteNumber(endRaw);
    const endS = endParsed !== null && endParsed >= startS ? endParsed : startS;

    // Skip units that ended entirely before the clip's visible in-point handle
    if (offsetS > 0 && endS <= offsetS) continue;

    const relStartS = Math.max(0, startS - offsetS);
    const relEndS = Math.max(relStartS, endS - offsetS);
    const frameOffset = baseFrame + Math.round(relStartS * rate);
    const rawDuration = Math.round((relEndS - relStartS) * rate);
    if (!Number.isSafeInteger(frameOffset) || frameOffset < 0 || !Number.isSafeInteger(rawDuration)) {
      continue;
    }
    const durationFrames = Math.max(1, rawDuration);

    const unitIdRaw = unit.unit_id ?? unit.unitId;
    const unitId = parseFiniteNumber(unitIdRaw);

    const rawName = unitId !== null
      ? `HawaVoClean Guard: ${classification.decision} (unit ${unitId})`
      : `HawaVoClean Guard: ${classification.decision}`;
    const name = sanitizeBoundedText(rawName, MAX_MARKER_NAME_LEN);

    const noteParts = [];
    if (typeof unit.decision_reason === 'string' && unit.decision_reason.trim().length > 0) {
      noteParts.push(sanitizeBoundedText(unit.decision_reason.trim(), MAX_MARKER_NOTE_LEN));
    }
    const chosenStrength = parseFiniteNumber(unit.chosen_strength);
    if (classification.decision === 'LOW_STRENGTH' && chosenStrength !== null) {
      noteParts.push(`chosen_strength=${formatGuardScoreValue(chosenStrength)}`);
    }

    const scoreEntries = [];
    let scoreChars = 0;
    const appendScore = (entry) => {
      if (scoreChars >= MAX_MARKER_NOTE_LEN) return;
      scoreEntries.push(entry);
      scoreChars += entry.length + 2;
    };
    if (isPlainObject(unit.guard_a_scores)) {
      for (const key of Object.keys(unit.guard_a_scores).sort()) {
        const val = unit.guard_a_scores[key];
        if (val !== undefined && val !== null) {
          appendScore(`${sanitizeBoundedText(key, MAX_SCORE_VALUE_LEN)}=${formatGuardScoreValue(val)}`);
        }
      }
    }
    if (isPlainObject(unit.guard_b_scores)) {
      for (const key of Object.keys(unit.guard_b_scores).sort()) {
        const val = unit.guard_b_scores[key];
        if (val !== undefined && val !== null) {
          appendScore(`guard_b.${sanitizeBoundedText(key, MAX_SCORE_VALUE_LEN)}=${formatGuardScoreValue(val)}`);
        }
      }
    }
    if (scoreEntries.length > 0) {
      noteParts.push(`scores: ${scoreEntries.join(', ')}`);
    }

    const rawNote = noteParts.length > 0 ? noteParts.join(' | ') : `Decision: ${classification.decision}`;
    const note = sanitizeBoundedText(rawNote, MAX_MARKER_NOTE_LEN);

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
  if (isPlainObject(entry) && isPlainObject(entry.report)) {
    return entry.report;
  }

  const candidatePaths = [];
  if (isPlainObject(entry) && typeof entry.reportPath === 'string' && entry.reportPath.length > 0) {
    candidatePaths.push(entry.reportPath);
  }
  if (typeof cleanedFilePath === 'string' && cleanedFilePath.length > 0) {
    const parsed = path.parse(cleanedFilePath);
    candidatePaths.push(path.join(parsed.dir, `${parsed.name}.hawavoclean.json`));
  }

  for (const candidate of candidatePaths) {
    if (candidate.includes('\0') || candidate.length > MAX_PATH_LEN || !path.isAbsolute(candidate)) continue;
    try {
      if (fs.existsSync(candidate)) {
        const stat = fs.statSync(candidate);
        if (stat.isFile() && stat.size <= MAX_REPORT_FILE_BYTES) {
          const parsed = JSON.parse(fs.readFileSync(candidate, 'utf8'));
          if (isPlainObject(parsed)) return parsed;
        }
      }
    } catch (_) {
      // Malformed or unreadable sidecar report does not abort audio delivery
    }
  }

  return null;
}

/**
 * Validates an untrusted IPC payload for `hawa:resolve:batch-clean-track` before touching Resolve API.
 * Strips unknown fields (such as test-only `failPoint` or `timeline`) and returns a clean DTO.
 *
 * @param {unknown} payload
 * @returns {{ clips: Array<object>, cleanedByClipId: Record<string, any> | Map<string, any>, targetTrackName: string | undefined, fps: number | undefined, addGuardMarkers: boolean | undefined }}
 */
function validateBatchCleanTrackPayload(payload) {
  if (!isPlainObject(payload)) {
    throw new Error('batchCleanTrack: payload must be a plain object.');
  }
  if (!Array.isArray(payload.clips) || payload.clips.length === 0) {
    throw new Error('batchCleanTrack: clips must be a non-empty array.');
  }
  if (payload.clips.length > MAX_BATCH_CLIPS) {
    throw new Error(`batchCleanTrack: clips exceeds maximum batch size (${MAX_BATCH_CLIPS}).`);
  }
  const mapIsMap = payload.cleanedByClipId instanceof Map;
  if (!mapIsMap && !isPlainObject(payload.cleanedByClipId)) {
    throw new Error('batchCleanTrack: cleanedByClipId must be an object or Map.');
  }
  const mapSize = mapIsMap ? payload.cleanedByClipId.size : Object.keys(payload.cleanedByClipId).length;
  if (mapSize === 0 || mapSize > MAX_BATCH_CLIPS) {
    throw new Error(`batchCleanTrack: cleanedByClipId must contain 1..${MAX_BATCH_CLIPS} entries.`);
  }

  let targetTrackName;
  if (payload.targetTrackName !== undefined) {
    if (
      typeof payload.targetTrackName !== 'string' ||
      payload.targetTrackName.includes('\0') ||
      payload.targetTrackName.trim().length === 0 ||
      payload.targetTrackName.length > MAX_TRACK_NAME_LEN
    ) {
      throw new Error(`batchCleanTrack: targetTrackName must be a non-empty string (<= ${MAX_TRACK_NAME_LEN} chars).`);
    }
    targetTrackName = payload.targetTrackName;
  }

  let fps;
  if (payload.fps !== undefined) {
    const rate = parseFiniteNumber(payload.fps);
    if (rate === null || rate <= 0 || rate > MAX_FPS) {
      throw new Error(`batchCleanTrack: fps must be a finite number in (0, ${MAX_FPS}].`);
    }
    fps = rate;
  }

  let addGuardMarkers;
  if (payload.addGuardMarkers !== undefined) {
    if (typeof payload.addGuardMarkers !== 'boolean') {
      throw new Error('batchCleanTrack: addGuardMarkers must be a boolean.');
    }
    addGuardMarkers = payload.addGuardMarkers;
  }

  const seenClipIds = new Set();
  for (let i = 0; i < payload.clips.length; i += 1) {
    const clip = payload.clips[i];
    if (!isPlainObject(clip)) {
      throw new Error(`batchCleanTrack: clip at index ${i} must be an object.`);
    }
    const rawId = clip.clipId ?? clip.mediaId ?? clip.id;
    if (typeof rawId !== 'string' || rawId.includes('\0')) {
      throw new Error(`batchCleanTrack: clip at index ${i} must have a valid string clipId.`);
    }
    const clipId = rawId.trim();
    if (!clipId || clipId.length > MAX_CLIP_ID_LEN) {
      throw new Error(`batchCleanTrack: clip at index ${i} clipId is empty or too long.`);
    }
    if (seenClipIds.has(clipId)) {
      throw new Error(`batchCleanTrack: duplicate clipId '${clipId}' at index ${i}.`);
    }
    seenClipIds.add(clipId);

    const startRaw = clip.startFrame ?? clip.recordFrame;
    if (startRaw !== undefined) {
      const parsedStart = parseFiniteNumber(startRaw);
      if (parsedStart === null || parsedStart < 0) {
        throw new Error(`batchCleanTrack: invalid startFrame for clip '${clipId}'.`);
      }
    }

    const cleanedEntry = mapIsMap
      ? payload.cleanedByClipId.get(clipId)
      : (Object.prototype.hasOwnProperty.call(payload.cleanedByClipId, clipId) ? payload.cleanedByClipId[clipId] : undefined);
    if (!cleanedEntry) {
      throw new Error(`batchCleanTrack: missing cleaned audio entry for clip '${clipId}'.`);
    }
    const filePath = typeof cleanedEntry === 'string'
      ? cleanedEntry
      : (isPlainObject(cleanedEntry) ? cleanedEntry.filePath : null);
    if (
      typeof filePath !== 'string' ||
      filePath.length === 0 ||
      filePath.length > MAX_PATH_LEN ||
      filePath.includes('\0') ||
      !path.isAbsolute(filePath)
    ) {
      throw new Error(`batchCleanTrack: cleaned audio path for clip '${clipId}' must be a bounded absolute path.`);
    }
    if (isPlainObject(cleanedEntry)) {
      if (cleanedEntry.reportPath !== undefined) {
        if (
          typeof cleanedEntry.reportPath !== 'string' ||
          cleanedEntry.reportPath.length === 0 ||
          cleanedEntry.reportPath.length > MAX_PATH_LEN ||
          cleanedEntry.reportPath.includes('\0') ||
          !path.isAbsolute(cleanedEntry.reportPath)
        ) {
          throw new Error(`batchCleanTrack: invalid reportPath for clip '${clipId}'.`);
        }
      }
      if (cleanedEntry.report !== undefined && cleanedEntry.report !== null) {
        if (!isPlainObject(cleanedEntry.report)) {
          throw new Error(`batchCleanTrack: report for clip '${clipId}' must be an object.`);
        }
        if (cleanedEntry.report.units !== undefined) {
          if (!Array.isArray(cleanedEntry.report.units) || cleanedEntry.report.units.length > MAX_GUARD_REPORT_UNITS) {
            throw new Error(`batchCleanTrack: report.units for clip '${clipId}' is invalid or oversized.`);
          }
        }
      }
    }
  }

  return {
    clips: payload.clips,
    cleanedByClipId: payload.cleanedByClipId,
    targetTrackName,
    fps,
    addGuardMarkers,
  };
}

/**
 * Validates an untrusted IPC payload for `hawa:resolve:apply-guard-markers` before touching Resolve API.
 *
 * @param {unknown} payload
 * @returns {{ report: object | null | undefined, reportPath: string | undefined, clipStartFrame: number | undefined, fps: number | undefined, sourceOffsetS: number | undefined, markers: Array<object> | undefined }}
 */
function validateApplyGuardMarkersPayload(payload) {
  if (!isPlainObject(payload)) {
    throw new Error('applyGuardMarkers: payload must be a plain object.');
  }

  let fps;
  if (payload.fps !== undefined) {
    const rate = parseFiniteNumber(payload.fps);
    if (rate === null || rate <= 0 || rate > MAX_FPS) {
      throw new Error(`applyGuardMarkers: fps must be a finite number in (0, ${MAX_FPS}].`);
    }
    fps = rate;
  }

  let clipStartFrame;
  if (payload.clipStartFrame !== undefined) {
    const frame = parseFiniteNumber(payload.clipStartFrame);
    if (frame === null || frame < 0) {
      throw new Error('applyGuardMarkers: clipStartFrame must be a finite non-negative number.');
    }
    clipStartFrame = frame;
  }

  let sourceOffsetS;
  if (payload.sourceOffsetS !== undefined) {
    const offset = parseFiniteNumber(payload.sourceOffsetS);
    if (offset === null || offset < 0) {
      throw new Error('applyGuardMarkers: sourceOffsetS must be a finite non-negative number.');
    }
    sourceOffsetS = offset;
  }

  let reportPath;
  if (payload.reportPath !== undefined) {
    if (
      typeof payload.reportPath !== 'string' ||
      payload.reportPath.length === 0 ||
      payload.reportPath.length > MAX_PATH_LEN ||
      payload.reportPath.includes('\0') ||
      !path.isAbsolute(payload.reportPath)
    ) {
      throw new Error('applyGuardMarkers: reportPath must be a bounded absolute path.');
    }
    reportPath = payload.reportPath;
  }

  let report = payload.report;
  if (report !== undefined && report !== null) {
    if (!isPlainObject(report)) {
      throw new Error('applyGuardMarkers: report must be a plain object.');
    }
    if (report.units !== undefined) {
      if (!Array.isArray(report.units) || report.units.length > MAX_GUARD_REPORT_UNITS) {
        throw new Error(`applyGuardMarkers: report.units must be an array of at most ${MAX_GUARD_REPORT_UNITS} items.`);
      }
    }
  }

  let markers;
  if (payload.markers !== undefined) {
    if (!Array.isArray(payload.markers)) {
      throw new Error('applyGuardMarkers: markers must be an array.');
    }
    if (payload.markers.length > MAX_GUARD_MARKERS) {
      throw new Error(`applyGuardMarkers: markers exceeds maximum count (${MAX_GUARD_MARKERS}).`);
    }
    for (let i = 0; i < payload.markers.length; i += 1) {
      const m = payload.markers[i];
      if (!isPlainObject(m)) {
        throw new Error(`applyGuardMarkers: marker at index ${i} must be an object.`);
      }
      const frameOffset = parseFiniteNumber(m.frameOffset);
      if (frameOffset === null || frameOffset < 0) {
        throw new Error(`applyGuardMarkers: marker at index ${i} has invalid frameOffset.`);
      }
      if (m.color !== undefined && (typeof m.color !== 'string' || m.color.includes('\0') || m.color.length > MAX_MARKER_COLOR_LEN)) {
        throw new Error(`applyGuardMarkers: marker at index ${i} has invalid color.`);
      }
      if (m.name !== undefined && (typeof m.name !== 'string' || m.name.includes('\0') || m.name.length > MAX_MARKER_NAME_LEN * 4)) {
        throw new Error(`applyGuardMarkers: marker at index ${i} has invalid or oversized name.`);
      }
      if (m.note !== undefined && (typeof m.note !== 'string' || m.note.includes('\0') || m.note.length > MAX_IPC_MARKER_NOTE_LEN)) {
        throw new Error(`applyGuardMarkers: marker at index ${i} has invalid or oversized note (> ${MAX_IPC_MARKER_NOTE_LEN} chars).`);
      }
      if (m.durationFrames !== undefined) {
        const dur = parseFiniteNumber(m.durationFrames);
        if (dur === null || dur < 1) {
          throw new Error(`applyGuardMarkers: marker at index ${i} has invalid durationFrames.`);
        }
      }
    }
    markers = payload.markers;
  }

  return {
    report,
    reportPath,
    clipStartFrame,
    fps,
    sourceOffsetS,
    markers,
  };
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
   * @param {number} [options.startFrame] Target timeline record frame (aliases: `recordFrame`)
   * @param {number} [options.recordFrame] Target timeline record frame
   * @param {number} [options.leftOffset] Source clip in-point frame offset (aliases: `sourceStartFrame`, `startFrameInSource`)
   * @param {number} [options.sourceStartFrame] Source clip in-point frame offset
   * @param {number} [options.sourceEndFrame] Source clip out-point frame offset
   * @param {number} [options.durationFrames] Clip duration in frames
   * @param {number | string} [options.mediaType] Optional Resolve mediaType (`2`/'audio' or `1`/'video')
   * @param {number} [options.fps] Optional explicit timeline fps (defaults to timeline's `timelineFrameRate`)
   * @param {object} [options.report] Optional `.hawavoclean.json` report for Guard markers
   * @param {string} [options.reportPath] Optional path to `.hawavoclean.json` report
   * @param {boolean} [options.addGuardMarkers] Whether to place Guard markers when a report is provided
   * @param {string} [options.failPoint] Injected failure point for tests
   * @returns {Promise<{ success: boolean, trackIndex: number, trackName: string, recordFrame: number, fps: number, markersAdded?: number }>}
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
    const resolvedFps = await resolveTimelineFps(timeline, options.fps);

    // Resolve exact timeline placement frame (honoring caller startFrame/recordFrame,
    // then matching source clip on audio/video timeline tracks, then timeline start frame, else 0).
    let resolvedRecordFrame = parseFiniteNumber(options.startFrame ?? options.recordFrame);
    let resolvedHandleStart = parseFiniteNumber(
      options.sourceStartFrame ?? options.startFrameInSource ?? options.leftOffset
    );
    let resolvedDuration = parseFiniteNumber(options.durationFrames);
    const resolvedHandleEndExplicit = parseFiniteNumber(options.sourceEndFrame ?? options.endFrameInSource);

    if (resolvedRecordFrame === null || (resolvedHandleStart === null && resolvedDuration === null)) {
      const detectedTiming = await findTimelineClipTiming(timeline, mediaId);
      if (detectedTiming) {
        if (resolvedRecordFrame === null) resolvedRecordFrame = detectedTiming.startFrame;
        if (resolvedHandleStart === null && detectedTiming.leftOffset !== null) {
          resolvedHandleStart = detectedTiming.leftOffset;
        }
        if (resolvedDuration === null && detectedTiming.durationFrames !== null) {
          resolvedDuration = detectedTiming.durationFrames;
        }
      }
    }
    if (resolvedRecordFrame === null && typeof timeline.GetStartFrame === 'function') {
      try {
        const tlStart = parseFiniteNumber(await timeline.GetStartFrame());
        if (tlStart !== null && tlStart >= 0) resolvedRecordFrame = tlStart;
      } catch (_) {}
    }
    const recordFrame = resolvedRecordFrame !== null && resolvedRecordFrame >= 0
      ? Math.round(resolvedRecordFrame)
      : 0;

    const handleStartFrame = resolvedHandleStart !== null && resolvedHandleStart >= 0
      ? Math.round(resolvedHandleStart)
      : null;
    let handleEndFrame = null;
    if (resolvedHandleEndExplicit !== null && resolvedHandleEndExplicit >= 0) {
      handleEndFrame = Math.round(resolvedHandleEndExplicit);
    } else if (handleStartFrame !== null && resolvedDuration !== null && resolvedDuration > 0) {
      handleEndFrame = handleStartFrame + Math.round(resolvedDuration);
    }
    const normalizedMediaType = normalizeResolveMediaType(options.mediaType);

    const txId = this.journal.begin('new-track', {
      mediaId,
      filePath,
      trackSubtype,
      desiredTrackName,
      recordFrame,
      fps: resolvedFps,
    });
    let createdTrackIndex = null;
    let importedItem = null;
    let appendedItem = null;
    /** @type {Array<number>} */
    const addedMarkerFrames = [];
    /** @type {Set<string>} */
    const addedMarkerColors = new Set();

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

      // 3. Append / Insert into the newly created track at the exact recordFrame
      let inserted = false;
      if (typeof mediaPool.AppendToTimeline === 'function') {
        /** @type {Record<string, any>} */
        const clipSpec = {
          mediaPoolItem: importedItem,
          trackIndex: createdTrackIndex,
          recordFrame,
        };
        if (handleStartFrame !== null) clipSpec.startFrame = handleStartFrame;
        if (handleEndFrame !== null && (handleStartFrame === null || handleEndFrame > handleStartFrame)) {
          clipSpec.endFrame = handleEndFrame;
        }
        if (normalizedMediaType !== undefined) clipSpec.mediaType = normalizedMediaType;

        const result = await mediaPool.AppendToTimeline([clipSpec]);
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

      // 4. Optional Guard markers placed with exact recordFrame and timeline fps
      let markersAdded = 0;
      if (
        options.addGuardMarkers !== false &&
        (options.report || options.reportPath) &&
        typeof timeline.AddMarker === 'function'
      ) {
        const parsedReport = resolveClipReport({ report: options.report, reportPath: options.reportPath }, filePath);
        const sourceOffsetS = handleStartFrame !== null ? handleStartFrame / resolvedFps : 0;
        const markers = extractGuardMarkersFromReport(parsedReport, recordFrame, resolvedFps, { sourceOffsetS });
        for (const marker of markers) {
          const added = await timeline.AddMarker(
            marker.frameOffset,
            marker.color,
            marker.name,
            marker.note,
            marker.durationFrames
          );
          if (added !== false) {
            addedMarkerFrames.push(marker.frameOffset);
            addedMarkerColors.add(marker.color);
            markersAdded += 1;
          }
        }
      }

      if (options.failPoint === 'after_insertion') {
        throw new Error('Injected failure after insertion');
      }

      this.journal.cleanup(txId);
      return {
        success: true,
        trackIndex: createdTrackIndex,
        trackName: desiredTrackName,
        recordFrame,
        fps: resolvedFps,
        markersAdded,
      };
    } catch (err) {
      this.log(`transactionalNewTrack failed: ${err.message}. Initiating compensating rollback...`);
      this.journal.update(txId, 'ROLLING_BACK');

      // Compensating action 0: delete markers added during this transaction
      if (addedMarkerFrames.length > 0) {
        if (typeof timeline.DeleteMarkerAtFrame === 'function') {
          for (const frameId of [...addedMarkerFrames].reverse()) {
            try {
              await timeline.DeleteMarkerAtFrame(frameId);
            } catch (_) {}
          }
        } else if (typeof timeline.DeleteMarkersByColor === 'function') {
          for (const color of addedMarkerColors) {
            try {
              await timeline.DeleteMarkersByColor(color);
            } catch (_) {}
          }
        }
      }

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
    fps,
    addGuardMarkers = true,
    failPoint,
  } = {}) {
    if (!Array.isArray(clips) || clips.length === 0) {
      throw new Error('batchTrackClean: clips must be a non-empty array.');
    }
    if (clips.length > MAX_BATCH_CLIPS) {
      throw new Error(`batchTrackClean: clips exceeds maximum batch size (${MAX_BATCH_CLIPS}).`);
    }
    const mapIsMap = cleanedByClipId instanceof Map;
    if (!mapIsMap && !isPlainObject(cleanedByClipId)) {
      throw new Error('batchTrackClean: cleanedByClipId map is required.');
    }
    let explicitRate = null;
    if (fps !== undefined) {
      explicitRate = parseFiniteNumber(fps);
      if (explicitRate === null || explicitRate <= 0 || explicitRate > MAX_FPS) {
        throw new Error(`batchTrackClean: invalid fps (${fps}).`);
      }
    }
    if (
      typeof targetTrackName !== 'string' ||
      targetTrackName.includes('\0') ||
      targetTrackName.trim().length === 0 ||
      targetTrackName.length > MAX_TRACK_NAME_LEN
    ) {
      throw new Error('batchTrackClean: targetTrackName must be a non-empty string.');
    }

    // 1. Pre-validate every clip, duplicate clipId, and cleaned 48 kHz WAV before touching Resolve API.
    const prepared = [];
    const seenClipIds = new Set();
    let maxChannels = 1;

    for (let i = 0; i < clips.length; i += 1) {
      const clip = clips[i];
      if (!isPlainObject(clip)) {
        throw new Error(`batchTrackClean: clip at index ${i} must be an object.`);
      }
      const clipId = String(clip.clipId ?? clip.mediaId ?? clip.id ?? '').trim();
      if (!clipId || clipId.includes('\0') || clipId.length > MAX_CLIP_ID_LEN) {
        throw new Error(`batchTrackClean: clip at index ${i} is missing clipId/mediaId/id.`);
      }
      if (seenClipIds.has(clipId)) {
        throw new Error(`batchTrackClean: duplicate clipId '${clipId}' at index ${i}.`);
      }
      seenClipIds.add(clipId);

      const cleanedEntry = mapIsMap
        ? cleanedByClipId.get(clipId)
        : (Object.prototype.hasOwnProperty.call(cleanedByClipId, clipId) ? cleanedByClipId[clipId] : undefined);
      if (!cleanedEntry) {
        throw new Error(`batchTrackClean: missing cleaned audio entry for clip '${clipId}'.`);
      }

      const filePath = typeof cleanedEntry === 'string'
        ? cleanedEntry
        : (isPlainObject(cleanedEntry) ? cleanedEntry.filePath : null);
      if (typeof filePath !== 'string' || filePath.includes('\0') || filePath.length > MAX_PATH_LEN || !path.isAbsolute(filePath)) {
        throw new Error(`batchTrackClean: cleaned audio path for clip '${clipId}' must be absolute.`);
      }

      const audioCheck = validateAudioFormat(filePath);
      if (!audioCheck.valid) {
        throw new Error(`batchTrackClean rejected clip '${clipId}': ${audioCheck.error}`);
      }
      if (audioCheck.channels > maxChannels) {
        maxChannels = audioCheck.channels;
      }

      let startRaw = clip.startFrame ?? clip.recordFrame;
      if (startRaw === undefined && typeof clip.GetStart === 'function') {
        startRaw = await clip.GetStart();
      }
      const startFrame = parseFiniteNumber(startRaw);
      if (startFrame === null || startFrame < 0) {
        throw new Error(`batchTrackClean: invalid startFrame for clip '${clipId}'.`);
      }

      const handleStartRaw = clip.sourceStartFrame ?? clip.startFrameInSource ?? clip.leftOffset;
      const handleStartParsed = parseFiniteNumber(handleStartRaw);
      const handleStartFrame = handleStartParsed !== null
        ? Math.max(0, Math.round(handleStartParsed))
        : null;

      let handleEndFrame = null;
      const handleEndRaw = clip.sourceEndFrame ?? clip.endFrameInSource;
      const handleEndParsed = parseFiniteNumber(handleEndRaw);
      const durationParsed = parseFiniteNumber(clip.durationFrames);
      if (handleEndParsed !== null) {
        handleEndFrame = Math.max(0, Math.round(handleEndParsed));
      } else if (handleStartFrame !== null && durationParsed !== null && durationParsed > 0) {
        handleEndFrame = handleStartFrame + Math.round(durationParsed);
      }
      if (handleStartFrame !== null && handleEndFrame !== null && handleEndFrame <= handleStartFrame) {
        throw new Error(`batchTrackClean: invalid source handle range for clip '${clipId}' (${handleStartFrame}..${handleEndFrame}).`);
      }

      const offsetParsed = parseFiniteNumber(clip.sourceOffsetS);
      const report = addGuardMarkers ? resolveClipReport(cleanedEntry, filePath) : null;
      const normalizedMediaType = normalizeResolveMediaType(clip.mediaType);

      prepared.push({
        clipId,
        filePath,
        startFrame: Math.round(startFrame),
        handleStartFrame,
        handleEndFrame,
        explicitSourceOffsetS: offsetParsed !== null && offsetParsed > 0 ? offsetParsed : null,
        sourceOffsetS: 0,
        mediaType: normalizedMediaType,
        report,
      });
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

    const rate = explicitRate !== null
      ? explicitRate
      : await resolveTimelineFps(activeTimeline);
    for (const entry of prepared) {
      entry.sourceOffsetS = entry.explicitSourceOffsetS !== null
        ? entry.explicitSourceOffsetS
        : (entry.handleStartFrame !== null ? entry.handleStartFrame / rate : 0);
    }

    const mediaPool = await this.getMediaPool();
    if (!mediaPool) {
      throw new Error('MediaPool unavailable.');
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
    /** @type {Set<string>} */
    const addedMarkerColors = new Set();
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
              addedMarkerColors.add(marker.color);
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
        fps: rate,
        clipsPlaced: prepared.length,
        markersAdded: placedMarkers.length,
        markers: placedMarkers,
      };
    } catch (err) {
      this.log(`transactionalBatchTrackClean failed: ${errorMessage(err)}. Initiating compensating rollback...`);
      const compensations = [];
      try {
        this.journal.update(txId, 'ROLLING_BACK', compensations);
      } catch (jErr) {
        this.log(`Rollback journal ROLLING_BACK update failed: ${errorMessage(jErr)}`);
      }

      // Compensating action 1: delete markers added during this transaction
      if (addedMarkerFrames.length > 0) {
        if (typeof activeTimeline.DeleteMarkerAtFrame === 'function') {
          for (const frameId of [...addedMarkerFrames].reverse()) {
            try {
              await activeTimeline.DeleteMarkerAtFrame(frameId);
              compensations.push({ action: 'delete_marker', frameId, status: 'ok' });
            } catch (markerErr) {
              const msg = errorMessage(markerErr);
              compensations.push({ action: 'delete_marker', frameId, status: 'error', error: msg });
              this.log(`Rollback DeleteMarkerAtFrame(${frameId}) failed: ${msg}`);
            }
          }
        } else if (typeof activeTimeline.DeleteMarkersByColor === 'function') {
          for (const color of addedMarkerColors) {
            try {
              await activeTimeline.DeleteMarkersByColor(color);
              compensations.push({ action: 'delete_markers_by_color', color, status: 'ok' });
            } catch (markerErr) {
              const msg = errorMessage(markerErr);
              compensations.push({ action: 'delete_markers_by_color', color, status: 'error', error: msg });
              this.log(`Rollback DeleteMarkersByColor(${color}) failed: ${msg}`);
            }
          }
        }
      }

      // Compensating action 2: delete appended timeline items
      if (appendedItems.length > 0) {
        let deletedViaBatch = false;
        if (typeof activeTimeline.DeleteClips === 'function') {
          try {
            await activeTimeline.DeleteClips(appendedItems, false);
            compensations.push({ action: 'delete_timeline_items', count: appendedItems.length, status: 'ok' });
            deletedViaBatch = true;
          } catch (itemsErr) {
            const msg = errorMessage(itemsErr);
            compensations.push({ action: 'delete_timeline_items', count: appendedItems.length, status: 'error', error: msg });
            this.log(`Rollback DeleteClips(timeline) failed: ${msg}`);
          }
        }
        if (!deletedViaBatch && typeof activeTimeline.DeleteItem === 'function') {
          for (const item of [...appendedItems].reverse()) {
            try {
              await activeTimeline.DeleteItem(item);
              compensations.push({ action: 'delete_timeline_item', status: 'ok' });
            } catch (itemErr) {
              const msg = errorMessage(itemErr);
              compensations.push({ action: 'delete_timeline_item', status: 'error', error: msg });
              this.log(`Rollback DeleteItem failed: ${msg}`);
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
          const msg = errorMessage(trackErr);
          compensations.push({ action: 'delete_track', trackIndex: createdTrackIndex, status: 'error', error: msg });
          this.log(`Rollback DeleteTrack(${createdTrackIndex}) failed: ${msg}`);
        }
      }

      // Compensating action 4: purge imported media pool items
      if (importedItems.length > 0 && typeof mediaPool.DeleteClips === 'function') {
        try {
          await mediaPool.DeleteClips(importedItems);
          compensations.push({ action: 'delete_media_pool_clips', count: importedItems.length, status: 'ok' });
        } catch (poolErr) {
          const msg = errorMessage(poolErr);
          compensations.push({ action: 'delete_media_pool_clips', count: importedItems.length, status: 'error', error: msg });
          this.log(`Rollback mediaPool.DeleteClips failed: ${msg}`);
        }
      }

      try {
        this.journal.update(txId, 'ROLLED_BACK', compensations);
      } catch (jErr) {
        this.log(`Rollback journal ROLLED_BACK update failed: ${errorMessage(jErr)}`);
      }
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
    fps,
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
    if (markers !== undefined) {
      if (!Array.isArray(markers)) {
        throw new Error('applyGuardMarkers: markers must be an array.');
      }
      if (markers.length > MAX_GUARD_MARKERS) {
        throw new Error(`applyGuardMarkers: markers exceeds maximum count (${MAX_GUARD_MARKERS}).`);
      }
      resolvedMarkers = markers.map((m, idx) => {
        if (!isPlainObject(m)) {
          throw new Error(`applyGuardMarkers: invalid marker descriptor at index ${idx}.`);
        }
        const parsedOffset = parseFiniteNumber(m.frameOffset);
        if (parsedOffset === null || parsedOffset < 0) {
          throw new Error(`applyGuardMarkers: invalid marker descriptor at index ${idx}.`);
        }
        const parsedDuration = parseFiniteNumber(m.durationFrames);
        const color = typeof m.color === 'string' && m.color.trim().length > 0
          ? sanitizeBoundedText(m.color.trim(), MAX_MARKER_COLOR_LEN)
          : 'Yellow';
        const name = typeof m.name === 'string' && m.name.trim().length > 0
          ? sanitizeBoundedText(m.name.trim(), MAX_MARKER_NAME_LEN)
          : 'HawaVoClean Guard';
        const note = typeof m.note === 'string'
          ? sanitizeBoundedText(m.note, MAX_MARKER_NOTE_LEN)
          : '';
        return {
          frameOffset: Math.round(parsedOffset),
          color,
          name,
          note,
          durationFrames: Math.max(1, parsedDuration !== null ? Math.round(parsedDuration) : 1),
        };
      });
    } else {
      const effectiveFps = await resolveTimelineFps(activeTimeline, fps);
      const parsedReport = resolveClipReport({ report, reportPath }, '');
      resolvedMarkers = extractGuardMarkersFromReport(parsedReport, clipStartFrame, effectiveFps, { sourceOffsetS });
    }

    if (resolvedMarkers.length === 0 || typeof activeTimeline.AddMarker !== 'function') {
      return { success: true, markersAdded: 0, markers: [] };
    }

    const addedFrames = [];
    const addedColors = new Set();
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
          addedColors.add(marker.color);
          placed.push(marker);
        }
      }
      return { success: true, markersAdded: placed.length, markers: placed };
    } catch (err) {
      if (addedFrames.length > 0) {
        if (typeof activeTimeline.DeleteMarkerAtFrame === 'function') {
          for (const frameId of [...addedFrames].reverse()) {
            try {
              await activeTimeline.DeleteMarkerAtFrame(frameId);
            } catch (_) {}
          }
        } else if (typeof activeTimeline.DeleteMarkersByColor === 'function') {
          for (const color of addedColors) {
            try {
              await activeTimeline.DeleteMarkersByColor(color);
            } catch (_) {}
          }
        }
      }
      throw err;
    }
  }
}

module.exports = {
  ALLOWED_TRANSACTION_TRANSITIONS,
  GUARD_MARKER_COLORS,
  MAX_MARKER_NAME_LEN,
  MAX_MARKER_NOTE_LEN,
  MAX_IPC_MARKER_NOTE_LEN,
  MAX_BATCH_CLIPS,
  MAX_GUARD_MARKERS,
  MAX_GUARD_REPORT_UNITS,
  validateAudioFormat,
  resolveTimelineFps,
  normalizeResolveMediaType,
  extractGuardMarkersFromReport,
  validateBatchCleanTrackPayload,
  validateApplyGuardMarkersPayload,
  TransactionJournal,
  TimelineTransactionManager,
  TRANSACTIONS_DIR,
};
