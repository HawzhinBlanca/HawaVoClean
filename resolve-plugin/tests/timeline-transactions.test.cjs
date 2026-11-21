// @ts-check
'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');

const {
  validateAudioFormat,
  extractGuardMarkersFromReport,
  validateBatchCleanTrackPayload,
  validateApplyGuardMarkersPayload,
  GUARD_MARKER_COLORS,
  MAX_MARKER_NAME_LEN,
  MAX_MARKER_NOTE_LEN,
  MAX_IPC_MARKER_NOTE_LEN,
  MAX_BATCH_CLIPS,
  MAX_GUARD_MARKERS,
  MAX_GUARD_REPORT_UNITS,
  TransactionJournal,
  TimelineTransactionManager,
} = require('../com.hawavoclean.resolve/timeline-transactions.js');

/**
 * Helper to generate synthetic WAV files with specific sample rate and channels.
 */
function createSyntheticWav(filePath, { sampleRate = 48000, channels = 1, bitsPerSample = 16, format = 1, numFrames = 100 } = {}) {
  const bytesPerSample = bitsPerSample / 8;
  const blockAlign = channels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = numFrames * blockAlign;
  const totalSize = 44 + dataSize;

  const buf = Buffer.alloc(totalSize);
  buf.write('RIFF', 0);
  buf.writeUInt32LE(totalSize - 8, 4);
  buf.write('WAVE', 8);

  buf.write('fmt ', 12);
  buf.writeUInt32LE(16, 16); // subchunk size
  buf.writeUInt16LE(format, 20); // 1 = PCM, 3 = Float
  buf.writeUInt16LE(channels, 22);
  buf.writeUInt32LE(sampleRate, 24);
  buf.writeUInt32LE(byteRate, 28);
  buf.writeUInt16LE(blockAlign, 32);
  buf.writeUInt16LE(bitsPerSample, 34);

  buf.write('data', 36);
  buf.writeUInt32LE(dataSize, 40);

  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, buf);
}

test('validateAudioFormat validates 48 kHz delivery and rejects non-48k audio', (t) => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'wav-test-'));
  t.after(() => fs.rmSync(tmpDir, { recursive: true, force: true }));

  const wav48kMono = path.join(tmpDir, 'audio_48k_mono.wav');
  createSyntheticWav(wav48kMono, { sampleRate: 48000, channels: 1, bitsPerSample: 16 });
  const res48kMono = validateAudioFormat(wav48kMono);
  assert.equal(res48kMono.valid, true);
  assert.equal(res48kMono.sampleRate, 48000);
  assert.equal(res48kMono.channels, 1);
  assert.equal(res48kMono.format, 'pcm');

  const wav48kStereoFloat = path.join(tmpDir, 'audio_48k_stereo_float.wav');
  createSyntheticWav(wav48kStereoFloat, { sampleRate: 48000, channels: 2, bitsPerSample: 32, format: 3 });
  const res48kStereo = validateAudioFormat(wav48kStereoFloat);
  assert.equal(res48kStereo.valid, true);
  assert.equal(res48kStereo.sampleRate, 48000);
  assert.equal(res48kStereo.channels, 2);
  assert.equal(res48kStereo.format, 'float');

  const wav44k = path.join(tmpDir, 'audio_44k.wav');
  createSyntheticWav(wav44k, { sampleRate: 44100, channels: 2, bitsPerSample: 16 });
  const res44k = validateAudioFormat(wav44k);
  assert.equal(res44k.valid, false);
  assert.ok(res44k.error.includes('must be delivered at 48 kHz'));

  const textFile = path.join(tmpDir, 'corrupt.wav');
  fs.writeFileSync(textFile, 'NOT A WAV FILE CONTENT');
  const resCorrupt = validateAudioFormat(textFile);
  assert.equal(resCorrupt.valid, false);
});

test('TransactionJournal records stages and cleans up on completion', (t) => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'journal-test-'));
  t.after(() => fs.rmSync(tmpDir, { recursive: true, force: true }));

  const journal = new TransactionJournal(tmpDir);
  const txId = journal.begin('replace', { test: true });
  assert.ok(txId.startsWith('tx_'));

  const logFile = path.join(tmpDir, `${txId}.json`);
  assert.ok(fs.existsSync(logFile));

  const content1 = JSON.parse(fs.readFileSync(logFile, 'utf8'));
  assert.equal(content1.stage, 'INIT');
  assert.equal(content1.operation, 'replace');

  journal.update(txId, 'ROLLING_BACK', [{ action: 'restore' }]);
  const content2 = JSON.parse(fs.readFileSync(logFile, 'utf8'));
  assert.equal(content2.stage, 'ROLLING_BACK');
  assert.equal(content2.compensations.length, 1);

  journal.cleanup(txId);
  assert.ok(!fs.existsSync(logFile));
});

test('transactionalReplace preserves sub-clips, verifies 48k, and rolls back on failure', async (t) => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'tx-replace-test-'));
  t.after(() => fs.rmSync(tmpDir, { recursive: true, force: true }));

  const journal = new TransactionJournal(path.join(tmpDir, 'journal'));
  const validAudio = path.join(tmpDir, 'clean_48k.wav');
  createSyntheticWav(validAudio, { sampleRate: 48000, channels: 1 });

  let clipCurrentFile = '/Volumes/Media/original.wav';
  let preserveCalled = false;
  let fallbackReplaceCalled = false;
  let restoredTo = null;

  const mockItem = {
    GetClipProperty: async () => ({ 'File Path': clipCurrentFile }),
    ReplaceClipPreserveSubClip: async (newPath) => {
      preserveCalled = true;
      clipCurrentFile = newPath;
      return true;
    },
    ReplaceClip: async (newPath) => {
      fallbackReplaceCalled = true;
      restoredTo = newPath;
      clipCurrentFile = newPath;
      return true;
    },
  };

  const manager = new TimelineTransactionManager({
    getProject: async () => ({}),
    getMediaPool: async () => ({}),
    findItemByMediaId: async () => mockItem,
    log: () => {},
    journal,
  });

  // 1. Happy path: ReplaceClipPreserveSubClip called
  const res = await manager.transactionalReplace('clip-1', validAudio);
  assert.equal(res.success, true);
  assert.equal(res.method, 'ReplaceClipPreserveSubClip');
  assert.equal(preserveCalled, true);
  assert.equal(clipCurrentFile, validAudio);

  // 2. Injected failure after replace: triggers rollback to original
  let failed = false;
  try {
    await manager.transactionalReplace('clip-1', validAudio, { failPoint: 'after_replace' });
  } catch (err) {
    failed = true;
    assert.ok(err.message.includes('Injected failure after replace'));
  }
  assert.equal(failed, true);
  assert.equal(fallbackReplaceCalled, true);
  assert.equal(restoredTo, validAudio); // was restored to prior known state
});

test('transactionalAppend imports media and rolls back on timeline append failure', async (t) => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'tx-append-test-'));
  t.after(() => fs.rmSync(tmpDir, { recursive: true, force: true }));

  const journal = new TransactionJournal(path.join(tmpDir, 'journal'));
  const validAudio = path.join(tmpDir, 'clean_48k.wav');
  createSyntheticWav(validAudio, { sampleRate: 48000, channels: 2 });

  let deletedClips = [];
  const mockMediaPoolItem = {
    GetMediaId: async () => 'media-item-123',
  };

  const mockMediaPool = {
    ImportMedia: async (paths) => [mockMediaPoolItem],
    AppendToTimeline: async (items) => {
      // Simulate append failure
      return [];
    },
    DeleteClips: async (clips) => {
      deletedClips.push(...clips);
      return true;
    },
  };

  const manager = new TimelineTransactionManager({
    getProject: async () => ({}),
    getMediaPool: async () => mockMediaPool,
    findItemByMediaId: async () => null,
    log: () => {},
    journal,
  });

  let failed = false;
  try {
    await manager.transactionalAppend(validAudio);
  } catch (err) {
    failed = true;
    assert.ok(err.message.includes('AppendToTimeline returned false/empty'));
  }
  assert.equal(failed, true);
  // Verified rollback purged imported media item
  assert.equal(deletedClips.length, 1);
  assert.equal(deletedClips[0], mockMediaPoolItem);
});

test('transactionalNewTrack creates dedicated track, aligns timecode, and executes full rollback on failure', async (t) => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'tx-newtrack-test-'));
  t.after(() => fs.rmSync(tmpDir, { recursive: true, force: true }));

  const journal = new TransactionJournal(path.join(tmpDir, 'journal'));
  const validAudio = path.join(tmpDir, 'clean_48k_stereo.wav');
  createSyntheticWav(validAudio, { sampleRate: 48000, channels: 2 });

  let tracks = ['audio 1'];
  let trackNames = {};
  let deletedTracks = [];
  let deletedItems = [];
  let deletedClips = [];

  const mockMediaItem = { GetMediaId: async () => 'media-imported-1' };
  const mockTimelineItem = { GetName: () => 'timeline-item-1' };

  const mockTimeline = {
    GetTrackCount: async (type) => tracks.length,
    AddTrack: async (type, subtype) => {
      tracks.push(`${type} ${tracks.length + 1} (${subtype})`);
      return true;
    },
    SetTrackName: async (type, index, name) => {
      trackNames[index] = name;
      return true;
    },
    DeleteTrack: async (type, index) => {
      deletedTracks.push(index);
      return true;
    },
    DeleteItem: async (item) => {
      deletedItems.push(item);
      return true;
    },
  };

  const mockMediaPool = {
    ImportMedia: async (paths) => [mockMediaItem],
    AppendToTimeline: async (clips) => [mockTimelineItem],
    DeleteClips: async (clips) => {
      deletedClips.push(...clips);
      return true;
    },
  };

  const mockProject = {
    GetCurrentTimeline: async () => mockTimeline,
  };

  const manager = new TimelineTransactionManager({
    getProject: async () => mockProject,
    getMediaPool: async () => mockMediaPool,
    findItemByMediaId: async () => null,
    log: () => {},
    journal,
  });

  // 1. Happy path: creates stereo track named 'HawaVoClean Cleaned'
  const res = await manager.transactionalNewTrack('source-clip-1', validAudio);
  assert.equal(res.success, true);
  assert.equal(res.trackIndex, 2);
  assert.equal(trackNames[2], 'HawaVoClean Cleaned');

  // 2. Injected failure after insertion: tests full compensating rollback
  let failed = false;
  try {
    await manager.transactionalNewTrack('source-clip-1', validAudio, { failPoint: 'after_insertion' });
  } catch (err) {
    failed = true;
    assert.ok(err.message.includes('Injected failure after insertion'));
  }
  assert.equal(failed, true);
  // Verifying rollback cleaned up:
  assert.equal(deletedItems.length, 1); // timeline item deleted
  assert.equal(deletedTracks.length, 1); // track deleted
  assert.equal(deletedClips.length, 1); // media pool item purged
});

test('TransactionJournal enforces explicit stage transition table', () => {
  const { ALLOWED_TRANSACTION_TRANSITIONS, TransactionJournal } = require('../com.hawavoclean.resolve/timeline-transactions.js');
  assert.deepEqual([...ALLOWED_TRANSACTION_TRANSITIONS.COMMITTED], []);
  assert.deepEqual([...ALLOWED_TRANSACTION_TRANSITIONS.ROLLED_BACK], []);

  const txDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hawa-tx-fsm-'));
  try {
    const journal = new TransactionJournal(txDir);
    const txId = journal.begin('replace', { mediaId: 'm1' });
    assert.throws(() => journal.update(txId, 'ROLLED_BACK'), /Illegal timeline transaction transition: INIT -> ROLLED_BACK/);
    journal.update(txId, 'ROLLING_BACK');
    journal.update(txId, 'ROLLED_BACK');
    assert.throws(() => journal.update(txId, 'COMMITTED'), /Illegal timeline transaction transition: ROLLED_BACK -> COMMITTED/);
  } finally {
    fs.rmSync(txDir, { recursive: true, force: true });
  }
});

test('extractGuardMarkersFromReport handles missing/empty inputs and maps decisions, offsets, and notes deterministically', () => {
  assert.deepEqual(extractGuardMarkersFromReport(null, 86400, 24), []);
  assert.deepEqual(extractGuardMarkersFromReport(undefined, 86400, 24), []);
  assert.deepEqual(extractGuardMarkersFromReport({}, 86400, 24), []);
  assert.deepEqual(extractGuardMarkersFromReport({ units: [] }, 86400, 24), []);
  assert.deepEqual(extractGuardMarkersFromReport({ units: [{ final_decision: 'REVERTED', start_time_s: 1, end_time_s: 2 }] }, 86400, 0), []);
  assert.deepEqual(extractGuardMarkersFromReport({ units: [{ final_decision: 'REVERTED', start_time_s: 1, end_time_s: 2 }] }, 86400, -24), []);

  const report = {
    units: [
      {
        unit_id: 0,
        start_time_s: 0.0,
        end_time_s: 0.4,
        final_decision: 'REVERTED',
        decision_reason: 'Pre-roll unit before in-point',
      },
      {
        unit_id: 1,
        start_time_s: 0.5,
        end_time_s: 2.0,
        final_decision: 'enhanced',
        chosen_strength: 1.0,
        guard_a_verdict: 'PASS',
        decision_reason: 'Passed Guard A with strength s=1.00',
      },
      {
        unit_id: 2,
        start_time_s: 2.0,
        end_time_s: 3.5,
        final_decision: 'original_no_speech',
        guard_a_verdict: 'NO_SPEECH',
      },
      {
        unit_id: 5,
        start_time_s: 6.5,
        end_time_s: 8.0,
        final_decision: 'ERROR_PASSTHROUGH',
        guard_a_verdict: 'ERROR',
        decision_reason: 'Enhancer timed out',
        guard_a_scores: { enhancer_error: true },
      },
      {
        unit_id: 3,
        start_time_s: 3.5,
        end_time_s: 5.0,
        final_decision: 'original_reverted',
        guard_a_verdict: 'REVERT',
        decision_reason: 'Spectral hole exceeded threshold',
        guard_a_scores: { spectral_hole: 0.14231, hf_preservation: 0.41 },
      },
      {
        unit_id: 4,
        start_time_s: 5.0,
        end_time_s: 6.0,
        final_decision: 'original_continuity',
        guard_a_verdict: 'PASS',
        decision_reason: 'Continuity policy reverted isolated island',
        guard_a_scores: { seam_delta_db: 4.25 },
      },
      {
        unit_id: 6,
        start_time_s: 8.5,
        end_time_s: 9.5,
        final_decision: 'enhanced',
        chosen_strength: 0.25,
        guard_a_verdict: 'PASS',
        decision_reason: 'Passed Guard A with strength s=0.25',
        guard_a_scores: { spectral_hole: 0.03 },
      },
    ],
  };

  const markers = extractGuardMarkersFromReport(report, 86400, 24, { sourceOffsetS: 0.5 });
  assert.equal(markers.length, 4);

  // Sorted deterministically by frameOffset
  assert.equal(markers[0].unitId, 3);
  assert.equal(markers[0].decision, 'REVERTED');
  assert.equal(markers[0].color, GUARD_MARKER_COLORS.REVERTED);
  assert.equal(markers[0].frameOffset, 86400 + Math.round((3.5 - 0.5) * 24));
  assert.equal(markers[0].durationFrames, Math.round(1.5 * 24));
  assert.ok(markers[0].note.includes('Spectral hole exceeded threshold'));
  assert.ok(markers[0].note.includes('hf_preservation=0.41'));
  assert.ok(markers[0].note.includes('spectral_hole=0.1423'));

  assert.equal(markers[1].unitId, 4);
  assert.equal(markers[1].decision, 'CONTINUITY_REVERTED');
  assert.equal(markers[1].color, 'Yellow');
  assert.equal(markers[1].frameOffset, 86400 + Math.round((5.0 - 0.5) * 24));
  assert.equal(markers[1].durationFrames, 24);
  assert.ok(markers[1].note.includes('seam_delta_db=4.25'));

  assert.equal(markers[2].unitId, 5);
  assert.equal(markers[2].decision, 'ERROR_PASSTHROUGH');
  assert.equal(markers[2].color, 'Red');
  assert.equal(markers[2].frameOffset, 86400 + Math.round((6.5 - 0.5) * 24));
  assert.ok(markers[2].note.includes('Enhancer timed out'));

  assert.equal(markers[3].unitId, 6);
  assert.equal(markers[3].decision, 'LOW_STRENGTH');
  assert.equal(markers[3].color, 'Yellow');
  assert.equal(markers[3].frameOffset, 86400 + Math.round((8.5 - 0.5) * 24));
  assert.ok(markers[3].note.includes('chosen_strength=0.25'));

  // Pure & deterministic: calling twice yields byte-identical JSON
  const markersAgain = extractGuardMarkersFromReport(report, 86400, 24, { sourceOffsetS: 0.5 });
  assert.deepEqual(markersAgain, markers);
});

test('transactionalBatchTrackClean creates track, aligns clips with handles, places Guard markers, and persists 0o600 journal', async (t) => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'tx-batch-happy-'));
  t.after(() => fs.rmSync(tmpDir, { recursive: true, force: true }));

  const journalDir = path.join(tmpDir, 'journal');
  const journal = new TransactionJournal(journalDir);
  const wav1 = path.join(tmpDir, 'clip1_clean.wav');
  const wav2 = path.join(tmpDir, 'clip2_clean.wav');
  createSyntheticWav(wav1, { sampleRate: 48000, channels: 1 });
  createSyntheticWav(wav2, { sampleRate: 48000, channels: 2 });

  const tracks = [{ name: 'Dialogue 1', subtype: 'mono' }];
  const appendedSpecs = [];
  const addedMarkers = [];

  const mockTimeline = {
    GetTrackCount: async () => tracks.length,
    GetTrackName: async (_type, idx) => tracks[idx - 1]?.name || '',
    AddTrack: async (_type, subtype) => {
      tracks.push({ name: `Audio ${tracks.length + 1}`, subtype });
      return true;
    },
    SetTrackName: async (_type, idx, name) => {
      tracks[idx - 1].name = name;
      return true;
    },
    AddMarker: async (frameId, color, name, note, duration) => {
      addedMarkers.push({ frameId, color, name, note, duration });
      return true;
    },
    DeleteMarkerAtFrame: async () => true,
    DeleteClips: async () => true,
    DeleteTrack: async () => true,
  };

  let importCounter = 0;
  const mockMediaPool = {
    ImportMedia: async (paths) => [{ id: `mp-${++importCounter}`, path: paths[0] }],
    AppendToTimeline: async (specs) => {
      appendedSpecs.push(...specs);
      return specs.map((s, i) => ({ tlItemId: `tl-${appendedSpecs.length}-${i}`, spec: s }));
    },
    DeleteClips: async () => true,
  };

  const manager = new TimelineTransactionManager({
    getProject: async () => ({ GetCurrentTimeline: async () => mockTimeline }),
    getMediaPool: async () => mockMediaPool,
    findItemByMediaId: async () => null,
    log: () => {},
    journal,
  });

  const clips = [
    { clipId: 'c1', startFrame: 86400, sourceStartFrame: 24, durationFrames: 240 },
    { clipId: 'c2', startFrame: 86800, sourceStartFrame: 0, sourceEndFrame: 360 },
  ];
  const cleanedByClipId = {
    c1: {
      filePath: wav1,
      report: {
        units: [
          {
            unit_id: 1,
            start_time_s: 2.0,
            end_time_s: 3.5,
            final_decision: 'REVERTED',
            decision_reason: 'Consonant attenuation guard triggered',
            guard_a_scores: { hf_preservation: 0.32 },
          },
        ],
      },
    },
    c2: {
      filePath: wav2,
      report: {
        units: [
          {
            unit_id: 2,
            start_time_s: 1.0,
            end_time_s: 2.0,
            final_decision: 'LOW_STRENGTH',
            chosen_strength: 0.5,
            decision_reason: 'Reduced strength selected',
            guard_a_scores: { spectral_hole: 0.04 },
          },
        ],
      },
    },
  };

  const res = await manager.transactionalBatchTrackClean({
    timeline: mockTimeline,
    clips,
    cleanedByClipId,
    targetTrackName: 'HawaVoClean Cleaned',
    fps: 24,
    addGuardMarkers: true,
  });

  assert.equal(res.success, true);
  assert.equal(res.trackIndex, 2);
  assert.equal(res.trackName, 'HawaVoClean Cleaned');
  assert.equal(res.reusedTrack, false);
  assert.equal(res.clipsPlaced, 2);
  assert.equal(res.markersAdded, 2);
  assert.equal(tracks[1].subtype, 'stereo'); // max channel count across batch
  assert.equal(tracks[1].name, 'HawaVoClean Cleaned');

  // Verify exact frame alignment + handle preservation
  assert.equal(appendedSpecs.length, 2);
  assert.equal(appendedSpecs[0].trackIndex, 2);
  assert.equal(appendedSpecs[0].recordFrame, 86400);
  assert.equal(appendedSpecs[0].startFrame, 24);
  assert.equal(appendedSpecs[0].endFrame, 264);
  assert.equal(appendedSpecs[1].trackIndex, 2);
  assert.equal(appendedSpecs[1].recordFrame, 86800);
  assert.equal(appendedSpecs[1].startFrame, 0);
  assert.equal(appendedSpecs[1].endFrame, 360);

  // Verify Guard Timeline Markers (clip 1 had sourceStartFrame=24 -> 1.0s offset, unit at 2.0s -> +24 frames)
  assert.equal(addedMarkers.length, 2);
  assert.equal(addedMarkers[0].frameId, 86424);
  assert.equal(addedMarkers[0].color, 'Red');
  assert.equal(addedMarkers[0].duration, 36);
  assert.equal(addedMarkers[1].frameId, 86824);
  assert.equal(addedMarkers[1].color, 'Yellow');

  // Verify 0o600 transaction journal persisted in COMMITTED stage
  assert.ok(fs.existsSync(res.journalPath));
  const journalData = JSON.parse(fs.readFileSync(res.journalPath, 'utf8'));
  assert.equal(journalData.stage, 'COMMITTED');
  assert.equal(journalData.operation, 'batch-track-clean');
  if (process.platform !== 'win32') {
    const mode = fs.statSync(res.journalPath).mode & 0o777;
    assert.equal(mode, 0o600);
  }
});

test('transactionalBatchTrackClean executes full compensating rollback on mid-batch and post-marker failures', async (t) => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'tx-batch-rollback-'));
  t.after(() => fs.rmSync(tmpDir, { recursive: true, force: true }));

  const journalDir = path.join(tmpDir, 'journal');
  const journal = new TransactionJournal(journalDir);
  const wav1 = path.join(tmpDir, 'clip1.wav');
  const wav2 = path.join(tmpDir, 'clip2.wav');
  createSyntheticWav(wav1, { sampleRate: 48000, channels: 1 });
  createSyntheticWav(wav2, { sampleRate: 48000, channels: 1 });

  // Case A: Mid-batch failure after clip 0 placed and clip 1 imported
  {
    const tracks = [{ name: 'Dialogue 1', subtype: 'mono' }];
    const deletedTimelineItems = [];
    const deletedTracks = [];
    const deletedPoolClips = [];
    const deletedMarkerFrames = [];

    const mockTimeline = {
      GetTrackCount: async () => tracks.length,
      GetTrackName: async (_type, idx) => tracks[idx - 1]?.name || '',
      AddTrack: async (_type, subtype) => {
        tracks.push({ name: `Audio ${tracks.length + 1}`, subtype });
        return true;
      },
      SetTrackName: async (_type, idx, name) => {
        tracks[idx - 1].name = name;
        return true;
      },
      AddMarker: async () => true,
      DeleteMarkerAtFrame: async (frameId) => {
        deletedMarkerFrames.push(frameId);
        return true;
      },
      DeleteClips: async (items) => {
        deletedTimelineItems.push(...items);
        return true;
      },
      DeleteTrack: async (_type, idx) => {
        deletedTracks.push(idx);
        return true;
      },
    };

    let idxCounter = 0;
    const mockMediaPool = {
      ImportMedia: async (paths) => [{ id: `mp-${++idxCounter}`, path: paths[0] }],
      AppendToTimeline: async (specs) => [{ id: `tl-${idxCounter}`, spec: specs[0] }],
      DeleteClips: async (clips) => {
        deletedPoolClips.push(...clips);
        return true;
      },
    };

    const manager = new TimelineTransactionManager({
      getProject: async () => ({ GetCurrentTimeline: async () => mockTimeline }),
      getMediaPool: async () => mockMediaPool,
      findItemByMediaId: async () => null,
      log: () => {},
      journal,
    });

    await assert.rejects(
      manager.transactionalBatchTrackClean({
        timeline: mockTimeline,
        clips: [
          { clipId: 'c1', startFrame: 100 },
          { clipId: 'c2', startFrame: 300 },
        ],
        cleanedByClipId: { c1: wav1, c2: wav2 },
        failPoint: 'mid_batch',
      }),
      /Injected mid-batch failure/
    );

    assert.equal(deletedTimelineItems.length, 1, 'first appended clip must be rolled back');
    assert.deepEqual(deletedTracks, [2], 'newly created track must be deleted');
    assert.equal(deletedPoolClips.length, 2, 'both imported media pool items must be purged');
    assert.equal(deletedMarkerFrames.length, 0);

    const journalFiles = fs.readdirSync(journalDir).filter((f) => f.endsWith('.json'));
    assert.equal(journalFiles.length, 1);
    const jPath = path.join(journalDir, journalFiles[0]);
    const rec = JSON.parse(fs.readFileSync(jPath, 'utf8'));
    assert.equal(rec.stage, 'ROLLED_BACK');
    if (process.platform !== 'win32') {
      assert.equal(fs.statSync(jPath).mode & 0o777, 0o600);
    }
  }

  // Case B: Post-marker failure on a REUSED track rolls back markers, items, pool clips, but preserves pre-existing track
  {
    const tracks = [
      { name: 'Dialogue 1', subtype: 'mono' },
      { name: 'HawaVoClean Cleaned', subtype: 'mono' },
    ];
    const deletedTimelineItems = [];
    const deletedTracks = [];
    const deletedPoolClips = [];
    const deletedMarkerFrames = [];

    const mockTimeline = {
      GetTrackCount: async () => tracks.length,
      GetTrackName: async (_type, idx) => tracks[idx - 1]?.name || '',
      AddTrack: async () => {
        throw new Error('AddTrack must not be called when reusing targetTrackName');
      },
      AddMarker: async () => true,
      DeleteMarkerAtFrame: async (frameId) => {
        deletedMarkerFrames.push(frameId);
        return true;
      },
      DeleteItem: async (item) => {
        deletedTimelineItems.push(item);
        return true;
      },
      DeleteTrack: async (_type, idx) => {
        deletedTracks.push(idx);
        return true;
      },
    };

    const mockMediaPool = {
      ImportMedia: async (paths) => [{ id: paths[0] }],
      AppendToTimeline: async (specs) => [{ id: `tl-${specs[0].recordFrame}` }],
      DeleteClips: async (clips) => {
        deletedPoolClips.push(...clips);
        return true;
      },
    };

    const manager = new TimelineTransactionManager({
      getProject: async () => ({ GetCurrentTimeline: async () => mockTimeline }),
      getMediaPool: async () => mockMediaPool,
      findItemByMediaId: async () => null,
      log: () => {},
      journal,
    });

    await assert.rejects(
      manager.transactionalBatchTrackClean({
        timeline: mockTimeline,
        clips: [
          { clipId: 'c1', startFrame: 100 },
          { clipId: 'c2', startFrame: 400 },
        ],
        cleanedByClipId: {
          c1: {
            filePath: wav1,
            report: { units: [{ unit_id: 1, start_time_s: 1.0, end_time_s: 2.0, final_decision: 'REVERTED' }] },
          },
          c2: {
            filePath: wav2,
            report: { units: [{ unit_id: 2, start_time_s: 2.0, end_time_s: 3.0, final_decision: 'CONTINUITY_REVERTED' }] },
          },
        },
        targetTrackName: 'HawaVoClean Cleaned',
        fps: 24,
        failPoint: 'after_markers',
      }),
      /Injected failure after guard marker placement/
    );

    assert.deepEqual(deletedMarkerFrames, [448, 124], 'markers added in tx must be deleted in reverse order');
    assert.equal(deletedTimelineItems.length, 2, 'all appended items must be deleted');
    assert.deepEqual(deletedTracks, [], 'pre-existing reused track must NOT be deleted on rollback');
    assert.equal(deletedPoolClips.length, 2, 'all imported media pool clips must be purged');
  }
});

test('transactionalBatchTrackClean rejects non-48kHz WAV before mutating timeline', async (t) => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'tx-batch-44k-'));
  t.after(() => fs.rmSync(tmpDir, { recursive: true, force: true }));

  const wav48k = path.join(tmpDir, 'ok_48k.wav');
  const wav44k = path.join(tmpDir, 'bad_44k.wav');
  createSyntheticWav(wav48k, { sampleRate: 48000, channels: 1 });
  createSyntheticWav(wav44k, { sampleRate: 44100, channels: 1 });

  let mutated = false;
  const mockTimeline = {
    GetTrackCount: async () => 1,
    AddTrack: async () => {
      mutated = true;
      return true;
    },
  };
  const mockMediaPool = {
    ImportMedia: async () => {
      mutated = true;
      return [];
    },
  };

  const manager = new TimelineTransactionManager({
    getProject: async () => ({ GetCurrentTimeline: async () => mockTimeline }),
    getMediaPool: async () => mockMediaPool,
    findItemByMediaId: async () => null,
    log: () => {},
    journal: new TransactionJournal(path.join(tmpDir, 'journal')),
  });

  await assert.rejects(
    manager.transactionalBatchTrackClean({
      timeline: mockTimeline,
      clips: [
        { clipId: 'c1', startFrame: 0 },
        { clipId: 'c2', startFrame: 100 },
      ],
      cleanedByClipId: { c1: wav48k, c2: wav44k },
    }),
    /must be delivered at 48 kHz/
  );
  assert.equal(mutated, false, 'invalid batch must fail before any timeline or media pool mutation');
});

test('applyGuardMarkers places markers and rolls back partial writes on error', async () => {
  const placed = [];
  const deleted = [];
  let failOnSecond = false;

  const mockTimeline = {
    AddMarker: async (frameId, color, name, note, duration) => {
      if (failOnSecond && placed.length === 1) {
        throw new Error('Resolve AddMarker IPC failure');
      }
      placed.push({ frameId, color, name, note, duration });
      return true;
    },
    DeleteMarkerAtFrame: async (frameId) => {
      deleted.push(frameId);
      return true;
    },
  };

  const manager = new TimelineTransactionManager({
    getProject: async () => ({ GetCurrentTimeline: async () => mockTimeline }),
    getMediaPool: async () => ({}),
    findItemByMediaId: async () => null,
    log: () => {},
  });

  const report = {
    units: [
      { unit_id: 1, start_time_s: 1.0, end_time_s: 2.0, final_decision: 'REVERTED', decision_reason: 'Hole' },
      { unit_id: 2, start_time_s: 3.0, end_time_s: 4.0, final_decision: 'CONTINUITY_REVERTED', decision_reason: 'Seam' },
    ],
  };

  const ok = await manager.applyGuardMarkers({ timeline: mockTimeline, report, clipStartFrame: 1000, fps: 24 });
  assert.equal(ok.success, true);
  assert.equal(ok.markersAdded, 2);
  assert.equal(placed.length, 2);

  placed.length = 0;
  failOnSecond = true;
  await assert.rejects(
    manager.applyGuardMarkers({ timeline: mockTimeline, report, clipStartFrame: 1000, fps: 24 }),
    /Resolve AddMarker IPC failure/
  );
  assert.deepEqual(deleted, [1024], 'marker placed prior to failure must be rolled back');
});

test('main.js and preload.js wire resolve:batch-clean-track and resolve:apply-guard-markers behind requireTrustedIpcSender', () => {
  const mainSource = fs.readFileSync(
    path.join(__dirname, '..', 'com.hawavoclean.resolve', 'main.js'),
    'utf8'
  );
  const preloadSource = fs.readFileSync(
    path.join(__dirname, '..', 'com.hawavoclean.resolve', 'preload.js'),
    'utf8'
  );

  assert.match(
    mainSource,
    /ipcMain\.handle\(\s*'hawa:resolve:batch-clean-track',\s*async\s*\(event,\s*payload\)\s*=>\s*\{\s*requireTrustedIpcSender\(event\);\s*return\s+resolveBatchCleanTrack\(payload\);/
  );
  assert.match(
    mainSource,
    /ipcMain\.handle\(\s*'hawa:resolve:apply-guard-markers',\s*async\s*\(event,\s*payload\)\s*=>\s*\{\s*requireTrustedIpcSender\(event\);\s*return\s+resolveApplyGuardMarkers\(payload\);/
  );
  assert.ok(
    !mainSource.includes('failPoint: payload'),
    'main.js IPC bridge must never forward untrusted test failPoint from renderer payload'
  );
  assert.match(
    mainSource,
    /async function resolveBatchCleanTrack\(payload\)\s*\{\s*const validated = validateBatchCleanTrackPayload\(payload\);\s*const mgr = getTimelineTxManager\(\);/
  );
  assert.match(
    mainSource,
    /async function resolveApplyGuardMarkers\(payload\)\s*\{\s*const validated = validateApplyGuardMarkersPayload\(payload\);\s*const mgr = getTimelineTxManager\(\);/
  );
  assert.match(preloadSource, /batchCleanTrack:\s*\(payload\)\s*=>\s*invoke\('hawa:resolve:batch-clean-track',\s*payload\)/);
  assert.match(preloadSource, /applyGuardMarkers:\s*\(payload\)\s*=>\s*invoke\('hawa:resolve:apply-guard-markers',\s*payload\)/);
});

test('extractGuardMarkersFromReport survives corrupt reports, invalid fps, negative frames, inverted times, and >10KB notes', () => {
  // Corrupt / non-object reports and non-array units
  for (const badReport of [null, undefined, 42, 'corrupt', true, [], { units: null }, { units: 'bad' }, { units: {} }]) {
    assert.deepEqual(extractGuardMarkersFromReport(badReport, 100, 24), []);
  }

  // Invalid fps values (NaN, Infinity, -Infinity, 0, negative, booleans, empty string)
  const minimalReport = { units: [{ unit_id: 1, start_time_s: 1, end_time_s: 2, final_decision: 'REVERTED' }] };
  for (const badFps of [NaN, Infinity, -Infinity, 0, -24, true, false, '', null, undefined, 'abc']) {
    assert.deepEqual(extractGuardMarkersFromReport(minimalReport, 100, badFps), []);
  }

  // Null options must not crash destructuring
  assert.equal(extractGuardMarkersFromReport(minimalReport, 100, 24, null).length, 1);

  // Negative clipStartFrame, inverted start_time_s > end_time_s, huge >10KB note & score flood, NUL bytes
  const hugeReason = 'A'.repeat(16_384) + '\0tail';
  const hugeScores = {};
  for (let i = 0; i < 500; i += 1) {
    hugeScores[`metric_${String(i).padStart(4, '0')}`] = 'X'.repeat(200);
  }
  const adversarialReport = {
    units: [
      null,
      'string-unit',
      [],
      { final_decision: 'REVERTED', start_time_s: null, end_time_s: 2 }, // null start_time_s must be skipped
      { final_decision: 'REVERTED', start_time_s: NaN, end_time_s: 2 },
      {
        unit_id: 7,
        start_time_s: 5.0,
        end_time_s: 2.0, // inverted start > end
        final_decision: 'REVERTED',
        decision_reason: hugeReason,
        guard_a_scores: hugeScores,
      },
      {
        unit_id: 8,
        start_time_s: -10.0,
        end_time_s: -5.0,
        final_decision: 'ERROR_PASSTHROUGH',
        decision_reason: 'Negative range\0sanitized',
      },
    ],
  };

  const markers = extractGuardMarkersFromReport(adversarialReport, -500, 24);
  assert.equal(markers.length, 2);

  for (const m of markers) {
    assert.ok(Number.isSafeInteger(m.frameOffset) && m.frameOffset >= 0, `frameOffset must be non-negative safe int, got ${m.frameOffset}`);
    assert.ok(Number.isSafeInteger(m.durationFrames) && m.durationFrames >= 1, `durationFrames must be >= 1, got ${m.durationFrames}`);
    assert.ok(m.note.length <= MAX_MARKER_NOTE_LEN, `marker note exceeded cap: ${m.note.length}`);
    assert.ok(m.name.length <= MAX_MARKER_NAME_LEN, `marker name exceeded cap: ${m.name.length}`);
    assert.ok(!m.note.includes('\0'), 'marker note must not contain NUL');
    assert.ok(!m.name.includes('\0'), 'marker name must not contain NUL');
  }
  // Inverted start_time_s > end_time_s clamps cleanly to 1-frame marker
  const inverted = markers.find((m) => m.unitId === 7);
  assert.ok(inverted);
  assert.equal(inverted.frameOffset, 120);
  assert.equal(inverted.durationFrames, 1);
});

test('transactionalBatchTrackClean rejects duplicate clip IDs, missing WAV paths, and prototype keys before touching Resolve', async (t) => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'tx-batch-adv-'));
  t.after(() => fs.rmSync(tmpDir, { recursive: true, force: true }));

  const wav1 = path.join(tmpDir, 'c1.wav');
  createSyntheticWav(wav1, { sampleRate: 48000, channels: 1 });

  let resolveTouched = false;
  const manager = new TimelineTransactionManager({
    getProject: async () => {
      resolveTouched = true;
      return { GetCurrentTimeline: async () => ({}) };
    },
    getMediaPool: async () => {
      resolveTouched = true;
      return {};
    },
    findItemByMediaId: async () => null,
    log: () => {},
    journal: new TransactionJournal(path.join(tmpDir, 'journal')),
  });

  // Duplicate clipId
  await assert.rejects(
    manager.transactionalBatchTrackClean({
      clips: [
        { clipId: 'dup-1', startFrame: 0 },
        { clipId: 'dup-1', startFrame: 240 },
      ],
      cleanedByClipId: { 'dup-1': wav1 },
    }),
    /duplicate clipId 'dup-1'/
  );
  assert.equal(resolveTouched, false, 'duplicate clipId must fail before touching Resolve API');

  // Missing WAV file on disk
  await assert.rejects(
    manager.transactionalBatchTrackClean({
      clips: [{ clipId: 'c1', startFrame: 0 }],
      cleanedByClipId: { c1: path.join(tmpDir, 'missing.wav') },
    }),
    /batchTrackClean rejected clip 'c1'/
  );
  assert.equal(resolveTouched, false, 'missing WAV must fail before touching Resolve API');

  // Missing cleaned entry via prototype key ('toString')
  await assert.rejects(
    manager.transactionalBatchTrackClean({
      clips: [{ clipId: 'toString', startFrame: 0 }],
      cleanedByClipId: { c1: wav1 },
    }),
    /missing cleaned audio entry for clip 'toString'/
  );
  assert.equal(resolveTouched, false, 'prototype key must not satisfy cleanedByClipId lookup');

  // Invalid fps (NaN / Infinity / <=0)
  for (const badFps of [NaN, Infinity, 0, -24, true]) {
    await assert.rejects(
      manager.transactionalBatchTrackClean({
        clips: [{ clipId: 'c1', startFrame: 0 }],
        cleanedByClipId: { c1: wav1 },
        fps: badFps,
      }),
      /invalid fps/
    );
  }
  assert.equal(resolveTouched, false);
});

test('partial rollback continues cleaning up remaining resources when DeleteMarkersByColor, DeleteClips, and DeleteTrack throw', async (t) => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'tx-batch-partial-rb-'));
  t.after(() => fs.rmSync(tmpDir, { recursive: true, force: true }));

  const journalDir = path.join(tmpDir, 'journal');
  const journal = new TransactionJournal(journalDir);
  const wav1 = path.join(tmpDir, 'c1.wav');
  const wav2 = path.join(tmpDir, 'c2.wav');
  createSyntheticWav(wav1, { sampleRate: 48000, channels: 1 });
  createSyntheticWav(wav2, { sampleRate: 48000, channels: 1 });

  const tracks = [{ name: 'Dialogue 1', subtype: 'mono' }];
  const colorDeletesAttempted = [];
  const itemDeletesAttempted = [];
  const trackDeletesAttempted = [];
  const poolDeletesAttempted = [];

  // Timeline implements DeleteMarkersByColor (not DeleteMarkerAtFrame) and throws during rollback!
  const mockTimeline = {
    GetTrackCount: async () => tracks.length,
    GetTrackName: async (_type, idx) => tracks[idx - 1]?.name || '',
    AddTrack: async (_type, subtype) => {
      tracks.push({ name: `Audio ${tracks.length + 1}`, subtype });
      return true;
    },
    SetTrackName: async (_type, idx, name) => {
      tracks[idx - 1].name = name;
      return true;
    },
    AddMarker: async () => true,
    DeleteMarkersByColor: async (color) => {
      colorDeletesAttempted.push(color);
      if (color === 'Red') throw null; // non-Error throw must NOT abort rollback!
      throw new Error(`DeleteMarkersByColor(${color}) IPC failure`);
    },
    DeleteClips: async () => {
      throw new Error('Timeline DeleteClips failed');
    },
    DeleteItem: async (item) => {
      itemDeletesAttempted.push(item.id);
      return true;
    },
    DeleteTrack: async (_type, idx) => {
      trackDeletesAttempted.push(idx);
      throw new Error(`DeleteTrack(${idx}) locked`);
    },
  };

  let mpCounter = 0;
  const mockMediaPool = {
    ImportMedia: async (paths) => [{ id: `mp-${++mpCounter}`, path: paths[0] }],
    AppendToTimeline: async (specs) => [{ id: `tl-${specs[0].recordFrame}` }],
    DeleteClips: async (clips) => {
      poolDeletesAttempted.push(...clips.map((c) => c.id));
      return true;
    },
  };

  const manager = new TimelineTransactionManager({
    getProject: async () => ({ GetCurrentTimeline: async () => mockTimeline }),
    getMediaPool: async () => mockMediaPool,
    findItemByMediaId: async () => null,
    log: () => {},
    journal,
  });

  await assert.rejects(
    manager.transactionalBatchTrackClean({
      timeline: mockTimeline,
      clips: [
        { clipId: 'c1', startFrame: 100 },
        { clipId: 'c2', startFrame: 300 },
      ],
      cleanedByClipId: {
        c1: {
          filePath: wav1,
          report: { units: [{ unit_id: 1, start_time_s: 1, end_time_s: 2, final_decision: 'REVERTED' }] },
        },
        c2: {
          filePath: wav2,
          report: { units: [{ unit_id: 2, start_time_s: 1, end_time_s: 2, final_decision: 'LOW_STRENGTH', chosen_strength: 0.25 }] },
        },
      },
      fps: 24,
      failPoint: 'after_markers',
    }),
    /Injected failure after guard marker placement/
  );

  // Every rollback stage must have run despite throws in DeleteMarkersByColor, DeleteClips, and DeleteTrack!
  assert.deepEqual(colorDeletesAttempted.sort(), ['Red', 'Yellow']);
  assert.deepEqual(itemDeletesAttempted, ['tl-300', 'tl-100'], 'DeleteItem fallback must clean up timeline items when DeleteClips throws');
  assert.deepEqual(trackDeletesAttempted, [2]);
  assert.deepEqual(poolDeletesAttempted, ['mp-1', 'mp-2'], 'mediaPool.DeleteClips must STILL run after DeleteTrack throws');

  const journalFiles = fs.readdirSync(journalDir).filter((f) => f.endsWith('.json'));
  assert.equal(journalFiles.length, 1);
  const rec = JSON.parse(fs.readFileSync(path.join(journalDir, journalFiles[0]), 'utf8'));
  assert.equal(rec.stage, 'ROLLED_BACK');
  assert.ok(rec.compensations.some((c) => c.action === 'delete_markers_by_color' && c.status === 'error'));
  assert.ok(rec.compensations.some((c) => c.action === 'delete_track' && c.status === 'error'));
  assert.ok(rec.compensations.some((c) => c.action === 'delete_media_pool_clips' && c.status === 'ok'));
});

test('applyGuardMarkers continues rollback when DeleteMarkersByColor throws and truncates oversized notes', async () => {
  const placed = [];
  const colorDeletes = [];
  let failOnThird = false;

  const mockTimeline = {
    AddMarker: async (frameId, color, name, note, duration) => {
      if (failOnThird && placed.length === 2) {
        throw new Error('Marker 3 IPC crash');
      }
      placed.push({ frameId, color, name, note, duration });
      return true;
    },
    DeleteMarkersByColor: async (color) => {
      colorDeletes.push(color);
      throw new Error(`DeleteMarkersByColor(${color}) threw`);
    },
  };

  const manager = new TimelineTransactionManager({
    getProject: async () => ({ GetCurrentTimeline: async () => mockTimeline }),
    getMediaPool: async () => ({}),
    findItemByMediaId: async () => null,
    log: () => {},
  });

  // Huge >10KB note is truncated safely when applied directly
  const ok = await manager.applyGuardMarkers({
    timeline: mockTimeline,
    markers: [{ frameOffset: 50, color: 'Red', name: 'N'.repeat(1000), note: 'Z'.repeat(20_000), durationFrames: 12 }],
  });
  assert.equal(ok.markersAdded, 1);
  assert.ok(placed[0].note.length <= MAX_MARKER_NOTE_LEN);
  assert.ok(placed[0].name.length <= MAX_MARKER_NAME_LEN);

  // Partial rollback with throwing DeleteMarkersByColor attempts every color and rethrows original error
  placed.length = 0;
  failOnThird = true;
  await assert.rejects(
    manager.applyGuardMarkers({
      timeline: mockTimeline,
      markers: [
        { frameOffset: 10, color: 'Red', name: 'M1', note: 'n1', durationFrames: 1 },
        { frameOffset: 20, color: 'Yellow', name: 'M2', note: 'n2', durationFrames: 1 },
        { frameOffset: 30, color: 'Red', name: 'M3', note: 'n3', durationFrames: 1 },
      ],
    }),
    /Marker 3 IPC crash/
  );
  assert.deepEqual(colorDeletes.sort(), ['Red', 'Yellow']);
});

test('IPC validators reject malformed, non-array, duplicate, and oversized payloads before Resolve API', () => {
  // validateBatchCleanTrackPayload
  for (const bad of [null, undefined, [], 'str', 123]) {
    assert.throws(() => validateBatchCleanTrackPayload(bad), /payload must be a plain object/);
  }
  assert.throws(() => validateBatchCleanTrackPayload({ clips: 'not-array', cleanedByClipId: {} }), /clips must be a non-empty array/);
  assert.throws(() => validateBatchCleanTrackPayload({ clips: [], cleanedByClipId: {} }), /clips must be a non-empty array/);
  assert.throws(
    () => validateBatchCleanTrackPayload({
      clips: Array.from({ length: MAX_BATCH_CLIPS + 1 }, (_, i) => ({ clipId: `c${i}`, startFrame: i })),
      cleanedByClipId: { c0: '/tmp/a.wav' },
    }),
    /exceeds maximum batch size/
  );
  assert.throws(
    () => validateBatchCleanTrackPayload({
      clips: [{ clipId: 'c1', startFrame: 0 }, { clipId: 'c1', startFrame: 10 }],
      cleanedByClipId: { c1: '/tmp/a.wav' },
    }),
    /duplicate clipId/
  );
  assert.throws(
    () => validateBatchCleanTrackPayload({
      clips: [{ clipId: 'c1', startFrame: -5 }],
      cleanedByClipId: { c1: '/tmp/a.wav' },
    }),
    /invalid startFrame/
  );
  assert.throws(
    () => validateBatchCleanTrackPayload({
      clips: [{ clipId: 'c1', startFrame: 0 }],
      cleanedByClipId: { c1: 'relative/path.wav' },
    }),
    /bounded absolute path/
  );
  assert.throws(
    () => validateBatchCleanTrackPayload({
      clips: [{ clipId: 'c1', startFrame: 0 }],
      cleanedByClipId: { c1: '/tmp/a.wav' },
      fps: NaN,
    }),
    /fps must be a finite number/
  );
  assert.throws(
    () => validateBatchCleanTrackPayload({
      clips: [{ clipId: 'c1', startFrame: 0 }],
      cleanedByClipId: { c1: '/tmp/a.wav' },
      addGuardMarkers: 'yes',
    }),
    /addGuardMarkers must be a boolean/
  );

  // Strips untrusted failPoint / timeline from valid DTO
  const cleanBatch = validateBatchCleanTrackPayload({
    clips: [{ clipId: 'c1', startFrame: 10 }],
    cleanedByClipId: { c1: '/tmp/a.wav' },
    fps: 24,
    failPoint: 'mid_batch',
    timeline: {},
  });
  assert.equal('failPoint' in cleanBatch, false);
  assert.equal('timeline' in cleanBatch, false);

  // validateApplyGuardMarkersPayload
  for (const bad of [null, undefined, [], 'str', 99]) {
    assert.throws(() => validateApplyGuardMarkersPayload(bad), /payload must be a plain object/);
  }
  assert.throws(() => validateApplyGuardMarkersPayload({ markers: 'not-array' }), /markers must be an array/);
  assert.throws(
    () => validateApplyGuardMarkersPayload({
      markers: Array.from({ length: MAX_GUARD_MARKERS + 1 }, (_, i) => ({ frameOffset: i })),
    }),
    /exceeds maximum count/
  );
  assert.throws(
    () => validateApplyGuardMarkersPayload({
      markers: [{ frameOffset: 10, note: 'X'.repeat(MAX_IPC_MARKER_NOTE_LEN + 1) }],
    }),
    /oversized note/
  );
  assert.throws(
    () => validateApplyGuardMarkersPayload({
      markers: [{ frameOffset: -1 }],
    }),
    /invalid frameOffset/
  );
  assert.throws(
    () => validateApplyGuardMarkersPayload({
      report: { units: Array.from({ length: MAX_GUARD_REPORT_UNITS + 1 }, () => ({})) },
    }),
    /at most/
  );
  assert.throws(() => validateApplyGuardMarkersPayload({ reportPath: 'relative.json' }), /bounded absolute path/);
  assert.throws(() => validateApplyGuardMarkersPayload({ fps: Infinity }), /fps must be a finite number/);
  assert.throws(() => validateApplyGuardMarkersPayload({ clipStartFrame: -10 }), /clipStartFrame must be a finite non-negative/);
});
