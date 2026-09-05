// @ts-check
'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');

const {
  validateAudioFormat,
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
