// @ts-check
'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const crypto = require('node:crypto');
const { execFileSync } = require('node:child_process');

const mainModule = require('../com.hawavoclean.resolve/main.js');
const {
  CPU_TYPE_ARM64,
  CPU_TYPE_X86_64,
  computeFileSha256,
  acquireWorkflowIntegrationNode,
} = require('../com.hawavoclean.resolve/sdk-preflight.js');
const {
  TimelineTransactionManager,
  TransactionJournal,
} = require('../com.hawavoclean.resolve/timeline-transactions.js');

const RESOLVE_PLUGIN_DIR = path.join(__dirname, '..');
const INSTALL_SH = path.join(RESOLVE_PLUGIN_DIR, 'install.sh');
const ACTIVATE_SH = path.join(RESOLVE_PLUGIN_DIR, 'activate.sh');
const BUILD_PKG_SH = path.join(RESOLVE_PLUGIN_DIR, 'build-pkg.sh');
const MAIN_JS_PATH = path.join(RESOLVE_PLUGIN_DIR, 'com.hawavoclean.resolve', 'main.js');

function createSyntheticUniversalNode(filePath) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const buf = Buffer.alloc(128);
  buf.writeUInt32BE(0xcafebabe, 0); // FAT_MAGIC
  buf.writeUInt32BE(2, 4);
  buf.writeUInt32BE(CPU_TYPE_ARM64, 8);
  buf.writeUInt32BE(CPU_TYPE_X86_64, 28);
  fs.writeFileSync(filePath, buf);
}

function createSynthetic48kWav(filePath, channels = 1) {
  const numFrames = 100;
  const bitsPerSample = 16;
  const blockAlign = channels * (bitsPerSample / 8);
  const dataSize = numFrames * blockAlign;
  const totalSize = 44 + dataSize;
  const buf = Buffer.alloc(totalSize);
  buf.write('RIFF', 0);
  buf.writeUInt32LE(totalSize - 8, 4);
  buf.write('WAVE', 8);
  buf.write('fmt ', 12);
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20);
  buf.writeUInt16LE(channels, 22);
  buf.writeUInt32LE(48000, 24);
  buf.writeUInt32LE(48000 * blockAlign, 28);
  buf.writeUInt16LE(blockAlign, 32);
  buf.writeUInt16LE(bitsPerSample, 34);
  buf.write('data', 36);
  buf.writeUInt32LE(dataSize, 40);
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, buf);
}

test('sdk-preflight.js is staged in install.sh, verified in activate.sh, and packaged with postinstall acquisition in build-pkg.sh', (t) => {
  const installSrc = fs.readFileSync(INSTALL_SH, 'utf8');
  const activateSrc = fs.readFileSync(ACTIVATE_SH, 'utf8');
  const buildPkgSrc = fs.readFileSync(BUILD_PKG_SH, 'utf8');

  assert.ok(installSrc.includes('sdk-preflight.js'), 'install.sh must copy sdk-preflight.js');
  assert.ok(activateSrc.includes('sdk-preflight.js'), 'activate.sh must verify sdk-preflight.js');
  assert.ok(buildPkgSrc.includes('sdk-preflight.js'), 'build-pkg.sh must package and invoke sdk-preflight.js');
  assert.ok(buildPkgSrc.includes('acquireWorkflowIntegrationNode'), 'build-pkg.sh postinstall must invoke acquireWorkflowIntegrationNode');

  // Verify build-pkg.sh actually packages sdk-preflight.js into the generated macOS .pkg payload
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'wave-g-pkg-'));
  t.after(() => fs.rmSync(tmpDir, { recursive: true, force: true }));

  const stageDir = path.join(tmpDir, 'stage');
  fs.mkdirSync(stageDir, { recursive: true });
  fs.writeFileSync(path.join(stageDir, 'PLUGIN_ID'), 'com.hawavoclean.resolve\n');
  fs.writeFileSync(path.join(stageDir, 'VERSION'), '3.3.0\n');
  fs.writeFileSync(path.join(stageDir, 'main.js'), '// main\n');
  const mainHash = computeFileSha256(path.join(stageDir, 'main.js'));
  fs.writeFileSync(path.join(stageDir, 'SHA256SUMS'), `${mainHash}  ./main.js\n`);

  const outputPkg = path.join(tmpDir, 'com.hawavoclean.resolve.pkg');
  execFileSync(BUILD_PKG_SH, ['--stage', stageDir, '--output', outputPkg]);
  assert.ok(fs.existsSync(outputPkg));

  if (process.platform === 'darwin') {
    const payload = execFileSync('pkgutil', ['--payload-files', outputPkg], { encoding: 'utf8' });
    assert.match(payload, /com\.hawavoclean\.resolve\/sdk-preflight\.js/);
  }
});

test('loadWorkflowIntegration acquires local WorkflowIntegration.node with SHA-256 verification and falls back to user cache when pluginDir is read-only', (t) => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'wave-g-sdk-load-'));
  t.after(() => fs.rmSync(tmpDir, { recursive: true, force: true }));

  const sourceNode = path.join(tmpDir, 'resolve-sdk', 'WorkflowIntegration.node');
  createSyntheticUniversalNode(sourceNode);
  const expectedSha = computeFileSha256(sourceNode);

  // 1. Writable plugin dir: acquires directly into pluginDir with SHA-256 verification
  const writablePluginDir = path.join(tmpDir, 'plugin-writable');
  fs.mkdirSync(writablePluginDir, { recursive: true });
  const loadedPaths = [];
  const fakeBridge = { Initialize: () => true };

  const mod = mainModule.loadWorkflowIntegration({
    pluginDir: writablePluginDir,
    cacheDir: path.join(tmpDir, 'cache-unused'),
    sdkOptions: { platform: 'darwin', targetArch: 'arm64', overrideNodePath: sourceNode },
    requireFn: (p) => {
      loadedPaths.push(p);
      return fakeBridge;
    },
  });

  assert.equal(mod, fakeBridge);
  assert.equal(loadedPaths.length, 1);
  assert.equal(loadedPaths[0], path.join(writablePluginDir, 'WorkflowIntegration.node'));
  assert.equal(computeFileSha256(loadedPaths[0]), expectedSha);

  // 2. Checksum mismatch rejected cleanly by acquireWorkflowIntegrationNode
  const mismatch = acquireWorkflowIntegrationNode(path.join(tmpDir, 'bad-checksum'), {
    platform: 'darwin',
    targetArch: 'arm64',
    overrideNodePath: sourceNode,
    expectedSha256: '0'.repeat(64),
  });
  assert.equal(mismatch.ok, false);
  assert.equal(mismatch.code, 'ERR_CHECKSUM_MISMATCH');

  // 3. Read-only plugin dir simulation: falls back to user cache dir
  const cacheDir = path.join(tmpDir, 'user-cache');
  const cacheLoadedPaths = [];
  const modCached = mainModule.loadWorkflowIntegration({
    pluginDir: path.join(tmpDir, 'plugin-ro'),
    cacheDir,
    sdkOptions: { platform: 'darwin', targetArch: 'arm64', overrideNodePath: sourceNode },
    acquireWorkflowIntegrationNode: (dir, opts) => {
      if (dir.endsWith('plugin-ro')) {
        return { ok: false, code: 'ERR_ACQUISITION_FAILED', error: 'EACCES: permission denied' };
      }
      return acquireWorkflowIntegrationNode(dir, opts);
    },
    requireFn: (p) => {
      cacheLoadedPaths.push(p);
      return fakeBridge;
    },
  });

  assert.equal(modCached, fakeBridge);
  assert.equal(cacheLoadedPaths[0], path.join(cacheDir, 'WorkflowIntegration.node'));
  assert.equal(computeFileSha256(cacheLoadedPaths[0]), expectedSha);
});

test('resolveGetSelectedClip inspects audio timeline tracks near playhead and returns exact startFrame, leftOffset, durationFrames, trackIndex, mediaType, and fps', async () => {
  const makeMediaPoolItem = ({ mediaId, name, filePath, audioCh = '1', duration = '00:00:10:00', fps = '29.97' }) => ({
    GetMediaId: async () => mediaId,
    GetName: async () => name,
    GetClipProperty: async (key) => {
      if (key === 'File Path') return filePath;
      if (key === 'Audio Ch') return audioCh;
      if (key === 'Duration') return duration;
      if (key === 'FPS') return fps;
      return '';
    },
  });

  const makeTimelineItem = ({ start, end, leftOffset, mpItem, trackIndex = 1 }) => ({
    GetStart: async () => start,
    GetEnd: async () => end,
    GetDuration: async () => end - start,
    GetLeftOffset: async () => leftOffset,
    GetTrackIndex: async () => trackIndex,
    GetMediaPoolItem: async () => mpItem,
  });

  // Case A: Video-only B-roll clip on V1 (Audio Ch = '0') + active dialogue clip on A2 under playhead at 29.97 fps
  const brollMp = makeMediaPoolItem({ mediaId: 'v-broll', name: 'BROLL.mov', filePath: '/Media/BROLL.mov', audioCh: '0' });
  const a1Mp = makeMediaPoolItem({ mediaId: 'a-earlier', name: 'INTRO.wav', filePath: '/Media/INTRO.wav', audioCh: '1' });
  const a2Mp = makeMediaPoolItem({ mediaId: 'a-dialogue', name: 'DIALOGUE_A2.wav', filePath: '/Media/DIALOGUE_A2.wav', audioCh: '1' });

  const timeline2997 = {
    GetSetting: async (k) => (k === 'timelineFrameRate' ? '29.97' : null),
    GetCurrentTimecode: async () => '01:00:10:00', // 3610 * 30 = 108300
    GetCurrentVideoItem: async () => makeTimelineItem({ start: 108000, end: 108600, leftOffset: 0, mpItem: brollMp, trackIndex: 1 }),
    GetTrackCount: async (type) => (type === 'audio' ? 2 : 1),
    GetItemListInTrack: async (type, idx) => {
      if (type !== 'audio') return [];
      if (idx === 1) return [makeTimelineItem({ start: 107000, end: 107300, leftOffset: 10, mpItem: a1Mp })];
      if (idx === 2) return [makeTimelineItem({ start: 108200, end: 108500, leftOffset: 48, mpItem: a2Mp })];
      return [];
    },
  };

  const selectedAudio = await mainModule.resolveGetSelectedClip({
    getMediaPool: async () => ({ GetSelectedClips: async () => [] }),
    getProject: async () => ({ GetCurrentTimeline: async () => timeline2997 }),
  });

  assert.ok(selectedAudio);
  assert.equal(selectedAudio.mediaId, 'a-dialogue');
  assert.equal(selectedAudio.startFrame, 108200);
  assert.equal(selectedAudio.leftOffset, 48);
  assert.equal(selectedAudio.durationFrames, 300);
  assert.equal(selectedAudio.trackIndex, 2);
  assert.equal(selectedAudio.mediaType, 'audio');
  assert.equal(selectedAudio.fps, 29.97);

  // Case B: Audio-only timeline (GetCurrentVideoItem returns null) at 23.976 fps
  const podcastMp = makeMediaPoolItem({ mediaId: 'a-pod', name: 'PODCAST.wav', filePath: '/Media/PODCAST.wav', audioCh: '2', fps: '23.976' });
  const timeline23976 = {
    GetSetting: async (k) => (k === 'timelineFrameRate' ? '23.976' : null),
    GetCurrentTimecode: async () => '01:00:05:00', // 3605 * 24 = 86520
    GetCurrentVideoItem: async () => null,
    GetTrackCount: async (type) => (type === 'audio' ? 1 : 0),
    GetItemListInTrack: async (type, idx) => {
      if (type === 'audio' && idx === 1) {
        return [makeTimelineItem({ start: 86400, end: 86880, leftOffset: 24, mpItem: podcastMp })];
      }
      return [];
    },
  };

  const selectedPodcast = await mainModule.resolveGetSelectedClip({
    getMediaPool: async () => ({ GetSelectedClips: async () => [] }),
    getProject: async () => ({ GetCurrentTimeline: async () => timeline23976 }),
  });

  assert.ok(selectedPodcast);
  assert.equal(selectedPodcast.mediaId, 'a-pod');
  assert.equal(selectedPodcast.startFrame, 86400);
  assert.equal(selectedPodcast.leftOffset, 24);
  assert.equal(selectedPodcast.durationFrames, 480);
  assert.equal(selectedPodcast.trackIndex, 1);
  assert.equal(selectedPodcast.mediaType, 'audio');
  assert.equal(selectedPodcast.fps, 23.976);

  // Verify drop-frame timecode parsing at 29.97 fps: 00:01:00;02 = 1800 frames (2 frames dropped at minute 1)
  assert.equal(mainModule.parseTimecodeFrames('00:01:00;02', 29.97), 1800);
});

test('transactionalNewTrack and transactionalBatchTrackClean honor real startFrame and timeline fps across 23.976/25/29.97/50/60 fps', async (t) => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'wave-g-fps-'));
  t.after(() => fs.rmSync(tmpDir, { recursive: true, force: true }));

  const wavPath = path.join(tmpDir, 'cleaned_48k.wav');
  createSynthetic48kWav(wavPath, 1);

  const report = {
    units: [
      {
        unit_id: 1,
        start_time_s: 2.0,
        end_time_s: 3.0,
        final_decision: 'REVERTED',
        decision_reason: 'guard_a_rejected',
      },
    ],
  };

  for (const fps of [23.976, 25, 29.97, 50, 60]) {
    const journal = new TransactionJournal(path.join(tmpDir, `journal-${fps}`));
    const appendedSpecs = [];
    const addedMarkers = [];
    const tracks = ['audio 1'];

    const mockTimeline = {
      GetSetting: async (k) => (k === 'timelineFrameRate' ? String(fps) : null),
      GetTrackCount: async () => tracks.length,
      GetTrackName: async () => 'Audio 1',
      AddTrack: async (_t, sub) => {
        tracks.push(`audio ${tracks.length + 1} (${sub})`);
        return true;
      },
      SetTrackName: async () => true,
      AddMarker: async (frameOffset, color, name, note, durationFrames) => {
        addedMarkers.push({ frameOffset, color, name, note, durationFrames });
        return true;
      },
    };

    const mockMediaPool = {
      ImportMedia: async () => [{ GetMediaId: async () => `imp-${fps}` }],
      AppendToTimeline: async (specs) => {
        appendedSpecs.push(...specs);
        return specs.map((_, i) => ({ id: i }));
      },
    };

    const mgr = new TimelineTransactionManager({
      getProject: async () => ({ GetCurrentTimeline: async () => mockTimeline }),
      getMediaPool: async () => mockMediaPool,
      findItemByMediaId: async () => null,
      log: () => {},
      journal,
    });

    // 1. transactionalNewTrack honors explicit startFrame (never blindly 0) & reads real timeline fps
    const nt = await mgr.transactionalNewTrack('clip-a', wavPath, {
      startFrame: 90000,
      leftOffset: 12,
      durationFrames: 480,
      mediaType: 'audio',
      report,
    });

    assert.equal(nt.success, true);
    assert.equal(nt.recordFrame, 90000);
    assert.equal(nt.fps, fps);
    assert.equal(appendedSpecs[0].recordFrame, 90000);
    assert.equal(appendedSpecs[0].startFrame, 12);
    assert.equal(appendedSpecs[0].endFrame, 492);
    assert.equal(appendedSpecs[0].mediaType, 2);

    // Marker offset accounts for sourceOffsetS = 12 / fps: relStartS = 2.0 - 12/fps => frameOffset = 90000 + round((2.0 - 12/fps) * fps)
    const expectedNewTrackMarkerFrame = 90000 + Math.round((2.0 - 12 / fps) * fps);
    assert.equal(addedMarkers[0].frameOffset, expectedNewTrackMarkerFrame);

    // 2. transactionalBatchTrackClean with omitted fps reads real timelineFrameRate from activeTimeline
    addedMarkers.length = 0;
    appendedSpecs.length = 0;

    const batch = await mgr.transactionalBatchTrackClean({
      clips: [{ clipId: 'c1', startFrame: 86400, mediaType: 'audio' }],
      cleanedByClipId: { c1: { filePath: wavPath, report } },
    });

    assert.equal(batch.success, true);
    assert.equal(batch.fps, fps);
    assert.equal(appendedSpecs[0].recordFrame, 86400);
    assert.equal(appendedSpecs[0].mediaType, 2);
    assert.equal(addedMarkers[0].frameOffset, 86400 + Math.round(2.0 * fps));
    assert.equal(addedMarkers[0].durationFrames, Math.round(1.0 * fps));
  }
});

test('stale-rendezvous fallback resolves Desktop app engine before local engine.json', (t) => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'wave-g-fallback-'));
  t.after(() => fs.rmSync(tmpDir, { recursive: true, force: true }));

  const fakeDesktopEngine = path.join(tmpDir, 'hawavoclean-engine');
  fs.writeFileSync(fakeDesktopEngine, '#!/bin/sh\necho ok\n', { mode: 0o700 });

  // 1. When Desktop app engine exists, resolveDesktopEngineFallback returns installed_desktop_engine
  const fallback = mainModule.resolveDesktopEngineFallback({
    findDesktopEngineExecutable: () => fakeDesktopEngine,
  });
  assert.ok(fallback);
  assert.equal(fallback.type, 'installed_desktop_engine');
  assert.equal(fallback.executable, fakeDesktopEngine);
  assert.deepEqual(fallback.command, [fakeDesktopEngine, 'serve']);

  // 2. When Desktop app engine is absent, returns null so spawnLocalEngine falls back to engine.json
  const missing = mainModule.resolveDesktopEngineFallback({
    findDesktopEngineExecutable: () => null,
  });
  assert.equal(missing, null);

  // 3. Verify main.js stale rendezvous branch invokes resolveDesktopEngineFallback() instead of spawnLocalEngine(null)
  const mainSrc = fs.readFileSync(MAIN_JS_PATH, 'utf8');
  assert.match(mainSrc, /desktopFallback = resolveDesktopEngineFallback\(\)/);
  assert.match(mainSrc, /spawnLocalEngine\(desktopFallback\)/);
});

test('end-user error HTML contains calm actionable guidance and zero developer commands', () => {
  const html = mainModule.errorPageHtml(
    'The HawaVoClean engine could not be started',
    'Could not connect to the audio engine.'
  );

  // Must not leak developer tooling instructions to end users
  assert.doesNotMatch(html, /brew install/i);
  assert.doesNotMatch(html, /Homebrew/i);
  assert.doesNotMatch(html, /npm start/i);
  assert.doesNotMatch(html, /engine\.json\.example/i);
  assert.doesNotMatch(html, /resolve-plugin\/install\.sh/i);

  // Must guide end users calmly to Desktop app or installer package
  assert.match(html, /HawaVoClean Desktop/);
  assert.match(html, /installer package/i);
  assert.match(html, /Workflow Integrations/);

  // Verify missing-UI fallback page in main.js is also free of developer commands
  const mainSrc = fs.readFileSync(MAIN_JS_PATH, 'utf8');
  assert.doesNotMatch(mainSrc, /re-run <code>npm start<\/code>/);
  assert.doesNotMatch(mainSrc, /Homebrew: <code>\/opt\/homebrew\/bin<\/code>/);
  assert.doesNotMatch(mainSrc, /Copy engine\.json\.example/);
});

test('install.sh supports --shared-desktop-engine and activate.sh verifies shared-desktop-engine stages without bundled engine/', (t) => {
  const installSrc = fs.readFileSync(INSTALL_SH, 'utf8');
  assert.ok(installSrc.includes('--shared-desktop-engine'), 'install.sh must support --shared-desktop-engine');
  assert.ok(installSrc.includes('/Applications/HawaVoClean.app'), 'install.sh must reference /Applications/HawaVoClean.app');

  // Verify activate.sh --verify-only accepts a valid shared-desktop-engine stage without engine/ directory
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'wave-g-shared-stage-'));
  t.after(() => fs.rmSync(tmpDir, { recursive: true, force: true }));

  const stage = path.join(tmpDir, 'com.hawavoclean.resolve');
  fs.mkdirSync(stage, { recursive: true });
  const files = {
    PLUGIN_ID: 'com.hawavoclean.resolve\n',
    VERSION: '3.3.0\n',
    ENGINE_MODE: 'shared-desktop-engine\n',
    SYMLINKS: '',
    'manifest.xml': '<Plugin><Id>com.hawavoclean.resolve</Id></Plugin>\n',
    'package.json': '{"name":"com.hawavoclean.resolve","version":"3.3.0"}\n',
    'main.js': '// main\n',
    'preload.js': '// preload\n',
    'session-auth.js': '// auth\n',
    'engine-discovery.js': '// discovery\n',
    'sdk-preflight.js': '// preflight\n',
    'timeline-transactions.js': '// tx\n',
    'engine.json': '{"command":["/Applications/HawaVoClean.app/Contents/Resources/engine/hawavoclean-engine","serve"],"cwd":"."}\n',
    'index.html': '<!doctype html><html></html>\n',
  };
  for (const [name, content] of Object.entries(files)) {
    fs.writeFileSync(path.join(stage, name), content);
  }
  const sumLines = Object.keys(files)
    .sort()
    .map((name) => `${computeFileSha256(path.join(stage, name))}  ./${name}`)
    .join('\n') + '\n';
  fs.writeFileSync(path.join(stage, 'SHA256SUMS'), sumLines);

  const verified = execFileSync(ACTIVATE_SH, ['--stage', stage, '--verify-only'], { encoding: 'utf8' }).trim();
  assert.equal(verified, stage);
});
