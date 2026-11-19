import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { getBridge, resetBridgeCache, type HawaBridge } from './index';

describe('bridge Resolve forwarding', () => {
  const prevHawa = window.hawa;

  beforeEach(() => {
    resetBridgeCache();
    delete window.hawa;
  });

  afterEach(() => {
    resetBridgeCache();
    if (prevHawa === undefined) delete window.hawa;
    else window.hawa = prevHawa;
  });

  it('forwards batchCleanTrack and applyGuardMarkers when present on window.hawa.resolve', async () => {
    const batchCleanTrack = vi.fn(async () => ({
      success: true,
      trackIndex: 2,
      trackName: 'HawaVoClean',
      clipsPlaced: 1,
      markersAdded: 3,
    }));
    const applyGuardMarkers = vi.fn(async () => ({
      success: true,
      markersAdded: 2,
      markers: [],
    }));
    window.hawa = {
      host: 'resolve',
      engine: { getEndpoint: async () => ({ baseUrl: 'http://127.0.0.1:8765' }) },
      files: {
        pickAudio: async () => null,
        pathForFile: () => null,
        revealInFinder: async () => undefined,
      },
      resolve: {
        getSelectedClip: async () => null,
        importMedia: async () => null,
        replaceClip: async () => true,
        appendToTimeline: async () => true,
        getContext: async () => ({ project: 'P', timeline: 'T', page: 'fairlight' }),
        batchCleanTrack,
        applyGuardMarkers,
      },
    } satisfies HawaBridge;

    const bridge = getBridge();
    expect(typeof bridge.resolve?.batchCleanTrack).toBe('function');
    expect(typeof bridge.resolve?.applyGuardMarkers).toBe('function');

    const batchRes = await bridge.resolve!.batchCleanTrack!({
      clips: [{ clipId: 'c1', startFrame: 0 }],
      cleanedByClipId: { c1: '/out/c1.wav' },
      addGuardMarkers: true,
    });
    expect(batchCleanTrack).toHaveBeenCalledTimes(1);
    expect(batchRes.markersAdded).toBe(3);

    const markerRes = await bridge.resolve!.applyGuardMarkers!({
      reportPath: '/out/c1.hawavoclean.json',
      clipStartFrame: 0,
      fps: 24,
    });
    expect(applyGuardMarkers).toHaveBeenCalledTimes(1);
    expect(markerRes.markersAdded).toBe(2);
  });

  it('leaves batchCleanTrack and applyGuardMarkers undefined when absent from window.hawa.resolve', () => {
    window.hawa = {
      host: 'resolve',
      engine: { getEndpoint: async () => ({ baseUrl: 'http://127.0.0.1:8765' }) },
      files: {
        pickAudio: async () => null,
        pathForFile: () => null,
        revealInFinder: async () => undefined,
      },
      resolve: {
        getSelectedClip: async () => null,
        importMedia: async () => null,
        replaceClip: async () => true,
        appendToTimeline: async () => true,
        getContext: async () => ({ project: null, timeline: null, page: null }),
      },
    };

    const bridge = getBridge();
    expect(bridge.resolve?.batchCleanTrack).toBeUndefined();
    expect(bridge.resolve?.applyGuardMarkers).toBeUndefined();
  });
});
