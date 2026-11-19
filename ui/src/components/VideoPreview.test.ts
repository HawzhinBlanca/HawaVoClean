import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import axe from 'axe-core';
import {
  VIDEO_SYNC_DRIFT_THRESHOLD_S,
  VideoPreview,
  isVideoFilePath,
} from './VideoPreview';

declare global {
  var IS_REACT_ACT_ENVIRONMENT: boolean | undefined;
}
globalThis.IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLElement;
let root: Root;

beforeEach(() => {
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(async () => {
    root.unmount();
  });
  host.remove();
});

async function runAxe(element: HTMLElement) {
  const results = await axe.run(element, {
    runOnly: {
      type: 'tag',
      values: ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'best-practice'],
    },
    rules: {
      'color-contrast': { enabled: false },
      region: { enabled: false },
    },
  });
  return results.violations;
}

describe('VideoPreview', () => {
  it('detects supported video container extensions (.mp4, .mov, .mkv)', () => {
    expect(isVideoFilePath('/clips/interview.mp4')).toBe(true);
    expect(isVideoFilePath('/clips/take01.MOV')).toBe(true);
    expect(isVideoFilePath('http://127.0.0.1:8765/media/clip.mkv?token=abc')).toBe(true);
    expect(isVideoFilePath('/clips/voice.wav')).toBe(false);
    expect(isVideoFilePath('/clips/voice.flac')).toBe(false);
    expect(isVideoFilePath(null)).toBe(false);
    expect(isVideoFilePath('')).toBe(false);
  });

  it('degrades cleanly to null when audio-only (videoUrl is null or empty)', async () => {
    await act(async () => {
      root.render(
        createElement(VideoPreview, {
          videoUrl: null,
          currentTime: 12.5,
          playing: true,
          collapsed: false,
          onToggleCollapse: vi.fn(),
        }),
      );
    });
    expect(host.innerHTML).toBe('');
  });

  it('supports click-to-collapse PiP toggle between compact pill and expanded video panel', async () => {
    const onToggleCollapse = vi.fn();

    await act(async () => {
      root.render(
        createElement(VideoPreview, {
          videoUrl: 'http://127.0.0.1:8765/api/file?path=/clip.mp4',
          currentTime: 65.4,
          playing: false,
          collapsed: true,
          onToggleCollapse,
        }),
      );
    });

    const expandBtn = host.querySelector<HTMLButtonElement>('button.video-preview-pill');
    expect(expandBtn).not.toBeNull();
    expect(expandBtn?.getAttribute('aria-expanded')).toBe('false');
    expect(expandBtn?.textContent).toContain('01:05.4');
    expect(host.querySelector('video')).toBeNull();

    await act(async () => {
      expandBtn!.click();
    });
    expect(onToggleCollapse).toHaveBeenCalledTimes(1);

    await act(async () => {
      root.render(
        createElement(VideoPreview, {
          videoUrl: 'http://127.0.0.1:8765/api/file?path=/clip.mp4',
          currentTime: 65.4,
          playing: false,
          collapsed: false,
          onToggleCollapse,
          duration: 120,
          onSeek: vi.fn(),
        }),
      );
    });

    const video = host.querySelector<HTMLVideoElement>('video');
    expect(video).not.toBeNull();
    expect(video?.muted).toBe(true);
    expect(video?.hasAttribute('playsinline')).toBe(true);

    const collapseBtn = host.querySelector<HTMLButtonElement>('button.vp-collapse-btn');
    expect(collapseBtn?.getAttribute('aria-expanded')).toBe('true');
    await act(async () => {
      collapseBtn!.click();
    });
    expect(onToggleCollapse).toHaveBeenCalledTimes(2);
  });

  it('keeps <video> frame-synced within 40 ms of DualPlayer currentTime and mirrors play/pause', async () => {
    const playSpy = vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined);
    const pauseSpy = vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});

    try {
      await act(async () => {
        root.render(
          createElement(VideoPreview, {
            videoUrl: 'http://127.0.0.1:8765/api/file?path=/clip.mov',
            currentTime: 10.0,
            playing: true,
            collapsed: false,
            onToggleCollapse: vi.fn(),
            duration: 120,
          }),
        );
      });

      const video = host.querySelector<HTMLVideoElement>('video')!;
      expect(video.muted).toBe(true);
      expect(Math.abs(video.currentTime - 10.0)).toBeLessThanOrEqual(VIDEO_SYNC_DRIFT_THRESHOLD_S);
      expect(playSpy).toHaveBeenCalled();

      // Drift <= 40ms should NOT force a redundant seek
      const timeSetterSpy = vi.spyOn(video, 'currentTime', 'set');
      await act(async () => {
        root.render(
          createElement(VideoPreview, {
            videoUrl: 'http://127.0.0.1:8765/api/file?path=/clip.mov',
            currentTime: 10.02,
            playing: true,
            collapsed: false,
            onToggleCollapse: vi.fn(),
            duration: 120,
          }),
        );
      });
      expect(timeSetterSpy).not.toHaveBeenCalled();

      // Drift > 40ms MUST resync video.currentTime and pause when playing transitions true -> false
      Object.defineProperty(video, 'paused', { value: false, configurable: true });
      await act(async () => {
        root.render(
          createElement(VideoPreview, {
            videoUrl: 'http://127.0.0.1:8765/api/file?path=/clip.mov',
            currentTime: 10.25,
            playing: false,
            collapsed: false,
            onToggleCollapse: vi.fn(),
            duration: 120,
          }),
        );
      });
      expect(timeSetterSpy).toHaveBeenCalledWith(10.25);
      expect(pauseSpy).toHaveBeenCalled();
    } finally {
      playSpy.mockRestore();
      pauseSpy.mockRestore();
    }
  });

  it('forwards scrub slider seeks to onSeek and degrades gracefully on video media errors', async () => {
    const onSeek = vi.fn();
    await act(async () => {
      root.render(
        createElement(VideoPreview, {
          videoUrl: 'http://127.0.0.1:8765/api/file?path=/clip.mkv',
          currentTime: 5,
          playing: false,
          onSeek,
          collapsed: false,
          onToggleCollapse: vi.fn(),
          duration: 60,
        }),
      );
    });

    const scrub = host.querySelector<HTMLInputElement>('input.vp-scrub')!;
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
    await act(async () => {
      setter?.call(scrub, '24.5');
      scrub.dispatchEvent(new Event('input', { bubbles: true }));
      scrub.dispatchEvent(new Event('change', { bubbles: true }));
    });
    expect(onSeek).toHaveBeenCalledWith(24.5);

    // Media decode error falls back cleanly without breaking the panel
    const video = host.querySelector<HTMLVideoElement>('video')!;
    await act(async () => {
      video.dispatchEvent(new Event('error'));
    });
    expect(host.querySelector('video')).toBeNull();
    expect(host.querySelector('.vp-error')?.textContent).toContain('audio playback continues');
  });

  it('ignores delayed async seeked events from programmatic sync seeks without feeding back to onSeek', async () => {
    const onSeek = vi.fn();
    const playSpy = vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined);
    try {
      await act(async () => {
        root.render(
          createElement(VideoPreview, {
            videoUrl: 'http://127.0.0.1:8765/api/file?path=/clip.mp4',
            currentTime: 10.0,
            playing: true,
            onSeek,
            collapsed: false,
            onToggleCollapse: vi.fn(),
            duration: 60,
          }),
        );
      });

      const video = host.querySelector<HTMLVideoElement>('video')!;
      expect(video.currentTime).toBeCloseTo(10.0, 2);

      // Flush microtasks & advance DualPlayer clock by >40ms before async `seeked` arrives
      await Promise.resolve();
      await Promise.resolve();
      await act(async () => {
        root.render(
          createElement(VideoPreview, {
            videoUrl: 'http://127.0.0.1:8765/api/file?path=/clip.mp4',
            currentTime: 10.12,
            playing: true,
            onSeek,
            collapsed: false,
            onToggleCollapse: vi.fn(),
            duration: 60,
          }),
        );
      });

      // Simulate browser firing `seeked` for the settled programmatic seek at 10.12s
      // (even after safeTime has moved on by >40ms) -> must NOT fire onSeek loop
      video.currentTime = 10.12;
      await act(async () => {
        video.dispatchEvent(new Event('seeked'));
      });
      expect(onSeek).not.toHaveBeenCalled();

      // Genuine user-initiated seek after programmatic seek settled MUST still call onSeek
      video.currentTime = 24.5;
      await act(async () => {
        video.dispatchEvent(new Event('seeked'));
      });
      expect(onSeek).toHaveBeenCalledTimes(1);
      expect(onSeek).toHaveBeenCalledWith(24.5);
    } finally {
      playSpy.mockRestore();
    }
  });

  it('passes axe-core WCAG 2.2 AA accessibility checks in collapsed and expanded states', async () => {
    await act(async () => {
      root.render(
        createElement(VideoPreview, {
          videoUrl: 'http://127.0.0.1:8765/api/file?path=/clip.mp4',
          currentTime: 12,
          playing: false,
          onSeek: vi.fn(),
          collapsed: true,
          onToggleCollapse: vi.fn(),
          duration: 90,
        }),
      );
    });
    expect(await runAxe(host)).toEqual([]);

    await act(async () => {
      root.render(
        createElement(VideoPreview, {
          videoUrl: 'http://127.0.0.1:8765/api/file?path=/clip.mp4',
          currentTime: 12,
          playing: false,
          onSeek: vi.fn(),
          collapsed: false,
          onToggleCollapse: vi.fn(),
          duration: 90,
        }),
      );
    });
    expect(await runAxe(host)).toEqual([]);
  });
});
