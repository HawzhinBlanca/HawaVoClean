import { useEffect, useId, useRef, useState } from 'react';
import '../styles/studio-rack.css';

/** Maximum tolerated audio/video drift in seconds (40 ms). */
export const VIDEO_SYNC_DRIFT_THRESHOLD_S = 0.04;

const VIDEO_EXTENSION_RE = /\.(mp4|mov|mkv)(?:[?#].*)?$/i;

/** Returns true when a path or URL references a supported video container (.mp4/.mov/.mkv). */
export function isVideoFilePath(pathOrUrl: string | null | undefined): boolean {
  if (!pathOrUrl) return false;
  return VIDEO_EXTENSION_RE.test(pathOrUrl.trim());
}

export interface VideoPreviewProps {
  /** URL of the video source, or null/undefined/empty when audio-only. */
  videoUrl: string | null | undefined;
  /** Master playback time in seconds from DualPlayer. */
  currentTime: number;
  /** Whether DualPlayer is currently playing. */
  playing: boolean;
  /** Called when the user scrubs or seeks from the PiP video preview. */
  onSeek?: ((timeSeconds: number) => void) | undefined;
  /** Whether the PiP panel is collapsed to a compact toggle pill. */
  collapsed: boolean;
  /** Toggles between expanded PiP and collapsed pill. */
  onToggleCollapse: () => void;
  /** Optional clip title displayed in the PiP header. */
  title?: string | undefined;
  /** Optional clip duration in seconds for the PiP scrub bar. */
  duration?: number | undefined;
  className?: string | undefined;
}

function formatTimecode(seconds: number): string {
  const safe = Number.isFinite(seconds) && seconds > 0 ? seconds : 0;
  const mins = Math.floor(safe / 60);
  const secs = Math.floor(safe % 60);
  const tenths = Math.floor((safe % 1) * 10);
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}.${tenths}`;
}

export function VideoPreview({
  videoUrl,
  currentTime,
  playing,
  onSeek,
  collapsed,
  onToggleCollapse,
  title = 'Video Preview',
  duration,
  className,
}: VideoPreviewProps) {
  const uid = useId();
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const programmaticSeekRef = useRef(false);
  const [mediaDuration, setMediaDuration] = useState<number>(0);
  const [hasError, setHasError] = useState(false);

  const normalizedUrl = videoUrl?.trim() ?? '';
  const safeTime = Number.isFinite(currentTime) && currentTime >= 0 ? currentTime : 0;
  const effectiveDuration =
    duration !== undefined && Number.isFinite(duration) && duration > 0 ? duration : mediaDuration;

  // Reset media error state whenever the source URL changes.
  useEffect(() => {
    setHasError(false);
    setMediaDuration(0);
  }, [normalizedUrl]);

  // Keep <video> frame-synced within 40 ms of DualPlayer's playback clock.
  useEffect(() => {
    const video = videoRef.current;
    if (!video || collapsed || !normalizedUrl || hasError) return;

    video.muted = true;
    const targetTime =
      effectiveDuration > 0 ? Math.min(safeTime, effectiveDuration) : safeTime;
    const drift = Math.abs(video.currentTime - targetTime);

    if (drift > VIDEO_SYNC_DRIFT_THRESHOLD_S) {
      programmaticSeekRef.current = true;
      try {
        video.currentTime = targetTime;
      } catch {
        // Ignore seek errors before metadata is ready; onLoadedMetadata will resync.
      }
      queueMicrotask(() => {
        programmaticSeekRef.current = false;
      });
    }
  }, [safeTime, collapsed, normalizedUrl, hasError, effectiveDuration]);

  // Mirror DualPlayer play/pause state on the muted <video> element.
  useEffect(() => {
    const video = videoRef.current;
    if (!video || collapsed || !normalizedUrl || hasError) return;

    video.muted = true;
    if (playing) {
      if (video.paused) {
        const playResult = video.play();
        if (playResult && typeof playResult.catch === 'function') {
          playResult.catch(() => {
            // Autoplay or abort rejections in headless/background tabs are non-fatal.
          });
        }
      }
    } else if (!video.paused) {
      video.pause();
    }
  }, [playing, collapsed, normalizedUrl, hasError]);

  // Degrade cleanly to nothing for audio-only sources.
  if (!normalizedUrl) {
    return null;
  }

  if (collapsed) {
    return (
      <button
        type="button"
        className={`video-preview-pill${className ? ` ${className}` : ''}`}
        aria-expanded={false}
        aria-label={`Expand ${title}`}
        onClick={onToggleCollapse}
      >
        <span aria-hidden="true">▣</span>
        <span>{title}</span>
        <span className="vp-time">{formatTimecode(safeTime)}</span>
      </button>
    );
  }

  return (
    <aside
      className={`video-preview${className ? ` ${className}` : ''}`}
      aria-label={title}
    >
      <div className="vp-header">
        <div className="vp-title-wrap">
          <span className="vp-title">{title}</span>
          <span className="vp-time" data-testid="video-preview-time">
            {formatTimecode(safeTime)}
          </span>
        </div>
        <button
          type="button"
          className="vp-collapse-btn"
          aria-expanded={true}
          aria-label={`Collapse ${title}`}
          title="Collapse picture-in-picture preview"
          onClick={onToggleCollapse}
        >
          <span aria-hidden="true">−</span>
        </button>
      </div>

      <div className="vp-stage">
        {hasError ? (
          <p className="vp-error" role="status">
            Video preview unavailable for this stream — audio playback continues normally.
          </p>
        ) : (
          <video
            ref={videoRef}
            className="vp-video"
            src={normalizedUrl}
            muted
            playsInline
            preload="metadata"
            aria-label={`${title} video frame`}
            onLoadedMetadata={(e) => {
              const el = e.currentTarget;
              if (Number.isFinite(el.duration) && el.duration > 0) {
                setMediaDuration(el.duration);
              }
              if (Math.abs(el.currentTime - safeTime) > VIDEO_SYNC_DRIFT_THRESHOLD_S) {
                programmaticSeekRef.current = true;
                try {
                  el.currentTime = safeTime;
                } catch {
                  // Ignore if not seekable yet.
                }
                queueMicrotask(() => {
                  programmaticSeekRef.current = false;
                });
              }
            }}
            onSeeked={(e) => {
              if (programmaticSeekRef.current || !onSeek) return;
              const nextTime = e.currentTarget.currentTime;
              if (
                Number.isFinite(nextTime) &&
                Math.abs(nextTime - safeTime) > VIDEO_SYNC_DRIFT_THRESHOLD_S
              ) {
                onSeek(nextTime);
              }
            }}
            onError={() => setHasError(true)}
          />
        )}
      </div>

      <div className="vp-footer">
        {onSeek && effectiveDuration > 0 ? (
          <>
            <label htmlFor={`${uid}-scrub`} className="sr-only">
              Seek video preview
            </label>
            <input
              id={`${uid}-scrub`}
              type="range"
              className="vp-scrub"
              min={0}
              max={effectiveDuration}
              step={0.04}
              value={Math.min(safeTime, effectiveDuration)}
              aria-valuetext={formatTimecode(safeTime)}
              onChange={(e) => {
                const next = Number(e.target.value);
                if (Number.isFinite(next)) {
                  onSeek(Math.min(effectiveDuration, Math.max(0, next)));
                }
              }}
            />
          </>
        ) : null}
        <span className="vp-sync-badge">SYNC ≤40ms</span>
      </div>
    </aside>
  );
}
