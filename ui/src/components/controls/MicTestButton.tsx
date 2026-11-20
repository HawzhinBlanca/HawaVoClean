import { useCallback, useEffect, useRef, useState } from 'react';
import { ingestFile } from '../../state/actions';
import { useI18n } from '../../state/i18n';
import { jobInFlight, useStore } from '../../state/store';

const RECORD_DURATION_S = 5;

/** Encode mono Float32 PCM samples [-1, 1] into a valid 16-bit RIFF WAVE buffer. */
export function encodeWavPcm16(samples: Float32Array, sampleRate: number): ArrayBuffer {
  const safeRate = Math.max(8000, Math.min(48000, Math.round(sampleRate || 48000)));
  const numSamples = samples.length;
  const dataBytes = numSamples * 2;
  const buffer = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buffer);

  const writeStr = (offset: number, str: string): void => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + dataBytes, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true); // PCM fmt chunk size
  view.setUint16(20, 1, true); // PCM format
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, safeRate, true);
  view.setUint32(28, safeRate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeStr(36, 'data');
  view.setUint32(40, dataBytes, true);

  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    const raw = samples[i] ?? 0;
    const clamped = Math.max(-1, Math.min(1, Number.isFinite(raw) ? raw : 0));
    view.setInt16(offset, clamped < 0 ? Math.round(clamped * 0x8000) : Math.round(clamped * 0x7fff), true);
    offset += 2;
  }

  return buffer;
}

function concatChunks(chunks: readonly Float32Array[]): Float32Array {
  let total = 0;
  for (const c of chunks) total += c.length;
  const out = new Float32Array(total);
  let pos = 0;
  for (const c of chunks) {
    out.set(c, pos);
    pos += c.length;
  }
  return out;
}

export interface MicTestButtonProps {
  /** Optional override for ingesting the recorded WAV file (defaults to `ingestFile`). */
  onRecorded?: (file: File) => void | Promise<void>;
}

/**
 * 1-Click "Test with Mic (5s)" button for the Simple Studio welcome card.
 * Captures up to 5 seconds of live microphone PCM, encodes a 16-bit WAV file
 * client-side, and hands it straight to `ingestFile()` for instant testing.
 */
export function MicTestButton({ onRecorded }: MicTestButtonProps = {}) {
  const { t } = useI18n();
  const engineReady = useStore((s) => s.engineStatus === 'ready');
  const busy = useStore((s) => jobInFlight(s.job) || s.analyzing || Boolean(s.upload));
  const pushToast = useStore((s) => s.pushToast);

  const [recording, setRecording] = useState(false);
  const [remainingS, setRemainingS] = useState(RECORD_DURATION_S);
  const [notice, setNotice] = useState<string | null>(null);

  const chunksRef = useRef<Float32Array[]>([]);
  const sampleRateRef = useRef(48000);
  const streamRef = useRef<MediaStream | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const procRef = useRef<ScriptProcessorNode | null>(null);
  const stopTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const tickTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const activeRef = useRef(false);

  const cleanupMedia = useCallback(() => {
    if (stopTimerRef.current !== null) {
      clearTimeout(stopTimerRef.current);
      stopTimerRef.current = null;
    }
    if (tickTimerRef.current !== null) {
      clearInterval(tickTimerRef.current);
      tickTimerRef.current = null;
    }
    if (procRef.current) {
      procRef.current.onaudioprocess = null;
      try {
        procRef.current.disconnect();
      } catch {
        // ignore
      }
      procRef.current = null;
    }
    if (streamRef.current) {
      for (const track of streamRef.current.getTracks()) {
        try {
          track.stop();
        } catch {
          // ignore
        }
      }
      streamRef.current = null;
    }
    if (ctxRef.current) {
      const c = ctxRef.current;
      ctxRef.current = null;
      if (c.state !== 'closed') {
        void c.close().catch(() => undefined);
      }
    }
  }, []);

  const finishRecording = useCallback(async () => {
    if (!activeRef.current) return;
    activeRef.current = false;
    setRecording(false);
    setRemainingS(RECORD_DURATION_S);
    const sr = sampleRateRef.current;
    const chunks = chunksRef.current;
    chunksRef.current = [];
    cleanupMedia();

    let pcm = concatChunks(chunks);
    if (pcm.length === 0) {
      // Ensure non-empty valid 0.5s silent WAV fallback if browser emitted no frames before early stop.
      pcm = new Float32Array(Math.round(sr * 0.5));
    }
    const wavBuf = encodeWavPcm16(pcm, sr);
    const file = new File([wavBuf], 'mic-test-5s.wav', { type: 'audio/wav' });
    if (onRecorded) {
      await onRecorded(file);
    } else {
      await ingestFile(file);
    }
  }, [cleanupMedia, onRecorded]);

  useEffect(() => {
    return () => {
      activeRef.current = false;
      cleanupMedia();
    };
  }, [cleanupMedia]);

  const handleClick = async (): Promise<void> => {
    if (recording) {
      void finishRecording();
      return;
    }
    if (!engineReady || busy) return;
    setNotice(null);

    const mediaDevices = typeof navigator !== 'undefined' ? navigator.mediaDevices : undefined;
    if (!mediaDevices || typeof mediaDevices.getUserMedia !== 'function') {
      setNotice(t.micUnavailable);
      pushToast(t.micUnavailable, 'err');
      return;
    }

    let stream: MediaStream;
    try {
      stream = await mediaDevices.getUserMedia({ audio: true });
    } catch (e) {
      const name = e instanceof Error ? e.name : '';
      const msg =
        name === 'NotAllowedError' || name === 'PermissionDeniedError'
          ? t.micDenied
          : t.micUnavailable;
      setNotice(msg);
      pushToast(msg, 'err');
      return;
    }

    const AudioCtx = typeof window !== 'undefined' ? window.AudioContext : undefined;
    if (!AudioCtx) {
      for (const track of stream.getTracks()) track.stop();
      setNotice(t.micUnavailable);
      pushToast(t.micUnavailable, 'err');
      return;
    }

    try {
      const ctx = new AudioCtx();
      sampleRateRef.current = ctx.sampleRate || 48000;
      chunksRef.current = [];
      streamRef.current = stream;
      ctxRef.current = ctx;

      const source = ctx.createMediaStreamSource(stream);
      const silentGain = ctx.createGain();
      silentGain.gain.value = 0;

      if (typeof ctx.createScriptProcessor === 'function') {
        const proc = ctx.createScriptProcessor(4096, 1, 1);
        proc.onaudioprocess = (ev: AudioProcessingEvent) => {
          if (!activeRef.current) return;
          const ch = ev.inputBuffer.getChannelData(0);
          if (ch && ch.length > 0) {
            chunksRef.current.push(new Float32Array(ch));
          }
        };
        source.connect(proc);
        proc.connect(silentGain);
        silentGain.connect(ctx.destination);
        procRef.current = proc;
      }

      activeRef.current = true;
      setRecording(true);
      setRemainingS(RECORD_DURATION_S);

      const startedAt = Date.now();
      tickTimerRef.current = setInterval(() => {
        const elapsed = Math.floor((Date.now() - startedAt) / 1000);
        setRemainingS(Math.max(1, RECORD_DURATION_S - elapsed));
      }, 250);

      stopTimerRef.current = setTimeout(() => {
        void finishRecording();
      }, RECORD_DURATION_S * 1000);
    } catch {
      cleanupMedia();
      setNotice(t.micUnavailable);
      pushToast(t.micUnavailable, 'err');
    }
  };

  return (
    <div className="mic-test-wrap">
      <button
        type="button"
        className={`sw-mic-btn${recording ? ' recording' : ''}`}
        disabled={(!engineReady || busy) && !recording}
        aria-pressed={recording}
        onClick={() => void handleClick()}
        title={recording ? t.micStopEarly : t.micTestBtn}
      >
        <span className="sw-mic-dot" aria-hidden="true" />
        <span>
          {recording ? `${t.micRecordingPrefix} ${remainingS}s` : t.micTestBtn}
        </span>
      </button>
      {notice ? (
        <span className="sw-mic-notice" role="status">
          {notice}
        </span>
      ) : null}
    </div>
  );
}
