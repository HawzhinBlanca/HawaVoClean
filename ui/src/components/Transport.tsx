import { useEffect, useRef, useState } from 'react';
import { getPlayer } from '../audio/player';
import { formatTime } from '../render/ticks';
import { setAb, toggleLoudnessMatch, togglePlay } from '../state/actions';
import { useI18n } from '../state/i18n';
import { useStore, type AbMode } from '../state/store';
import { IconPause, IconPlay, IconWarn } from './Icons';
import { LiveMeter } from './LiveMeter';
import { Segmented } from './Segmented';

export function Transport() {
  const { t } = useI18n();
  const abMode = useStore((s) => s.abMode);
  const loudnessMatch = useStore((s) => s.loudnessMatch);
  const gainOffsetDb = useStore((s) => s.gainOffsetDb);
  const cleaned = useStore((s) => s.cleanedPath);
  const previewActive = useStore((s) => s.previewActive);
  const artifacts = useStore((s) => s.artifacts);
  const deckFault = useStore((s) => s.deckFault);
  const original = useStore((s) => s.original);
  const playing = useStore((s) => s.playing);
  const [clock, setClock] = useState({ t: '00:00.0', d: '00:00.0' });
  const [deltaMode, setDeltaMode] = useState(false);
  const [loopEnabled, setLoopEnabled] = useState(false);
  const [wetDryMix, setWetDryMix] = useState(1);
  const holdingOrigRef = useRef(false);
  const prevDeckRef = useRef<AbMode>('cleaned');

  useEffect(() => {
    let lastText = '';
    let lastDelta: boolean | null = null;
    let lastLoop: boolean | null = null;
    let lastMix: number | null = null;
    const unsub = getPlayer().subscribe((snap) => {
      const nextDelta = snap.deltaMode ?? false;
      if (nextDelta !== lastDelta) {
        lastDelta = nextDelta;
        setDeltaMode(nextDelta);
      }
      const nextLoop = snap.loopEnabled ?? false;
      if (nextLoop !== lastLoop) {
        lastLoop = nextLoop;
        setLoopEnabled(nextLoop);
      }
      const nextMix = snap.wetDryMix ?? 1;
      if (nextMix !== lastMix) {
        lastMix = nextMix;
        setWetDryMix(nextMix);
      }
      // Tenth-of-a-second resolution keeps React updates at ~10 Hz, not 60.
      const tStr =
        formatTime(Math.floor(snap.time * 10) / 10, false) + '.' + Math.floor((snap.time * 10) % 10);
      const dStr =
        formatTime(Math.floor(snap.duration * 10) / 10, false) + '.' + Math.floor((snap.duration * 10) % 10);
      const text = `${tStr}|${dStr}`;
      if (text !== lastText) {
        lastText = text;
        setClock({ t: tStr, d: dStr });
      }
    });
    return unsub;
  }, []);

  const hasAudio = Boolean(original);
  // A path is not a deck. The master can be gone — deleted, moved, or written
  // over by a later run — while the report that names it is still on screen,
  // and in that case there is nothing to switch to.
  const masterServed =
    ((Boolean(cleaned) && artifacts?.master !== false) || previewActive) &&
    deckFault?.deck !== 'cleaned';

  const startHoldOrig = (): void => {
    if (!hasAudio || !masterServed || holdingOrigRef.current) return;
    holdingOrigRef.current = true;
    prevDeckRef.current = abMode;
    setAb('original');
  };

  const endHoldOrig = (): void => {
    if (!holdingOrigRef.current) return;
    holdingOrigRef.current = false;
    if (prevDeckRef.current === 'cleaned' && masterServed) {
      setAb('cleaned');
    }
  };
  // Key bindings live in one map (App.useKeyboardMap) so Space/A/B/D/L cannot be
  // claimed twice; this component only draws the transport.
  const mixPct = Math.round(wetDryMix * 100);

  return (
    <div className="transport">
      <button
        type="button"
        className={`playbtn${playing ? ' on' : ''}`}
        disabled={!hasAudio}
        onClick={togglePlay}
        aria-label={playing ? 'Pause' : 'Play'}
        title={playing ? 'Pause (space)' : 'Play (space)'}
      >
        {playing ? <IconPause size={16} /> : <IconPlay size={16} />}
      </button>
      <Segmented<AbMode>
        ariaLabel="A/B source"
        className="ab"
        thumbClassName={abMode === 'cleaned' ? 'clean' : 'orig'}
        value={abMode}
        onChange={setAb}
        options={[
          { value: 'original', label: 'Original', className: 'orig', disabled: !hasAudio },
          { value: 'cleaned', label: 'Cleaned', className: 'clean', disabled: !masterServed },
        ]}
      />
      <span className="timecode">
        {clock.t}
        <span className="dim dur"> / {clock.d}</span>
      </span>

      <div
        className="transport-pro"
        style={{
          gridColumn: '1 / -1',
          display: 'flex',
          flexWrap: 'wrap',
          alignItems: 'center',
          gap: '6px',
        }}
      >
        <button
          type="button"
          className={`btn-level-match${loudnessMatch && masterServed ? ' on' : ''}`}
          disabled={!hasAudio || !masterServed}
          onClick={toggleLoudnessMatch}
          aria-pressed={loudnessMatch}
          aria-label="Level match A/B loudness"
          title={
            hasAudio && masterServed
              ? `Level match A/B: ${loudnessMatch ? 'ON' : 'OFF'} (${gainOffsetDb >= 0 ? '+' : ''}${gainOffsetDb.toFixed(1)} dB on Cleaned) — eliminates loudness bias`
              : 'Level match requires both Original and Cleaned audio'
          }
          style={{ minHeight: '28px', height: '28px', padding: '0 8px' }}
        >
          <span className="lm-label">Level Match</span>
          {masterServed && Number.isFinite(gainOffsetDb) && Math.abs(gainOffsetDb) > 0.05 ? (
            <span className="lm-badge mono" aria-hidden="true">
              {gainOffsetDb > 0 ? `+${gainOffsetDb.toFixed(1)}` : gainOffsetDb.toFixed(1)} dB
            </span>
          ) : null}
        </button>
        <button
          type="button"
          className="btn-level-match btn-hold-orig"
          disabled={!hasAudio || !masterServed}
          onPointerDown={(e) => {
            if (e.button === 0) startHoldOrig();
          }}
          onPointerUp={endHoldOrig}
          onPointerCancel={endHoldOrig}
          onPointerLeave={endHoldOrig}
          onKeyDown={(e) => {
            if ((e.key === ' ' || e.key === 'Enter') && !e.repeat) {
              e.preventDefault();
              startHoldOrig();
            }
          }}
          onKeyUp={(e) => {
            if (e.key === ' ' || e.key === 'Enter') {
              e.preventDefault();
              endHoldOrig();
            }
          }}
          onBlur={endHoldOrig}
          aria-label={t.holdForOriginal}
          title={t.holdForOriginalTitle}
          style={{ minHeight: '28px', height: '28px', padding: '0 8px' }}
        >
          <span>{t.holdOrigShort}</span>
        </button>

        <button
          type="button"
          className={`btn-level-match btn-delta${deltaMode && masterServed ? ' on' : ''}`}
          disabled={!hasAudio || !masterServed}
          onClick={() => getPlayer().toggleDeltaMode()}
          aria-pressed={deltaMode && masterServed}
          aria-label="Removed — listen to removed audio only"
          title="Listen to removed audio only (D)"
          style={{ minHeight: '28px', height: '28px', padding: '0 8px' }}
        >
          <span aria-hidden="true">Δ</span>
          <span>Removed</span>
        </button>

        <button
          type="button"
          className={`btn-level-match btn-loop${loopEnabled ? ' on' : ''}`}
          disabled={!hasAudio}
          onClick={() => getPlayer().toggleLoop()}
          aria-pressed={loopEnabled}
          aria-label="Loop selection or clip"
          title="Loop selection or clip (L)"
          style={{ minHeight: '28px', height: '28px', padding: '0 8px' }}
        >
          <span>Loop</span>
        </button>

        {masterServed ? (
          <div
            className="transport-mix"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '6px',
              padding: '0 8px',
              minHeight: '28px',
              borderRadius: 'var(--r-sm, 4px)',
              background: 'var(--panel-2, #1a1e24)',
              border: '1px solid var(--hair, rgba(255, 255, 255, 0.07))',
              fontSize: '11px',
              color: 'var(--fg-3, #949daa)',
            }}
          >
            <span
              className="mix-label"
              aria-hidden="true"
              style={{
                textTransform: 'uppercase',
                letterSpacing: 'var(--ls-caps, 0.08em)',
                fontWeight: 500,
              }}
            >
              Mix
            </span>
            <input
              type="range"
              className="mix-slider"
              min={0}
              max={100}
              step={1}
              value={mixPct}
              disabled={!hasAudio || !masterServed}
              onChange={(e) => getPlayer().setWetDryMix(Number(e.target.value) / 100)}
              aria-label="Wet dry mix"
              aria-valuetext={`${mixPct}% cleaned`}
              title={`Wet/Dry mix: ${mixPct}% Cleaned / ${100 - mixPct}% Original`}
              style={{
                width: '74px',
                height: '24px',
                margin: 0,
                accentColor: 'var(--cyan, #39d0ff)',
                cursor: hasAudio && masterServed ? 'pointer' : 'not-allowed',
              }}
            />
            <span
              className="mix-pct mono"
              aria-hidden="true"
              style={{
                minWidth: '32px',
                textAlign: 'right',
                fontFamily: 'var(--font-mono, ui-monospace, monospace)',
                fontVariantNumeric: 'tabular-nums',
                color: 'var(--fg-2, #aab2bf)',
              }}
            >
              {mixPct}%
            </span>
          </div>
        ) : null}

        <LiveMeter playing={playing} />
      </div>

      {/* A deck that could not be played is stated here, under the switch it
          is about, and it stays until a deck is asked for again. It is not the
          red error bar: nothing has just gone wrong with the engine — a file
          the run left behind is not there, which is a standing fact about what
          this screen can offer. The wording is announced once through the
          app's single polite live region (App.useJobAnnouncer). */}
      {deckFault ? (
        <p className="deckfault">
          <IconWarn size={12} />
          <span>
            <b>{deckFault.headline}</b>
            {deckFault.detail}
          </span>
        </p>
      ) : null}
    </div>
  );
}
