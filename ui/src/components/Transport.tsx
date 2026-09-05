import { useEffect, useState } from 'react';
import { getPlayer } from '../audio/player';
import { formatTime } from '../render/ticks';
import { setAb, toggleLoudnessMatch, togglePlay } from '../state/actions';
import { useStore, type AbMode } from '../state/store';
import { IconPause, IconPlay, IconWarn } from './Icons';
import { Segmented } from './Segmented';

export function Transport() {
  const abMode = useStore((s) => s.abMode);
  const setAbMode = useStore((s) => s.setAbMode);
  const loudnessMatch = useStore((s) => s.loudnessMatch);
  const gainOffsetDb = useStore((s) => s.gainOffsetDb);
  const setGainOffsetDb = useStore((s) => s.setGainOffsetDb);
  const cleaned = useStore((s) => s.cleanedPath);
  const artifacts = useStore((s) => s.artifacts);
  const deckFault = useStore((s) => s.deckFault);
  const original = useStore((s) => s.original);
  const setPlaying = useStore((s) => s.setPlaying);
  const playing = useStore((s) => s.playing);
  const [clock, setClock] = useState({ t: '00:00.0', d: '00:00.0' });

  useEffect(() => {
    let lastText = '';
    let lastPlaying: boolean | null = null;
    let lastDeck: AbMode | null = null;
    let lastGain: number | null = null;
    const unsub = getPlayer().subscribe((snap) => {
      if (snap.playing !== lastPlaying) {
        lastPlaying = snap.playing;
        setPlaying(snap.playing);
      }
      // The switch renders the player, not its own memory of what was pressed.
      // `pending` is the deck a `setActive` is still waiting on (a fetch in
      // flight); `active` is the deck actually making sound. Reading the pair
      // is what makes it impossible for the control to claim CLEANED while the
      // ORIGINAL element is the one being heard — the exact lie that used to
      // survive a cleaned deck failing to load.
      const deck = snap.pending ?? snap.active;
      if (deck !== lastDeck) {
        lastDeck = deck;
        setAbMode(deck);
      }
      if (snap.gainOffsetDb !== undefined && snap.gainOffsetDb !== lastGain) {
        lastGain = snap.gainOffsetDb;
        setGainOffsetDb(snap.gainOffsetDb);
      }
      // Tenth-of-a-second resolution keeps React updates at ~10 Hz, not 60.
      const t = formatTime(Math.floor(snap.time * 10) / 10, false) + '.' + Math.floor((snap.time * 10) % 10);
      const d = formatTime(Math.floor(snap.duration * 10) / 10, false) + '.' + Math.floor((snap.duration * 10) % 10);
      const text = `${t}|${d}`;
      if (text !== lastText) {
        lastText = text;
        setClock({ t, d });
      }
    });
    return unsub;
  }, [setPlaying, setAbMode, setGainOffsetDb]);

  const hasAudio = Boolean(original);
  // A path is not a deck. The master can be gone — deleted, moved, or written
  // over by a later run — while the report that names it is still on screen,
  // and in that case there is nothing to switch to.
  const masterServed =
    Boolean(cleaned) && artifacts?.master !== false && deckFault?.deck !== 'cleaned';
  // Key bindings live in one map (App.useKeyboardMap) so Space/A/B cannot be
  // claimed twice; this component only draws the transport.

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
      >
        <span className="lm-label">Level Match</span>
        {masterServed && Number.isFinite(gainOffsetDb) && Math.abs(gainOffsetDb) > 0.05 ? (
          <span className="lm-badge mono">
            {gainOffsetDb > 0 ? `+${gainOffsetDb.toFixed(1)}` : gainOffsetDb.toFixed(1)} dB
          </span>
        ) : null}
      </button>
      <span className="timecode">
        {clock.t}
        <span className="dim dur"> / {clock.d}</span>
      </span>
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
