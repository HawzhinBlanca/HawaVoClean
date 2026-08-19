import { useEffect, useState } from 'react';
import { getPlayer } from '../audio/player';
import { formatTime } from '../render/ticks';
import { setAb, togglePlay } from '../state/actions';
import { useStore, type AbMode } from '../state/store';
import { IconPause, IconPlay } from './Icons';
import { Segmented } from './Segmented';

export function Transport() {
  const abMode = useStore((s) => s.abMode);
  const cleaned = useStore((s) => s.cleanedPath);
  const original = useStore((s) => s.original);
  const setPlaying = useStore((s) => s.setPlaying);
  const playing = useStore((s) => s.playing);
  const [clock, setClock] = useState({ t: '00:00.0', d: '00:00.0' });

  useEffect(() => {
    let lastText = '';
    let lastPlaying: boolean | null = null;
    const unsub = getPlayer().subscribe((snap) => {
      if (snap.playing !== lastPlaying) {
        lastPlaying = snap.playing;
        setPlaying(snap.playing);
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
  }, [setPlaying]);

  const hasAudio = Boolean(original);

  useEffect(() => {
    const onKey = (e: KeyboardEvent): void => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;
      if (e.code === 'Space') {
        e.preventDefault();
        if (hasAudio) togglePlay();
      } else if (e.key === 'a' || e.key === 'A') {
        setAb('original');
      } else if (e.key === 'b' || e.key === 'B') {
        if (cleaned) setAb('cleaned');
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [hasAudio, cleaned]);

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
          { value: 'cleaned', label: 'Cleaned', className: 'clean', disabled: !cleaned },
        ]}
      />
      <span className="timecode">
        {clock.t}
        <span className="dim dur"> / {clock.d}</span>
      </span>
    </div>
  );
}
