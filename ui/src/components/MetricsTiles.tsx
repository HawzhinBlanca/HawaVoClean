import { useStore } from '../state/store';

interface TileProps {
  label: string;
  unit: string;
  orig: number | null | undefined;
  clean: number | null | undefined;
  hasCleaned: boolean;
  /** 'lower' → a drop is good (noise floor); 'higher' → a rise is good; 'none' → neutral */
  better: 'lower' | 'higher' | 'none';
  digits?: number;
}

function fmt(v: number | null | undefined, digits: number): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—';
  return v.toFixed(digits);
}

function Tile({ label, unit, orig, clean, hasCleaned, better, digits = 1 }: TileProps) {
  const hasOrig = orig !== null && orig !== undefined && Number.isFinite(orig);
  const hasClean = hasCleaned && clean !== null && clean !== undefined && Number.isFinite(clean);
  let delta: number | null = null;
  if (hasOrig && hasClean) delta = (clean as number) - (orig as number);
  let deltaCls = '';
  if (delta !== null && Math.abs(delta) >= 0.05 && better !== 'none') {
    const improved = better === 'lower' ? delta < 0 : delta > 0;
    deltaCls = improved ? ' good' : ' bad';
  }
  const deltaText =
    delta !== null ? `${delta > 0 ? '+' : delta < 0 ? '−' : '±'}${Math.abs(delta).toFixed(digits)}` : null;
  // A meter with no signal is still a meter: the key dims, the value holds an
  // em-dash at full size so the tile keeps its height, and the delta row
  // carries a skeleton bar where the number will land (it shimmers while the
  // clip is being analyzed — see interaction.css).
  const dataState = hasClean ? 'ab' : hasOrig ? 'single' : 'empty';
  // D1 · read as separate nodes the tile is a pile of loose numbers — "Noise
  // floor", "−84.7", "dB", "−48.5", "−36.2" — with the one mark that says how
  // they relate (the delta operator) already hidden as decoration. The tile is
  // therefore named as a whole and its interior is hidden from assistive
  // technology, so it is announced once, as the sentence it draws.
  const say = hasClean
    ? `${label}: ${fmt(clean, digits)} ${unit}, from ${fmt(orig, digits)} ${unit}, change ${deltaText ?? '—'} ${unit}`
    : hasOrig
      ? `${label}: ${fmt(orig, digits)} ${unit}, original only`
      : `${label}: not measured yet`;
  return (
    <div className="tile" data-state={dataState} role="group" aria-label={say}>
      <div className="k" aria-hidden="true">
        {label}
      </div>
      <div className="vals" aria-hidden="true">
        {hasClean ? (
          <span className="clean">{fmt(clean, digits)}</span>
        ) : hasOrig ? (
          <span className="single">{fmt(orig, digits)}</span>
        ) : (
          <span className="none">—</span>
        )}
        <span className="unit">{unit}</span>
      </div>
      <div className="sub" aria-hidden="true">
        {hasClean ? (
          <>
            {/* This row used to read `−24.9 → +3.2`, which every reader parses
                as before → after — and +3.2 is not the after value, it is the
                change. The arrow is replaced by the delta operator, which can
                only be read one way. */}
            <span className="orig" title="Original">
              {fmt(orig, digits)}
            </span>
            <span className="op" aria-hidden="true">
              Δ
            </span>
            <span className={`delta${deltaCls}`} title="Change after cleaning">
              {deltaText ?? '—'}
            </span>
          </>
        ) : hasOrig ? (
          <span className="tag">original</span>
        ) : (
          <span className="skel" aria-hidden="true" />
        )}
      </div>
    </div>
  );
}

export function MetricsTiles() {
  const original = useStore((s) => s.original);
  const cleaned = useStore((s) => s.cleaned);
  const hasCleaned = Boolean(cleaned);
  return (
    <div className="metrics" role="group" aria-label="Loudness and noise measurements">
      <Tile
        label="Integrated"
        unit="LUFS"
        orig={original?.loudness.integrated_lufs}
        clean={cleaned?.loudness.integrated_lufs}
        hasCleaned={hasCleaned}
        better="none"
      />
      <Tile
        label="True peak"
        unit="dBTP"
        orig={original?.loudness.true_peak_dbtp}
        clean={cleaned?.loudness.true_peak_dbtp}
        hasCleaned={hasCleaned}
        better="none"
      />
      <Tile
        label="Noise floor"
        unit="dB"
        orig={original?.noise_floor_db}
        clean={cleaned?.noise_floor_db}
        hasCleaned={hasCleaned}
        better="lower"
      />
    </div>
  );
}
