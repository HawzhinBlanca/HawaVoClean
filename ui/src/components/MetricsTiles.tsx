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
  return (
    <div className="tile">
      <div className="k">{label}</div>
      <div className="vals">
        {hasClean ? (
          <span className="clean">{fmt(clean, digits)}</span>
        ) : hasOrig ? (
          <span className="single">{fmt(orig, digits)}</span>
        ) : (
          <span className="none">—</span>
        )}
        <span className="unit">{unit}</span>
      </div>
      <div className="sub">
        {hasClean ? (
          <>
            <span className="orig" title="Original">
              {fmt(orig, digits)}
            </span>
            <span className="arrow">→</span>
            <span className={`delta${deltaCls}`} title="Change after cleaning">
              {deltaText ?? '—'}
            </span>
          </>
        ) : (
          <span className="tag">{hasOrig ? 'original' : ''}</span>
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
    <div className="metrics">
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
