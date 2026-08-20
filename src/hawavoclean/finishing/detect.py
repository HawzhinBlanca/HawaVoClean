"""Acoustic defect and imbalance detectors for deterministic local finishing."""

from dataclasses import dataclass
from typing import Any

import numpy as np

# Low-mid (250-500 Hz) vs presence (2-5 kHz) level ratio of a normal voice,
# and how far above it counts as a defect worth correcting. Measured
# 2026-08-19 on four real recordings: +30.7, +41.2, +10.8 dB and a +43.9 dB
# fixture; the reference sits at the upper-middle so only genuine boom
# (proximity effect, resonant room) exceeds it.
NORMAL_VOICE_MUD_REFERENCE_DB = 36.0
MUD_EXCESS_THRESHOLD_DB = 6.0


@dataclass(frozen=True)
class DefectDetectionReport:
    """Detection scores for each prospective finishing stage."""

    has_dc_offset: bool
    dc_level: float
    has_hum: bool
    hum_freq_hz: float
    click_count: int
    has_plosives: bool
    mud_imbalance_db: float
    has_mud: bool
    has_harsh_sibilance: bool
    sibilance_ratio: float


def detect_defects(
    waveform: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
) -> DefectDetectionReport:
    """Analyze unit waveform for DC offset, electrical hum, clicks, plosives, and sibilance."""
    n = len(waveform)
    if n < 512:
        return DefectDetectionReport(
            has_dc_offset=False,
            dc_level=0.0,
            has_hum=False,
            hum_freq_hz=0.0,
            click_count=0,
            has_plosives=False,
            mud_imbalance_db=0.0,
            has_mud=False,
            has_harsh_sibilance=False,
            sibilance_ratio=1.0,
        )

    # 1. DC Offset
    dc_level = float(np.mean(waveform))
    has_dc = abs(dc_level) > 0.005

    # 2. Spectral analysis across frames of the waveform
    n_fft = min(2048, 2 ** int(np.floor(np.log2(n))))
    hop = n_fft // 2
    win = np.hanning(n_fft)
    num_frames = max(1, (n - n_fft) // hop + 1)

    # Sample up to 64 evenly spaced frames for performance and accuracy
    if num_frames > 64:
        frame_indices = np.linspace(0, num_frames - 1, 64, dtype=int)
    else:
        frame_indices = np.arange(num_frames)

    stft_mags = np.zeros((len(frame_indices), n_fft // 2 + 1), dtype=np.float32)
    for idx, f_idx in enumerate(frame_indices):
        chunk = waveform[f_idx * hop : f_idx * hop + n_fft] * win
        stft_mags[idx] = np.abs(np.fft.rfft(chunk, n=n_fft))

    fft_mag = np.mean(stft_mags, axis=0)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

    # 50/60 Hz mains hum: a dedicated long FFT on the low band. With the
    # 2048-point analysis above there are only ~3 bins between 30-100 Hz at
    # 48 kHz, so "hum bin > 4x the mean of the band" was mathematically
    # impossible and de-hum never ran on real inputs. Here: 16384-point
    # (~2.9 Hz bins at 48 kHz) and the hum bin is compared against the
    # MEDIAN of the 30-150 Hz band EXCLUDING its own neighbourhood.
    has_hum = False
    hum_freq = 0.0
    n_hum_fft = 16384
    if len(waveform) >= n_hum_fft:
        hum_frames = max(1, min(16, (len(waveform) - n_hum_fft) // (n_hum_fft // 2) + 1))
        hum_win = np.hanning(n_hum_fft)
        hum_mag = np.zeros(n_hum_fft // 2 + 1, dtype=np.float64)
        for i in range(hum_frames):
            start = i * (n_hum_fft // 2)
            hum_mag += np.abs(
                np.fft.rfft(waveform[start : start + n_hum_fft] * hum_win, n=n_hum_fft)
            )
        hum_mag /= hum_frames
        hum_freqs = np.fft.rfftfreq(n_hum_fft, d=1.0 / sample_rate)
        band = (hum_freqs >= 30.0) & (hum_freqs <= 150.0)
        best_ratio = 0.0
        for target in (50.0, 60.0):
            idx = int(np.argmin(np.abs(hum_freqs - target)))
            lo = max(0, idx - 2)
            hi = min(len(hum_mag), idx + 3)
            peak = float(np.max(hum_mag[lo:hi]))
            neighbourhood = np.zeros_like(band)
            neighbourhood[lo:hi] = True
            ref_bins = band & ~neighbourhood
            floor = float(np.median(hum_mag[ref_bins])) + 1e-9 if np.any(ref_bins) else 1e-9
            ratio = peak / floor
            if ratio > 8.0 and ratio > best_ratio:
                best_ratio = ratio
                has_hum = True
                hum_freq = float(hum_freqs[lo + int(np.argmax(hum_mag[lo:hi]))])

    # 3. Click / Transient spike detection
    diff = np.abs(np.diff(waveform))
    threshold_click = float(np.mean(diff) + 6.0 * np.std(diff))
    click_count = int(np.sum(diff > max(0.20, threshold_click)))

    # 4. Low-frequency plosive detection (<120Hz sudden energy burst)
    low_band_bins = freqs <= 120
    low_band_energy = float(np.sum(fft_mag[low_band_bins] ** 2))
    total_energy = float(np.sum(fft_mag**2)) + 1e-9
    has_plosives = (low_band_energy / total_energy) > 0.45

    # 5. Mud / Presence imbalance (250-500Hz vs 2k-5kHz)
    mud_bins = (freqs >= 250) & (freqs <= 500)
    pres_bins = (freqs >= 2000) & (freqs <= 5000)
    e_mud = float(np.mean(fft_mag[mud_bins])) + 1e-9
    e_pres = float(np.mean(fft_mag[pres_bins])) + 1e-9
    # Natural speech carries FAR more energy at 250-500 Hz than at 2-5 kHz —
    # measured on real recordings: +11 to +41 dB, median ~+30 dB. "Mud" is
    # EXCESS over that, not the mere presence of low-mids. The old +2 dB
    # threshold flagged every real voice and thinned it.
    mud_imbalance_db = float(20.0 * np.log10(e_mud / e_pres))
    has_mud = mud_imbalance_db > NORMAL_VOICE_MUD_REFERENCE_DB + MUD_EXCESS_THRESHOLD_DB

    # 6. Sibilance (5kHz - 10kHz vs overall mid band); at low sample rates
    # the sibilance band may lie above Nyquist — then there is no sibilance
    # to measure, not a NaN.
    sib_bins = (freqs >= 5000) & (freqs <= 10000)
    mid_bins = (freqs >= 1000) & (freqs <= 4000)
    if np.any(sib_bins) and np.any(mid_bins):
        e_sib = float(np.mean(fft_mag[sib_bins])) + 1e-9
        e_mid = float(np.mean(fft_mag[mid_bins])) + 1e-9
        sib_ratio = float(e_sib / e_mid)
    else:
        sib_ratio = 0.0
    has_harsh_sibilance = sib_ratio > 1.8

    return DefectDetectionReport(
        has_dc_offset=has_dc,
        dc_level=dc_level,
        has_hum=has_hum,
        hum_freq_hz=hum_freq,
        click_count=click_count,
        has_plosives=has_plosives,
        mud_imbalance_db=mud_imbalance_db,
        has_mud=has_mud,
        has_harsh_sibilance=has_harsh_sibilance,
        sibilance_ratio=sib_ratio,
    )


# ---------------------------------------------------------------------------
# Speech tilt: bounded, measured tonal restoration
# ---------------------------------------------------------------------------
#
# WHY A SECOND MEASURE. `mud_imbalance_db` above is a single number (250-500 Hz
# against 2-5 kHz) and it answers a single question: are the low-mids in
# excess? It is blind to a recording whose low end measures NORMAL but whose
# consonant region was never captured. A user reported exactly that: a very
# muffled 24 s take came back with every band moved by exactly the same
# +15.7 dB — a flat loudness gain and nothing else — and shipped as muffled as
# it arrived, only louder.
#
# WHAT THIS MEASURES INSTEAD. Wide band levels relative to the voice's own body
# band, each compared against a target, per band. The correction is the bounded
# difference. The bands are few and wide on purpose: this is a tilt corrector,
# not a room-correction curve, and every extra degree of freedom is another way
# to re-voice a speaker who did not need it.
#
# HOW THE TARGET WAS CALIBRATED. Not from a textbook long-term average speech
# spectrum — from the recordings this project is judged on. Two references have
# to measure as "already acceptable" and receive zero:
#   * the synthetic natural-voice fixture pinned by the 3.1.1 transparency gate
#     (tests/unit/test_finishing_tonal_transparency.py), and
#   * "Flute 09", a real 94 s recording whose finished sound the user approved.
# Measured (p75 of speech-active frames, dB relative to the 300-1000 Hz body):
#
#                            90-300   1.5-3k    3-6k
#   natural-voice fixture     +13.0    -22.0   -33.4
#   Flute 09 (approved)        +7.0    -27.8   -34.1
#   Teat1vo (reported bad)     +3.0    -26.4   -44.8
#
# Read that table before changing anything here. The reported file and the
# approved one are within 1.4 dB of each other from 90 Hz to 3 kHz. They differ
# in exactly one place: above 3 kHz the reported file is ~11 dB lower and still
# falling at ~20 dB per octave. So a correction that fires on the reported
# file's low-mids or presence would fire just as hard on the approved one —
# which is the 3.1.1 regression ("harsh and treble sounding, dialogs bass
# removed lot") all over again. The targets below sit BELOW both acceptable
# references in every band, a deadband is added on top of that, and the
# correction stops at the deadband edge. A voice inside the band gets exactly
# 0.0 dB, and no voice can be pushed past the edge. Over-brightening is
# impossible by construction rather than by tuning.
TILT_LOW_BAND_HZ = (90.0, 300.0)
TILT_BODY_BAND_HZ = (300.0, 1000.0)
TILT_PRESENCE_BAND_HZ = (1500.0, 3000.0)
TILT_BRILLIANCE_BAND_HZ = (3000.0, 6000.0)

# Target level of each band relative to the body band. Nothing is corrected
# above 6 kHz at all: air that was never recorded cannot be restored, only
# imitated with hiss.
TILT_TARGET_LOW_DB = 14.0
TILT_TARGET_PRESENCE_DB = -24.0
TILT_TARGET_BRILLIANCE_DB = -30.0

# Half-width of the "already acceptable" band around each target. Sized from
# the measured spread BETWEEN units of the approved recording (its 3-6 kHz
# level moves 4.8 dB across its five units), so that normal variation inside
# one good recording never crosses the line.
TILT_DEADBAND_DB = 7.0

# Caps. Low shelf: 6 dB is the most a corrective broadcast shelf takes before
# it stops correcting and starts re-voicing the speaker — which is exactly what
# 3.1.1 was about. Presence/brilliance: an octave-wide +10/+12 dB restorative
# bell is where the lifted band's own noise floor becomes the loudest new thing
# in it. The total-lift budget keeps a file that is short in both bands from
# receiving the sum of two full corrections.
TILT_MAX_LOW_CUT_DB = 6.0
TILT_MAX_LOW_LIFT_DB = 4.0
TILT_MAX_PRESENCE_LIFT_DB = 10.0
TILT_MAX_BRILLIANCE_LIFT_DB = 12.0
TILT_MAX_TOTAL_LIFT_DB = 14.0

# GATE 1 — the band must carry DYNAMICS, not a constant floor.
# Headroom is the band's own loud level (p95 over speech-active frames) minus
# its own quiet level (p10 over all frames). Speech swings tens of dB across
# syllables in every band it occupies; tape hiss, mains buzz, codec noise and
# dither do not. Measured: the approved recording carries +42 dB of headroom at
# 3-6 kHz and the reported one +18 dB (both real signal, one much weaker),
# against +1.4 dB for a brick-wall-lowpassed control and +4.1 dB for the
# no-pauses synthetic fixture (both correctly refused). The gate is a RAMP, not
# a cliff: a hard threshold made adjacent units of the same recording land on
# opposite sides of it and swap 10 dB of EQ mid-file.
TILT_HEADROOM_FULL_DB = 16.0
TILT_HEADROOM_ZERO_DB = 10.0

# GATE 2 — the band must have been CAPTURED AT ALL. A backstop below the
# headroom gate for material whose upper bands are so far down that no amount
# of gain is going to find speech in them. Also ramped.
TILT_REACH_FULL_DB = -48.0
TILT_REACH_ZERO_DB = -58.0

# A file whose presence is in genuine SURPLUS may have its low end lifted
# instead of cut. Gated on that surplus so a boomy recording can never argue
# its way into more bass.
TILT_THIN_SURPLUS_DB = 6.0

# Below this peak there is no voice here to balance — only dither and the
# quantisation floor, whose tilt means nothing.
TILT_MIN_PEAK_AMPLITUDE = 1e-4

TILT_MIN_ANALYSIS_SAMPLES = 8192
_TILT_FFT_SIZE = 2048
_TILT_HOP = 512
_TILT_MIN_FRAMES = 8
# Speech-active frames sit within this of the unit's loud reference. Silence and
# room tone must not drag the body level down: the target curve describes the
# VOICE, not the gaps between phrases.
_TILT_ACTIVE_RANGE_DB = 25.0


@dataclass(frozen=True)
class BandTilt:
    """One analysis band: what was measured, what was wanted, what was awarded."""

    name: str
    low_hz: float
    high_hz: float
    level_rel_body_db: float
    target_rel_body_db: float
    headroom_db: float
    correction_db: float
    gate_reason: str


@dataclass(frozen=True)
class SpeechTiltReport:
    """Measured spectral tilt of speech and the bounded correction earned by it."""

    measured: bool
    body_level_db: float
    bands: tuple[BandTilt, ...]
    low_shelf_db: float
    presence_db: float
    brilliance_db: float

    @property
    def is_correction(self) -> bool:
        """True when some band earned a move worth applying."""
        return (
            abs(self.low_shelf_db) >= 0.25 or self.presence_db >= 0.25 or self.brilliance_db >= 0.25
        )

    def summary(self) -> str:
        """Single-line provenance for the audit report."""
        return " ".join(
            f"{b.name}[{b.level_rel_body_db:+.1f}vs{b.target_rel_body_db:+.0f}"
            f",hr{b.headroom_db:.0f},{b.correction_db:+.1f}{b.gate_reason}]"
            for b in self.bands
        )


_EMPTY_TILT = SpeechTiltReport(
    measured=False,
    body_level_db=-200.0,
    bands=(),
    low_shelf_db=0.0,
    presence_db=0.0,
    brilliance_db=0.0,
)


def _ramp(value: float, zero_at: float, full_at: float) -> float:
    """Linear 0..1 ramp between two thresholds, in either direction."""
    if full_at == zero_at:
        return 1.0 if value >= full_at else 0.0
    return float(np.clip((value - zero_at) / (full_at - zero_at), 0.0, 1.0))


@dataclass(frozen=True)
class _BandStats:
    """Raw per-band statistics before any target or gate is applied."""

    level_db: float
    headroom_db: float
    present: bool


def _band_stats(
    stft_power: np.ndarray[Any, np.dtype[np.float64]],
    freqs: np.ndarray[Any, np.dtype[np.float64]],
    active: np.ndarray[Any, np.dtype[np.bool_]],
    low_hz: float,
    high_hz: float,
) -> _BandStats:
    """Level (p75 of active frames) and headroom (p95 active - p10 overall)."""
    mask = (freqs >= low_hz) & (freqs < high_hz)
    if not np.any(mask):
        return _BandStats(level_db=-200.0, headroom_db=0.0, present=False)
    band = np.mean(stft_power[:, mask], axis=1)
    level = 10.0 * np.log10(float(np.percentile(band[active], 75)) + 1e-30)
    loud = 10.0 * np.log10(float(np.percentile(band[active], 95)) + 1e-30)
    floor = 10.0 * np.log10(float(np.percentile(band, 10)) + 1e-30)
    return _BandStats(level_db=float(level), headroom_db=float(loud - floor), present=True)


def _size_correction(
    low: _BandStats,
    presence: _BandStats,
    brilliance: _BandStats,
    body_db: float,
) -> tuple[float, str, float, str, float, str]:
    """Turn three band measurements into three bounded, gated corrections."""

    def lift(band: _BandStats, target: float, cap: float) -> tuple[float, str]:
        if not band.present:
            return 0.0, ":above-nyquist"
        rel = band.level_db - body_db
        deficit = (target - TILT_DEADBAND_DB) - rel
        if deficit <= 0.0:
            return 0.0, ":in-band"
        reach = _ramp(rel, TILT_REACH_ZERO_DB, TILT_REACH_FULL_DB)
        dynamics = _ramp(band.headroom_db, TILT_HEADROOM_ZERO_DB, TILT_HEADROOM_FULL_DB)
        gain = min(deficit, cap) * reach * dynamics
        if gain < 0.05:
            return 0.0, ":not-captured" if reach < dynamics else ":no-dynamics"
        reason = ":lift" if reach * dynamics > 0.99 else ":lift-gated"
        return float(gain), reason

    presence_db, presence_reason = lift(
        presence, TILT_TARGET_PRESENCE_DB, TILT_MAX_PRESENCE_LIFT_DB
    )
    brilliance_db, brilliance_reason = lift(
        brilliance, TILT_TARGET_BRILLIANCE_DB, TILT_MAX_BRILLIANCE_LIFT_DB
    )

    # One budget for the two lifts, scaled together so the corrected shape is
    # still the shape that was measured.
    total = presence_db + brilliance_db
    if total > TILT_MAX_TOTAL_LIFT_DB:
        scale = TILT_MAX_TOTAL_LIFT_DB / total
        presence_db *= scale
        brilliance_db *= scale

    # Low shelf. A CUT needs only its own evidence — excess is excess. A LIFT
    # needs proof the file is genuinely thin (presence in surplus), so a boomy
    # recording can never talk the shelf into adding bass.
    if not low.present:
        return 0.0, ":above-nyquist", presence_db, presence_reason, brilliance_db, brilliance_reason
    low_rel = low.level_db - body_db
    presence_rel = presence.level_db - body_db if presence.present else -200.0
    excess = low_rel - (TILT_TARGET_LOW_DB + TILT_DEADBAND_DB)
    deficit = (TILT_TARGET_LOW_DB - TILT_DEADBAND_DB) - low_rel
    thin = presence_rel > TILT_TARGET_PRESENCE_DB + TILT_THIN_SURPLUS_DB
    if excess > 0.0:
        low_db, low_reason = -float(min(excess, TILT_MAX_LOW_CUT_DB)), ":cut"
    elif deficit > 0.0 and thin:
        low_db, low_reason = float(min(deficit, TILT_MAX_LOW_LIFT_DB)), ":thin-lift"
    else:
        low_db, low_reason = 0.0, ":in-band"
    return low_db, low_reason, presence_db, presence_reason, brilliance_db, brilliance_reason


def measure_speech_tilt(
    waveform: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
) -> SpeechTiltReport:
    """Measure band tilt against the speech-intelligibility target and size the fix.

    Deterministic: fixed transform size, fixed percentiles, no randomness,
    float64 throughout. Returns an all-zero correction when the unit is too
    short or too quiet to measure, when every band sits inside the deadband, or
    when the bands outside it fail their gates.
    """
    mono = np.asarray(waveform, dtype=np.float64)
    if mono.ndim > 1:
        mono = np.mean(mono, axis=0)
    if len(mono) < TILT_MIN_ANALYSIS_SAMPLES or sample_rate < 8000:
        return _EMPTY_TILT
    if not np.all(np.isfinite(mono)):
        return _EMPTY_TILT
    if float(np.max(np.abs(mono))) < TILT_MIN_PEAK_AMPLITUDE:
        return _EMPTY_TILT

    num_frames = (len(mono) - _TILT_FFT_SIZE) // _TILT_HOP + 1
    if num_frames < _TILT_MIN_FRAMES:
        return _EMPTY_TILT

    window = np.hanning(_TILT_FFT_SIZE)
    stft_power = np.empty((num_frames, _TILT_FFT_SIZE // 2 + 1), dtype=np.float64)
    for i in range(num_frames):
        chunk = mono[i * _TILT_HOP : i * _TILT_HOP + _TILT_FFT_SIZE] * window
        stft_power[i] = np.abs(np.fft.rfft(chunk, n=_TILT_FFT_SIZE)) ** 2
    freqs = np.asarray(np.fft.rfftfreq(_TILT_FFT_SIZE, d=1.0 / sample_rate), dtype=np.float64)

    frame_energy = np.sum(stft_power, axis=1)
    loud_ref = float(np.percentile(frame_energy, 90))
    active = frame_energy >= loud_ref * 10 ** (-_TILT_ACTIVE_RANGE_DB / 10.0)
    if int(np.sum(active)) < _TILT_MIN_FRAMES:
        active = np.ones(num_frames, dtype=bool)

    body = _band_stats(stft_power, freqs, active, *TILT_BODY_BAND_HZ)
    if not body.present or body.level_db < -180.0:
        return _EMPTY_TILT
    low = _band_stats(stft_power, freqs, active, *TILT_LOW_BAND_HZ)
    presence = _band_stats(stft_power, freqs, active, *TILT_PRESENCE_BAND_HZ)
    brilliance = _band_stats(stft_power, freqs, active, *TILT_BRILLIANCE_BAND_HZ)

    low_db, low_why, pres_db, pres_why, bril_db, bril_why = _size_correction(
        low, presence, brilliance, body.level_db
    )
    specs = (
        ("low", TILT_LOW_BAND_HZ, TILT_TARGET_LOW_DB, low, low_db, low_why),
        (
            "presence",
            TILT_PRESENCE_BAND_HZ,
            TILT_TARGET_PRESENCE_DB,
            presence,
            pres_db,
            pres_why,
        ),
        (
            "brilliance",
            TILT_BRILLIANCE_BAND_HZ,
            TILT_TARGET_BRILLIANCE_DB,
            brilliance,
            bril_db,
            bril_why,
        ),
    )
    bands = tuple(
        BandTilt(
            name=name,
            low_hz=edges[0],
            high_hz=edges[1],
            level_rel_body_db=stats.level_db - body.level_db,
            target_rel_body_db=target,
            headroom_db=stats.headroom_db,
            correction_db=gain,
            gate_reason=why,
        )
        for name, edges, target, stats, gain, why in specs
    )
    return SpeechTiltReport(
        measured=True,
        body_level_db=body.level_db,
        bands=bands,
        low_shelf_db=low_db,
        presence_db=pres_db,
        brilliance_db=bril_db,
    )


def aggregate_speech_tilt(reports: list[SpeechTiltReport]) -> SpeechTiltReport:
    """Combine per-unit measurements into ONE correction for the whole file.

    Units are finished independently, so a per-unit correction lets the tone
    step between adjacent blocks — measured at up to 2.8 dB in the 3-6 kHz band
    across two 12 s units of the reported recording, which is an audible pump
    every time a unit boundary goes by. The band levels are combined by MEDIAN
    (one loud outlying unit must not set the tone for a whole recording) and the
    correction is then sized once from those medians, so every unit of a file
    receives the identical filter.
    """
    usable = [r for r in reports if r.measured and r.bands]
    if not usable:
        return _EMPTY_TILT

    body = float(np.median([r.body_level_db for r in usable]))
    by_name: dict[str, _BandStats] = {}
    for index, name in enumerate(("low", "presence", "brilliance")):
        present = [r.bands[index] for r in usable if r.bands[index].gate_reason != ":above-nyquist"]
        if not present:
            by_name[name] = _BandStats(level_db=-200.0, headroom_db=0.0, present=False)
            continue
        by_name[name] = _BandStats(
            level_db=body + float(np.median([b.level_rel_body_db for b in present])),
            headroom_db=float(np.median([b.headroom_db for b in present])),
            present=True,
        )

    low_db, low_why, pres_db, pres_why, bril_db, bril_why = _size_correction(
        by_name["low"], by_name["presence"], by_name["brilliance"], body
    )
    specs = (
        ("low", TILT_LOW_BAND_HZ, TILT_TARGET_LOW_DB, by_name["low"], low_db, low_why),
        (
            "presence",
            TILT_PRESENCE_BAND_HZ,
            TILT_TARGET_PRESENCE_DB,
            by_name["presence"],
            pres_db,
            pres_why,
        ),
        (
            "brilliance",
            TILT_BRILLIANCE_BAND_HZ,
            TILT_TARGET_BRILLIANCE_DB,
            by_name["brilliance"],
            bril_db,
            bril_why,
        ),
    )
    return SpeechTiltReport(
        measured=True,
        body_level_db=body,
        bands=tuple(
            BandTilt(
                name=name,
                low_hz=edges[0],
                high_hz=edges[1],
                level_rel_body_db=stats.level_db - body,
                target_rel_body_db=target,
                headroom_db=stats.headroom_db,
                correction_db=gain,
                gate_reason=why,
            )
            for name, edges, target, stats, gain, why in specs
        ),
        low_shelf_db=low_db,
        presence_db=pres_db,
        brilliance_db=bril_db,
    )
