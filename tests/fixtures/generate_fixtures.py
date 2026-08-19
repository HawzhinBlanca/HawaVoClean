"""Generate deterministic reference synthetic audio fixtures for unit and integration testing."""

from pathlib import Path

import numpy as np
import soundfile as sf


def generate_speech_like_waveform(
    duration_s: float = 8.0,
    sample_rate: int = 48000,
    f0: float = 140.0,
    add_hum: bool = False,
    add_clicks: bool = False,
) -> np.ndarray:
    """Generate harmonic voice-like signal with natural pauses, formants, and gentle noise."""
    t = np.linspace(
        0.0, duration_s, int(sample_rate * duration_s), endpoint=False, dtype=np.float32
    )
    n_samples = len(t)

    # Fundamental + harmonics
    harmonics = [1.0, 0.7, 0.5, 0.35, 0.2, 0.15, 0.1, 0.05]
    voice = np.zeros(n_samples, dtype=np.float32)
    for h_idx, amp in enumerate(harmonics, start=1):
        voice += amp * np.sin(2.0 * np.pi * (f0 * h_idx) * t).astype(np.float32)

    # Speech rhythm envelope: active utterances and pauses
    # E.g. [0..2s speech], [2..2.6s pause], [2.6..5.5s speech], [5.5..6.2s pause], [6.2..8s speech]
    envelope = np.zeros(n_samples, dtype=np.float32)
    p1 = (t >= 0.2) & (t < 2.2)
    p2 = (t >= 2.8) & (t < 5.2)
    p3 = (t >= 5.8) & (t < 7.8)

    envelope[p1] = 0.5 + 0.3 * np.sin(2.0 * np.pi * 3.0 * t[p1])
    envelope[p2] = 0.6 + 0.2 * np.sin(2.0 * np.pi * 2.5 * t[p2])
    envelope[p3] = 0.5 + 0.3 * np.sin(2.0 * np.pi * 3.5 * t[p3])

    signal = (voice * envelope * 0.4).astype(np.float32)

    # Add gentle pink noise floor
    noise = np.random.default_rng(42).normal(0.0, 0.005, size=n_samples).astype(np.float32)
    signal += noise

    if add_hum:
        signal += 0.04 * np.sin(2.0 * np.pi * 50.0 * t).astype(np.float32)
        signal += 0.015 * np.sin(2.0 * np.pi * 100.0 * t).astype(np.float32)

    if add_clicks:
        click_positions = [int(0.8 * sample_rate), int(3.5 * sample_rate), int(6.5 * sample_rate)]
        for pos in click_positions:
            if pos < n_samples:
                signal[pos] += 0.45

    return np.clip(signal, -0.95, 0.95)


def main() -> None:
    fixtures_dir = Path("tests/fixtures")
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    sr = 48000

    # 1. Clean speech-like podcast file (mono)
    sig_mono = generate_speech_like_waveform(duration_s=8.0, sample_rate=sr)
    sf.write(fixtures_dir / "sample_sorani_podcast.wav", sig_mono, sr, subtype="PCM_24")

    # 2. Dual mono identical
    sig_dual = np.column_stack([sig_mono, sig_mono])
    sf.write(fixtures_dir / "sample_dual_mono.wav", sig_dual, sr, subtype="PCM_24")

    # 3. Split speakers (ch0 speaks first, ch1 speaks second)
    spk1 = generate_speech_like_waveform(duration_s=8.0, sample_rate=sr, f0=120.0)
    spk2 = generate_speech_like_waveform(duration_s=8.0, sample_rate=sr, f0=220.0)
    # alternate active regions
    t = np.linspace(0.0, 8.0, len(spk1), endpoint=False)
    spk1[t >= 4.0] = 0.001 * spk1[t >= 4.0]
    spk2[t < 4.0] = 0.001 * spk2[t < 4.0]
    sig_split = np.column_stack([spk1, spk2])
    sf.write(fixtures_dir / "sample_split_speakers.wav", sig_split, sr, subtype="PCM_24")

    # 4. Ambiguous stereo (stereo chorus / panning)
    pan_l = sig_mono * 0.8 + 0.1 * np.roll(sig_mono, 200)
    pan_r = sig_mono * 0.5 + 0.4 * np.roll(sig_mono, 500)
    sig_ambig = np.column_stack([pan_l, pan_r])
    sf.write(fixtures_dir / "sample_ambiguous_stereo.wav", sig_ambig, sr, subtype="PCM_24")

    # 5. Noisy with hum and clicks
    sig_hum = generate_speech_like_waveform(
        duration_s=6.0, sample_rate=sr, add_hum=True, add_clicks=True
    )
    sf.write(fixtures_dir / "sample_noisy_hum.wav", sig_hum, sr, subtype="PCM_24")

    print("Generated all audio fixtures in tests/fixtures/")


if __name__ == "__main__":
    main()
