"""Unit tests for sample-accurate timeline assembly and postcondition validation."""

import numpy as np
import pytest

from voiceclean.assembly.stitch import assemble_channel_timeline
from voiceclean.assembly.validate import validate_assembled_timeline
from voiceclean.audio.types import AudioBuffer
from voiceclean.errors import OutputValidationError
from voiceclean.segmentation.types import SpeechUnit


@pytest.mark.unit
def test_assemble_and_validate_clean_timeline() -> None:
    sr = 48000
    u0 = SpeechUnit(0, 0, 0, 10000, 0, 10000, is_speech=True)
    u1 = SpeechUnit(1, 0, 10000, 24000, 10000, 24000, is_speech=False)

    w0 = 0.3 * np.ones(10000, dtype=np.float32)
    w1 = np.zeros(14000, dtype=np.float32)

    timeline = assemble_channel_timeline([u0, u1], [w0, w1], total_samples=24000, sample_rate=sr)
    assert len(timeline) == 24000

    buf = AudioBuffer(data=timeline, sample_rate=sr)
    # Validate postconditions - must pass without error
    validate_assembled_timeline(
        buf, expected_channels=1, expected_samples=24000, expected_sample_rate=sr, units=[u0, u1]
    )


@pytest.mark.unit
def test_validate_assembled_timeline_detects_gap() -> None:
    sr = 48000
    # Gap between 9000 and 10000
    u0 = SpeechUnit(0, 0, 0, 9000, 0, 9000, is_speech=True)
    u1 = SpeechUnit(1, 0, 10000, 20000, 10000, 20000, is_speech=False)
    buf = AudioBuffer(data=np.zeros((1, 20000), dtype=np.float32), sample_rate=sr)

    with pytest.raises(OutputValidationError, match="Timeline gap detected"):
        validate_assembled_timeline(
            buf,
            expected_channels=1,
            expected_samples=20000,
            expected_sample_rate=sr,
            units=[u0, u1],
        )
