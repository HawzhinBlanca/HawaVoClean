"""Data models and records for segmentation and speech units."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SpeechInterval:
    """A contiguous interval of detected speech in samples."""

    start_sample: int
    end_sample: int

    @property
    def length_samples(self) -> int:
        return self.end_sample - self.start_sample


@dataclass
class SpeechUnit:
    """Utterance group representation with context windows and speech masks as in BLUEPRINT.md section 10."""

    unit_id: int
    channel_id: int
    start_sample: int
    end_sample: int
    context_start_sample: int
    context_end_sample: int
    is_speech: bool
    forced_boundary: bool = False
    speech_mask: np.ndarray[Any, np.dtype[np.bool_]] = field(
        default_factory=lambda: np.empty(0, dtype=np.bool_)
    )
    input_sha256: str = ""

    @property
    def core_length_samples(self) -> int:
        return self.end_sample - self.start_sample

    @property
    def total_context_samples(self) -> int:
        return self.context_end_sample - self.context_start_sample

    @property
    def left_context_samples(self) -> int:
        return self.start_sample - self.context_start_sample

    @property
    def right_context_samples(self) -> int:
        return self.context_end_sample - self.end_sample
