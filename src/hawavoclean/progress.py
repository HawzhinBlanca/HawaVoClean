"""Pipeline progress events.

``run_pipeline`` accepts an optional ``on_progress`` callback and invokes it
with a :class:`ProgressEvent` at every stage boundary. The callback is an
observer only: an exception raised inside it is logged and swallowed, so a
broken progress sink can never break a processing run.

Stage weights (contract ``docs/ui-contract.md`` section 2): preflight 0.02,
decode 0.05, segment 0.08, enhancement + guard 0.08 -> 0.80 linearly over
units, finish 0.80 -> 0.95, publish 0.98, done 1.0.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from hawavoclean.logging import get_logger

logger = get_logger("progress")

Stage = Literal[
    "preflight", "decode", "segment", "enhance", "guard", "finish", "publish", "done", "error"
]

PROGRESS_PREFLIGHT = 0.02
PROGRESS_DECODE = 0.05
PROGRESS_SEGMENT = 0.08
PROGRESS_UNITS_END = 0.80
PROGRESS_FINISH_START = 0.80
PROGRESS_FINISH_END = 0.95
PROGRESS_PUBLISH = 0.98
PROGRESS_DONE = 1.0


@dataclass(frozen=True)
class ProgressEvent:
    """One progress notification: where the pipeline is and how far along."""

    stage: Stage
    progress: float
    message: str
    unit_index: int | None = None
    unit_total: int | None = None
    #: Multi-pass runs only: which pass this event belongs to. ``pass_total``
    #: stays ``None`` in auto mode, where the total is unknown until the run
    #: stands itself down. Single-pass runs never set ``pass_index``, so their
    #: event stream is byte-identical to the pre-multipass contract.
    pass_index: int | None = None
    pass_total: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Contract JSON shape: ``{"event":"progress","stage":..,"progress":..,"message":..}``
        plus ``"unit":{"index":..,"total":..}`` when the event is about one unit,
        plus ``"pass":{"index":..,"total":..}`` when it belongs to a multi-pass
        run (``total`` is ``null`` while auto mode has not decided the total)."""
        out: dict[str, Any] = {
            "event": "progress",
            "stage": self.stage,
            "progress": round(float(self.progress), 4),
            "message": self.message,
        }
        if self.unit_index is not None and self.unit_total is not None:
            out["unit"] = {"index": self.unit_index, "total": self.unit_total}
        if self.pass_index is not None:
            out["pass"] = {"index": self.pass_index, "total": self.pass_total}
        return out


ProgressCallback = Callable[[ProgressEvent], None]


def unit_progress(index: int, total: int, *, done: bool) -> float:
    """Overall progress for unit ``index`` (1-based) of ``total``.

    The enhancement + guard span (0.08 -> 0.80) is divided into ``total``
    equal slices; ``done=False`` is the start of the unit's slice (emitted
    before enhancement), ``done=True`` its end (emitted after the guard)."""
    if total <= 0:
        return PROGRESS_UNITS_END
    frac = (index if done else index - 1) / total
    frac = min(1.0, max(0.0, frac))
    return PROGRESS_SEGMENT + (PROGRESS_UNITS_END - PROGRESS_SEGMENT) * frac


def emit_progress(callback: ProgressCallback | None, event: ProgressEvent) -> None:
    """Invoke ``callback`` with ``event``; never let a sink error escape."""
    if callback is None:
        return
    try:
        callback(event)
    except Exception as e:  # observer errors must not break the run
        logger.warning(f"Progress callback failed at stage {event.stage!r}: {e}")
