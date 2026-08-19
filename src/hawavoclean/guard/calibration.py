"""Guard threshold calibration loader and validator."""

import json
from pathlib import Path
from typing import Any

from hawavoclean.config import GuardConfig
from hawavoclean.errors import CalibrationError


def load_calibration_artifact(calibration_path: Path | str) -> dict[str, Any]:
    """Load and validate guard calibration artifact."""
    p = Path(calibration_path).resolve()
    if not p.exists():
        raise CalibrationError(f"Guard calibration artifact not found: {p}")

    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise CalibrationError(f"Failed to read calibration JSON from {p}: {e}") from e

    required_keys = ["schema_version", "calibration_id", "thresholds", "guard_id"]
    for k in required_keys:
        if k not in data:
            raise CalibrationError(f"Invalid calibration artifact: missing required key '{k}'")

    return dict(data)


def apply_calibrated_thresholds(
    config: GuardConfig,
    calibration_data: dict[str, Any],
) -> GuardConfig:
    """Return updated GuardConfig with locked calibrated thresholds."""
    thresh = calibration_data.get("thresholds", {})
    return GuardConfig(
        guard_id=str(calibration_data.get("guard_id", config.guard_id)),
        probe_id=str(calibration_data.get("probe_id", config.probe_id)),
        calibration_file=config.calibration_file,
        min_anchor_confidence=float(
            thresh.get("min_anchor_confidence", config.min_anchor_confidence)
        ),
        max_posterior_js_div=float(thresh.get("max_posterior_js_div", config.max_posterior_js_div)),
        max_peak_js_div=float(thresh.get("max_peak_js_div", config.max_peak_js_div)),
        mode=config.mode,
        max_timing_drift_ms=float(thresh.get("max_timing_drift_ms", config.max_timing_drift_ms)),
        enforce_signal_integrity=config.enforce_signal_integrity,
        spectral_hole_thresh=float(thresh.get("spectral_hole_thresh", config.spectral_hole_thresh)),
        musical_noise_thresh=float(thresh.get("musical_noise_thresh", config.musical_noise_thresh)),
        min_hf_preservation_ratio=float(
            thresh.get("min_hf_preservation_ratio", config.min_hf_preservation_ratio)
        ),
    )
