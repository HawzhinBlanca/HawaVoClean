"""Acceptance testing harness verifying hard release gates on the locked acceptance corpus."""

from pathlib import Path
from typing import Any

from eval.corpus import load_corpus_manifest
from voiceclean.pipeline import run_pipeline


def evaluate_acceptance_gates(
    manifest_path: Path | str,
    output_dir: Path | str = "data/acceptance/outputs",
) -> dict[str, Any]:
    """Execute all hard release gates against the locked acceptance dataset."""
    manifest = load_corpus_manifest(manifest_path)
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    total_items = len(manifest.items)
    passed_items = 0

    for item in manifest.items:
        audio_file = Path(item.audio_path)
        out_file = out_dir / f"{item.id}_clean.wav"

        # Run full production pipeline
        report = run_pipeline(
            input_path=audio_file,
            output_path=out_file,
            profile="production",
            overwrite=True,
        )

        # Verify acceptance invariants
        assert report.output.samples == report.input.samples, f"Sample mismatch for {item.id}"
        assert report.output.channels == report.input.channels, f"Channel mismatch for {item.id}"
        assert report.output.sample_rate == report.input.sample_rate, (
            f"Sample rate mismatch for {item.id}"
        )
        if report.output.true_peak_dbtp is not None:
            assert report.output.true_peak_dbtp <= -0.9, f"True peak violated for {item.id}"

        # Invariant: No UNVERIFIED unit selected enhanced audio
        for u in report.units:
            if u.guard_a_verdict == "UNVERIFIED":
                assert u.final_decision != "enhanced", (
                    f"Unit {u.unit_id} was UNVERIFIED but enhanced!"
                )

        passed_items += 1
        results.append(
            {
                "id": item.id,
                "passed": True,
                "enhanced_units": report.summary.enhanced,
                "reverted_units": report.summary.reverted,
                "true_peak": report.output.true_peak_dbtp,
            }
        )

    return {
        "manifest_sha256": manifest.manifest_sha256,
        "total_items": total_items,
        "passed_items": passed_items,
        "release_gate_status": "PASSED" if passed_items == total_items else "FAILED",
        "results": results,
    }
