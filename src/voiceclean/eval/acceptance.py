"""Acceptance harness: hard release gates over the locked acceptance corpus.

Gates are explicit conditionals that record structured failures and keep
going — never bare asserts (which python -O deletes) and never exceptions
that mask which item failed. A run that fails any gate reports
release_gate_status="FAILED" with the failing items and reasons named.
"""

from pathlib import Path
from typing import Any

from voiceclean.eval.corpus import load_corpus_manifest
from voiceclean.pipeline import run_pipeline

# At least this fraction of speech units must be enhanced across the corpus,
# or the tool is functionally a no-op and the release gate fails. The floor
# is deliberately modest: on the current synthetic corpus the conservative
# guard reverts aggressively (measured 2025-08-19: 5 of 13 units enhanced
# across all bundled samples). A floor of 0.15 catches "nothing was
# enhanced at all" without demanding the guard loosen.
MIN_ENHANCED_SPEECH_FRACTION = 0.15


def _gate_failures(report: Any) -> list[str]:
    """Evaluate every hard gate for one item; return the failures."""
    failures: list[str] = []

    if report.output.samples != report.input.samples:
        failures.append(
            f"sample count mismatch: output {report.output.samples} != "
            f"input {report.input.samples}"
        )
    if report.output.channels != report.input.channels:
        failures.append(
            f"channel count mismatch: output {report.output.channels} != "
            f"input {report.input.channels}"
        )
    if report.output.sample_rate != report.input.sample_rate:
        failures.append(
            f"sample rate mismatch: output {report.output.sample_rate} != "
            f"input {report.input.sample_rate}"
        )
    if report.output.true_peak_dbtp is not None and report.output.true_peak_dbtp > -1.0:
        failures.append(
            f"true peak {report.output.true_peak_dbtp:.3f} dBTP exceeds the -1.0 ceiling"
        )
    for u in report.units:
        if u.guard_a_verdict == "UNVERIFIED" and u.final_decision == "enhanced":
            failures.append(f"unit {u.unit_id} was UNVERIFIED but selected enhanced audio")
        if u.final_decision == "enhanced" and u.guard_a_verdict != "PASS":
            failures.append(
                f"unit {u.unit_id} enhanced without a PASS verdict ({u.guard_a_verdict})"
            )
    return failures


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
    speech_units_total = 0
    speech_units_enhanced = 0

    for item in manifest.items:
        audio_file = Path(item.audio_path)
        out_file = out_dir / f"{item.id}_clean.wav"

        try:
            report = run_pipeline(
                input_path=audio_file,
                output_path=out_file,
                profile="production",
                overwrite=True,
            )
        except Exception as e:
            results.append(
                {
                    "id": item.id,
                    "passed": False,
                    "failures": [f"pipeline raised {type(e).__name__}: {e}"],
                }
            )
            continue

        failures = _gate_failures(report)
        speech_units_total += sum(1 for u in report.units if u.is_speech)
        speech_units_enhanced += sum(
            1 for u in report.units if u.is_speech and u.final_decision == "enhanced"
        )

        if not failures:
            passed_items += 1
        results.append(
            {
                "id": item.id,
                "passed": not failures,
                "failures": failures,
                "enhanced_units": report.summary.enhanced,
                "reverted_units": report.summary.reverted,
                "true_peak": report.output.true_peak_dbtp,
            }
        )

    corpus_failures: list[str] = []
    if speech_units_total > 0:
        enhanced_fraction = speech_units_enhanced / speech_units_total
        if enhanced_fraction < MIN_ENHANCED_SPEECH_FRACTION:
            corpus_failures.append(
                f"only {speech_units_enhanced}/{speech_units_total} speech units "
                f"({enhanced_fraction:.0%}) were enhanced — below the "
                f"{MIN_ENHANCED_SPEECH_FRACTION:.0%} floor; the tool did essentially nothing"
            )

    status = "PASSED" if (passed_items == total_items and not corpus_failures) else "FAILED"
    return {
        "manifest_sha256": manifest.manifest_sha256,
        "total_items": total_items,
        "passed_items": passed_items,
        "speech_units_total": speech_units_total,
        "speech_units_enhanced": speech_units_enhanced,
        "corpus_failures": corpus_failures,
        "release_gate_status": status,
        "results": results,
    }
