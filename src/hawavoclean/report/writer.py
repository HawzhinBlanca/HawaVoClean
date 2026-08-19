"""JSON audit report serialization and validation."""

from pathlib import Path

from hawavoclean.report.schema import HawaVoCleanReport


def serialize_json_report(report: HawaVoCleanReport) -> str:
    """Serialize HawaVoCleanReport model to formatted JSON string."""
    return report.model_dump_json(indent=2)


def write_json_report(report: HawaVoCleanReport, output_path: Path | str) -> Path:
    """Write validated JSON audit report to disk."""
    dest = Path(output_path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(serialize_json_report(report), encoding="utf-8")
    return dest


def load_json_report(report_path: Path | str) -> HawaVoCleanReport:
    """Load and validate an existing JSON audit report."""
    return HawaVoCleanReport.model_validate_json(
        Path(report_path).resolve().read_text(encoding="utf-8")
    )
