from pathlib import Path

from scripts import generate_schemas


def test_check_schemas_is_non_mutating_and_detects_drift(tmp_path: Path) -> None:
    expected = generate_schemas.rendered_schemas()
    for name, content in expected.items():
        (tmp_path / name).write_text(content, encoding="utf-8")

    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert generate_schemas.check_schemas(tmp_path) == []
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before

    stale = tmp_path / next(iter(expected))
    stale.write_text("{}\n", encoding="utf-8")
    assert generate_schemas.check_schemas(tmp_path) == [f"generated schema differs: {stale}"]


def test_check_schemas_reports_missing_and_unexpected_files(tmp_path: Path) -> None:
    unexpected = tmp_path / "retired.schema.json"
    unexpected.write_text("{}\n", encoding="utf-8")

    drift = generate_schemas.check_schemas(tmp_path)

    assert len(drift) == len(generate_schemas.MODELS) + 1
    assert all(
        f"missing generated schema: {tmp_path / name}" in drift
        for name, _model in generate_schemas.MODELS
    )
    assert f"unexpected generated schema: {unexpected}" in drift
