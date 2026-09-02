from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

from hawavoclean.eval.auto_calibration import run_auto_calibration


@dataclass
class DummyPass:
    pass_index: int
    enhanced: bool
    separation_db: float
    cumulative_drift_db: float
    discarded: bool
    discard_reason: str | None = None


@dataclass
class DummyMultipassReport:
    passes: list[DummyPass]


@dataclass
class DummyItem:
    id: str
    audio_path: str


@dataclass
class DummyManifest:
    manifest_sha256: str
    items_count: int
    items: list[DummyItem]


def test_run_auto_calibration_comprehensive(tmp_path: Path) -> None:
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text("{}", encoding="utf-8")
    output_file = tmp_path / "out" / "proof.json"

    dummy_manifest = DummyManifest(
        manifest_sha256="abc123sha",
        items_count=3,
        items=[
            DummyItem(id="item_pass", audio_path=str(tmp_path / "item1.wav")),
            DummyItem(id="item_discards", audio_path=str(tmp_path / "item2.wav")),
            DummyItem(id="item_error", audio_path=str(tmp_path / "item3.wav")),
        ],
    )

    def fake_run_multipass(
        input_path: Path, output_path: Path | None = None, **_kwargs: Any
    ) -> DummyMultipassReport:
        _ = output_path
        if "item1" in str(input_path):
            return DummyMultipassReport(
                passes=[
                    DummyPass(
                        pass_index=0,
                        enhanced=True,
                        separation_db=14.5,
                        cumulative_drift_db=0.05,
                        discarded=False,
                    ),
                    DummyPass(
                        pass_index=1,
                        enhanced=True,
                        separation_db=15.0,
                        cumulative_drift_db=0.10,
                        discarded=True,
                        discard_reason="guard regressed",
                    ),
                ]
            )
        elif "item2" in str(input_path):
            return DummyMultipassReport(
                passes=[
                    DummyPass(
                        pass_index=0,
                        enhanced=True,
                        separation_db=10.0,
                        cumulative_drift_db=0.8,
                        discarded=True,
                        discard_reason="excessive drift detected",
                    ),
                    DummyPass(
                        pass_index=1,
                        enhanced=True,
                        separation_db=10.1,
                        cumulative_drift_db=0.9,
                        discarded=True,
                        discard_reason="separation gain below floor",
                    ),
                ]
            )
        else:
            raise RuntimeError("synthetic processing crash")

    with (
        patch(
            "hawavoclean.eval.auto_calibration.load_corpus_manifest", return_value=dummy_manifest
        ),
        patch("hawavoclean.multipass.run_multipass", side_effect=fake_run_multipass),
    ):
        proof = run_auto_calibration(manifest_file, output_file)

    assert output_file.is_file()
    saved = json.loads(output_file.read_text(encoding="utf-8"))
    assert saved == proof

    assert proof["corpus"]["manifest_sha256"] == "abc123sha"
    assert proof["corpus"]["items_count"] == 3

    assert proof["summary"]["total_passes_run"] == 4
    assert proof["summary"]["total_items_shipped"] == 2
    assert proof["summary"]["halted_by_guard"] == 1
    assert proof["summary"]["halted_by_drift"] == 1
    assert proof["summary"]["halted_by_separation"] == 1

    per_item = proof["per_item"]
    assert len(per_item) == 3
    assert per_item[0]["id"] == "item_pass"
    assert per_item[0]["shipped_pass_index"] == 0
    assert per_item[0]["halt_reason"] == "guard"

    assert per_item[1]["id"] == "item_discards"
    assert per_item[1]["shipped_pass_index"] == 0

    assert per_item[2]["id"] == "item_error"
    assert "synthetic processing crash" in per_item[2]["error"]
