"""Tests for the Broadcast Quality Certificate HTML renderer, server route, and Studio 4.5 chat commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from hawavoclean.assistant import AudioPlanRequest, PlanPayload, _rules_plan, _schema
from hawavoclean.report.certificate import render_qc_certificate_html
from hawavoclean.server.app import create_app


def _sample_report() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "created_at": "2026-11-21T08:00:00Z",
        "input": {
            "path": "/Users/private/secret_client/interview_01.wav",
            "duration_s": 12.5,
            "sample_rate": 48000,
            "channels": 2,
            "integrated_lufs": -24.2,
            "true_peak_dbtp": -2.1,
        },
        "output": {
            "path": r"C:\Users\private\exports\interview_01.clean.wav",
            "duration_s": 12.5,
            "sample_rate": 48000,
            "channels": 2,
            "integrated_lufs": -16.1,
            "true_peak_dbtp": -1.05,
            "loudness_range_lu": 4.8,
        },
        "provenance": {
            "profile": "studio_pro",
            "version": "4.5.0",
        },
        "summary": {
            "units_total": 3,
            "enhanced": 2,
            "reverted": 1,
            "no_speech": 0,
            "error_passthrough": 0,
        },
        "units": [
            {
                "unit_id": 0,
                "start_time_s": 0.0,
                "end_time_s": 4.2,
                "final_decision": "enhanced",
                "guard_a_verdict": "PASS",
                "guard_b_verdict": "PASS",
                "chosen_strength": 0.85,
                "decision_reason": "Clean vocal enhancement",
            },
            {
                "unit_id": 1,
                "start_time_s": 4.2,
                "end_time_s": 8.4,
                "final_decision": "reverted",
                "guard_a_verdict": "FAIL",
                "guard_b_verdict": "PASS",
                "chosen_strength": 0.0,
                "decision_reason": "Preserved original consonant <script>alert(1)</script>",
            },
            {
                "unit_id": 2,
                "start_time_s": 8.4,
                "end_time_s": 12.5,
                "final_decision": "enhanced",
                "guard_a_verdict": "PASS",
                "guard_b_verdict": "PASS",
                "chosen_strength": 0.78,
                "decision_reason": "OK",
            },
        ],
    }


def test_render_qc_certificate_html_en_sanitizes_paths_and_xss() -> None:
    html_doc = render_qc_certificate_html(_sample_report(), lang="en")
    assert 'lang="en" dir="ltr"' in html_doc
    assert "GUARD VERIFIED · BROADCAST READY" in html_doc
    assert "interview_01.wav" in html_doc
    assert "interview_01.clean.wav" in html_doc
    # Host filesystem paths must never leak into the client certificate.
    assert "/Users/private" not in html_doc
    assert "C:\\Users\\private" not in html_doc
    # XSS payloads must be strictly escaped.
    assert "<script>" not in html_doc
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_doc
    assert 'class="timeline-svg"' in html_doc
    assert "-16.1 LUFS" in html_doc
    assert "-1.05 dBTP" in html_doc
    assert "4.8 LU" in html_doc


def test_render_qc_certificate_html_ckb_rtl_and_warn_states() -> None:
    rep = _sample_report()
    rep["output"]["true_peak_dbtp"] = -0.2  # exceeds -0.85 ceiling -> warn badge
    rep["units"] = [
        {
            "unit_id": i,
            "start_time_s": float(i),
            "end_time_s": float(i + 1),
            "final_decision": "error_passthrough" if i == 0 else "no_speech",
        }
        for i in range(505)
    ]
    rep["summary"]["error_passthrough"] = 1
    html_doc = render_qc_certificate_html(rep, lang="ckb")
    assert 'lang="ckb" dir="rtl"' in html_doc
    assert "بڕوانامەی کوالێتی دەنگی HawaVoClean" in html_doc
    assert "badge warn" in html_doc
    assert "Showing first 500 of 505 speech units." in html_doc


def test_render_qc_certificate_handles_empty_and_nonfinite() -> None:
    html_empty = render_qc_certificate_html(
        {
            "input": {"duration_s": float("nan"), "integrated_lufs": True},
            "output": {"true_peak_dbtp": float("inf")},
            "units": [],
        }
    )
    assert "No speech units recorded." in html_empty
    assert "badge warn" in html_empty

    html_bad_unit = render_qc_certificate_html(
        {
            "input": {"duration_s": 5.0},
            "units": [{"start_time_s": "bad", "end_time_s": 1.0}],
        }
    )
    assert "0:00.0 – 0:01.0" in html_bad_unit


def test_certificate_endpoint_auth_errors_and_headers(tmp_path: Path) -> None:
    token = "test-secret-token-12345"
    app = create_app(token=token)
    client = TestClient(app, base_url="http://127.0.0.1:8765")

    # Unauthenticated rejected
    res_unauth = client.get("/api/v1/jobs/missing-id/certificate")
    assert res_unauth.status_code in (401, 403)

    headers = {"X-Hawa-Token": token}
    # Unknown job -> 404
    res_404 = client.get("/api/v1/jobs/does-not-exist/certificate", headers=headers)
    assert res_404.status_code == 404

    # Seed a completed job record directly into the manager with an inline report
    manager = app.state.job_manager
    job_id = "job-cert-proof-1"
    report_path = tmp_path / "cert_report.json"
    report_path.write_text(json.dumps(_sample_report()), encoding="utf-8")

    # Monkeypatch manager.get_status cleanly for deterministic route testing
    orig_get = manager.get_status

    def fake_get_status(jid: str) -> dict[str, Any] | None:
        if jid == job_id:
            return {"job_id": job_id, "state": "done", "report": _sample_report()}
        if jid == "job-no-report":
            return {"job_id": jid, "state": "queued"}
        result = orig_get(jid)
        return result if isinstance(result, dict) else None

    manager.get_status = fake_get_status

    res_409 = client.get("/api/v1/jobs/job-no-report/certificate", headers=headers)
    assert res_409.status_code == 409

    res_ok = client.get(
        f"/api/v1/jobs/{job_id}/certificate?lang=ckb&download=1",
        headers=headers,
    )
    assert res_ok.status_code == 200
    assert "text/html" in res_ok.headers["content-type"]
    assert res_ok.headers["x-content-type-options"] == "nosniff"
    assert res_ok.headers["cache-control"] == "no-store"
    assert "default-src 'none'" in res_ok.headers["content-security-policy"]
    assert "attachment; filename=" in res_ok.headers["content-disposition"]
    assert 'dir="rtl"' in res_ok.text
    assert "/Users/private" not in res_ok.text


@pytest.mark.parametrize(
    ("prompt", "expected_op", "expected_field", "expected_val"),
    [
        ("show spectrogram", "ui_action", "ui_action", "show_spectrogram"),
        ("شەبەنگی ٢ ڕەهەندی", "ui_action", "ui_action", "show_spectrogram"),
        ("show waveform", "ui_action", "ui_action", "show_waveform"),
        ("open batch queue", "ui_action", "ui_action", "open_batch"),
        ("show keyboard shortcuts", "ui_action", "ui_action", "open_shortcuts"),
        ("download qc report", "ui_action", "ui_action", "download_certificate"),
        ("بڕوانامەی QC", "ui_action", "ui_action", "download_certificate"),
        ("solo 3000 Hz", "playback", "playback_action", "solo_freq"),
        ("listen to 7.2 kHz", "playback", "playback_action", "solo_freq"),
        ("stop solo", "playback", "playback_action", "clear_solo"),
    ],
)
def test_assistant_studio45_chat_rules(
    prompt: str, expected_op: str, expected_field: str, expected_val: str
) -> None:
    plan = _rules_plan(AudioPlanRequest(message=prompt))
    assert plan.operation == expected_op
    assert getattr(plan, expected_field) == expected_val
    if expected_val == "solo_freq":
        assert plan.solo_hz is not None and 20.0 <= plan.solo_hz <= 18000.0


def test_assistant_solo_hz_rejects_bool_and_present_in_schema() -> None:
    raw_bool: Any = True
    with pytest.raises(ValueError, match="numeric plan parameters"):
        PlanPayload(
            operation="playback",
            playback_action="solo_freq",
            solo_hz=raw_bool,
            explanation="Invalid bool solo_hz",
        )
    schema = _schema()
    playback_branch = next(
        b for b in schema["anyOf"] if b["properties"]["operation"]["const"] == "playback"
    )
    assert "solo_hz" in playback_branch["properties"]
    assert "solo_freq" in playback_branch["properties"]["playback_action"]["enum"]
    assert "clear_solo" in playback_branch["properties"]["playback_action"]["enum"]
    ui_branch = next(
        b for b in schema["anyOf"] if b["properties"]["operation"]["const"] == "ui_action"
    )
    assert "download_certificate" in ui_branch["properties"]["ui_action"]["enum"]
    assert "show_spectrogram" in ui_branch["properties"]["ui_action"]["enum"]
