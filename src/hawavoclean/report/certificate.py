"""Self-contained, offline, print-ready Broadcast Quality Certificate (HTML) renderer.

Generates a single-file HTML report from a completed HawaVoClean JSON report dict
with zero external assets, strict HTML escaping, and basename-only file naming so
host filesystem paths are never leaked to clients.
"""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _basename(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "untitled"
    # Normalize both POSIX and Windows separators before taking basename.
    cleaned = raw.replace("\\", "/")
    name = Path(cleaned).name
    return name or "untitled"


def _fmt_num(value: Any, decimals: int = 1, suffix: str = "") -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    if not math.isfinite(value):
        return "—"
    return f"{value:.{decimals}f}{suffix}"


def _fmt_time(seconds: Any) -> str:
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        return "0:00.0"
    if not math.isfinite(seconds) or seconds < 0:
        return "0:00.0"
    mins = int(seconds // 60)
    rem = seconds - mins * 60
    return f"{mins}:{rem:04.1f}"


def _decision_class(decision: str) -> str:
    d = decision.lower()
    if "enhance" in d:
        return "enhanced"
    if "revert" in d or "unverified" in d:
        return "reverted"
    if "error" in d:
        return "error"
    return "bypass"


def _build_timeline_svg(units: list[dict[str, Any]], duration_s: float) -> str:
    if not units or not math.isfinite(duration_s) or duration_s <= 0:
        return ""
    colors = {
        "enhanced": "#2cd4ff",
        "reverted": "#ffb44c",
        "error": "#ff5a5f",
        "bypass": "#4c5666",
    }
    rects: list[str] = []
    for u in units[:2000]:
        t0 = u.get("start_time_s", 0.0)
        t1 = u.get("end_time_s", t0)
        if not isinstance(t0, (int, float)) or not isinstance(t1, (int, float)):
            continue
        if not (math.isfinite(t0) and math.isfinite(t1)) or t1 <= t0:
            continue
        x = max(0.0, min(1000.0, (t0 / duration_s) * 1000.0))
        w = max(1.5, min(1000.0 - x, ((t1 - t0) / duration_s) * 1000.0))
        cls = _decision_class(str(u.get("final_decision") or ""))
        fill = colors.get(cls, "#4c5666")
        rects.append(
            f'<rect x="{x:.1f}" y="2" width="{w:.1f}" height="20" rx="2" fill="{fill}" opacity="0.88" />'
        )
    return (
        '<svg viewBox="0 0 1000 24" preserveAspectRatio="none" class="timeline-svg" '
        'role="img" aria-label="Per-unit guard decision timeline">'
        '<rect x="0" y="0" width="1000" height="24" rx="4" fill="#10151f" />'
        + "".join(rects)
        + "</svg>"
    )


def render_qc_certificate_html(
    report: dict[str, Any],
    *,
    lang: str = "en",
) -> str:
    """Render a self-contained HTML Broadcast Quality Certificate from a report dict."""
    is_ckb = lang == "ckb"
    direction = "rtl" if is_ckb else "ltr"
    html_lang = "ckb" if is_ckb else "en"

    raw_inp = report.get("input")
    inp: dict[str, Any] = raw_inp if isinstance(raw_inp, dict) else {}
    raw_out = report.get("output")
    out: dict[str, Any] = raw_out if isinstance(raw_out, dict) else {}
    raw_summary = report.get("summary")
    summary: dict[str, Any] = raw_summary if isinstance(raw_summary, dict) else {}
    raw_prov = report.get("provenance")
    prov: dict[str, Any] = raw_prov if isinstance(raw_prov, dict) else {}
    raw_units = report.get("units")
    units: list[dict[str, Any]] = (
        [u for u in raw_units if isinstance(u, dict)] if isinstance(raw_units, list) else []
    )

    source_name = _basename(inp.get("path") or inp.get("filename") or report.get("source"))
    master_name = _basename(out.get("path") or out.get("filename") or "master.wav")
    duration_s = float(inp.get("duration_s") or out.get("duration_s") or 0.0)
    if not math.isfinite(duration_s) or duration_s < 0:
        duration_s = 0.0
    sample_rate = inp.get("sample_rate") or out.get("sample_rate") or 48000
    channels = inp.get("channels") or out.get("channels") or 1

    in_lufs = inp.get("integrated_lufs")
    out_lufs = out.get("integrated_lufs")
    in_peak = inp.get("true_peak_dbtp")
    out_peak = out.get("true_peak_dbtp")
    out_lra = out.get("loudness_range_lu")

    total = int(summary.get("units_total") or len(units))
    enhanced = int(summary.get("enhanced") or 0)
    reverted = (
        int(summary.get("reverted") or 0)
        + int(summary.get("continuity_reverted") or 0)
        + int(summary.get("unverified") or 0)
    )
    bypassed = int(summary.get("no_speech") or 0)
    errored = int(summary.get("error_passthrough") or 0)

    profile = str(
        prov.get("profile") or report.get("profile") or prov.get("enhancement_core") or "studio"
    )
    app_version = str(prov.get("version") or report.get("schema_version") or "HawaVoClean")
    created_at = str(report.get("created_at") or prov.get("timestamp_utc") or "")

    peak_ok = (
        not isinstance(out_peak, bool)
        and isinstance(out_peak, (int, float))
        and math.isfinite(out_peak)
        and out_peak <= -0.85
    )
    guard_ok = errored == 0
    badge_text = (
        ("پشتڕاستکراوە · ئامادەی پەخش" if is_ckb else "GUARD VERIFIED · BROADCAST READY")
        if (peak_ok and guard_ok)
        else ("پشکنین تەواوبوو · پێویستی بە بینین هەیە" if is_ckb else "QC COMPLETE · REVIEW NOTES")
    )
    badge_cls = "pass" if (peak_ok and guard_ok) else "warn"

    timeline_svg = _build_timeline_svg(units, duration_s)

    max_rows = 500
    rows_html: list[str] = []
    for u in units[:max_rows]:
        uid = u.get("unit_id", "")
        dec = str(u.get("final_decision") or "")
        cls = _decision_class(dec)
        span = f"{_fmt_time(u.get('start_time_s'))} – {_fmt_time(u.get('end_time_s'))}"
        ga = str(u.get("guard_a_verdict") or "—")
        gb = str(u.get("guard_b_verdict") or "—")
        strength = _fmt_num(u.get("chosen_strength"), 2)
        reason = _esc(u.get("decision_reason") or "")
        rows_html.append(
            f"<tr>"
            f'<td class="mono">#{_esc(uid)}</td>'
            f'<td class="mono" dir="ltr">{_esc(span)}</td>'
            f'<td><span class="pill {cls}">{_esc(dec)}</span></td>'
            f'<td class="mono">{_esc(ga)} / {_esc(gb)}</td>'
            f'<td class="mono">{_esc(strength)}</td>'
            f"<td>{reason}</td>"
            f"</tr>"
        )
    overflow_note = ""
    if len(units) > max_rows:
        overflow_note = (
            f'<p class="muted">Showing first {max_rows} of {len(units)} speech units.</p>'
        )

    title_text = (
        "بڕوانامەی کوالێتی دەنگی HawaVoClean"
        if is_ckb
        else "HawaVoClean Broadcast Quality Certificate"
    )
    subtitle_text = (
        "ڕاپۆرتی پشکنینی خۆکار · پێوانەی ITU-R BS.1770 و پارێزەری گەڕاوە"
        if is_ckb
        else "Offline Fail-Closed Speech Cleanup & Broadcast Loudness Verification"
    )

    return f"""<!DOCTYPE html>
<html lang="{html_lang}" dir="{direction}">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{_esc(title_text)} — {_esc(source_name)}</title>
<style>
  :root {{
    --bg: #080b11; --panel: #10151f; --border: rgba(255,255,255,0.1);
    --fg: #eef3fb; --muted: #93a0b4; --cyan: #2cd4ff; --amber: #ffb44c;
    --ok: #4ade80; --err: #ff5a5f;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 28px; background: var(--bg); color: var(--fg);
    font: 14px/1.5 -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", sans-serif;
  }}
  .wrap {{ max-width: 980px; margin: 0 auto; }}
  .header {{
    display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
    gap: 14px; padding-bottom: 18px; border-bottom: 1px solid var(--border);
  }}
  h1 {{ margin: 0; font-size: 22px; letter-spacing: -0.02em; }}
  .sub {{ margin: 4px 0 0; color: var(--muted); font-size: 13px; }}
  .badge {{
    padding: 6px 12px; border-radius: 999px; font-weight: 600; font-size: 12px;
    letter-spacing: 0.04em; border: 1px solid transparent;
  }}
  .badge.pass {{ background: rgba(74,222,128,0.14); border-color: rgba(74,222,128,0.45); color: #86efac; }}
  .badge.warn {{ background: rgba(255,180,76,0.16); border-color: rgba(255,180,76,0.5); color: #ffd694; }}
  .grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
    gap: 12px; margin: 20px 0;
  }}
  .card {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 12px 14px;
  }}
  .card .k {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }}
  .card .v {{ margin-top: 4px; font-size: 19px; font-weight: 600; font-family: ui-monospace, Menlo, monospace; }}
  .card .d {{ margin-top: 2px; font-size: 12px; color: var(--muted); }}
  .section {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 16px; margin-bottom: 16px;
  }}
  .section h2 {{ margin: 0 0 10px; font-size: 13px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }}
  .meta-list {{ display: flex; flex-wrap: wrap; gap: 16px; font-size: 13px; color: var(--muted); }}
  .meta-list strong {{ color: var(--fg); font-weight: 600; }}
  .timeline-svg {{ width: 100%; height: 24px; display: block; margin: 8px 0 6px; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 14px; font-size: 12px; color: var(--muted); }}
  .dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-inline-end: 5px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
  th, td {{ text-align: start; padding: 7px 8px; border-bottom: 1px solid var(--border); }}
  th {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }}
  .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-variant-numeric: tabular-nums; }}
  .pill {{ display: inline-block; padding: 2px 7px; border-radius: 999px; font-size: 11px; font-weight: 600; }}
  .pill.enhanced {{ background: rgba(44,212,255,0.16); color: #9cecff; }}
  .pill.reverted {{ background: rgba(255,180,76,0.16); color: #ffd694; }}
  .pill.error {{ background: rgba(255,90,95,0.18); color: #ffb3b5; }}
  .pill.bypass {{ background: rgba(255,255,255,0.08); color: #cbd5e1; }}
  .muted {{ color: var(--muted); font-size: 12px; }}
  @media print {{
    body {{ background: #fff; color: #111827; padding: 0; }}
    .card, .section {{ background: #fff; border-color: #d1d5db; }}
    .card .k, .sub, .muted, th, .meta-list, .legend {{ color: #4b5563; }}
    .meta-list strong {{ color: #111827; }}
    .badge.pass {{ background: #ecfdf5; border-color: #10b981; color: #065f46; }}
    .badge.warn {{ background: #fffbeb; border-color: #f59e0b; color: #92400e; }}
    .pill.enhanced {{ background: #e0f2fe; color: #0369a1; }}
    .pill.reverted {{ background: #fef3c7; color: #92400e; }}
    .pill.bypass {{ background: #f3f4f6; color: #374151; }}
  }}
</style>
</head>
<body>
<div class="wrap">
  <header class="header">
    <div>
      <h1>{_esc(title_text)}</h1>
      <p class="sub">{_esc(subtitle_text)}</p>
    </div>
    <span class="badge {badge_cls}">{_esc(badge_text)}</span>
  </header>

  <section class="grid" aria-label="Broadcast metrics">
    <div class="card">
      <div class="k">{"ئاستی دەنگ (Output LUFS)" if is_ckb else "Output Loudness"}</div>
      <div class="v" dir="ltr">{_esc(_fmt_num(out_lufs, 1, " LUFS"))}</div>
      <div class="d" dir="ltr">Input: {_esc(_fmt_num(in_lufs, 1, " LUFS"))}</div>
    </div>
    <div class="card">
      <div class="k">{"لوتکەی ڕاستەقینە (True Peak)" if is_ckb else "True Peak"}</div>
      <div class="v" dir="ltr">{_esc(_fmt_num(out_peak, 2, " dBTP"))}</div>
      <div class="d" dir="ltr">Input: {_esc(_fmt_num(in_peak, 2, " dBTP"))}</div>
    </div>
    <div class="card">
      <div class="k">{"پارچە پاککراوەکان" if is_ckb else "Guard Decisions"}</div>
      <div class="v" dir="ltr">{enhanced} / {total}</div>
      <div class="d">{reverted} reverted · {bypassed} bypassed{
        f" · {errored} err" if errored else ""
    }</div>
    </div>
    <div class="card">
      <div class="k">{"مەودای داینامیک (LRA)" if is_ckb else "Loudness Range"}</div>
      <div class="v" dir="ltr">{_esc(_fmt_num(out_lra, 1, " LU"))}</div>
      <div class="d" dir="ltr">{_esc(_fmt_time(duration_s))} · {_esc(sample_rate)} Hz · {
        _esc(channels)
    }ch</div>
    </div>
  </section>

  <section class="section">
    <h2>{"زانیاری پڕۆسەکردن" if is_ckb else "Provenance & Media"}</h2>
    <div class="meta-list">
      <span>{"سەرچاوە" if is_ckb else "Source"}: <strong>{_esc(source_name)}</strong></span>
      <span>{"ماستەر" if is_ckb else "Master"}: <strong>{_esc(master_name)}</strong></span>
      <span>{"پڕۆفایل" if is_ckb else "Profile"}: <strong class="mono">{
        _esc(profile)
    }</strong></span>
      <span>{"وەشان" if is_ckb else "Engine"}: <strong class="mono">{
        _esc(app_version)
    }</strong></span>
      {
        f'<span>{"کات" if is_ckb else "Generated"}: <strong class="mono" dir="ltr">{_esc(created_at)}</strong></span>'
        if created_at
        else ""
    }
    </div>
  </section>

  {
        f'''<section class="section">
    <h2>{"نەخشەی پارێزەری پارچەکان" if is_ckb else "Per-Unit Guard Timeline"}</h2>
    {timeline_svg}
    <div class="legend">
      <span><i class="dot" style="background:#2cd4ff"></i>Enhanced ({enhanced})</span>
      <span><i class="dot" style="background:#ffb44c"></i>Reverted ({reverted})</span>
      <span><i class="dot" style="background:#4c5666"></i>Bypassed ({bypassed})</span>
    </div>
  </section>'''
        if timeline_svg
        else ""
    }

  <section class="section">
    <h2>{"وردەکاری بڕیارەکان" if is_ckb else "Unit Decision Ledger"}</h2>
    <table>
      <thead>
        <tr>
          <th>Unit</th>
          <th>Time</th>
          <th>Decision</th>
          <th>Guard A / B</th>
          <th>Strength</th>
          <th>Reason</th>
        </tr>
      </thead>
      <tbody>
        {
        "".join(rows_html)
        if rows_html
        else '<tr><td colspan="6" class="muted">No speech units recorded.</td></tr>'
    }
      </tbody>
    </table>
    {overflow_note}
  </section>
</div>
</body>
</html>
"""
