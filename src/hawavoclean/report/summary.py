"""Human-readable TXT review summary generation according to BLUEPRINT.md section 18.3."""

from typing import Any

from hawavoclean.report.schema import HawaVoCleanReport

_INTERVENTION_COST_DESC: dict[str, str] = {
    "preserve": "0 (None / Bit-Exact Passthrough)",
    "production": "1 (Low / Conservative Wiener Filtering)",
    "lowband": "1 (Low / Band-Split DeepFilterNet3)",
    "studio": "2 (Moderate / Full-Band DeepFilterNet3)",
    "lowband_then_production": "2 (Moderate / Lowband DFN3 + Wiener)",
    "restore_source": "3 (High / Generative Source-Conditioned Synthesis)",
    "restore_enrolled": "4 (Maximum / Generative Enrolled-Identity Reconstruction)",
}

_KURDISH_ROUTE_NAMES: dict[str, str] = {
    "preserve": "پاراستن (Preserve)",
    "production": "بەرهەمهێنان (Production)",
    "lowband": "نزم-باند (Lowband)",
    "studio": "ستۆدیۆ (Studio)",
    "lowband_then_production": "نزم-باند پاشان بەرهەمهێنان (Lowband -> Production)",
    "restore_source": "گەڕاندنەوەی سەرچاوە (Restore Source)",
    "restore_enrolled": "گەڕاندنەوەی تۆمارکراو (Restore Enrolled)",
}


def _format_smart_safe_summary(rest: dict[str, Any], lang: str = "en") -> list[str]:
    lines: list[str] = []
    is_ckb = lang == "ckb"
    selected_route = str(rest.get("selected_route") or "preserve")
    route_name = (
        _KURDISH_ROUTE_NAMES.get(selected_route, selected_route)
        if is_ckb
        else selected_route.upper()
    )
    cost_desc = _INTERVENTION_COST_DESC.get(selected_route, "Unknown")
    confidence = float(rest.get("confidence") or 0.0)
    abstained = bool(rest.get("abstained", False))
    reason = str(rest.get("reason") or "No decision rationale recorded")
    decision_digest = str(rest.get("decision_sha256") or "unknown")
    ranker_version = str(rest.get("ranker_version") or "unknown")
    ranker_sha = str(rest.get("ranker_sha256") or "unknown")

    if is_ckb:
        lines.append("--- ڕاپۆرتی بڕیاری زیرەکی پارێزراو (SMART SAFE DECISION REPORT) ---")
        lines.append(f"ڕێگای هەڵبژێردراو:         {route_name}")
        lines.append(f"تێچووی دەستێوەردان:        {cost_desc}")
        lines.append(f"متمانەی هەڵبژاردن:         {confidence * 100:.1f}% ({confidence:.3f})")
        if abstained:
            lines.append(
                "دۆخی پەشیمانبوونەوە:       چالاک کرا (گەڕانەوە بۆ کەمترین دەستێوەردانی پارێزراو)"
            )
            lines.append(f"هۆکاری پەشیمانبوونەوە:     {reason}")
        else:
            lines.append("دۆخی پەشیمانبوونەوە:       ئاسایی (هەموو پشکنین و پارێزەرەکان سەرکەوتن)")
            lines.append(f"هۆکاری بڕیار:              {reason}")
        if selected_route in {"restore_source", "restore_enrolled"}:
            lines.append(
                "ئاشکراکردنی بنیاتنانەوە:   ئاشکراکراو — دەنگەکە بەرهەمهێنانی زیرەکی دەستکردی تێدایە"
            )
        else:
            lines.append(
                "ئاشکراکردنی بنیاتنانەوە:   ئاشکراکراو — دەنگەکە هیچ بنیاتنانەوەیەکی دەستکردی تێدا نییە (تەنها فلتەر)"
            )
        lines.append(f"مۆری بڕیار (SHA-256):      {decision_digest[:16]}...")
        lines.append(f"وەشانی پلەبەندکەر:         {ranker_version} ({ranker_sha[:16]}...)")
        lines.append("")
    else:
        lines.append("--- SMART SAFE DECISION REPORT ---")
        lines.append(f"Selected Route:       {route_name}")
        lines.append(f"Intervention Cost:    {cost_desc}")
        lines.append(f"Selection Confidence: {confidence * 100:.1f}% ({confidence:.3f})")
        if abstained:
            lines.append(
                "Abstention Status:    TRIGGERED (Fallback down least-intervention ladder engaged)"
            )
            lines.append(f"Abstention Reason:    {reason}")
        else:
            lines.append("Abstention Status:    NORMAL (Passed all invariant and quality gates)")
            lines.append(f"Decision Reason:      {reason}")
        if selected_route in {"restore_source", "restore_enrolled"}:
            lines.append(
                "Reconstruction:       DISCLOSED — Generative neural bandwidth extension / acoustic reconstruction applied."
            )
        else:
            lines.append(
                "Reconstruction:       DISCLOSED — Zero generative reconstruction. Bounded classical DSP/filtering only."
            )
        lines.append(f"Decision Digest:      {decision_digest[:16]}...")
        lines.append(f"Ranker Model:         {ranker_version} (SHA-256: {ranker_sha[:16]}...)")
        lines.append("")

    # Acoustic Evidence
    ev = rest.get("acoustic_evidence") or {}
    if ev:
        if is_ckb:
            lines.append("--- بەڵگە و شیکارییە دەنگییەکان (ACOUSTIC EVIDENCE) ---")
            lines.append(
                f"ئەگەری ئاخاوتن:            {float(ev.get('speech_probability', 0.0)) * 100:.1f}%"
            )
            lines.append(
                f"مەترسیی مۆسیقا:            {float(ev.get('music_probability', 0.0)) * 100:.1f}%"
            )
            lines.append(
                f"مەترسیی تێکەڵبوونی دەنگ:   {float(ev.get('crosstalk_probability', 0.0)) * 100:.1f}%"
            )
            lines.append(
                f"لێواری باند (Cutoff):       {float(ev.get('estimated_cutoff_hz', 0.0)):.1f} Hz "
                f"(متمانە: {float(ev.get('cutoff_confidence', 0.0)):.2f})"
            )
            lines.append(
                f"ئاستی ژاوەژاو:             {float(ev.get('noise_floor_db', 0.0)):.1f} dB "
                f"(ڕێژەی هێزی سیگناڵ: {float(ev.get('snr_db', 0.0)):.1f} dB)"
            )
            lines.append(f"کاتژمێری دەنگدانەوە (RT60):  {float(ev.get('rt60_s', 0.0)):.2f} چرکە")
            lines.append(f"ڕێژەی بڕان (Clipping):     {float(ev.get('clipping_ratio', 0.0)):.4f}")
            lines.append(
                f"پەیوەندیی کەناڵەکان:       {float(ev.get('channel_coherence', 0.0)):.2f}"
            )
            lines.append("")
        else:
            lines.append("--- ACOUSTIC EVIDENCE & DETECTIONS ---")
            lines.append(
                f"Speech Probability:   {float(ev.get('speech_probability', 0.0)) * 100:.1f}%"
            )
            lines.append(
                f"Music Risk:           {float(ev.get('music_probability', 0.0)) * 100:.1f}%"
            )
            lines.append(
                f"Crosstalk Risk:       {float(ev.get('crosstalk_probability', 0.0)) * 100:.1f}%"
            )
            lines.append(
                f"Estimated Cutoff:     {float(ev.get('estimated_cutoff_hz', 0.0)):.1f} Hz "
                f"(confidence: {float(ev.get('cutoff_confidence', 0.0)):.2f})"
            )
            lines.append(
                f"Noise Floor:          {float(ev.get('noise_floor_db', 0.0)):.1f} dB "
                f"(Estimated SNR: {float(ev.get('snr_db', 0.0)):.1f} dB)"
            )
            lines.append(f"Reverb RT60:          {float(ev.get('rt60_s', 0.0)):.2f} s")
            lines.append(f"Clipping Ratio:       {float(ev.get('clipping_ratio', 0.0)):.4f}")
            lines.append(f"Channel Coherence:    {float(ev.get('channel_coherence', 0.0)):.2f}")
            lines.append("")

    # Candidates Breakdown
    candidates = rest.get("candidates") or []
    if candidates:
        if is_ckb:
            lines.append("--- هەڵسەنگاندنی بەربژێرەکان (CANDIDATE EVALUATION MATRIX) ---")
        else:
            lines.append("--- CANDIDATE EVALUATION MATRIX ---")
        for c in candidates:
            c_route = str(c.get("route") or "unknown")
            c_name = _KURDISH_ROUTE_NAMES.get(c_route, c_route) if is_ckb else c_route
            eligible = bool(c.get("eligible", False))
            status_str = "ELIGIBLE" if eligible else "REJECTED"
            score = c.get("rank_score")
            score_str = f"{float(score):.3f}" if score is not None else "N/A"
            conf = c.get("confidence")
            conf_str = f"{float(conf):.2f}" if conf is not None else "N/A"
            reasons = c.get("reasons") or (["Passed all hard guards"] if eligible else ["Rejected"])
            reasons_str = "; ".join(str(r) for r in reasons)
            sha = str(c.get("evidence_sha256") or "unknown")[:16]
            cost = _INTERVENTION_COST_DESC.get(c_route, "?").split()[0]
            lines.append(
                f"  [{status_str:8s}] {c_name} (Cost: {cost}) | MOS: {score_str} | Conf: {conf_str} | Evidence: {sha}..."
            )
            lines.append(f"             Reasons: {reasons_str}")
        lines.append("")

    return lines


def generate_human_summary(report: HawaVoCleanReport, *, lang: str = "en") -> str:
    """Generate structured plain-text review summary for audio engineers."""
    lines: list[str] = [
        "================================================================================",
        "                       HAWAVOCLEAN - AUDIT SUMMARY                       ",
        "================================================================================",
        f"Job ID:               {report.job_id}",
        f"Schema Version:       {report.schema_version}",
        *(
            [
                f"Release:              {report.release.version}",
                f"Release Identity:     {report.release.identity_sha256[:16]}...",
            ]
            if report.release is not None
            else ["Release:              legacy schema-v1 report (not recorded)"]
        ),
        f"Config Hash:          {report.config_hash[:16]}...",
        "",
        "--- INPUT MEDIA ---",
        f"Path:                 {report.input.path}",
        f"SHA-256:              {report.input.sha256}",
        f"Sample Rate:          {report.input.sample_rate} Hz",
        f"Channels:             {report.input.channels}",
        f"Samples:              {report.input.samples:,}",
        f"Duration:             {report.input.duration_s:.2f} s",
        "",
        "--- OUTPUT MASTER ---",
        f"Path:                 {report.output.path}",
        f"SHA-256:              {report.output.sha256}",
        f"Sample Rate:          {report.output.sample_rate} Hz",
        f"Channels:             {report.output.channels}",
        f"Samples:              {report.output.samples:,}",
        f"Duration:             {report.output.duration_s:.2f} s",
        f"Integrated Loudness:  {report.output.integrated_lufs:.1f} LUFS"
        if report.output.integrated_lufs is not None
        else "Integrated Loudness:  N/A",
        f"True Peak:            {report.output.true_peak_dbtp:.1f} dBTP"
        if report.output.true_peak_dbtp is not None
        else "True Peak:            N/A",
        "",
        "--- RUNTIME MODELS & CALIBRATION ---",
        f"Enhancement Core:     {report.core.id} ({report.core.algorithm})",
        f"Core Params Hash:     {report.core.params_hash[:16]}...",
        f"Phase Coherent:       {report.core.phase_coherent}",
        f"Fidelity Guard:       {report.guard.id}",
        f"Calibration ID:       {report.guard.calibration_id[:16]}...",
        "",
        "--- PROCESSING STATISTICS ---",
        f"Total Speech Units:   {report.summary.units_total}",
        f"  - Enhanced (PASS):  {report.summary.enhanced} ({report.summary.enhanced / max(1, report.summary.units_total) * 100:.1f}%)",
        f"  - Reverted to Orig: {report.summary.reverted} ({report.summary.reverted / max(1, report.summary.units_total) * 100:.1f}%)",
        f"  - Unverified:       {report.summary.unverified}",
        f"  - Error Fallbacks:  {report.summary.error_passthrough}",
        f"  - Continuity Revert: {report.summary.continuity_reverted}",
        f"  - Continuity Crossfade: {report.summary.continuity_crossfaded}",
        f"  - Non-Speech:       {report.summary.no_speech}",
        f"  - Finish Applied:   {report.summary.finish_applied}",
        f"  - Finish Bypassed:  {report.summary.finish_bypassed}",
        "",
    ]

    if report.restoration:
        rest = report.restoration
        if rest.get("mode") == "smart_safe":
            lines.extend(_format_smart_safe_summary(rest, lang=lang))
        else:
            lines.append("--- SPECTRAL RESTORATION (HawaRestore-KD) ---")
            lines.append(f"Mode:                 {rest.get('mode')}")
            lines.append(f"Speaker ID:           {rest.get('speaker_id')}")
            lines.append(f"Profile Hash:         {str(rest.get('profile_hash'))[:16]}...")
            bw = rest.get("bandwidth", {})
            evidence = bw.get("evidence", {})
            lines.append(
                f"Cutoff Frequency:     {bw.get('effective_cutoff_hz', 0.0):.1f} Hz "
                f"({bw.get('shape', 'unknown')}, confidence {bw.get('confidence', 0.0):.2f}, "
                f"SNR above cutoff {evidence.get('above_cutoff_snr_db', 0.0):.1f} dB)"
            )
            model = rest.get("restorer", {})
            lines.append(
                f"Restoration Model:    {model.get('name')} "
                f"(commit: {str(model.get('commit', ''))[:8]}..., solver: {model.get('solver')})"
            )
            lines.append(f"Weights SHA-256:      {str(model.get('weights_sha256'))[:16]}...")
            segs = rest.get("segments", {})
            lines.append(
                f"Segments:             restored={segs.get('restored')}, reduced={segs.get('reduced')}, "
                f"reverted={segs.get('reverted')}, bypassed={segs.get('bypassed')}, "
                f"errors={segs.get('errors')}"
            )
            guard_r = rest.get("guard_r", {})
            lines.append(
                f"Guard R Verdict:      {guard_r.get('verdict', 'n/a')} "
                f"(accepted strength {guard_r.get('accepted_strength', 0.0):.2f})"
            )
            lines.append(f"Guard R Reason:       {guard_r.get('reason', 'n/a')}")
            lines.append("")

    if len(report.passes) > 1:
        lines.append("--- MULTI-PASS AUDIT TRAIL ---")
        for p in report.passes:
            if p.discarded:
                lines.append(
                    f"  Pass {p.pass_index}: DISCARDED — {p.discard_reason or 'no reason recorded'}"
                )
                continue
            strengths = ", ".join(f"{s:.2f}" for s in p.chosen_strengths) or "none"
            lufs = f"{p.integrated_lufs:.1f} LUFS" if p.integrated_lufs is not None else "N/A"
            lines.append(
                f"  Pass {p.pass_index}: {p.enhanced}/{p.units_total} enhanced, "
                f"{p.reverted} reverted, strengths [{strengths}], "
                f"separation {p.separation_db:.1f} dB, {lufs}"
            )
        lines.append("")

    lines.append("--- FLAGGED REVIEW TIMECODES ---")

    if not report.review_timecodes:
        lines.append("  None. All speech units passed verification cleanly.")
    else:
        for tc in report.review_timecodes:
            lines.append(
                f"  [{tc.start_time_s:07.2f}s - {tc.end_time_s:07.2f}s] (Ch {tc.channel}) [{tc.verdict}] {tc.reason}"
            )

    lines.extend(
        [
            "",
            "================================================================================",
            "Guard scope: the fidelity guard detects SPECTRAL change between the",
            "original and processed audio. It does not verify linguistic content;",
            "flagged timecodes above are where processing was rejected or unverifiable.",
            "================================================================================",
        ]
    )

    return "\n".join(lines) + "\n"
