"""Human-readable TXT review summary generation according to BLUEPRINT.md section 18.3."""

from voiceclean.report.schema import VoiceCleanReport


def generate_human_summary(report: VoiceCleanReport) -> str:
    """Generate structured plain-text review summary for audio engineers."""
    lines: list[str] = [
        "================================================================================",
        "                       HAWZHIN VOICECLEAN - AUDIT SUMMARY                       ",
        "================================================================================",
        f"Job ID:               {report.job_id}",
        f"Schema Version:       {report.schema_version}",
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
        f"Enhancement Core:     {report.core.id} (commit {report.core.commit[:8]})",
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
        f"  - Non-Speech:       {report.summary.no_speech}",
        f"  - Finish Applied:   {report.summary.finish_applied}",
        f"  - Finish Bypassed:  {report.summary.finish_bypassed}",
        "",
        "--- FLAGGED REVIEW TIMECODES ---",
    ]

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
            "Linguistic Preservation Invariant: Zero tolerated substitutions/deletions.",
            "================================================================================",
        ]
    )

    return "\n".join(lines) + "\n"
