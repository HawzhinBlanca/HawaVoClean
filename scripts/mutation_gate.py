#!/usr/bin/env python3
"""Mutation gate: prove the test suite can detect real regressions.

Applies 12 behavior-breaking mutations one at a time, runs the full suite
against each, and requires at least one test to fail for every mutation.
A mutation that leaves the suite green is an untested behavior — the gate
fails and the missing test must be written.

Run from the repo root: python scripts/mutation_gate.py
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

MUTATIONS: list[tuple[str, str, str, str]] = [
    (
        "M1: limiter ceiling raised 0.5 dB",
        "src/voiceclean/finishing/limiter.py",
        'ceiling_linear = float(10.0 ** (ceiling_dbtp / 20.0))',
        'ceiling_linear = float(10.0 ** ((ceiling_dbtp + 0.5) / 20.0))',
    ),
    (
        "M2: lookahead anticipation deleted (shift semantics)",
        "src/voiceclean/finishing/limiter.py",
        """    if lookahead_samples > 0:
        size = lookahead_samples + 1
        windowed_min = scipy.ndimage.minimum_filter1d(
            inst_gain, size=size, origin=size // 2, mode="nearest"
        )
    else:
        windowed_min = inst_gain
    anticipated = _slope_limited_min_envelope(windowed_min, lookahead_samples)""",
        """    windowed_min = inst_gain
    anticipated = windowed_min""",
    ),
    (
        "M3: guard verdict always PASS",
        "src/voiceclean/guard/verdict.py",
        "verdict = GuardVerdict.PASS if len(reasons) == 0 else GuardVerdict.REVERT",
        "verdict = GuardVerdict.PASS",
    ),
    (
        "M4: policy returns enhanced audio on guard REVERT",
        "src/voiceclean/policy/decision.py",
        """    return (
        UnitPolicyDecision(
            selected_waveform=orig_core_waveform.copy(),
            is_enhanced=False,
            chosen_strength=0.0,
            guard_verdict=final_verdict,
            guard_scores=final_scores,
            decision_reason=f"Reverted to original audio: {failure_reasons}",
        ),
        orig_probe,
    )""",
        """    return (
        UnitPolicyDecision(
            selected_waveform=candidates[0][1],
            is_enhanced=True,
            chosen_strength=candidates[0][0],
            guard_verdict=final_verdict,
            guard_scores=final_scores,
            decision_reason=f"Reverted to original audio: {failure_reasons}",
        ),
        orig_probe,
    )""",
    ),
    (
        "M5: unit reported enhanced when the enhancer failed",
        "src/voiceclean/policy/decision.py",
        """                selected_waveform=orig_core_waveform.copy(),
                is_enhanced=False,
                chosen_strength=0.0,
                guard_verdict=GuardVerdict.ERROR,""",
        """                selected_waveform=orig_core_waveform.copy(),
                is_enhanced=True,
                chosen_strength=1.0,
                guard_verdict=GuardVerdict.ERROR,""",
    ),
    (
        "M6: stitch writes every unit one sample late",
        "src/voiceclean/assembly/stitch.py",
        "        timeline[start:end] = wave",
        "        timeline[start + 1 : min(end + 1, total_samples)] = wave[: min(end + 1, total_samples) - (start + 1)]",
    ),
    (
        "M7: assembly sample-count invariant disabled",
        "src/voiceclean/assembly/validate.py",
        """    if samples != expected_samples:
        raise OutputValidationError(
            f"Assembled output samples {samples} != expected input samples {expected_samples}"
        )""",
        """    if False:
        raise OutputValidationError(
            f"Assembled output samples {samples} != expected input samples {expected_samples}"
        )""",
    ),
    (
        "M8: ambiguous stereo silently classified dual-mono",
        "src/voiceclean/audio/channels.py",
        None,  # filled dynamically below
        None,
    ),
    (
        "M9: report and summary never staged (audio published alone)",
        "src/voiceclean/job.py",
        """            shutil.copyfile(temp_audio_path, staged_audio)
            for staged, content in ((staged_json, json_report_str), (staged_txt, txt_summary_str)):
                with open(staged, "w", encoding="utf-8") as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())

            renamed: list[tuple[Path, Path]] = []
            try:
                for staged, dest in (
                    (staged_audio, dest_audio),
                    (staged_json, dest_json),
                    (staged_txt, dest_txt),
                ):""",
        """            shutil.copyfile(temp_audio_path, staged_audio)

            renamed: list[tuple[Path, Path]] = []
            try:
                for staged, dest in (
                    (staged_audio, dest_audio),
                ):""",
    ),
    (
        "M10: Wiener gain floor removed",
        "src/voiceclean/enhancement/production.py",
        '"gain_floor": 0.05,',
        '"gain_floor": 0.0,',
    ),
    (
        "M11: encoder drops the final sample",
        "src/voiceclean/audio/encode.py",
        None,  # filled dynamically below
        None,
    ),
    (
        "M12: mono loudness target off by 6 dB",
        "src/voiceclean/resources/configs/production.toml",
        "target_lufs_mono = -19.0",
        "target_lufs_mono = -13.0",
    ),
]


def fill_dynamic_mutations() -> None:
    channels = (REPO / "src/voiceclean/audio/channels.py").read_text()
    # Find the ambiguous branch and neutralize it
    if 'raise AmbiguousStereoError(' in channels:
        MUTATIONS[7] = (
            MUTATIONS[7][0],
            "src/voiceclean/audio/channels.py",
            "raise AmbiguousStereoError(",
            'return ChannelMode.DUAL_MONO_SAME\n        raise AmbiguousStereoError(',
        )
    MUTATIONS[10] = (
        MUTATIONS[10][0],
        "src/voiceclean/audio/encode.py",
        "        sf.write(\n            str(dest_path),\n            interleaved,",
        "        sf.write(\n            str(dest_path),\n            interleaved[:-1],",
    )


def apply(path: Path, old: str, new: str) -> None:
    s = path.read_text()
    if old not in s:
        raise SystemExit(f"MUTATION TARGET NOT FOUND in {path}:\n{old[:120]}")
    path.write_text(s.replace(old, new, 1))


def run_suite() -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=1200,
    )
    out = proc.stdout + proc.stderr
    first_fail = ""
    for line in out.splitlines():
        if line.startswith("FAILED") or line.startswith("ERROR"):
            first_fail = line.strip()
            break
    return proc.returncode == 0, first_fail


def git_clean() -> bool:
    proc = subprocess.run(
        ["git", "diff", "--stat"], capture_output=True, text=True, cwd=REPO
    )
    return proc.stdout.strip() == ""


def main() -> int:
    fill_dynamic_mutations()
    if not git_clean():
        print("ABORT: working tree is dirty; commit or stash before the mutation gate.")
        return 2

    caught = 0
    results: list[str] = []
    for name, rel, old, new in MUTATIONS:
        path = REPO / rel
        apply(path, old, new)
        try:
            green, first_fail = run_suite()
        finally:
            subprocess.run(["git", "checkout", "--", rel], cwd=REPO, check=True)
        if green:
            results.append(f"[MISSED] {name} — suite stayed GREEN")
        else:
            caught += 1
            results.append(f"[caught] {name} — {first_fail or 'suite failed'}")

    print()
    for r in results:
        print(r)
    print()
    print(f"mutation gate: {caught}/{len(MUTATIONS)} caught")
    if not git_clean():
        print("ABORT: tree left dirty after gate!")
        return 2
    return 0 if caught == len(MUTATIONS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
