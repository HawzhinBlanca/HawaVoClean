#!/usr/bin/env python3
"""Mutation gate: prove the test suite can detect real regressions — and prove
it was the *right* test that detected each one.

Applies behaviour-breaking mutations one at a time and requires, for every
mutation, that at least one of its **declared owning tests** fails. A mutation
that leaves its owners green is an untested behaviour: the gate fails and the
missing test must be written.

Why it is built this way
------------------------
The first version of this gate ran ``pytest -x`` and treated *any* non-zero
exit as "caught". That is vacuous in two separate ways, and an audit caught it
doing both at once:

1. ``-x`` stops at the first failure, so the reported failure is very often not
   the mutated behaviour at all — it is simply whatever ran first. ``tests/``
   collects ``chaos`` before ``unit``, so a single unstable chaos test
   short-circuited the run **before the mutated module was ever imported**. The
   audit's run printed "12/12 caught" with 7 of the 12 credited to
   ``tests/chaos/test_interrupt_cleanup.py`` — a test that touches none of the
   mutated code.
2. Any pre-existing failure on the tree — one broken test, one flake — makes
   every mutation "caught" forever. The gate reports a perfect score precisely
   when the suite is at its least trustworthy.

So this version:

* **Establishes a green baseline first.** The whole suite must pass before a
  single mutation is applied. A red baseline aborts loudly; it is not a gate
  result, it is a broken tree, and any score computed on top of it is a lie.
* **Credits a mutation only to its owners.** Each mutation declares the tests
  that exercise the behaviour it breaks. Credit requires one of *those* to
  fail. No unrelated failure — flake, environment, or collateral damage — can
  ever be mistaken for detection.
* **Records which tests failed**, and asserts that set is non-empty and
  disjoint from the baseline's failures and from the documented
  :data:`QUARANTINE` list.
* **Verifies every owner exists** before running anything, so a renamed or
  deleted test is an abort rather than a silent hole.
* **Restores each file from an in-memory copy**, not ``git checkout --``, so a
  concurrent edit elsewhere in the tree can never be destroyed by the gate.

Run from the repo root::

    python scripts/mutation_gate.py           # owner-scoped: fast, exact credit
    python scripts/mutation_gate.py --full    # whole suite per mutation, also
                                              # reports collateral damage
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Tests that must never be allowed to credit a mutation, whatever they do.
#: Each entry is a node-id prefix plus the reason it is here. This list is not
#: an excuse to leave a test broken — it is a statement that a *pass* from this
#: test is not evidence about the code under mutation.
QUARANTINE: dict[str, str] = {
    "tests/chaos/": (
        "Chaos tests drive real subprocesses, signals and the process table. "
        "Their verdict depends on machine state, and none of them exercises a "
        "mutated module's arithmetic — so a failure here is never evidence "
        "that a mutation was detected."
    ),
    "tests/fuzz/": (
        "The fuzz gate is deselected by default (`-m 'not fuzz'`) and runs the "
        "CLI per case; it is a separate gate with its own runtime budget."
    ),
}


@dataclass(frozen=True)
class Mutation:
    """One behaviour-breaking edit, and the tests that must notice it."""

    code: str
    name: str
    rel: str
    old: str
    new: str
    #: pytest node ids (or `file::test` prefixes) that exercise the behaviour
    #: this mutation breaks. At least one of them MUST fail.
    owners: tuple[str, ...] = field(default=())

    @property
    def title(self) -> str:
        return f"{self.code}: {self.name}"


MUTATIONS: list[Mutation] = [
    Mutation(
        "M1",
        "limiter ceiling raised 0.5 dB",
        "src/hawavoclean/finishing/limiter.py",
        "ceiling_linear = float(10.0 ** (ceiling_dbtp / 20.0))",
        "ceiling_linear = float(10.0 ** ((ceiling_dbtp + 0.5) / 20.0))",
        # Owners: the ceiling is the limiter's one hard promise. Discovered by
        # running the whole suite against this mutation: 58 tests broke, and
        # these are the three that are ABOUT the ceiling.
        owners=(
            "tests/unit/test_limiter_loudness.py::test_limiter_enforces_ceiling",
            "tests/property/test_limiter_ceiling.py::test_limiter_enforces_ceiling_without_flat_tops",
            "tests/property/test_properties.py::test_property_limiter_strictly_enforces_ceiling",
        ),
    ),
    Mutation(
        "M2",
        "lookahead anticipation deleted (shift semantics)",
        "src/hawavoclean/finishing/limiter.py",
        """    if lookahead_samples > 0 and samples > 1:
        size = min(lookahead_samples + 1, samples)
        # Look-AHEAD window [i, i+size-1]: origin = -(size//2) is always within
        # scipy's valid range (-(size//2) .. (size-1)//2) for any size parity.
        windowed_min = scipy.ndimage.minimum_filter1d(
            inst_gain, size=size, origin=-(size // 2), mode="nearest"
        )
    else:
        windowed_min = inst_gain
    del peak_envelope, inst_gain  # consumed; free before the next full-size array
    anticipated = _slope_limited_min_envelope(windowed_min, lookahead_samples)""",
        """    windowed_min = inst_gain
    del peak_envelope, inst_gain
    anticipated = windowed_min""",
        # Owner: the ONLY test in the suite that distinguishes a look-ahead
        # envelope from a plain one. If it is ever deleted, this mutation
        # becomes invisible — which is exactly what the gate must refuse.
        owners=(
            "tests/property/test_limiter_ceiling.py::test_gain_reduction_anticipates_the_peak",
        ),
    ),
    Mutation(
        "M3",
        "guard verdict always PASS",
        "src/hawavoclean/guard/verdict.py",
        "verdict = GuardVerdict.PASS if len(reasons) == 0 else GuardVerdict.REVERT",
        "verdict = GuardVerdict.PASS",
        # Owners: the guard is only worth having if it can say no.
        owners=(
            "tests/unit/test_guard_modes.py::test_integrity_mode_still_rejects_broken_audio",
            "tests/unit/test_guard_modes.py::test_anchor_gating_applies_only_in_strict_mode",
        ),
    ),
    Mutation(
        "M4",
        "policy returns enhanced audio on guard REVERT",
        "src/hawavoclean/policy/decision.py",
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
        # Owner: a REVERT verdict must hand back the ORIGINAL waveform.
        owners=(
            "tests/unit/test_hunt_round3_dsp_guard.py::test_empty_candidate_never_passes_the_guard",
        ),
    ),
    Mutation(
        "M5",
        "unit reported enhanced when the enhancer failed",
        "src/hawavoclean/policy/decision.py",
        """                selected_waveform=orig_core_waveform.copy(),
                is_enhanced=False,
                chosen_strength=0.0,
                guard_verdict=GuardVerdict.ERROR,""",
        """                selected_waveform=orig_core_waveform.copy(),
                is_enhanced=True,
                chosen_strength=1.0,
                guard_verdict=GuardVerdict.ERROR,""",
        # Owner: an enhancer crash must fail closed, not be reported enhanced.
        owners=("tests/unit/test_policy_verdict.py::test_policy_enhancer_error_fails_closed",),
    ),
    Mutation(
        "M6",
        "stitch writes every unit one sample late",
        "src/hawavoclean/assembly/stitch.py",
        "        timeline[start:end] = wave",
        "        timeline[start + 1 : min(end + 1, total_samples)] = wave[: min(end + 1, total_samples) - (start + 1)]",
        # Owners: sample-exact placement on the timeline.
        owners=(
            "tests/unit/test_stitch_no_duplication.py::test_two_unit_butt_joint_duplicates_nothing",
            "tests/unit/test_hunt_round3_dsp_guard.py::test_stitch_declick_skips_forced_cut_and_ramps_natural_joint",
        ),
    ),
    Mutation(
        "M7",
        "assembly sample-count invariant disabled",
        "src/hawavoclean/assembly/validate.py",
        """    if samples != expected_samples:
        raise OutputValidationError(
            f"Assembled output samples {samples} != expected input samples {expected_samples}"
        )""",
        """    if False:
        raise OutputValidationError(
            f"Assembled output samples {samples} != expected input samples {expected_samples}"
        )""",
        # Owner: the invariant has exactly one test, and it is this one.
        owners=("tests/unit/test_assembly.py::test_validate_rejects_sample_count_mismatch",),
    ),
    Mutation(
        "M8",
        "ambiguous stereo silently classified dual-mono",
        "src/hawavoclean/audio/channels.py",
        "raise AmbiguousStereoError(",
        "return ChannelMode.DUAL_MONO_SAME\n        raise AmbiguousStereoError(",
        # Owner: ambiguous stereo must raise, never be silently downmixed.
        owners=("tests/unit/test_channels.py::test_channel_mode_ambiguous_stereo_rejected",),
    ),
    Mutation(
        "M9",
        "report and summary never staged (audio published alone)",
        "src/hawavoclean/job.py",
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
        # Owners: `verify` reads the published report back, so it is the test
        # that notices when audio is published without its report and summary.
        owners=(
            "tests/unit/test_cli_surface.py::test_verify_happy_path",
            "tests/unit/test_cli.py::test_cli_verify_success",
        ),
    ),
    Mutation(
        "M10",
        "Wiener gain floor removed",
        "src/hawavoclean/enhancement/production.py",
        '"gain_floor": 0.05,',
        '"gain_floor": 0.0,',
        # Owners: the gain floor is part of the core's locked parameter set, so
        # the lockfile invariant is its first and sharpest detector; the
        # mastering regression is the one that hears the difference.
        # (44 tests broke under this mutation — nearly all of them the
        # lockfile check cascading through every CLI entry point.)
        owners=(
            "tests/unit/test_studio_core.py::test_every_registered_core_has_a_consistent_lockfile",
            "tests/unit/test_mastering_regression.py::test_published_file_hits_loudness_target_and_structure",
        ),
    ),
    Mutation(
        "M11",
        "encoder drops the final sample",
        "src/hawavoclean/audio/encode.py",
        "        sf.write(\n            str(dest_path),\n            interleaved,",
        "        sf.write(\n            str(dest_path),\n            interleaved[:-1],",
        # Owners: encode/probe round-trips that count samples.
        owners=(
            "tests/unit/test_audio_io.py::test_encode_and_probe_float32",
            "tests/unit/test_edge_paths.py::test_encode_float32_subtype",
        ),
    ),
    Mutation(
        "M12",
        "mono loudness target off by 6 dB",
        "src/hawavoclean/resources/configs/production.toml",
        "target_lufs_mono = -19.0",
        "target_lufs_mono = -13.0",
        # Owner: the only test that measures the published file's loudness.
        owners=(
            "tests/unit/test_mastering_regression.py::test_published_file_hits_loudness_target_and_structure",
        ),
    ),
]

_FAILED_LINE = re.compile(r"^(?:FAILED|ERROR)\s+(\S+)")
PYTEST_BASE = [sys.executable, "-m", "pytest", "-q", "--tb=no", "-rf", "-p", "no:cacheprovider"]


def run_pytest(selection: list[str] | None = None, timeout: int = 2400) -> tuple[int, set[str]]:
    """Run pytest and return ``(returncode, failing node ids)``.

    Deliberately NOT ``-x``: the whole point is to know *which* tests failed,
    and a run that stops at the first failure cannot answer that.
    """
    proc = subprocess.run(
        PYTEST_BASE + (selection or []),
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=timeout,
    )
    out = proc.stdout + proc.stderr
    failed = {m.group(1) for line in out.splitlines() if (m := _FAILED_LINE.match(line))}
    return proc.returncode, failed


def collect_ids(selection: list[str]) -> set[str]:
    """Node ids pytest can actually resolve for ``selection`` (empty on error)."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            # `-q -q`: the project's ini adds `-v`, and only a NEGATIVE net
            # verbosity makes --collect-only print one node id per line rather
            # than a tree. A single -q cancels the -v and prints the tree.
            "-q",
            "-q",
            "-p",
            "no:cacheprovider",
            *selection,
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=600,
    )
    if proc.returncode != 0:
        return set()
    ids: set[str] = set()
    for line in proc.stdout.splitlines():
        line = line.strip()
        if "::" in line and not line.startswith(("=", "-", "no tests ran")):
            ids.add(line)
    return ids


def quarantined(node_id: str) -> str | None:
    for prefix, reason in QUARANTINE.items():
        if node_id.startswith(prefix):
            return reason
    return None


def owner_matches(node_id: str, owner: str) -> bool:
    """Whether ``node_id`` is ``owner``, a parametrisation of it
    (``file::test[param]``), or a test inside it when ``owner`` names a file or
    a class."""
    return node_id == owner or node_id.startswith(f"{owner}[") or node_id.startswith(f"{owner}::")


def owned_by(node_id: str, owners: tuple[str, ...]) -> bool:
    return any(owner_matches(node_id, o) for o in owners)


def dirty_paths(paths: list[str]) -> list[str]:
    """Which of ``paths`` git reports as modified (staged or unstaged)."""
    proc = subprocess.run(
        ["git", "status", "--porcelain", "--", *paths],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    return [line[3:].strip() for line in proc.stdout.splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--full",
        action="store_true",
        help="run the WHOLE suite per mutation instead of only its owners; "
        "credit still requires an owner failure, and everything else that "
        "broke is reported as collateral damage.",
    )
    args = ap.parse_args()

    print("=" * 78)
    print("  MUTATION GATE — credit is only ever given to a mutation's own tests")
    print("=" * 78)
    for prefix, reason in QUARANTINE.items():
        print(f"  quarantined (never credits a mutation): {prefix}")
        print(f"      {reason}")

    targets = sorted({m.rel for m in MUTATIONS})

    # ---------------------------------------------------------------- preflight
    dirty = dirty_paths(targets)
    if dirty:
        print("\nABORT: a file this gate mutates has uncommitted changes:")
        for d in dirty:
            print(f"  {d}")
        print("Commit or stash them — the gate rewrites these files in place.")
        return 2

    missing: list[str] = []
    for m in MUTATIONS:
        if not m.owners:
            missing.append(f"{m.code}: no owning tests declared")
            continue
        resolved = collect_ids(list(m.owners))
        for owner in m.owners:
            if not any(owner_matches(r, owner) for r in resolved):
                missing.append(f"{m.code}: owner does not exist — {owner}")
            elif quarantined(owner):
                missing.append(f"{m.code}: owner is quarantined — {owner}")
    if missing:
        print("\nABORT: the mutation-to-test pinning is broken:")
        for line in missing:
            print(f"  {line}")
        print("Every mutation must name tests that exist and are creditable.")
        return 2

    # ---------------------------------------------------------------- baseline
    print("\n[baseline] running the whole suite on the clean tree…")
    rc, baseline_failed = run_pytest()
    if rc != 0 or baseline_failed:
        print(f"\nABORT: the suite is NOT green on the clean tree (pytest rc={rc}).")
        for f in sorted(baseline_failed):
            print(f"  FAILED {f}")
        if not baseline_failed:
            print("  (pytest failed without naming a test — collection or plugin error)")
        print(
            "\nA mutation gate run on a red suite is meaningless: every mutation\n"
            "would be 'caught' by the failure that was already there. Fix the\n"
            "suite, then run the gate."
        )
        return 2
    print("[baseline] green — 0 failures. Mutations may proceed.\n")

    # --------------------------------------------------------------- mutations
    rows: list[tuple[str, str, str]] = []
    caught = 0
    for m in MUTATIONS:
        path = REPO / m.rel
        original = path.read_text()
        if m.old not in original:
            print(f"ABORT: mutation target not found in {m.rel} for {m.code}:\n{m.old[:160]}")
            return 2
        path.write_text(original.replace(m.old, m.new, 1))
        try:
            selection = None if args.full else list(m.owners)
            _, failed = run_pytest(selection)
        finally:
            # Restored from the copy taken above, never `git checkout --`: the
            # gate must not be able to destroy a concurrent edit it did not make.
            path.write_text(original)
        if path.read_text() != original:  # pragma: no cover - filesystem fault
            print(f"ABORT: failed to restore {m.rel}!")
            return 2

        regressions = failed - baseline_failed
        owner_fails = sorted(f for f in regressions if owned_by(f, m.owners))
        collateral = sorted(
            f for f in regressions if not owned_by(f, m.owners) and not quarantined(f)
        )
        credited_flakes = sorted(f for f in regressions if quarantined(f))

        if owner_fails:
            caught += 1
            detail = owner_fails[0].split("::", 1)[-1]
            if len(owner_fails) > 1:
                detail += (
                    f" (+{len(owner_fails) - 1} more owner{'s' if len(owner_fails) > 2 else ''})"
                )
            rows.append(("caught", m.title, detail))
        elif regressions:
            rows.append(
                (
                    "MISSED",
                    m.title,
                    "no owning test failed; only non-owners broke: "
                    + ", ".join((collateral + credited_flakes)[:3]),
                )
            )
        else:
            rows.append(("MISSED", m.title, "suite stayed GREEN — this behaviour is untested"))

        print(f"[{rows[-1][0]:>6}] {m.title}")
        for f in owner_fails:
            print(f"           owner failed: {f}")
        if args.full:
            for f in collateral:
                print(f"           collateral:   {f}")
            for f in credited_flakes:
                print(f"           quarantined:  {f}  (never credited)")

    # ----------------------------------------------------------------- report
    print()
    print("-" * 78)
    for status, title, detail in rows:
        print(f"[{status:>6}] {title} — {detail}")
    print("-" * 78)
    mode = "whole suite" if args.full else "owning tests only"
    print(f"mutation gate ({mode}): {caught}/{len(MUTATIONS)} caught by their own tests")

    still_dirty = dirty_paths(targets)
    if still_dirty:
        print(f"ABORT: tree left dirty after the gate: {still_dirty}")
        return 2
    return 0 if caught == len(MUTATIONS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
