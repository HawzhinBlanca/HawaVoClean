# Implementation status

HawaVoClean 3.3.0 is an evidence-backed **release candidate in progress**, not a finished release.
The core code, crash-safe publication, integrated profiles, local release gate and controllable
runtime hardening are strong. The project must not be called 10/10 or production-released until the
remaining human, host, governance and vendor-risk gates close.

The exact volatile counts, proof commits, approval states and advisory totals are generated from
tracked evidence in [the release-status snapshot](docs/generated-release-status.md). The release
gate fails if that file drifts from its sources.

## What is implemented

- Three selectable profiles: classical production Wiener DSP, full-band studio DFN3/WPE restoration,
  and DFN3-below-1-kHz lowband restoration with the original high band preserved.
- Per-speech-unit fail-closed selection, two guards, sample-exact assembly, continuity taper,
  deterministic finishing, BS.1770 loudness and 8×-oversampled true-peak limiting.
- Immutable content-addressed output generations committed through one authoritative `current`
  pointer. The previous complete generation survives failed overwrite and crash recovery.
- Schema-v2 provenance, deterministic CycloneDX 1.6 SBOM, bounded loopback server storage/queues,
  a CPU-only non-root read-only container, and a transactional self-contained Resolve installer.
- One two-pass local release gate covering Python, fuzz, mutations, UI, packaging, real engineering
  audio, Resolve staging, container scanning, SBOM and reproducible artifact identities.
- Result-free Sorani protocol and corpus-source designs that are machine-valid but deliberately
  refuse approval-dependent execution.
- An opt-in generative restore mode (`--mode restore`) behind Restoration Guard R: protected-band
  invariance by construction, fail-closed revert with the rejecting layer's evidence, CPU-pinned
  deterministic inference, and a mandatory self-attested checkpoint. Its engineering behavior is
  tested and mutation-covered; its *quality* evidence is currently synthetic-only — the committed
  checkpoint has never seen real speech, the ten speaker profiles are generated fixtures, and the
  committed benchmark records restoration degrading LSD on those fixtures. Real-corpus training,
  real speaker enrollment, and the four-condition human protocol remain open (R-14, T5.6–T5.8).

## What the strongest proof currently establishes

The latest full proof is bound to source commit `13d43a7`, not automatically to later evidence-summary
or final-release commits. It passed twice from detached clean checkouts with 1,018 default tests, one
skip, 41 separate fuzz cases, 23/23 declared mutations, 342 UI tests and 92.73% branch coverage in
each pass. All ten promised release/engineering identities reproduced. The final candidate must still
rerun this complete gate after every human, host and governance blocker closes.

The proof is engineering evidence, not Sorani product validation. Private real speech regressions and
tracked synthetic fixtures can detect drift; they cannot establish population-level content safety,
listening quality or dialect coverage.

## Real-audio engineering measurements

The measurements below are frozen engineering references, not corpus averages.

### Lowband reference (24 seconds)

| Chain | Speech/floor separation | Pause rumble | Guard result |
|---|---:|---:|---|
| Source | 15.1 dB | −32.3 dB | — |
| Production | 19.8 dB | −34.6 dB | Enhanced |
| Studio | 15.1 dB | −30.5 dB | All speech units reverted |
| Lowband | 29.4 dB | −71.6 dB | Enhanced; hole 0.066; consonants 0.999 |
| Lowband → production | 35.2 dB | −83.3 dB | Enhanced in both runs |

The earlier 40.0 dB prototype number came from an unguarded, unmastered intermediate. The guarded
product chain's 35.2 dB is the valid comparison.

### Continuity reference (94.6 seconds)

The production continuity taper preserves five of six guard-passing units instead of cascading one
failure across the file. Speech/floor separation improved by 7.23 dB over the old fixed-point revert
behavior. Studio and unrelated lowband outputs remained unchanged except for the separately explained
version-seeded PCM24 dither identity.

## Open release blockers

1. **GitHub governance (U1/T3.2–T3.3):** the billing blocker is gone — the repository was made public
   on 2026-08-24, hosted Actions now run, and four pull requests have taken the full hosted matrix
   green. The two owner-only decisions are also closed: the ephemeral `hawavoclean-release` runner is
   registered, and `HAWAVOCLEAN_RELEASE_EVIDENCE_ROOT` was set on 2026-08-26 to
   `/Users/hawzhin/HawaVoCleanEvidence`, whose fourteen manifest-named files hydrated into a hosted
   job for the first time in the repository's history. What remains is one green
   `release / required`, and the two things queued behind it: the `main` ruleset, and T3.3's
   deliberately-failing-pull-request proof.

   The gate's second execution failed, and on nothing in the product: the listener had been started
   with `nohup ./run.sh &`, which leaves SIGINT, SIGQUIT and SIGHUP at `SIG_IGN` for every
   descendant, so ten interrupt-dependent tests could not deliver the signal they exist to test. One
   of those failures is itself the proof the product was right — the watchdog detected the ignored
   disposition and escalated to SIGTERM, exactly as
   `test_watchdog_escalates_to_sigterm_when_sigint_is_inherited_ignored` asserts it must. The
   listener is now started through `start-ephemeral-runner.sh`, which restores the three dispositions
   before exec; the ten tests fail and pass on the same machine according to that one difference, and
   no test was changed. See `docs/u1-governance-runbook.md`.

   Applying the ruleset is one-way for a single-maintainer repository: it requires one approving
   review with `enforce_admins: true`, and nobody may approve their own pull request. Everything that
   still needs to land — blocker 7 below — must land before it.
2. **Vendor Electron (T4.6):** Resolve 21.0.3 embeds Electron 36.3.2 with 33 captured advisories,
   including seven high. HawaVoClean hardens reachable boundaries, but the residual vendor risk needs
   explicit acceptance or a qualifying Resolve update.
3. **Sorani evidence (U3/T5):** protocol `896dfc12…` and source design `1f46b23e…` are unapproved.
   No licensed held-out split, dual-review verdict ledger or listening result exists.
4. **Real Resolve product (U2/T6):** the transactional installer and staged lifecycle are proved, but
   the plugin has not completed the real in-host workflow, timeline, keyboard or VoiceOver matrix.
5. **Final release (T7.2–T7.4/U4):** the eventual source commit must be rebuilt twice, challenged,
   signed, merged through protection, tagged and published with matching hashes.
6. ~~**Audio-regression references encode a report bug (U1):**~~ **Resolved 2026-08-25.** The
   references held `input.integrated_lufs` and `input.true_peak_dbtp` measured on the pre-master
   buffer — the audio *after* enhancement — while every other field in that block described the
   source. The proof was that three profiles reported three different input loudnesses for one file
   (−24.107 / −24.671 / −24.400 LUFS); a source file has one loudness, and the value tracked the
   profile. Regenerated under v3.3: all six cases now report a single consistent figure per input
   (−24.887 / −1.195 for Flute, −31.841 / −17.492 for teat1vo), matching what this entry predicted
   to the digit. Every regenerated master reproduced the `candidate_audio_sha256` the manifest had
   already predicted, so the v3.2→v3.3 dither-seed transition is complete and the drift contract is
   now historical rather than active — `scripts/audio_regression_gate.py` passes with **0 changed
   samples and 0.0 max LSB** across all six cases, two runs each, where it previously had to tolerate
   up to 2.0 LSB.
7. **The Resolve engine is built by no hosted job (U1):** `scripts/build_resolve_engine.py` assembles
   a shipped surface, and only the self-hosted release gate runs it. It installs the wheel without
   the optional extras, so while `import hawavoclean.cli` briefly required torch the engine could not
   have started either, and no hosted job would have said so — the wheel smoke in `core` caught the
   same defect for the CLI. Verified by hand on this branch: the engine builds, and
   `hawavoclean-engine --version` and `doctor` both succeed with torch absent. The fix is a step in
   the existing macos-15 `web-resolve` job, which `required` already depends on, so it needs no new
   check name and no branch-protection change — but `.github/workflows/ci.yml` is pinned by
   `ci.workflow_sha256` in the governance contract and attested in ledger entry 38, so editing it
   amends an approved design and belongs with U1 rather than to a passing change.

## Historical correction

The 1.0-era repository claimed “production ready,” blueprint compliance, a Sorani ASR guard and zero
false accepts. Those claims were false: the registry and metrics were fabricated and cached reruns
could misreport decisions. The 2026-08-19 rebuild removed those mechanisms. `BLUEPRINT.md` and the
old changelog entries are retained as historical context, not as current product evidence.
