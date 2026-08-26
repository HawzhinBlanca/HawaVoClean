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

1. ~~**GitHub governance (U1/T3.2–T3.3):**~~ **T3.2 and T3.3 closed 2026-08-26.** `main` is protected
   with the contracted ruleset, read back from GitHub and compared field-for-field against
   `evidence/release/github-governance-contract.json`: eleven fields, zero mismatches. The exact
   release gate passed twice — two isolated passes of forty-one steps each, all ten release and
   engineering identities reproduced across them — and `required` reported success for the first time
   in this repository's history. T3.3 is proved in
   `evidence/release/t3.3-branch-protection-proof.json`: a disposable pull request carrying one
   deliberately failing test was refused by GitHub twice, first by `required_linear_history` and then
   with `Required status check "required" is failing`, and `main` never moved.

   Getting there cost five gate executions and each one found something the design had asserted but
   never run. The listener had been started with `nohup ./run.sh &`, leaving SIGINT, SIGQUIT and
   SIGHUP at `SIG_IGN` for every descendant, so ten interrupt-dependent tests could not deliver the
   signal they exist to test — the product was right, and the watchdog escalating to SIGTERM proved
   it. The release SBOM inherited a narrower analysis from the vulnerability scan's Trivy cache and
   lost all ninety-two Wolfi package digests, so its contents depended on cache history rather than
   on the image. The contract pinned the branch-protection context as `release / required`, which
   GitHub never reports; measured with review and admin enforcement switched off, that string left a
   green pull request `BLOCKED` while `required` left it `CLEAN`. And the `release-candidate`
   environment admitted only protected branches, which is incompatible with a gate that triggers on
   `pull_request` — inert while the repository had no protected branch, and locking out every pull
   request the moment `main` gained one, `main` included.

   All four are the same shape: a control that passed because there was nothing for it to check. Only
   executing them exposed it, which is the argument for this checkpoint existing at all.

   Two deviations stand, recorded rather than met. The self-hosted runner is registered under the
   owner's own account on the owner's workstation, alongside `gh` credentials and a working copy,
   where `docs/operations.md` asks for a dedicated unprivileged account; it is ephemeral and
   single-purpose per job, which is the other half of that requirement. And the environment's
   `deployment_branch_policy` is removed rather than `protected_branches_only`; the required reviewer,
   which is the control that was doing the work, is unchanged.

   What remains under U1 is not governance mechanism but people: with `enforce_admins: true` and one
   approving review required, and nobody able to approve their own pull request, `main` now needs a
   second reviewer for every merge. That is the contract behaving as designed — T7 assumes an
   independent challenger and U4 assumes a protected merge — and it is why the pull request carrying
   this very paragraph cannot merge itself.

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
7. ~~**The Resolve engine is built by no hosted job (U1):**~~ **Resolved 2026-08-26.**
   `scripts/build_resolve_engine.py` assembles a shipped surface, and only the self-hosted release
   gate ran it, so a bundle that could not start — or could start and not run — was invisible to
   every hosted job, and therefore to `required`, and therefore to branch protection. The macos-15
   `web-resolve` job now builds it on the platform it ships for and makes it do the job:
   `--version` proves the launcher, `doctor` proves the bundled numeric stack and all four packaged
   profiles and three core locks, and process/verify proves the runtime end to end against the
   generation's own digests. `required` already depends on that job, so this needed no new check
   name and no branch-protection change. Measured locally first: a 711 MB bundle in 21 s, doctor
   clean, and an 8-second fixture processed and verified in under 3 s. Editing
   `.github/workflows/ci.yml` amends an approved design — it is pinned by `ci.workflow_sha256` and
   attested in ledger entry 38 — so both digests are re-pinned and the amendment is recorded in the
   ledger, which is why this belonged with U1 rather than to a passing change.

   The same work found that the engine build command in `docs/operations.md` and
   `resolve-plugin/README.md` had never worked: both prefixed it with
   `SOURCE_DATE_EPOCH="$(git show -s --format=%ct HEAD)"`, and in a Git checkout the backend derives
   both anchors itself and refuses an explicit one supplied without
   `HAWAVOCLEAN_SOURCE_REVISION`. The pair belongs to sdist builds, which have no `.git` to read.

## Historical correction

The 1.0-era repository claimed “production ready,” blueprint compliance, a Sorani ASR guard and zero
false accepts. Those claims were false: the registry and metrics were fabricated and cached reruns
could misreport decisions. The 2026-08-19 rebuild removed those mechanisms. `BLUEPRINT.md` and the
old changelog entries are retained as historical context, not as current product evidence.
