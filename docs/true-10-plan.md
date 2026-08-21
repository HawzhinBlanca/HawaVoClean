# HawaVoClean: True 10/10 Release Plan

Status: **implementation in progress — Phases 0–2 complete; Phase 3 active; Phase 4 partially closed**
Baseline: `continuity-taper` at `bf6d932`, audited 2026-08-21  
Target: one evidence-backed `v3.3.0` release candidate containing continuity taper, lowband,
crash-safe publication, complete release hardening, real Sorani validation, and in-Resolve proof.

## What “10/10” means

Ten out of ten is an evidence standard, not a claim that software has zero risk. HawaVoClean may be
called 10/10 ship-ready only when all of the following are true on the **same release commit**:

1. No open P0 or P1 defect, no known data-loss path, and no silently accepted critical/high
   vulnerability in an artifact HawaVoClean controls.
2. Every automated release gate passes from a clean checkout locally and in required CI.
3. The exact wheel, UI bundle, Resolve plugin and container/reproducibility artifact intended for
   release are the artifacts that were tested; their hashes are recorded.
4. A speaker-disjoint, held-out Sorani corpus and blinded listening exercise meet declared thresholds
   without a confirmed word/content change.
5. The plugin passes the real DaVinci Resolve 21 workflow matrix on representative media.
6. Every claim in the README, status, architecture, risk register and release notes is generated from
   or linked to reproducible evidence.
7. Any risk owned by Blackmagic, Apple, GitHub or another vendor is explicitly named, bounded by
   compensating controls, and accepted by the user. An unaccepted external high-risk blocker prevents
   a 10/10 declaration.

## Execution rules

- Work in dependency order. A later phase cannot hide or waive an earlier failed gate.
- Start each defect with a failing regression or executable probe whenever technically possible.
- Preserve `main`, `continuity-taper`, and `claude/magical-jackson-ce2782` until the integrated release
  commit is independently verified. Do not delete historical branches or test evidence as cleanup.
- Never weaken a threshold, skip a test, or replace a real dependency with a mock merely to turn a
  gate green. Any justified contract change requires an ADR and before/after measurements.
- Use small commits whose message states the invariant established. Record evidence by commit hash.
- Do not push, alter repository visibility/protection, install with sudo, tag, or publish a release
  without the corresponding user checkpoint below.
- A checked box means its completion proof exists, not merely that code was written.

## Effort model and critical path

Effort bands are planning aids, not delivery promises: **S** ≤ half a focused day, **M** roughly one
to two days, **L** roughly three to five days, and **H** requires substantial human/external work.

```text
P0 publication safety
        ↓
single integrated release tree
        ↓
unified release gate + CI
        ↓
runtime/security/installer hardening
        ↓
Sorani acceptance + real Resolve acceptance
        ↓
reproducible release and final adversarial audit
```

Sorani corpus preparation may progress while code hardening runs, but the held-out set must be locked
before calibration begins and must remain unseen until the release candidate is frozen.

## Phase 0 — Freeze truth and establish the evidence ledger

Goal: every later result is attributable to an exact source and artifact.

- [x] **T0.1 — Create the integration branch and baseline record** (P0, S)
  - Create `codex/v3.3-release` from the audited `continuity-taper` commit.
  - Record `main`, continuity, lowband, tag, lockfile and real-audio reference hashes.
  - Confirm the working tree is clean and preserve all three branches.
  - **Proof:** committed baseline manifest whose tracked hashes reproduce from a clean checkout;
    local real-audio references verify separately on the audited workstation without pretending the
    private recording is part of the repository.
- [x] **T0.2 — Create an append-only evidence log** (P0, S)
  - Record task ID, commit, command/tool version, input artifact, result, output hash and known limits.
  - Failed attempts remain in the log; corrections append rather than rewrite history.
  - **Proof:** schema-validation test plus one entry reproducing the 2026-08-21 baseline.
- [x] **T0.3 — Pin the release contract** (P0, S)
  - Confirm supported Python and operating-system versions, container support, Resolve version, output
    bundle contract and backwards-compatibility policy.
  - Keep guard-reverted units fully original unless the user explicitly changes that safety contract.
  - **Proof:** accepted ADRs with migration and rollback consequences.

**Exit gate:** the release scope and evidence format are immutable enough that every subsequent test
can name exactly what it verified.

## Phase 1 — Eliminate the publication data-loss defect

Goal: a crash or overwrite failure can never destroy the last complete generation or make an
incomplete generation authoritative.

- [x] **T1.1 — Specify truthful publication semantics** (P0, M; depends on T0.3)
  - Write an ADR explicitly rejecting the impossible claim that three independent flat-file renames
    are one atomic filesystem operation.
  - Recommended design: immutable generation directories plus a single atomically replaced commit
    manifest/pointer. Official readers resolve WAV, JSON and TXT from that one committed generation.
  - Preserve a documented compatibility path for callers that request a flat WAV, without treating
    uncommitted aliases as authoritative.
  - Define startup recovery, overwrite, cancellation, cross-filesystem and concurrent-reader behavior.
  - **Proof:** state-machine model enumerating every durable state and recovery transition.
- [x] **T1.2 — Build the transaction with durability barriers** (P0, L; depends on T1.1)
  - Stage on the destination filesystem; flush each artifact; flush directories; validate hashes;
    commit through one atomic pointer; retain the previous generation until commit is durable.
  - Recovery must be idempotent after process crash, `SIGKILL`, power-loss simulation and repeated
    recovery interruption.
  - **Proof:** implementation contains no destructive removal of the prior committed generation before
    the new commit is durable.
- [x] **T1.3 — Add the full publication failure matrix** (P0, L; depends on T1.2)
  - Inject failure before/after every write, flush, rename, directory flush, pointer update and cleanup.
  - Exercise new destination, overwrite, cancellation, disk-full, permission loss and concurrent read.
  - Assert: old generation byte-identical or new generation complete; never mixed; no invisible data
    loss; recovery repeatable; reports hash the exact WAV they describe.
  - Add a mutation owner for transaction ordering and rollback/recovery behavior.
  - **Proof:** failure matrix and mutation gate pass on APFS plus a second supported filesystem/runtime.
- [x] **T1.4 — Migrate CLI, UI, verifier and reports** (P0, M; depends on T1.2)
  - All first-party consumers must resolve only committed generations and reject mismatched artifacts.
  - Old complete triplets remain readable; partial legacy triplets fail explicitly.
  - **Proof:** compatibility fixtures and end-to-end process/download/verify tests.

**Exit gate:** repeated hard-kill and injected-failure runs cannot lose or expose an authoritative
partial output. This gate blocks every merge and release task below.

## Phase 2 — Produce one coherent release tree

Goal: continuity taper, multipass and lowband exist together in one versioned source tree.

- [x] **T2.1 — Integrate lowband semantically** (P0, L; depends on Phase 1)
  - Reconcile `CHANGELOG.md`, mutation ownership and `studio.py` conflicts deliberately; do not choose
    conflict sides mechanically.
  - Register lowband in factory, CLI, server, UI, schemas, docs and provenance locks.
  - Preserve full-band DFN3 inference with the verified 1 kHz crossover behavior.
  - **Proof:** profile-surface tests and real lowband regression hashes pass beside continuity tests.
- [x] **T2.2 — Unify version and artifact identity** (P0, M; depends on T2.1)
  - Make one source authoritative for Python, UI, plugin, report and release versions.
  - Advance report schema compatibly; reject fabricated or internally inconsistent build identity.
  - **Proof:** test that all reported/package versions derive from the same release identity.
- [x] **T2.3 — Run semantic regression comparisons** (P0, M; depends on T2.1)
  - Compare production, studio and lowband against frozen pre-integration references.
  - Explain every changed hash with an intended code path and measured audio/report difference.
  - **Proof:** zero unexplained drift and deterministic repetition on the chosen release platform.

**Exit gate:** one clean commit contains all intended cores and safety work; no release feature exists
only in another worktree or unmerged branch.

## Phase 3 — Make one command mean “release candidate is valid”

Goal: eliminate the gap between locally green subsets and the release claim.

- [ ] **T3.1 — Build a hermetic full release gate** (P0, L; depends on Phase 2)
  - One non-mutating command runs formatting, lint, strict types, default tests, branch coverage, fuzz,
    mutation, UI type/build/tests, wheel/sdist install smoke, CLI E2E, real regression fixtures, plugin
    self-test, container build/run if supported, SBOM validation, audits and documentation consistency.
  - Pin tool versions and record durations/hashes. Fail on dirty tree or generated-file drift.
  - Coverage may not regress below the audited 92.49%; changed safety-critical lines require direct
    tests. Mutation remains 100% caught by declared owners.
  - **Proof:** two consecutive clean-checkout runs produce the same artifact hashes where determinism
    is promised.
- [ ] **T3.2 — Expand GitHub Actions** (P0, L; depends on T3.1 and user checkpoint U1)
  - Run the same gate components in CI, not a weaker handwritten approximation.
  - Cover the declared Python/OS support matrix, with macOS required for the Resolve-facing package.
  - Pin actions by immutable commit; use least permissions; upload immutable evidence artifacts.
  - **Proof:** required checks pass on the exact release-candidate commit.
- [ ] **T3.3 — Protect release governance** (P0, S; depends on T3.2)
  - Protect `main`; require review and all release checks; block force pushes and tag movement.
  - **Proof:** an intentionally failing test prevents merge in a disposable PR.

**Exit gate:** “green” has one definition locally and remotely, and it covers every shipped surface.

## Phase 4 — Close runtime, security and operational gaps

Goal: every install is reproducible, bounded, recoverable and truthfully described.

- [x] **T4.1 — Repair or retire the Docker contract** (P1, M)
  - Decide whether the release supports CPU-only, studio/GPU, or both; remove unsupported claims.
  - Use available pinned bases, install the matching extras, run non-root with a read-only-friendly
    filesystem design, health check and explicit cache/work mounts.
  - **Proof:** clean build, vulnerability/misconfiguration scan and representative process/verify run.
- [ ] **T4.2 — Emit a real multi-ecosystem SBOM** (P1, M)
  - Produce validated CycloneDX JSON for Python, UI/plugin JavaScript, model weights and system/runtime
    artifacts; include licenses, hashes and relationships.
  - Sign/checksum the SBOM and bind it to release artifact hashes.
  - **Proof:** schema validation and independent inventory spot-checks.
- [x] **T4.3 — Complete report provenance** (P1, M)
  - Record tool/release version, source/build ID, lock digest, model/core/guard hashes, Python and all
    relevant DSP/neural/runtime versions, device and deterministic settings.
  - Retain compatibility readers for schema v1.
  - **Proof:** a fresh installed wheel produces a self-verifiable schema-v2 report; tampering fails.
- [x] **T4.4 — Bound server state and uploaded data** (P1, M)
  - Add queue limits, terminal-job TTL/count retention, upload quotas/TTL, cleanup on success/failure,
    startup scavenging and observable disk-pressure errors.
  - Never delete a committed user output as retention cleanup.
  - **Proof:** fake-clock, restart and disk-pressure tests demonstrate bounded growth and safe cleanup.
- [x] **T4.5 — Make the Resolve installer transactional and self-contained** (P1, L)
  - No mutable lock fallback; assemble from exact locked inputs; self-test before install.
  - Install to a staging target, back up the prior plugin, atomically activate, verify, and roll back on
    failure. Do not hardcode the repository virtual environment as the production engine.
  - **Proof:** install/upgrade/failure/rollback tests in a disposable plugin directory, then real host.
- [ ] **T4.6 — Resolve Electron and host-boundary risk** (P1, M)
  - Separate the standalone test-runtime advisories from Resolve's vendor-owned Electron runtime.
  - Upgrade controlled dependencies; deny unexpected permissions; keep sandbox/context isolation;
    validate CSP, loopback-only networking, navigation/popups and preload surface.
  - If Resolve still embeds a vulnerable major, document the exact advisory exposure and compensating
    controls for explicit user acceptance; never report the dependency estate as clean.
  - **Proof:** full lock audit, source-boundary tests and host-version evidence.

**Exit gate:** no controllable high/critical finding, no unbounded storage path, no non-reproducible
installer fallback, and no misleading SBOM/provenance claim.

## Phase 5 — Establish real Sorani quality and content-integrity evidence

Goal: replace four synthetic files and one-recording calibration with a locked, human-verified product
evaluation that can actually falsify the quality promise.

- [ ] **T5.1 — Write the evaluation protocol before selecting results** (P0, M)
  - Declare primary outcomes: confirmed word/content change, guard false accept/revert, intelligibility,
    artifact severity, speech/noise separation and blinded preference.
  - Predefine exclusion, adjudication, stopping and regression rules. ASR may assist triage but is not
    the linguistic oracle.
  - **Proof:** versioned protocol and sample-size rationale approved before held-out evaluation.
- [ ] **T5.2 — Build licensed, speaker-disjoint splits** (P0, H; depends on T5.1 and U3)
  - Use real Sorani across dialect, gender, age range where available, microphone, room, reverb, hum,
    fan/street noise, codec, clipping risk and music bleed.
  - Minimum statistical target: at least 300 independently reviewed held-out speech units, enough that
    zero observed content changes gives an approximate one-sided 95% upper bound near 1%; increase
    the sample if independence/stratification reduces effective size.
  - Keep calibration and acceptance speakers disjoint; hash-lock the held-out manifest.
  - **Proof:** validated manifest, provenance/licensing classification, transcripts and split-leak test.
- [ ] **T5.3 — Dual-review content integrity** (P0, H; depends on T5.2)
  - Two Sorani-capable reviewers independently compare source and each candidate; disagreements are
    adjudicated without revealing the profile until after verdict lock.
  - Acceptance requires zero confirmed introduced/deleted/substituted words in shipped candidates.
  - **Proof:** anonymized verdict ledger with inter-reviewer agreement and adjudication trail.
- [ ] **T5.4 — Calibrate only on the calibration split** (P0, L; depends on T5.2)
  - Measure thresholds, strength selection and guard operating points without examining held-out labels.
  - Preserve failure examples and report uncertainty, not just aggregate averages.
  - **Proof:** reproducible calibration artifact whose provenance names the exact split and code commit.
- [ ] **T5.5 — Run locked held-out and blinded listening gates** (P0, H; depends on T5.3–T5.4)
  - Evaluate production, studio and lowband against original and appropriate baselines.
  - Require no content-integrity failure, no objective safety regression, and a statistically supported
    improvement or non-inferiority on declared listening outcomes per intended profile.
  - Report strata and worst cases; an aggregate score cannot hide a failing dialect/noise condition.
  - **Proof:** signed result artifact, analysis script and reproducible plots/tables.

**Exit gate:** the strongest product promise is backed by locked held-out human evidence, with known
limits stated. Any confirmed content alteration is a release blocker until understood and corrected.

## Phase 6 — Prove the actual DaVinci Resolve product

Goal: the tested experience is the host workflow users will run, not just a standalone Electron shell.

- [ ] **T6.1 — Install through the hardened installer** (P0, M; depends on T4.5 and U2)
  - Capture installed artifact hashes, permissions, engine path and Resolve/Electron versions.
  - Verify upgrade and rollback from a prior plugin installation.
- [ ] **T6.2 — Execute the Resolve workflow matrix** (P0, H; depends on T6.1)
  - Mono/stereo, short/long, production/studio/lowband, clips with handles, non-48 kHz media, Unicode
    paths, missing media, cancel, engine crash, Resolve restart and project reopen.
  - Verify select → process → review A/B → import → append/replace → report access and timeline sync.
  - Assert no orphan engine, duplicate media, timing shift, partial publication or silent failure.
  - **Proof:** host logs, report/output hashes, screenshots/video and a completed matrix.
- [ ] **T6.3 — Human UX, keyboard and VoiceOver pass** (P1, H; depends on T6.1)
  - Run the actual screen reader, keyboard-only navigation and constrained-window layouts inside Resolve.
  - Correct modal background exposure and any host-specific focus behavior.
  - **Proof:** recorded VoiceOver/keyboard checklist with no unresolved severe accessibility defect.

**Exit gate:** the user signs off on real timeline behavior and listening quality inside Resolve.

## Phase 7 — Release truth, reproducibility and final challenge

Goal: publish exactly what was proved, with no stale or inflated claim.

- [ ] **T7.1 — Generate truthful documentation** (P1, M)
  - Rewrite README, STATUS, architecture, model provenance, operations, risks and changelog from the
    release evidence. Remove v1 language, stale counts and unsupported “atomic”/quality claims.
  - Generate volatile counts/versions where practical and test commands/examples.
  - **Proof:** documentation consistency tests and a fresh-user walkthrough from the README.
- [ ] **T7.2 — Assemble and reproduce the candidate** (P0, L)
  - Build wheel/sdist, UI, plugin, supported container and SBOM twice from clean checkouts.
  - Compare deterministic artifacts; explain and normalize permitted metadata differences.
  - Sign artifacts/checksums and bind every item to the source commit and evidence log.
  - **Proof:** reproducibility report and successful install/process/verify from release artifacts only.
- [ ] **T7.3 — Independent adversarial release audit** (P0, L)
  - Attack publication, interruption, overwrite, provenance, security boundary, profile routing, corpus
    leakage, installer rollback, long-run resources and documentation claims.
  - Every finding is reproduced or rejected with executable evidence; all P0/P1 findings close before
    release.
  - **Proof:** final finding register with no open P0/P1 and all gates rerun after the final fix.
- [ ] **T7.4 — Protected merge, tag and GitHub release** (P0, S; depends on U4)
  - Merge the exact audited commit, create an immutable signed/annotated tag, publish tested artifacts,
    SBOM, checksums, limitations and recovery instructions.
  - **Proof:** release asset hashes match T7.2 and required checks remain green on the tag.

## Required user checkpoints

| ID | When | User action/decision | Default recommendation |
|---|---|---|---|
| U1 | Before T3.2 | Fix GitHub billing or authorize a visibility change | Fix billing; keep the private-repo boundary |
| U2 | Before T6.1 | Approve and authenticate the sudo plugin install | Install only the hash-recorded hardened candidate |
| U3 | Before T5.2 | Approve corpus sources/licensing and identify Sorani reviewers | Use only sources with recorded rights; two reviewers |
| U4 | Before T7.4 | Final go/no-go after listening, Resolve and risk review | Release only with no open P0/P1 |
| U5 | During T0.3/T5 | Decide whether reverted units receive finishing | Keep `REVERT = original` until human evidence supports a change |

## Final release gate

The release candidate is blocked unless all rows are green:

| Gate | Required evidence |
|---|---|
| Publication safety | Full injected-failure matrix, hard-kill recovery and overwrite preservation |
| Source topology | One commit contains continuity, multipass and lowband; no release-only worktree |
| Python | Format/lint/mypy, default/fuzz/property/chaos/integration tests, ≥92.49% branch coverage |
| Mutation | 100% of declared mutations caught by owning tests |
| UI | Typecheck, build, 342+ tests, browser smoke, performance and accessibility checks |
| Packaging | Fresh wheel/sdist install, doctor, process and verify from outside the repository |
| Security | Secret/source/dependency/image scans; no controllable critical/high finding |
| Supply chain | Valid signed CycloneDX SBOM, lock/model/artifact hashes and provenance schema v2 |
| Operations | Bounded jobs/uploads, crash recovery, disk-pressure and cleanup tests |
| Docker | Build/run/scan for every supported image, or explicit removal from support contract |
| Resolve | Hardened install/upgrade/rollback plus completed real-host workflow matrix |
| Product | Locked speaker-disjoint Sorani acceptance and blinded listening thresholds met |
| Documentation | Commands and claims verified against the exact release artifacts |
| Governance | Required CI green, protected merge, immutable tag and user go/no-go |

## Stop conditions

Stop and report rather than paper over the result if any of these occurs:

- a confirmed content change, old-output loss, mixed-generation publication or unexplained audio drift;
- a gate can pass without exercising the behavior it claims to own;
- corpus leakage or reviewer unblinding invalidates the held-out evaluation;
- Resolve/vendor runtime leaves a high risk that cannot be bounded or explicitly accepted;
- the tested artifact differs from the artifact proposed for release.

The project reaches true 10/10 only after the final adversarial audit survives without reopening a
P0/P1 item. Until then, progress is reported by completed task IDs and evidence—not by a rounded score.
