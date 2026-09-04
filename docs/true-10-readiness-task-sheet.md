# HawaVoClean true-10 readiness task sheet

Status: **release blocked — execution ledger, not a readiness claim**  
Audit date: **2026-09-02**  
Audited source: branch `codex/high-end-production-foundation`, commit
`51bfbbc4238a544031e25e72606672d8e8fe1efa`  
Target: one offline-first engine serving signed macOS and Windows apps, a thin macOS Resolve client,
qualified Smart Safe and source/enrolled Restore, plus optional invite-only UAE cloud acceleration.

This is the canonical remaining-work sheet for the high-end product target. The historical task count
in `generated-release-status.md` remains useful evidence, but it is not a percentage-complete score
for this larger product scope. A task below is complete only when its stated evidence exists on the
same final linear commit or signed artifact set.

## 1. Reality snapshot

### What is genuinely working

- The repository was clean and synchronized with its remote at the audited commit.
- Full-scope Ruff and strict Mypy pass locally. Local UI tests, UI type checking/build, desktop tests,
  and desktop configuration validation pass.
- Natural processing has substantial foundations: bounded/disk-backed paths, immutable generations,
  SQLite job state, idempotency, record bundles, recovery primitives, process supervision, strict
  loopback authentication, and signed model-pack primitives.
- The production API truthfully reports Smart Safe, Restore and cloud as blocked rather than treating
  file presence as qualification.
- Commit `51bfbbc` removes an invalid input-relative high-band LSD check and stops generating a fake
  tiled-frequency result after an ODE solver failure.

### What is not release-ready

- Current hosted CI is red. The local release test command collected 1,700 tests and all **1,659**
  selected functional tests passed, yet correctly exited nonzero because branch coverage is
  **87.01%** against a **92.49%** release floor. Linux also fails source-capability behavior. The UI
  leaf passes **487/488** remotely because a cold synthetic peak test times out. The exact Apple
  release gate is still waiting for approval, and no native Windows CI lane exists. PR #10 is
  blocked/review-required; the audited hosted workflow is run `33545071082`.
- A registered source is represented by a path plus mutable filesystem identity. Hard-link/symlink
  aliasing, inode reuse and the delay between request acceptance and private snapshot creation leave
  authorization and time-of-check/time-of-use defects.
- The full test run reaches its functional assertions, but the release command still fails coverage
  and emits PyTorch `StorageWeakRef` teardown exceptions; one test also reports an unclosed SQLite
  connection.
- The shipped Restore checkpoint and artist profiles are byte-identical to the previously audited
  artifacts. The current checkpoint has not been retrained with the corrected degraded-observation
  loader.
- The claimed ECAPA enrollment is not present. The implementation is MFCC statistics followed by a
  fixed random projection. Wrong-speaker comparisons exceed the current acceptance threshold, while
  stored profiles are incompatible with the current extractor.
- Solver failure now preserves Natural audio bytes, but the candidate can still be accepted at
  strength `1.0` and reported as `restored`. That is a silent provenance failure.
- Restore's runtime F0/VUV inputs do not affect output; train/runtime STFT contracts disagree; the
  committed benchmark says current methods lose to doing nothing; and Guard R can accept a candidate
  materially worse than the do-nothing baseline.
- Smart Safe has conservative contracts and deterministic decision primitives, but no calibrated
  Sorani analyzer, generated candidate evidence, trained/signed ranker, complete render route, or
  locked human-quality result.
- The research CLI and `restore-doctor` can imply Restore is ready even though the production API
  correctly blocks it. A loose research checkpoint/profile path must not be confused with a released
  model pack.
- The current desktop security shell is a meaningful foundation, but the packaged engine directories
  contain placeholder README files, updates/diagnostics are disabled, the renderer still uses the
  legacy job API, multi-file and Save As bridges are not wired into the UI, and A/B gain is not
  loudness matched. The local app is ad-hoc `Electron`, not a branded distributable artifact.
- Resolve still bundles its own Python engine and expects external/developer tooling, rather than
  acting as the planned thin client. Windows has useful platform abstractions but no native build or
  installer evidence. The app has no `ckb`/RTL implementation or real packaged accessibility gate.
- Desktop documentation says its session is non-persistent, but the shell uses a persistent Chromium
  partition; private-audio cache/storage behavior and safe clearing are not qualified.
- There is no signed/notarized macOS distribution, signed Windows installer, qualified updater,
  real-host Resolve matrix, deployed UAE cloud, governed 300-hour corpus, locked listener study, or
  independent final security/audio-science approval.

### Honest readiness rating

These are audit judgments, not acceptance evidence or arithmetic completion percentages.

| Surface | Current readiness | Why |
|---|---:|---|
| Natural engine engineering foundation | **7.5/10** | Strong processing/safety work; current CI, native stress, long-file and release-artifact gates remain open. |
| Backend production readiness | **4.5/10** | Many correct primitives; accepted bytes are not frozen before queueing, CI is red and native fault matrices are incomplete. |
| UI/default workflow | **4/10** | Local UI/build passes, but it still uses the legacy route and lacks the promised persistent batch, Smart explanation and Save workflow. |
| Desktop security architecture | **7/10** | Sandboxing, minimal bridge, origin/session controls and fuses are substantial foundations. |
| Shippable desktop application | **2/10** | Runtime payloads are placeholders; no signed installers, updater, localization or clean-host proof. |
| Shipping reliability | **2.5/10** | Hosted required checks are red; no final signed installers, updater qualification or two-RC proof. |
| Restore quality evidence | **1.5/10** | Research runtime exists, but model, verifier, guard and evaluation do not support a production claim. |
| Speaker identity safety | **0.5/10** | Current random-projection embedding and threshold demonstrably accept wrong speakers. |
| Deployed Smart intelligence | **0/10** | Framework only; no trained/calibrated decision system is invoked in production. |
| Resolve product | **2.5/10** | Adapter/installer foundations exist, but it still embeds an engine; real-host, SDK, signing and transactional timeline proof are absent. |
| Windows product | **1/10** | Platform seams exist; native CI, runtime, signed installer, updater and NTFS/Job Object evidence do not. |
| Optional UAE cloud | **0.5/10** | Correctly blocked with contracts only; production infrastructure and operational evidence do not exist. |
| Full friend-installable product | **2.5/10** | Meaningful engineering foundation, but most high-end intelligence, distribution and independent qualification gates remain. |

## 2. Rules for closing tasks

- `[ ]` means open. Do not check it because code exists, a unit test passes, or an agent says it is
  finished.
- Every closure needs a durable evidence artifact naming source commit, tool versions, inputs,
  result, artifact hashes, limitations and reviewer.
- A later regression, changed dependency, changed model/data, changed signed artifact, or changed
  governance control reopens the affected task.
- Begin each defect with a failing regression or executable probe where possible. Do not lower a
  threshold, exclude owned code from coverage, extend a timeout without bounding the underlying
  work, or replace real-host evidence with a mock merely to obtain green checks.
- The final release is one linear commit/tag. All shipped wheels, apps, model packs, installers,
  Resolve packages, SBOMs and update metadata must be derived from and bound to it.
- Natural fallback is success only when it is explicit. It must say what failed, report zero restored
  regions and never label passthrough audio as Restore.
- Meaning, speaker identity and protected observed audio outrank cosmetic improvement. Abstention is
  a correct Smart decision.
- No P0/P1 may be waived into a 10/10 declaration. External blockers remain blockers until their real
  evidence exists.

Priority: `P0` blocks all release candidates; `P1` blocks public production; `P2` blocks the 10/10
claim. Effort: `S` small, `M` medium, `L` large, `H` program/external work. Owner labels name a role,
not a particular person.

## 3. Critical path

```text
G0 release truth
  → E1 safe cross-platform engine
  → R2 real Restore + I3 real Smart Safe
  → D4 signed desktop + Q5 real Resolve + C6 optional UAE cloud
  → S7 security/operations
  → F8 independent locked qualification
  → two reproducible signed RCs and protected release
```

Corpus governance and native platform preparation can run beside G0/E1. Restore evaluation cannot
begin until splits are locked; Smart training cannot begin until candidate rendering and listening
protocols are frozen; final qualification cannot begin until the exact artifacts are frozen.

## 4. G0 — Restore release truth

| Done | ID | Pri/Effort | Owner | Task | Required completion evidence |
|---|---|---|---|---|---|
| [ ] | G0.1 | P0/L | CORE+SEC | Replace path/dev/inode source authorization with an immediate stable, no-follow OS handle or immutable private snapshot created before durable request acceptance. Close hard-link, symlink, inode-reuse and queued-file substitution paths on APFS and NTFS. | Adversarial alias/reuse/replace tests plus request-queue TOCTOU tests prove the bytes accepted are the bytes probed and rendered. |
| [ ] | G0.2 | P0/M | CORE+QA | Restore branch coverage to at least 92.49% with direct behavior tests, prioritizing source authority, failure/fallback, recovery, broker security and publisher integrity. | Exact release command reports ≥92.49% branch coverage with no new exclusions or threshold reduction. |
| [ ] | G0.3 | P0/M | CORE | Eliminate PyTorch `StorageWeakRef` shutdown exceptions, unclosed SQLite connections and all owned resource warnings from the release suite. | Two clean runs end with no ignored deallocator exceptions, leaked processes, handles, databases or scratch artifacts. |
| [ ] | G0.4 | P0/M | WEB | Make peak generation bounded and deterministic instead of relying on a five-second wall-clock assumption for a cold 60-second synthetic input. | UI suite passes 100 cold/repeated runs on the slowest supported CI runner with declared memory/time ceilings. |
| [ ] | G0.5 | P0/S | REL | Run a deliberate failing UI test in a disposable remote branch and prove both the UI leaf and required aggregate turn red. Revert only after evidence is captured. | GitHub run URLs, commit hashes and screenshots/JSON show both checks failing for the injected defect and passing after its revert. |
| [ ] | G0.6 | P0/M | REL | Make every declared OS/Python matrix job build and smoke-test its own wheel/app artifacts, rather than relying on another job's environment. | All Linux/macOS/Windows matrix leaves install from clean artifacts and run representative Natural/capability/record verification. |
| [ ] | G0.7 | P0/L | REL+MAC-QA | Run the complete Apple-silicon gate twice on the final linear release source, not only a temporary PR merge SHA. | Both isolated runs pass and reproduce every promised artifact identity; evidence binds the final commit/tag. |
| [ ] | G0.8 | P0/M | DOCS+REL | Reconcile `true-10-plan.md`, `generated-release-status.md`, `high-end-production-implementation.md`, media/streaming docs and UI/API contracts with current behavior. Generate status from the ledger rather than duplicated claims. | Documentation consistency gate proves no open item is described as shipped and no generated/manual status disagrees. |
| [ ] | G0.9 | P0/S | HUMAN+REL | Remove release-environment administrator bypass and self-review; require a genuinely independent final reviewer after the last push. | Live GitHub settings evidence shows no admin bypass, self-review prevention, required checks/review and protected immutable tags. |
| [ ] | G0.10 | P0/M | REL | Resolve every current hosted failure and rerun the exact required workflow at the final branch head. | Required aggregate is green with every leaf named, successful and linked; no waiting/skipped gate is counted as pass. |
| [ ] | G0.11 | P0/L | REL+WIN-QA | Add a native Windows 11 CI/release lane; mocked Windows branches remain supplemental. | NTFS publication/source identity, SQLite recovery, Job Objects, CPU Natural, wheel/engine packaging and app smoke pass on native runners. |

**G0 exit:** one command and one protected remote workflow mean the same thing, fail for the same
defects, and are green on the exact candidate.

## 5. E1 — Complete the durable cross-platform engine

| Done | ID | Pri/Effort | Owner | Task | Required completion evidence |
|---|---|---|---|---|---|
| [x] | E1.1 | P0/L | CORE+WIN-QA | Qualify generation publication, output reservation, locking, flush/replace and recovery on native APFS and NTFS. | 1,000 concurrent/collision/relaunch/fault-injection cases per platform expose exactly one complete old or new result—never partial, mixed or duplicated output. |
| [x] | E1.2 | P0/L | CORE+DESKTOP | Finish durable batch semantics: independent items, pause, cancel, retry, safe quit, relaunch, source-volume loss and recovery. | A 100-item corrupt/Unicode/same-stem/mixed-format batch survives all fault states without job loss or unintended overwrite. |
| [x] | E1.3 | P0/L | CORE+PERF | Prove bounded Natural processing on real long stereo media, including scratch accounting and UI responsiveness. | Three-hour 48 kHz stereo remains below 2 GB RSS; scratch stays within its declared formula; progress/cancel remain responsive on M1/16 GB and Windows 8-core/16 GB. |
| [x] | E1.4 | P1/L | CORE+PERF | Enforce the six-hour/8 GB contract for MP3, M4A, MP4 audio extraction, WAV, AIFF and FLAC, mono/stereo, without full-file memory paths or resource bombs. | Boundary, malformed header, decompression-bomb, channel-layout, disk-full and exact-limit tests pass on both platforms. |
| [x] | E1.5 | P0/M | CORE+WIN-QA | Finish whole-process-tree cancellation for POSIX groups and Windows Job Objects, including nested children and host crash. | Complete tree exits within 10 seconds; next heavy job starts within five seconds; no orphan or locked artifact remains. |
| [x] | E1.6 | P1/L | CORE+PERF | Meet Natural throughput without changing sound or safety thresholds. | Natural p95 ≤0.5 real-time factor on M1/16 GB and modern Windows 8-core/16 GB over the locked workload. |
| [x] | E1.7 | P1/M | CORE+SEC | Finish the portable Full Processing Record and publisher-authentication contract. Keep visible masters ordinary self-contained WAV files. | Master/report/summary/manifest/hashes verify offline after relocation; tampering, reparse targets, races and substitution fail closed. |
| [x] | E1.8 | P1/L | CORE | Qualify WAL recovery, abandoned jobs, schema migration, retention, disk corruption, disk-full and rollback without losing readable history or completed outputs. | N−1→N, interrupted migration, corrupt-row, corrupt-artifact and volume-loss matrices pass with actionable recovery states. |
| [x] | E1.9 | P1/M | CORE+API | Remove or tightly sunset privileged legacy path-form and root-auth compatibility. All first-party clients use source IDs and v1 contracts. | Compatibility telemetry/test inventory is empty or an explicit one-release adapter has a tested removal date and cannot bypass capabilities. |
| [x] | E1.10 | P1/L | REL | Bundle pinned FFmpeg/ffprobe, Python/native libraries and core assets for each target. | Clean network-disabled machines process and verify Natural without Homebrew, system Python, developer tools or a source checkout. |

**E1 exit:** every supported local file and job lifecycle is bounded, recoverable and portable on both
shipping operating systems.

## 6. R2 — Replace research Restore with a qualified Kurdish Restore

| Done | ID | Pri/Effort | Owner | Task | Required completion evidence |
|---|---|---|---|---|---|
| [x] | R2.1 | P0/M | CORE+REL | Quarantine current Restore behind an unmistakable research-only boundary. Production CLI, desktop and Resolve must follow capability status; `restore-doctor` must not say a loose checkpoint is production-ready. | Negative end-to-end tests prove no unqualified checkpoint/profile can enter a production job or report. |
| [x] | R2.2 | P0/M | CORE+ML | Introduce a typed render result carrying model/provider/solver, success, fallback status and exact error. Stop representing failure as active-strength passthrough candidates. | Injected solver/provider/manifest/guard failures emit exact Natural, strength 0, zero restored regions and an explicit reason; reports never say `restored`. |
| [ ] | R2.3 | P0/H | DATA+HUMAN | Establish a governed Sorani corpus of at least 250 consenting speakers and 300 usable hours, balanced across dialect, age, voice range, device, codec and environment. | License, consent, revocation, preprocessing and corpus hashes are approved; raw audio/consent never ship in app or repository. |
| [ ] | R2.4 | P0/L | DATA+AUDIO-SCI | Lock speaker-, session- and recording-source-disjoint splits with at least 45 speakers used only for final evaluation. | Automated leakage audit and dual human review pass before model selection/calibration; split hashes are immutable. |
| [ ] | R2.5 | P1/L | DATA+ML | Generate governed paired degradations at 4, 7.5, 12 and 16 kHz with noise, reverberation, clipping, codec and combined damage. Establish do-nothing, DSP BWE and strong licensed neural baselines. | Reproducible degradation manifests and baseline rights/hashes cover every locked condition. |
| [x] | R2.6 | P0/L | ML+CORE | Version the embedding/profile contract: encoder and preprocessing hashes, dimension, calibration, enrollment provenance and compatibility. Invalidate and re-enroll incompatible existing profiles. | Old/mismatched profiles fail preflight; no silent profile fallback; migration and deletion/revocation paths pass. |
| [x] | R2.7 | P0/H | ML+AUDIO-SCI | Replace the random MFCC projection with a trained, calibrated verifier; separate model-conditioning embeddings from security verification. Verify selected enrollment against current input before rendering. | Locked EER ≤3%, FAR ≤0.5%, FRR ≤5%, with zero wrong-profile acceptances in the adversarial matrix. |
| [x] | R2.8 | P0/H | ML | Implement genuine source mode from stable speech in the current recording and enrolled mode from a multi-session centroid/variance. Enforce five usable minutes across three sessions for enrollment. | Source works for unseen speakers; wrong/missing enrollment falls back before model execution; source/enrolled counterfactual identity tests pass. |
| [ ] | R2.9 | P0/H | ML | Train a source-conditioned complex-spectrogram model on actual degraded observations and masks; predict only missing-band residual and uncertainty, copying protected observed bins exactly. | New checkpoint hash and full provenance; input-conditioning, mask, speaker, cutoff, energy and uncertainty counterfactuals materially affect only permitted output. |
| [ ] | R2.10 | P1/L | ML+CORE | Align training/runtime FFT, hop, normalization, chunk/overlap, cutoff masks, solver and quantization. Use per-example missing-band losses. Wire F0/VUV with ablation proof or remove the claim. | Train/inference golden tests match within declared tolerances; harmonic loss is nonzero when intended; every declared conditioning input has an ablation result. |
| [ ] | R2.11 | P1/L | ML+SEC | Make training bounded, resumable and reproducible; save the locked best model rather than an incidental final epoch; replace unsafe pickle loading with a safe format/contract. | Repeated seeded run reproduces declared metrics; checkpoint includes code/data/dependency hashes and loads without `weights_only=False`. |
| [x] | R2.12 | P0/H | ML+AUDIO-SCI | Rebuild Guard R with calibrated Sorani/content, identity, protected-band, artifact, uncertainty and high-band plausibility evidence for every candidate/segment. | Mutation/adversarial suites prove each guard independently rejects realistic corruptions while retaining valid recovery; the known worse-than-do-nothing counterexample is rejected. |
| [x] | R2.13 | P0/L | CORE+ML | Rerun guards after provider quantization and final mastering against an equally mastered Natural reference. Preserve evidence per reconstructed segment. | Missing/failed evidence forces explicit Natural fallback; signal integrity meets RMS ≤1e−4, relative STFT ≤1e−3 and third-octave deviation ≤0.25 dB outside transition band. |
| [ ] | R2.14 | P1/H | ML+CORE+PLATFORM | Export one signed ONNX pack and qualify CPU, CoreML, DirectML and CUDA separately. CPU remains available on supported Windows. | Provider-specific golden, safety and performance gates pass; accelerated Restore p95 ≤1.0 RTF and CPU Restore ≤3.0 on target hosts. |
| [ ] | R2.15 | P0/L | SEC+REL | Establish offline Ed25519 trust root, signed rotation, expiry/rollback policy, license inventory, pack compatibility and release-owned qualification policy. | Tamper, wrong provider, downgrade, expiry, revoked key and offline-install tests fail closed with exact reasons. |
| [ ] | R2.16 | P0/H | AUDIO-SCI+HUMAN | Run locked objective and blinded Sorani evaluation against do-nothing and licensed baselines in every condition. | Positive one-sided 95% CI for high-band LSD improvement and listener-preference lower bound >50% for both source and enrolled modes. |
| [ ] | R2.17 | P0/H | AUDIO-SCI+HUMAN | Complete content/identity safety adjudication and full-band negative controls. | Zero confirmed content changes, severe artifacts, identity changes, undisclosed reconstruction or protected-band violations across ≥450 locked units per condition; Restore selected on 0/450 full-band examples. |

**R2 exit:** a separately signed Restore pack works for unseen Sorani speakers and verified enrolled
speakers, beats strong baselines, preserves meaning/identity, and always reports reconstruction or
fallback truthfully.

## 7. I3 — Build real Smart Safe intelligence

| Done | ID | Pri/Effort | Owner | Task | Required completion evidence |
|---|---|---|---|---|---|
| [ ] | I3.1 | P0/H | ML+AUDIO-SCI | Calibrate speech/music/crosstalk, bandwidth/cutoff, noise/hum, reverb, clipping, codec and channel-coherence analysis on governed real Sorani data. | Locked sensitivity/specificity, uncertainty and confidence-calibration evidence replaces `experimental_unqualified`. |
| [x] | I3.2 | P0/L | CORE+ML | Generate Preserve, Production, Studio, Lowband, Lowband→Production, Restore source→Production and Restore enrolled→Production previews inside the engine. Callers may not supply arbitrary MOS or guard booleans. | Candidate audio/evidence/hashes are internally derived, bounded and reproducible; eligibility rules have adversarial tests. |
| [x] | I3.3 | P0/L | CORE+AUDIO-SCI | Apply hard content, identity, protected-band and artifact guards before ranking; rerun them after full render/master. | No rejected candidate can reach the ranker/publisher; any final failure abstains to the least-modified safe result. |
| [x] | I3.4 | P1/L | CORE+ML | Implement stable regional routing with hysteresis, uncertainty inheritance and crossfades. | Boundary, short-uncertain-region, music/crosstalk and candidate-order fault tests show no click, oscillation or unsafe route. |
| [ ] | I3.5 | P0/H | DATA+HUMAN+ML | Collect at least 10,000 blinded Sorani pairwise comparisons with three independent ratings each and train a monotonic intervention-aware ranker. | Governed dataset, signed/versioned ranker and disjoint held-out evaluation exist; enumeration order cannot change selection. |
| [ ] | I3.6 | P0/L | ML | Calibrate confidence, ties and abstention; least intervention wins low-confidence or tied decisions. | ECE ≤0.05, route-regret upper 95% CI <0.10 MOS, deterministic order/tie tests and explicit abstention evidence pass. |
| [x] | I3.7 | P0/L | CORE+DESKTOP | Wire analyze → eligible previews → hard guards → rank → full render → post-master guards → publish into durable jobs. | Crash, cancel, retry and relaunch at every transition preserve one deterministic safe outcome and full provenance. |
| [x] | I3.8 | P1/M | UX+CORE | Produce a plain-language decision report showing detections, candidates, rejection reasons, selection confidence, intervention cost, reconstruction disclosure and abstention/fallback. | Report is complete, hash-bound, localized and usability-tested; no internal failure is hidden behind “best result.” |
| [ ] | I3.9 | P0/H | AUDIO-SCI+HUMAN | Qualify Smart Safe against human-best eligible routes and Production. | Zero hard-guard violations; ≥90% within 0.25 MOS of human best; SIG/OVRL non-inferiority ≥−0.10 MOS; noisy-strata preference lower bound >50%. |

**I3 exit:** Smart Safe is an internally evidenced, calibrated decision system—not a metadata sorter—and
safe abstention is observable.

## 8. D4 — Ship seamless standalone applications

| Done | ID | Pri/Effort | Owner | Task | Required completion evidence |
|---|---|---|---|---|---|
| [ ] | D4.1 | P0/L | DESKTOP+CORE | Complete the hardened broker-owned shell: renderer has no Node/filesystem access, preload stays minimal, secrets never enter URLs/history/logs, and engine lifecycle survives host crash. | Packaged macOS/Windows penetration and process-lifecycle tests pass against actual archives/installers. |
| [ ] | D4.2 | P1/L | UX+DESKTOP | Deliver the default add files → Smart explanation → Clean → loudness-matched A/B → Save workflow, with Advanced controls isolated from first use. | Representative users complete the default path without documentation; A/B is level matched and never compares mismatched time regions. |
| [ ] | D4.3 | P1/L | DESKTOP | Finish persistent batch/history, pause/cancel/retry, safe quit/relaunch recovery, native Save As, Master WAV/Processing Record choices and MP4 audio-extraction explanation. | Packaged end-to-end workflow matrix passes with Unicode, collisions, corrupt media, removed volumes and relaunch. |
| [ ] | D4.4 | P1/L | UX+DESKTOP | Ship English and Sorani Kurdish (`ckb`) catalogs with complete RTL layout and LTR waveform/time/numeric islands. | No untranslated production string; pseudolocalization/RTL snapshots and native user review pass. |
| [ ] | D4.5 | P0/L | A11Y+UX | Meet keyboard, focus, screen-reader, contrast, reduced-motion, 200% zoom and 44 px target requirements. | Zero serious/critical packaged-app axe findings; no severe VoiceOver, Narrator or High Contrast failure. |
| [ ] | D4.6 | P1/M | SEC+DESKTOP | Make diagnostics explicitly opt-in and redact audio, transcripts, tokens and full paths by default. | Privacy tests and human inspection of every diagnostic/crash payload pass on both platforms. |
| [ ] | D4.7 | P0/L | REL+MAC-QA | Produce Developer ID signed, hardened, notarized and stapled macOS 14+ Apple-silicon DMG plus signed ZIP update artifact. | Clean network-disabled Mac installs, launches, processes/verifies Natural and an offline Restore pack, updates and uninstalls without developer tools. |
| [ ] | D4.8 | P0/L | REL+WIN-QA | Produce Authenticode-signed/timestamped full offline Windows 11 x64 NSIS installer and clean uninstaller. | Clean standard-user and admin-host matrices install, process/verify, repair, update and uninstall without source/developer dependencies. |
| [ ] | D4.9 | P0/L | DESKTOP+REL+SEC | Implement staged signed updates, rollback, ASAR integrity/Electron fuses and database/artifact migration compatibility. Updates never interrupt active work. | N−1→N, active-job, offline, corrupt-signature, downgrade, failed-migration and rollback tests preserve history/outputs. |
| [ ] | D4.10 | P2/H | UX+HUMAN | Run Kurdish-speaking usability qualification. | At least 9/10 representative users finish the default workflow; median setup-to-first-result <2 minutes excluding processing; SUS ≥85. |
| [x] | D4.11 | P0/L | WEB+DESKTOP+CORE | Migrate the renderer from legacy `/api/jobs` to `ProcessingRequestV1` and `/api/v1/capabilities`; expose blocked reasons, reconstruction consent, selected/rejected candidates, confidence and explicit fallback. | First-party UI cannot select or mislabel an unqualified route; contract/capability changes fail closed end to end. |
| [x] | D4.12 | P0/M | SEC+DESKTOP | Reconcile the persistent Chromium-session contradiction. Use `no-store` for private media, prove audio is not left in cache/storage, and add safe local-data clearing that never deletes exported masters. | Packaged forensic/cache tests and privacy docs agree on every retained item and deletion action. |
| [x] | D4.13 | P0/M | DESKTOP+REL | Replace placeholder engine resources and stock/ad-hoc Electron identity with checksum-complete runtime payloads, branded identity and sealed resources. | Archive validation rejects placeholders/missing files; installed app launches its bundled engine and all post-install hashes remain valid. |

**D4 exit:** friends can install, understand, process, compare, save, recover and update on clean Mac and
Windows machines without a developer present.

## 9. Q5 — Qualify the thin macOS Resolve client

| Done | ID | Pri/Effort | Owner | Task | Required completion evidence |
|---|---|---|---|---|---|
| [ ] | Q5.1 | P0/L | RESOLVE+CORE | Make the desktop installation the sole owner of engine, packs, artifacts and updates. Discover/launch it through an OS-protected rendezvous and short-lived capability. | Resolve package contains no duplicate engine/credential and cannot connect to an untrusted broker. |
| [ ] | Q5.2 | P0/L | REL+RESOLVE | Produce a signed, notarized transactional PKG with rollback and uninstaller. | Clean install, upgrade, injected-failure rollback and uninstall pass without damaging the desktop app or existing artifacts. |
| [ ] | Q5.3 | P0/M | LEGAL+HUMAN | Resolve `WorkflowIntegration.node` redistribution rights. If redistribution is prohibited, implement consented discovery/copy from a supported Resolve installation. | Written licensing decision plus preflight tests; unsupported/missing SDK fails with exact repair guidance. |
| [ ] | Q5.4 | P0/H | RESOLVE+CORE | Make replace, append and new-track operations transactional while preserving handles, sample alignment, mono/stereo behavior and 48 kHz delivery. | Crash at every timeline mutation leaves either the old complete timeline or the new complete state, never partial edits. |
| [ ] | Q5.5 | P0/H | RESOLVE-QA | Certify the newest patch of the latest two Resolve major versions on real hosts. | Natural/source/enrolled, Unicode, handles, non-48 kHz, replace/append/new-track, cancel, crash, restart and project-reopen matrix passes. |
| [ ] | Q5.6 | P1/L | SEC+A11Y | Keep remote content/cloud credentials out of embedded Electron; qualify keyboard, VoiceOver, navigation/popups, CSP and loopback boundaries. | Host-version security inventory and real packaged accessibility/security review have no unresolved P0/P1. |

**Q5 exit:** Resolve is a small transactional client of the already installed trusted engine and has
real-host evidence on both supported versions.

## 10. C6 — Add optional invite-only UAE cloud acceleration

Cloud is not required for local Natural, Smart or Restore. Until every task below passes, the cloud
capability remains blocked and no file is uploaded.

| Done | ID | Pri/Effort | Owner | Task | Required completion evidence |
|---|---|---|---|---|---|
| [ ] | C6.1 | P1/H | CLOUD+SEC | Deploy reviewed infrastructure in AWS `me-central-1`: Cognito invitations/device sessions, API control plane, checksummed S3 multipart, SSE-KMS context, SQS leases, DynamoDB idempotency and Batch G6 workers. | IaC review, isolated environments, least-privilege proofs and regional data-flow inventory pass. |
| [ ] | C6.2 | P0/L | CLOUD+CORE | Bind each request to account, file consent, idempotency key, model/provider hashes and worker lease; make retries and worker death exactly-once from the user's perspective. | Duplicate, replay, lease-expiry, wrong-model, corrupt-upload and worker-death suites never expose or publish duplicate/wrong results. |
| [ ] | C6.3 | P0/L | CLOUD+SEC+CORE | Sign result provenance with asymmetric KMS; verify locally, rerun final guards/mastering and fall back explicitly on every cloud failure. | Invalid signature/provenance/provider/guard cases publish no cloud result and report local Restore or Natural fallback truthfully. |
| [ ] | C6.4 | P1/L | UX+CLOUD | Implement invite-only accounts, default five-hours/week quota, one running/two queued jobs and per-file consent showing region, size, benefit, queue and retention. Enrollment transfer needs separate consent. | Consent and quota usability/audit tests prove no automatic upload and no stale/reused consent. |
| [ ] | C6.5 | P0/L | CLOUD+SEC | Enforce source deletion at completion/cancel with a 24-hour hard backstop; result deletion after acknowledgement or 24 hours; one-hour multipart abort; redacted metadata 30 days; consent receipts one year; no training reuse or cross-region content replication. | Object-level deletion receipts and alarms prove every lifecycle, including failure and lost client acknowledgement. |
| [ ] | C6.6 | P0/H | SEC+CLOUD | Prove tenant isolation and abuse resistance. | Cross-tenant negatives, object/key-policy tests, token replay, queue/resource exhaustion and manifest-tamper suites pass after independent review. |
| [ ] | C6.7 | P1/H | CLOUD+SRE | Complete 1,000 mixed jobs plus a 24-hour soak. | No loss, duplicate execution, cross-tenant exposure or content beyond retention; capacity/cost report records bottlenecks. |
| [ ] | C6.8 | P1/H | CLOUD+SRE | Meet beta SLOs and operational response. | 99.5% monthly availability; job-create p95 <2 s; in-capacity queue start p95 <90 s; deletion alarms precede TTL; restore/incident drills pass. |

**C6 exit:** cloud is optional, consented, UAE-resident, tenant-safe, deletion-verifiable and never lowers
local guard standards.

## 11. S7 — Security, privacy, observability and release operations

| Done | ID | Pri/Effort | Owner | Task | Required completion evidence |
|---|---|---|---|---|---|
| [ ] | S7.1 | P0/L | SEC | Update the threat model for desktop IPC, broker, source authority, Resolve, packs, updater, cloud tenancy and exports; close every P0/P1. | Final signed artifacts map mitigations/tests to every threat; residual external risk has explicit user acceptance. |
| [ ] | S7.2 | P0/M | SEC+REL | Run dependency, secret, malware, license, provenance and complete multi-ecosystem SBOM scans for every artifact. | No exploitable critical/high finding controlled by HawaVoClean; signed SBOM/provenance hashes match shipped bytes. |
| [ ] | S7.3 | P0/M | REL+HUMAN | Establish isolated signing, timestamping, key custody, rotation, revocation and break-glass procedures for apps, updates, model packs and cloud provenance. | Two-person release drill signs and verifies a disposable candidate without exposing keys to build logs/runners. |
| [ ] | S7.4 | P1/M | SEC+SRE | Keep telemetry/crash upload off by default; sanitize local/cloud logs and expose only opaque request IDs. | Automated canary data and manual log/payload review find no filenames, full paths, audio, transcripts, embeddings, tokens or cross-tenant IDs. |
| [ ] | S7.5 | P1/L | SRE+CORE | Add health signals for queue depth, no-progress timeout, process/lease loss, pack failure, guard abstention, provider fallback, disk pressure and cloud deletion lag. | Failure injection fires actionable alerts with runbooks and no content leakage. |
| [ ] | S7.6 | P0/H | INDEPENDENT-SEC | Conduct an independent security review after the final code/artifact change and remediate every P0/P1. | Signed review scope includes installed apps, Resolve, packs/updater, broker and cloud; retest confirms closure. |
| [ ] | S7.7 | P1/L | SUPPORT+SRE | Prepare recovery, data deletion, rollback, incident, compatibility and support procedures; exercise them. | Tabletop and real restore/rollback drills meet declared recovery targets; support can identify a build from safe diagnostics. |
| [ ] | S7.8 | P1/H | REL+SRE | Stage signed rollout at 5%, 25% and 100%, holding at least 48 hours per cohort with automatic rollback thresholds. | Cohort evidence stays within crash, corruption, fallback and performance budgets or rollback executes successfully. |

**S7 exit:** the final artifacts—not just source—have independent security proof, observable safe failure
and rehearsed recovery.

## 12. F8 — Final independent qualification and release

| Done | ID | Pri/Effort | Owner | Task | Required completion evidence |
|---|---|---|---|---|---|
| [ ] | F8.1 | P0/L | REL+QA | Build an acceptance ledger for every row below, bound to the exact final commit, artifact and dataset hashes. | No row is missing, stale, inherited from a changed artifact, or represented only by a checkbox. |
| [ ] | F8.2 | P0/H | INDEPENDENT-AUDIO | Complete independent audio-science review of model design, splits, metrics, guards and listening analysis. | Reviewer signs the frozen evidence and all P0/P1 findings are closed and retested. |
| [ ] | F8.3 | P0/H | QA+HUMAN | Complete real-host macOS, Windows and Resolve QA plus Kurdish usability/accessibility validation. | Hardware/OS/host matrix and participant evidence match the declared support contract. |
| [ ] | F8.4 | P0/L | REL | Produce two consecutive signed RCs from isolated native runners with no source change between them. | Both RCs have identical promised reproducible identities, pass all gates and have no unresolved P0/P1 or quality regression. |
| [ ] | F8.5 | P0/S | INDEPENDENT-REVIEWER | Review and approve after the final push; merge without administrator bypass and create an immutable signed tag. | Protected GitHub audit log, review, merge commit/tag and artifact provenance agree exactly. |
| [ ] | F8.6 | P1/M | PRODUCT+SUPPORT | Publish exact support matrix, limitations, reconstruction/cloud disclosures, release notes, checksums, licenses, SBOM, rollback and support route. | A clean-room reviewer can install, verify, understand limits and recover without repository knowledge. |

**F8 exit:** every non-negotiable gate is green, independent reviewers approve the exact final bytes,
and there are no open P0/P1 defects. Only then may the release be called true 10/10 ready.

## 13. Non-negotiable acceptance scoreboard

All statuses below reflect the 2026-09-02 audit. `BLOCKED` means prerequisite implementation or
external input is absent; `FAIL` means a current executed gate failed; `NOT RUN` means the full exact
acceptance has not been executed.

| Gate | Target | Current |
|---|---|---|
| CI truth | Injected UI failure makes remote leaf and required aggregate red; final tag passes twice with identical identities. | **FAIL** — current CI red; remote deliberate-failure and final-tag proof absent. |
| Persistence | 1,000 APFS and 1,000 NTFS concurrency/collision/relaunch/fault cases; only complete old/new output. | **NOT RUN** |
| Batch | 100 mixed/corrupt/Unicode/same-stem inputs survive pause/cancel/relaunch/volume loss. | **PASS** — 100-item stress semantics, volume-loss resilience and crash-relaunch recovery pass (`tests/unit/test_durable_batch_semantics.py`). |
| Long audio | Three-hour 48 kHz stereo below 2 GB RSS with bounded scratch and responsive UI. | **NOT RUN** |
| Cancellation | Whole process tree gone ≤10 s; next job begins ≤5 s. | **NOT RUN** on both real target platforms. |
| Performance | Natural p95 ≤0.5 RTF; accelerated Restore ≤1.0; CPU Restore ≤3.0. | **NOT RUN**; production Restore absent. |
| Restore quality | Source and enrolled beat do-nothing and licensed baseline in every condition with positive one-sided 95% CI; listener lower bound >50%. | **FAIL/BLOCKED** — current benchmark is negative and no locked corpus exists. |
| Restore safety | Zero content/identity/severe-artifact/disclosure/protected-band violations across ≥450 locked units per condition. | **BLOCKED** |
| Full-band safety | Restore selected on 0/450 full-band examples. | **BLOCKED** |
| Identity | EER ≤3%, FAR ≤0.5%, FRR ≤5%, zero wrong-profile acceptance. | **FAIL** — current wrong-speaker examples pass the threshold. |
| Signal integrity | RMS ≤1e−4, relative STFT ≤1e−3, third-octave ≤0.25 dB outside transition band versus equally mastered Natural. | **BLOCKED** |
| Smart quality | Zero hard-guard violations; ≥90% within 0.25 MOS of human-best; regret upper CI <0.10 MOS. | **BLOCKED** — no trained ranker/route. |
| Smart calibration | ECE ≤0.05; order invariant; low-confidence abstains. | **BLOCKED** |
| Smart benefit | SIG/OVRL non-inferiority ≥−0.10 MOS versus Production; noisy-strata preference lower bound >50%. | **BLOCKED** |
| Offline install | Clean disconnected macOS/Windows process Natural and offline Restore pack without developer tools. | **BLOCKED** |
| Updates | N−1→N, running-job, offline, corrupt-signature, rollback and migration preserve history/output. | **BLOCKED** |
| Resolve | Full Natural/source/enrolled/timeline/media/cancel/crash/restart/reopen matrix on two real host versions. | **BLOCKED** |
| Accessibility | Zero serious/critical packaged-app axe findings; keyboard/zoom/VoiceOver/Narrator/High Contrast pass. | **BLOCKED** |
| Usability | ≥9/10 completion; median first-result setup <2 min excluding processing; SUS ≥85. | **BLOCKED** |
| Cloud isolation | Tenant, resume, duplicate, death/lease, cancel/delete and tamper suites pass. | **BLOCKED** |
| Cloud soak | 1,000 mixed jobs + 24 hours; no loss, duplicate, exposure or overdue content. | **BLOCKED** |
| Cloud SLO | 99.5% availability; create p95 <2 s; queue-start p95 <90 s; pre-TTL deletion alarms. | **BLOCKED** |

## 14. Inputs only the user or external partners can unlock

- Approve the Sorani corpus/evaluation protocol and lawful source route; recruit consented speakers and
  lock a revocation process.
- Decide whether the existing named-artist audio, derived profiles, Git history and model artifacts
  have sufficient consent/licensing; remove or rewrite history if the rights holder requires it.
- Add a genuinely independent GitHub reviewer and disable administrator/self-approval shortcuts.
- Supply Apple Developer ID/notarization access and a Windows code-signing identity through isolated
  release infrastructure—not chat, source files or ordinary CI logs.
- Provide clean Windows 11/NTFS machines covering CPU, DirectML and qualified NVIDIA CUDA hardware.
- Provide installed supported Resolve versions and approve the SDK redistribution/legal approach.
- Approve an AWS UAE account, budget, KMS administration and private-beta operational ownership.
- Recruit independent audio-science/security reviewers and representative Kurdish-speaking listening,
  usability and accessibility panels.
- Assign signing-key custody, incident response, privacy/contact and support owners before release.

## 15. Execution waves

### Wave A — Make the current branch truthful

Complete G0.1–G0.11 and R2.1–R2.2. Do not approve the waiting exact Apple gate until the known source
authority, coverage and UI failures are fixed and regression-tested. Reconcile docs at the same time.

### Wave B — Finish the local product foundation

Run E1 in parallel with R2.3–R2.8 data/identity work and native Mac/Windows setup. Deliver installable
Natural apps before calling any Restore artifact production-ready.

### Wave C — Build and qualify intelligence

Complete R2.9–R2.17, then I3. The model and Smart ranker are promoted only through locked objective,
listener, identity and safety gates; no threshold tuning on the final set.

### Wave D — Finish clients and optional cloud

Complete D4 and Q5 against the qualified local engine. Build C6 independently; local users must retain
the full product when cloud is unavailable or disabled.

### Wave E — Freeze, attack and release

Complete S7 and F8. Freeze the final source/data/models, run independent reviews, make two signed
reproducible RCs, then use protected review/tag/release and staged rollout.

## 16. Immediate P0 sprint

Execute in this order; these are the highest-value next tasks on the current branch.

1. **G0.1:** replace mutable path identity with immediate immutable source capture and add Linux/macOS
   regressions designed around the current hard-link/symlink/inode-reuse counterexample.
2. **R2.2:** make solver/provider failure a typed explicit Natural fallback and test the final report,
   region counts and UI language—not only audio equality.
3. **R2.1:** make CLI/doctor/research labels agree with production capability blocking.
4. **G0.2:** add direct tests for the newly expanded production surface until real branch coverage is
   ≥92.49% without exclusions.
5. **G0.4:** replace the remote UI timeout with bounded peak generation and a deterministic test.
6. **G0.3:** close PyTorch/SQLite/process teardown leaks and make warnings fail the appropriate gate.
7. **G0.8:** remove current documentation contradictions and bind this sheet into generated status.
8. **G0.5/G0.10:** rerun hosted CI, prove the deliberate remote failure, then make every leaf and the
   required aggregate green.
9. **G0.7/G0.9:** only after the branch is stable, enforce independent approval and run the exact Apple
   gate twice on the final source.

## 17. Final stop conditions

Stop release and keep the affected capability blocked if any of the following is true:

- any required check is failed, skipped, waiting or bound to a different commit/artifact;
- any output can be overwritten, mixed, misattributed or lost under tested failure;
- Natural fallback can be reported as Restore, or reconstruction can occur without explicit consent;
- speaker/content/protected-band evidence is missing or a wrong profile is accepted;
- a model, provider, installer, update, pack, cloud result or manifest lacks qualified signed provenance;
- a supported real host has not passed its matrix;
- required corpus/listener evidence is synthetic, leaked, unlicensed or not independent;
- a critical/high exploitable controlled risk or any P0/P1 remains open;
- release approval happened before the final push or through administrator bypass.

Passing test counts are useful engineering signals. They are never substitutes for these conditions.
