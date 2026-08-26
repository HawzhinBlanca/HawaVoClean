# U1 — GitHub governance activation runbook

State of the live repository as of the T3.3 evidence entries (ledger sequences
38 and 39). Everything here was measured against `HawzhinBlanca/HawaVoClean`,
not assumed.

## U1 decision: resolved by visibility change

On 2026-08-24 the owner made the repository **public**. That is the second of
U1's two documented paths, and it resolves both blockers at once: GitHub Actions
minutes are free for public repositories, and required-reviewer environment
protection — which the free private plan refused with HTTP 422 — becomes
available. The plan's stated preference had been to repair billing and keep the
private boundary; that preference is superseded by this decision, and
`RISKS.md` R-12 records it.

Before anything else was touched, the newly public surface was audited: all 27
tracked audio files are synthetic fixtures, no private Sorani corpus is in the
repository, there are no credential or `.env` files, and a pattern scan across
200 commits of history found no tokens or keys. Secret scanning and push
protection were then enabled — both free on public repositories and unavailable
before.

## Where U1 actually stands

| Contract requirement | State | Blocked by |
|---|---|---|
| `immutable-release-tags` ruleset on `refs/tags/v*` | **Active** (id 21254065) | — |
| Release workflow is the attested file | **Verified** (`50c7974a…`) | — |
| Branch rules available on this plan | **Confirmed by probe** | — |
| Any GitHub Actions job starting | **Running** | — |
| `release-candidate` environment with a required reviewer | **Active** | — |
| Secret scanning and push protection | **Enabled** | — |
| Protected `main` requiring `release / required` | Not applied | The runner — see below |
| Self-hosted `hawavoclean-release` runner | **Registered and proven** | — |
| `HAWAVOCLEAN_RELEASE_EVIDENCE_ROOT` | **Set — 2026-08-26** (`/Users/hawzhin/HawaVoCleanEvidence`) | — |
| Private evidence hydrated by a hosted job | **Yes — 2026-08-26** | — |
| `exact-release-gate` has executed at least once | **Yes — 2026-08-25** | — |
| Fork pull requests require approval from all outside contributors | **Active** | — |

Re-measured against the live repository on 2026-08-25, after PRs #1, #2 and #3
merged: 0 runners registered; one ruleset (`immutable-release-tags`, tag target);
`main` returns HTTP 404 for branch protection; `release-candidate` carries
`required_reviewers` and `branch_policy`; no repository variables exist.

**The remaining order is strictly serial, and the runner is its head.** The
earlier note that protected `main` needed "one green run first" understates the
dependency. `required` is an aggregate job that asserts
`test "$EXACT_RELEASE_GATE" = success`, and `exact-release-gate` runs on
`[self-hosted, macOS, ARM64, hawavoclean-release]`. With no such runner the gate
never starts, so `required` never starts either — it does not appear in a pull
request's checks at all. Applying a ruleset with a strict `release / required`
context in that state would not merely defer merges; it would make `main`
**permanently unmergeable** until a runner exists. Hosted jobs going green,
which they now do, does not change this.

## The two blockers as they stood before the visibility change

Both are now resolved; they are kept here because the evidence ledger references
them and because they explain why the contract's repository boundary changed.

**1. Actions billing.** Every one of the 18 historical runs failed in about two
seconds with zero steps. The check-run annotation is unambiguous:

> The job was not started because recent account payments have failed or your
> spending limit needs to be increased. Please check the 'Billing & plans'
> section in your settings

**2. Environment protection is plan-gated.** Creating the contract's
`release-candidate` environment returns HTTP 422:

> Please ensure the billing plan supports the required reviewers protection rule

GitHub still creates the environment shell when this fails. The shell was
deleted: an environment with no required reviewer would let the exact-release
gate reach private evidence without the approval the contract requires, which is
worse than having no environment at all.

## Public-repository consequence worth naming

A self-hosted runner attached to a public repository is a known attack surface:
a pull request from a fork can otherwise execute untrusted code on the owner's
machine. Two independent controls stand in the way here, and both were verified:

- `exact-release-gate` carries
  `if: github.event_name != 'pull_request' || github.event.pull_request.head.repo.full_name == github.repository`,
  so a fork pull request never reaches the self-hosted runner at all.
- The job additionally targets the `release-candidate` environment, whose
  required reviewer must approve the deployment before it runs.

Repository settings also report `default_workflow_permissions: read` and
`can_approve_pull_request_reviews: false` (re-verified 2026-08-25).

The third control is now in place. The fork pull-request approval policy was
`first_time_contributors`, which would have let a returning outside contributor
reach the self-hosted runner without a fresh approval; it is now
`all_external_contributors`. This had to precede runner registration rather than
follow it, so it is done ahead of the owner's step:

```
gh api --method PUT \
  repos/HawzhinBlanca/HawaVoClean/actions/permissions/fork-pr-contributor-approval \
  -f approval_policy=all_external_contributors
```

The repository currently has 0 forks, so nothing was exposed in the interval.

## Proven on 2026-08-25: the gate's first execution

`exact-release-gate` had never run once in this repository's history — it was
pending on every pull request, including the four that merged. With the runner
registered and the `release-candidate` deployment approved by its required
reviewer, it executed, and it failed at its first step:

```
Run test -n "$HAWAVOCLEAN_RELEASE_EVIDENCE_ROOT"
  HAWAVOCLEAN_RELEASE_EVIDENCE_ROOT:
##[error]Process completed with exit code 1
```

That is the gate behaving correctly: it fail-closes on absent private evidence
rather than hydrating nothing and reporting success. `required` then failed on
`test "$EXACT_RELEASE_GATE" = success`, which is also correct.

Two things this bought that reading the workflow could not. It established the
gate is genuinely reachable — the runner labels match, the environment approval
path works, the checkout and toolchain steps all pass. And it exposed the
ordering error corrected below: the evidence root is a *prerequisite* of the
branch ruleset, not a step after it.

## What only the account owner can do

Both are now done; they are kept here because they are the two decisions that
cannot be derived from the repository.

1. ~~**Register the release runner**~~ *Done 2026-08-25*, on the Apple-silicon
   host, labelled exactly `self-hosted, macOS, ARM64, hawavoclean-release`. The
   contract requires it to be ephemeral or single-purpose; it is ephemeral. See
   the recorded deviation on the account it runs under, below.
2. ~~**Choose the private-evidence mirror path**~~ *Done 2026-08-26*:
   `/Users/hawzhin/HawaVoCleanEvidence`, outside every runner workspace, holding
   the fourteen manifest-named files at their manifest paths, mode `0444` inside
   `0555` directories. Verified before use by hydrating a fresh clone that
   contained no `test_output/` at all.

## The two owner-only steps, as commands

Both need the owner's own credentials on the owner's own machine, which is why
they cannot be scripted from here.

**1. Register the runner.** *Done on 2026-08-25.* Recorded here because it is
reproducible and because the published checksum step is not optional:

```
V=2.336.0
EXPECT=8e8839c49b7060b6b2154f4931f815df330c27f167d53ef2239ee3dfce28b079
mkdir -p ~/actions-runner && cd ~/actions-runner
curl -sSL -o "actions-runner-osx-arm64-$V.tar.gz" \
  "https://github.com/actions/runner/releases/download/v$V/actions-runner-osx-arm64-$V.tar.gz"
shasum -a 256 "actions-runner-osx-arm64-$V.tar.gz"   # must equal $EXPECT before extracting
tar xzf "actions-runner-osx-arm64-$V.tar.gz"

./config.sh --unattended \
  --url https://github.com/HawzhinBlanca/HawaVoClean \
  --token "$(gh api --method POST \
      repos/HawzhinBlanca/HawaVoClean/actions/runners/registration-token --jq .token)" \
  --name hawavoclean-release-mac --labels hawavoclean-release --ephemeral
nohup ./start-ephemeral-runner.sh > run.log 2>&1 &   # NOT `nohup ./run.sh &` — see below
```

`self-hosted`, `macOS` and `ARM64` are applied by the installer; only
`hawavoclean-release` needs declaring. The registration token is short-lived —
mint it into the command as above rather than writing it down.

`--ephemeral` satisfies the contract and has an operational consequence worth
stating: the runner deregisters and the listener exits after **one** job, so
the runner must be started again for each subsequent job. No launchd service was
installed.

### Why the listener is not started with `nohup ./run.sh &`

*Measured 2026-08-26. The gate's second execution failed on this and nothing
else.*

A POSIX shell sets SIGINT and SIGQUIT to `SIG_IGN` for a background job,
`nohup` adds SIGHUP, and `SIG_IGN` is **inherited across both fork and exec** by
every descendant. CPython compounds it: at interpreter start it installs
`default_int_handler` only when SIGINT is currently `SIG_DFL`, so an inherited
ignore is left in place and a self-directed SIGINT becomes a silent no-op.

Starting the listener with `nohup ./run.sh &` therefore hands *every job the
runner ever runs* a process tree in which SIGINT cannot be delivered. Ten tests
in the release gate's default suite failed for that reason alone:

```
tests/unit/test_publication_transaction.py::…recover_idempotently[…-SIGINT]  (6)
tests/chaos/test_interrupt_cleanup.py::…no_partials_and_no_orphans[2]
tests/chaos/test_multipass_interrupt.py::…no_partials_and_no_litter[2]
tests/unit/test_watchdog.py::test_watchdog_interrupts_then_hard_exits
tests/unit/test_watchdog.py::test_a_child_reparented_before_arming_still_tears_down
```

The product was correct throughout, and one of the failures proves it: the
watchdog *noticed* the ignored disposition and escalated to SIGTERM, which is
precisely the behaviour
`test_watchdog_escalates_to_sigterm_when_sigint_is_inherited_ignored` exists to
assert. The suite was reporting a broken harness, accurately.

`start-ephemeral-runner.sh` restores the three dispositions and execs `run.sh`:

```sh
exec /usr/bin/python3 -c '
import os, signal, sys
for name in ("SIGINT", "SIGQUIT", "SIGHUP"):
    signal.signal(getattr(signal, name), signal.SIG_DFL)
os.execvp(sys.argv[1], sys.argv[1:])
' ./run.sh "$@"
```

Reproduced and fixed on the same machine in one command each: the ten tests fail
under `nohup sh -c 'pytest …' &` and all forty-seven pass under
`nohup python3 sigdfl.py sh -c 'pytest …' &`. The tests were not touched.

`trap - INT` cannot substitute for this: POSIX states that signals ignored on
entry to a non-interactive shell cannot be trapped or reset from within it.

### Recorded deviation: the runner account

`docs/operations.md` asks for the runner to be registered "under a dedicated
unprivileged OS account… do not leave repository credentials or unrelated
working copies on it." It is registered as the owner's own account on the
owner's workstation, alongside `gh` credentials and this checkout. `--ephemeral`
and single-purpose-per-job are satisfied; **account isolation is not**. This is
a stated deviation rather than a met requirement, and it is the reason
`fork_pull_requests_forbidden` and the `release-candidate` reviewer gate carry
more weight here than the contract assumes.

**2. Choose the evidence mirror path**, outside any runner workspace, then:

```
gh variable set HAWAVOCLEAN_RELEASE_EVIDENCE_ROOT \
  --repo HawzhinBlanca/HawaVoClean --body '/absolute/path/you/choose'
```

Only the path is the owner's decision; setting the variable afterwards is not.
*Done 2026-08-26* with `/Users/hawzhin/HawaVoCleanEvidence`. Build the mirror by
copying each `input` / `reference_audio` / `reference_report` named in
`evidence/release/audio-regressions.json` to the same relative path under the
root, verifying its `*_sha256` on write. Prove it before trusting CI to it:

```
git clone --no-hardlinks . /tmp/hydrate-proof     # no test_output/ at all
uv run --frozen python scripts/hydrate_release_evidence.py \
  --source-root ~/HawaVoCleanEvidence \
  --destination-root /tmp/hydrate-proof \
  --manifest /tmp/hydrate-proof/evidence/release/audio-regressions.json
```

## Remaining activation order

Order matters here:

1. ~~Open a pull request from `codex/v3.3-release` into `main`.~~ Done: PR #1
   registers the `release` workflow, and its hosted jobs now execute for the
   first time in the repository's history.
2. Let the hosted jobs run green. `exact-release-gate` stays pending until the
   self-hosted runner exists — that is expected, and `required` cannot report
   success before it does.
3. **Only after the runner exists AND the evidence root is set** apply the
   branch ruleset to `main`. Two dependencies, not one, and both were learned
   the hard way rather than read:

   - `required` cannot report success while `exact-release-gate` has no runner
     to execute on, so applying the ruleset before the runner locks `main`
     outright.
   - The gate's first step is `test -n "$HAWAVOCLEAN_RELEASE_EVIDENCE_ROOT"`.
     With the variable unset the gate fails immediately at hydration, so
     `required` fails too. Step 5 below is therefore a *prerequisite of this
     step*, not a successor to it — the numbering said otherwise until the gate
     ran for the first time on 2026-08-25 and proved it.

   The rules to apply are already proven
   to be accepted on this plan: pull request with one approval, dismiss stale
   reviews, require last push approval, required conversation resolution,
   required linear history, no deletion, no force push, and a strict
   `release / required` status check.
4. ~~Create `release-candidate` with the owner as required reviewer and a
   protected-branches-only deployment policy.~~ Done: the environment exists and
   carries `required_reviewers` and `branch_policy`.
5. ~~Set `HAWAVOCLEAN_RELEASE_EVIDENCE_ROOT`.~~ Done 2026-08-26. **This had to
   happen before step 3**, for the reason recorded there: the gate refuses to
   hydrate without it, so the `release / required` context cannot report success
   until it is set. The first job to run with it set hydrated all fourteen
   artifacts and went on to fail on the launcher defect recorded above, not on
   the evidence.
6. **T3.3 proof:** open a disposable pull request that deliberately fails a gate
   and record that it cannot be merged. T3.3 stays open until that exists.

### A third dependency of step 3, discovered 2026-08-26

The ruleset is **one-way for a single-maintainer repository.** It requires one
approving review with `enforce_admins: true`, and GitHub does not let anyone
approve their own pull request; administrator enforcement closes the override
as well. From the moment it is applied, the owner cannot merge anything alone.

That is the contract behaving as designed — T7 assumes an independent
challenger and U4 assumes a protected merge — but it makes ordering load
bearing in a way the numbering above does not show: **everything that still
needs to land must land before step 3.** That was the `web-resolve` engine-build
step (STATUS blocker 7), which edits `.github/workflows/ci.yml` and therefore
re-pins both `ci.workflow_sha256` and the contract's own digest; it is folded
into the same pull request as this runbook change, so one gate execution covers
both U1 items rather than two.

Step 6 is unaffected: the T3.3 proof needs a pull request that *cannot* merge,
which is the state protection creates rather than one it obstructs.

## Note on merge settings

The repository currently allows merge commits, and `main` now contains three of
them (PRs #1, #2 and #3). Existing history is unaffected — `required_linear_history`
governs new pushes — but once the rule is active every later pull request must
squash or rebase. Disabling merge commits in repository settings is optional and
makes that constraint visible in the UI before someone attempts a merge; it is
left set as it is, because it changes how pull requests are landed today and
that is a workflow decision rather than a governance one.

### `required_linear_history` conflicts with evidence pinning

Found while preparing this pull request's own merge, and it is not a workflow
preference — it is a way to lose evidence.

`evidence/release/audio-regressions.json` pins `d37ef29a…` as the
`reference_source_commit` of all six regenerated cases. That commit is
*introduced by this pull request*: it is not yet an ancestor of `main`. Squash
and rebase merges both rewrite branch SHAs, so merging this branch either of
those ways orphans the very commit it pins — the same failure as deleting the
branch that held `3f2ec25b…`, arrived at from the opposite direction, and the
one `evidence/baseline-lowband-branch` exists to remember.

Today a merge commit avoids it, because the original SHAs survive. Once
`required_linear_history` is active a merge commit is no longer available, so
**any later change that both regenerates evidence and pins it to its own commit
must tag that commit before merging.** `tests/unit/test_release_evidence.py`
accepts a tag as a durable anchor precisely for this, alongside trunk ancestry.

Done here rather than left as advice: `evidence/audio-regression-references-v3.3`
now points at `d37ef29a…` and is pushed, so the pin holds under every merge
strategy including the ones the ruleset will force.
