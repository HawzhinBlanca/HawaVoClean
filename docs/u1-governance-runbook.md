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
| `HAWAVOCLEAN_RELEASE_EVIDENCE_ROOT` | Not set | Mirror location decision |
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

1. **Register the release runner** on the Apple-silicon host, labelled exactly
   `self-hosted, macOS, ARM64, hawavoclean-release`, from
   Settings → Actions → Runners. The contract requires it to be ephemeral or
   single-purpose.
2. **Choose the private-evidence mirror path**, outside any runner workspace,
   and it will be set as the repository variable
   `HAWAVOCLEAN_RELEASE_EVIDENCE_ROOT`.

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
nohup ./run.sh > run.log 2>&1 &
```

`self-hosted`, `macOS` and `ARM64` are applied by the installer; only
`hawavoclean-release` needs declaring. The registration token is short-lived —
mint it into the command as above rather than writing it down.

`--ephemeral` satisfies the contract and has an operational consequence worth
stating: the runner deregisters and the listener exits after **one** job, so
`run.sh` must be started again for each subsequent job. No launchd service was
installed.

**2. Choose the evidence mirror path**, outside any runner workspace, then:

```
gh variable set HAWAVOCLEAN_RELEASE_EVIDENCE_ROOT \
  --repo HawzhinBlanca/HawaVoClean --body '/absolute/path/you/choose'
```

Only the path is the owner's decision; setting the variable afterwards is not.

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
5. Set `HAWAVOCLEAN_RELEASE_EVIDENCE_ROOT`. **Do this before step 3**, for the
   reason recorded there: the gate refuses to hydrate without it, so the
   `release / required` context cannot report success until it is set.
6. **T3.3 proof:** open a disposable pull request that deliberately fails a gate
   and record that it cannot be merged. T3.3 stays open until that exists.

## Note on merge settings

The repository currently allows merge commits, and `main` now contains three of
them (PRs #1, #2 and #3). Existing history is unaffected — `required_linear_history`
governs new pushes — but once the rule is active every later pull request must
squash or rebase. Disabling merge commits in repository settings is optional and
makes that constraint visible in the UI before someone attempts a merge; it is
left set as it is, because it changes how pull requests are landed today and
that is a workflow decision rather than a governance one.
