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
| Protected `main` requiring `release / required` | Deliberately deferred | Needs one green run first |
| Self-hosted `hawavoclean-release` runner | Not registered | Owner's machine |
| `HAWAVOCLEAN_RELEASE_EVIDENCE_ROOT` | Not set | Mirror location decision |

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
`can_approve_pull_request_reviews: false`. Before the runner is registered, set
*Settings → Actions → Fork pull request workflows from outside collaborators* to
require approval for all outside collaborators.

## What only the account owner can do

1. **Register the release runner** on the Apple-silicon host, labelled exactly
   `self-hosted, macOS, ARM64, hawavoclean-release`, from
   Settings → Actions → Runners. The contract requires it to be ephemeral or
   single-purpose.
2. **Choose the private-evidence mirror path**, outside any runner workspace,
   and it will be set as the repository variable
   `HAWAVOCLEAN_RELEASE_EVIDENCE_ROOT`.

## Remaining activation order

Order matters here:

1. ~~Open a pull request from `codex/v3.3-release` into `main`.~~ Done: PR #1
   registers the `release` workflow, and its hosted jobs now execute for the
   first time in the repository's history.
2. Let the hosted jobs run green. `exact-release-gate` stays pending until the
   self-hosted runner exists — that is expected, and `required` cannot report
   success before it does.
3. **Only then** apply the branch ruleset to `main`. Applying it earlier would
   require a status context that has never reported, which would block the very
   pull request that lands the workflow. The rules to apply are already proven
   to be accepted on this plan: pull request with one approval, dismiss stale
   reviews, require last push approval, required conversation resolution,
   required linear history, no deletion, no force push, and a strict
   `release / required` status check.
4. Create `release-candidate` with the owner as required reviewer and a
   protected-branches-only deployment policy.
5. Set `HAWAVOCLEAN_RELEASE_EVIDENCE_ROOT`.
6. **T3.3 proof:** open a disposable pull request that deliberately fails a gate
   and record that it cannot be merged. T3.3 stays open until that exists.

## Note on merge settings

The repository currently allows merge commits. The branch rule for
`required_linear_history` will reject them on `main` once active. Disabling
merge commits in repository settings is optional but makes the constraint
visible in the UI before a merge is attempted.
