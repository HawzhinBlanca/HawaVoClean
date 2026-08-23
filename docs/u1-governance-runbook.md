# U1 — GitHub governance activation runbook

State of the live repository as of the T3.3 evidence entry (ledger sequence 38).
Everything here was measured against `HawzhinBlanca/HawaVoClean`, not assumed.

## Where U1 actually stands

| Contract requirement | State | Blocked by |
|---|---|---|
| `immutable-release-tags` ruleset on `refs/tags/v*` | **Active** (id 21254065) | — |
| Release workflow is the attested file | **Verified** (`50c7974a…`) | — |
| Branch rules available on this plan | **Confirmed by probe** | — |
| Any GitHub Actions job starting | **Blocked** | Actions billing |
| `release-candidate` environment with a required reviewer | **Blocked** | Private-repo plan |
| Protected `main` requiring `release / required` | Deliberately deferred | Needs one green run first |
| Self-hosted `hawavoclean-release` runner | Not registered | Owner's machine |
| `HAWAVOCLEAN_RELEASE_EVIDENCE_ROOT` | Not set | Mirror location decision |

## The two blockers, exactly as GitHub reports them

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

## What only the account owner can do

1. **Repair Actions billing** at <https://github.com/settings/billing> — fix the
   failed payment method and raise the spending limit above zero. This is the
   U1 checkpoint proper and cannot be delegated to an agent.
2. **Decide the private-repo boundary.** Required-reviewer environments need
   GitHub Pro on a private repository. The plan's stated preference is *fix
   billing, keep the repo private*; making the repository public is the
   documented alternative and makes both Actions minutes and environment
   protections free. Either satisfies U1 — they are not equivalent in every
   other respect, so this is a real decision.
3. **Register the release runner** on the Apple-silicon host, labelled exactly
   `self-hosted, macOS, ARM64, hawavoclean-release`, from
   Settings → Actions → Runners. The contract requires it to be ephemeral or
   single-purpose.
4. **Choose the private-evidence mirror path**, outside any runner workspace,
   and it will be set as the repository variable
   `HAWAVOCLEAN_RELEASE_EVIDENCE_ROOT`.

## What happens the moment billing is repaired

In this order, because the order matters:

1. Open a pull request from `codex/v3.3-release` into `main`. This registers the
   `release` workflow and produces the first real `release / required` context.
2. Let the hosted jobs run green. The `exact-release-gate` job will stay pending
   until the self-hosted runner and the environment exist — that is expected.
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
