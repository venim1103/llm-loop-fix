# Project Agent Workflow

These instructions apply repository-wide to assistants working on llm-loop-fix,
within higher-priority platform, tool, security, and safety requirements.
The assistant owns routine execution within the agreed scope; the user reviews
in the background and can redirect or stop work.

## Goal and Documents

llm-loop-fix is a research prototype for training-free weight-space repair of
demonstrable small-language-model generation failures. Follow the evidence
sequence: exact published-artifact replay, matched behavioral reproduction,
controlled automatic discovery, then verified export. No edit is a valid result.
The skeleton is not a validated repair method or a contract that must survive
unchanged. Editor assistance, attention/quantization research, and broad model
support remain conditional follow-ups.

- [DESIGN.md][design]: technical contracts, scope, rationale, acceptance gates.
- [PLAN.md][plan]: current milestone, actionable queue, material blockers, and
  dated execution evidence. Historical next-step proposals are not a live queue.
- [REVIEW.md][review]: historical publication/repository analysis. Do not rewrite
  old failures as current passes or assume dated bug findings remain unfixed.
- [README.md][readme]: public purpose, capabilities, maturity, and licensing;
  not routine test counts, session handovers, or commit history.
- [Container guide][container-guide], [CI][ci], and [pyproject.toml][packaging]:
  reusable setup, interpreter/backend locks, and ordinary validation.

Keep stable workflow rules here. Keep Current State short; replace it when the
milestone, blocker, or authority changes. Historical receipts belong in the plan
or Git, not an expanding copy in this file. Use existing documents and
reference-style links; do not create a new ledger for routine work.

## Start With the Current Outcome

- Read Current State and the [live queue][queue], then only the design sections,
  owning implementation, nearby tests, and evidence needed for that outcome.
  Do not repeat a repository audit or follow historical "continue next" text.
- Check the actual worktree, branch/upstream, relevant files, and applicable
  instructions. Preserve unrelated changes, untracked files, and ignored assets.
  Imported material under `TMP/` is not the task queue.
- For runtime work, verify the selected interpreter, required tools/assets, and
  resource headroom after interruption. Reuse environments/downloads and verify
  interrupted commands before repeating them. Documentation-only work does not
  require a fresh model/cache/toolchain audit.
- Distinguish proposed, implemented, tested, integrated, and measured.
  Historical verification is not a fresh check or renewed experiment allowance.

## Implement First; Investigate to Unblock

- Choose the smallest useful change that advances the current package's exit
  gate: a working feature, reproduced/fixed bug, integrated path, or decisive
  research result. Prefer an end-to-end increment over more general scaffolding.
  Do not invent unrelated features merely to produce code.
- Before ordinary engineering, state a concise intent: the behavior to change,
  owning code/tests, and cheapest discriminating check. Use the existing task
  entry or session; do not require a separate experiment contract for every
  file read, helper edit, syntax check, or unit-test invocation.
- Inspect current owners/tests and prior art, make the smallest coherent edit,
  then run the focused check immediately. Fix defects exposed in the same slice
  before expanding scope. Reuse existing helpers/tests; avoid speculative
  abstractions and fixtures that cannot settle the missing real-world evidence.
- A prerequisite-only investigation must name the exact blocked implementation
  or experiment, the missing fact, and how resolving it changes the next action.
  Bound the investigation to that question. Do not recursively turn every new
  source reference into another audit or seek blanket provenance/legal/ABI
  certification beyond the active contract. Actual rights and safety gates
  still apply; unresolved required evidence is a blocker, not assumed permission.
- During implementation continuation, after two consecutive investigation/
  documentation-only slices without a useful product-code integration, package
  exit-gate result, or resolution that makes a named product integration or
  real-model run executable, reassess the critical path before doing more.
  Return to useful implementation, perform an already-authorized gated check,
  or give a precise blocked handoff. Private analysis-helper fixes, renamed
  contracts, extra inventory, and documentation commits do not reset this guard.
  Do not hide an unbounded investigation inside one nominal slice.
- An assistant-owned parser/guard/checklist defect is not missing upstream
  evidence. Diagnose and fix routine helper bugs within the engineering task;
  do not generate a chain of research checkpoints around them. This prospective
  rule does not reopen previously closed attempts or override explicit caps.
- Diagnose failures before retrying. After three substantively different failed
  approaches to the same blocker, stop that path and retain concise evidence.
  Renaming a script, contract, or experiment does not reset a hard budget.
- Report progress as behavior fixed, integration demonstrated, gate passed, or
  a decision-changing negative result. Pages, hashes, test totals, file counts,
  commits, and static declarations alone are not progress toward empirical repair.
- Use standing authority for routine reversible code/tests, compatible docs,
  controlled checks, and checkpoint commits/pushes. Do not ask for routine
  planning/approval or a finish/continue vote, including through questionnaire
  dialogs. Do not spawn subagents unless explicitly requested by the user.

## Keep Records Proportionate

- Update the live queue when the next action or material blocker changes.
  Record one concise entry per coherent verified increment, meaningful
  experiment, or material blocker: change/question, relevant inputs, check,
  result, and remaining limit. Reuse that entry for related diagnostics.
- Keep raw logs, helper output, and research assets under ignored `.runs/`.
  Commit only needed compact evidence/manifests with applicable rights.
  Do not add a numbered PLAN section or research JSON record for every lookup,
  failed helper command, notice fragment, whitespace check, or instruction edit.
- Acquisition/native/model experiments still require the prospective contracts
  below. Group genuinely related preparation under a meaningful bounded scope;
  do not create micro-contracts to simulate progress or renew a closed allowance.
- Preserve dated historical results, failures, immutable identities, and hard
  limits. Keep history reachable without requiring readers to traverse it before
  finding the current queue. An old proposal is neither a current task nor assent.

## Continuation, Blocking, and Stopping

- During an explicit continuation, execute the next useful authorized action
  after a checkpoint; a commit or progress report alone is not a reason to stop.
  Process newly delivered user direction before starting another slice.
- Continue only meaningful in-scope work. If the critical path and genuinely
  useful independent prerequisites are blocked, state the exact dependency,
  bounded recovery attempted, and smallest missing input/authority or next
  action, then stop. Do not manufacture peripheral audits, generic fixtures,
  documentation, or jobs to satisfy a persistence rule.
- Before a blocked handoff, check the live queue and owned pending jobs.
  A real authorized job still running is not a failure: reuse its identity and
  supported bounded/event-based waits rather than duplicate or repeatedly poll
  it. Do not use sleep loops or dummy work to simulate indefinite execution.
- A missing routine plan/test budget is work to prepare, not an external
  blocker. A genuine protected rights/data/native/experiment dependency cannot
  be removed by more routine fixtures or silently enlarged authority.
- A current explicit pause/stop takes priority: start no new slice, safely
  contain owned work, restore trial mutations, and give a concise handoff.
  Do not delay stopping for unrelated checks, commits, pushes, or a dialog.
- Respect bounded requests. A documentation cleanup, review, status question,
  or proposal is not permission for separate implementation or experiments.
  Finish that scope and stop when requested. After a pause, only an actual
  continuation request resumes implementation; a status question or terminal
  notification does not.
- Within an active continuation, a scoped correction alone is not a stop;
  resume the already-agreed queue afterward unless the user limits the request.
  Never carry earlier finish permission or continuation authority across a
  newer pause or explicitly bounded task.

## Research, Data, and Mutation Boundaries

- Autonomy is not evidence. Use `source_reported`, `locally_reproduced`,
  `implemented_unvalidated`, and `proposed` labels. Hash parity proves an
  artifact property; toy/smoke checks prove wiring, not useful model repair.
  Shorter output or fewer cap hits alone is not reasoning improvement.
- Keep `source_reproduction` and `validated_repair` separate. Preserve exact
  base/tokenizer/backend revisions, generation/grader settings, paired
  identities, protected answers, and immutable dataset roles. Never select on
  confirmation/benchmark answers, silently recycle holdouts, or relax frozen
  acceptance thresholds.
- Preserve the immutable baseline, numerical/edit invariants, exact
  stored-dtype audits, exception-safe rollback, and complete checkpoint/MTP
  tensors, including runtime-omitted auxiliary state. Never leave an unaccepted
  edit active. Failed restoration is fatal. Export only through atomic staging
  and fresh-load verification; byte-only replay/export is not accepted repair.
- Before acquisition, native installation/build/probes, or pretrained-model
  experiments, verify the active package and prerequisite gates; prospectively
  record pinned inputs, applicable rights, disk/memory/time/token/attempt caps,
  outputs, retention, and cleanup.
  Routine authority is not blanket experiment permission. A source paper's
  request/token counts are not a budget. Full evaluation needs a measured pilot;
  failed/aborted attempts consume the declared allowance.
- Public metadata/source inspection can resolve a concrete prerequisite, but
  does not authorize acquisition, adoption, or execution beyond its own scope.
  Do not reopen closed runs by relabeling them as setup or routine debugging.
  Keep large-memory/GPU jobs serial, check actual shared-host headroom, and stop
  on OOM, invalid identity, or budget exhaustion.
- Pin and inspect third-party code before execution. Do not automatically
  deserialize untrusted pickle or enable arbitrary remote code. Respect
  artifact/dataset/script rights; repository MIT does not relicense them.
- Do not change project goals, protected data boundaries, frozen profiles,
  holdouts, acceptance tolerances, or hard experiment caps; start training,
  an editor service, or deferred model/edit families; or publish unvalidated
  model artifacts without fresh user direction.
- Required active-package dependency/container maintenance is delegated:
  prefer installed tools, preserve Python `>=3.10`, justify changes, update
  affected inputs/locks, and validate matching profiles. Do not silently mutate
  the selected environment, perform unrelated upgrades, or bypass TLS, hashes,
  security controls, or privilege boundaries.
- Branch creation, remote changes, force-push/history rewriting, user-data
  deletion, external accounts/payments/licenses/contact, and privileged host
  changes require explicit direction. Do not request credentials or route
  secrets through the model. User approval cannot grant third-party rights.
  Retain local work if synchronization is blocked.

## Verification and Checkpoints

Ordinary tests use controlled fixtures without pretrained downloads or a GPU.
Tiny random-model integration is allowed only as explicitly offline wiring,
not a benchmark or repair claim. Run the focused check after substantive edits.
Before a meaningful source checkpoint, run from the repository root:

```sh
python -m pytest -q
ruff check src tests .devcontainer/smoke.py
git diff --check
```

Use the interpreter/backend commands in the [container guide][container-guide].
Run shellcheck for bootstrap changes, offline smoke for inference/dependency/
container wiring, and actionlint when available for workflow changes.
Ruff's declared `E4/E7/E9/F` policy is authoritative; do not add a blanket
formatting gate or reformat untouched code. Do not weaken assertions or rerun
unchanged successful gates for reassurance.

For documentation/instruction-only changes, check content, references, scope
consistency, and whitespace; disclose that the code suite was not rerun.
Unavailable hosted CI, image builds, and research runs remain unverified.

Review the intended diff and stage only checkpoint files. Commit meaningful
verified increments using Conventional Commits and a useful bullet body.
Check branch/upstream/worktree first and push periodically through the configured
upstream without routine approval. If the remote advanced, fetch/reconcile
conservatively and validate affected work; never discard either history or
force-push. A blocked push does not invalidate local work or authorize expansion.

## Current State (2026-11-18)

- **WP02 execution is blocked after the explicit November 18 continuation.**
  The focused prompt/native-record/grading slice passes its existing offline
  checks; no concrete unblocking source defect was established. Resume only
  from the exact dependencies in the [live queue][queue], not another historical
  "next" receipt or recursive source census. Closed budgets are not renewed.
- The last production-source checkpoint is `ccceb81`. Prior CPU 3.10/3.12
  full regressions, offline smokes, and source CI remain historical checks.
  The selected `/opt/venv` lacks the math-grading stack; the retained complete
  CPU profile was used without installing packages or switching the editor.
  Python floor, locks, environments, pinned assets, and ignored research assets
  are unchanged.
- WP00/WP01 declared gates pass, including exact published-artifact replay.
  Matched real-model behavioral reproduction, automatic discovery/causal
  utility, independent confirmation, and accepted-repair export do not pass.
  Substantial offline/recorded/toy infrastructure exists; it is not empirical
  repair. Validated export remains disabled; recorded exports are byte-only.
- The critical path is WP02 matched behavioral reproduction: an admitted native
  backend with effective stopping/complete checkpoint preservation, permitted
  source-compatible graders and role-admitted data, then a measured bounded
  pilot. WP07 additionally lacks independent eligible clean/confirmation
  inputs. Static source inventories cannot substitute for these gates.
- All closed acquisitions/attempts and frozen constraints in the plan remain
  closed. In particular, native approaches 369/373/376 permit neither a delayed
  376 probe nor a fourth renamed candidate; the failed 387/389 source census
  is not reopened. The 78,235-byte system C++ `limits` body remains unread/
  unhashed under its unchanged 64-KiB candidate envelope: no segmentation,
  hashing, or adoption follows from this cleanup.
- Existing SDK/CUDA/CUTLASS assent does not supply unseen HSTU, source-family,
  model-resource, or dataset rights. No whole-native integration is admitted.
  The plan retains the precise rights, failed attempts, and partial component
  evidence without turning them into a new source-audit queue.
- Section 391's committed grader inventory is static evidence only.
  Section 392's completed private getter-key analysis is retained unintegrated;
  do not rerun it or automatically open another documentation-only slice.
  No owned local command, model run, or trial is active.

[design]: docs/DESIGN.md
[plan]: docs/PLAN.md
[queue]: docs/PLAN.md#4-immediate-implementation-queue
[review]: docs/REVIEW.md
[readme]: README.md
[container-guide]: .devcontainer/README.md
[ci]: .github/workflows/ci.yml
[packaging]: pyproject.toml
