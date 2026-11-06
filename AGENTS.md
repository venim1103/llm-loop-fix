# Project Agent Workflow

These instructions apply repository-wide to assistants working on llm-loop-fix,
within higher-priority platform, tool, security, and safety requirements. The
user reviews in the background and provides course corrections; the assistant
owns routine planning and execution within the agreed project scope.

## Project and Documents

llm-loop-fix is a research prototype for training-free weight-space repair of
demonstrable small-language-model generation failures. Follow the evidence
sequence: exact published-artifact replay, matched behavioral reproduction,
controlled automatic discovery, then verified export. No edit is a valid result.
The existing source is a skeleton, not a validated repair method or a contract
that must survive unchanged. Editor assistance, attention/quantization research,
and broad model support remain conditional follow-ups.

- [DESIGN.md][design]: scope, technical contracts, rationale, and acceptance gates.
- [PLAN.md][plan]: work-package dependencies, current status, dated execution
  evidence, failed checks, budgets, and the next concrete action.
- [REVIEW.md][review]: dated publication/repository analysis and research findings.
  Preserve its historical baseline; do not rewrite old failures as current passes.
- [README.md][readme]: public purpose, capabilities, maturity limits, and licensing.
  Do not add routine test counts, session handovers, or commit history.
- [Container guide][container-guide]: reusable setup and interpreter/backend lock
  instructions. [CI][ci] and [pyproject.toml][packaging] define ordinary gates.

Keep stable workflow rules here. Replace Current State when it changes; do not
turn this file into an append-only experiment ledger or duplicate the roadmap.
Use existing documents and reference-style links. This audience split does not
authorize unrelated cleanup of existing documentation.

## Start With Verified Context

- Read Current State and the immediate queue/latest relevant evidence in the
  plan. Read only the design sections and review findings needed for the next
  slice; do not repeat a whole-repository audit at every checkpoint.
- Check the actual worktree, branch/upstream, owning code, nearby tests, and
  applicable instructions. Preserve user changes and untracked files. Imported
  material under `TMP/` is not this project's task queue; do not alter or stage
  it incidentally.
- Recheck selected interpreter, available tools, caches, pinned assets, and
  resource headroom after interruption or reload. Reuse existing environments
  and downloads. Verify an interrupted command's outcome before repeating it.
- Treat proposed, adopted, implemented, tested, integrated, and measured as
  different states. Old evidence is not a fresh check; a historical run budget
  is not renewed permission. Current standing authority governs routine work.

## Work Autonomously

- During project continuation, independently choose and execute the first useful
  unblocked task in the agreed queue. Choose a conservative recommended default
  for compatible implementation decisions; record material assumptions and
  rationale rather than asking the user to plan each step.
- Routine authority covers reversible local code/tests, compatible design and
  planning updates, bounded synthetic checks, required regression gates, and
  meaningful checkpoint commits/periodic normal upstream pushes. Do not request
  routine approval, a next-task choice, or a finish/continue vote. Do not use
  agent-controlled questionnaire/choice dialogs for routine decisions.
- For each slice, record a finite hypothesis, owning files, prerequisites,
  focused check, resource/attempt limits proportionate to risk, outputs, and
  rollback/stop condition. A missing routine plan or test budget is work to
  complete independently, not an approval blocker.
- Gather enough nearby evidence for a falsifiable hypothesis, make the smallest
  useful edit, and immediately run its cheapest discriminating check. Repair
  the same slice if it exposes a local defect before expanding scope. Reuse
  existing tests/helpers; avoid unrelated cleanup and speculative abstractions.
- At material checkpoints, reassess the design and queue against new evidence,
  record results, commit a verified increment, and take the next useful in-scope
  step. A checkpoint, commit, review, or progress report alone is not a reason
  to end an open-ended implementation session.
- Mark isolated blockers precisely and continue authorized prerequisites or
  another genuinely unblocked task. For example, unavailable image-build tools
  do not prevent a pinned WP01 artifact inventory. Do not bypass a dependency
  gate, claim the blocked package complete, or manufacture peripheral work.
- Diagnose failures before retrying. After three failed approaches to the same
  problem, retain the evidence, mark the blocker, and choose another useful
  path. Do not repeatedly poll or rerun an unchanged failed command.
- Give concise progress updates about changes, evidence, and remaining limits.
  Do not spawn subagents unless the user explicitly requests delegation.

## Pause, Stop, and Task Boundaries

- Process new user messages delivered by the platform before starting the next
  slice. Apply the newest compatible direction promptly; background review is
  not a requirement to wait for another approval after every checkpoint.
- An explicit current `stop`, `pause`, or `finish` request takes priority. Start
  no new slice. Cancel or contain owned work safely, restore any trial mutation
  where possible, retain diagnostics, and leave a concise handoff. Do not delay
  stopping for unrelated checks, commits, pushes, or a confirmation dialog.
- After a pause, resume implementation only on an actual continuation request;
  a status question or terminal notification alone is not one. An earlier stop
  does not prohibit a newly requested task or later explicit continuation.
- Respect bounded requests. Instruction/document creation, a review, or a
  proposal request is not permission to start separate implementation,
  acquisition, or inference. Complete that request and report its outcome.
- Continue useful authorized work during an active implementation session, but
  do not promise indefinite turn lifetime, background message monitoring, or
  execution after the platform ends the session. Use supported event-based
  waiting only for actual authorized work; never sleep loops, dummy jobs, or
  invented work to keep a turn alive.
- If every useful in-scope path is blocked or platform/tool limits prevent
  further execution, report the exact constraint, retained state, and pending
  action. Do not pretend to remain active.

## Research, Data, and Mutation Boundaries

- Autonomy is not evidence. Use `source_reported`, `locally_reproduced`,
  `implemented_unvalidated`, and `proposed` claim labels. Hash parity proves an
  artifact property; a toy/smoke pass proves wiring, not useful language-model
  repair. Shorter output or fewer cap hits alone is not reasoning improvement.
- Follow the design's separate `source_reproduction` and `validated_repair`
  profiles. Preserve exact base/tokenizer revisions, generation/grader settings,
  complete paired identities, and immutable dataset roles. Never select using
  confirmation or benchmark answers, recycle a holdout silently, or relax a
  frozen acceptance threshold to obtain a favorable result.
- Keep numerical/edit invariants, immutable baseline identity, exact stored-dtype
  auditing, exception-safe restoration, and atomic fresh-load-verified export.
  Never leave an unaccepted trial edit active or publish a partial artifact as
  a completed repair. Failure to restore is a fatal validation error.
- Routine authority is not blanket authority for model experiments. Before
  any acquisition or empirical run, verify that it belongs to an active package,
  its prerequisite gates pass, and pinned inputs, licenses, disk/memory/time/
  token/attempt budgets, retention, and cleanup are recorded prospectively.
  Plan an in-scope bounded replay/pilot independently; never infer its budget
  from the source's 50-request or 32K-token protocol. Full evaluation needs the
  measured pilot envelope first. Failed/aborted attempts consume the budget.
- Public metadata/source inspection and preparing missing experiment contracts
  are useful prerequisites, not permission to launch an unbounded run. Keep
  expensive inference separate from ordinary checks; run large-memory/GPU jobs
  serially. Check actual hardware and headroom rather than inheriting another
  project's GPU limits. Stop on OOM, budget exhaustion, or invalid identity.
- Pin and inspect third-party code before execution. Do not deserialize
  untrusted pickle or enable arbitrary remote code automatically. Store local
  assets and raw results under ignored `.runs/<experiment_id>/`; commit only
  compact, license-compatible evidence and manifests, never weights or caches.
- Preserve published-artifact exactness and upstream model/data/script rights.
  Repository MIT licensing does not relicense downloaded artifacts or datasets.

## Protected Changes and Escalation

Do not treat standing authority as permission to change the project's goal,
protected data boundaries, frozen profiles/holdouts, acceptance tolerances, or
hard experiment caps; start training, an editor service, or deferred model/edit
families; or publish unvalidated model artifacts. A compatible technical choice
is delegated, but expansion of explicit constraints needs fresh user direction.

For a protected action, retain the conservative state, record the precise
missing authority/evidence and recommended next action, and continue useful
authorized prerequisites. Do not turn the missing fact into a favorable claim
or repeatedly ask the same routine question.

Dependency/container maintenance required by an active package is delegated:
prefer installed tools, preserve the Python floor, justify changes, update all
affected inputs/locks, and validate the matching profiles. Do not perform
unrelated upgrades, silently mutate the selected environment, or disable TLS,
hash checks, security controls, or privilege boundaries as a workaround.

Branch creation, remote changes, force-push/history rewriting, deletion of user
data, external accounts/payments/licenses/contact, and privileged host changes
need explicit direction. Never request credentials or route secrets through the
model. Retain local progress if authentication or synchronization is blocked.

## Verification and Checkpoint Commits

Ordinary tests use controlled fixtures and require neither pretrained downloads
nor a GPU. Tiny random-model forward/generation integration is allowed when
explicitly scoped as offline wiring, not a benchmark or repair claim.

After each substantive edit, run the focused check first. Before a meaningful
source checkpoint, run the applicable regression gate from the repository root:

```sh
python -m pytest -q
ruff check src tests .devcontainer/smoke.py
git diff --check
```

Run shellcheck for bootstrap changes, offline smoke for inference/dependency/
container wiring, and actionlint when available for workflow changes. Use the
interpreter/backend-specific commands in the container guide. Do not introduce
a new blanket formatting gate or reformat untouched files; Ruff's declared
`E4/E7/E9/F` policy, not the copied template's rules, is authoritative.

Required focused/regression checks for new in-scope work have standing authority.
Do not weaken assertions or rerun a successful unchanged gate for reassurance.
For documentation/instruction-only changes, check content, reference targets,
scope consistency, and whitespace; disclose that the code suite was not rerun.
Report unavailable hosted CI, image builds, or research runs as unverified.

Review the diff and stage only the checkpoint's files. Use Conventional Commit
subjects with a useful bullet body. Commit meaningful verified increments and
push periodically through the current branch's configured upstream without a
routine approval prompt. Check branch/tracking/worktree state first; if the
remote advanced, fetch and reconcile conservatively without discarding either
history and validate affected work. Never force-push or include unrelated user
changes. A blocked push is not a reason to discard commits or stop other work.

## Current State (2026-11-06)

- WP00 local baseline and CI definition are committed in `ea03aa0`; the earlier
  planning/fixture checkpoint is `c13a81e` and research review is `1d000ff`.
- The plan records passing clean CPU checks on Python 3.10/3.12 and existing
  CUDA wiring. Hosted CPU CI passed at `ff80c16`; fresh CPU/CUDA image builds
  remain unverified. Recheck tools/results rather than assuming the old
  limitation persists.
- Python `>=3.10` is retained. The 3.10 CPU lock uses Transformers 5.20.2;
  the 3.12 CPU/CUDA locks use 5.22.1, which requires Python 3.11+. Locks are
  not interchangeable. Reuse ignored `.runs/wp00/` environments if still valid.
- No local source-artifact replay, full behavioral reproduction, or automatic
  real-model repair is established. BF16 auditing, rollback, and utility-aware
  selection remain pending in their roadmap packages.
- WP01's pinned metadata inventory and proposed replay limits are recorded in
  the plan. Metadata checks pass; no checkpoint/NPZ was acquired or replayed.
- On project continuation, check the remaining WP00 gates first. If unavailable,
  retain their blocker and consult WP01's prepared inventory and prerequisites;
  inspect bounded hosted image checks without claiming WP00 closure or starting
  a weight download. Follow the plan's dependency gates thereafter.

[design]: docs/DESIGN.md
[plan]: docs/PLAN.md
[review]: docs/REVIEW.md
[readme]: README.md
[container-guide]: .devcontainer/README.md
[ci]: .github/workflows/ci.yml
[packaging]: pyproject.toml