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
  routine approval, a next-task choice, or a finish/continue vote.
- Do not invoke `vscode_askQuestions` or equivalent agent-controlled
  questionnaire/choice dialogs. Resolve routine choices using standing
  authority; preserve protected boundaries rather than manufacturing an
  approval or finish/continue cycle.
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

- An explicit project continuation keeps the agreed implementation session
  active. Do not voluntarily issue a final Copilot answer merely because a
  task, checkpoint, commit, push, review, or progress report is complete.
  Give checkpoint updates in commentary and continue the next useful agreed
  work unless a current stop request or the completion gate below applies.
- Treat continuation as an active execution loop, not a series of checkpoints
  requiring another user message. After a checkpoint or scoped correction,
  invoke the next useful authorized tool action in the same active turn.
  A promise to continue later is not continuation. Keep progress updates in
  commentary; never use a final answer to deliver checkpoint narration or a
  transcript of claimed ongoing work.
- Only a current explicit user pause/stop or equivalent voluntarily ends this
  loop. The completion gate below handles unavoidable execution constraints,
  not discretionary checkpoint exits or routine reasons to request permission.
- Process new user messages delivered by the platform before starting the next
  slice and before considering a final answer. Apply the newest compatible
  direction promptly; background review is not a requirement to wait for
  another approval after every checkpoint. Absence of a delivered stop is not
  permission to expand scope, but it is not implicit finish permission either.
- An explicit current `stop`, `pause`, or `finish` request takes priority. Start
  no new slice. Cancel or contain owned work safely, restore any trial mutation
  where possible, retain diagnostics, and leave a concise handoff. Do not delay
  stopping for unrelated checks, commits, pushes, or a confirmation dialog.
- After a pause, resume implementation only on an actual continuation request;
  a status question or terminal notification alone is not one. An earlier stop
  does not prohibit a newly requested task or later explicit continuation.
- Finish permission applies only to the current requested scope. Never carry
  it into later continuation, new work, reloads, or context/cursor gaps. A
  changed checkpoint is not covered by an earlier checkpoint's finish vote.
- Respect explicitly bounded requests. Instruction/document creation, a
  review, or a proposal request is not permission to start separate
  implementation, acquisition, or inference. Complete only that scope and
  report its outcome. Do not relabel an active open-ended continuation as a
  bounded request just because its current checkpoint is finished.
- A scoped correction or status request within an active continuation is not
  itself a stop. Resume the already-agreed queue after the detour unless the
  user explicitly limits work to that request. This preserves existing active
  authority; it does not create authority for separate work or experiments.
- A task blocker, pending CI/build, missing routine test approval, or tool
  selection/invocation failure is not by itself permission to end an active
  continuation. Diagnose it, use existing routine authority, repair a local
  defect, or continue another useful authorized prerequisite. Stopping a
  failed or budget-exhausted experiment does not end the whole session.

### Completion Gate for Active Continuation

Before ending without a current explicit stop, check and record all of these:

- The newest user messages actually delivered by the platform contain no
  overlooked continuation or course correction. Do not infer finish permission
  from silence, a completed checkpoint, or an earlier session's stop.
- The plan's next useful agreed task and genuinely unblocked prerequisites
  have been considered. Any remaining blocker identifies its exact dependency,
  attempted bounded recovery, and why another authorized path cannot proceed.
  A missing routine plan/test budget must be prepared, not used as an excuse.
- Registered work still in progress has been distinguished from failed work.
  While an actual authorized job or input is pending, use supported bounded
  event-based waits and process delivered messages before deciding to end.
  A pending external job is not an exhausted path when such waiting is
  available. Reuse its identity; do not duplicate it or repeatedly poll it.

If useful authorized work or supported waiting remains, continue it rather
than issue a final answer. Only an established execution constraint with no
such path permits a constrained end-of-turn handoff without an explicit stop.
State the exact runtime/tool/authority limit, retained work, and pending action;
do not present this as user-requested completion or ask for a finish vote.

Repository instructions guide decisions; they cannot guarantee indefinite turn
lifetime, receive messages the platform has not delivered, or execute after the
platform ends the session. Never use sleep loops, dummy jobs, unrelated work,
unauthorized scope expansion, or renewed hard budgets to simulate persistence.
On an explicit stop, give a concise handoff with commit IDs, verification,
remaining boundaries, and worktree state, then start no further slice.

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

## Current State (2026-11-16)

- Start with [the harness handover][handover] and the [post-WIP review][latest-review].
  The plan owns historical receipts, failed attempts and budgets; the former
  long Current State is retained in Git at `c9a7965`, not repeated here.
- Latest source is `ccceb81`, committed/pushed on `main` to `origin/main`;
  [its CI][current-ci] completed successfully at actual run `43826903795`,
  with both CPU jobs independently checked. The plan retains source checkpoints
  and historical pending observations. No owned local command/model/trial is active.
- The user confirms there were no user-authored edits: the previously retained
  workflow/handover notes are assistant-created. [Section 210][current-continuation]
  owns their reviewed cleanup and the new explicit continuation/approval.
  Commit reviewed documentation, preserve ignored assets, and continue the
  agreed prerequisites rather than treating that checkpoint as a finish.
- WP00/WP01 declared gates pass, including exact published-artifact replay.
  WP02-06 and WP08/09 have offline source/recorded/toy fixtures, not empirical
  repair. WP06 has measured stored-control comparison and frozen construction
  identities, not validated norm matching or causal utility. WP07 has bounded development-grid admission,
  complete recorded joins, separate usability/quality/health counts and pinned
  baseline reuse, plus explicit zero-baseline and finite-metric legacy health
  guards. Attached thinking flags match stored text and declared repetition
  spans match actual periodic token blocks. Complete-run admission binds tail
  terminal-EOS exclusion to the frozen profile, not inferred native stop detail.
  These checks do not prove scan completeness, tail vocabulary/threshold
  validity or calibrated pathology. Positional legacy counts are not independent utility selection.
  Independent selection/confirmation is absent. Validated export remains
  disabled; recorded checkpoint/slice exports assert byte-only scope.
  Do not restore fabricated acceptance or drop auxiliary checkpoint tensors.
- Prior retained CPU 3.12/3.10 suites passed 1,437 tests each at `ccceb81`;
  declared Ruff/whitespace and owning diagnostics have no errors. The unchanged
  selected CUDA environment has two pre-existing optional math-verify import
  warnings; both CPU suites exercise that stack. Both CPU offline smokes passed
  at `c928aca`, without pretrained downloads; this is prior wiring evidence,
  not fresh smokes for the import/signature-neutral periodic-block/EOS predicates.
  Python `>=3.10`, CPU/CUDA locks, editor selection and asset pins are unchanged.
  Fixture/count/cache checks do not establish native execution or empirical repair.
- All earlier acquisition/resolver/decoder/template/rights/empirical attempts
  remain closed. The new approval permits separately budgeted in-scope
  prerequisite acquisition/setup/admission/experiments after their gates pass,
  not an unbounded run or unseen license assent. Section 211 retrieved the
  exact referenced NumPy code permission from a dated author-site archive;
  raw payload/index digest parity passes, not complete transitive certification.
  Sections 212-214 inspected all 22 exact native members and installed NumPy
  2.3.5 in the isolated native-contract environment with a passing numerical
  smoke, complete stored-file/notice audit and seven-distribution consistency.
  The bounded readelf/installer failures remain failed; narrower metadata and
  single-thread installation passed without raised caps. Ordinary profiles
  are unchanged. Torch/HSTU build permission, complete native compatibility,
  dataset quarantine/admission, source MATH/official IFEval compatibility,
  effective native stopping, measured pilot and accepted-repair gates remain.
  Prior explicit SDK/CUDA/CUTLASS acceptance is recorded, not permission for
  unseen HSTU terms or complete transitive rights; do not repeat that assent.
- [Sections 199-209][source-prerequisites] own the bounded source evidence and
  dependency reassessment. The next useful WP07 decision requires independently
  prepared eligible clean controls and a predeclared rule; confirmation size/
  calibrated uncertainty follows a measured pilot. Missing rights/native/data/
  pilot evidence must be obtained and checked under a new prospective bounded
  contract, not replaced by generic fixtures or recycled old attempts.
  Approval does not establish those facts or package completion.
  Sections 202/207 narrowed earlier immediate-queue handoffs; identified
  numerical, legacy-metric and text/periodic/EOS-claim slices are now checked.
  The neighboring token-tail assessment is complete within existing declarations:
  content/threshold claims require the actual eligible policy, not invented
  classes/thresholds or generic fixtures. The direct NumPy prerequisite is now
  checked for isolated private-use wiring. Sections 215/216 diagnosed comment
  syntax in the exact exposed example and derived its raw-bound normalized
  exclusion marker through an explicit comments-only source projection followed
  by unchanged strict parsing. Old failures remain failed; no dataset admitted.
  Section 217's complete 1,319-prompt literal comparison has no raw/normalized/
  near findings at the frozen 0.9 threshold, not semantic independence or data
  admission. Sections 218/219 preserve the failed Contents metadata cap and
  corrected filename-only Git tree acquisition; all 1,319 original JSONL rows
  match frozen benchmark questions/raw answers at the separately pinned current
  source commit. This is paired-artifact parity, not historical unpinned source
  loader identity. Next reassess native/official-source/semantic and independent
  clean/pilot prerequisites. Express-NVIDIA permission is not supplied by
  user approval; native/pilot/independent clean/confirmation gates remain.
  Section 220's new exact bundled langdetect README review names the original
  Java port/date and profile-generation recipes, but supplies no exact bundled-
  profile corpus/revision grant; MIT metadata/Apache notice disagreement remains.
  No profile/table adoption or build follows. Section 221 measures ten
  selected-method cases on each retained CPU:
  five normalized kwargs refused/five canonical kwargs matched, with two
  representative portable-controller cases using an explicit synthetic checker.
  This is not official compliance/whole-registry integration. Keep the generic
  drift guard; future source-specific constraint projection needs complete
  identity/resource compatibility and immutable raw-to-derived data roles.
  Section 222 owns the remaining execution gate: actual permitted native build,
  source/resource rights, independent admitted clean/confirmation inputs and
  measured pilot are missing. No live job/trial is active, and more generic
  fixtures cannot discharge those dependencies. Retain progress, not fabricated
  admission or renewed old allowances.
  Renewed continuation rechecks `07cfa85`, caches/interpreter and headroom.
  Sections 223/224 preserve a failed search transport and a passing exact primary
  NLTK data-page observation with no model permission identified; no resource
  adoption follows. Next assess a separately scoped source-built native path
  excluding the quarantined component before acquisition/build/compatibility
  admission; a flag cannot certify original-wheel absence or change frozen
  evaluation inputs.
  Sections 225/226 identify a potential same-source `USE_FBGEMM=OFF` build,
  but exact pre-build preparation still requires real listed submodules and
  FBGEMM/asmjit unless separately reviewed system dependencies replace them.
  No fake files or uncontrolled recursive init; this is not proof HSTU source
  itself is required. No source build/adoption occurred and `nvcc` is not found
  in the bounded tooling check. Next inspect exact process-local compiler-kit
  metadata and source acquisition/ownership boundaries under finite contracts.
  Sections 227-231 establish separate exact CUDA13.0 compiler13.0.88/
  CCCL13.0.85/runtime13.0.96 inputs, not the inference graph's compiler13.5.
  Acquired compiler SDK notices match the accepted notice; all 1,872 stored
  files pass full SHA-256/size/CRC. Ten actual x86-64 ELF declarations and
  bounded host readelf NEEDED/version metadata pass; temporary ELFs are removed.
  These are static metadata, not runtime resolution or transitive certification.
  CCCL prefixes identify BSD-3-Clause/Apache-2.0 WITH LLVM-exception leads.
  Sections 232/233 retain a manual display-budget failure and pass a separate
  counted exact SDK internal-use/third-party/NVIDIA-GPU scope review. Actual
  host libc/selected library headers support planning, not ABI certification.
  Section 234 now installs four exact SDK packages in a separate private CPU3.12
  environment and identifies nvcc13.0.88 once; installed stored-file hashes and
  ordinary metadata/editor selection parity pass. This is compiler identification,
  not header/GPU/model execution or native engine adoption; CCCL is uninstalled.
  Return to exact Torch/FBGEMM/asmjit source ownership/preparation boundaries.
  HSTU/full-source rights, independent data/grader/resource gates and a measured
  empirical pilot remain absent. Ordinary CPU/CUDA environments are unchanged.
  Sections 235/236 retain a wrong internal-directory assumption and verify the
  actual FBGEMM `external/asmjit` Gitlink/configuration plus Meta's exact root
  BSD notice. This supports scoped genuine source preparation research, not a
  source checkout/HSTU exclusion or superseded nested terms. Next exact asmjit
  root permission/core legal-tree declarations precede any source acquisition.

[design]: docs/DESIGN.md
[plan]: docs/PLAN.md
[review]: docs/REVIEW.md
[readme]: README.md
[container-guide]: .devcontainer/README.md
[ci]: .github/workflows/ci.yml
[packaging]: pyproject.toml
[handover]: docs/PLAN.md#198-copilot-harness-handover-2026-11-13
[latest-review]: docs/REVIEW.md#9-post-wip-code-review-2026-11-13
[source-prerequisites]: docs/PLAN.md#209-wp02wp07-frozen-profile-token-tail-eos-admission-2026-11-13
[current-ci]: https://github.com/venim1103/llm-loop-fix/actions/runs/43826903795
[current-continuation]: docs/PLAN.md#210-reviewed-documentation-cleanup-and-renewed-continuation-2026-11-16