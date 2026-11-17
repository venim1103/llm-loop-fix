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

## Current State (2026-11-17)

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
  Section237 now binds the exact asmjit root/LICENSE.md Zlib-style grant and
  genuine CMake descriptor, not its implementation or compilation. Next
  complete recursive metadata/legal/gitlink inventory for a genuine scoped
  source subset; GPU/HSTU source and source-family adoption remain excluded.
  Sections238/239 reconstruct 231 Git-tree identities and bind asmjit
  db/LICENSE.md to the already reviewed root Zlib notice, not complete embedded
  rights. Proposed non-GPU FBGEMM plus asmjit is531 blob descriptors/12.7 MB,
  not531 unique objects; seven FBGEMM Gitlinks remain unresolved.
  Sections240/241 preserve a logged-copy parser failure and verify identical
  forwarded v2/SHA1/shallow/filter advertisements on both public repositories.
  FBGEMM's exact failed response was reused without a duplicate request; one new
  asmjit probe passed with unchanged project refs/config/status.
  Sections242-244 retain the high-level filtered-fetch private-config failure,
  audit actual cached bodies offline and import only verified packs into two
  separate clean metadata caches. These contain230 unique trees/two pinned
  commits/zero blobs, with unchanged empty refs/config/HEAD and no configured
  remote/promisor state. Acquisition quarantine is not the clean source store;
  the old failure remains failed. Sections245/246 derive530 unique blobs from
  531 source-bound descriptors and acquire precisely the genuine25,693-byte
  asmjit root CMake blob under one explicit SHA want. Complete stored framing/
  hash/size and unchanged project/private-ref checks pass; no CMake execution.
  Sections247-249 acquire exactly527 remaining objects, reuse three cached
  notices/CMake objects and verify all531 source files/bytes/executable modes
  in an inert declared non-GPU path snapshot. Six other Gitlinks remain
  unprepared. Actual source-rights census finds Arm Apache/BSD declarations,
  complete Broken Unlicense text and a PDCLib public-domain provenance pointer;
  a191-line review receipt is retained with a two-line per-window budget
  overrun (aggregate cap passed, not the complete review). Sections251/252
  retrieve/read complete canonical Apache terms with202 lines below each320
  cap; Arm notices, redistribution/NOTICE/patent/trademark conditions remain.
  Section253 identifies actual public nonfork DevSolar/pdclib metadata and
  CC0-1.0 pointer.254 retains a filename-selector failure after exact current
  commit/root verification;255/256 acquire/read actual COPYING.CC0 and both
  surrounding scope documents,188 guarded lines. This binds present-day CC0
  waiver/fallback/nonwaived-rights conditions and declared allocator/IANA
  references, not their upstream certification or historical rewritten-sort
  identity. Remaining native frontend/dependency rights still need actual
  verification. Section250's explicit12/13-line guard and narrow
  10-line source review pass without rewriting249 as successful.
  This is not semantic HSTU absence or complete source-rights/native admission; no full
  GPU archive, automatic lazy/submodule fetch, project ref mutation or native
  build/adoption is authorized by this evidence.
  Sections257/258 bind the same candidate Torch2.13.0 root, cached CMake,
  actual BSD-style root conditions and37-path preparation manifest, not37
  verified dependencies. Actual paths differ from four NNPACK section labels;
  two branch hints are not Gitlink pins. Ordinary/editor Torch2.10.0 stays
  unchanged.259-262 bind/read the actual flag forwarder, pass four sealed
  standalone OFF/0/predefined-variable/cache cases and bind root ordering
  (forwarding24/preparation30/option281), not executed root preparation.
  Helper is unchanged; no native build/cache file/GPU/model operation occurred.
  Section263 resumes the unused metadata-only contract after explicit
  continuation: four exact tree responses reconstruct successfully and bind
  all37 actual paths to immutable Gitlinks, including the existing FBGEMM pin.
  No missing/extra declarations; four label/path differences and two branch
  hints are preserved. This is metadata identity, not37 licensed/prepared
  dependencies.264 binds the actual preparation owner and seven nested links:
  only googletest/json exactly alias Torch pins; cutlass is a different fork.
  NCCL has a separately gated fetch path outside37 declarations; automatic
  submodule init requires ALL listed folders empty plus Git/.git, not any
  missing folder.265's ten-dependency notice batch failed on ONNX's real
  LICENSES directory after8 requests; unused requests remain closed.
  Two rootBSD-style notices were completely read.266 separately binds/
  reads ONNX's actual three legal templates and pybind11's contribution scope,
  reusing prior complete Apache/CC0 readings for exact aliases. Templates do
  not assign licenses to unseen implementation files.267 binds and completely
  reads the seven previously unattempted root notices; nlohmann's LICENSES
  remains explicit.268 fails its32-KiB cap on the actual34,670-byte GPL
  template, with no raw fetch.269 separately binds/reads exact measured
  supplemental terms and .reuse/dep5: upstream declares GPL for
  tests/thirdparty/imapdl/*, CC0 for hedley.hpp, Apache for test Fuzzer and
  BSD-3 for tools/amalgamate, over its default MIT. This is source-reported
  scope, not actual header/build verification or native adoption. Both
  failed allowances remain failed. Main legal reads and remaining dependency/
  preparation/NCCL boundaries remain; no implementation/build.
  Root grants and isolated wiring do not prove embedded rights/HSTU exclusion.
  Section270 inventories the complete pinned Torch main namespace:23 subtree
  responses,1,310 reconstructed trees,22,836 entries/21,490 blob descriptors/
  202,691,924 declared blob bytes; all37 Gitlinks join with no differences.
  Four symlinks remain metadata only.24 legal filename leads include ten
  non-notice test markers/scripts.271 binds all14 actual legal documents,
  reuses root LICENSE and completely reads1,407 new lines under guards.
  Source/binary notices, dual-license choices, LLVM exceptions/third-party
  caveats and source-specific attributions remain. Concurrentqueue explicitly
  points to an embedded semaphore Zlib license; roots are not a full header/
  source census.272 acquires one exact inert CPU header for that permission
  inspection but fails its leading-prefix hypothesis.273 reuses it with zero
  GETs and completely reads two comment spans (26 lines), including the actual
  Jeff Preshing grant/three restrictions at44-64. The272 failure remains
  failed. This single unexecuted header is not a native source-family
  adoption, compiled exclusion, model execution or original-wheel/HSTU grant.
  Section274 now fetches/audits the genuine pinned Torch commit plus1,309
  unique trees, corresponding to1,310 inventory tree paths, then imports
  identical bodies into a clean no-promisor/no-ref metadata cache. Zero new
  source blobs; this is not source preparation/adoption or native permission.
  Section275 distinguishes34 actual inline blob descriptors in five groups
  from36 third_party Gitlinks.276 finds both Valgrind headers byte-identical
  in the installed ordinary package, verifies candidate Git/RECORD hashes
  without downloads, and completely reads130 leading-comment lines. Their
  bespoke file-only BSD-style four conditions must not be relabeled standard
  BSD-3; the stated GPLv2 rest-of-Valgrind scope is not adopted. Two inert
  copies are retained, with installed files/environments unchanged. Next
  Sections277/278 reuse six bodies and acquire exactly28 remaining blobs/
  904,226 bytes, yielding a verified34-object/1,365,907-byte inert inline
  payload index. Original modes are metadata, not executable checkout
  parity; no submodule/GPU subtree/archive or native execution follows.
  Sections279/280 bind the actual literal census and completely read21
  selected windows (353 displayed lines), retaining queue BSD/Boost and
  miniz MIT/Unlicense distinctions/additional holders.281 reads the newly
  exposed patent tail: author opinion, not patent clearance.282/283 acquire
  the exact cited PNG Gist and read Alex Evans' actual byline/public-domain
  declaration; retain its reported anonymous fork and unverified historical
  join, not complete native rights.284 reuses byte-exact cached Dependencies
  and reads the actual miniz Bazel target: only c/h there, but CMake's observed
  CRC flags do not establish its source list. Rosetta example provenance/
  whole-graph CMake scope remain unresolved.285/286 bind two actual owners
  and the678-byte serialize leaf: parent calls serialize, its leaf explicitly
  appends miniz.c (not examples) to Caffe2_CPU_SRCS, and the parent declares
  torch_cpu from that variable unless BUILD_LIBTORCHLESS. This is static
  source-list evidence, not executed configuration or whole-graph exclusion.
  287 preserves public GitHub's failed tree:1 request (maximum depth0),
  despite the local fixture pass.288's new commit-only/tree:0 plus canonical
  root-API import passes for pinned FP16: two metadata objects/zero blobs,
  unchanged project/private-state guards.289 completely reads its exact MIT
  root grant (Facebook/Georgia Tech/Google notices), not embedded/dependency
  clearance or source adoption.290/291 read FP16's real build requirements
  and find its actual Dependencies.cmake caller, not the earlier ARM-flag
  false lead: vendored-path guards request genuine PSimd/FP16 directories
  and cache-OFF tests/benchmarks (not FORCE). Standalone FP16 can otherwise
  download PSimd even with tests disabled. No configuration executed.
  Sections292-296 bind PSimd's real commit/root, read its MIT grant and
  prepare a fresh byte/mode-verified four-file CMake-input subset47,162
  bytes, not a full checkout/Torch tree. Header census/review includes
  CUDA/OpenCL qualifiers; no source/configuration execution or native
  admission follows.297-299 bind/read all seven FP16 install sources and
  prepare a genuine ten-file/44,102-byte input subset for tests/benchmarks
  OFF with an explicit real PSimd path, not default/full-checkout coverage.
  Empty __init__.py is the actual empty Git object; PeachPy imports remain
  unexecuted.300 now passes one genuine standalone host component configure/
  compile/run with actual C99/C++11 commands and six tiny half patterns per
  consumer, unchanged14 source files and no scoped dependency subprocess/
  download command. This is not Torch/GPU/ABI or broad numerical admission.
  Sections301-305 bind the next actual CPUinfo owner/root and complete
  42-tree metadata inventory, read root/bundled-CLog two-clause BSD grants,
  not implementation or native admission. No Gitlinks/symlinks in that
  inventory; reported20-MB whole-tree bytes are not an acquisition budget.
  Sections306-322 acquire31 genuine CPUinfo host C/header/template files/
  289,890 bytes; all eight private headers read,177 declared literal includes/
  four guarded mock exclusions. GNU <cpuid.h> is external, not self-resolution.
  Full931-line CMake credits active pkg-config code to JsonCpp. The33-file/
  336,403-byte subset is now byte/mode-exactly prepared with separate
  required notices, not a full checkout. JsonCpp candidate commit/root
  now authenticated; root notice/actual helper completely read. Helper has
  its own MIT OR CC0-1.0/Jan Tojnar2020 declaration, not just JsonCpp ownership.
  Predeclared strict13-command function/two caller-argument comparisons
  match current code, not original historical version. Actual external
  GNU cpuid.h/full runtime exception and private-use sections checked.
  One serial standalone host attempt compiled20 genuine C units and
  passed C99/C++11 initialization, rejecting unsupported fallback;
  measured29 local/135 external dependencies. Pkg-config ON retains
  upstream blank Version/URL. No install/full native/data/pilot/
  confirmation admission or jurisdictional CC0 assumption. Reassess
  remaining native preparation dependencies, not another CPUinfo run.
  Sections323-334 bind the actual conditional pthreadpool caller, genuine
  commit/root (two metadata objects/zero blobs), reused real BSD notice
  and complete220-line CMake. C11 sources/optional ccache and unconditional
  FXdiv download when its source variable is undefined are declared,
  even with tests/benchmarks OFF. FXdiv's actual Torch pin/root is now
  authenticated; complete MIT notice and117-line CMake read. Its real
  13,149-byte header is now acquired/fully read, not substituted with the
  different installed Torch header. Initial70-marker census missed three
  Hacker's Delight references; preserved old result plus separate complete
  supplement yields73. Default GNU division code's referenced permission
  is unresolved, holding genuine preparation/native use. One GitHub-MCP
  search failed at transport (no results/remote execution unverified);
  do not retry unchanged or claim no source exists. Public REST discovery
  then pins one mirror's canonical90-entry root and completely reads its
  exact267-byte README: it names the book author/second edition and original
  hackersdelight.org package, not actual permission or primary authority.
  Next separately bounded historical-primary-site permission/source-scope
  lookup; no live-domain/archive link followed yet. No pthreadpool/FXdiv
  configuration, fake target/system bypass/full-native/data/pilot/
  confirmation admission.

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