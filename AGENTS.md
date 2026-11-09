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

## Current State (2026-11-09)

- WP00 local baseline and CI definition are committed in `ea03aa0`; the earlier
  planning/fixture checkpoint is `c13a81e` and research review is `1d000ff`.
- The plan records passing clean CPU checks on Python 3.10/3.12 and existing
  CUDA wiring. Hosted CPU CI passed at `ff80c16`; fresh CPU/CUDA dependency
  images and offline CPU checks passed at `a56d60d`, closing WP00. CUDA-image
  construction is not hosted GPU-runtime evidence.
- Python `>=3.10` is retained. The 3.10 CPU lock uses Transformers 5.20.2;
  the 3.12 CPU/CUDA locks use 5.22.1, which requires Python 3.11+. Locks are
  not interchangeable. Reuse ignored `.runs/wp00/` environments if still valid.
- WP01's exact published-artifact replay is locally reproduced: both BF16
  tensor hashes/support and all 486 unaffected tensors pass. The two model
  files remain immutable/ignored in `.runs/wp01-p50c-replay/`; the plan records
  compact provenance and consumed budgets. No generation was run.
- Full behavioral reproduction, automatic discovery, and validated repair
  remain unverified. Full immutable proposal/baseline identity integration and
  utility selection remain pending in their
  roadmap packages; raw-dtype audit fixtures alone do not close those gates.
- WP01's safe replay utility and 23 offline byte/transaction checks pass on
  Python 3.10/3.12, including enforced six-GiB containment; hosted source CI
  passes at `bf9fa90`. The automatic repair skeleton is outside this result.
- WP02's initial item/generation records, immutable roles, role-sensitive dataset
  digests, and complete expected-key joins pass offline fixtures. The legacy
  CLI has not been integrated. Frozen data/profile/code run declarations now
  bind complete items and planned experiment/profile/sample identities at optional
  math/controlled-IFEval, diagnosis and native-output handoffs. Canonical dataset
  reorder parity and pre-grader reference/instruction drift refusal pass fixtures.
  This is declared-only identity, not authenticated execution/model provenance;
  unbound legacy compatibility and baseline/proposal/utility integration remain
  unvalidated. Run-specific canonical JSON now restores complete frozen nested
  declarations under a one-MiB UTF-8 cap and refuses altered checksums/schema/
  scope/fields and mismatched caller expected identity. Self-consistent checksums
  without an external pin are not authentication. Complete candidate batch
  validation now preflights independently planned keys, exact per-item prompt/
  thinking snapshots, declared profile/native stops and aggregate record/token/
  text envelopes, preserving original immutable quality/health records. This is
  recorded declaration handoff, not authenticated engine/template/model execution.
  Bounded serial grading now preflights complete identities and framing before
  any parser/checker, then refuses result changes outside quality/new health or
  loss of attached health. Independent unavailable/error outcomes and fatal
  no-partial-batch exits pass fixtures. Sixteen-record admission and existing
  per-operation guards are explicit; no end-to-end deadline is claimed. Strict
  complete-run readonly diagnosis now requires expected candidate, exact prompt/
  context snapshots and diagnostic/development-only roles before the conservative
  report. Missing/literal observations remain insufficient and caps neutral; no
  report authorizes localization. Separate Linux `diagnose-records` now requires
  an external run pin/candidate and bounded regular non-symlink UTF-8 inputs,
  returns one readonly JSON report before pretrained loading, and refuses bad
  identity/roles/collections/bytes/file types without reporting. Legacy
  experimental `diagnose` is unchanged; `run` loading/export stays disabled.
  This is offline recorded evidence, not authenticated execution or accepted
  repair. Explicit wrapped/unwrapped nested text discovery now precedes the
  legacy first-ModuleList fallback; vision-before-text toy canonical names/
  indices and both retained CPU smoke profiles pass. This is naming maintenance,
  not a validated Qwen3.5 adapter, position/cache/probe or empirical result.
  A separately bounded 15,176-parameter random hybrid Qwen3.5 wrapper now passes
  exact four-projection names/shapes and one finite eight-token CPU forward on
  both retained packages. Offline/thread/time/four-GiB gates and both smoke
  profiles pass; measured 3.12 regression uses one construction/forward and about
  444 MiB peak process RSS. This is tiny random wiring, not pretrained adapter,
  native EOS or empirical repair validation. A separate fixed-tolerance single
  eight-token/seven-plus-one cached-prefix check passes on both CPUs with
  unchanged registered parameter hashes. Measured 3.12 difference is about
  `1.49e-8`, two constructions/four forwards and 445 MiB process peak RSS.
  This does not validate broader position/cache, padding/batching, native
  generation, empirical probing or accepted repair. Return to unblocked recorded
  evaluation and experiment/baseline/proposal identity prerequisites next.
  Hosted bound-run CI passed at `1b362a6`; persistence CI passed at `a942fe9`.
  Complete-batch CI passed at `245d53d`. These do not validate subsequent grading
  source or empirical provenance. Batch grading CI passed at `b383ae4`; it does
  not validate subsequent strict diagnosis or empirical provenance. Source text
  extractors and initial pinned-profile
  fingerprints pass offline fixtures; backend adapters remain pending. Initial
  independent finish/thinking health records pass fixtures with explicit prompt
  context; repetition/tail calibration and legacy integration remain pending.
  A bounded prompt-only textual overlap
  audit passes fixtures; actual dataset audit/review and semantic independence
  remain unverified. This is
  not behavioral reproduction or empirical profile validation.
- WP03's bounded exact tandem observer now returns frozen nonoverlapping token
  spans under a frozen spec digest, with exact block ends before continuation.
  Empty/threshold/partial spans and separate blocks pass offline fixtures; caps
  are not inferred loops. Backend failures and over-budget inputs are explicitly
  unavailable. This is literal pattern observation, not semantic degeneration
  or empirical calibration. Optional health attachment now binds exact generated
  token hashes, restores frozen nested JSON and enforces consistent observation
  versions/thinking requirements/specs within and across paired samples. Different
  observed patterns remain comparable; no-spec defaults remain unchanged. Frozen
  declared-tokenizer tail specs now report exact token-bound suffixes, excluding
  one terminal declared EOS only on observed stop; EOS-only, thresholds, non-tail
  whitespace and unavailable outcomes pass fixtures. Tail health now requires
  matching evaluation-profile/tokenizer/EOS identity before scanning, preserves
  exact token/profile-bound JSON and enforces tail-spec comparison identity.
  Semantic tail-degeneration flags stay unavailable for unreviewed mappings.
  Mapping labels are not reviewed decoder semantics. Math/controlled-IFEval
  grading now preserves attached immutable health across quality outcomes and
  refuses incompatible actual framing before grader execution. Backend handoff
  and empirical calibration remain pending. Legacy classification is unchanged.
  Standalone observer CI passed at `c038424`; repetition health CI passed at
  `0aa8fd2`; standalone tail CI passed at `1350762`. Subsequent profile-bound
  health CI passed at `4748556`; grading handoff CI passed at `36862c2`.
  Declared sampling bridge CI passed at `c622d26`.
  Exact pinned tokenizer JSON/config sizes and source hashes are now inspected
  and retained read-only with compact Apache-2.0 provenance. Builtin ByteLevel
  metadata is not decoder runtime validation. Tokenizer EOS `248046` differs
  from model text-config EOS `248044`; generation-config/native-engine precedence
  is an unresolved profile prerequisite. No real token mapping is adopted.
  Download/retention/request caps pass; end-to-end inspection timing is unverified
  and its one-attempt allowance is consumed. Plan separately bounded stop/decoder
  prerequisites without reusing these acquisition attempts.
  Cached exact-revision metadata has no generation-config path. The one stable
  SDK-candidate inspection stopped on GitHub's `v0.33.0rc3` result despite its
  false prerelease flag; two metadata attempts, no code assets/runtime, clock
  deadline pass. Do not adopt it or expand that frozen one-version allowance.
  Effective backend stopping remains blocked; bounded independent offline
  decoder prerequisites may proceed without real profile/mapping adoption.
  A separate tokenizers `0.23.3` CPU fixture passes eight synthetic strings and
  two productive prefixes on Python 3.10/3.12 with exact token snapshots and
  explicit special-token retention. Thirty calls/profile, serial process caps
  and persisted deadline pass. This is low-level decoder fixture evidence, not
  a full token-class table, native-engine EOS or empirical degeneration validation.
  Bounded readonly no-edit diagnosis now binds complete diagnostic/development
  keys and declared dataset/profile/record snapshots. Correct/usable framing and
  available negative literal-pattern channels can abstain as `no_patch`; caps
  alone do not change that result. Missing/positive channels, quality/backend
  failures remain insufficient; role/key/profile/framing and report invariants
  fail closed. No report authorizes localization, mutation or accepted repair.
  Experiment/reference authenticity, semantic calibration and legacy probe
  integration remain pending; this is standalone recorded-evidence wiring.
  Hosted readonly diagnosis CI passed at `a7e19a6`; it does not validate the
  subsequent frozen run binding or empirical no-edit interpretation.
- The source math-grader dependencies are added to matching CPU/CUDA locks
  without changing old pins/hashes; both retained CPU environments pass bounded
  grader import/equivalence and dependency/smoke checks. Selected CUDA packages
  are unchanged. The portable timed math wrapper and immutable quality records
  pass offline fixtures. Hosted dependency CPU CI and matching fresh CPU/CUDA
  image validation passed at `92f7e25`; math-wrapper source CI passed at `670969d`.
  Immutable IFEval metadata and the strict fixed-argument grading controller
  pass offline fixtures with independent instruction outcomes, error retention,
  timeout/abort restoration, and no empty-constraint pass. The official checker
  runtime/argument compatibility, deterministic language detection, pinned
  plaintext tokenizer resources, backend integration, and empirical grading
  remain unverified. Hosted controller source CI passed at `20468fb`.
  No hosted GPU-runtime evidence is implied.
  The official checker source/notice inventory is pinned and inspected. Full
  IFEval runtime is isolated-blocked on plaintext table rights (retained archive
  notice has no license terms), normalized-kwargs compatibility, and a reviewed
  Python-3 langdetect packaging path. Inputs remain quarantined/ignored; their
  one-attempt inspection allowance is consumed. Continue unblocked backend,
  numerical, and health prerequisites rather than bypassing these gates.
  Native vLLM-output record adaptation passes offline request/sample/exact prompt
  token identity, native finish/stop retention and immutable JSON fixtures.
  Same-length different prompt snapshots cannot join. Frozen profiles now expose
  fresh declared native sampling kwargs with backend/version/seed-policy checks;
  source task/sample and non-source repair values pass mutation-isolation fixtures.
  Explicit EOS-stop kwargs are unvalidated runtime choices. This is not installed
  vLLM, native generation, sampler/template validation or legacy CLI integration;
  those remain unverified. Hosted adapter source CI passed at `1ca3275`.
  WP04 raw BF16/FP16/FP32 hashing/stored-bit counts and actual unrounded post-cast
  norms pass offline fixtures, including erased casts and exact toy restoration.
  Hosted numerical CI passed at `959c6ba`. Canonical target preflight, atomic
  mutation rollback and full-batch byte-verified restoration pass offline CPU
  fixtures, including OOM/cancellation and fatal restoration classification with
  independent test cleanup. Hosted atomic CI at `4efac3c` passed unit/lint gates
  but failed offline smoke on invented projection aliases. Registered canonical
  projection paths now pass both retained CPU smoke profiles; this is naming
  maintenance, not a validated Qwen3.5 adapter. Evaluation-time failure/success
  restoration passes focused fixtures; health-only sweep selection no longer
  reapplies an unaccepted winner. The legacy `run` command refuses model loading
  and export until utility confirmation and fresh-load validation exist.
  Configuration, all-target direction/shape/dtype/support preflight and complete
  stored-dtype materialization now pass offline fixtures before any copy. Missing
  support cannot fall back to static alignment. Alpha-zero/unchanged stored
  candidates are explicit no-ops without live copies or published backups.
  Explicit immutable numerical budgets pass offline alpha, matrix/distinct-layer,
  intended/effective support and stored per-matrix/full-model norm fixtures.
  Selected weights tied to unselected registered parameters/buffers are refused.
  Unbudgeted legacy calls remain experimental compatibility paths, not validated
  proposals. Projection module/parameter/storage replacement, changed projection
  registries, tampered backups, and unbacked projection changes during evaluation
  now fail fatally in offline fixtures. Retained original edited storage is
  recovered where trustworthy backups remain; invalid graphs are never accepted.
  Duplicate-preserving whole-model module/parameter/buffer registries and stored
  bytes are now audited across callbacks, including unrelated state corruption,
  aliases and weightless module replacement. Unbacked corruption is detected,
  not claimed restored. These remain offline lifecycle fixtures, not a model
  adapter or empirical repair result.
  Canonical frozen registered-tensor baselines now preserve aliases/nonpersistent
  buffers and stored-byte identity. Baseline-bound calls require explicit budgets
  and reject stale keys/shapes/dtypes/bytes before copies, including alpha zero;
  repeat application cannot compound a bound trial. Frozen numerical proposals
  now bind complete baseline target identity, working-FP32 directions, exact
  ordered support, alpha and budgets under canonical digests. Caller report/config
  mutation cannot alter frozen trials; unequal support widths, no-op, stored
  budget refusal and interrupted-copy restoration pass offline fixtures. Frozen
  proposal contexts now share the whole-model restoration guard with legacy
  sweeps; normal/no-op, ordinary/OOM/cancellation, stale pre-yield and fatal
  identity/restore exits pass offline fixtures. Experiment-profile/utility/export
  integration remains pending; a frozen numerical proposal is not accepted repair
  evidence.
  Hosted budget/alias CI passed at `22c0666`, callback identity CI at `c0769af`,
  and whole-registry CI at `00539a0`. Frozen-baseline CI passed at `345f499`;
  frozen-proposal CI passed at `8bf9cfc`; shared-context CI passed at `2f28214`.
  Repetition/health source evidence is tracked in the plan; no hosted GPU runtime
  or empirical detector validation is implied.
  Hosted lifecycle/naming CI passed at `02bb02e`; it does not validate the
  subsequent proposal-preflight/materialization checkpoint.
  Hosted preparation/no-op CI passed at `676349b`; it does not validate the
  subsequent explicit-budget checkpoint.
  No automatic real-weight trial is authorized from these fixtures. Continuation
  is active; proceed with unblocked offline health/profile prerequisites.
- On project continuation, inspect/pin portable strict IFEval checker code,
  integrate grading/backend adapters, and retain data-audit review gates,
  using offline fixtures.
  Plan and satisfy a separately bounded pilot preflight before inference; do
  not reuse WP01's budget or treat artifact parity as behavior. Follow the
  plan's dependency gates thereafter.

[design]: docs/DESIGN.md
[plan]: docs/PLAN.md
[review]: docs/REVIEW.md
[readme]: README.md
[container-guide]: .devcontainer/README.md
[ci]: .github/workflows/ci.yml
[packaging]: pyproject.toml