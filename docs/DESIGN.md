# Design: Evidence-Gated Weight-Space Repair

Status: revised specification, not an implemented or validated pipeline.
Design revision: 2026-11-06.
Evidence and limitations: [REVIEW.md](REVIEW.md).
Execution order and acceptance gates: [PLAN.md](PLAN.md).

## 1. Goal and Scope

Build a reproducible research tool that first replays a published small-model weight edit, then tests whether an automatic, training-free method can discover a useful repair of a demonstrable generation failure. A valid outcome can be no edit. Lower repetition or shorter output alone is not evidence of improved reasoning.

The existing Python modules are initial skeletons. Their names suggest useful ownership boundaries, but their current behavior does not define this specification. Implemented helpers, passing smoke checks, and source-reported research results must never be labeled a completed repair method.

| Track | Question | Evidence required before claiming success |
| --- | --- | --- |
| Released-artifact replay | Can we reconstruct the author's exact final edit? | Exact base and post-edit tensor hashes |
| Behavioral reproduction | Does that artifact reproduce the reported effect? | Complete matched base/edit evaluation, with protocol and uncertainty |
| Automatic discovery | Can our estimator discover a safe, useful new edit? | Causal controls, frozen selection, independent utility and preservation checks |
| Editor-assisted discovery | Does an LLM editor improve on the automatic method? | Verified editor access, constrained proposals, matched-budget comparisons |

The first three tracks form the initial project. Editor-assisted discovery is conditional later work; replay does not require an editor model. No initial claim is made about adding missing capabilities, universal loop directions, auditing arbitrary behavior from weights alone, quantized/sharded editing, model merging, or recursive improvement.

## 2. Research Anchor and Claim Discipline

The positive anchor is the published text-model edit of `Qwen/Qwen3.5-0.8B` at revision `2fc06364715b967f1860aea9cf38778875588b17`. The research-log snapshot is `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`; [REVIEW.md](REVIEW.md) contains the pinned sources and reproduction caveats.

The source reports edits to `model.language_model.layers.15.mlp.down_proj.weight` and `model.language_model.layers.19.mlp.down_proj.weight`, each `[1024,3584]`, with 64 columns selected per matrix and alpha 6. Intended support is 131,072 values; 130,188 stored BF16 entries actually change. The directions came from four matched original productive/repetitive contexts, not a universal direction supplied by the publication.

| Layer | Base tensor SHA-256 | Edited tensor SHA-256 |
| --- | --- | --- |
| 15 | `ef60d2c687bab19464dca9c8bfd7777e46ad226df0c9e25643b0bab04b5a1cca` | `82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53` |
| 19 | `638eedeb701b94aa861f4438e615c07ec6c60ee703db1bfee2de577bb6498c3e` | `8ef3c61bc36af113ad58e89dbb0a4a14dcba5a45cb750c15ab515c82ddd12eb2` |

Source-reported gains include MATH-500 7.9% to 48.1% and cap-hit share 93.0% to 33.1%. Cap-hit share is not measured loop rate. Held-out practice confirmation also reports five correctness regressions and 51 whitespace-tail cutoffs. A faithful reproduction reports those tradeoffs; it does not imply acceptance under a stricter export policy.

Use four claim labels in reports: `source_reported`, `locally_reproduced`, `implemented_unvalidated`, and `proposed`. Formula similarity or equal parameter counts do not establish artifact parity. Failed, inconclusive, and no-patch experiments remain publishable project results.

## 3. Runtime and Data Contracts

These are semantic contracts, not a requirement to add a separate file for every row. Reuse existing modules where their ownership fits.

| Record | Required contents |
| --- | --- |
| ExperimentSpec | Version/track; code/environment identity; model/tokenizer revisions; adapter; dataset roles; generation/evaluator profiles; constraints; selection rule; budgets |
| EvaluationItem | Stable ID; source/revision; prompt/messages; reference/evaluator metadata; subject/length group; immutable dataset role |
| GenerationRecord | Experiment/candidate/item/sample IDs; seed; exact prompt tokens when available, length and thinking-prefix context; generated tokens/text; actual finish and native request/stop details; independent health observations; extraction and correctness |
| ProbeSpec | Exact module keys; hook boundary; predictive/observed token convention; matched spans; weighting; collection budget |
| LocalizationEvidence | Estimator/version; support counts; direction artifact/hash; stability; signed writer scores; control results; abstention reason |
| PatchProposal | Baseline hashes; exact parameter keys/shapes/dtypes; replacement or direction/mask artifact; alpha/support; perturbation limits |
| VerificationResult | Complete paired item outcomes; candidate status/rejection reasons; frozen choice; utility/preservation/health metrics; uncertainty; actual resources |
| ExportRecord | Explicit outcome; manifest and patch/checkpoint digests; fresh-load validation; provenance; applicable licenses/attribution |

Persist versioned records in auditable JSON plus safe tensor artifacts. Do not deserialize untrusted pickle objects. A manifest digest detects consistency, not authorship; signature/authentication requirements, if introduced, are separate.

### 3.1 Dataset Boundaries

Assign each item an immutable role: `diagnostic`, `development`, `confirmation`, `benchmark`, or `clean_reference`. Identify exact and near-duplicate overlaps before selection. Keep benchmark labels and confirmation records out of discovery/selection inputs.

Source-format projections are fresh linked declarations, not in-place edits of
raw dataset IDs or references. The GSM8K projection binds the raw manifest,
independently expected contiguous row IDs/count, derived source IDs/references
and exact evaluator revision/rule in one frozen identity. Preserve benchmark
roles and raw solution references; the published last-`####`, strip/comma-removal
rule does not imply numeric parsing or underlying row parity. A separately named
source-checker item label must refuse current optional math-verify grading.
Declaration agreement does not admit data, clear quarantine, authenticate the
source runtime or adopt a native evaluation profile.

Paired comparisons join by `(item_id, sample_index, seed)` and require identical complete key sets. Do not pair by list position or truncate using `zip`. Missing/duplicate outputs invalidate the candidate. Empty input and unavailable quality metrics are explicit errors/unavailable states, not excellent scores.

Confirmation is read only after the entire candidate is frozen: direction, columns, alpha, generation/evaluator settings, and selection rule. If confirmation influences a revised patch, it becomes development data and a new untouched confirmation set is required. A healthy/no-loop negative-control corpus must be available before automatic selection/export, not added only after a positive result.

### 3.2 Generation and Evaluation Profiles

The first source-reproduction profile pins the exact model/tokenizer revision, chat template, thinking enabled, BF16 inference, backend/version, stop/EOS behavior, and seeds. The math user turn is the problem plus `Please reason step by step, and put your final answer within \boxed{}.` Sampling is temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, and at most 32,768 new tokens. MATH-500 uses four samples per problem and GSM8K two; report mean accuracy, not pass@4. IFEval uses its official strict prompt-level checks and two samples.

The pinned thinking-enabled chat template supplies `<think>\n` before generation.
Record that prompt-boundary state explicitly: a generated closing tag without a
generated opening can be valid. Unknown prompt context gives unavailable framing
conclusions, not proof of malformed thinking or healthy output.

Model, tokenizer and generation-config EOS metadata can disagree. Inspect the
exact generation configuration and native-engine precedence; pin the effective
stop rule rather than selecting one field or silently merging IDs. Constructor
argument fixtures alone do not validate EOS/stop or token/text retention semantics.

The source math extractor uses the text after the last `</think>` when present, otherwise the full response. It reads the last `\boxed` marker, falling back to the last `\fbox` only if no `\boxed` exists. A malformed or empty selected box gives no answer, even if an earlier box was valid. Braced nested answers and the source's space-delimited `\boxed 5` form are supported. Truncated answers can still be benchmark-correct. Thinking-mode IFEval instead requires the usable response after closure. Pin portable grader implementations and dependencies; the source evaluation imports some unavailable private helpers, so copying its script is not a complete reproduction setup.

Define `source_reproduction` and `validated_repair` as distinct profiles. Source reproduction preserves reported scoring; validated repair adds usable-answer, clean-behavior, and degeneration gates. Neither profile may silently change decoding while comparing baseline and edited weights. Non-thinking mode and decoding controls are separately named comparison arms.

## 4. Independent Generation-Health Observations

Replace the assumption of one exclusive failure class with independently recorded observations:

| Observation | Meaning |
| --- | --- |
| Finish reason | Actual backend stop, length limit, cancellation, or error; never inferred solely from sequence length |
| Repetition evidence | Measured contiguous or near-periodic spans, density, onset/end, confidence, and recovery |
| Tail degeneration | Tokenizer-aware whitespace/control floods, including runs before EOS |
| Thinking closure | Required/not required; observed open/close delimiters; missing or malformed closure |
| Answer usability | Empty, invalid format, valid task answer, or evaluator-specific unavailable result |
| Task correctness | Separate task/evaluator result, including benchmark-correct truncated answers |

A cap hit alone is not a proven loop. Legitimate lists, repeated code, formulas, boilerplate, and recovered repetition are negative cases for detector calibration. Do not label the entire suffix after the first repeated phrase pathological by default. Diagnosis, probing, and verification consume the same record/configuration; probing never recomputes a different label from incomplete metadata.

If there is no observed pathological repetition, no matched evidence, or no stable direction, emit an informative `no_patch` or `insufficient_evidence` report without forcing localization.

## 5. Adapters and Contrastive Probing

The first validated adapter targets the Qwen text stack. It exposes real state-dict keys, explicit layer order, residual/output boundaries, projection orientation, actual dtype/device, and supported inference/probe backends. AutoModel/config import success is not an adapter integration test. Unsupported stacks fail explicitly rather than choosing the first ModuleList or guessing from suffixes.

Use embedding output $h_t^{(0)}$ and block $\ell$ producing $h_t^{(\ell+1)}$. For a conventional block, a schematic update is:

$$
h_t^{(\ell+1)} = h_t^{(\ell)} + W_{\mathrm{attn}}^{(\ell)}a_t^{(\ell)} + W_{\mathrm{mlp}}^{(\ell)}m_t^{(\ell)}.
$$

The adapter owns the actual normalization, scaling, bias, sublayer ordering, and hybrid-attention semantics; the equation does not assert identical internals across architectures. The source's residual outputs 16/20 map to zero-based blocks 15/19. Observed-token states at position $t$ and predictive states at $t-1$ are different attribution choices; record and test the selected convention.

Construct matched productive/repetitive contexts on original weights. Balance by prompt/context, control lengths/positions/topics where possible, and avoid treating prompt-boundary tokens as clean generated reasoning. Capture residual outputs and pre-projection inputs only for required layers/spans; aggregate bounded statistics inside hooks. Test teacher-forced and cached-generation behavior on bounded fixtures before claiming equivalence.

## 6. Automatic Localization as a Falsifiable Hypothesis

Candidate direction estimation starts with matched, prompt-balanced differences:

$$
\delta^{(\ell)} = \mu_{\mathrm{loop}}^{(\ell)} - \mu_{\mathrm{productive}}^{(\ell)}, \qquad
u^{(\ell)} = \frac{\delta^{(\ell)}}{\|\delta^{(\ell)}\|_2}.
$$

Reject negligible/nonfinite differences and unstable directions using criteria fixed before confirmation. Estimate stability across matched contexts and diagnostic subsets. A post-layer difference may reflect topic, token identity, or trajectory length; it does not prove a cyclic attractor or identify a causal writer.

For compatible residual-writing projections, estimate signed column contributions:

$$
c_j = (u^\top W_{:,j})(\bar m_{\mathrm{loop},j}-\bar m_{\mathrm{productive},j}).
$$

Positive contributions generate writer candidates. Negative contributions may be protective; do not silently replace signed ranking with absolute alignment or include nonpositive writers merely to fill K. Static alignment, if studied, is a separately declared estimator/control. Fewer than K supported columns leads to a recorded smaller candidate or abstention according to the fixed experiment rule, not an undocumented fallback.

Before claiming a discovered repair mechanism, compare targeted interventions with norm/support-matched random directions/columns, layer-isolated changes, and dose response. Measure both generated-answer utility and productive/numeric preservation; teacher-forced repetition likelihood alone is insufficient. A targeted effect indistinguishable from controls falsifies the current causal hypothesis.

## 7. Numerical Edit and Transaction Invariants

For baseline projection $W_0$, unit direction $u$, and column mask $M_K$:

$$
W' = W_0 - \alpha u\left(M_K \odot (u^\top W_0)\right), \qquad
u^\top w'_j = (1-\alpha)u^\top w_{0,j}
$$

for selected column $j$ in exact arithmetic. Alpha one removes that column's directional component; alpha six reverses it at five times the original magnitude. Context-dependent activation and downstream nonlinearities determine whether this helps. Small global support does not establish small functional impact or capability preservation.

Before any mutation, validate every target, baseline digest, dimension, dtype, unit/finite direction, unique in-range support, alpha bound, and per-matrix/global perturbation budget. Matrix count and distinct-layer count are separate configuration fields. Every candidate starts from the same immutable baseline; no repeated application can compound a trial.

Compute the proposed update in the declared working dtype, then materialize the actual stored dtype. Hash contiguous raw bytes using a versioned encoding compatible with source BF16 replay. Distinguish intended touched support from bit-effective changes. Verify unselected columns remain bit-identical, and record post-cast relative Frobenius change. Ineffective casts and alpha zero are no-op outcomes.

Prevalidate all modules before editing any. Restore original tensor bits on generation/probing/evaluation errors, OOM, invalid metrics, and export failure. Missing targets must fail closed; a failed restore is an explicit fatal validation error, never silently ignored. Final commit/export occurs only after confirmation and fresh-load checks, not as a side effect of choosing an alpha.

## 8. Selection, Confirmation, and Preservation

Evaluate an explicit immutable zero-edit baseline independently of alpha order. Validate complete paired outputs, finite measured metrics, and declared candidate grids. Cache baselines under the full experiment profile; sampling comparisons use the same recorded item/seed schedule without promising identical backend RNG paths.

Keep correctness rescues/regressions distinct from health improvements/regressions. Task rescue means baseline incorrect and candidate correct; a usable-answer contract is additionally required for a validated-repair result. Early EOS or a shorter incorrect answer is never a task rescue. Corpus perplexity protects only its measured reference distribution and remains a secondary guard.

Predeclare the development selection rule, minimum useful improvement, uncertainty procedure, clean-suite tolerance, and degeneration budget. Prefer no edit or the smallest effective edit on ties. Freeze direction/support/alpha and the entire profile before untouched confirmation. Cluster statistical uncertainty by problem rather than treating its multiple samples as independent problems.

The initial `validated_repair` policy is conservative: a positive lower bound of the predeclared 95% problem-level paired improvement interval on confirmation; no observed new whitespace/control degeneration or correctness losses on the fixed clean-preservation suite; and an independently measured clean-corpus perplexity ratio at most 1.25. These are proposed gate defaults, not a guarantee of population-level safety or a threshold derived from the publication. Fix the confirmation size and resampling procedure after the hardware/data pilot and before automatic edit selection. Any changed tolerance creates a new profile before evaluating it, not an after-the-fact exception.

Source-artifact reproduction reports all tradeoffs even when this stricter policy rejects the published edit for validated export. Holdouts never tune the selected patch. An inconclusive interval, violation, or missing metric produces an explicit no-patch/validation-failed result.

## 9. Resource and Artifact Policy

Do not promise 20-50 literal model forwards. Count generation requests, teacher-forced requests, sequences, prompt/generated tokens, CPU/GPU peak memory, wall time, and failures. A harness request can generate many long sequences. Under source dimensions, retaining 32K residuals and MLP inputs in float32 alone consumes about 13.5 GiB; bounded collection is a requirement, not an optimization to postpone.

The first hardware pilot declares time/token/memory caps and checkpoints progress. It uses small public subsets and cannot claim full benchmark reproduction. Full evaluation starts only after a measured resource plan is recorded. Budget exhaustion restores weights and leaves resumable diagnostics, not a committed repair. No pretrained downloads or long GPU runs are required by ordinary unit CI.

Artifact replay can use the audited upstream exact-replacement format before our discovery code exists. The first project delta format should use exact stored-dtype replacement slices bound to base hashes, not an approximate arithmetic delta. Source BF16 payload size is 256 KiB before metadata; FP32 payload is 512 KiB. Preserve the actual base-model, research-artifact, script, and dataset licenses/attribution rather than labeling all outputs MIT.

Export to a staging directory; validate a strict typed manifest and cross-field invariants; reload/apply to the exact baseline; verify all edited hashes and every untouched tensor; then publish atomically. Refuse incompatible bases and accidental overwrites. Repeated application must either explicitly report already-applied exact values or refuse; it must never compound an edit.

Complete artifact coverage is distinct from registered runtime state. An adapter
must account for every baseline checkpoint tensor, including auxiliary tensors
the builtin loader explicitly ignores. Such tensors remain baseline-owned and
must be preserved bit for bit outside the runtime graph; dropping them from a
framework save is not a verified full-checkpoint export. Explicit tied aliases
and nonpersistent runtime buffers require separate declared coverage, not guessed
key equivalence. A registered-tensor baseline alone proves neither complete
checkpoint preservation nor model/config provenance. Missing coverage fails
closed; no partial-load or artifact-export authorization follows from a loader's
ignore declaration.

## 10. Lifecycle, Ownership, and Acceptance

The required lifecycle is `baseline -> diagnose -> candidate evidence -> reversible development trials -> freeze -> independent confirmation -> staged export -> fresh-load validation -> published artifact`.

Terminal outcomes are `patched`, `no_patch`, `insufficient_evidence`, `unsupported_model`, `validation_failed`, and `budget_exhausted`. Reports retain rejected candidates and negative results. `patched` under a research-reproduction profile is not interchangeable with `patched` under the validated-repair profile; the profile/track must accompany the outcome.

| Existing owner | Intended responsibility; currently skeleton behavior |
| --- | --- |
| [detector.py](../src/llm_loop_fix/detector.py) | Shared independent health observations and calibrated spans |
| [probe.py](../src/llm_loop_fix/probe.py) | Explicit adapter use, matched-span collection, bounded statistics |
| [localize.py](../src/llm_loop_fix/localize.py) | Candidate evidence, signed ranking, stability, abstention |
| [patch.py](../src/llm_loop_fix/patch.py) | Raw-dtype auditing, validated proposal materialization, exact rollback |
| [verify.py](../src/llm_loop_fix/verify.py) | Identity-complete pairing, explicit baseline, utility selection, confirmation |
| [manifest.py](../src/llm_loop_fix/manifest.py) | Versioned typed provenance bound to actual artifact contents |
| [cli.py](../src/llm_loop_fix/cli.py) | Structured input/profile handling, explicit outcomes, atomic orchestration |
| [test_pipeline.py](../tests/test_pipeline.py) | Focused algebra/lifecycle tests; real-architecture integration and empirical gates labeled separately |

Acceptance proceeds through replay, behavior, discovery, export, and expansion gates defined in [PLAN.md](PLAN.md). The controlled toy fixture has been repaired and its two tests pass; it uses a one-sided writer and establishes only a synthetic baseline. The separate [replay utility](../src/llm_loop_fix/replay.py) now establishes exact published-artifact tensor parity, not behavioral improvement. The automatic pipeline's BF16 hash path, verifier acceptance, and rollback defects remain pending implementation tasks. This revision establishes their contract; the separate replay result does not declare those defects fixed.
