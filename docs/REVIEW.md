# Weight-Space Programming and llm-loop-fix: Research and Design Review

Review date: 2026-11-06.
Repository baseline: `287d120c40397ca8efb87f98f4c74ac1ed3f5181`.
Research-log snapshot: `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`.

Scope: the publication and released experimental evidence; the repository's README, design, roadmap, all eight Python modules, tests, sample prompts, packaging, and development-container configuration. This is a review and proposed backlog, not an implementation change or a completed implementation guide. Existing source, design, and roadmap files were left unchanged.

## 1. Prioritized Findings

Priorities are implementation gates, not production-incident severity:

- **P0:** resolve before trusting the pipeline's repair claims or using its automated edits as validated results.
- **P1:** resolve before an evaluated, reproducible MVP.
- **P2:** subsequent expansion, after the core mechanism and safety contracts are demonstrated.

Evidence labels distinguish **executed** local checks, **inspection** of current code, **source-reported** research results, and **hypotheses** requiring experiments. Research results below were not reproduced locally in this review.

| ID | Priority | Finding | Main owner |
| --- | --- | --- | --- |
| F01 | P0 | The verifier can reward loss of useful output | Verification and evaluation |
| F02 | P0 | Completed/parity claims exceed the available evidence | Design, roadmap, README |
| F03 | P0 | Failure categories are conflated and not propagated through the pipeline | Detection and generation records |
| F04 | P0 | Correlational localization is treated as a causal repair mechanism | Research protocol and localization |
| F05 | P0 | BF16 hashing fails; changed-entry counts are not genuinely bitwise | Patching and numerical auditing |
| F06 | P0 | Failed verification can leave edited weights active; export is not transactional | Sweep and export lifecycle |
| F07 | P0 | The baseline test suite fails and does not establish real-model repair | Tests and milestone definitions |
| F08 | P1 | Item identity, answers, dataset roles, and evaluation boundaries are missing | Data contracts |
| F09 | P1 | The generation protocol does not match the motivating experiment | Model loading and inference |
| F10 | P1 | Architecture, module identity, and activation-position contracts are implicit | Model adapters and probing |
| F11 | P1 | Sweep inputs and selection do not reliably enforce a true baseline or abstention | Candidate selection |
| F12 | P1 | The manifest is not a strict schema or a sufficient reproduction record | Provenance |
| F13 | P1 | Memory and compute budgets are undefined; long-context probing is expensive | Runtime and experiment planning |
| F14 | P1 | The CPU build input is absent and dependency/CI policy is incomplete | Packaging and containers |
| F15 | P1 | Projection mathematics need explicit numerical and edit-budget invariants | Patch specification |
| F16 | P2 | The proposed LLM editor is a separate, currently unspecified product track | Scope and agent design |
| F17 | P2 | Delta export needs an exact application contract and licensing policy | Artifact distribution |
| F18 | P2 | Attention, quantization, and broad model support are premature expansions | Research roadmap |

### F01 - P0: Stopping repetition is not sufficient evidence of repair

**Evidence:** [verify.py](../src/llm_loop_fix/verify.py#L140) defines rescue as a change from looping to non-looping. Selection considers repetition metrics and perplexity, not answer usability, correctness, or instruction following. [cli.py](../src/llm_loop_fix/cli.py#L102) supplies prompt prefixes as the clean perplexity references. **Executed:** an isolated sweep selected `alpha=1.0` and counted one rescue when the candidate output was only EOS. Patch application and restoration were mocked in that acceptance-rule check.

**Risk:** an empty answer, early EOS, wrong short answer, or premature reasoning closure can appear to improve the model. Prompt-prefix perplexity does not protect the model's ability to produce useful answers.

**Change:** track termination, repetition, answer extraction, task correctness, and instruction compliance independently. Keep clean-corpus perplexity as a secondary guard, not the repair objective. Define correctness rescues and correctness regressions separately from generation-health transitions. Decide explicitly how much regression or degeneration, if any, a research or product profile permits.

**Done when:** an EOS-only candidate and a non-looping incorrect-answer candidate cannot qualify as task rescues; the selected edit improves held-out utility while meeting predeclared clean-behavior and degeneration gates. A benchmark-correct but truncated answer remains representable without being automatically declared incorrect.

### F02 - P0: Research-inspired implementation is being presented as established parity

**Evidence:** [DESIGN.md](DESIGN.md#L1) describes a dominant cyclic-attractor direction and directly implemented research mechanics. [PLAN.md](PLAN.md#L3) marks infrastructure and exact Qwen parity completed. [README.md](../README.md#L1) presents a roughly 20-50-forward-pass repair pipeline. No published-checkpoint replay, matched benchmark evaluation, or automatic-discovery experiment exists in the repository.

**Risk:** formula parity, artifact parity, behavioral reproduction, and a newly proposed automatic method are four different achievements. The current wording collapses them. A single direction is a working hypothesis, not an established description of all small-model loops.

**Change:** label claims as source-reported, locally reproduced, implemented but unvalidated, or proposed. Replace completed phases with artifact-backed milestones. Describe the present code as a research prototype. Distinguish replaying the released edit, reproducing its behavior, reproducing its discovery process, and evaluating our own locator.

**Done when:** every completed milestone links to passing checks or an experiment artifact; exact-edit parity requires the published tensor hashes, and behavioral parity requires the published evaluation protocol. Automatic discovery has its own separate acceptance gate.

### F03 - P0: The failure taxonomy is neither independent nor wired end to end

**Evidence:** [detector.py](../src/llm_loop_fix/detector.py#L21) treats every non-EOS cap hit as looping, treats empty output as clean, and flags a repeated block anywhere in the sequence even if the output subsequently recovers. It does not parse thinking delimiters or establish normal termination. [probe.py](../src/llm_loop_fix/probe.py#L126) passes only n-gram settings. [cli.py](../src/llm_loop_fix/cli.py#L89) does not provide whitespace-token metadata to probing or verification. **Executed:** the same distinct-token output was `clean` in the probe but `cap_truncation` when the detector received cap metadata; clean-only localization then raised `ValueError`.

**Risk:** diagnosis and verification use different labels. A healthy long answer can become a contrastive failure example, real truncation can be missed during localization, and whitespace regressions can go uncounted. A model with no observed loops cannot complete an ordinary diagnosis.

**Change:** create one generation record and one detector configuration shared by every stage. Separate finish reason, repetition evidence, tail degeneration, thinking closure, and answer usability. Record observed spans and confidence rather than assuming the entire suffix after a repeated phrase is pathological. Calibrate thresholds against legitimate lists, repeated code, boilerplate, and long productive reasoning. Tokenizer-aware whitespace/control handling must cover runs preceding EOS as well as literal final runs.

**Done when:** identical records have identical labels in diagnosis and verification; a cap hit alone is not called a proven loop; recovered repetition, unclosed thinking, empty output, and whitespace tails have explicit tests. No-loop and insufficient-evidence cases produce a useful report and no edit.

### F04 - P0: Localization correlation is being used as causal evidence

**Evidence:** [localize.py](../src/llm_loop_fix/localize.py#L58) forms a direction from pooled loop/clean means and ranks projections by directional contribution. At [localize.py](../src/llm_loop_fix/localize.py#L133), nonpositive activation scores fall back to static alignment; top-K selection can include nonpositive columns. [probe.py](../src/llm_loop_fix/probe.py#L124) pools unequal-length spans, including a prompt-boundary token. There is no direction-stability estimate, matched-context protocol, or causal control experiment.

**Risk:** the direction can reflect position, length, token identity, topic, or reasoning style rather than a loop mechanism. Long failed trajectories can dominate the estimate. Static alignment does not establish that a column caused the failure, and editing negatively contributing columns can remove protective behavior.

**Change:** keep the contrastive estimator as a candidate generator. Specify matched productive/repetitive contexts, prompt-balanced weighting, uncertainty and stability checks, and an abstention rule for weak directions. Make positive-writer selection explicit instead of silently substituting another ranking. Test activation interventions, dose response, layer isolation, and norm-matched random-direction/column controls before committing to persistent weight edits.

**Done when:** on unseen diagnostic examples, the targeted intervention outperforms declared controls and improves generated-answer utility, not just teacher-forced repetition scores. If it does not, the design records the hypothesis as falsified and does not export a purported causal repair.

### F05 - P0: BF16 provenance currently fails at the first hash

**Evidence:** [patch.py](../src/llm_loop_fix/patch.py#L13) hashes `tensor.numpy().tobytes()`. **Executed:** hashing a BF16 tensor raised `TypeError: Got unsupported ScalarType BFloat16`. [patch.py](../src/llm_loop_fix/patch.py#L134) counts numeric inequality, not raw-bit inequality. Numeric comparison equates positive/negative zero and handles NaNs differently from a bit comparison.

**Risk:** the advertised BF16 audit path cannot run. Converting to float32 merely to make hashing work would produce a different hash from the evaluated BF16 checkpoint. Nominal touched entries, numerically changed values, and bit-changed stored values are not interchangeable.

**Change:** hash a specified contiguous raw-byte representation of the actual storage dtype. Compare per-entry storage bits for bitwise counts, and reject nonfinite weights before editing. Report intended support, effective post-cast changes, and both pre-cast and stored-dtype perturbation norms where useful. Preserve compatibility with the upstream BF16 hashing convention for replay.

**Done when:** BF16, FP16, and FP32 fixtures hash and round-trip correctly; selected values that round back are not counted as changed; untouched columns retain identical bits. The published replay yields 130,188 bit-changed entries, not an assumed 131,072.

### F06 - P0: Verification and export must be transactional

**Evidence:** [verify.py](../src/llm_loop_fix/verify.py#L74) restores between normal candidates but has no exception-safe rollback around the sweep. **Executed:** an intentionally failed second generation call left a real toy projection edited, changing its tested entry from `10.0` to `0.0`. [cli.py](../src/llm_loop_fix/cli.py#L118) saves the model before building and validating the final manifest. Missing module names can also be silently skipped during patching/restoration.

**Risk:** a generation error, OOM, interrupted evaluation, invalid configuration, or failed manifest write can leave a live model changed or a partial output directory that looks like a completed patch.

**Change:** prevalidate the entire edit, preserve an immutable baseline, restore on every failure path, and commit the winning patch only after its gates pass. Export into a staging directory, validate the manifest against the actual stored weights, reload and verify, then publish atomically. Refuse unintended overwrites and report explicit failure/no-patch outcomes.

**Done when:** injected exceptions at each mutation/evaluation/export boundary restore all original tensor bits and leave no committed partial artifact. Missing targets fail closed rather than silently weakening the edit or rollback.

### F07 - P0: The existing unit suite is not a successful pipeline baseline

**Evidence:** **Executed:** [test_pipeline.py](../tests/test_pipeline.py#L107) produced one pass and one failure. Its intended clean sequence is `[2,3,4,5,6,7,8,9,7,8,9,7,8,9,7,8]`, so the assertion that it is non-looping fails before patching. The toy generator also hardcodes token transitions from a hidden-coordinate threshold instead of using the LM logits, and never truly stops on EOS. The container smoke uses supplied trajectories through random Qwen2 weights.

**Risk:** the test cannot currently validate the claimed successful localization/patch/manifest path, and neither the toy nor the smoke test demonstrates repair of a naturally looping language model.

**Change:** repair the fixture as a focused follow-up, without weakening the detector to accommodate it. Keep synthetic algebra and lifecycle tests, add controlled autoregressive integration tests, and establish a separately labeled real-model experiment gate. Reuse the existing test home/helpers before multiplying test files.

**Done when:** the normal test command is green, the fixture's clean behavior is genuinely clean, failure/no-op paths are tested, and roadmap labels distinguish numerical tests, wiring smoke tests, integration tests, and empirical repair evidence.

### F08 - P1: Evaluation needs structured items and sealed dataset roles

**Evidence:** [cli.py](../src/llm_loop_fix/cli.py#L16) reduces inputs to strings and drops IDs and other fields. [sample_prompts.jsonl](../examples/sample_prompts.jsonl#L1) contains five illustrative prompts without expected answers or split roles. The same prompt list is used for localization and alpha selection. [verify.py](../src/llm_loop_fix/verify.py#L145) pairs records with `zip`, which silently truncates unequal result lists.

**Risk:** missing/reordered outputs, duplicate problems, and reuse of selection examples can create invalid paired statistics and apparent generalization. Five easy prompts are not a loop dataset or a benchmark.

**Change:** define items with stable IDs, source revision, answer/evaluator metadata, subject/length group, and immutable role. Separate diagnostic discovery, development selection, frozen confirmation, final benchmark, and clean preservation corpora. Check exact/near-duplicate overlap. Reject missing, duplicate, or mismatched sample IDs instead of zipping by position. Sample identity must include problem ID and seed/sample index.

**Done when:** all paired comparisons are complete identity joins; selection cannot read final benchmark labels; confirmation is not recycled into tuning. Empty datasets or unavailable reference metrics are errors or explicit unavailable values, never apparently excellent scores.

### F09 - P1: The default inference path is not a Qwen reproduction protocol

**Evidence:** [cli.py](../src/llm_loop_fix/cli.py#L42) defaults to 192 new tokens; [cli.py](../src/llm_loop_fix/cli.py#L62) loads float32; generation uses raw tokenized prompts and greedy decoding without a chat template, thinking control, or recorded sampling profile. Experiment 8 used thinking-enabled, BF16, long-context sampled evaluation with a presence penalty and multiple samples per problem [S10].

**Risk:** both the baseline failure and the edited behavior can change before any surgical edit is considered. A 192-token cap can manufacture truncation on productive reasoning. Reloading a BF16 checkpoint as FP32 also prevents storage-level parity with the original artifact.

**Change:** introduce explicit reproducibility and product-evaluation profiles with pinned model/tokenizer revisions, dtype, chat template, thinking mode, EOS/stop configuration, seeds, backend, and generation settings. Cache the baseline under that complete profile. Treat non-thinking mode and decoding penalties as comparison baselines, not as settings silently changed while measuring a weight edit.

**Done when:** baseline and candidate use identical declared profiles. A parity run can reproduce the source prompt/sampling/grading contract, while a lightweight smoke run is clearly identified as a different protocol.

### F10 - P1: Model adapters must own module identity and activation semantics

**Evidence:** [probe.py](../src/llm_loop_fix/probe.py#L30) falls back to the first nonempty `ModuleList`; [probe.py](../src/llm_loop_fix/probe.py#L57) recognizes selected `nn.Linear` suffixes and invents names such as `layers.15.mlp.down_proj`. These are not necessarily state-dict keys. Conv1D-oriented, fused, quantized, multimodal, hybrid-attention, or sharded models do not automatically satisfy that contract. **Executed:** the current Transformers environment does map both Qwen3.5 configuration classes to `Qwen3_5ForCausalLM`; lack of an AutoModel mapping was not confirmed as a current defect.

**Risk:** a successful config import or an AutoModel mapping can be mistaken for full adapter compatibility. The fallback can find the wrong stack. A token's observed hidden state and the previous position that predicted that token are distinct attribution choices.

**Change:** start with an explicit Qwen text adapter exposing real checkpoint names, layer order, residual-output boundaries, projection orientation, and supported dtypes/backends. Record unsupported architectures rather than guessing. Specify post-layer hook versus `hidden_states[layer+1]` conventions and whether onset attribution uses observed-token or predictive-position states. Test teacher-forced versus cached-generation consistency on bounded examples.

**Done when:** tiny Qwen2 and Qwen3.5 integration fixtures resolve the intended text blocks and exact weight keys; shape/position assertions fail on incompatible modules. Broader support is a tested capability table, not an implication of name matching.

### F11 - P1: Sweep selection must enforce baseline identity and a no-edit preference

**Evidence:** [verify.py](../src/llm_loop_fix/verify.py#L167) assumes the first candidate is the baseline although the API accepts an arbitrary alpha sequence. It does not require a unique zero candidate, complete generated results, finite metrics, or nonempty references. Empty references return perplexity `1.0`. Tie-breaking can choose a nonzero edit for a small perplexity difference even when repair metrics do not improve.

**Risk:** nonzero-first sweeps can compare against an already edited model. Missing outputs and unmeasured quality can look harmless. A needless mutation can be selected without demonstrated benefit.

**Change:** evaluate an explicit immutable `alpha=0` baseline independently of candidate order. Validate the candidate grid and metrics, use per-item seeds for sampling comparisons, and distinguish unavailable perplexity from a measured value. Require meaningful development improvement before nonzero selection; prefer no edit or the smallest effective perturbation on ties. Freeze all selection choices before confirmation.

**Done when:** shuffled alpha grids have the same baseline/decision, malformed grids and incomplete outputs are rejected, and no-supported-gain runs return `no_patch` with unchanged weights and an explanatory report.

### F12 - P1: A self-digest is not strict schema validation or full provenance

**Evidence:** [manifest.py](../src/llm_loop_fix/manifest.py#L137) checks the version string, required top-level keys, and digest. **Executed:** a correctly digested object with `model_id=None`, a list for `intervention`, a string for `summary`, and an integer for `edited_matrices` passed. The manifest omits model/tokenizer revisions, direction artifacts, runtime profile, code/environment revision, item-level results, and dataset roles. Its abbreviated localization report retains only the first 16 listed columns per summary.

**Risk:** a malformed or internally contradictory manifest can be accepted, and a valid round-trip JSON file cannot reproduce the edited checkpoint or establish the reported evaluation. An unsigned digest detects accidental inconsistency, not authorship or trustworthiness.

**Change:** specify a versioned, typed schema and cross-field invariants: shape/dtype, unique in-range columns, support/count consistency, finite metrics, selected alpha, and agreement with stored tensor hashes. Record pinned baseline identity, actual parameter keys, materialized patch or direction/mask artifacts, dataset/profile/environment digests, candidate outcomes, and rejection reasons. Validate against freshly loaded artifact contents, not just the manifest's own claims. Authentication/signing, if needed later, is a separate requirement.

**Done when:** recomputing the digest cannot make malformed or inconsistent metadata pass. Applying the referenced patch to its exact base yields the declared post-edit hashes and leaves every other tensor unchanged.

### F13 - P1: Forward-pass counts conceal the actual compute and memory cost

**Evidence:** [probe.py](../src/llm_loop_fix/probe.py#L77) retains full CPU float32 residual outputs and projection inputs before aggregation and calls the model on complete sequences. For the source dimensions, residuals plus MLP inputs alone require approximately `32,768 * 24 * (1,024 + 3,584) * 4` bytes: **13.5 GiB**, excluding attention inputs, model/logits/KV memory, and the prompt. The CLI generates prompts serially and regenerates the baseline during the sweep. No request, token, wall-time, or memory budget is enforced.

**Risk:** the long-context experiment motivating the project is not a cheap extension of the short toy run. Generation internally uses many decoding forwards; a research harness request is not one literal model forward. The claimed 20-50 passes is not a cost guarantee.

**Change:** define budgets in generation requests, teacher-forced requests, sequences, prompt/generated tokens, wall time, and peak CPU/GPU memory. Cache baselines, aggregate selected span statistics inside hooks, probe selected layers/windows, and avoid retaining unnecessary full logits or caches. Bound long-context processing while preserving positional/context semantics; arbitrary prefix cropping is not a faithful substitute. Establish a measured pilot before committing to full benchmark runs.

**Done when:** a representative long-context probe fits a declared memory envelope, budget exhaustion is recoverable, and every experiment reports actual resources. Only measured results may become efficiency claims.

### F14 - P1: Container and packaging reproducibility are incomplete

**Evidence:** [Dockerfile](../.devcontainer/Dockerfile#L3) defaults to the CPU dependency lock and copies it at [Dockerfile](../.devcontainer/Dockerfile#L56), but that lockfile is absent from the worktree and tracked files. The CUDA lock is present. [pyproject.toml](../pyproject.toml#L1) allows much older dependencies than [requirements.in](../.devcontainer/requirements.in#L1), and its dev extra includes pytest but not Ruff. No CI workflow exists. **Executed:** CUDA offline smoke and shellcheck passed; Ruff reported 105 findings in the active environment, largely annotation/import/style policy, with three unused imports and one unused variable.

**Risk:** a working prebuilt CUDA environment does not prove the default CPU image can be rebuilt. A plain editable installation can legally select versions lacking the intended Qwen support. The promised pytest/Ruff/shellcheck gate is not yet enforced.

**Change:** restore a generated, hash-locked CPU dependency input as a separate task. Align declared support with tested package combinations; keep heavyweight evaluation/backend dependencies in deliberate extras/environments. Commit an explicit lint policy before treating all 105 findings as mandatory cleanup. Add unit/lint/shell CI, offline tiny-model integration gates, and separately scheduled real-model experiments. Pin the container base digest if reproducible image identity is a requirement; a Python lock alone does not freeze apt/base-image contents.

**Done when:** fresh CPU and CUDA builds consume existing validated inputs, the documented dev installation has its advertised tools, and ordinary CI is green without claiming that smoke checks reproduce the research.

### F15 - P1: Numerical validity and sparsity must be explicit patch invariants

**Evidence:** [patch.py](../src/llm_loop_fix/patch.py#L65) accepts unchecked alpha, direction, support, and budget values; a nonpositive top-K is clamped to one, and fractional selection can exceed available columns. Zero directions can still produce nominally touched matrices. The top-matrix count does not guarantee distinct layers despite the CLI's `--top-k-layers` name. Norm change is measured before casting, and there is no hard intervention-size/stability limit.

**Risk:** an invalid configuration can become an unintended edit rather than a rejection. A small global parameter fraction can conceal a large intervention on the selected matrices. Zero separation is absence of evidence, not a valid discovered loop direction.

**Change:** validate finite unit directions, compatible dimensions, unique support, positive in-range K/fractions, allowed alpha bounds, actual dtype, and per-matrix/global budgets before mutation. Make number of matrices versus distinct layers unambiguous. Record effective post-cast perturbation, and return a no-op for an ineffective update. Treat alpha greater than one as a measured behavioral intervention, not ordinary projection removal.

**Done when:** invalid and degenerate inputs fail before any mutation; intended support and bit-effective support are audited; alpha zero and no-effect casts retain original bits; repeat application cannot accidentally compound a candidate instead of starting from the same baseline.

### F16 - P2: An LLM-guided editor is not the current automatic heuristic

**Evidence:** [PLAN.md](PLAN.md#L18) names an agentic weight-space loop, but the implementation contains no editor-model client, proposal protocol, bounded inspection interface, sandbox, or hypothesis record. The current method is a conventional activation-guided, gradient-free heuristic.

**Risk:** the project can appear to reproduce the publication's native LLM programming capability while actually testing a different algorithm. Adding an unconstrained agent before stable evaluation also increases cost and makes attribution of gains harder.

**Change:** first validate the deterministic toolchain. If an editor is later desired, give it bounded inspection tools and declarative patch proposals over an allowlist; enforce budgets and safety in the trusted runner. Keep all benchmark holdouts inaccessible. Log proposals, evidence, rejected edits, model/version/prompt settings, and human approval policy. Prefer structured proposals to arbitrary code execution for the MVP.

**Done when:** an editor-assisted arm can be compared fairly with the same-budget automatic heuristic and controls. Editor availability is verified for that experiment; source use of `gpt-6-astra` is not assumed to mean we can reproduce its discovery runs.

### F17 - P2: Delta distribution needs exactness, base binding, and rights metadata

**Evidence:** [PLAN.md](PLAN.md#L18) proposes a roughly 256 KB slice exporter, but there is no patch loading/application API or artifact schema. Upstream publishes exact BF16 replacement values/indices and verifies the base and rebuilt tensors [S13], [S14]. Research logs/networks, scripts, model weights, and benchmark data do not all share this repository's MIT license [S2], [S7], [S18].

**Risk:** a numerical delta can round differently when reapplied or be applied to an incompatible revision. A small edit may still carry base-model license obligations, and copied experiment material can introduce separate attribution or redistribution requirements.

**Change:** prefer exact target-dtype replacement slices for the first distribution format, bound to base tensor hashes and canonical parameter keys. Define encoding, dtype, shape, selected indices, checksums, refusal on mismatches, rollback, and repeat-application behavior. Keep the artifact separate from the audit manifest but bind their digests. Track upstream/model/dataset licenses and attribution; do not relabel everything MIT.

**Done when:** export/load/apply matches a full checkpoint bit for bit on the exact base and rejects different bases. Size claims specify dtype and metadata: 131,072 BF16 values are 256 KiB before metadata; FP32 values are 512 KiB.

### F18 - P2: Broader interventions should follow, not precede, mechanism validation

**Evidence:** [PLAN.md](PLAN.md#L18) proposes q/k head-periodicity probing and several different model families as near-term work. The publication's positive Qwen result is specific, and its non-looping Gemma follow-up did not establish a capability improvement [S17]. Qwen3.5's hybrid attention also makes ordinary full-attention-head assumptions insufficient for general support.

**Risk:** moving immediately to attention circuits, quantized weights, model merging, or general reasoning repair creates several new hypotheses and compatibility boundaries before the main one is tested.

**Change:** use one pinned, known-looping Qwen checkpoint as the positive anchor and a healthy/non-looping model as a no-edit negative control. Test another model/seed/domain only after the first automatic method passes its gates. Put attention-periodicity research, quantized/sharded patching, multidirectional edits, and native editor discovery in explicitly conditional later phases.

**Done when:** every expansion has a motivating observed failure, a discriminating experiment, an adapter contract, and separate held-out preservation checks. Failure to find a safe edit is an acceptable result.

## 2. Publication Analysis

### 2.1 What the publication establishes and what it does not

The linked work is a November 2, 2026 research blog by Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab, with released experimental logs [S1], [S2]. Its citation describes a blog post; this review does not assume a separate peer-reviewed paper. The reported editor is `gpt-6-astra`, run through Codex with different reasoning-effort levels. That model identity is source-reported, not a capability independently tested here.

The central contribution is a collection of demonstrations that a general-purpose LLM can infer selected behavior from weights and can express task-directed changes directly in weight space without gradient training. Some demonstrations are tool-free text-only interactions; the larger ones use substantial linear algebra, Python execution, and behavioral feedback. The results support feasibility in those settings, not a universal optimizer or a universal interpretable failure direction.

The evidence needs to be read at three levels:

1. **Demonstrated:** the released networks, edits, logs, and evaluations support particular reported successes and failures, subject to reproduction.
2. **Plausible project hypothesis:** targeted weight editing can remove an identifiable generation obstacle in some pretrained models without retraining.
3. **Not established:** arbitrary hidden-behavior auditing without inference, universally reliable automatic repair, adding missing capabilities, no catastrophic forgetting from sparse edits, or efficient general self-improvement.

The source release deliberately excludes earlier prompt-version trials [S2]. Therefore the displayed runs are not a predeclared random sample of all attempts. The release also omits some intermediate activations/generated work files and parts of the Experiment 8 agent/evaluation harness [S10]. This limits exact discovery-process reproduction even when final-artifact replay is possible.

### 2.2 Experiments 1-3: Reading weights

**Experiment 1: game identification [S3].** Four policies have identical architecture, channel semantics, and action space: a 17,253-parameter `500 -> 32 -> 32 -> 5` MLP. Astra receives architecture/input metadata and rounded weights, but no game list or tools, and identifies all four games. The policies shown were chosen as the best of three trained seeds per game; there is one reader reply per displayed network. Spatial input/output semantics are substantial available information, so this is not behavior inference from unlabeled numbers alone. It is a convincing small-network example, not a calibrated estimate across arbitrary policies or basis transformations.

**Experiment 2: diffusion-generated shapes [S4].** Three identical 23,824-parameter DDPM MLPs generate filled circles, squares, and triangles. Astra names the shape categories without forwards or tools, but describes all three as hollow outlines. The keyword-based category score counts these as correct while the detailed behavioral predictions are wrong. The paper's interpretation that this error reflects geometric weight reading rather than a simulated forward pass is plausible, but the error alone does not prove that mechanism. Lesson: evaluate the specificity of explanations and predictions, not only a broad label.

**Experiment 3: language and induction heads [S5].** Three approximately 7.2M-parameter character models have their embeddings and output layers withheld. Astra gets the middle blocks, forward-code/training metadata, and three candidate corpora, and may perform weight-matrix algebra but not run layers. It identifies the corpus in two conversations per network and names the measured copying heads. These are six reader conversations over three trained models, not six independently trained networks. Reported corpus probabilities are approximately 0.48-0.62, not near-certainty. Copy-head agreement uses an induction-score threshold of 0.2; another released scoring field uses 0.3 and gives different labels. The threshold must be explicit in any reproduction. Part of the original Python corpus could not be fully recovered, which further separates replay of released weights from retraining reproduction.

**Project implications:** exploit architecture-aware inspection and known circuit tools, but retain calibrated uncertainty. A model-written explanation is a hypothesis. Counterfactual intervention and behavioral prediction tests are stronger than a plausible narrative or static similarity score.

### 2.3 Experiments 4-6: Writing and repairing weights

**Experiment 4: Kinetix car repair [S6].** Astra sees a 171,664-weight trained actor, the scene/rules/observation layout, the baseline's 256-episode result, and a recorded trajectory. One tool-free reply edits 6,073 values and succeeds in all 1,024 separate test episodes. It wires a small position/velocity-dependent override that backs up and then accelerates over the ramp. This is strong evidence of translating a known control strategy into an existing network. It remains one level and one repair, with comparisons against the rounded network actually shown to the editor. It is not a zero-behavior-information diagnosis.

**Experiment 5: MinAtar initialization [S7].** Astra receives exact game source/rules and a zero-initialized 20,486-parameter architecture; it produces one weight-setting reply per game, without rollout feedback. All three policies beat the reported best-seed PPO 100M-step baselines under the same 1,000-episode evaluation. Against 500M-step PPO, Breakout is much stronger (`517.4` versus `25.3`), while Seaquest (`53.1` versus `88.9`) and Asterix (`53.2` versus `55.2`) do not win. This shows useful domain-program compilation into weights, not across-the-board superiority to training or independence from domain knowledge. PPO checkpoints are not included. Parser fixes reread the same replies rather than obtaining new editor attempts; a reproduction must use the final parsed artifacts.

**Experiment 6: Go initialization with feedback [S8].** A 1,480,274-weight network is written through a CPU sandbox. With zero feedback it wins about 54% against two smaller hidden PPO opponents, but loses to the tuned equal-size PPO baselines. With 50 feedback requests it scores 70-79% against the three equal-size PPO seeds. A games request can play up to 128 full games; a position request can inspect up to 256 positions. There is one editor run per reported budget, so the curve is not an average over editor seeds. The hidden opponents and tuned same-size comparison make this stronger than comparing only against weak practice opponents, but it remains an environment-specific study.

**Project implications:** behavioral feedback and fair baselines matter. Preserve held-out opponents/tasks, compare under matched resources, and count the work behind a feedback request. Training-free is not compute-free, and a sparse handcrafted policy need not imply sparse repair of a distributed language capability.

### 2.4 Experiment 7: A consequential negative result

In the sycophancy experiment, three honest/sycophantic Gemma pairs use the same training setup with frozen tied embeddings/output weights [S9]. Four weights-only investigation variants fail to establish the behavior; the first three produce forced tie-breaks at probability 0.5, and the fourth is partly unfinished. The attention-tool variant identifies all three pairs using input-conditioned attention patterns, without seeing model outputs. One successful pair has only 55% reported confidence.

This directly limits the publication's prospective idea of auditing hidden behavior from weights without evaluation. Input-conditioned observations were necessary in this example. Attention to a suggested answer is evidence of a computation, not by itself proof of a general psychological property or a uniquely causal agreement circuit. The practical architecture should allow limited behavioral/activation probes and abstention, rather than promise weights-only diagnosis.

### 2.5 Experiment 8: The project-relevant positive result

The main Qwen study edits the text part of `Qwen/Qwen3.5-0.8B`, a 24-layer model, with thinking enabled [S10]. The editor has practice problems/answers, a CPU weight-editing kernel, and a bounded inference tool. It is forbidden to train, run autograd/optimizers, or run the model outside that tool. Embeddings, output head, final norm, and trivial whole-tensor rescaling are protected. This is tool-assisted, feedback-driven weight-space programming, not a raw-weight-only one-shot intervention.

| Submission | MATH-500 accuracy | GSM8K accuracy | IFEval accuracy | MATH-500 cap-hit share |
| --- | --- | --- | --- | --- |
| Base | 7.9% | 13.8% | 21.0% | 93.0% |
| 10-request edit | 12.3% | 25.7% | 49.0% | 87.2% |
| 25-request edit | 14.1% | 25.1% | 39.9% | 86.4% |
| 50-request edit | 48.1% | 67.1% | 62.3% | 33.1% |

These are source-reported mean accuracies over samples, **not pass@4** [S10], [S15]. The rows describe separately conducted editor runs, not a controlled single trajectory where only the budget changes. Their final interventions differ: the 10-request run edits 30 output matrices, the 25-request run zeros an attention-head output block, and the 50-request run edits 64 MLP columns in each of two layers. More requests did not guarantee monotonic improvement on every metric.

The 50-request improvement is +40.2 percentage points on MATH-500. Its strongest interpretation is removal of a generation obstacle that prevented existing reasoning from becoming useful output. Cap-hit share is **not a measured loop rate**; the source itself diagnoses repetition, truncation, missing answers, wrong math, and whitespace degeneration separately.

### 2.6 The exact final Qwen intervention

The frozen notes describe layer-specific unit directions from **four matched productive/repetitive original contexts**, measured at residual outputs 16 and 20, and writer selection using original forward request 4 [S11], [S12]. Mapping these residual-output numbers to zero-based layers 15 and 19 must be part of the adapter contract. Pooling all late failed tokens against arbitrary clean spans is not automatically equivalent to these matched contexts.

For each selected matrix and its direction, the final formula is:

$$
W' = W_0 - 6 u\left(M_{64} \odot (u^\top W_0)\right).
$$

The source tensors are `model.language_model.layers.15.mlp.down_proj.weight` and `model.language_model.layers.19.mlp.down_proj.weight`, each of shape `[1024,3584]`, with exactly 64 selected columns. Nominal support is:

$$
2 \times 64 \times 1024 = 131{,}072.
$$

The percentage against an advertised 800M parameters is approximately 0.0164%, not an invariant for other models or the complete multimodal checkpoint. The final BF16 artifact actually changes **130,188** stored entries. Whole-matrix relative Frobenius changes are approximately **10.64%** and **8.11%** [S13]. Small global support does not mean a small functional perturbation.

| Layer | BF16 bit-changed entries | Published post-edit tensor SHA-256 |
| --- | --- | --- |
| 15 | 65,260 | `82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53` |
| 19 | 64,928 | `8ef3c61bc36af113ad58e89dbb0a4a14dcba5a45cb750c15ab515c82ddd12eb2` |

Replay uses base revision `2fc06364715b967f1860aea9cf38778875588b17`. The published application script verifies base dtype/shape/hash, applies exact stored changes, and verifies reconstructed hashes [S14]. A recreated formula with newly estimated directions is a different edit unless it produces those exact tensors.

### 2.7 What made the discovery evidence stronger

The frozen notes report dose-response measurements, separate layer-15/layer-19 interventions, productive/numeric controls, and a norm-matched orthogonal random control [S11]. The edit was frozen before independent confirmation on 1,152 unique practice problems. Paired correctness rose from 10.6% to 60.4%, with **579 correctness rescues and five correctness regressions**. The source reports a paired standard error of about 1.5 percentage points. These are held-out practice results, not the final benchmark results.

The confirmation also exposes an important tradeoff: the edited model has **51 whitespace-tail cutoffs**, versus zero in the baseline, plus wrong answers, missing boxes, and remaining truncation. It is not a complete or universally benign repair. A product policy requiring zero new whitespace degeneration would reject this edit, even though a research reproduction must still report its benchmark gains. Do not silently redefine product acceptance to make a published success pass.

This is the minimum evidential pattern our design should learn from: matched diagnosis, falsifiable interventions, comparable controls, frozen selection, independent confirmation, and explicit regressions. Exact K and alpha are details of the final hypothesis, not substitutes for this process.

### 2.8 Evaluation details that materially change the result

The Qwen profile uses the model's chat template with `enable_thinking=True`, the math instruction requesting a boxed final answer, BF16/vLLM evaluation, temperature `0.6`, top-p `0.95`, top-k `20`, presence penalty `1.5`, seed `0`, and up to `32,768` new tokens [S10], [S15]. MATH-500 uses four samples per problem, GSM8K two, and IFEval two with strict prompt-level checking. Source/checker versions must be pinned as part of reproduction.

The math extractor reads the text after the last `</think>` **when that delimiter exists**; otherwise it reads the full response, and extracts the last valid boxed answer [S16]. A truncated response can therefore still be benchmark-correct. IFEval, in thinking mode, treats absence of `</think>` as an empty usable answer [S15]. These are distinct task rules. Implement explicit evaluator profiles, not one global assumption that cap hit or missing closure always means incorrect.

The released evaluation script has machine-specific paths and imports from a private `weightenv` package; some multiple-choice/IFEval helpers and the full agent harness are not included [S10], [S15], [S16]. Reproduction therefore needs a documented portable evaluator/backend integration, not just a command copied from the logs. Benchmark versions and licenses must remain visible.

The source's final unedited GPQA Diamond score is **22.7%**; the blog figure used an interim **24.5%** over 149 of 198 questions [S10]. Treat final machine-readable results as the baseline of record and document this discrepancy when reproducing the wider benchmark figure.

### 2.9 Independent corroboration and the negative follow-up

The linked reconstructed checkpoint reports an external MATH-500 rerun with 500 problems and four samples each: **51.0% accuracy and 31.6% cap hits**, using vLLM `0.33.0`, Transformers `5.20.2`, and math-verify `0.9.0` [S18]. It reports 43.4M generated tokens and about 51 minutes on an H200. Its base-model comparison is only a 25-problem smoke test, so this is not a fresh complete paired replication of both models. The full multimodal release differs from the source's text-only export, while the language tensors are reported hash-identical. This is external corroboration of the artifact's behavior, not independently repeated discovery and not a local result of this review.

On non-looping Gemma 3 270M IT, Astra's four budgeted runs do not establish a math gain [S17]. Format-agnostic MATH-500 accuracy is 6.8% base versus 7.0%, 7.6%, 7.7%, and 7.0%. These are **lenient/format-agnostic** scores; strict boxed-only scores are separately lower. The Gemma/Qwen comparison also changes architecture, size, sampling, and token limit. It supports a bounded negative result, not a theorem that weight editing can never add capability. Our proposed method needs both known-looping positives and healthy/no-edit negatives.

## 3. Recommended Design Revision

### 3.1 Project goal and scope

Recommended first goal: **reproduce and then evaluate targeted, training-free repair of a demonstrable generation failure in one pinned small language model, while preserving useful behavior and allowing abstention.** Keep the current module boundaries where they fit; the main deficiencies are evidence, data contracts, and lifecycle guarantees, not a need to replace the whole repository.

Use four separate research tracks:

| Track | Question | Required evidence |
| --- | --- | --- |
| Released-artifact replay | Can we rebuild the author's actual edit? | Exact base and post-edit tensor hashes |
| Behavioral reproduction | Does that frozen artifact reproduce the effect? | Matched complete protocol and separately evaluated base/edit |
| Automatic discovery | Can our contrastive algorithm find a useful safe edit? | Causal controls, frozen selection, held-out utility/preservation |
| LLM-guided discovery | Does an editor improve on the deterministic method? | Verified editor access, constrained tools, matched-budget comparison |

Do not make native LLM editing a hidden prerequisite for the deterministic MVP. Do not call the deterministic method a reproduction of the editor's native programming capability. Keep weights-only auditing, adding missing general reasoning ability, universal repair, model merging, quantization, and self-improvement outside the initial acceptance contract.

### 3.2 Preserve the useful implementation foundations

Retain PyTorch/no-gradient execution, the distinction between residual writers and projection inputs, float32 edit computation followed by a deliberate target-dtype cast, candidate restoration from baseline on normal paths, per-matrix support/norm records, and duplicate-key/nonfinite JSON guards. Hook cleanup already has a `finally` block; extend that discipline to mutation and export. The direct masked formula and contrastive contribution score are useful candidate mechanisms when their assumptions are stated and tested.

### 3.3 Make the runtime contract explicit

The architecture needs persisted contracts around the current five stages:

| Contract | Minimum contents |
| --- | --- |
| Experiment specification | Purpose/track, base and tokenizer revisions, dataset roles, evaluation profile, budgets, constraints, selection rule |
| Generation record | Item/sample identity, tokens/text, prompt length, seed, finish reason, thinking markers, extraction/correctness, failure observations |
| Probe specification | Adapter version, real parameter keys, hook boundary, token-position convention, matched spans, weighting, bounded collection policy |
| Localization evidence | Estimator, support counts, direction artifact/hash, stability/confidence, ranking rule, control/intervention results, abstention reason |
| Patch proposal | Exact baseline identity, target keys/shapes/dtypes, direction or replacement artifact, support, alpha, budgets, expected checksums |
| Verification result | Full paired item outcomes, candidate status/rejection reasons, frozen choice, clean preservation, uncertainty, actual resources |
| Export record | Outcome, strict manifest, patch/checkpoint hashes, fresh-load verification, license/attribution metadata |

These are semantic interfaces, not a mandate for seven new source files. Reuse appropriate existing modules; add an abstraction only when a contract genuinely crosses ownership boundaries. Model-generated or user-provided text must be treated as data by any future editor tools, not trusted execution instructions.

The operational lifecycle should be: immutable baseline -> diagnose -> candidate evidence -> bounded reversible edit -> development evaluation -> freeze -> independent confirmation -> verified export. Outcomes include `patched`, `no_patch`, `insufficient_evidence`, `unsupported_model`, `validation_failed`, and `budget_exhausted`. Repeatedly looking at confirmation results while adjusting the edit turns that set into development data; it must not remain labeled held out.

### 3.4 State the mathematics as a testable model

For a unit direction and a selected column, the projection rule is:

$$
w'_j = w_j - \alpha u(u^\top w_j), \qquad
u^\top w'_j = (1-\alpha)u^\top w_j.
$$

At alpha one, that column's contribution along the direction is removed in exact arithmetic. At alpha six, it is reversed with five times its original directional magnitude. This can counteract other writers, but the benefit depends on activation/context and downstream nonlinear computation. It is not a theorem that 64 modified neurons suppress the full residual stream or preserve unrelated capabilities. Reapplying the update without restoration can compound the factor, which is why every candidate must start from the same baseline.

Use unambiguous block indexing, for example embedding output `h^(0)` and block `l` producing `h^(l+1)`. Each adapter must define sublayer ordering, normalization, residual scaling, biases, and hybrid attention where relevant. A post-block direction does not prove which sublayer created it. The contribution estimate

$$
c_j = (u^\top W_{:,j})(\bar m_{\mathrm{loop},j}-\bar m_{\mathrm{clean},j})
$$

is a candidate attribution statistic, not causal certification. Score calibration across layers and handling of negative/uncertain contributions require validation. If matched-context differences are negligible or unstable, the correct result is insufficient evidence, not a forced rank list.

### 3.5 Define success before selecting candidates

Keep at least these metrics separate: repetition evidence, cap-hit share, actual finish/closure, empty/invalid answers, whitespace/control degeneration, task correctness, strict instruction compliance, clean-reference loss/perplexity, and resource use. Maintain both health-transition and correctness-transition tables; do not publish one generic rescue count.

For sampled evaluation, use fixed item/sample seeds and complete paired outputs. Report uncertainty clustered by problem, rather than treating four samples of the same problem as four independent problems. The development selection rule, minimum improvement, clean-behavior tolerance, degeneration budget, and statistical acceptance criterion must be written before examining confirmation results. Require a new holdout if confirmation is used to revise the patch.

Use source-parity and product-safety profiles separately. The former reproduces reported scoring, including correct truncated answers; the latter can require usable final output and stricter degeneration constraints. Research usefulness does not automatically mean the patch is safe to distribute for general use.

### 3.6 Required comparison arms

At minimum compare the immutable baseline, the published edit where applicable, our automatic edit, and norm/support-matched random controls. Include layer-isolated and direction/dose ablations for mechanism claims. Compare non-thinking mode, a sensible longer-token budget, and standard decoding controls as cheaper behavioral alternatives where relevant, with those protocol changes reported separately. If a decoding configuration fixes the observed issue sufficiently, document that tradeoff before claiming weight surgery is necessary.

Test preservation beyond the discovery domain: previously correct math, simple non-math QA, ordinary conversation, system/instruction compliance, formatting, code/list repetition that should remain valid, and a clean language corpus. Broader benchmark and safety claims require broader tests; a math gain and acceptable prompt perplexity alone do not establish them.

## 4. Proposed Work Packages

These packages are the basis for a later step-by-step guide. They are not completed tasks and do not prescribe implementing every future feature now. Dependencies distinguish replay/evaluation work from new-method development.

| Package | Deliverable | Dependencies | Exit gate | Findings |
| --- | --- | --- | --- | --- |
| WP00 | Honest milestone/claim inventory; repaired toy fixture; CPU lock and explicit ordinary CI policy | None | Green numerical/wiring baseline; no unsupported completed/parity claims | F02, F07, F14 |
| WP01 | Pinned source-artifact acquisition/replay and checksum report | Source snapshot and lawful access to base weights | Exact base/post hashes; 2 changed tensors and 130,188 BF16 changes | F02, F05, F12, F17 |
| WP02 | Portable structured data/evaluation profile; measured pilot; complete base/published-edit behavioral reproduction | WP01 for edited artifact | Frozen matched protocol; complete paired benchmark outputs and declared uncertainty/resources | F01, F08, F09, F13 |
| WP03 | Shared generation records and calibrated independent failure observations | WP02 data/profile contract | Detector consistency; negative/recovered/truncated/empty cases handled; no-loop diagnosis succeeds | F03, F08, F10 |
| WP04 | Validated raw-dtype numerical audit and reversible patch transaction | WP00; WP01 artifact contract | Dtype/support invariants and every injected failure rollback pass | F05, F06, F11, F15 |
| WP05 | Explicit Qwen text adapters and bounded matched-span activation capture | WP02, WP03 | Correct names/indexing/positions; measured memory envelope; tiny-model integration checks | F09, F10, F13 |
| WP06 | Automatic-localization feasibility study with causal controls | WP02, WP05; WP04 before persistent edits | Targeted effect beats controls; direction stability and held-out relevance demonstrated, or hypothesis rejected | F04, F15 |
| WP07 | Utility-aware development sweep, freeze, confirmation, and abstention | WP02, WP03, WP04, WP06 | Predeclared utility/preservation gates pass on untouched confirmation; no-output/no-gain candidates rejected | F01, F08, F11 |
| WP08 | Strict provenance, fresh-load checks, and atomic CLI export/reporting | WP04, WP07 | Reloaded artifact agrees with the accepted evaluation and manifest; no-patch/failure states are explicit | F06, F12 |
| WP09 | Exact target-dtype slice distribution and application API | WP08 | Apply/reload is bit-exact on the correct base; incompatible bases refused; rights metadata complete | F17 |
| WP10 | Second-model/domain validation and a healthy-model negative control | WP07, WP08 | Generalization and abstention/preservation reported separately; support matrix updated from evidence | F18 |
| WP11 | Optional editor-assisted or attention/quantization research arm | WP10 and an explicit scope decision | New hypothesis, constrained interface, matched-budget controls, separate gates | F16, F18 |

WP01 can use the upstream replay utility without first implementing our own automatic locator. WP02 should establish whether the motivating effect can be measured on our hardware before extensive algorithm expansion. A small pilot validates feasibility/configuration only; it cannot be labeled full benchmark reproduction. WP04 and WP03 are implementation prerequisites, not evidence that our automatic method works.

The decisive gates are:

1. **Replay gate:** the published edit is reproducible as an artifact, independently of our heuristic.
2. **Behavior gate:** a complete, pinned base/edit evaluation supports the reported effect; discrepancies are explained rather than tuned away.
3. **Discovery gate:** our automatically chosen intervention has causal-control and held-out utility evidence.
4. **Export gate:** the accepted result can be safely reproduced, reloaded, and applied only to its declared base.
5. **Expansion gate:** another model/domain and no-edit negative control justify any broader claims.

If the behavior gate fails, stop and resolve profile/artifact/backend/grader differences before interpreting localization experiments. If the discovery gate fails, retain replay/diagnostic tooling and record a negative result; do not compensate by adding more intervention families without a new justified hypothesis.

## 5. Test and Experiment Requirements

Prefer focused tests in the existing suite and expand coverage as contracts grow. Keep expensive empirical evaluations separate from ordinary unit CI.

| Slice | Required discriminating checks |
| --- | --- |
| Detector | Exact-loop positive; legitimate repetition and recovered suffix negatives; cap without repetition; unclosed thinking; empty/EOS-only output; whitespace before EOS; complete stop metadata |
| Probing | Shared labels; exact layer/key/shape and predictive/observed-position convention; prompt-balanced matched means; no-loop/zero-direction handling; bounded memory; hook cleanup on exception |
| Localization | Known synthetic writer plus protective/random alternatives; sign/scale behavior; stability across held-out contexts; matched intervention controls |
| Patching | BF16/FP16/FP32 raw bytes; cast-roundback; unchanged unselected columns; invalid support/alpha/dimensions; zero/no-op updates; exact restore; partial-mutation failure rollback |
| Sweep | Independent zero baseline; reordered/invalid grids; missing/duplicate IDs; empty references; nonfinite metrics; EOS-only/wrong-short output rejected; no-gain baseline preference; failure rollback |
| Manifest/export | Typed-schema failures despite recomputed digest; cross-field inconsistencies; duplicate/nonfinite JSON; wrong-base refusal; actual stored hash agreement; atomic export and fresh-load parity |
| CLI/integration | Loader/generation profile fidelity; clean/no-patch reports; distinct matrix/layer semantics; no silent overwrites; tiny real-architecture hooks and true autoregressive paths |
| Empirical study | Published-artifact baseline; automatic method versus controls; frozen confirmation; problem-clustered uncertainty; previously good behavior preserved; measured resources; negative-result reporting |

The later task guide should expand one package at a time into: purpose and falsifiable hypothesis; prerequisites/pinned inputs; smallest owning-file changes; focused tests; experiment command and resource budget; expected output artifacts; rollback/no-patch behavior; acceptance and stop conditions. Add the exact commands after the contracts and hardware pilot are agreed, so the guide does not encode today's unsupported assumptions.

## 6. Decisions to Resolve Before the Task Guide

| Decision | Recommended starting point |
| --- | --- |
| Initial product identity | Research reproducer plus validated deterministic de-looping tool; editor discovery is optional later |
| Positive model | The exact published Qwen3.5-0.8B revision, not an arbitrary short-greedy default |
| Negative control | A healthy/non-looping model that should receive no edit; Gemma follow-up informs expectations but is not a direct size-controlled comparison |
| Definition of repair | Held-out usable-answer/task benefit with explicit preservation and degeneration budgets; not lower cap/loop rate alone |
| Research versus export policy | Separate profiles, particularly for the published edit's five regressions and 51 whitespace cutoffs |
| Resource envelope | Measured pilot on available hardware; record tokens/memory/time before promising a fixed request count |
| First distributable format | Exact stored-dtype replacements bound to base hashes; formula factors remain an optional later compression format |
| Breadth of support | One explicit text adapter and dtype/backend profile first; other support only after evidence |
| Candidate-selection confidence | Predeclared development rule and untouched problem-level confirmation; baseline wins without established benefit |

No decision above blocks completing this review. They are the questions the implementation guide should settle explicitly rather than inherit from the prototype.

## 7. Local Verification Record

All local checks used the existing implementation; no implementation fixes were applied. The VS Code selected interpreter was `/opt/venv/bin/python`, with Python 3.12, PyTorch `2.10.0+cu128`, Transformers `5.22.1`, NumPy `2.2.6`, and Safetensors `0.8.0` observed in the active environment.

| Check | Result and scope |
| --- | --- |
| VS Code tests for the existing pipeline test file | **1 passed, 1 failed**; clean-fixture assertion fails before patching |
| Isolated EOS-only acceptance check | Confirmed selection as a rescue; mutation/restoration mocked, actual verifier/detector executed |
| BF16 tensor hashing | Confirmed unsupported-dtype `TypeError` |
| Structurally invalid but correctly digested manifest | Confirmed accepted by current validator |
| Candidate generation exception with real toy patch | Confirmed edited weights remain active after exception |
| Same probe output with/without cap metadata | Confirmed inconsistent `clean` versus `cap_truncation` labeling; no-loop localization raises |
| Current Qwen3.5 AutoModel mappings | Both outer and text config classes map to a causal-LM class; no weights downloaded or full compatibility claimed |
| Existing offline container smoke | **Passed** on CUDA / NVIDIA RTX A3000 12GB Laptop GPU; tiny random Qwen2 wiring only |
| Certificate-bootstrap shellcheck | **Passed** |
| Ruff over source, tests, and smoke script | **105 findings** in active environment; largely style/annotation/import policy, not 105 behavioral defects |
| Default CPU build inputs | Default lockfile absent by directory/tracked-file inspection; no container rebuild attempted |

No pretrained weights were downloaded for this review. No MATH-500/GSM8K/IFEval benchmark, published-edit replay, editor call, full container rebuild, or real-model repair experiment was run. Reported research outcomes remain source-reported. The document's proposed acceptance checks are future work, not claimed test coverage.

## 8. Sources

Publication and linked model-card content were consulted on the review date. Research repository sources below are pinned to the recorded log snapshot. The analysis paraphrases the sources; the original logs remain the authoritative record of what the editor saw and submitted.

- [S1]: publication, authorship/date, experiment summaries, and prospective claims.
- [S2]: release scope, editor/tool execution, omitted earlier trials, artifact layout, and licensing.
- [S3]-[S9]: individual protocols and limitations for Experiments 1-7.
- [S10]: Qwen experiment setup, evaluation, source-number mapping, omissions, and final GPQA discrepancy.
- [S11], [S12]: frozen Qwen discovery/confirmation evidence and exact selected-neuron metadata.
- [S13], [S14]: evaluated BF16 edit manifest and bit-exact application procedure.
- [S15], [S16]: actual evaluation and math extraction/grading code.
- [S17]: Gemma negative follow-up and strict versus format-agnostic scores.
- [S18]: reconstructed public checkpoint, external rerun, version/hardware details, and base-model license.

[S1]: https://rikiyat.github.io/blog/weight-space-programming/
[S2]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/README.md
[S3]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp1_which_game/README.md
[S4]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp2_diffusion/README.md
[S5]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp3_language/README.md
[S6]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp4_fix_car/README.md
[S7]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp5_write_from_scratch/README.md
[S8]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp6_go_from_scratch/README.md
[S9]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp7_sycophancy/README.md
[S10]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/README.md
[S11]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/work_notes/final_report.md
[S12]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/work_notes/frozen_weights_manifest.json
[S13]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/edit/manifest.json
[S14]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/code/apply_edit.py
[S15]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/code/evaluate.py
[S16]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/code/mathqa.py
[S17]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/gemma_followup/README.md
[S18]: https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming

## 9. Post-WIP Code Review (2026-11-13)

Scope: compare `dd3e629` with `96eafd1` (five commits), review the changed
probing/localization/export and synthetic tests, and fix findings before further
project work. The initial worktree was clean. Sections 1-8 above remain the
historical publication/repository review, not current failure/pass claims.

### Prioritized Findings and Fixes

1. **P0: unbacked repair publication.** [The exporter][review-manifest] accepted
	a caller-written `accepted=true` dictionary with no independent confirmation
	owner. It also accepted arbitrary edited target bits rather than the frozen
	proposal. Tampered-target fixtures reproduced publication. Fixed exact stored
	proposal/baseline/budget checks; `export_validated_checkpoint` is disabled.
	`export_recorded_checkpoint` and its new schema explicitly claim only recorded
	bytes and `repair_acceptance_validated=false`. A safetensors re-audit is not
	a fresh native model/config/tokenizer load or utility confirmation.
2. **P0: export path and concurrent overwrite.** Bundle filenames could escape
	staging; checked `os.rename` could replace a concurrent empty directory.
	Fixed pre-I/O plain/distinct filename checks and Linux `renameat2` no-replace
	promotion. Unsupported promotion platforms fail closed. Temporary outputs
	are removed on failure; existing destination identity is preserved.
3. **P1: incomplete artifact idempotence.** An artifact with the correct edited
	target but corrupted untouched tensors could return `already_applied`.
	Reproduced/refused via a corrupted embedding fixture. Recognition now requires
	the complete accepted/recorded checkpoint file digest, not target-only hashes.
4. **P1: stale or weak manifest handoffs.** Frozen coverage maps remained mutable;
	untouched checkpoint weights were not bound to runtime baseline hashes.
	Manifest pins were checked on one read and parsed on another; typed column
	fields and serialized slice contents were insufficiently checked. Fixed
	copied immutable maps, complete mapping/registered-state checks, bounded
	single-read pinned JSON and typed shapes/support, plus fresh slice/source
	audits. A corrupt serializer fixture now refuses before promotion.
5. **P1: recorded-span contamination.** [Recorded probing][review-probe] marked
	every token after first onset as looping, ignoring block ends/gaps/recovery.
	A two-block fixture counted 21 loop tokens instead of 18. Fixed exact spans
	with independent raw-state sums. Duplicate/mixed declarations and aggregate
	token overflow now refuse before model mode/allocation; declared Qwen text
	inspection cannot silently select vision fallback. This is not authenticated
	run/role/engine provenance or semantic productive/repetitive matching.
6. **P1: malformed writer evidence.** [Localization][review-localize] accepted
	broadcastable vector-width mismatches, boolean/fractional counts and nonfinite
	weights/scores. Fixed shared strict contrastive input checks and finite scores.
	Random-direction construction uses one checked draw; matching unit-vector norm
	and support is explicitly not matching actual post-cast perturbation norm.
	Control builders alone are not measured causal evidence.
7. **P1: unsupported roadmap claims.** The plan listed four WP07 confirmation
	APIs and a CI revision absent from the reviewed source. Corrected WP07 to
	pending, removed the unsupported pass claim, and separated recorded-byte
	WP08/WP09 fixtures from accepted-repair export/distribution. Detector controls
	remain synthetic literal-token observations, not real decoder/loop calibration.

### Verification and Remaining Limits

Focused [export/replay controls][review-replay-tests] and
[probe/localization controls][review-pipeline-tests] pass. Final suites pass
1,288 tests each on retained Python 3.12/3.10 CPUs; both offline CPU smokes,
declared Ruff, whitespace, editor diagnostics and document references pass.
Immutable coverage reconstruction also passes. These are local source/fixture
gates, not a hosted CI or empirical repair result.
No pretrained acquisition/generation, real data use, accepted repair/export,
native runtime/profile adoption or environment/lock change occurred. Hosted CI,
independent confirmation, true norm-matched controls, native adapter semantics,
data/rights admission and empirical calibration remain unverified. This bounded
review does not resume the paused research queue or renew closed allowances.

[review-manifest]: ../src/llm_loop_fix/manifest.py
[review-probe]: ../src/llm_loop_fix/probe.py
[review-localize]: ../src/llm_loop_fix/localize.py
[review-replay-tests]: ../tests/test_replay.py
[review-pipeline-tests]: ../tests/test_pipeline.py