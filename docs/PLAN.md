# Roadmap and Step-by-Step Execution Guide

Revision: 2026-11-09.
Current work: WP00/WP01 complete for their declared gates; WP02 records/evaluators and WP04 frozen numerical proposals/verified trial contexts offline-tested; WP03 repetition/tail observations next; live backend/pilot remain pending.
Technical contract: [DESIGN.md](DESIGN.md).
Evidence, findings, and historical baseline: [REVIEW.md](REVIEW.md).

The automatic repair source is initial skeleton code, not an implementation plan we must preserve. Reuse useful modules and tests, but evaluate changes against the design's contracts. Exact published-artifact replay is now locally reproduced in the separate audited utility. No behavioral reproduction, automatic discovery, or validated real-model repair has been completed locally.

## 1. Status and Working Rules

- [x] Publish the research/repository review: commit `1d000ff`, pushed to `origin/main`.
- [x] Revise the design to distinguish source replay, behavioral reproduction, automatic discovery, and conditional editor work; documentation contract checks passed.
- [x] Repair the controlled toy fixture: clean/looping paths are distinct, end-token termination is asserted, and patched clean behavior is preserved; both existing tests pass.
- [x] Establish repository-owned lint/dependency policy, restore the CPU lock, and add a separate Python 3.10 compatibility lock plus ordinary CI.
- [x] Verify clean hash-checked CPU installs, pytest, Ruff, shellcheck, offline CPU/CUDA wiring, and workflow syntax locally.
- [x] Verify hosted CPU CI on Python 3.10/3.12 at `ff80c16`; both matrix jobs and required steps succeeded.
- [x] Record fresh CPU/CUDA image-build validation at `a56d60d`; both hosted jobs and offline image checks passed.
- [x] Reproduce the exact published BF16 artifact with all original/edited hashes, stored-bit support, and 486 unaffected tensors verified.
- [ ] Establish portable profiles, datasets, and measured behavioral evaluation.
- [ ] Validate the automatic method before describing it as a working repair tool.

Checkboxes above describe actual document/delivery work, not completion of the automatic pipeline. WP00's local/hosted/image gates and WP01's exact-artifact gate pass. Hash parity proves a tensor artifact property, not useful generation or a safe automatic repair. Subsequent packages are pending unless a results artifact establishes their exit gate.

For every task, record the hypothesis, pinned inputs, intended behavior, owning code, focused validation, artifact outputs, and stop condition. Make the smallest testable change, run its discriminating check, and update status from evidence. Keep unrelated refactors out of the task. Use Conventional Commits when committing; never mark a whole phase complete because skeleton APIs exist.

Store local experiment products under the already ignored `.runs/<experiment_id>/` rather than committing weights or generated benchmark text. Commit compact, license-compatible summaries and source/profile manifests when a gate is met. Use safe tensor formats, checksums, and per-item IDs; do not silently overwrite previous results.

## 2. Dependency Map

| Package | Status | Depends on | Result needed to proceed |
| --- | --- | --- | --- |
| WP00 | Complete | None | Honest documentation, green focused tests, explicit CI/dependency policy, fresh CPU/CUDA image checks |
| WP01 | Complete: exact artifact parity | Pinned upstream inputs; WP00 before integrating project code | Exact published-artifact replay report |
| WP02 | In progress: item/pairing contracts offline-tested | WP01 for edited artifact | Portable data/profile contracts, hardware pilot, matched behavioral evaluation |
| WP03 | In progress: finish/thinking records offline-tested | WP02's record/profile contracts | Shared independent health observations and safe no-loop diagnosis |
| WP04 | In progress: frozen numerical lifecycle offline-tested | WP00, WP01's artifact contract | Validated proposal/materialization and exception-safe patch lifecycle |
| WP05 | Pending | WP02, WP03 | Explicit text adapter, position semantics, bounded probing |
| WP06 | Pending | WP02, WP05; WP04 before persistent edits | Automatic-localization feasibility and causal-control evidence |
| WP07 | Pending | WP02, WP03, WP04, WP06 | Utility-aware selection, frozen confirmation, abstention |
| WP08 | Pending | WP04, WP07 | Strict provenance and atomically validated export |
| WP09 | Conditional follow-up | WP08 | Bit-exact distributable patch and correct-base application |
| WP10 | Conditional follow-up | WP07, WP08 | Second-model/domain generalization beyond the initial controls |
| WP11 | Deferred research | WP10 and a new scope decision | Separately validated editor/attention/quantization hypotheses |

Record definitions from WP02 can be established before its full benchmark is finished, allowing WP03/WP04 groundwork without claiming empirical success. Upstream replay does not depend on our automatic locator. The practical initial sequence is WP00 -> WP01 -> WP02 contracts/pilot -> WP03 and WP04 -> WP05 -> WP06 -> WP07 -> WP08, with complete behavioral reproduction required before interpreting automatic discovery.

## 3. Work Packages

### WP00 - Specification and Development Baseline

Findings: F02, F07, F14. Owner: [DESIGN.md](DESIGN.md), this roadmap, [README.md](../README.md), [test_pipeline.py](../tests/test_pipeline.py), [pyproject.toml](../pyproject.toml), [Dockerfile](../.devcontainer/Dockerfile), and [CI](../.github/workflows/ci.yml).

**Hypothesis:** we can establish a truthful, green numerical/wiring baseline without claiming the automatic method works or changing the detector to accommodate a broken fixture.

1. Align the public documentation with the revised design; remove unsupported completed/parity and fixed-compute claims. Keep [REVIEW.md](REVIEW.md) as the dated evidence record, not a live checkbox list.
2. Run the existing focused test and reproduce why the supposedly clean toy trajectory enters tokens 7/8/9. Repair the fixture to provide genuinely clean behavior; do not weaken repetition detection. Verify the remaining assertions rather than assuming they will pass.
3. Declare repository-owned Ruff rules and documented development dependencies. Address required lint failures narrowly; do not automatically rewrite all 105 historical style findings. Preserve the declared Python support range unless a separate tested support decision changes it.
4. Generate the missing Python 3.12 CPU dependency lock and a Python 3.10 compatibility lock from the existing inputs, verify the CUDA lock, and align packaging with the combinations actually tested. Interpreter-specific dependencies must not be forced into the wrong profile. Distinguish a successful running CUDA container from fresh-build reproducibility.
5. Add ordinary CI for focused pytest, the chosen Ruff policy, shellcheck, and offline tiny-model wiring. Record any unavailable image-build validation separately; CI success is not research reproduction.

**Checks:** `python -m pytest -q tests/test_pipeline.py`, `ruff check src tests .devcontainer/smoke.py`, `shellcheck .devcontainer/prepare-build.sh`, and `python .devcontainer/smoke.py` after each relevant slice. The historical test/lint commands are known to fail; the task is to make the agreed gates green, not merely list them. Lock/build checks must use the correct CPU/CUDA profile.

**Output:** focused regression tests, explicit lint/dependency/CI policy, reproducible inputs, and a compact baseline report with tool versions and executed/unavailable checks.

**Exit/stop:** no failing mandatory test or missing build input may be described as completed. If a repaired fixture exposes another defect, capture and fix that local slice before widening scope. Do not start large model downloads just to prove WP00.

### WP01 - Replay the Published Artifact

Findings: F02, F05, F12, F17. Inputs: the log snapshot and exact Qwen revision in [DESIGN.md](DESIGN.md#L24); inspect the upstream application code and licenses before running it.

**Hypothesis:** the published BF16 replacement artifact reconstructs exactly two changed tensors without needing an editor model or our estimator.

1. Inventory required files, expected hashes, total download/storage cost, licenses, and base identity. Acquire only the pinned inputs needed for replay; record versions and paths.
2. Verify original tensor dtype, shape, and SHA-256 against the source manifest. Refuse a different base instead of trying to adapt its values.
3. Reconstruct the source edit with the audited upstream exact-application utility or a tested portable equivalent. Retain the upstream artifact as the source of truth, not newly estimated directions.
4. Verify both published post-edit hashes, 130,188 bit-changed entries, intended 64-column support per matrix, and unchanged remaining tensors. Keep the untouched baseline separate.

**Checks:** raw BF16 byte fixtures and full replay checksum comparisons; no generation is required for the replay gate.

**Output:** pinned acquisition manifest and replay report; large base/edited artifacts stay outside version control.

**Exit/stop:** exact hashes establish artifact parity only. Stop on any base/post hash mismatch; do not proceed by accepting a numerically similar tensor. If inputs are inaccessible, record the blockage without claiming parity.

### WP02 - Data, Profiles, Hardware Pilot, and Behavioral Reproduction

Findings: F01, F08, F09, F13. Owners: [cli.py](../src/llm_loop_fix/cli.py), [verify.py](../src/llm_loop_fix/verify.py), and the smallest shared record/profile abstraction needed by both.

**Hypothesis:** complete matched evaluation can measure the published effect on our hardware, and explicit data boundaries prevent selection leakage.

1. Implement/version structured items, generation records, immutable dataset roles, and identity-complete joins. Add duplicate/missing/reordered sample and overlap checks before inference integration.
2. Define portable `source_reproduction` and `validated_repair` profiles exactly as specified in the design. Pin chat template, thinking, sampling, stop metadata, model/tokenizer/backend versions, and graders. Test boxed-answer/truncation/closure differences independently of model inference.
3. Prepare diagnostic/development/confirmation/final benchmark roles and a separate clean-preservation corpus. Prepare a healthy/no-loop negative control now, before automatic selection, rather than postponing it to WP10.
4. Run a bounded hardware pilot on a small identified subset with no selection use of final benchmark answers. Record memory, tokens, throughput, wall time, caching/storage, and budget-exhaustion behavior. Declare the full-run resource budget and confirmation sample-size/uncertainty procedure before automatic trials.
5. Run the pinned base and published edit under identical complete benchmark profiles, initially MATH-500; extend to GSM8K/IFEval as specified for the reproduction claim. Store all paired IDs, correctness, finishes, cap/degeneration counts, profiles, and uncertainty.

**Checks:** evaluator golden cases, complete-key joins, profile serialization/cache identity, tiny offline inference integration, then separately budgeted empirical runs. The pilot is not the full benchmark gate.

**Output:** portable profiles/evaluators, dataset manifests, hardware budget, baseline cache, complete paired evaluation records, and a reproduction report with discrepancies.

**Exit/stop:** the measured result must support a predeclared reproduction criterion under the matched protocol. Define that statistical criterion before the full run; do not tune the method against published percentages. Stop and resolve artifact/backend/profile/grader differences if reproduction fails. Positive reproduction does not validate our automatic estimator.

### WP03 - Shared Health Observations and No-Edit Diagnosis

Findings: F03, F08, F10. Owner: [detector.py](../src/llm_loop_fix/detector.py), with the callers in [probe.py](../src/llm_loop_fix/probe.py) and [verify.py](../src/llm_loop_fix/verify.py).

**Hypothesis:** calibrated independent observations distinguish repetition, healthy long reasoning, truncation, and invalid output without changing labels between stages.

1. Add shared records for actual finish reason, repetition spans/confidence/recovery, tokenizer-aware tail degeneration, thinking closure, answer usability, and independent task correctness.
2. Calibrate on legitimate list/code repetition, recovered repeated phrases, non-repetitive caps, empty/EOS-only output, malformed thinking, and whitespace before EOS.
3. Change probing/verification to consume the same recorded analysis/configuration. No caller may silently drop cap/stop/tokenizer metadata and reclassify an output.
4. Return useful clean/no-patch and insufficient-evidence reports rather than forcing a contrastive direction or throwing for every healthy dataset.

**Checks:** focused detector and call-site consistency tests; no pretrained model is needed.

**Output:** shared health schema, calibrated fixtures/thresholds, consistent diagnosis reports.

**Exit/stop:** a cap alone is not a proven loop; recovered/healthy examples are not forced into the failure spans. Reject unsupported tokenizer/record assumptions before localization.

### WP04 - Numerical Audit and Reversible Patch Lifecycle

Findings: F05, F06, F11, F15. Owner: [patch.py](../src/llm_loop_fix/patch.py) and patch lifecycle boundaries in [verify.py](../src/llm_loop_fix/verify.py).

**Hypothesis:** every proposed update can be materialized in its actual dtype, verified, and undone bit for bit even when trial evaluation fails.

1. Fix raw-dtype SHA-256 and bit-comparison semantics using BF16/FP16/FP32 fixtures. Distinguish intended support, effective stored changes, and post-cast norm perturbation.
2. Prevalidate all targets, real keys, baseline hashes, unit/finite directions, dimensions, support, alpha, matrix/layer semantics, and edit budgets before mutation. Invalid/zero/ineffective candidates fail or abstain explicitly.
3. Preserve an immutable baseline and introduce exception-safe mutation boundaries. Every trial starts from it; restore on partial mutation, generation/evaluation error, OOM, invalid metrics, and rejected candidates.
4. Inject failures before/during/after mutation and confirm exact restoration of all affected weights. Make missing targets and failed restoration explicit errors.

**Checks:** focused dtype/cast/support/no-op/rollback tests; source replay fixtures where appropriate.

**Output:** validated proposal/materialization contract and tested rollback lifecycle. This task does not yet claim utility-aware alpha selection.

**Exit/stop:** no exception path can leave an accidental live edit. A hash or rollback discrepancy blocks subsequent persistent automatic trials.

### WP05 - Explicit Text Adapter and Bounded Probing

Findings: F09, F10, F13. Owner: [probe.py](../src/llm_loop_fix/probe.py), with a small adapter boundary only where required.

**Hypothesis:** architecture-aware matched-span summaries preserve meaningful attribution without retaining all long-context activations.

1. Implement the initial Qwen text adapter with canonical state-dict keys, block order, projection orientation, hook/output semantics, dtype/device, and supported backend. Reject incompatible stacks rather than selecting the first ModuleList.
2. Test source layer/residual-output indexing and predictive versus observed-token positions on tiny Qwen2/Qwen3.5 configurations. Check bounded teacher-forced versus cached-generation behavior explicitly.
3. Collect matched original productive/repetitive contexts with prompt-balanced weights and declared confound controls. Do not pool arbitrary cap tails as proven loop examples.
4. Aggregate only required spans/layers in hooks; record memory limits and cleanup on exception. Measure a representative longer probe before scaling diagnostic requests.

**Checks:** offline adapter/name/shape/position tests, matched-summary fixtures, hook cleanup, measured CPU/GPU resource limits.

**Output:** supported-adapter table, probe specification, bounded summaries, and memory evidence.

**Exit/stop:** wrong names/positions, unstable context semantics, or exceeding the pilot envelope blocks localization claims. A config import alone never passes this gate.

### WP06 - Automatic-Localization Feasibility Study

Findings: F04, F15. Owner: [localize.py](../src/llm_loop_fix/localize.py), with the established probe/evaluation/patch contracts.

**Hypothesis:** a stable matched-context direction and positive writer selection identify a more useful intervention than norm/support-matched random controls.

1. Implement prompt-balanced direction estimates, diagnostic stability checks, signed writer scores, and explicit abstention. Declare any alternate static-alignment estimator as a separate arm, not a fallback.
2. Fix diagnostic/development choices, permitted K/alpha grids, matrices/layers, perturbation constraints, and control construction before comparison runs. Protect negatively contributing columns from undocumented removal.
3. Compare targeted, random-direction, random-column, layer-isolated, and dose-response interventions with productive/numeric controls under equal resource budgets.
4. Measure generated-answer utility as well as repetition signals; retain unsuccessful hypotheses and rejected edits. Do not look at untouched confirmation to choose the mechanism.

**Checks:** synthetic writer/protective-control fixtures; development-only causal experiments; direction stability and measured comparison reports.

**Output:** estimator specification, falsifiable control results, and a discovery-feasibility decision.

**Exit/stop:** if controls explain the apparent gain or utility does not improve, reject the current hypothesis. Retain replay/diagnostic tooling; do not silently add new edit families to manufacture a success.

### WP07 - Utility-Aware Selection, Freeze, and Confirmation

Findings: F01, F08, F11. Owner: [verify.py](../src/llm_loop_fix/verify.py).

**Hypothesis:** the selected candidate improves usable task outcomes on unseen confirmation without violating declared preservation gates.

1. Evaluate an explicit immutable zero baseline independently of grid order, with complete identity joins, finite measured metrics, and profile-keyed cached baseline outputs.
2. Track correctness rescues/losses separately from health transitions. Reject EOS-only, short-wrong, missing-output, unavailable-quality, and no-supported-gain candidates; baseline wins ties without useful improvement.
3. Apply the predeclared development rule and choose the smallest effective supported edit on ties. Freeze direction, support, alpha, constraints, and the entire evaluation profile.
4. Evaluate untouched confirmation and the prepared clean/no-loop preservation suite. Compute the predeclared problem-level paired uncertainty and enforce the design's proposed validated-repair gates. A relaxed tolerance is a new profile fixed before a new evaluation, not a retrospective exception.

**Checks:** reordered/invalid grid, empty references, incomplete samples, early EOS, correctness regression, no-op selection, and exception restoration; then the frozen empirical confirmation.

**Output:** all candidate decisions/rejection reasons, frozen patch identity, paired confirmation/preservation records, and an explicit outcome.

**Exit/stop:** positive confirmation improvement interval plus declared clean/degeneration/perplexity gates permits validated export work. Inconclusive or failed confirmation produces no patch. Revising after confirmation requires a new holdout.

### WP08 - Strict Provenance and Atomic Export

Findings: F06, F12. Owner: [manifest.py](../src/llm_loop_fix/manifest.py) and [cli.py](../src/llm_loop_fix/cli.py).

**Hypothesis:** another run can reproduce the accepted artifact from its declared baseline, and failed publication cannot masquerade as a completed repair.

1. Implement a versioned typed schema and cross-field invariants covering base/model/profile/dataset/code/environment identity, shapes/dtypes/support, effective counts, frozen choice, candidate outcomes, and actual tensor digests.
2. Ensure a recomputed self-digest cannot validate wrong section types, impossible counts, inconsistent alpha/support, or incompatible base identity.
3. Stage model/patch and manifest outputs, reload/apply to the exact baseline, and verify edited and untouched tensors. Bind the accepted evaluation to the actual materialized artifact.
4. Publish atomically only after all checks pass. Refuse accidental overwrites; expose patched/no-patch/unsupported/insufficient-evidence/validation-failed/budget-exhausted outcomes, restoring on failed export.

**Checks:** malformed and tampered manifests, actual-hash mismatch, partial-write/reload failures, fresh-load parity, and CLI lifecycle integration.

**Output:** a reload-verified artifact plus strict provenance or an honest non-patch report.

**Exit/stop:** no committed partial output and no unsupported success status. Failure restores baseline and preserves diagnostic evidence.

### WP09 - Exact Patch Distribution

Findings: F17. Conditional follow-up, after WP08 proves the export contract.

1. Define exact stored-dtype replacement slices, base binding, safe encoding, support metadata, artifact/manifest checksums, and attribution/license obligations.
2. Add export/load/apply with wrong-base refusal and explicit already-applied/refusal behavior; never compound arithmetic updates.
3. Compare with the accepted full checkpoint bit for bit and report size including metadata. For the source support, BF16 values alone are 256 KiB; FP32 values are 512 KiB.

**Checks/output:** deterministic distribution round-trip and wrong-base/repeat-application fixtures; documented license-aware application contract.

**Exit/stop:** distribution parity is an artifact property, not a new behavioral-repair claim. No approximate replacement of source exactness to achieve a size target.

### WP10 - Generalization Beyond the Initial Controls

Findings: F18. Conditional follow-up; the initial clean/no-loop negative control is already required by WP02/WP07.

1. Choose a second model/domain or independent failure setting from observed evidence; declare an explicit compatible adapter/profile and independent dataset roles.
2. Test discovery again rather than assuming source layer numbers, alpha, K, directions, or success transfer. Include healthy/no-edit and cheaper decoding comparison arms.
3. Report utility, abstention, preservation, uncertainty, and resources separately for each setting; update the support/claim matrix only from results.

**Checks/output:** independently frozen evaluation reports and an evidence-backed support matrix.

**Exit/stop:** one-model success is not universal repair. No safe edit on a healthy/non-looping model is a valid desired outcome.

### WP11 - Conditional Editor and New-Mechanism Research

Findings: F16, F18. Deferred until WP10 and an explicit scope decision.

1. For an editor arm, verify model access and define trusted, bounded inspection tools and declarative allowlisted proposals; benchmark/confirmation data remain inaccessible. Prefer no arbitrary code execution for the initial interface.
2. Compare editor assistance with the automatic method under the same measured resource budget; record model/version/prompt/proposals/rejections/approval policy.
3. Treat attention-periodicity, multidirectional, quantized/sharded, or other new mechanisms as separately motivated experiments with new adapter/numerical/preservation gates.

**Checks/output:** independent research protocol and constrained-interface/control evidence, not unconditional new MVP requirements.

**Exit/stop:** no editor-availability assumption, general capability claim, or unbounded new intervention family substitutes for a failed deterministic-discovery gate.

## 4. Immediate Implementation Queue

WP00 step 2 is complete for the controlled fixture: both existing tests pass, with explicit failing-loop, clean-termination, patched-termination, and clean-preservation assertions. The fixture now uses a one-sided ReLU writer; the earlier signed tanh writer showed that over-projection can activate negative clean inputs. The green fixture does not establish safety for signed activations or natural language models; retain such cases as adversarial preservation tests in the later numerical/selection work.

WP00's development/image gates and WP01's exact-artifact replay gate pass. WP02's initial records/profiles, keyed pairing, role-sensitive datasets/overlap audit, source extraction, portable math/quality wrapper, and immutable IFEval instruction metadata pass offline fixtures. WP03's finish/thinking groundwork records explicit prompt context while leaving uncalibrated repetition/tail observations unavailable. Next inspect/pin portable strict IFEval checker code and integrate grading/backend adapters; continue calibrated detector and numerical-lifecycle prerequisites according to their gates. The legacy CLI remains disconnected from the new records. Prepare a separately pinned/budgeted hardware pilot before generation; do not spend WP01's unused time/bytes on inference or infer behavioral reproduction from hashes. BF16 hashing/rollback/selection repairs for the automatic pipeline follow WP04/WP07 rather than a blanket rewrite of all skeleton modules.

Before each code task, inspect its current owning implementation and test; do not assume the historical review baseline is unchanged. Preserve compatible APIs where possible, but explicit identity/outcome/transaction contracts take precedence over accidental skeleton behavior.

Do not start full benchmarks, editor integrations, or attention/quantization expansion simply because the old roadmap listed them next. The first research experiment is exact artifact replay; the first measured inference work is the bounded behavioral pilot.

## 5. WP00 Baseline Report

Recorded 2026-11-06 against code baseline `c13a81ead143fdca717ba6bd88cb5db1e2ca414b` plus the WP00 lint/dependency/workflow changes described here. The existing toy tests are controlled numerical/wiring checks, not naturally generated language-model repair evidence.

| Profile | Python | PyTorch | Transformers | Validation |
| --- | --- | --- | --- | --- |
| Fresh CPU compatibility environment | 3.10.18 | 2.10.0+cpu | 5.20.2 | Hash-verified wheel install, editable project, pip check, Ruff, offline smoke, 2 tests passed |
| Fresh CPU container/CI environment | 3.12.3 | 2.10.0+cpu | 5.22.1 | Hash-verified wheel install, editable project/CLI help, pip check, Ruff, offline smoke, 2 tests passed |
| Existing CUDA environment | 3.12 | 2.10.0+cu128 | 5.22.1 | pip check and offline smoke on RTX A3000 / CUDA 12.8; not a fresh image or wheel install |

All three use NumPy 2.2.6, Safetensors 0.8.0, Accelerate 1.15.0, and SciPy 1.15.3. The lock generator is uv 0.8.22; isolated CPU installs use pip 25.2. Ruff 0.17.4 passes under the explicit `E4/E7/E9/F` policy; ShellCheck 0.9.0 passes the certificate-bootstrap script. Actionlint 1.7.12 validates the new workflow. CUDA re-resolution preserved every locked version and hash, ignoring only generated provenance comments.

The initial attempt to use the 3.12 pins on Python 3.10 correctly failed: Transformers 5.22.1 requires Python 3.11+. Resolving the same inputs for 3.10 produced the separately tested compatible profile. The package's Python `>=3.10` floor is unchanged; other allowed interpreter/dependency combinations remain untested.

| Input | SHA-256 |
| --- | --- |
| [Shared requirements](../.devcontainer/requirements.in) | `ce22d0a58fffef9b533b747ad6270917f5c6c747197ece3f0cbf994b7d301092` |
| [Python 3.12 CPU lock](../.devcontainer/requirements.lock) | `606c63839bc9da12ed99b1b0cabb1e1856296a6dab276fe47b02fd65b0a02fff` |
| [Python 3.10 CPU lock](../.devcontainer/requirements-py310.lock) | `074f4567df69ff9c18303dfa9acfb3c1ae38c414e8e9202fb104bc9d6560fc70` |
| [Python 3.12 CUDA lock](../.devcontainer/gpu/requirements.lock) | `f3b5e631c62639eaea13a4126cda623ecdb99a4bb1ba689fe683cef75b8af27e` |

Local environments, re-resolution files, and the workflow validator are under ignored `.runs/wp00/`. VS Code's selected interpreter remains `/opt/venv/bin/python`. No pretrained weights, benchmark answers/generations, or model edit artifacts were downloaded or committed.

**Hosted CI evidence (2026-11-06):** the [initial run][ci-initial] at `ea03aa0` and the [workflow-instruction run][ci-workflow] at `ff80c16101f604bde044aa685fbc8b98b445b8b1` completed successfully. The latter's public jobs API records two successful CPU matrix jobs, Python 3.10 and 3.12; locked installation, Ruff, pytest, shellcheck, and offline smoke all succeeded in each. This is hosted CPU wiring evidence, not an image build or research result.

**Reload check (2026-11-06):** the worktree was clean at `ff80c16`; VS Code still selects `/opt/venv/bin/python`; ignored `.runs/wp00/` environments survive. Approximately 333 GiB disk, 13 GiB available host RAM, and 12,108 MiB free GPU memory were observed. These are headroom observations, not a replay/inference budget. No pinned Qwen3.5-0.8B cache directory was found at the checked default cache path.

**WP00 closure (2026-11-06):** fresh CPU/CUDA dependency-image builds and offline image wiring/tests passed in [attempt 1][image-attempt-1] at `a56d60d`. CPU ran 12:57:56-12:59:32 UTC (96 seconds); CUDA ran 12:59:34-13:02:21 UTC (167 seconds), serially, 263 runner seconds total. Each passed runner headroom, certificate preparation, fresh build, and offline checks. The CUDA image checks its packaged CUDA 12.8 version but runs smoke on CPU, not a hosted GPU. These results close WP00's declared gates.

**Limits retained:** no local container engine is available. Base-image tags, apt sources, and editable build dependencies remain unfrozen; passing fresh builds does not establish bit-reproducible images, source-artifact parity, behavioral reproduction, or safe automatic repair. Existing local CUDA-runtime smoke remains separate evidence.

## 6. WP01 Inventory Preparation

**Scope recorded before inspection (2026-11-06):** public metadata and pinned source text only. Hypothesis: the log snapshot `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f` and base revision `2fc06364715b967f1860aea9cf38778875588b17` expose enough identity, sizes, rights, and exact-application details to prepare a bounded replay contract without acquiring weights or executing upstream code.

Owner: this plan and the smallest compact, machine-readable inventory if needed. Prerequisites: verified source identities in [DESIGN.md](DESIGN.md), public HTTPS access, and the current WP00 image-build blocker retained. The focused check will compare inventory hashes/counts/shapes with the published manifest, validate revision/file identities and byte totals against metadata, and check all references. Source inspection is not an executed replay.

This preparation permits at most 24 public HTTP requests, at most 5 MiB total response text, 30 seconds per request, and 20 minutes wall time, including failed attempts. It permits zero checkpoint/tensor downloads, upstream-code execution, model loads, generation requests, and benchmark-data acquisition. Prefer bounded model/directory APIs and small source files; do not clone the full research-log repository. Retain only compact license-compatible metadata and findings. Stop on inaccessible/ambiguous identity or rights; report the missing evidence instead of adapting the base or claiming parity.

Acquisition/replay stays blocked until its dependency gate and a separate prospective disk/memory/time/attempt/retention contract are satisfied. A future budget must be based on this inventory, not the source's request/token limits. Completing this inventory does not close WP00, pass WP01's replay gate, or authorize inference.

### Pinned Inventory and Replay Prerequisites

The [machine-readable inventory][wp01-inventory] records nine required files, raw-file digests where inspected, LFS/Git blob identities where payloads were not acquired, source/base rights, exact expected tensor checks, and proposed replay limits. The pinned model API, config, and index agree on the release identity and the two parameter names. The index lists 488 tensors and 1,746,882,752 tensor-data bytes; the checkpoint file is 1,746,942,600 bytes. Including the audit metadata, two license files, NPZ, and both script dependencies gives **1,747,403,950 bytes** of required inputs. Tokenizer/model-class/GPU/generation acquisition is unnecessary for tensor replay.

The actual replay payload is `edit/edit.npz` (387,619 bytes), not the submission `.npy` filenames retained in the source manifest. Its Git blob identity is pinned; a raw SHA-256 remains deliberately unknown until authorized acquisition. The expected BF16 changes remain source-reported: 65,260 entries in layer 15 and 64,928 in layer 19, 130,188 total. Both have 64 changed columns; layer 19 has 1,023 bit-changed rows, not 1,024. These metadata checks are not measured local tensor changes.

**Rights:** the [pinned upstream license][wp01-source-license] assigns CC BY 4.0 to logs/data/networks and MIT to its scripts; the base's pinned license is Apache-2.0. Preserve attribution, upstream URLs, modification notices, and the relevant license files with future acquired artifacts. Repository MIT does not relicense them. No benchmark question text, GPQA content, model weights, or replacement arrays are copied into this repository.

**Application inspection:** [apply_edit.py][wp01-apply] imports the sibling [st_numpy.py][wp01-helper]; both file SHA-256 values are recorded. Sparse replay copies raw BF16 `uint16` bits and assigns stored values at stored indices, rather than re-estimating a direction. The helper can read only the named tensors for `.npy` output, but complete `.safetensors` output reads all tensor payloads and retains another full-sized byte list during saving: approximately 3,493,765,504 payload bytes before temporary arrays/overhead. Its output header is reserialized, so unchanged tensor-byte parity, not whole-container byte equality, is the preservation check.

The upstream utility uses optimization-sensitive `assert` statements for dtype/shape/hash checks and writes the final output directly. Do not treat its printed manifest counts as independent measurements or invoke it unwrapped as a validated export. A bounded replay needs explicit failures, safe numeric NPZ validation without pickle, an immutable baseline, measured raw-bit/support checks for both edits and all 486 unaffected tensors, an owned staging path, and independent fresh-load verification before atomic promotion. These prerequisites are not implemented or executed by this inventory.

**Proposed, not started:** one acquisition attempt per required file and one reconstruction attempt, at most 1.8 GB download, 8 GiB working disk with 10 GiB initially free, 6 GiB process RAM, 45 minutes total, and zero generation/GPU work. Raw inputs/outputs would live under ignored `.runs/wp01-p50c-replay/`; retain one immutable base, one verified output, the licenses, and compact resource/hash diagnostics, deleting only owned failed staging data. Recheck headroom, exact runtime versions, cap enforcement, and dependency gates before activating this proposal. No source-size inference budget is inherited.

**Checks passed (2026-11-06):** all recorded tensor fields and the manifest's raw-file digest match the pinned source; base file sizes/Git blobs/LFS digest match pinned model metadata; required-input, support, changed-entry, unaffected-tensor, and export-copy arithmetic agree. JSON/document diagnostics are clean. This is metadata validation only: none of the expected tensor-byte properties has been measured locally.

The completed preparation used 12 bounded public requests and 127,430 response bytes, with no failed requests, checkpoint/NPZ download, model load, or upstream execution. WP00's fresh-image gate subsequently passed. Actual WP01 replay remains unverified; the next useful step is safe synthetic replay validation and the separate prospective acquisition/replay preflight.

## 7. WP00 Hosted Image-Check Slice

**Prospective scope (2026-11-06):** [the image workflow][image-ci] tests the unchanged shared Dockerfile with the existing Python 3.12 CPU and CUDA locks. Hypothesis: hosted Ubuntu runners can provide fresh-build evidence while no local engine is available. Focused check: actionlint, including embedded shell checks; then the ordinary regression gates and existing offline CPU smoke before pushing. The hosted build result remains `implemented_unvalidated` until both jobs actually pass.

Trigger only relevant container/packaging/workflow changes or explicit workflow dispatch, not routine documentation checkpoints. Run the CPU/CUDA image jobs serially, without a Docker layer cache; each job has a 25-minute timeout. The initial attempt therefore consumes at most 50 hosted runner minutes. This slice allows at most three diagnosed workflow attempts, including cancelled/failed attempts, before retaining a blocker; no unchanged retry or inherited research budget. Each runner must expose at least 24 GiB free disk and 6 GiB available RAM before building. Reject an image above 12 GiB after construction. These are headroom/output gates, not a claimed hard memory/disk cap on Docker's build daemon; the disposable runner and job timeout bound containment. Offline smoke has enforced limits of two CPUs and 6 GiB RAM, no swap expansion, and no network.

Acquisition is limited to the existing image/OS/runtime dependency inputs; there are no pretrained weights, datasets, inference jobs, image publication, new credentials, or privileged local-host changes. Preserve the approved certificate preparation and hash-checked runtime locks. Base-image tags and apt sources retain their documented unfrozen status: a fresh build is not bit-reproducible image parity. Logs retain checkout identity, Docker version, image ID/size, packaged CUDA version, dependency check, tests, Ruff, and smoke results; do not publish the built images. Hosted runner disposal contains build data; no user cache/data cleanup is performed.

Both dependency images run tiny-model smoke on CPU. The CUDA job also verifies that Torch was compiled for CUDA 12.8; it does not establish hosted GPU passthrough or fresh GPU-runtime validation. Existing local CUDA smoke remains separate evidence. Stop on headroom failure, OOM, dependency/hash/TLS failure, timeout, or failed wiring, diagnose before a bounded repair, and keep WP00 open until the recorded fresh-image criteria pass.

**Local checks (2026-11-06):** actionlint 1.7.12 accepted the workflow, including its embedded shell. The surviving Python 3.12 CPU environment passed `pip check`, the full pytest suite (two tests), the declared Ruff scope, and offline CA/tiny-Qwen wiring with Torch 2.10.0+cpu and Transformers 5.22.1. Whitespace/reference checks pass. Neither local Docker execution nor a hosted image result is inferred from these checks. The first push containing the workflow will start attempt 1; do not rerun unchanged successful local gates while waiting for hosted evidence.

**Hosted attempt 1 (2026-11-06):** [run 42221682907][image-attempt-1] is registered at `a56d60dfbb79572ec4fc012c4189e07c5cecd401`. Initially CPU was building and CUDA queued. The resumed exact-commit jobs check now confirms both completed successfully, including all headroom/certificate/build/offline steps. CPU took 96 seconds and CUDA 167 seconds; no retry was needed. This establishes fresh dependency-image construction and offline CPU wiring for both locks, not hosted GPU execution or research parity. No local Docker/Podman engine is available for independent execution.

## 8. WP01 Safe Replay Slice

**Prospective scope (2026-11-06):** add a small [replay utility][replay-code] and [focused fixtures][replay-tests], separate from automatic localization/selection. Hypothesis: validated Safetensors headers and raw BF16 `uint16` bytes let us apply the published sparse replacement without a model class or BF16-to-float conversion, while preserving an immutable baseline and atomically promoting only a freshly audited staging copy.

Before the first code edit, verify selected-interpreter/dependency facts and the official Safetensors API with a tiny offline BF16 file. Focused checks: exact raw hashes and signed-zero bit changes, valid sparse edits and unaffected tensors, dtype/shape/base/edit hash rejection, duplicate/out-of-bounds/noninteger indices, object/pickle archive rejection, existing-output protection, and failure cleanup before promotion. Reuse the installed NumPy/Safetensors and existing pytest conventions. Fixtures are each under 1 MiB; focused tests have a two-minute envelope, at most three diagnosed local repair attempts per defect. No downloaded checkpoint/NPZ, upstream execution, model load, or generation belongs to this slice.

This utility is implemented-unvalidated until focused tests pass; synthetic success proves file/byte/transaction behavior only. Required source checkpoint gates follow the repository instructions. Acquisition remains a separate prospective preflight with the inventory's unreplenished proposed caps; no asset is acquired merely because a utility exists. Stop on failed restoration/cleanup, incompatible library behavior, malformed identity, or an unsafe serialization path; preserve diagnostics and use another genuinely unblocked prerequisite rather than bypassing validation.

**Dependency/check recovery (2026-11-06):** raw BF16 probing exposed the Safetensors 0.8 `TensorSpec` API; Pylance confirmed its keyword signature/buffer lifetime and the adapted offline probe passed. Packaging and container inputs now explicitly require Safetensors `>=0.8,<1`; every existing lock already pins 0.8.0, so no lock versions/hashes change. The updated requirements input SHA-256 is `cfbb8275f8d85c7d9cefa716f23809a13e24adc78bfcb25f15ab723522e25d90`; the earlier WP00 input hash remains a historical baseline. The generated lock line continuation caused one local-validator false failure, corrected without changing a lock.

The resumed Python 3.10 environment points at a disappeared managed Python 3.10.18 runtime; its installed environment remains intact. One 120-second `uv` restore of that exact managed version is prospectively allowed for the required minimum-version check, with system TLS verification, no package reinstall, and no editor-interpreter switch. Recheck the existing interpreter link and run the focused suite/Ruff/dependency check after restoration. This is active-package development maintenance, not model acquisition or an empirical run. Diagnose a failed restore before another path; do not mask a missing compatibility check as a pass.

**Implemented and synthetically checked (2026-11-06):** the utility validates official Safetensors layouts, streams whole-file/per-tensor hashes, pins all input hashes, validates NPZ entry identity/NPY sizes/numeric dtypes before allocation, rejects pickle/object/duplicate/out-of-range inputs, measures stored-bit counts/support, and patches only an owned copy. Fresh official-reader/hash/header checks cover every tensor before a non-overwriting atomic link. Baseline size/hash are rechecked, and failure cleanup removes only owned staging. The hash-required `python -m llm_loop_fix.replay` command emits observed input/output/tensor digests and byte-change measurements, not a useful-repair claim. Package initialization still imports the project's existing PyTorch helpers; no model is instantiated and no tensor conversion/inference is performed.

The full Python 3.12 CPU suite passed 25 tests; all 23 replay tests also passed on restored Python 3.10.18 and under enforced six-GiB address-space/two-minute limits. Both CPU environments passed Ruff and dependency checks, CPU offline smoke passed, and Pylance/editor diagnostics are clear with `/opt/venv/bin/python` still selected. Actionlint accepted ordinary CI's update to run the complete test suite, including replay fixtures. No lock was regenerated because all pinned versions already satisfy the new input minimum. The managed-runtime restore succeeded in 3.05 seconds without reinstalling its environment. Actual published-artifact acquisition/replay and its resource report remain pending the separate preflight.

## 9. WP01 Acquisition/Replay Preflight

**Recorded before acquisition (2026-11-06):** the safe runner is committed in `bf9fa90`, WP00's exact-commit image gate passes, and both Python CPU profiles pass the complete 25-test suite. The 23 replay fixtures pass under enforced six-GiB containment. VS Code selects `/opt/venv/bin/python`; execution uses the retained `.runs/wp00/cpu/bin/python` with NumPy 2.2.6/Safetensors 0.8.0, avoiding CUDA-runtime dependencies. No model is instantiated; existing package initialization imports its installed PyTorch helpers only.

The preflight observed 356,747,194,368 free disk bytes and 19,208,296 KiB available host RAM, above the planned 10-GiB initial disk/six-GiB process limits. No prior `.runs/wp01-p50c-replay/` directory exists. Curl 8.5.0 supplies transfer byte/time limits, HTTPS-only redirects, and default system TLS verification; jq supplies structured inventory parsing; `timeout` and `prlimit` enforce the replay deadline/address-space limit. The pinned inventory contains nine files totalling 1,747,403,950 bytes. All have a known source SHA-256 or Git blob identity; the NPZ is verified against its pinned Git blob and its raw SHA-256 recorded before loading.

Activate the existing proposal, not a renewed budget: one curl acquisition invocation per file, one reconstruction invocation, at most 1.8 GB aggregate download and 45 minutes total from the first acquisition. Record the start/deadline in ignored run state and derive remaining time from it; no repeated or silently resumed failed attempt. Limit each download to its pinned file size and remaining time, verify size/SHA-256/Git blob as applicable before moving its owned partial into the immutable input tree, and stop on any failure/mismatch. Ordinary HTTPS redirects are permitted; no model token, remote-code execution, TLS bypass, benchmark acquisition, or automatic retry is used.

Working disk is bounded by construction: the verified input set is below 1.8 GB, and one staging/output hard-link pair occupies at most one 1.8-GB payload, together below the existing 8-GiB cap. Refuse any pre-existing destination/partial. Replay enforces a six-GiB address-space cap and a 1.8-GB per-file limit, disables excess numerical threads, and is timed using only the unreplenished remainder. Record wall time/peak resident memory when available. Keep one read-only base, one accepted fresh-load-verified output, licenses, and compact hashes/resource diagnostics under `.runs/wp01-p50c-replay/`; never commit those model files. Remove only owned failed staging. Stop on OOM, disk/time/byte exhaustion, invalid identity, unsafe NPZ metadata, hash/support mismatch, or failed cleanup.

The output must preserve all 488 tensor names and the other 486 tensors' exact dtype/shape/bytes, match both published original/edited BF16 hashes, and measure 65,260/64,928 bit changes with 64 changed columns each. Only then record `locally_reproduced` published-artifact parity. No generation, behavioral improvement, or `validated_repair` claim is authorized by this pass. The input and reconstruction budgets are consumed by failed/aborted attempts too.

## 10. WP01 Exact Replay Result

**`locally_reproduced`, `source_reproduction` (2026-11-06):** the [compact replay report][replay-report] records exact original/edited raw BF16 hashes for both published tensors, measured 65,260/64,928 bit changes (130,188 total), 64 changed columns per matrix, and unchanged dtype/shape/bytes for all other 486 tensors. All 488 names and the full base header were preserved. The source's layer-19 bit-changed row count is 1,023 and matches the measurement. The portable utility copied raw stored replacements, not a newly estimated formula or direction.

The nine inputs were acquired once each, 1,747,403,950 bytes total. The NPZ's pinned Git blob matched and its observed SHA-256 is `e92a92aafd1f1068b4fd94aa0fbe081ba3ddea2047ee7bcac7161d861be299e7`. Both model files are read-only and ignored under `.runs/wp01-p50c-replay/`. Output SHA-256 is `bd5c223dc7d0ed48a6857f798a94da6ec46c40b9ff4e7bc31872dba10f943ae5`; this identifies our full-release container, not a published text-only container hash. Published parity is established by tensor hashes and unchanged-base tensor bytes, not whole-container equality with a differently packaged source export.

One reconstruction attempt passed, with fresh official-reader/header/hash verification before non-overwriting atomic promotion. Replay body time was 9.447 seconds; peak RSS 268,521,472 bytes; observed total working disk 3,494,350,156 bytes; completion was observed 206 seconds after the original acquisition start, within the unreplenished 45-minute/six-GiB/eight-GiB bounds. No failed/aborted acquisition, reconstruction retry, unsafe pickle, upstream-code execution, model instantiation, GPU work, or generation occurred. No partial/staging files remained. Only compact license-compatible metadata is committed; base/edit assets and the relevant Apache-2.0/CC-BY-4.0/MIT notices remain in ignored local storage.

Code checkpoint `bf9fa90` passed both local CPU suites and [hosted ordinary CI][replay-source-ci]. WP01's exact-artifact gate is complete. The automatic patch hash/rollback/selection paths are still skeleton behavior and not fixed by this separate utility. Behavioral reproduction, automatic discovery, reasoning improvement, and validated repair remain unverified; WP02's separately scoped contracts/pilot are next.

## 11. WP02 Offline Record Contracts (2026-11-06)

**Prospective slice:** the legacy prompt loader discards stable IDs and cannot
represent dataset boundaries. Add the smallest shared versioned, frozen item
records and strict JSONL ingestion in [records][records], with controlled fixtures
in [record tests][record-tests]. Require explicit source/revision/role, preserve
IDs and evaluator/reference/group fields, and reject missing IDs, duplicate
records/JSON keys, empty prompts, and empty inputs rather than silently dropping
them. Leave legacy inference disconnected until generation/profile contracts
and complete pairing are tested.

**Prerequisites/budget:** WP00 and WP01 gates pass; reuse the existing CPU locks
and selected editor interpreter. Offline synthetic fixtures only, no model or
dataset acquisition, generation, GPU, or model mutation. Up to three local
repair attempts per focused defect; ordinary required regression checks have
standing authority. Focused check: `python -m pytest -q tests/test_records.py`.
Output: auditable contracts/tests, not behavioral reproduction. Stop the slice
on unresolved identity/role loss; do not route benchmark/confirmation answers
into discovery to work around a missing contract.

**Implemented/offline-tested; empirical use `implemented_unvalidated`:** frozen
version-1 items support prompt or immutable chat messages and preserve source,
role, references, evaluator ID, and groups. Generation records retain actual
finish outcomes, including empty/error outputs. Pairing requires an independent
nonempty expected `(item_id, sample_index, seed)` set, rejects duplicates and
missing/extra samples on either side (including a sample omitted by both), and
checks experiment/profile/candidate consistency and matched prompt length.
Strict canonical JSON round-trips restore immutable containers and reject
unknown versions/fields, duplicate keys, and nonfinite values. Dataset digests
are order-independent and include roles/content/provenance; exact cross-role
prompt overlaps and duplicate IDs are rejected. Digests identify content, not
authorship or proof of an untouched holdout.

**Executed checks:** focused stages passed 13, 32, then 42 tests. Full CPU suites
passed 67 tests each on Python 3.10/3.12; declared Ruff, editor diagnostics, and
whitespace passed. No failed approach, acquisition, or inference occurred.
Near-duplicate auditing, full pinned profiles/graders, quality/health enrichment,
and legacy integration remain pending; neither WP02's pilot nor its behavioral
reproduction gate is complete. Next: portable profile and extractor contracts.

## 12. WP02 Portable Extraction Contract (2026-11-06)

**Prospective hypothesis:** portable source extraction must preserve math's
full-response fallback without a closing thinking delimiter, but thinking-mode
IFEval must produce an empty usable response in that case. Pinned source
inspection also shows `\boxed` precedence over `\fbox` and no recovery from a
malformed last selected box. The earlier phrase "last valid box" was imprecise;
the design now records the exact selection rule rather than relaxing it.

**Owners/prerequisites/budget:** [evaluation.py][evaluation] and its [golden
tests][evaluation-tests]; the versioned identity slice is committed in
`9581ea4`. Inspect [pinned math source][math-extraction-source] and [pinned
evaluation source][evaluation-source] without executing their private imports.
This is public source inspection and offline synthetic testing only: no model,
benchmark download, pretrained generation, or GPU job. No dependency/environment
change. Up to three local repair attempts per focused defect, then retain the
failure and choose another unblocked prerequisite. Focused check:
`python -m pytest -q tests/test_evaluation.py` and the profile cases in
`tests/test_records.py`. Output: prompt/extractor and pinned profile identities,
not a completed math/IFEval grader, correctness claim, or behavioral replay.
Stop on unresolved source rule mismatch rather than accepting a more favorable
answer; portable graders and separate source/repair profile identity follow.

**Implemented/offline-tested; empirical use `implemented_unvalidated`:** portable
math prompt/extraction and IFEval usable-response extraction pass 18 synthetic
golden cases, including nested braces, `\fbox` precedence, space-delimited boxes,
malformed/empty selected boxes, and last-delimiter selection. Math extraction
does not use finish reason to override a possible answer. No task correctness
grader or instruction checker is implemented by these text functions.

Initial immutable profile fingerprints include track/task, exact model and
tokenizer revisions, template and dependency-lock digests, backend/grader
versions, dtype, thinking/template mode, decoding, context/token limits,
engine/sampling seeds, seed policy, sample count, and EOS/stop identities. Source
profiles enforce the pinned Qwen revision, BF16 thinking settings, source
sample counts, no additional stop strings, 34,816-token context, math-verify
0.9.0 identity, and strict prompt-level IFEval checker identity. The profile
stores a declared backend version; it does not certify equivalence to source
vLLM or verify the actual tokenizer/template/lock files. `validated_repair`
has a separate identity, but constructing one does not pass its frozen
utility/preservation gates. Fixtures use synthetic digests, EOS values, and
backend identities; they do not establish valid empirical runtime inputs.

**Executed checks:** combined record/extractor focused suites passed 79 tests;
full suites passed 104 tests each on Python 3.10/3.12, with declared Ruff,
editor diagnostics, and whitespace clean. Pylance confirmed all then-resolved
serializer/parser call sites compatible. No dependency/environment mutation,
pretrained generation, benchmark download, or source-script execution occurred.
Next: bounded near-duplicate data audit, portable grading and independent
quality/health records, then backend integration and a prospective pilot.

## 13. WP02 Bounded Cross-Role Overlap Audit (2026-11-06)

**Prospective hypothesis:** identity/role digests and raw exact overlap rejection
alone miss whitespace/case/Unicode variants and similar prompts. A deterministic
prompt-only audit can flag cross-role review candidates without accessing gold
answers, changing roles, or certifying semantic equivalence. Use NFKC/lowercase/
collapsed whitespace normalization and standard-library `difflib.SequenceMatcher`
with autojunk disabled. A proposed 0.9 similarity threshold flags review only;
it is not a frozen repair-acceptance tolerance or evidence of dataset independence.

**Owners/check/budget:** [records][records] and [record tests][record-tests], after
the profile/extractor checkpoint `543409f`. Synthetic offline fixtures only;
no dataset/model acquisition, inference, mutation, environment change, or GPU.
Audit defaults bound work to 100,000 cross-role comparisons and 16,384 content
characters per item; exceeded limits raise before returning any partial audit.
Tests exercise smaller explicit budgets, normalized prompt/chat duplicates,
changed reference answers, similar-but-distinct questions, ordering, and same-role
exclusion. Focused check: `python -m pytest -q tests/test_records.py`. Up to three
local repairs per defect; retain unresolved mismatch evidence and choose another
unblocked prerequisite. Findings require review before any downstream selection
use; no findings cannot prove absence of semantic duplicates or holdout leakage.

**Implemented/offline-tested; empirical use `implemented_unvalidated`:** immutable
audit findings carry dataset digest, method/version, threshold, compared-pair
count, IDs/roles, kind, and similarity. Comparisons exclude reference answers
and same-role pairs; raw prompts and equivalent user-message inputs normalize
consistently. Budget refusal returns no partial audit. Character bounds apply
both before and after NFKC expansion. Reordering items preserves findings.
Similar-but-different arithmetic questions are intentionally flagged for review,
not declared equivalent or removed automatically.

**Executed checks:** nine audit-focused cases pass, with full suites passing
113 tests each on Python 3.10/3.12, declared Ruff and whitespace clean. Earlier
editor diagnostics were clean. No empirical dataset audit, automatic role
assignment, or inference occurred. The default threshold remains a proposed
review heuristic, not a validated semantic duplicate detector. Before discovery,
actual datasets still need pinned manifests, completed audit review, immutable
role membership, and separate holdout-access controls. Next: portable grading
and independent quality/health outcomes.

## 14. WP02 Portable Math Grader Prerequisites (2026-11-06)

**Prospective hypothesis:** source math grading can use its pinned public
math-verify stack on both retained CPU/Python profiles, without private
`weightenv` imports or a hand-written mathematical equivalence checker.
Metadata reports Python >=3.10 for math-verify 0.9.0 and its exact dependency
latex2sympy2-extended 1.11.0. Pin ANTLR runtime 4.13.2 as in the source, retaining
the current SymPy/mpmath and every other existing version. Licenses are
Apache-2.0 (math-verify), MIT (LaTeX parser), and BSD (ANTLR); inspect retained
wheel notices/code before execution. No third-party data or model rights change.

**Owners/check:** [packaging][packaging], [dependency inputs][dependency-inputs],
all three interpreter/backend locks, and the existing evaluation/tests owner.
First check packaging/input pins against version-specific metadata. Then acquire
only three pure-Python wheels under ignored `.runs/wp02-graders/`, verify their
exact sizes/SHA-256, inspect ZIP members/notices/entry points and public parser/
verifier APIs without execution, refresh locks with existing pins constrained,
and hash-install matching CPU profiles only. Preserve selected `/opt/venv`
and its CUDA packages; CUDA lock resolution is not CUDA runtime validation.

**Finite maintenance budget:** wheels total 383,479 bytes; hard allowance two
MiB, one acquisition attempt per wheel, HTTPS/TLS/hash checks, no source builds
or arbitrary remote code. Require one GiB free disk and four GiB available RAM;
allow at most 512 MiB new retained tooling/wheel products. Lock resolution is
serial, at most two attempts per profile (cached/offline first, then a single
bounded metadata resolution only if diagnosed as cache absence), within an
original 15-minute total deadline, no existing-package upgrades. CPU installs
use the generated hash-locked local wheels/current package cache; one attempt
per profile within the same remaining deadline. Stop on identity/hash mismatch,
incompatible floor, unavailable resolution, deadline/OOM, or unrelated pin churn;
retain evidence and continue another authorized prerequisite. Do not retry an
unchanged failure or mutate the selected environment as a workaround.

The later grader wrapper/golden cases remain offline controlled checks, not a
benchmark: serial main-thread library timeouts, explicit unavailable/error
quality states, no pretrained inference or dataset acquisition. Pilot budgets
remain separate. Fresh hosted images/CI must be reported as unverified until
their matching changed-input checks actually pass.

**Implemented/locally tested prerequisites:** optional `evaluation` packaging
and all three runtime locks add only math-verify 0.9.0, latex2sympy2-extended
1.11.0, and ANTLR 4.13.2. Offline uv 0.8.22 resolution passed once per profile,
with prior locks as constraints and verified local wheels. Parsed comparisons
preserved all 41/42/59 existing CPU-3.12/CPU-3.10/CUDA-3.12 pins and hash sets;
only ignored-path constraint annotations were mechanically removed. Both CPU
environments received exactly three packages in one offline hash-checked install
each; dependency checks and bounded synthetic fraction equivalence passed.
The editor still selects `/opt/venv`; no CUDA environment install occurred.

Acquisition used one request per wheel, 383,479 bytes total. One first-wheel
hash check stopped before execution because a manually transcribed hash had
one wrong character; canonical release JSON confirmed the received bytes and
same URL/version/size, with no redownload. A second parser-entry transcription
error was caught by direct metadata comparison before download. Both diagnostics
are retained ignored; the verified hashes are
`3703e7c4885354027fa84409d762a596a2906d1fd4deb78361876bd905a76194`,
`aebb77d52ce269e25028e4bea89ddb14d242ba36bcf7b636496fb5fd9728d234`, and
`fe3835eb8d33daece0e799090eda89719dbccee7aa39ef94eed3818cafa5a7e8` respectively.
Wheel ZIP metadata/import surfaces, parser/verifier timeout machinery, entry
points, and license notices were inspected before execution. No source build,
extra inference dependency, or upstream script was executed. Use structured
metadata digest comparisons, not manually copied wrapped page excerpts.

Observed new package/tooling products total 6,535,688 bytes; maintenance was
observed at 329 seconds within its original 900-second deadline. No budget was
renewed. Initial smoke inherited the container's CUDA device and stopped before
CPU fixture execution; explicit CPU overrides then passed both tiny offline
smokes. Both full suites passed 113 tests, plus Ruff and whitespace. No pretrained
model/data acquisition or empirical grading/behavioral benchmark occurred.

Current dependency-input SHA-256:
`7e3c3ef1d13df61df0badc7353722f930651ed0172e6ba3a0d4e7653579461ed`.
CPU-3.12/CPU-3.10/CUDA-3.12 lock SHA-256s:
`7acbf945be655bf5f86a1eaf66693a957595d3b65e3fbd0e65d9b112ab689de9`,
`066c64625a8bf5454ab876912bc5496589330a991d1647db6ed7353336ead44f`, and
`3fcbd0cb91129b396147a031ec79ba834f0c5a821eb08dc1f52534d543b35feb`.
Next implement the portable timed grader wrapper and independent quality
outcomes; actual IFEval checkers/backend adapters/pilot remain pending.

**Changed-input hosted gate registration:** the normal checkpoint push may
trigger ordinary CPU CI and one fresh image matrix under [image CI][image-ci].
Authorize that required matching dependency check only: serial CPU/CUDA jobs,
existing 25-minute/job timeout, headroom/image/run containment, no image publish
or hosted GPU-runtime claim. At most one matrix attempt for this changed-input
checkpoint; retain its identity/results rather than retrying unchanged failure.
Earlier successful image runs do not validate these new dependency inputs.

## 15. WP02 Timed Math Grading and Quality Records (2026-11-06)

**Prospective hypothesis:** the pinned math-verify API can grade matched math
records while keeping correctness separate from absent references, backend
errors, token/profile violations, and infrastructure failures. A truncated or
unclosed-thinking math response may be source-correct; do not override grading
from sequence length or closure alone. Add immutable quality observations to
generation JSON, preserving IDs/profile and the untouched input record.

**Owners/check/budget:** [evaluation][evaluation], [records][records], and their
existing fixture tests after dependency checkpoint `92f7e25`. Reuse both retained
CPU environments. No acquisition, environment mutation, pretrained generation,
dataset evaluation, or GPU. Main-thread grading only, five-second parse and
verification timeouts; refuse an already active Unix alarm rather than overwrite
its resource guard. Golden equivalence/wrong/no-answer cases and mocked error/
unavailability paths use synthetic answers. At most 20 real controlled grading
calls per focused pass and three local repairs per defect; required regression
gates have standing authority. Focused check:
`python -m pytest -q tests/test_evaluation.py tests/test_records.py`.

Use the source's boxed parse/verify defaults, but request raised exceptions so
operational failures remain explicit `error` with unavailable correctness,
rather than silently becoming wrong answers as in the source helper. This is a
declared safety discrepancy: error records invalidate empirical comparison;
do not claim full grader parity or favorable source accuracy by dropping them.
Math/IFEval correctness, answer usability, thinking closure, repetition, and
finish observations remain independent. Stop on identity/timeout/library error
without relaxing the profile; IFEval checkers and backend integration remain
separate unblocked prerequisites.

**Registered hosted dependency checks:** `92f7e25` has ordinary CPU CI run
`42230675979` in progress and image run `42230676128` queued at registration.
Reuse those identities; no retry or hosted success claim is recorded yet.
Dependency maintenance completed in 434 seconds within the original 900-second
budget, as retained ignored completion metadata records; ordinary synthetic
grader checks do not renew an acquisition or inference budget.

**Implemented/offline-tested; empirical use `implemented_unvalidated`:** matched
math generation grading preserves immutable input identity/token/text/finish
fields and adds strict quality records with extracted answer, format usability,
grader identity/status, nullable correctness, and explicit reason. Wrong or
unboxed answers are distinct from missing references, cancelled/error finishes,
profile token-limit violations, missing/wrong dependency versions, unsupported
threads, pre-existing alarms, and library failures/timeouts. JSON round-trips
restore nested immutable quality records and reject contradictory states.

Real pinned-library fixtures cover equivalent fractions, polynomials, unordered
sets and radicals, wrong values, interval endpoint differences, and source-correct
truncated/unclosed math. Mocked exceptions, including the library's timeout
BaseException, remain `error` with unavailable correctness; existing signal
handlers are restored. Parse/comparison timeout values remain five seconds.
These are per-library operation guards, not a measured whole-benchmark budget.
No threads disable timeouts; active Unix timers are conservatively refused.

**Executed checks:** final full suites passed 135 tests each on Python 3.10/3.12,
plus Ruff, editor diagnostics, and whitespace. The new semantic golden subset
passed five cases before full regression. No new dependency mutation, acquisition,
inference, or benchmark grading occurred. Source helper operational-error scoring
is not claimed identical; retained errors must invalidate empirical comparisons
rather than be discarded from accuracy denominators. Health/prompt-boundary
contracts, IFEval checkers, backend integration, and pilot remain pending.

At this checkpoint, [hosted CPU dependency CI][grader-dependency-ci] completed
successfully at `92f7e25`. [Registered image validation][grader-dependency-images]
is in progress, not failed or completed. Source/test pushes do not trigger its
path-filtered workflow; reuse the same identity and do not duplicate/cancel it.
New wrapper source CI remains unverified until its own matching run passes.

## 16. WP02/WP03 Finish and Thinking Boundary Observations (2026-11-06)

**Prospective hypothesis:** backend-reported length and thinking framing are
independent observations, not an exclusive failure mode or inferred correctness.
The pinned tokenizer/template explicitly supplies `<think>\n` in thinking mode;
generated text may legitimately contain only its closing delimiter. Record
whether the opening was supplied by the prompt, with `None` as unavailable
context, rather than assuming unmatched generated text is malformed.

**Owners/check/budget:** [detector.py][detector], [records][records], and existing
record/math fixture tests after `670969d`. Public inspection of the exact pinned
tokenizer config and chat template only; no tokenizer/model/data acquisition or
inference. Synthetic fixtures, existing CPU environments, no mutation/GPU/new
dependencies. Focused check: `python -m pytest -q tests/test_records.py
tests/test_evaluation.py`. Up to three local repairs per defect; required
regressions remain authorized. Output: frozen cap/thinking observations and
JSON persistence, not calibrated repetition/tail detection or completed WP03.

Backend `finish_reason == "length"` alone supplies the cap observation; token
length/EOS heuristics must not replace it. Unknown prompt context yields unknown
framing conclusions. Known context distinguishes missing closure from nested/
unmatched markers. Repetition and tail-degeneration fields stay unavailable
until measured detectors/configuration can populate them. Matched pairing also
checks equal prompt-boundary context. Stop on mistaken prefix/finish/identity
classification without weakening the quality gate or claiming no loop from
unavailable observations.

**Implemented/offline-tested; empirical use `implemented_unvalidated`:** generation
records preserve nullable prompt-supplied thinking-open context and immutable
health observations. Cap status uses only actual backend finish; marker presence
is distinct from closure completion and malformed nesting/unmatched closure.
Unknown context yields unavailable framing conclusions. Known prompt openings
permit close-only generated text. Missing closure is not necessarily malformed,
and can coexist with a correct extracted math answer. Repetition/tail fields
remain unavailable, not false. Math grading retains these independent health
fields, strict JSON round-trips restore them, and pairing rejects mismatched
prompt context. The legacy exclusive `LoopAnalysis` path is unchanged and remains
pending calibration/integration; this does not complete WP03.

**Executed checks:** focused boundary/record/grader checks passed 121 tests;
two combined quality/framing cases then passed before full suites passed 147
tests each on Python 3.10/3.12. Ruff, diagnostics, and whitespace are clean.
No new dependency mutation, acquisition, inference, or empirical health claim.
The design now explicitly records the pinned template's opening-prefix behavior.

**Hosted checkpoint evidence:** registered dependency image run `42230676128`
completed successfully at `92f7e25`, covering both fresh CPU/CUDA images and
their offline checks. This remains image/CPU evidence, not hosted GPU runtime.
Math-wrapper source [CI at `670969d`][math-wrapper-ci] completed successfully
in run `42231363836`. The new finish/thinking source checkpoint's hosted CI is
not yet verified. Next: immutable IFEval evaluator metadata and portable strict
checking, backend contracts, and separately budgeted pilot preflight.

## 17. WP02 Immutable IFEval Instruction Metadata (2026-11-06)

**Prospective hypothesis:** portable strict checking needs ordered instruction
IDs and their complete kwargs, not a lost prompt-only representation. Add frozen
instruction specs to evaluation items, canonicalize safe JSON args, preserve
null versus empty args in dataset identity, and return fresh decoded mappings
so a checker cannot mutate the stored item. Serialized records expose kwargs as
ordinary JSON objects/null, not opaque escaped metadata strings.

**Owners/check/budget:** [records][records] and [record tests][record-tests],
after health checkpoint `3b145ad`. Source evaluator instruction IDs/kwargs were
inspected in the pinned public script; this slice executes no checker code.
Synthetic fixtures only, no acquisition, environment mutation, pretrained
generation, benchmark audit, or GPU. Instruction argument text is bounded to
65,536 characters before/after canonicalization; reject duplicate keys,
nonfinite values, unsupported shapes/fields, and mutable instruction containers.
Focused check: `python -m pytest -q tests/test_records.py`; up to three local
repairs per defect, then retain evidence and take another unblocked prerequisite.
Test metadata order, nested argument preservation, immutable role/source/item
identity, JSON round-trips, and digest sensitivity. Output is portable data
contracts, not a validated IFEval registry or instruction-compliance result.

**Implemented/offline-tested; empirical use `implemented_unvalidated`:** frozen
ordered instruction specs preserve IDs and safe canonical object/null kwargs.
Decoded kwargs are fresh copies, so nested checker mutations cannot change the
stored item. JSON records expose ordinary argument objects/null and round-trip
order/identity. Dataset digests include instruction metadata; null and empty
objects remain distinct even though a source-compatible checker can consume
both as empty kwargs. Unknown instruction object fields, duplicate JSON keys,
nonfinite/overflowed numbers, unsupported shapes, and argument-size exhaustion
are rejected. Math grading rejects instruction-bearing items instead of silently
ignoring their constraints. No registry code or dataset was acquired/executed.

**Executed checks:** item/record focused suite passed 94 tests and the new math
ownership case passed before final full suites passed 156 tests each on Python
3.10/3.12, with Ruff, diagnostics, and whitespace clean. Next: inspect/pin the
official strict IFEval registry and its runtime/resource dependencies; prepare
a separate bounded integration contract before acquisition/execution. This is
not an IFEval benchmark/compliance or complete WP02 result.

## 18. WP02 Strict IFEval Controller and Registry Prerequisites (2026-11-06)

**Prospective hypothesis:** strict prompt-level checking requires every fixed
instruction to pass; empty constraints, unknown IDs, mutable registries, random
fallback arguments, and operational failures cannot masquerade as compliance.
Implement the portable controller against an immutable declared registry, with
independent response/finish/thinking observations and per-instruction outcomes.
Controlled local checker fixtures prove wiring only, not official rule parity.

Public source inspection pins the official Google Research registry to
`e6890f85757dd84e27ca6df2dd30651dafad28e0` (Apache-2.0). The three core modules
are instructions.py (55,162 bytes), instructions_registry.py (7,240), and
instructions_util.py (19,538); requirements declare absl, langdetect, nltk,
immutabledict. Its kwargs defaults can randomize constraints. NLTK 3.9.2 source
redirects the historical English Punkt `.pickle` path to a pickle-free tokenizer,
but actual safe runtime/table versions/rights must be pinned and tested before
execution. Inspection used public source text only: no registry execution,
package installation, pickle deserialization, table or benchmark acquisition.
Private source `weightenv` checker parity remains unverified.

**Owners/check/budget:** [evaluation][evaluation], [records][records], and existing
tests after `f56e415`. Public source/metadata inspection and synthetic fixtures
only, no environment mutation, acquisition, pretrained generation, or GPU.
Require registry revision equality and fixed argument keys/values; refuse random
or normalized parameter changes rather than changing dataset metadata silently.
Only an implicit `prompt` argument binds to the stored literal prompt; preserve
the explicitly stored `prompt_to_repeat` target unchanged. Reject conflicting
implicit prompts. Five-second whole-controller Unix/main-thread timeout with
exception-safe signal and Python global RNG restoration; refuse active alarms
or unsupported context. Independent language-detector RNG/resources remain a
registry-loader prerequisite, not covered by Python RNG restoration.
At most 20 controlled fixture instructions per focused pass; three local repairs
per defect, required regression gates authorized. Focused check:
`python -m pytest -q tests/test_evaluation.py tests/test_records.py`.

Output: strict-checker control/data contract, not a verified official registry
loader, instruction-engine validation, IFEval score, or behavioral reproduction.
Prepare a separate pinned/license-safe acquisition/dependency/resource contract
for official code and plaintext Punkt tables before any such integration run.
Source checker random fallback behavior and operational errors are conservative
unavailable/error discrepancies; do not drop them to improve reported accuracy.

**Outcome:** the controller passes controlled strict-all/instruction-level,
empty-thinking, absent/unknown/fixed kwargs, immutable registry, explicit-repeat,
null parameterless, nested-mutation, nonboolean/error, timeout-handler, abort,
thread/alarm, and RNG/signal-restoration fixtures. Per-instruction booleans are
immutable and round-trip in generation records; partial operational-error results
are not scored. Both retained CPU interpreters pass 173 full tests; Ruff,
diagnostics, and whitespace pass. Selected Pylance interpreter remains
`/opt/venv/bin/python`; it was not changed. Official checker runtime parity,
normalized-argument compatibility, language detection determinism, pinned tables,
backend/pilot integration, and this checkpoint's hosted CI remain unverified.

## 19. WP02 Official Checker Source/Notice Inspection (2026-11-06)

**Prospective hypothesis/prerequisites:** the pinned official source can supply
strict-rule APIs, but fixed-argument compatibility, resource rights, and
deterministic language detection must be inspected before runtime integration.
The controller checkpoint `20468fb` passes both retained CPU floors. Disk/RAM
headroom rechecked: 329 GiB free disk and 18 GiB available RAM. Selected editor
interpreter remains unchanged. No empirical pilot authority is inherited.

**Pinned inputs/rights:** Google Research checker revision
`e6890f85757dd84e27ca6df2dd30651dafad28e0`, Apache-2.0 source headers and retained
LICENSE. Inspect instructions, registry, utility, strict evaluator, upstream
synthetic tests, README, and requirements. NLTK data revision
`4f15a3d89eefe9748ec1c05be495d91289197155` pins the plaintext `punkt_tab.zip`
archive, Git blob `5e5ff6137d5ee6025e400d1c3a7b21914c48b635`, 4,319,076 bytes.
Public resource XML identifies Jan Strunk but omits licensing terms: quarantine
the archive solely to inspect its notice; do not adopt, execute, or redistribute
tables unless rights are established. No dataset/benchmark questions or answers
are requested. Repository MIT licensing does not relicense these inputs.

**Finite inspection envelope:** ignored `.runs/wp02-ifeval/inspection-manifest.json`
records nine pinned files, exact sizes and Git blobs, totaling 4,469,370 bytes.
Five-MiB download cap, one attempt per file, HTTPS/TLS verification, 60-second
request limits and one original 900-second total deadline. Require at least one
GiB free disk/four GiB available RAM; at most 32 MiB new retained products.
Verify exact size/Git-blob identity, then record SHA-256 receipts. Read only a
bounded archive notice (32 KiB uncompressed), using ZIP metadata/APIs without
pickle, whole-archive extraction, or table execution. Stop on bad identity,
unsafe members, absent/unclear notice, deadline/OOM or cap; retain diagnostics,
do not redownload/renew. Inputs remain immutable ignored; no package install,
third-party execution, selected-environment mutation, pretrained inference,
or artifact publication. Rollback is no adoption/usage, not deletion of user data.

**Owning/check/output:** existing evaluation controller and PLAN. First validate
manifest schema/count/sum/caps and document whitespace; then perform bounded
source/notice inspection. Only compact provenance/rights/findings belong in
committed evidence. Package wheels/locks, safe plaintext tables, language-detector
seed isolation, and official fixture execution require a separate prospective
runtime/dependency contract after this inspection, not an expanded current cap.
Public PyPI metadata reports NLTK 3.9.2 requires Python >=3.9 and click/joblib/
regex/tqdm. Langdetect 1.0.9 declares MIT/six, but its only wheel is Python-2-tagged;
do not retag it or let a resolver build uninspected source. A later bounded,
reviewed source-build plan or an explicitly diagnosed compatible path is needed.

**Inspection outcome:** all nine exact sizes/Git blobs pass; [compact
inventory][ifeval-inventory] records SHA-256 identities and explicit limits.
One attempt/file consumed 4,469,370 bytes; inspection observed at 113 seconds,
4,476,609 retained bytes, within the original caps. Source/license remain
immutable ignored. The archive has 97 metadata-safe members and an 8,574-byte
README, but no explicit license terms. It attributes English model training to
Penn Treebank/Wall Street Journal and Jan Strunk/Tibor Kiss; attribution is not
a grant of resource rights. No table was extracted/used and no checker executed.

**Exact isolated blockers:** full official IFEval runtime needs explicit resource
rights or an in-scope legally established equivalent preserving tokenization;
the repository/index/retained notice inspections do not supply that grant.
Do not infer it from NLTK code licensing, a third-party mirror, or usual package
installation. Langdetect's Python-2-only wheel/source-build path separately
needs inspected, bounded Python-3 dependency work. Official kwargs also sort
keyword/forbidden lists, trim markers/keywords, and lowercase first words/letters:
the current conservative equality controller can refuse those valid normalized
constraints. A source-pinned normalization compatibility contract remains
required; do not treat its current fixture pass as full checker compatibility.

The acquisition attempt allowance is consumed; quarantine/keep receipts, do
not redownload or use this cap for wheels/runtime. Continue the unblocked native
backend-to-record adapter and numerical/health prerequisites by dependency gates.
No empirical or complete WP02/IFEval result follows from this inspection.

Hosted ordinary CPU source CI at `20468fb` passed in [controller CI][controller-ci],
run `42233882176`. Source changed-input gates now cover the offline controller,
health, and metadata tests; they do not validate official runtime, images with
new uninstalled dependencies, empirical profiles, or GPU generation. This
inspection-only checkpoint compares every inventory field against retained JSON
receipts and checks scope/references/whitespace; the unchanged code suite was
not rerun after documentation-only changes.

## 20. WP02 Native Backend-to-Record Adapter (2026-11-06)

**Prospective hypothesis:** complete native request IDs/sample indices and exact
prompt token snapshots can preserve the planned item/sample/seed identity and
observed stop/length/abort/error outcomes without EOS or token-length heuristics.
Same-length different prompts must not join; streaming, missing/duplicate samples,
extra stop rules, unknown finishes, and wrong backend/profile declarations fail.

**Owning/prerequisites/check:** [records][records], existing [record tests][record-tests],
and the GenerationRecord design contract, after `db1205d`. Public vLLM output,
v1/request and finish enum code inspected at
`88077b746cb21e1ad5b9f25d9cae613814e892c3` (native API reference, not an installed
backend pin). Native abort maps to cancellation while retaining the original
reason; a newer repetition stop is unsupported, never relabeled ordinary stop.
Snapshot exact expected prompt IDs and compare complete native IDs/indexes;
store native request/finish/stop details and compare prompt IDs in paired joins.
Prompt-prefix context must be known. Support only the declared engine/sampling
seed schedule; do not invent per-item/sample seeds. Match declared backend/version
and stop/EOS metadata, without claiming an input declaration authenticates a
live runtime. Existing legacy records lacking prompt snapshots remain unavailable
for that stronger identity check; native adapters must supply them.

**Finite offline slice:** no backend import/install, code/data acquisition,
model/tokenizer download, pretrained forward/generation, GPU, or weights mutation.
At most 100 synthetic native samples and 32 generated fixture tokens/sample per
focused pass; three local repairs/defect, required CPU regression gates authorized.
First check `python -m pytest -q tests/test_records.py`. Stop on identity/context/
unsupported finish or budget mismatch; produce no partial record tuple, retain
failures rather than infer a favorable completion. Outputs are immutable records
and synthetic fixtures only. Live request orchestration, exact template/tokenizer/
backend verification, sampler settings, exception capture, legacy CLI integration,
and empirical pilot remain separate pending gates.

**Outcome:** native-record fixtures pass request/version/exact prompt identity,
complete/reordered indices, empty stop/length/abort/error outcomes, preserved
raw finish/declared EOS stop details, immutable snapshots/JSON, context/seed/token
envelope refusal, stronger same-length prompt mismatch joins, and legacy
unavailable snapshots. Independent native request IDs may differ across correctly
matched candidates. Both retained CPU interpreters pass 203 full tests; Ruff,
editor diagnostics and whitespace pass. No native backend was imported or run;
the referenced API pin and fixture declaration are not runtime/profile evidence.
The legacy generation CLI remains disconnected; backend/tokenizer/sampler
integration, prompt-context proof, operational request exception retention,
hosted source CI at this checkpoint and the empirical pilot remain unverified.
Next useful unblocked task: WP04 stored-dtype numerical audit/lifecycle fixtures,
whose WP00/WP01 prerequisites pass independently of IFEval resource rights.

## 21. WP04 Stored-Dtype Numerical Audit (2026-11-06)

**Prospective hypothesis/owners:** [patch.py][patch-code] currently hashes BF16
through unsupported NumPy conversion, labels numerical inequality as stored-bit
changes, and measures a pre-cast norm. Raw byte views can preserve BF16/FP16/FP32
hashes, signed zeros and NaN payload identity, while actual post-cast float64
norm measurement avoids reporting a rounded-away edit as effective. Own the
existing patch module and [pipeline tests][pipeline-tests] after `1ca3275`.
WP00/WP01 prerequisites pass; this does not depend on IFEval resource rights.

**Finite slice/check:** ordinary CPU fixtures only: independent struct-packed
little-endian expected bytes; contiguous/noncontiguous/scalar/empty tensors,
identical NaN bits, signed zero, one-ULP changes, incompatible shape/dtype,
then tiny stored-cast integration. At most 256 elements per numerical fixture,
16 toy matrix edits per focused pass, three local repairs/defect. No pretrained
weights, downloads, GPU, grader resources or empirical inference. First check
`python -m pytest -q tests/test_pipeline.py`; retained 3.10/3.12 regression gates
authorized. Stop on hash/bit/restoration discrepancy or budget mismatch and retain
evidence; restore every synthetic trial. Baseline/input/target prevalidation,
all-target atomicity, ineffective/no-op policy, and exception-safe sweep lifecycle
remain next WP04 work, not implied by audit helpers passing. Automatic persistent
real-model trials remain blocked until those numerical/lifecycle gates pass.

**Outcome:** 15 focused pipeline/audit cases pass. Independent little-endian
BF16/FP16/FP32 byte hashes, scalar/empty/noncontiguous views, identical NaN payloads,
signed-zero and one-ULP entry counts pass. Tiny 2x2 matrix trials verify stored
cast erasure, exact unselected columns, nonzero sub-six-decimal float32 changes,
small BF16 norm stability and bit-identical all-parameter restoration. Post-cast
norms are measured in float64 and not rounded before recording. Both retained
CPU interpreters pass 216 full tests; Ruff, diagnostics and whitespace pass.
No pretrained weights, source model generation, grader resource use or GPU run.
This is audited fixture behavior, not a valid automatic repair/lifecycle result.
All-target prevalidation and exception-safe transaction/restore remain next.

Prior native adapter source CI at `1ca3275` passed in [adapter CI][adapter-ci],
run `42236352415`. Numerical-source CI at `959c6ba` passed in
[numerical CI][numerical-ci], run `42237373642`, at the exact matching head
`959c6baf26cd62a4ee0beb76964af57c0e310796`. This does not validate the subsequent
atomic-mutation changes; do not repeat the prior successful job.

## 22. WP04 All-Target Identity Preflight (2026-11-06)

**Prospective hypothesis/owners:** the patch loop currently skips missing targets
after earlier valid matrices may be mutated. Preflight every selected target
before backups/mutation, and require the legacy projection-map entry to resolve
to the exact real model submodule. Synthetic `layers.*` aliases cannot silently
stand in for a nested canonical model path. Own [patch.py][patch-code] and existing
[pipeline tests][pipeline-tests], after `959c6ba`; WP00/WP01 prerequisites pass.

**Finite check/output/stop:** first focused pipeline tests inject a missing second
target and a wrapped-model synthetic path, requiring explicit ValueError, unchanged
all-parameter hashes and empty backup stores. Tiny 32-parameter model only; at
most 16 new synthetic matrix mutation attempts/focused pass and three repairs/
defect, ordinary CPU regression gates authorized. No pretrained/GPU/inference/
data acquisition. Stop on any uncontained synthetic mutation or restore mismatch;
finally restore owned backups even when a fixture fails. This is target-identity
preflight only. Shape/dtype/unit/support/alpha/budget/baseline validation, finite
materialization and exception-safe copy/evaluation restoration remain pending;
do not describe this small preflight as a completed numerical transaction.

**Identity preflight outcome:** 17 focused pipeline cases pass. A missing later
target and wrapped synthetic name leave every parameter unchanged and create
no backups. Canonical path resolution is a guard, not a validated Qwen adapter.

## 23. WP04 Atomic Mutation and Verified Restoration (2026-11-06)

**Prospective hypothesis/owners:** a private mutation implementation inside an
exception-safe public boundary can restore independently owned original tensors
on partial copy, post-copy audit, OOM or cancellation failures. Preflight duplicate
targets/shared storage and reject stale or live-alias caller baselines. Restore
must validate every target/shape/dtype/canonical identity before any copy, preserve
dtype without silent conversion, and verify original raw hashes. Failure to
restore is fatal, not a normal rejected candidate. Own existing patch/pipeline
tests, following the tested target-identity slice; no new model family or source
acquisition. Reject nonfinite baseline/materialized weights and perturbation
metrics rather than returning an edited invalid candidate.

**Finite offline check:** same tiny 32-parameter fixtures, at most 64 trial/restore
matrix-copy operations/focused pass and three local repairs/defect. Inject one
second-copy failure, one post-copy audit failure, OOM/cancellation, stale/aliased
baselines and bad restore targets/shape/dtype. First run focused pipeline tests;
both retained CPU regression gates authorized. No pretrained/GPU/empirical work,
download, grader resources or environment mutation. On controlled fixture failure,
restore owned originals; fatal-restoration injection needs trusted test cleanup
and a retained diagnostic, never a favorable result. Stop an actual transaction
on restoration/hash/identity failure. This verifies mutation failure containment;
evaluation-time context cleanup, full proposal budgets/unit/support validation,
no-op policy and utility-aware acceptance remain subsequent pending gates.

**Outcome and pause boundary:** 28 focused pipeline cases and 229 full tests on
each retained Python 3.10/3.12 CPU interpreter pass. Ruff, touched-file diagnostics
and whitespace pass. Missing/noncanonical targets, shared selected storage,
stale/aliased baselines and invalid restore inputs fail explicitly. Partial copy,
post-copy audit, OOM and cancellation fixtures recover original tensor bits.
Restoration verifies all hashes after the complete copy batch. Deliberately failed
restoration raises fatal `RestorationError`; independently saved original copy
operations clean up that fixture, rather than reporting a valid recovered model.
No pretrained weights, acquisition, GPU or empirical generation was used.
The user requested a pause after checking; no evaluation-time cleanup slice has
started and no owned job or real-model trial remains active. That cleanup and
full proposal/no-op validation are the next useful WP04 work on an explicit
continuation. This checkpoint's hosted CI remains unverified at the pause.

## 24. WP04 Evaluation-Time Restoration (2026-11-09)

**Prospective hypothesis/owners:** the legacy sweep applies a trial before
generation, health analysis and perplexity, without an exception-safe evaluation
boundary. Inject failures at each stage to demonstrate the live-edit leak, then
restore independently owned baseline tensors when evaluation exits. Own existing
[verify.py](../src/llm_loop_fix/verify.py) and [pipeline tests][pipeline-tests].
The paused atomic checkpoint is `4efac3c`, clean and tracking `origin/main` at
continuation; WP00/WP01 and the offline mutation prerequisite are satisfied.

**Finite checks/outputs/stop:** use the existing 32-parameter fixture, no actual
generation callback or reference forward needed. At most 32 additional tiny
matrix trial/restore copies per focused pass; three local repairs per defect.
First run only the new interruption cases, then the pipeline tests and ordinary
CPU regression gates. RuntimeError, synthetic OOM and KeyboardInterrupt at
generation/health/perplexity must propagate after exact restoration. Fixtures
retain independently owned snapshots and trusted cleanup even on regression.
Stop a transaction on restore/hash/identity failure; classify it as fatal rather
than a rejected trial. No downloads, pretrained/GPU runs, dataset use, grader
resources, dependency changes or revived research budgets. Selection remains
legacy health-only; full proposal/no-op validation and utility acceptance are
not established by this cleanup slice.

**Observed outcome:** nine injected generation/health/perplexity interruptions
first failed their restored-hash assertions after observing an active edit.
Exception-safe evaluation cleanup now passes all nine, plus successful exit and
fatal cleanup after success/error. Successful sweeps return health-only candidate
evidence without winner reapplication; this is not utility selection. The legacy
CLI `run` fails before model dependency import, download or output creation,
because confirmation/accepted-artifact export is absent. Existing CLI refusal
fixture checks exit code, diagnostics and untouched output paths. Both retained
CPU interpreters passed 242 full tests before the subsequent naming repair;
Ruff, diagnostics and whitespace passed. No empirical evaluation was run.

## 25. Canonical Projection Naming Maintenance (2026-11-09)

**Observed failure/hypothesis:** hosted atomic-mutation CI at `4efac3c`, run
`42240014129`, passed pytest/Ruff/shellcheck on both CPU interpreters but failed
the offline tiny-model smoke. Matching local CPU smoke reproduces a missing
`layers.1.self_attn.o_proj` canonical path on Qwen2. Projection discovery must
return registered module paths instead of invented `layers.*` aliases; keep
the strict patch identity check. Own [probe.py](../src/llm_loop_fix/probe.py),
existing pipeline fixtures and the unchanged offline smoke. This is naming
maintenance, not a validated architecture/position/backend adapter.

**Finite check/outputs/stop:** first run matching CPU smoke after the naming edit,
then affected pipeline fixtures and the required retained CPU regression gates.
Reuse the two-block random Qwen2 smoke (hidden size 16, intermediate size 32,
vocabulary 64, synthetic sequences at most 14 tokens); at most three attempts
per defect/profile and one tiny matrix trial per smoke attempt. No pretrained
download, network smoke, GPU, data use, new dependency or selected-environment
mutation. Require canonical module identity and exact restoration; stop on
uncontained mutation, OOM or bad identity, retain diagnostics and repair locally.
Full Qwen3.5 adapter semantics remain WP05, not inferred from this smoke.

**Outcome:** matching Python 3.12/3.10 offline CPU smoke passes with the real
`model.layers.1.self_attn.o_proj` path. All 42 pipeline tests pass, including direct
nested registered-path/state-tensor identity and stale-alias refusal before
mutation. One existing fixture's exact error expectation changed from missing
canonical path to earlier unknown-target refusal; unchanged hashes/empty backups
remain required. Final checkpoint regression and current hosted CI are pending.
Final local regression now passes 243 tests on each retained Python 3.10/3.12
CPU interpreter; Ruff, touched-file diagnostics and whitespace pass. Selected
Pylance interpreter is still `/opt/venv/bin/python`, unchanged. Current hosted
source CI remains unverified; fresh dependency images were not rerun because
no lock/image inputs changed.
Next useful WP04 slice: prevalidate configuration, all directions/shapes/dtypes/
support and declared perturbation budgets before any matrix copy, then make
zero/ineffective stored edits explicit no-op outcomes. No empirical authority
follows from canonical naming or synthetic rollback.

## 26. WP04 Configuration and All-Target Proposal Preflight (2026-11-09)

**Hypothesis/owners:** the second mutation loop still validates directions,
dimensions, finite baseline and support after an earlier valid matrix may be
copied. The newly delivered [independent review](../PLAN_REVIEW.md), finding 2,
confirms this path at its historical `4efac3c` head. Own existing patch/pipeline
tests after pushed `02bb02e`. Preflight every selected target before any backup
or copy, preserve strict canonical identity and reject missing/invalid direction,
metadata, dtype or support. Remove the undocumented absolute-alignment fallback;
insufficient supplied support fails closed, not an invented writer selection.

**Finite check/outputs/stop:** tiny 32-parameter CPU fixtures, at most two selected
2x2 matrices per case and 128 trial/restore copy operations per focused pass;
three local repairs per defect. First run new late-invalid cases, then pipeline
and required retained CPU regression gates. Inject missing/nonunit/zero/nonfinite/
wrong-shape directions, wrong layer/shape metadata, duplicate/out-of-range support,
nonfinite baseline and unsupported dtype; require explicit error, no copy, no
backup publication and unchanged hashes. Trusted independent snapshots restore
fixtures even on regression. No pretrained/GPU/data/third-party acquisition or
environment change. Stop on uncontained mutation or restoration failure. Working
directions use float32 with an explicit 1e-5 absolute unit-norm check (numerical
representation tolerance, not a relaxed repair acceptance gate); retain the
source alpha envelope [0, 6]. Full materialization budgets, no-op policy,
immutable proposal identities and utility/empirical gates remain separate work.

**Focused outcome:** all twelve late-invalid cases first failed (accepted invalid
inputs, non-contract errors or earlier copies); they now pass with no copies and
no published backups. Configuration/fallback fixtures also pass; 71 pipeline
cases pass. This does not establish stored proposal materialization before copy
or declared perturbation budgets. No real-weight mutation was attempted.

## 27. WP04 Pre-Mutation Materialization and No-Op Outcomes (2026-11-09)

**Hypothesis/owners:** after input preflight, a later finite FP16/FP32 baseline
can produce nonfinite stored weights only after an earlier target was copied.
Prepare every actual-dtype candidate and audit metric before backing up/copying
any matrix. Own existing patch/pipeline tests; retain atomic rollback for failures
during the eventual copy/audit phase. Expose effective versus no-op stored
outcomes and avoid copying alpha-zero/unchanged stored tensors.

**Finite envelope/check/stop:** same tiny 32-parameter CPU fixtures, at most two
2x2 candidates/case and 64 new trial/restore copies per focused pass, three local
repairs per defect. First run two later-overflow fixtures, then pipeline and
required CPU regression/smoke checks. Require no copy/backup publication on
FP16 stored-cast or FP32 arithmetic overflow; independent snapshots perform
trusted cleanup on fixture regression. No pretrained/GPU/data acquisition,
new backend, dependencies or environment mutation. Retain diagnostics and stop
on OOM, identity/hash/rollback failure. Preparing all candidates increases bounded
temporary storage; a real-model pilot must account for this separately. Declared
support/norm/layer budgets, immutable proposal identity and utility acceptance
remain pending rather than implied by finite preparation/no-op fixtures.

**Outcome:** both later-overflow fixtures first failed after an earlier copy;
they now reject before any copy/backup publication. Complete-batch copied hashes
are compared with precomputed actual-dtype identities. Partial-copy/post-copy
audit/OOM/cancellation and fatal-restoration fixtures still pass. Alpha zero
avoids arithmetic (including signed-zero changes); erased and orthogonal stored
updates expose `is_noop`/`is_effective`, retain intended-support metrics, and
perform no live copies or publish backups. Both retained Python 3.10/3.12 CPU
interpreters pass 277 full tests; both matching offline tiny-Qwen2 smoke profiles,
Ruff, diagnostics and whitespace pass. This remains synthetic numerical behavior,
not an accepted real-model proposal, utility result or empirical profile proof.
Next: explicit finite perturbation/layer/support budget declarations and
immutable baseline/candidate identity enforcement. Hosted source CI for this
checkpoint is unverified; no dependency/image inputs changed or research cap
was renewed. The review's math-first pilot suggestion preserves the existing
isolated IFEval blocker; math still needs live-profile/data/hardware preflight
and a separately recorded finite pilot envelope before any inference.

Prior lifecycle/naming source CI passed at exact head
`02bb02efbb2cecbf15adb57e22bc3125d1e496f2`, run `42584814898`, in
[lifecycle CI][lifecycle-ci]. This includes the repaired offline smoke, not
subsequent numerical-preparation source or any GPU/empirical gate.

## 28. WP04 Declared Numerical Budgets (2026-11-09)

**Hypothesis/owners:** prepared finite updates are still unconstrained in size.
An immutable explicit budget can reject alpha, separate matrix/distinct-layer
counts, intended/effective support and actual stored per-matrix/full-model norm
changes before any backup/copy. Own existing patch/pipeline tests after `676349b`.
No default experimental tolerance is invented: all budget values are supplied.
Unbudgeted legacy calls remain experimental compatibility paths, not validated
proposals or permission for real-model trials. Accepted proposal integration must
require declared budget identity, pinned baselines and the empirical gates.

**Finite envelope/check/stop:** existing 32-parameter fixture, at most two 2x2
targets/case, 32 new trial/restore copies per focused pass and three repairs per
defect. First run budget cases; enforce no copy/backup publication for each
exceeded limit, frozen declarations and independent actual full-model norm on
a within-budget trial with exact final restoration. Then required retained CPU
regression gates. Model denominator includes all parameters, not just selected
matrices; measure stored-dtype deltas without rounding. Zero limits are valid
conservative declarations; nonfinite/negative/nonsensical counts fail. No
pretrained/GPU/data acquisition, new dependencies or environment mutation.
Stop on OOM, nonfinite metric or identity/restoration failure; retain diagnostics.
Full-model norm passes allocate one float64 parameter at a time, which needs
explicit accounting before any later real-model pilot. Immutable baseline/
proposal lifecycle identity, signed estimator validation and empirical/utility
acceptance remain pending.

**Adjacent measurement risk:** full-model delta measurement assumes selected
storage cannot also mutate unselected parameters/buffers. Reject every selected
weight tied to another registered state tensor, not only another selected target.
Two additional tiny alias fixtures fit this slice's finite envelope and require
unchanged complete state hashes. This is a conservative alias refusal, not
support for tied/specialized editing families. Export the budget via the existing
package surface and exercise that import in the zero-budget fixture.

**Outcome:** all seven over-budget paths refuse before live copies or backup
publication. Frozen/invalid declarations, measured zero-budget baseline and
independent actual full-model norm pass. Parameter/buffer aliases leave complete
state hashes unchanged. Both retained Python 3.10/3.12 CPU interpreters pass
295 full tests; both offline CPU smoke profiles, Ruff, diagnostics and whitespace
pass. Budget fields are explicit and optional only for legacy compatibility;
this is not accepted-proposal integration, a renewed experiment budget or an
empirical repair claim. Next useful slice: enforce original parameter/baseline
identity across trial callbacks and verified restoration; mutation-time guards
alone do not prove evaluation-time object identity. Current hosted CI is pending.

Prior preparation/no-op source CI passed at exact head
`676349b1aaee05c092b56d5d0d4fe59f1d450e56`, run `42585881378`, in
[preparation CI][preparation-ci]. This is not subsequent budget/alias source,
installed vLLM, official IFEval or GPU-runtime evidence.

## 29. WP04 Evaluation-Time Target/Baseline Identity (2026-11-09)

**Hypothesis/owners:** mutation-time identity guards do not extend through
generation/evaluation. Legacy restoration resolves current modules afresh, so
a callback can replace a projection/module/parameter or its storage and obtain
a successful result after copying baseline bytes into the replacement while
original storage remains edited. Own existing verify/pipeline tests after
`22c0666`; this is lifecycle identity containment, not utility selection.

**Finite check/outputs/stop:** existing 32-parameter CPU fixture plus at most one
four-parameter replacement, at most 48 new tiny trial/restore copies per focused
pass, three local repairs per defect. First run six identity cases after
successful/failed evaluation; require fatal identity classification and recovery
of retained original storage where possible. Independent trusted fixture cleanup
rebinds original objects/storage and restores the snapshot even on regression.
Capture original target module/parameter/shape/dtype/device/storage and baseline
hash identities before callbacks; do not treat a changed object as validated
merely because it now has matching bytes. A stale/tampered baseline must fail
before restoration can adopt it. Stop on identity/hash/restore failure, retain
diagnostics and no successful candidate result. No downloads, pretrained/GPU
jobs, real generation, dataset/grader resource use or dependency/environment
changes. Full profile-bound immutable proposal and utility integration remain
pending; later real-model pilots must account for identity-audit memory/time.

**Focused evidence/follow-up:** all six identity cases initially failed (successful
return or only the evaluation error); retained-original storage recovery and
fatal identity checks now pass alongside ordinary evaluation failure/success
cleanup. Within the same finite envelope, add a deliberately tampered baseline
handoff and unexpected unbacked projection changes during zero/nonzero trials.
Never adopt a digest-mismatched backup or return baseline evidence after an
unrestorable unbacked mutation. Fatal fixtures retain independent trusted cleanup;
they must not claim the controller recovered information it no longer owns.

Unbacked zero/nonzero cases initially returned evidence; final original-target
hash checks now refuse both. Add one registry-addition and three ordinary/OOM/
cancellation audit interruptions within the same envelope. Audit failure is fatal
because baseline identity cannot be verified; retain altered-state diagnostics
and trusted fixture cleanup instead of claiming restoration. Identity replacement
recovery does not silently rebind/accept the changed graph. A changed projection
registry cannot inherit the original graph's validation identity.

**Outcome:** all 18 focused identity/baseline/restoration cases pass; both retained
Python 3.10/3.12 CPU interpreters pass 308 full tests and matching offline smoke.
Ruff, diagnostics and whitespace pass. No pretrained acquisition or empirical
generation occurred. This protects the projection registry, not all registered
model state or a profile-bound immutable proposal. Next useful offline slice:
detect callback changes to unrelated registered parameters/buffers and graph
identity without inventing recovery data for unbacked mutations.

Prior budget/alias source CI passed at exact head
`22c06669a98cd4d911aef3ba7d9a775d07e5019d`, run `42587010016`, in
[budget CI][budget-ci]. This is not subsequent callback-identity source or
empirical repair evidence.

## 30. WP04 Whole-Registry Trial Identity (2026-11-09)

**Hypothesis/owners:** projection-only cleanup can return evidence after unrelated
registered parameters/buffers change, disappear, acquire aliases, or switch
objects/storage, and after a weightless module is replaced. Own existing
verify/pipeline tests after `c0769af`; extend identity containment, not acceptance
or a complete serialized proposal.

**Finite check/outputs/stop:** 32-parameter CPU fixture, two integer buffer entries,
one weightless marker and at most one eight-entry replacement; 20 new successful/
failed callback cases, at most 64 new tiny trial/restore copies per focused pass,
three repairs per defect. Trusted fixture cleanup independently restores original
objects/storage/bytes and removes injected aliases. Require fatal classification
while restoring controller-owned trial weights; never invent baseline bytes to
repair unrelated unbacked mutation. Snapshot duplicate-preserving module/tensor
registries, object/storage metadata and original tensor digests. No pretrained
downloads, GPU, empirical generation, datasets, resources or environment changes.
Stop the trial on registry/hash/audit failure and retain failure diagnostics;
later real-model pilots must separately budget full-state hashing costs.

**Outcome:** all 20 cases initially failed (successful evidence or the original
evaluation error). Duplicate-preserving registry/object/storage and final stored
byte audits now pass all 20; all 127 pipeline tests and 328 full tests pass on
both retained CPU interpreters. Both offline smoke profiles, Ruff, diagnostics
and whitespace pass. Controller-owned trial weights recover; unrelated unbacked
bytes/graphs are deliberately not claimed repaired. Full profile-bound proposal
and reusable immutable baseline identity are still pending. Next useful slice:
bind patch preflight to a frozen complete registered-tensor baseline identity so
a later model cannot silently become a new baseline before the first trial.

## 31. WP04 Frozen Registered-Tensor Baseline (2026-11-09)

**Hypothesis/owners:** an internally captured baseline cannot distinguish the
intended original model from pre-trial corruption. Own existing patch/pipeline
tests after `00539a0`. Add deeply immutable complete registered-tensor identities
and a canonical versioned digest, then bind numerical preflight to a declared
baseline plus explicit budget. Byte identity does not authenticate model/tokenizer
revision, architecture code, experiment profile or accepted proposal.

**Finite check/outputs/stop:** existing 32-parameter CPU fixture plus at most four
buffer entries/one alias, at most 64 tiny baseline captures and 32 trial/restore
copies per focused pass, three local repairs per defect. Check canonical immutable
records, duplicate-preserving aliases/nonpersistent buffers, stored signed zero,
malformed records and stale key/shape/dtype/byte identities before any copy or
backup publication. Independent cleanup restores fixture-owned mutations. No
downloads, pretrained/GPU/empirical generation, data or environment changes.
Reject unsupported encoding/layout/device or invalid baseline before mutation;
do not infer model provenance or authorize real-weight trials from a hash match.

**Outcome:** immutable/canonical records passed, while all 13 initial binding
checks failed (zero-edit acceptance, nonzero copy attempt, missing-budget
acceptance). Complete baseline plus explicit-budget preflight now passes all 29
record/binding/unsupported/permitted-repeat checks, including exact restoration
and independent cleanup. Both retained CPU interpreters pass 357 full tests and
matching offline smoke; Ruff, diagnostics and whitespace pass. This identifies
registered tensors, not architecture code, upstream revision, tokenizer or a
profile-bound accepted proposal. Next useful offline slice: freeze direction,
exact support, alpha and budgets with that baseline instead of allowing caller
mutation of the numerical proposal inputs.

Prior callback identity CI passed at exact head
`c0769afa3c44bd9cb2f16d4f33ac53828ebd143d`, run `42590073074`, in
[callback identity CI][callback-identity-ci]. Whole-registry CI passed at exact
head `00539a0bbf7cf4057b2bb8dd4fe8b9e9cf456351`, run `42590580380`, in
[whole-registry CI][whole-registry-ci]. Neither validates this subsequent
frozen-baseline source or an empirical model repair.

## 32. WP04 Frozen Numerical Proposal (2026-11-09)

**Hypothesis/owners:** baseline binding alone leaves caller-owned directions,
supports/configuration mutable. Own existing patch/pipeline tests after
`345f499`; freeze exact baseline target identity, working-FP32 direction, ordered
support, alpha and numerical budgets under a canonical digest. Reuse existing
all-target preparation and stored-dtype auditing, not a second edit algorithm.
This is a numerical proposal, not profile-bound utility acceptance or export.

**Finite check/outputs/stop:** 32-parameter CPU fixtures, at most 64 tiny captures/
direction materializations and 48 new trial/restore copies per focused pass,
three local repairs per defect. Check deep immutability/digest sensitivity,
malformed or baseline-incompatible targets, caller mutation after freezing,
exact support, zero-edit, budget refusal and restoration. Independent trusted
cleanup preserves original fixture tensors. No downloads, GPU/pretrained trials,
empirical generation, datasets, grader resources or environment changes. Refuse
invalid identity/direction/support before copying; do not infer confirmation or
publish artifacts. Profile-bound proposal use and fresh-load export remain gated.

**Outcome:** 20 focused proposal cases pass: canonical deep immutability, caller
mutation isolation, strict target/baseline/direction/support checks, unequal
ordered support widths, zero-budget no-copy, actual stored-budget refusal and
interrupted-copy restoration. One test-only budget diagnostic mismatch was
corrected to the existing message without changing no-copy/identity assertions.
Both retained CPU interpreters pass 377 full tests and matching offline smoke;
Ruff, diagnostics and whitespace pass. Results retain baseline/proposal digests.
Declared-column support uses the same directional projection formula and stored
auditing; this is no new edit family or real-model trial authorization. Next
useful slice: share verified whole-model restoration between legacy sweeps and
a context for frozen proposals, so downstream evaluation cannot leave trials
active. Full experiment-profile/utility/confirmation/export integration remains
unimplemented.

Prior frozen-baseline CI passed at exact head
`345f4994f0d16245914489e1f82637dd747f5f8e`, run `42591518749`, in
[frozen-baseline CI][frozen-baseline-ci]. This does not validate subsequent
frozen-proposal source or an empirical language-model repair.

## 33. WP04 Shared Frozen-Proposal Trial Context (2026-11-09)

**Hypothesis/owners:** frozen numerical application is safe during mutation but
downstream evaluation still needs the whole-model cleanup guard. Own existing
verify/pipeline tests after `8bf9cfc`; extract the already tested restoration
closure once, preserve legacy sweep behavior, then share it with a bounded
frozen-proposal context. No implicit selection, acceptance or export.

**Finite check/outputs/stop:** existing 32-parameter CPU fixture, at most 64 new
tiny trial/restore copies per focused pass, three repairs per defect. First run
existing sweep lifecycle cases after extraction; then check successful/no-op
context exit, ordinary/OOM/cancellation failure, stale baseline before yield,
identity/unbacked state changes and fatal restore failure with independent
trusted fixture cleanup. No downloads, pretrained/GPU jobs, empirical generation,
datasets, grader resources or dependency/environment changes. Context completion
requires exact original stored state and registry identity; failure is fatal,
not candidate evidence. Frozen profile and utility/confirmation/export gates
remain unimplemented, and no real-model trial authority follows.

**Outcome:** extraction preserves all 45 legacy sweep lifecycle cases; the shared
public context passes nine normal/no-op, evaluation failure, stale pre-yield,
unbacked/identity and fatal restoration cases. Both retained CPU interpreters
pass 386 full tests and matching offline smoke; Ruff, diagnostics and whitespace
pass. Independent trusted cleanup recovers deliberate fatal fixtures; production
code never treats failed restoration as accepted evidence. These are offline
lifecycle checks, not profile-bound useful repair. Next useful unblocked task:
WP03 independent repetition/tail observations and their bounded configuration
identity, while live backend/data/pilot and empirical interpretation stay gated.

Prior frozen-proposal CI passed at exact head
`8bf9cfc426b10688c6675559d14967630f07c881`, run `42593004044`, in
[frozen-proposal CI][frozen-proposal-ci]. This does not validate subsequent shared
context source or any real-model trial.

## 34. WP03 Bounded Independent Repetition Spans (2026-11-09)

**Hypothesis/owners:** legacy repetition spans extend through unrelated suffixes
and caps are mixed with loop labels. Own existing detector/records/record tests
after `2f28214`; first report exact immutable tandem spans with bounded immutable
configuration identity, independently of task correctness and pathology. A
literal repeated list/code fragment is observation, not proof of degeneration.

**Finite check/outputs/stop:** synthetic immutable generations of at most 128
tokens per new fixture, at most 24 new scans per focused pass, three repairs per
defect. Check exact block end and following continuation, cap without repetition,
separate spans, boundary/empty cases, invalid specs and budget/backend-unavailable
states. Retain existing unavailable health defaults until schema/pairing/JSON
integration is checked. No model forward, pretrained/GPU jobs, downloads,
datasets, grading resources or environment changes. Default scan budget is 4096
tokens; larger inputs are unavailable, not clean or silently truncated. This
is an offline exact-pattern observer, not empirical loop/tail calibration or
permission to diagnose all source-length samples.

**Outcome:** all 22 focused schema/spec/golden/unavailable cases pass; both
retained CPU interpreters pass 408 full tests, Ruff, diagnostics and whitespace.
Only 13 new tiny scan invocations occur per focused pass, within the prospective
24-scan envelope. No inference/dependency wiring changed, so offline smoke was
not repeated. Exact tandem observation does not label literal list/code patterns
as degeneration, infer loops from caps, or claim healthy recovered semantics.
Legacy loop classification and unavailable health defaults remain unchanged.
Next useful slice: attach observations to immutable health records, preserve
nested JSON identity and refuse differing detector specs in paired comparisons.

Prior shared-context CI passed at exact head
`2f28214f9c3762bba6e12ed2db65e2c51b3fffe5`, run `42593657315`, in
[shared-context CI][shared-context-ci]. This does not validate subsequent
repetition observer source or empirical calibration.

## 35. WP03 Token-Bound Repetition Health/Pairing (2026-11-09)

**Hypothesis/owners:** detached repetition records need exact generated-token
binding, nested immutable JSON restoration and comparison configuration checks
before health attachment. Own existing records/detector/record tests after
`c038424`; attach optional exact-pattern observations without changing no-spec
defaults or equating repetition with pathological degeneration.

**Finite check/outputs/stop:** synthetic records of at most 128 tokens, at most
32 new observer invocations per focused pass, three repairs per defect. First
check complete/unavailable health JSON and same-length wrong-token refusal; then
reject mismatched spec/version/thinking requirements across paired samples and
malformed nested observations. Pylance verifies the existing optional-argument
callers. No model forwards, downloads, GPU/pretrained trials, empirical grading,
datasets or environment changes. Unavailable scans cannot become clean claims;
different detector contracts cannot silently join. Legacy classification, tail
calibration and profile/backend integration remain pending.

**Focused follow-up:** three attachment round-trips passed; all five contract
mismatches initially joined. Shared health-contract identity now refuses them
while allowing actual observation differences. Fifteen JSON/token/pairing cases
pass, but initial pairing fixtures made 34 observer calls against the declared
32-call per-pass limit. Retain that overshoot; reuse immutable health for identical
fixture token snapshots and measure the corrected count before checkpoint gates.
Do not renew or increase the declared limit.

**Outcome:** all 15 attachment/token/JSON/pairing cases pass after correction;
an instrumented retained-CPU run measures 18 observer calls within the unchanged
32-call limit. Pylance confirms all 16 callers remain compatible with the optional
argument. Both retained CPU interpreters pass 423 full tests; Ruff, diagnostics
and whitespace pass. No inference/dependency wiring changed, so smoke was not
repeated. Generated-token digests prevent same-length observation reuse for
different token values. Paired comparison preserves health contract identity
while permitting different actual repetition. This is not pathological-loop
calibration, tokenizer-aware tail detection or profile/backend empirical use.
Next useful offline slice: explicit pinned-tokenizer assumptions and bounded
tail-token observations, without treating EOS-only or literal whitespace as
task-quality evidence.

Prior standalone observer CI passed at exact head
`c0384242e0ec4437194322c85aa74d4ca1f9ef54`, run `42594445884`, in
[repetition observer CI][repetition-observer-ci]. This does not validate subsequent
health integration source or semantic/empirical detector claims.

[repetition-observer-ci]: https://github.com/venim1103/llm-loop-fix/actions/runs/42594445884
[shared-context-ci]: https://github.com/venim1103/llm-loop-fix/actions/runs/42593657315
[frozen-proposal-ci]: https://github.com/venim1103/llm-loop-fix/actions/runs/42593004044
[frozen-baseline-ci]: https://github.com/venim1103/llm-loop-fix/actions/runs/42591518749
[callback-identity-ci]: https://github.com/venim1103/llm-loop-fix/actions/runs/42590073074
[whole-registry-ci]: https://github.com/venim1103/llm-loop-fix/actions/runs/42590580380
[budget-ci]: https://github.com/venim1103/llm-loop-fix/actions/runs/42587010016
[preparation-ci]: https://github.com/venim1103/llm-loop-fix/actions/runs/42585881378
[lifecycle-ci]: https://github.com/venim1103/llm-loop-fix/actions/runs/42584814898
[adapter-ci]: https://github.com/venim1103/llm-loop-fix/actions/runs/42236352415
[numerical-ci]: https://github.com/venim1103/llm-loop-fix/actions/runs/42237373642
[patch-code]: ../src/llm_loop_fix/patch.py
[pipeline-tests]: ../tests/test_pipeline.py
[controller-ci]: https://github.com/venim1103/llm-loop-fix/actions/runs/42233882176
[ifeval-inventory]: wp02-ifeval-source-inventory.json
[math-wrapper-ci]: https://github.com/venim1103/llm-loop-fix/actions/runs/42231363836
[detector]: ../src/llm_loop_fix/detector.py
[grader-dependency-ci]: https://github.com/venim1103/llm-loop-fix/actions/runs/42230675979
[grader-dependency-images]: https://github.com/venim1103/llm-loop-fix/actions/runs/42230676128
[packaging]: ../pyproject.toml
[dependency-inputs]: ../.devcontainer/requirements.in
[evaluation]: ../src/llm_loop_fix/evaluation.py
[evaluation-tests]: ../tests/test_evaluation.py
[math-extraction-source]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/code/mathqa.py
[evaluation-source]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/code/evaluate.py
[records]: ../src/llm_loop_fix/records.py
[record-tests]: ../tests/test_records.py
[replay-report]: wp01-replay-report.json
[replay-source-ci]: https://github.com/venim1103/llm-loop-fix/actions/runs/42225911261
[replay-code]: ../src/llm_loop_fix/replay.py
[replay-tests]: ../tests/test_replay.py
[image-ci]: ../.github/workflows/containers.yml
[image-attempt-1]: https://github.com/venim1103/llm-loop-fix/actions/runs/42221682907
[wp01-inventory]: wp01-artifact-inventory.json
[wp01-source-license]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/LICENSE
[wp01-apply]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/code/apply_edit.py
[wp01-helper]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/code/st_numpy.py

[ci-initial]: https://github.com/venim1103/llm-loop-fix/actions/runs/42218617917
[ci-workflow]: https://github.com/venim1103/llm-loop-fix/actions/runs/42219692091
