# Roadmap and Step-by-Step Execution Guide

Revision: 2026-11-06.
Current work: WP00 complete; WP01 pinned inventory verified, bounded replay preflight next.
Technical contract: [DESIGN.md](DESIGN.md).
Evidence, findings, and historical baseline: [REVIEW.md](REVIEW.md).

The source is initial skeleton code, not an implementation plan we must preserve. Reuse useful modules and tests, but evaluate changes against the design's contracts. This roadmap replaces earlier completed-phase claims; no real-model repair, exact replay, or behavioral reproduction has yet been completed locally.

## 1. Status and Working Rules

- [x] Publish the research/repository review: commit `1d000ff`, pushed to `origin/main`.
- [x] Revise the design to distinguish source replay, behavioral reproduction, automatic discovery, and conditional editor work; documentation contract checks passed.
- [x] Repair the controlled toy fixture: clean/looping paths are distinct, end-token termination is asserted, and patched clean behavior is preserved; both existing tests pass.
- [x] Establish repository-owned lint/dependency policy, restore the CPU lock, and add a separate Python 3.10 compatibility lock plus ordinary CI.
- [x] Verify clean hash-checked CPU installs, pytest, Ruff, shellcheck, offline CPU/CUDA wiring, and workflow syntax locally.
- [x] Verify hosted CPU CI on Python 3.10/3.12 at `ff80c16`; both matrix jobs and required steps succeeded.
- [x] Record fresh CPU/CUDA image-build validation at `a56d60d`; both hosted jobs and offline image checks passed.
- [ ] Establish portable profiles, datasets, artifact replay, and measured evaluation.
- [ ] Validate the automatic method before describing it as a working repair tool.

Checkboxes above describe actual document/delivery work, not completion of any pipeline module. WP00's local, hosted CPU, and fresh-image gates pass. WP01's metadata inventory is verified, but acquisition/replay has not started and its exit gate remains unverified. Subsequent packages are pending unless a results artifact establishes their exit gate.

For every task, record the hypothesis, pinned inputs, intended behavior, owning code, focused validation, artifact outputs, and stop condition. Make the smallest testable change, run its discriminating check, and update status from evidence. Keep unrelated refactors out of the task. Use Conventional Commits when committing; never mark a whole phase complete because skeleton APIs exist.

Store local experiment products under the already ignored `.runs/<experiment_id>/` rather than committing weights or generated benchmark text. Commit compact, license-compatible summaries and source/profile manifests when a gate is met. Use safe tensor formats, checksums, and per-item IDs; do not silently overwrite previous results.

## 2. Dependency Map

| Package | Status | Depends on | Result needed to proceed |
| --- | --- | --- | --- |
| WP00 | Complete | None | Honest documentation, green focused tests, explicit CI/dependency policy, fresh CPU/CUDA image checks |
| WP01 | In progress: inventory verified | Pinned upstream inputs; WP00 before integrating project code | Exact published-artifact replay report |
| WP02 | Pending | WP01 for edited artifact | Portable data/profile contracts, hardware pilot, matched behavioral evaluation |
| WP03 | Pending | WP02's record/profile contracts | Shared independent health observations and safe no-loop diagnosis |
| WP04 | Pending | WP00, WP01's artifact contract | Raw-dtype numerical audit and exception-safe patch lifecycle |
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

WP00 steps 3-5 and the fresh CPU/CUDA image gate pass. WP01's pinned inventory, rights/cost/budget report, and replay prerequisites are prepared and metadata-checked. Next prepare and validate a safe replay runner with synthetic BF16/NPZ fixtures, recheck interpreter/headroom/cap enforcement, and record the separate acquisition/replay preflight before acquiring pinned inputs. No inference is needed for this gate. BF16 hashing/rollback/selection repairs follow their contracts in WP04/WP07 rather than a blanket rewrite of all skeleton modules.

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
