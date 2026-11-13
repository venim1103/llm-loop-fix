# llm-loop-fix

Research prototype for training-free, gradient-free diagnosis and repair of small-model generation failures, inspired by [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/).

**Status:** exact published-artifact replay has been reproduced locally with verified BF16 hashes and unchanged remaining tensors. Behavioral reproduction and automatic discovery remain unverified; the automatic repair pipeline is still a skeleton, not a validated repair toolkit. Existing experimental CLI results must not be treated as validated repairs.

## Project Documents

| Document | Purpose |
| --- | --- |
| [docs/REVIEW.md](docs/REVIEW.md) | Dated publication/log analysis, 18 findings, evidence, and reproduction limits |
| [docs/DESIGN.md](docs/DESIGN.md) | Revised scope, data/evaluation contracts, numerical invariants, and acceptance policy |
| [docs/PLAN.md](docs/PLAN.md) | Step-by-step work packages, dependencies, focused checks, outputs, and stop conditions |

## Approach

1. Reconstruct the exact published Qwen3.5-0.8B edit and verify its baseline/edited tensor hashes.
2. Reproduce its behavior with pinned inference, data, sampling, and grading profiles, starting with a bounded hardware pilot.
3. Test whether matched activation contexts and controlled, reversible weight edits can discover a useful repair automatically.
4. Require usable-answer benefit, independent confirmation, clean-behavior preservation, and rollback-safe verified export. No edit is a valid result.

The publication supports a specific looping-model intervention, not a universal loop direction or a fixed 20-50-forward-pass cost guarantee. Cap hits, repetition, thinking closure, answer correctness, and degeneration are separate observations. LLM-editor integration, broad model support, attention/quantization research, and patch distribution are conditional later work.

## Development

For an existing compatible Python environment:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
ruff check src tests .devcontainer/smoke.py
llm-loop-fix --help
```

The dev extra includes pytest and Ruff. Mandatory lint rules are `E4`, `E7`, `E9`, and `F`, targeting Python 3.10 syntax; annotation/style modernization is not part of this gate. Runtime bounds follow the PyTorch 2.10 line, Transformers 5.17+, NumPy 2.2.6+, and Safetensors 0.8+, while preserving Python `>=3.10`. Those ranges are not an assertion that every allowed combination has been tested.

For hash-locked Linux x86-64 CPU installs, use the matching Python 3.12 or Python 3.10 instructions in [.devcontainer/README.md](.devcontainer/README.md). The 3.12 profile uses Transformers 5.22.1; the 3.10 compatibility profile uses 5.20.2 because 5.22.1 requires Python 3.11+. Do not reuse the 3.12 lock on Python 3.10.

[CI](.github/workflows/ci.yml) defines pytest, Ruff, shellcheck, and offline tiny-model wiring for both CPU profiles. Local checks, both hosted CPU CI jobs, and fresh CPU/CUDA dependency-image builds with offline CPU checks pass. Fresh CUDA-image construction is not hosted GPU-runtime validation. Exact inputs, checks, and limits are recorded in [docs/PLAN.md](docs/PLAN.md). Tests and CLI help do not establish real-model repair. Pretrained downloads and large inference jobs are not required for the ordinary unit baseline.

The five items in [examples/sample_prompts.jsonl](examples/sample_prompts.jsonl) are illustrative prompts, not a benchmark or a labeled development/confirmation dataset. `diagnose` remains experimental. `run` is disabled before model loading or output creation because utility confirmation and validated export are not implemented. Internal health-only sweeps return candidate evidence and restore trial weights; selecting a candidate does not accept or export it.

The [manifest module](src/llm_loop_fix/manifest.py) exposes recorded-byte checkpoint and exact-slice transactions for controlled fixtures. `export_recorded_checkpoint` explicitly declares unvalidated repair acceptance; `export_validated_checkpoint` refuses until independent confirmation exists. Byte preservation does not establish native model reload, utility, rights clearance or accepted repair. See the [dated code review](docs/REVIEW.md#9-post-wip-code-review-2026-11-13).

## Recorded Diagnosis

On Linux, complete persisted diagnostic/development evidence can be inspected without pretrained loading:

```bash
llm-loop-fix diagnose-records \
	--run-manifest RUN.json \
	--generations GENERATIONS.jsonl \
	--expected-run-sha256 RUN_SHA256 \
	--candidate-id base
```

Use the canonical run API and versioned generation records in [src/llm_loop_fix/records.py](src/llm_loop_fix/records.py). Inputs must be regular non-symlink UTF-8 files: at most one MiB for the run and four MiB for complete generation JSONL. The command prints a readonly JSON report; it does not grade, localize, edit, load a model or create an output artifact. Protected roles and incomplete/stale evidence refuse before reporting.

To check independently expected prompt/context snapshots, also supply `--expected-prompt-contexts PROMPTS.json --expected-prompt-contexts-sha256 PROMPTS_SHA256`. The optional one-MiB regular non-symlink UTF-8 file is a complete JSON mapping such as `{"item-a":[[1,4],true]}`: every run item needs its token IDs and explicit boolean thinking-prefix context. The paired hash pins the exact file bytes, including whitespace; duplicate fields, incomplete mappings and prompt/context drift refuse without a report. This does not authenticate tokenizer execution or change the readonly report schema.

The required external hash binds the caller's expected declaration, not authenticated engine/template/model execution. `no_patch` is conservative recorded-evidence abstention, not proof of healthy reasoning; `insufficient_evidence` does not authorize localization or repair. Empirical calibration and accepted utility remain unverified.

## Implementation Owners

The separate [replay utility](src/llm_loop_fix/replay.py) reconstructs the pinned sparse BF16 artifact without inference; [its report](docs/wp01-replay-report.json) establishes tensor parity only. The remaining automatic-pipeline owners below are still subject to their implementation and empirical gates.

| Skeleton | Intended ownership, subject to implementation gates |
| --- | --- |
| [detector.py](src/llm_loop_fix/detector.py) | Independent generation-health observations and calibrated spans |
| [probe.py](src/llm_loop_fix/probe.py) | Explicit model adapters and bounded matched-context statistics |
| [localize.py](src/llm_loop_fix/localize.py) | Candidate directions, signed attribution, controls, and abstention |
| [patch.py](src/llm_loop_fix/patch.py) | Validated numerical edits, actual-dtype auditing, and exact rollback |
| [verify.py](src/llm_loop_fix/verify.py) | Complete paired utility evaluation, frozen selection, and preservation |
| [manifest.py](src/llm_loop_fix/manifest.py) | Strict provenance bound to the actual accepted artifact |
| [cli.py](src/llm_loop_fix/cli.py) | Structured profiles/records, explicit outcomes, and atomic orchestration |
| [test_pipeline.py](tests/test_pipeline.py) | Synthetic numerical/lifecycle checks, distinct from empirical repair evidence |

## License

Repository code is MIT licensed; see [LICENSE](LICENSE). Upstream model weights, research artifacts, and benchmark data retain their respective licenses and attribution requirements.
