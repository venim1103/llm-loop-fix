# llm-loop-fix

Research prototype for training-free, gradient-free diagnosis and repair of small-model generation failures, inspired by [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/).

**Status:** design and implementation preparation. The source is initial skeleton code, not a validated repair toolkit. Published-artifact replay, behavioral reproduction, and automatic discovery have not yet been demonstrated locally. Existing CLI results must not be treated as validated edits.

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
python -m pytest -q tests/test_pipeline.py
llm-loop-fix --help
```

The editable install's dependency policy and repository-owned lint/CI gates are being established in WP00. Tests and CLI help do not establish real-model repair. Pretrained downloads and large inference jobs are not required for the ordinary unit baseline.

Container profiles and bootstrap instructions are in [.devcontainer/README.md](.devcontainer/README.md). The running CUDA environment passed its offline wiring smoke check during the review, but the default CPU dependency lockfile is currently missing; fresh-build reproducibility remains a WP00 task.

The five items in [examples/sample_prompts.jsonl](examples/sample_prompts.jsonl) are illustrative prompts, not a benchmark or a labeled development/confirmation dataset. The skeleton's `run` and `diagnose` commands remain experimental; their input/profile/outcome and export contracts will be revised against the design.

## Implementation Owners

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
