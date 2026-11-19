# Development Container (`llm-loop-fix`)

This setup adapts the multi-profile Dev Container and certificate-trust architecture from `micro_classifier` and `embedded-jev`. It provides hash-locked Python 3.12 CPU/CUDA inputs and offline numerical/wiring checks for the research prototype. Dependency locks do not pin the base image or apt contents; fresh image builds and real-model repair are separate validation gates.

## Profiles

| Configuration | Use |
| --- | --- |
| [CPU](devcontainer.json) | Default; weight-space probing, unit tests, and small model patching without a GPU requirement |
| [CUDA / Docker](gpu/devcontainer.json) | NVIDIA host with Docker and NVIDIA Container Toolkit |
| [CUDA / Podman / WSL](gpu-podman/devcontainer.json) | WSL2 passthrough with Podman and NVIDIA CDI (`--device nvidia.com/gpu=all`, `/dev/dxg`) |
| [NVIDIA NGC vLLM / Docker](gpu-ngc/devcontainer.json) | Opt-in WP02 backend candidate; uses NVIDIA's bundled Python and CUDA stack without the project's GPU wheel lock |
| [NVIDIA NGC vLLM / Podman / WSL][ngc-podman-profile] | Same NGC image using WSL2 GPU passthrough via NVIDIA CDI |

The CPU and two original CUDA profiles target Linux x86-64 and share the Ubuntu 24.04 base, `/opt/venv` Python 3.12 virtual environment, cache volume layout, and host CA certificate bootstrap. The original CUDA profiles install PyTorch 2.10.0 CUDA 12.8 runtime wheels (`+cu128`).

## Start the Environment

Open the repository in VS Code and run **Dev Containers: Reopen in Container** for the default CPU configuration.

Or launch a specific profile via the Dev Containers CLI:

```sh
# Default CPU profile (Podman host)
devcontainer up --workspace-folder "$PWD" --docker-path podman \
  --buildkit never --gpu-availability none

# CUDA / Podman / WSL profile
devcontainer up --workspace-folder "$PWD" \
  --config "$PWD/.devcontainer/gpu-podman/devcontainer.json" \
  --docker-path podman --buildkit never --gpu-availability detect

# CUDA / Docker profile
devcontainer up --workspace-folder "$PWD" \
  --config "$PWD/.devcontainer/gpu/devcontainer.json" \
  --gpu-availability all
```

### Experimental NVIDIA NGC backend

The [Docker][ngc-profile] and [Podman/WSL][ngc-podman-profile] NGC profiles
pin the same Linux/amd64 manifest for NVIDIA
vLLM `26.05-py3` (bundled vLLM `0.20.1`, Python 3.12, CUDA 13.2.1).
This is a distinct backend candidate, not an upgrade to the original
profiles or a substitution into the frozen reproduction settings. It uses
the image as shipped; it does not install the workspace or change NVIDIA's
bundled packages. This route avoids the standalone Torch wheel whose mixed
HSTU notice has not been resolved. It does not resolve that wheel's licensing
or clear the separate MATH/IFEval data and grader requirements.
Use the lock-based profiles for ordinary package development and grading.
Do not install the project dependencies into the NGC environment: their
Torch and Transformers requirements differ from the vendor stack.

**Prepare the host.** Use Linux/amd64 with `jq`,
the NVIDIA Container Toolkit, and a driver compatible with CUDA 13.2.1.
Check that `nvidia-smi` sees the intended GPU. For Docker, verify access
to the daemon. For Podman, use the same WSL user and image store for all
commands, and check that `nvidia-ctk cdi list` shows `nvidia.com/gpu=all`.
Check available disk in the engine's storage: the image has 9.10 GB of compressed
layers and needs additional space to unpack. Review the [image catalog][ngc-vllm],
[software license][ngc-eula], and [AI product terms][ngc-terms] before acquiring
or using the image. If the exact image is not cached, pull it only once the
applicable terms and a bounded acquisition scope are in place. From the
repository root:

```sh
# Docker
image=$(jq -er '.image' .devcontainer/gpu-ngc/devcontainer.json) &&
  docker pull --platform=linux/amd64 "$image"

# Podman on the WSL host, not inside the current devcontainer
image=$(jq -er '.image' .devcontainer/gpu-ngc-podman/devcontainer.json) &&
  podman pull --platform=linux/amd64 "$image"
```

Use the digest from the profile, not the release tag or `latest`.

**Run the model-free smoke before opening the workspace.**

```sh
# Choose the engine for your profile:
bash .devcontainer/gpu-ngc/check.sh smoke docker
bash .devcontainer/gpu-ngc/check.sh smoke podman
```

The [check][ngc-check] refuses missing or wrong-platform images; it never
pulls. The smoke uses one GPU with Docker or the named CDI group with
Podman, has a 60-second command timeout, and has no
network or workspace/model mounts. Success prints `NGC model-free smoke passed`
with the vLLM and Torch versions and GPU name. If the check fails, capture the
error and resolve the image, driver, runtime, or device issue before retrying.
Do not swap to an unpinned tag, override a CUDA requirement, or proceed to a
model run. The isolated Podman smoke passed on the operator's WSL host
with an RTX A3000; the new profile's creation check remains separate.

A pass verifies the bundled Python/vLLM, vLLM's CUDA platform and native
extension load, GPU arithmetic, and the stop-token API. It does not
demonstrate that generation obeys the configured stop
conditions or that the full checkpoint, including auxiliary MTP tensors,
is preserved. Those checks precede any matched behavioral pilot.

**Open the development container only after the smoke passes.**

```sh
# Docker
devcontainer up --workspace-folder "$PWD" \
  --config "$PWD/.devcontainer/gpu-ngc/devcontainer.json" \
  --gpu-availability all

# Podman/WSL; CDI in the profile supplies the GPU
devcontainer up --workspace-folder "$PWD" \
  --config "$PWD/.devcontainer/gpu-ngc-podman/devcontainer.json" \
  --docker-path podman --buildkit never --gpu-availability none
```

The profile checks the local image cache before launch and waits for the
same model-free smoke to pass inside the newly created container. Use the
creation log for the result, or rerun `bash .devcontainer/gpu-ngc/check.sh
runtime` inside the container. A failed check is a setup failure, not a
reason to move the project session or start a model run.

The development container is not the isolated host smoke: it mounts the
workspace and runs as the image's default user. Only open it on a checkout
you intend to expose to that container. For WSL, choose the NGC Podman
configuration, not the original lock-based Podman profile.

[ngc-profile]: gpu-ngc/devcontainer.json
[ngc-podman-profile]: gpu-ngc-podman/devcontainer.json
[ngc-check]: gpu-ngc/check.sh
[ngc-vllm]: https://catalog.ngc.nvidia.com/orgs/nvidia/-/containers/vllm/26.05-py3
[ngc-eula]: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/
[ngc-terms]: https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/

## Certificate Trust

`prepare-build.sh` exports public host root certificates from macOS keychains, Windows root stores (via WSL/PowerShell), or Linux system bundles into `.devcontainer/host-ca-certificates.crt` (git-ignored). The `Dockerfile` validates each certificate with OpenSSL and updates the Ubuntu trust store before running `apt` or `pip`.

To supply a custom PEM bundle explicitly on the host:

```sh
export LLM_LOOP_FIX_CA_BUNDLE=/absolute/path/to/approved-ca-bundle.pem
```

## Dependency Locks

Dependencies are specified in [requirements.in](requirements.in) and locked with transitive SHA-256 hashes using `uv==0.8.22`.

| Lock | Target | Purpose |
| --- | --- | --- |
| [requirements.lock](requirements.lock) | Python 3.12, Linux x86-64, CPU | Default container and ordinary CI |
| [gpu/requirements.lock](gpu/requirements.lock) | Python 3.12, Linux x86-64, CUDA 12.8 | Docker/Podman CUDA containers |
| [requirements-py310.lock](requirements-py310.lock) | Python 3.10, Linux x86-64, CPU | Minimum-version compatibility CI; not a separate container profile |

The Python 3.12 profiles use Transformers 5.22.1, which requires Python 3.11+. The Python 3.10 lock resolves Transformers 5.20.2 and other compatible transitive dependencies. Locks are interpreter/backend-specific, not interchangeable.

When running uv under a bounded address-space envelope, set
`UV_CONCURRENT_INSTALLS=1` for that process to bound its installation/unzip
worker pool. This [documented uv setting][uv-install-concurrency] does not raise
resource limits or change package pins, integrity checks, or the selected
interpreter.

The lock-based profiles also pin the source math-grading stack: math-verify 0.9.0,
latex2sympy2-extended 1.11.0, and ANTLR runtime 4.13.2. Standalone installs can
request the project's `evaluation` extra. These dependencies do not implement
the missing private source harness or establish behavioral reproduction.
When checking a CPU environment from a CUDA container, run smoke with an
explicit `LLM_LOOP_FIX_DEVICE=cpu` override; the selected interpreter and the
inherited device setting are independent.

To refresh the lockfiles inside a running container:

```sh
uv pip compile .devcontainer/requirements.in --torch-backend cpu \
  --python-version 3.12 --python-platform x86_64-unknown-linux-gnu \
  --generate-hashes --output-file .devcontainer/requirements.lock

uv pip compile .devcontainer/requirements.in --torch-backend cu128 \
  --python-version 3.12 --python-platform x86_64-unknown-linux-gnu \
  --generate-hashes --output-file .devcontainer/gpu/requirements.lock

uv pip compile .devcontainer/requirements.in --torch-backend cpu \
  --python-version 3.10 --python-platform x86_64-unknown-linux-gnu \
  --generate-hashes --output-file .devcontainer/requirements-py310.lock
```

For a fresh Linux x86-64 CPU virtual environment with Python 3.12, install the profile before the editable project:

```sh
python -m pip install pip==25.2
python -m pip install --require-hashes --only-binary=:all: \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r .devcontainer/requirements.lock
python -m pip install --no-deps -e '.[dev]'
python -m pip check
```

For Python 3.10, use [requirements-py310.lock](requirements-py310.lock) instead. CUDA containers use the CUDA lock and index from their build arguments. The editable project's build dependencies and the container's OS packages are not frozen by these runtime locks.

## Smoke Validation

[smoke.py](smoke.py) runs automatically as `postCreateCommand` without downloading pretrained model weights. It verifies:
- System CA trust store and TLS environment variables (`SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE`, `NODE_EXTRA_CA_CERTS`).
- CPU or CUDA device availability (`LLM_LOOP_FIX_DEVICE`).
- End-to-end synthetic `Qwen2ForCausalLM` contrastive probing, culprit matrix localization, surgical sparse-column patching, weight restoration, and `Qwen3_5Config` availability.

The full `pytest` suite runs separately with `--tests`, so application test failures do not block opening a working development environment. Test failures still return a nonzero exit code.

Run unit tests, live network verification, and lint checks inside the container with:

```sh
python .devcontainer/smoke.py --tests
python .devcontainer/smoke.py --network
shellcheck .devcontainer/prepare-build.sh
ruff check src tests .devcontainer/smoke.py
```

The repository-owned Ruff policy selects correctness rules `E4`, `E7`, `E9`, and `F` with a Python 3.10 target. ShellCheck is an OS tool, not part of the pip dev extra. For offline CPU smoke outside a container, set `LLM_LOOP_FIX_DEVICE=cpu`, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`, plus the four CA environment variables listed above to `/etc/ssl/certs/ca-certificates.crt`.

[Ordinary CI](../.github/workflows/ci.yml) uses the two CPU locks on Python 3.10/3.12 with pinned checkout/setup actions, read-only repository permissions, and a 15-minute job limit. It checks pytest, Ruff, shellcheck, and offline tiny-model wiring. Local checks, hosted CPU CI, and fresh CPU/CUDA dependency-image checks pass. See the baseline report in [PLAN.md](../docs/PLAN.md). None of these checks replays the published edit or evaluates real-model repair.

[Image CI](../.github/workflows/containers.yml) builds the shared Dockerfile with CPU/CUDA locks after relevant input changes or manual dispatch. Its jobs are serial, capped at 25 minutes each, and gated on runner headroom. The mounted source is read-only during network-disabled CPU smoke/tests; CUDA-image checks also verify the compiled CUDA version, not GPU availability. It publishes no images. Both jobs passed in the first hosted attempt; resource/attempt limits and remaining validation distinctions are in [PLAN.md](../docs/PLAN.md).

[uv-install-concurrency]: https://docs.astral.sh/uv/reference/environment/#uv_concurrent_installs

## Commit Messages

All contributors, including coding agents, must use Conventional Commits for every commit in this repository:

```text
<type>[optional scope]: <short imperative summary>
```

Use `fix` for bug fixes, `feat` for new functionality, `chore` for maintenance, and `test` for test changes. Other supported types are `docs`, `refactor`, `perf`, `build`, `ci`, and `revert`. Scopes are optional.

```text
fix: restore CUDA Podman devcontainer startup
feat(detector): support additional loop patterns
chore: refresh development dependencies
test: cover clean generation trajectories
```

Separate an optional explanatory body from the subject with a blank line. Mark breaking changes with `!` before the colon or a `BREAKING CHANGE:` footer.
