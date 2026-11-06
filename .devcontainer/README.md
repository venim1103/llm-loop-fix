# Development Container (`llm-loop-fix`)

This setup adapts the multi-profile Dev Container and certificate-trust architecture from `micro_classifier` and `embedded-jev`. It provides hash-locked Python 3.12 CPU/CUDA inputs and offline numerical/wiring checks for the research prototype. Dependency locks do not pin the base image or apt contents; fresh image builds and real-model repair are separate validation gates.

## Profiles

| Configuration | Use |
| --- | --- |
| [CPU](devcontainer.json) | Default; weight-space probing, unit tests, and small model patching without a GPU requirement |
| [CUDA / Docker](gpu/devcontainer.json) | NVIDIA host with Docker and NVIDIA Container Toolkit |
| [CUDA / Podman / WSL](gpu-podman/devcontainer.json) | WSL2 passthrough with Podman and NVIDIA CDI (`--device nvidia.com/gpu=all`, `/dev/dxg`) |

All profiles target Linux x86-64 and share the Ubuntu 24.04 base, `/opt/venv` Python 3.12 virtual environment, cache volume layout, and host CA certificate bootstrap. The CUDA profiles install PyTorch 2.10.0 CUDA 12.8 runtime wheels (`+cu128`).

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

[Ordinary CI](../.github/workflows/ci.yml) uses the two CPU locks on Python 3.10/3.12 with pinned checkout/setup actions, read-only repository permissions, and a 15-minute job limit. It checks pytest, Ruff, shellcheck, and offline tiny-model wiring. Clean CPU installs and the existing CUDA environment passed locally; hosted CI and fresh image builds have not yet run. See the baseline report in [PLAN.md](../docs/PLAN.md). None of these checks replays the published edit or evaluates real-model repair.

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
