# Development Container (`llm-loop-fix`)

This setup adapts the multi-profile Dev Container and certificate-trust architecture from `micro_classifier` and `embedded-jev`. It provides a reproducible, hash-locked Python 3.12 environment for diagnosing and surgically patching looping small language models on CPU or CUDA.

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

Dependencies are specified in [requirements.in](requirements.in) and locked with transitive SHA-256 hashes in [requirements.lock](requirements.lock) (CPU) and [gpu/requirements.lock](gpu/requirements.lock) (CUDA 12.8).

To refresh the lockfiles inside a running container:

```sh
uv pip compile .devcontainer/requirements.in --torch-backend cpu \
  --python-version 3.12 --python-platform x86_64-unknown-linux-gnu \
  --generate-hashes --output-file .devcontainer/requirements.lock

uv pip compile .devcontainer/requirements.in --torch-backend cu128 \
  --python-version 3.12 --python-platform x86_64-unknown-linux-gnu \
  --generate-hashes --output-file .devcontainer/gpu/requirements.lock
```

## Smoke Validation

[smoke.py](smoke.py) runs automatically as `postCreateCommand` without downloading pretrained model weights. It verifies:
- System CA trust store and TLS environment variables (`SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE`, `NODE_EXTRA_CA_CERTS`).
- CPU or CUDA device availability (`LLM_LOOP_FIX_DEVICE`).
- End-to-end synthetic `Qwen2ForCausalLM` contrastive probing, culprit matrix localization, surgical sparse-column patching, weight restoration, and `Qwen3_5Config` availability.
- Full `pytest` unit test suite execution.

Run live network verification and lint checks inside the container with:

```sh
python .devcontainer/smoke.py --network
shellcheck .devcontainer/prepare-build.sh
ruff check .
```
