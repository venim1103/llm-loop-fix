#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
    cached|smoke) mode="$1" ;;
    *) printf 'Usage: %s cached|smoke\n' "$0" >&2; exit 2 ;;
esac

directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
for tool in jq docker; do
    if ! command -v "$tool" >/dev/null; then
        printf 'The NGC profile requires %s on the Docker host.\n' "$tool" >&2
        exit 1
    fi
done

image=$(jq -er '.image | select(test("^nvcr\\.io/nvidia/vllm@sha256:[0-9a-f]{64}$"))' \
    "$directory/devcontainer.json")
if ! platform=$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "$image"); then
    printf 'Pinned NGC image not available through Docker. This check never pulls images.\n' >&2
    printf 'Review the container guide and applicable NVIDIA terms before obtaining %s.\n' "$image" >&2
    exit 1
fi
if [[ "$platform" != linux/amd64 ]]; then
    printf 'Expected linux/amd64 for %s; Docker reported %s.\n' "$image" "$platform" >&2
    exit 1
fi

if [[ "$mode" == smoke ]]; then
    docker run --rm --pull=never --platform=linux/amd64 --network=none \
        --memory=4g --cpus=2 --gpus 1 --cap-drop=ALL --security-opt=no-new-privileges \
        -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
        --entrypoint /usr/bin/timeout "$image" --kill-after=5s 60s /usr/bin/python3 -c '
from importlib.metadata import version
import sys

import torch
import vllm
from vllm import SamplingParams

if sys.version_info[:2] != (3, 12):
    raise SystemExit("Expected the bundled Python 3.12")
if (version("vllm"), vllm.__version__) != ("0.20.1", "0.20.1"):
    raise SystemExit("Expected bundled vLLM 0.20.1")
if torch.version.cuda != "13.2" or not torch.cuda.is_available():
    raise SystemExit("Expected a visible CUDA 13.2 device")
if SamplingParams(max_tokens=1, stop_token_ids=[1]).stop_token_ids != [1]:
    raise SystemExit("vLLM stop-token API did not retain the declared token")
if torch.tensor([1, 2, 3], device="cuda").sum().item() != 6:
    raise SystemExit("Unexpected result from the CUDA tensor smoke")
print("NGC model-free smoke passed:", version("vllm"), torch.__version__, torch.cuda.get_device_name(0))
'
fi
