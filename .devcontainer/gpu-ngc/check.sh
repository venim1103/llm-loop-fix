#!/usr/bin/env bash
set -euo pipefail

directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
case "${1:-}" in
    runtime)
        export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
        exec /usr/bin/timeout --kill-after=5s 60s /usr/bin/python3 -IB "$directory/smoke.py"
        ;;
    cached|smoke) mode="$1" ;;
    *) printf 'Usage: %s cached|smoke [docker|podman], or runtime\n' "$0" >&2; exit 2 ;;
esac

engine=${2:-docker}
case "$engine" in
    docker) profile="$directory/devcontainer.json"; gpu=(--gpus 1) ;;
    podman) profile="$directory/../gpu-ngc-podman/devcontainer.json"; gpu=(--device nvidia.com/gpu=all) ;;
    *) printf 'Unsupported container engine: %s\n' "$engine" >&2; exit 2 ;;
esac
container_cli() {
    case "$engine" in
        docker) docker "$@" ;;
        podman) podman "$@" ;;
    esac
}

for tool in jq "$engine"; do
    if ! command -v "$tool" >/dev/null; then
        printf 'The NGC profile requires %s on the %s host.\n' "$tool" "$engine" >&2
        exit 1
    fi
done

if ! image=$(jq -er '.image | select(test("^nvcr\\.io/nvidia/vllm@sha256:[0-9a-f]{64}$"))' "$profile"); then
    printf 'Expected a digest-pinned NVIDIA vLLM image in %s.\n' "$profile" >&2
    exit 1
fi
if ! platform=$(container_cli image inspect --format '{{.Os}}/{{.Architecture}}' "$image"); then
    printf 'Pinned NGC image not available through %s. This check never pulls images.\n' "$engine" >&2
    printf 'Review the container guide and applicable NVIDIA terms before obtaining %s.\n' "$image" >&2
    exit 1
fi
if [[ "$platform" != linux/amd64 ]]; then
    printf 'Expected linux/amd64 for %s; Docker reported %s.\n' "$image" "$platform" >&2
    exit 1
fi

if [[ "$mode" == smoke ]]; then
    container_cli run --rm --interactive --pull=never --platform=linux/amd64 --network=none \
        --memory=4g --cpus=2 "${gpu[@]}" --cap-drop=ALL --security-opt=no-new-privileges \
        -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
        --entrypoint /usr/bin/timeout "$image" --kill-after=5s 60s /usr/bin/python3 -IB - \
        < "$directory/smoke.py"
fi
