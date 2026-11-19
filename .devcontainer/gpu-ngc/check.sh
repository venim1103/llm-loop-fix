#!/usr/bin/env bash
set -euo pipefail

directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
case "${1:-}" in
    runtime)
        export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
        exec /usr/bin/timeout --kill-after=5s 60s /usr/bin/python3 -IB "$directory/smoke.py"
        ;;
    cached|smoke) mode="$1" ;;
    *) printf 'Usage: %s cached|smoke|runtime\n' "$0" >&2; exit 2 ;;
esac

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
    exec docker run --rm --interactive --pull=never --platform=linux/amd64 --network=none \
        --memory=4g --cpus=2 --gpus 1 --cap-drop=ALL --security-opt=no-new-privileges \
        -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
        --entrypoint /usr/bin/timeout "$image" --kill-after=5s 60s /usr/bin/python3 -IB - \
        < "$directory/smoke.py"
fi
