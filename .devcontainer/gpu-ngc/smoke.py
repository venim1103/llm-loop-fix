"""Model-free readiness check for the pinned NVIDIA vLLM image."""

from importlib import import_module
from importlib.metadata import version
import sys


def main() -> None:
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("Expected the bundled Python 3.12")

    import torch
    import vllm
    from vllm import SamplingParams
    from vllm.platforms import current_platform

    installed = version("vllm")
    if installed.partition("+")[0] != "0.20.1" or installed != vllm.__version__:
        raise SystemExit(f"Expected bundled vLLM 0.20.1; found {installed} / {vllm.__version__}")
    if torch.version.cuda != "13.2" or not torch.cuda.is_available():
        raise SystemExit("Expected bundled CUDA 13.2 and a visible GPU")
    if not current_platform.is_cuda():
        raise SystemExit("vLLM did not select its CUDA platform")
    import_module("vllm._C")

    if SamplingParams(max_tokens=1, stop_token_ids=[1]).stop_token_ids != [1]:
        raise SystemExit("vLLM stop-token API did not retain the declared token")
    if torch.tensor([1, 2, 3], device="cuda").sum().item() != 6:
        raise SystemExit("Unexpected result from the CUDA tensor smoke")
    print("NGC model-free smoke passed:", installed, torch.__version__, torch.cuda.get_device_name(0))


if __name__ == "__main__":
    main()
