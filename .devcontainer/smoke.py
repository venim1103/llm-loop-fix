"""Container environment and weight-space de-looping pipeline smoke check."""

import argparse
import json
import os
import ssl
import subprocess
import sys
import urllib.request
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen3_5Config


def check_certificates() -> None:
    bundle = Path("/etc/ssl/certs/ca-certificates.crt")
    for variable in (
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "NODE_EXTRA_CA_CERTS",
    ):
        if os.environ.get(variable) != str(bundle):
            raise RuntimeError(f"{variable} must use the system CA bundle")
    if not bundle.is_file() or ssl.create_default_context().cert_store_stats()["x509_ca"] == 0:
        raise RuntimeError("The system CA trust store is empty or missing")


def check_weight_space_pipeline() -> dict:
    repo_root = Path(__file__).resolve().parent.parent
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from llm_loop_fix.detector import detect_loop_onset
    from llm_loop_fix.localize import localize_loop_culprits
    from llm_loop_fix.patch import PatchConfig, apply_surgical_patch, restore_weights
    from llm_loop_fix.probe import collect_contrastive_activations

    torch.manual_seed(42)
    torch.set_num_threads(2)
    device = os.environ.get("LLM_LOOP_FIX_DEVICE", "cpu")
    if device not in {"cpu", "cuda"}:
        raise RuntimeError("LLM_LOOP_FIX_DEVICE must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The CUDA profile requires working NVIDIA GPU passthrough")

    config = Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).to(device).eval()

    loop_seq = torch.tensor([1, 2, 7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9], dtype=torch.long)
    clean_seq = torch.tensor([1, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16], dtype=torch.long)

    acts, analyses = collect_contrastive_activations(
        model, [loop_seq, clean_seq], prompt_lengths=[2, 2], min_ngram=3, min_repeats=3
    )
    if not analyses[0].is_looping or analyses[1].is_looping:
        raise RuntimeError("Loop detector smoke test failed on synthetic trajectories")

    report = localize_loop_culprits(model, acts)
    if not report.ranked_culprits:
        raise RuntimeError("Localization report did not rank culprit matrices")

    backup = {}
    patch_res = apply_surgical_patch(
        model,
        report,
        PatchConfig(alpha=0.5, top_k_matrices=1, mode="sparse_columns", sparse_column_fraction=0.1),
        backup_store=backup,
    )
    restore_weights(model, backup)

    if Qwen3_5Config().model_type != "qwen3_5":
        raise RuntimeError("Qwen3.5 configuration is unavailable")

    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(repo_root / "tests")],
        check=True,
        env={**os.environ, "PYTHONPATH": src_path},
    )

    return {
        "device": device,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "top_culprit_probe": report.ranked_culprits[0].module_name,
        "patched_fraction": round(patch_res.modified_fraction, 6),
    }


def check_network() -> None:
    import requests

    for address in (
        "https://pypi.org/simple/pip/",
        "https://huggingface.co/api/models/Qwen/Qwen2.5-0.5B-Instruct",
    ):
        with urllib.request.urlopen(address, timeout=30) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTPS check failed: {address}")
        response = requests.get(address, timeout=30)
        response.raise_for_status()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", action="store_true")
    options = parser.parse_args()
    check_certificates()
    results = check_weight_space_pipeline()
    if options.network:
        check_network()
    results["versions"] = {
        package: version(package)
        for package in ("torch", "transformers", "safetensors", "accelerate", "numpy", "scipy")
    }
    print(json.dumps(results, indent=2))
    print("Container smoke checks and unit tests passed; no pretrained model was downloaded.")


if __name__ == "__main__":
    main()
