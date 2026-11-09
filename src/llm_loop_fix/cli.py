"""Command-line interface for diagnosing and patching looping language models."""

import argparse
import json
from pathlib import Path
from typing import List
import torch

from .localize import localize_loop_culprits
from .probe import collect_contrastive_activations


def load_prompts(path: str) -> List[str]:
    """Load prompts from a JSONL or plain-text file."""
    prompts: List[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            obj = json.loads(line)
            prompts.append(obj.get("prompt") or obj.get("text") or "")
        else:
            prompts.append(line)
    return [p for p in prompts if p]


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="llm-loop-fix",
        description="Surgical weight-space de-looping for small language models.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for cmd in ("diagnose", "run"):
        sub = subparsers.add_parser(cmd)
        sub.add_argument("--model", required=True, help="Hugging Face model ID or path")
        sub.add_argument("--prompts", required=True, help="Path to JSONL or text prompts")
        sub.add_argument("--max-new-tokens", type=int, default=192)
        sub.add_argument("--top-k-layers", type=int, default=2)
        sub.add_argument(
            "--mode",
            choices=["top_k_columns", "sparse_columns", "rank1"],
            default="top_k_columns",
        )
        sub.add_argument("--top-k-columns", type=int, default=64)
        sub.add_argument("--sparse-fraction", type=float, default=0.02)
        sub.add_argument("--report-out", default="diagnostic_report.json")
        if cmd == "run":
            sub.add_argument("--output-dir", required=True, help="Reserved for validated export (currently disabled)")

    args = parser.parse_args()
    if args.command == "run":
        parser.error("run is disabled: utility confirmation and validated export are not implemented; use diagnose for experimental diagnostics")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    prompts = load_prompts(args.prompts)

    def generate_sequences(m):
        full_seqs = []
        gen_only = []
        p_lens = []
        for p in prompts:
            enc = tokenizer(p, return_tensors="pt").to(device)
            p_len = enc.input_ids.shape[1]
            out = m.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            seq = out[0].detach().cpu()
            full_seqs.append(seq)
            p_lens.append(p_len)
            gen_only.append(seq[p_len:].tolist())
        return full_seqs, p_lens, gen_only

    full_seqs, p_lens, _ = generate_sequences(model)
    acts, analyses = collect_contrastive_activations(
        model, full_seqs, p_lens, min_ngram=3, min_repeats=3
    )
    report = localize_loop_culprits(
        model, acts, top_columns_per_matrix=max(128, args.top_k_columns)
    )

    summary = report.to_summary_dict(top_k=10)
    summary["num_prompts"] = len(prompts)
    summary["num_looping_baseline"] = sum(1 for a in analyses if a.is_looping)
    Path(args.report_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
