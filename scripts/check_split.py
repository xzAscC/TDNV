"""Compare last-token states of one model on a single GPU vs. split over two GPUs.

    uv run python scripts/check_split.py --model Qwen/Qwen3-8B      # needs 2 GPUs

Runs: single GPU, split with plain peer copies, split with copies staged through the CPU.
"""

from __future__ import annotations

import argparse
import gc
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tdnv.extract import last_token_states
from tdnv.p2p import stage_cross_gpu_copies


def run(name, tok, texts, device_map, max_memory=None):
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, device_map=device_map, max_memory=max_memory).eval()
    placement = sorted({str(d) for d in getattr(model, "hf_device_map", {"": "0"}).values()})
    t0 = time.time()
    h = last_token_states(model, tok, texts, batch_size=16, max_length=256)
    dt = time.time() - t0
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return h, dt, placement


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    args = ap.parse_args()
    print("peer access 0->1:", torch.cuda.can_device_access_peer(0, 1))
    x = torch.randn(1 << 20, device="cuda:0")
    torch.cuda.synchronize()
    t0 = time.time()
    y = x.to("cuda:1")
    torch.cuda.synchronize(1)
    print(f"direct copy 4MB: {time.time() - t0:.3f}s, equal={torch.equal(x.cpu(), y.cpu())}")

    tok = AutoTokenizer.from_pretrained(args.model)
    texts = [f"The city of number {i} is in country {i % 7}." * (1 + i % 5) for i in range(64)]
    half = {0: "9GiB", 1: "30GiB"}  # forces a split even for a model that fits one GPU
    ref, t_ref, p = run(args.model, tok, texts, "cuda:0")
    print(f"single GPU {p}: {t_ref:.1f}s nan={ref.isnan().sum().item()}")
    for label in ("split, direct copies", "split, staged via CPU"):
        if "staged" in label:
            stage_cross_gpu_copies()
        h, dt, p = run(args.model, tok, texts, "auto", half)
        rel = ((h - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-6))
        print(f"{label} {p}: {dt:.1f}s nan={h.isnan().sum().item()} "
              f"max rel diff vs single={torch.nan_to_num(rel, nan=float('inf')).max().item():.2e}")


if __name__ == "__main__":
    main()
