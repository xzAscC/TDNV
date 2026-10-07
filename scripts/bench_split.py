"""Time one forward pass of a model split over the visible GPUs under a few settings.

    uv run python scripts/bench_split.py --model Qwen/Qwen3-32B
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def timed(model, enc, **kw) -> float:
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        model(**enc, **kw)
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)
    return time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--attn", default="sdpa")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    max_memory = {i: int(torch.cuda.get_device_properties(i).total_memory * 0.94)
                  for i in range(torch.cuda.device_count())}
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", max_memory=max_memory,
        attn_implementation=args.attn,
    ).eval()
    print("devices:", sorted({str(d) for d in model.hf_device_map.values()}))
    first = next(model.parameters()).device

    text = "The city of Krasnodar is in Russia. " * 20
    for bs in (1, 8):
        enc = tok([text] * bs, return_tensors="pt", padding=True).to(first)
        timed(model, enc)  # warm-up
        print(f"attn={args.attn} bs={bs} len={enc['input_ids'].shape[1]}: "
              f"plain {timed(model, enc):.2f}s | hidden {timed(model, enc, output_hidden_states=True):.2f}s"
              f" | hidden+keep1 {timed(model, enc, output_hidden_states=True, logits_to_keep=1):.2f}s")


if __name__ == "__main__":
    main()
