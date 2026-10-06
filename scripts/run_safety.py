"""Layerwise TDNV of last-token states: harmful vs benign (LLM-LAT safety data).

Example:
    uv run python scripts/run_safety.py --model meta-llama/Llama-3.1-8B-Instruct --mode prompt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tdnv.data import load_safety
from tdnv.extract import last_token_states, render
from tdnv.metrics import layerwise_report
from tdnv.plot import plot_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--mode", choices=["prompt", "response"], default="prompt")
    ap.add_argument("--n", type=int, default=1000, help="samples per class")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--max-response-tokens", type=int, default=32)
    ap.add_argument("--no-chat-template", action="store_true")
    ap.add_argument("--n-shuffles", type=int, default=5)
    ap.add_argument("--save-hidden", action="store_true")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    tag = "raw" if args.no_chat_template else "chat"
    out_dir = Path(args.out) / args.model.replace("/", "__") / f"{args.mode}_{tag}_n{args.n}_s{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    convs, labels = load_safety(args.mode, args.n, args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    texts = [render(tok, c, not args.no_chat_template, args.max_response_tokens) for c in convs]
    print("example (label=1):", repr(texts[0][-300:]))
    print("example (label=0):", repr(texts[-1][-300:]))

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    t0 = time.time()
    hidden = last_token_states(model, tok, texts, args.batch_size, args.max_length)
    print(f"hidden {tuple(hidden.shape)} in {time.time() - t0:.0f}s")
    labels_t = torch.tensor(labels)
    if args.save_hidden:
        torch.save({"hidden": hidden.half(), "labels": labels_t}, out_dir / "hidden.pt")

    del model
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = layerwise_report(hidden, labels_t, args.n_shuffles, args.seed, device)
    report["config"] = vars(args)
    report["label_meaning"] = (
        {"1": "harmful prompt", "0": "benign prompt"} if args.mode == "prompt"
        else {"1": "compliant response (rejected)", "0": "refusal (chosen)"}
    )
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=1))
    plot_report(report, out_dir / "tdnv.png", title=f"{args.model} | {args.mode} | {tag}")

    print(f"{'layer':>5} {'TDNV':>10} {'shuffled':>10} {'probe':>6}")
    for l, t, s, p in zip(report["layer"], report["tdnv"], report["tdnv_shuffled"], report["probe_acc"]):
        print(f"{l:>5} {t:>10.4f} {s:>10.2f} {p:>6.3f}")
    print("saved to", out_dir)


if __name__ == "__main__":
    main()
