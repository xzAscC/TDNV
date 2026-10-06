"""Layerwise TDNV of last-token states for every steering concept, one model per call.

Example:
    uv run python scripts/run_concepts.py --model Qwen/Qwen3-8B --concepts all
Results: <out>/<model>/<concept>/metrics.json (+ tdnv.png). Finished concepts are skipped.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tdnv.concepts import CONCEPTS, load_concept
from tdnv.extract import last_token_states, render
from tdnv.metrics import layerwise_report
from tdnv.plot import plot_report


def load_model(name: str):
    # Keep every layer on a GPU: accelerate's default budget can push a few layers of a 27-32B
    # model to CPU/disk on 2x40GB cards, which made each batch ~30x slower.
    max_memory = {i: int(torch.cuda.get_device_properties(i).total_memory * 0.94)
                  for i in range(torch.cuda.device_count())}
    kw = dict(torch_dtype=torch.bfloat16, device_map="auto", max_memory=max_memory or None)
    if "gemma-2" in name.lower():
        kw["attn_implementation"] = "eager"  # sdpa drops Gemma-2's attention soft-capping
    try:
        model = AutoModelForCausalLM.from_pretrained(name, **kw)
    except ValueError:  # e.g. multimodal Gemma-3 checkpoints
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(name, **kw)
    placement = {}
    for dev in getattr(model, "hf_device_map", {}).values():
        placement[str(dev)] = placement.get(str(dev), 0) + 1
    print("device map (modules per device):", placement)
    if any(d in ("cpu", "disk") for d in placement):
        raise RuntimeError("model does not fit on the GPUs; request more GPUs instead of offloading")
    return model.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--concepts", default="all", help="comma list or 'all'")
    ap.add_argument("--n", type=int, default=1000, help="max samples per class")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--max-response-tokens", type=int, default=32)
    ap.add_argument("--n-shuffles", type=int, default=3)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    names = list(CONCEPTS) if args.concepts == "all" else args.concepts.split(",")
    model_dir = Path(args.out) / args.model.replace("/", "__")
    todo = [c for c in names if not (model_dir / c / "metrics.json").exists()]
    print(f"{args.model}: {len(todo)} concepts to run, {len(names) - len(todo)} done")
    if not todo:
        return

    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for concept in todo:
        t0 = time.time()
        examples = load_concept(concept, args.n, args.seed)
        texts = [render(tok, e, args.max_response_tokens) for e in examples]
        labels = torch.tensor([e.label for e in examples])
        print(f"\n=== {concept}: {len(texts)} texts | pos={int(labels.sum())}")
        print("  last chars (label 1):", repr(texts[0][-80:]))
        hidden = last_token_states(model, tok, texts, args.batch_size, args.max_length)
        report = layerwise_report(hidden, labels, args.n_shuffles, args.seed, device)
        report["config"] = {**vars(args), "concept": concept, "n_examples": len(texts)}
        out_dir = model_dir / concept
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(report, indent=1))
        plot_report(report, out_dir / "tdnv.png", title=f"{args.model} | {concept}")
        t = report["tdnv"]
        best = min(range(1, len(t)), key=lambda l: t[l])
        print(f"  TDNV L1={t[1]:.2f} min={t[best]:.2f}@L{best} last={t[-1]:.2f} "
              f"| probe@min={report['probe_acc'][best]:.3f} | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
