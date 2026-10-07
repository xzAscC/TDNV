"""Layerwise TDNV and early-exit accuracy of ICL instances on the steering concepts, one model per call.

The paper's ICL setting (Sec. 3) on real data: for each concept, task 1 labels the inputs by the
concept and task 0 the opposite way (see tdnv.icl). For each K, every query is run under both
tasks with the same K demonstration inputs, and the representation h^(l) is the last token of
the query (the separator before the answer). Per layer l:
  tdnv            TDNV between the two tasks (classes = tasks), as in Sec. 3.2
  early_exit_t1/0 early-exit accuracy (App. G): argmax over the vocabulary of the final norm and
                  unembedding applied to h^(l), counted correct when it is the first token of
                  the task's answer for the query; index L (after the final norm) is the ICL
                  accuracy
Per concept, zero_shot_t1/0 is the zero-shot baseline: the same prediction with K = 0 (the two
tasks then see the same prompt and differ only in the target).

Example:
    uv run python scripts/run_icl.py --model Qwen/Qwen3-4B --concepts truth_cities --n-query 20
Results: <out>/<model>/<concept>/k<K>/metrics.json. Finished (concept, K) are skipped.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from run_concepts import load_model
from tdnv.extract import _decoder_stack, last_token_states
from tdnv.icl import CONCEPTS, answer_text, load_items, render, sample
from tdnv.metrics import layerwise_report


def target_ids(tok, concept: str, answers: list[str]) -> torch.Tensor:
    """First token of each answer as it follows the query."""
    return torch.tensor([tok(answer_text(concept, a), add_special_tokens=False)["input_ids"][0]
                         for a in answers])


@torch.no_grad()
def early_exit_preds(model, hidden: torch.Tensor, batch: int = 64) -> torch.Tensor:
    """[N, L+1] argmax token of unembed(norm(h^(l))); index L is already normed."""
    _, norm = _decoder_stack(model)
    head = model.get_output_embeddings()
    norm_dev = next(norm.parameters()).device
    n, n_idx, _ = hidden.shape
    out = torch.empty(n, n_idx, dtype=torch.long)
    for l in range(n_idx):
        for i in range(0, n, batch):
            h = hidden[i: i + batch, l]
            if l < n_idx - 1:
                h = norm(h.to(norm_dev, head.weight.dtype))
            h = h.to(head.weight.device, head.weight.dtype)
            out[i: i + batch, l] = head(h).float().argmax(-1).cpu()
    return out


def run_concept(model, tok, concept: str, args, model_dir: Path) -> None:
    ks = [int(k) for k in args.ks.split(",")]
    todo = [k for k in ks if not (model_dir / concept / f"k{k}" / "metrics.json").exists()]
    if not todo:
        return
    t0 = time.time()
    items = load_items(concept)
    insts = sample(concept, items, args.n_query, max(ks), args.seed)
    bos = tok.bos_token or ""
    tgt = {t: target_ids(tok, concept, [i.query.y(t) for i in insts]) for t in (1, 0)}
    assert (tgt[1] != tgt[0]).all(), "the two answers share their first token"

    # zero-shot baseline: K = 0, same prompt for both tasks
    zs = [render(concept, i, 0, 1, bos) for i in insts]
    h0 = last_token_states(model, tok, zs, args.batch_size, args.max_length)
    pred0 = early_exit_preds(model, h0[:, -1:])[:, 0]
    zero_shot = {f"zero_shot_t{t}": float((pred0 == tgt[t]).double().mean()) for t in (1, 0)}
    print(f"\n=== {concept}: {len(insts)} queries | zero-shot {zero_shot}")

    for k in todo:
        texts = [render(concept, i, k, t, bos) for t in (1, 0) for i in insts]
        labels = torch.tensor([t for t in (1, 0) for _ in insts])
        n_tok = [len(tok(s, add_special_tokens=False)["input_ids"]) for s in texts]
        hidden = last_token_states(model, tok, texts, args.batch_size, args.max_length)
        if not torch.isfinite(hidden[:, 1:]).all():
            raise RuntimeError(f"{concept} K={k}: non-finite hidden states; not writing metrics")
        report = layerwise_report(hidden, labels, args.n_shuffles, args.seed, hidden.device)
        pred = early_exit_preds(model, hidden)
        target = torch.cat([tgt[1], tgt[0]])
        hit = pred == target[:, None]
        for t in (1, 0):
            report[f"early_exit_t{t}"] = hit[labels == t].double().mean(0).tolist()
        cand = (pred[:, -1] == torch.cat([tgt[1], tgt[0]])) | (pred[:, -1] == torch.cat([tgt[0], tgt[1]]))
        report |= zero_shot
        report["answer_format_rate"] = float(cand.double().mean())  # final argmax is one of the two answers
        report["n_tokens"] = dict(mean=sum(n_tok) / len(n_tok), max=max(n_tok),
                                  n_truncated=sum(n > args.max_length for n in n_tok))
        report["example_prompts"] = {f"t{t}": texts[0 if t == 1 else len(insts)][-600:] for t in (1, 0)}
        report["config"] = {**vars(args), "concept": concept, "k": k, "n_query": len(insts)}
        out = model_dir / concept / f"k{k}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(report, indent=1))
        tv = report["tdnv"]
        best = min(range(1, len(tv)), key=lambda l: tv[l])
        print(f"  K={k:2d} | tokens mean {report['n_tokens']['mean']:.0f} max {report['n_tokens']['max']} "
              f"| TDNV L1={tv[1]:.2f} min={tv[best]:.3g}@L{best} last={tv[-1]:.2f} "
              f"| ICL acc t1={report['early_exit_t1'][-1]:.2f} t0={report['early_exit_t0'][-1]:.2f} "
              f"| format {report['answer_format_rate']:.2f} | {time.time() - t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--concepts", default="all", help="comma list or 'all' (12 concepts)")
    ap.add_argument("--ks", default="1,5,10,15", help="numbers of demonstrations")
    ap.add_argument("--n-query", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--n-shuffles", type=int, default=3)
    ap.add_argument("--out", default="outputs/icl")
    args = ap.parse_args()

    names = CONCEPTS if args.concepts == "all" else args.concepts.split(",")
    model_dir = Path(args.out) / args.model.replace("/", "__")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model)
    for concept in names:
        run_concept(model, tok, concept, args, model_dir)


if __name__ == "__main__":
    main()
