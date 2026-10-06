"""Contrastive (positive / negative) datasets as chat conversations with binary labels."""

from __future__ import annotations

import random

from datasets import load_dataset

HARMFUL = "LLM-LAT/harmful-dataset"
BENIGN = "LLM-LAT/benign-dataset"


def _sample_prompts(name: str, n: int, seed: int, max_chars: int) -> list[str]:
    ds = load_dataset(name, split="train")
    prompts = sorted({p.strip() for p in ds["prompt"] if p and 0 < len(p.strip()) <= max_chars})
    random.Random(seed).shuffle(prompts)
    if len(prompts) < n:
        raise ValueError(f"{name}: only {len(prompts)} prompts after filtering, need {n}")
    return prompts[:n]


def load_safety(mode: str, n: int, seed: int = 0, max_chars: int = 1500):
    """Return (conversations, labels).

    mode="prompt":   harmful prompt (label 1) vs benign prompt (label 0); user turn only.
    mode="response": same harmful prompt answered with compliance ("rejected", label 1)
                     vs refusal ("chosen", label 0); user + assistant turns.
    """
    if mode == "prompt":
        harmful = _sample_prompts(HARMFUL, n, seed, max_chars)
        benign = _sample_prompts(BENIGN, n, seed, max_chars)
        convs = [[{"role": "user", "content": p}] for p in harmful + benign]
        labels = [1] * n + [0] * n
    elif mode == "response":
        ds = load_dataset(HARMFUL, split="train").shuffle(seed=seed)
        ds = ds.filter(lambda r: r["prompt"] and r["chosen"] and r["rejected"])
        ds = ds.select(range(min(n, len(ds))))
        convs, labels = [], []
        for key, lab in (("rejected", 1), ("chosen", 0)):
            for r in ds:
                convs.append([{"role": "user", "content": r["prompt"]},
                              {"role": "assistant", "content": r[key]}])
                labels.append(lab)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return convs, labels
