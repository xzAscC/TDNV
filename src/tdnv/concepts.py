"""Registry of contrastive steering concepts.

Each loader returns a list of Example objects with binary labels (1 = positive / steered-toward
behavior, 0 = negative). An example is either a chat conversation or a raw string.
"""

from __future__ import annotations

import csv
import io
import json
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from tdnv.data import load_safety

CACHE = Path(__file__).resolve().parents[2] / "data"
CAA_URL = "https://raw.githubusercontent.com/nrimsky/CAA/main/datasets/generate/{}/generate_dataset.json"
GOT_URL = "https://raw.githubusercontent.com/saprmarks/geometry-of-truth/main/datasets/{}.csv"

CAA_BEHAVIORS = [
    "coordinate-other-ais", "corrigible-neutral-HHH", "hallucination", "myopic-reward",
    "refusal", "survival-instinct", "sycophancy",
]
TRUTH_SETS = ["cities", "sp_en_trans", "larger_than", "companies_true_false", "common_claim_true_false"]


@dataclass
class Example:
    user: str
    assistant: str | None = None  # appended after the generation prompt, no end-of-turn
    raw: bool = False  # True: plain text, no chat template
    label: int = 0


def _fetch(url: str, name: str) -> str:
    path = CACHE / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as r:
            path.write_bytes(r.read())
    return path.read_text()


def _caa(behavior: str, n: int, seed: int) -> list[Example]:
    """Same A/B question; label 1 ends on the behavior-matching letter, label 0 on the other.

    The last token is the answer letter, i.e. the position CAA reads its steering vector from.
    """
    rows = json.loads(_fetch(CAA_URL.format(behavior), f"caa/{behavior}.json"))
    random.Random(seed).shuffle(rows)
    out = []
    for r in rows[:n]:
        for key, lab in (("answer_matching_behavior", 1), ("answer_not_matching_behavior", 0)):
            letter = r[key].strip().strip("()")[0]
            out.append(Example(user=r["question"].strip(), assistant=f"({letter}", label=lab))
    return out


def _truth(name: str, n: int, seed: int) -> list[Example]:
    """True vs false factual statements, plain text, last token is the final period."""
    rows = list(csv.DictReader(io.StringIO(_fetch(GOT_URL.format(name), f"got/{name}.csv"))))
    random.Random(seed).shuffle(rows)
    out = []
    for lab in (1, 0):
        cls = [r for r in rows if int(r["label"]) == lab][:n]
        out += [Example(user=r["statement"].strip(), raw=True, label=lab) for r in cls]
    return out


def _safety(mode: str, n: int, seed: int) -> list[Example]:
    convs, labels = load_safety(mode, n, seed)
    return [Example(user=c[0]["content"], assistant=c[1]["content"] if len(c) > 1 else None,
                    label=l) for c, l in zip(convs, labels)]


CONCEPTS = {
    **{f"caa_{b}": (lambda n, s, b=b: _caa(b, n, s)) for b in CAA_BEHAVIORS},
    **{f"truth_{t}": (lambda n, s, t=t: _truth(t, n, s)) for t in TRUTH_SETS},
    "safety_prompt": lambda n, s: _safety("prompt", n, s),
    "safety_response": lambda n, s: _safety("response", n, s),
}


def load_concept(name: str, n: int, seed: int = 0) -> list[Example]:
    return CONCEPTS[name](n, seed)
