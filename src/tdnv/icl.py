"""ICL instances on the steering concepts, in the paper's setting (Sec. 3).

Each concept gives two ICL tasks over the same inputs x:
  task 1 labels every input by the concept (Geometry of Truth: its truth value; CAA: the answer
         that matches the behavior),
  task 0 labels every input the opposite way (the false truth value; the non-matching answer).
An instance is K demonstrations (x_k -> y_t(x_k)) followed by a query (x_q ->), as plain text
with a BOS token. Its representation is the last token of the query, the separator before the
answer. Both tasks use the same query and the same demonstration inputs, so the two instances of
a query differ only in the demonstration labels.

Formats (a demonstration ends with a blank line):
  truth_*: "Statement: <x>\\nAnswer: <True|False>"      query "Statement: <x>\\nAnswer:"
  caa_*:   "<question>\\nAnswer: (<A|B>)"               query "<question>\\nAnswer: ("
The answer token of the query is " True" / " False" for truth_*, "A" / "B" for caa_*.
"""

from __future__ import annotations

import csv
import io
import json
import random
from dataclasses import dataclass

from tdnv.concepts import CAA_BEHAVIORS, CAA_URL, GOT_URL, TRUTH_SETS, _fetch

CONCEPTS = [f"truth_{t}" for t in TRUTH_SETS] + [f"caa_{b}" for b in CAA_BEHAVIORS]


@dataclass(frozen=True)
class Item:
    x: str
    y1: str  # answer under task 1 (by the concept)
    y0: str  # answer under task 0 (opposite)

    def y(self, task: int) -> str:
        return self.y1 if task == 1 else self.y0


def load_items(concept: str) -> list[Item]:
    if concept.startswith("truth_"):
        name = concept.removeprefix("truth_")
        rows = csv.DictReader(io.StringIO(_fetch(GOT_URL.format(name), f"got/{name}.csv")))
        return [Item(r["statement"].strip(), *(("True", "False") if int(r["label"]) else ("False", "True")))
                for r in rows]
    behavior = concept.removeprefix("caa_")
    rows = json.loads(_fetch(CAA_URL.format(behavior), f"caa/{behavior}.json"))
    letter = lambda s: s.strip().strip("()")[0]  # noqa: E731
    return [Item(r["question"].strip(), letter(r["answer_matching_behavior"]),
                 letter(r["answer_not_matching_behavior"])) for r in rows]


def demo(concept: str, x: str, y: str) -> str:
    return f"Statement: {x}\nAnswer: {y}\n\n" if concept.startswith("truth_") else f"{x}\nAnswer: ({y})\n\n"


def query(concept: str, x: str) -> str:
    return f"Statement: {x}\nAnswer:" if concept.startswith("truth_") else f"{x}\nAnswer: ("


def answer_text(concept: str, y: str) -> str:
    """The answer as it follows the query (its first token is the target token)."""
    return f" {y}" if concept.startswith("truth_") else y


@dataclass
class Instance:
    query: Item
    demos: list[Item]  # max K; an instance with K demonstrations uses demos[:K]


def sample(concept: str, items: list[Item], n_query: int, k_max: int, seed: int) -> list[Instance]:
    """n_query queries, each with k_max demonstrations drawn from the other items.

    Truth sets: queries are half true, half false; each query's demonstrations are half true,
    half false (one extra of a random class when k_max is odd), in random order. The K-shot
    instance uses the first K, so instances with different K differ only in how many
    demonstrations they show.
    """
    rng = random.Random(seed)
    if concept.startswith("truth_"):
        by_cls = {c: [it for it in items if it.y1 == c] for c in ("True", "False")}
        queries = [it for c in ("True", "False") for it in rng.sample(by_cls[c], n_query // 2)]
    else:
        queries = rng.sample(items, n_query)
    out = []
    for q in queries:
        if concept.startswith("truth_"):
            first = rng.choice(["True", "False"])
            sizes = {first: (k_max + 1) // 2, ("False" if first == "True" else "True"): k_max // 2}
            demos = [it for c, k in sizes.items()
                     for it in rng.sample([it for it in by_cls[c] if it != q], k)]
            rng.shuffle(demos)
        else:
            demos = rng.sample([it for it in items if it != q], k_max)
        out.append(Instance(q, demos))
    return out


def render(concept: str, inst: Instance, k: int, task: int, bos: str) -> str:
    """Plain-text K-shot prompt for one task; with k = 0 both tasks give the same prompt."""
    shots = "".join(demo(concept, d.x, d.y(task)) for d in inst.demos[:k])
    return bos + shots + query(concept, inst.query.x)
