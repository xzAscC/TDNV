"""Additive difference-of-means steering at 10 layers, one model and one concept per call.

The steering vector at layer l is v_l = mu_1 - mu_0 of the same last-token states that TDNV uses
(see tdnv.concepts), fit on one half of the data. As in standard activation steering (CAA, ITI),
coef * v_l is added to the input of decoder layer l at every position, prompt and generated tokens,
with coef = +alpha or -alpha, and the model answers by greedy decoding. Index l is the same index
as the TDNV curve; index L is the final-norm output.

truth_*: the model is asked "Is the following statement true or false? Answer with one word, True
  or False." about each held-out statement (chat template if the tokenizer has one). True statements
  get -alpha (target answer "false"), false ones +alpha (target "true"). The first true/false word
  in the output is the answer.
caa_*: the held-out A/B question is followed by "(" after the generation prompt, as in CAA; the
  answer is whichever of the tokens "A" and "B" has the larger logit at the next position (free
  generation is not used: some models answer the A/B question in prose). Each question is run with +alpha (target: the behavior-matching
  letter) and with -alpha (target: the other letter). Split is by question.

A run succeeds when the answer equals the target; an output with no parsable answer is invalid.

Layers: 10 indices evenly spaced over 1..L, with the one nearest the TDNV minimum replaced by it.

Example:
    uv run python scripts/run_steering.py --model Qwen/Qwen3-8B --concepts truth_cities,caa_myopic-reward
Results: <out>/<model>/steer_<concept>/metrics.json. Finished concepts are skipped.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from run_concepts import load_model
from tdnv.concepts import load_concept
from tdnv.extract import _decoder_stack, last_token_states, render
from transformers import AutoTokenizer

QUESTION = "Is the following statement true or false? Answer with one word, True or False."


def split(examples, seed):
    """Half of the data to fit the vector, the other half to evaluate.

    Truth sets: half of each class. CAA: half of the questions (both answers of a question stay
    on the same side).
    """
    if not examples[0].raw:
        questions = sorted({e.user for e in examples})
        random.Random(seed).shuffle(questions)
        fit_q = set(questions[: len(questions) // 2])
        return ([e for e in examples if e.user in fit_q],
                [e for e in examples if e.user not in fit_q])
    fit, test = [], []
    for lab in (1, 0):
        cls = [e for e in examples if e.label == lab]
        random.Random(seed).shuffle(cls)
        fit += cls[: len(cls) // 2]
        test += cls[len(cls) // 2:]
    return fit, test


def pick_layers(tdnv: list[float], k: int) -> list[int]:
    L = len(tdnv) - 1
    best = 1 + int(np.nanargmin(tdnv[1:]))
    layers = sorted({int(round(x)) for x in np.linspace(1, L, k)})
    nearest = min(layers, key=lambda l: abs(l - best))
    return sorted(set(layers) - {nearest} | {best})


def prompt(tok, statement: str) -> str:
    if tok.chat_template:
        return tok.apply_chat_template(
            [{"role": "user", "content": f"{QUESTION}\n\n{statement}"}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
    return f"{tok.bos_token or ''}{QUESTION}\n\nStatement: {statement}\nAnswer:"


def parse(text: str) -> int:
    """1 = true, 0 = false, -1 = neither."""
    m = re.search(r"\b(true|false)\b", text, re.IGNORECASE)
    return -1 if m is None else int(m.group(1).lower() == "true")


def parse_letter(text: str) -> str:
    """First standalone A/B near the start, e.g. "B)", " B ) Yes", "Answer: **B)"."""
    m = re.search(r"(?<![A-Za-z])([AB])(?![A-Za-z])", text[:24])
    return m.group(1) if m else ""


def eval_items(tok, test):
    """(prompts, coef signs, targets, parser) for the held-out half."""
    if test[0].raw:
        prompts = [prompt(tok, e.user) for e in test]
        signs = [-1.0 if e.label == 1 else 1.0 for e in test]
        return prompts, signs, [1 - e.label for e in test], parse
    letter = {(e.user, e.label): e.assistant.strip("(") for e in test}
    prompts, signs, targets = [], [], []
    for q in sorted({e.user for e in test}):
        p = render(tok, type(test[0])(user=q, assistant="("))
        for sgn, lab in ((1.0, 1), (-1.0, 0)):
            prompts.append(p)
            signs.append(sgn)
            targets.append(letter[(q, lab)])
    return prompts, signs, targets, parse_letter


class Steerer:
    """Adds coef[b] * v to the residual stream of batch item b at one layer, every position."""

    def __init__(self, model):
        self.layers, self.norm = _decoder_stack(model)
        self.handle = None
        self.v = self.coef = None

    def _add(self, h):
        delta = self.coef.to(h.device, h.dtype)[:, None, None] * self.v.to(h.device, h.dtype)
        return h + delta

    def attach(self, layer: int):
        self.detach()
        if layer < len(self.layers):
            def pre(_m, args, kwargs):
                if args:
                    return (self._add(args[0]), *args[1:]), kwargs
                kwargs["hidden_states"] = self._add(kwargs["hidden_states"])
                return args, kwargs
            self.handle = self.layers[layer].register_forward_pre_hook(pre, with_kwargs=True)
        else:  # index L: output of the final norm, as in last_token_states
            self.handle = self.norm.register_forward_hook(lambda _m, _i, out: self._add(out))

    def detach(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


@torch.no_grad()
def answer(model, tok, prompts, coefs, steerer, batch_size, max_new_tokens, choices=None):
    """Greedy answers (decoded text) under steering with per-prompt coefficients.

    choices: {answer text: token id}; if given, the answer is the choice with the largest
    next-token logit instead of a generated continuation.
    """
    out = [""] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for i in range(0, len(prompts), batch_size):
        idx = order[i: i + batch_size]
        enc = tok([prompts[j] for j in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        steerer.coef = torch.tensor([coefs[j] for j in idx], dtype=torch.float32)
        if choices:
            logits = model(**enc, logits_to_keep=1).logits[:, -1, list(choices.values())].float()
            if not torch.isfinite(logits).all():
                raise RuntimeError("non-finite logits under steering")
            for j, k in zip(idx, logits.argmax(-1).tolist()):
                out[j] = list(choices)[k]
            continue
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for j, g in zip(idx, gen[:, enc["input_ids"].shape[1]:]):
            out[j] = tok.decode(g, skip_special_tokens=True)
    return out


def run_concept(model, tok, concept: str, args) -> None:
    model_dir = Path(args.out) / args.model.replace("/", "__")
    out_path = model_dir / f"steer_{concept}" / "metrics.json"
    tdnv_dir = Path(args.tdnv_dir or args.out) / model_dir.name
    tdnv = json.loads((tdnv_dir / concept / "metrics.json").read_text())["tdnv"]
    layers = pick_layers(tdnv, args.n_layers)
    alphas = [float(a) for a in args.alphas.split(",")]
    print(f"{args.model} | {concept} | layers {layers} "
          f"(TDNV min @ L{1 + int(np.nanargmin(tdnv[1:]))})")

    t0 = time.time()
    examples = load_concept(concept, args.n, args.seed)
    assert concept.startswith(("truth_", "caa_")), "only truth_* and caa_* concepts"
    fit, test = split(examples, args.seed)
    if args.max_test:
        random.Random(args.seed).shuffle(test)
        test = test[: args.max_test]
        if not test[0].raw:  # keep both answers of each kept question
            keep = {e.user for e in test}
            test = [e for e in split(examples, args.seed)[1] if e.user in keep]

    # Steering vectors from the TDNV representation, fit half only.
    hidden = last_token_states(model, tok, [render(tok, e) for e in fit], args.batch_size, 512)
    labels = torch.tensor([e.label for e in fit])
    vec = hidden[labels == 1].mean(0) - hidden[labels == 0].mean(0)  # [L+1, d]
    del hidden

    prompts, signs, targets, parser = eval_items(tok, test)
    signs, targets = np.array(signs), np.array(targets)
    pos = signs > 0
    choices = None
    if not test[0].raw:
        choices = {c: tok(c, add_special_tokens=False)["input_ids"][0] for c in "AB"}
        assert len(set(choices.values())) == 2, choices
    print(f"  {len(prompts)} eval prompts | prompt: {prompts[0][-160:]!r} | target {targets[0]!r}")

    def score(txt):
        pred = np.array([parser(t) for t in txt])
        hit = pred == targets
        invalid = pred == ("" if pred.dtype.kind == "U" else -1)
        return dict(success_rate=float(hit.mean()), success_pos=float(hit[pos].mean()),
                    success_neg=float(hit[~pos].mean()), invalid_rate=float(invalid.mean()))

    steerer = Steerer(model)
    steerer.v = torch.zeros(vec.shape[1])
    steerer.attach(layers[0])
    base_txt = answer(model, tok, prompts, [0.0] * len(prompts), steerer, args.batch_size,
                      args.max_new_tokens, choices)
    base = score(base_txt)  # share of items already at the target answer without steering
    print(f"  unsteered: {base} | e.g. {base_txt[:3]}")

    runs = []
    for layer in tqdm(layers, desc="layers"):
        steerer.v = vec[layer]
        steerer.attach(layer)
        for a in alphas:
            txt = answer(model, tok, prompts, list(a * signs), steerer, args.batch_size,
                         args.max_new_tokens, choices)
            runs.append(dict(
                layer=layer, rel_depth=layer / (len(tdnv) - 1), alpha=a, tdnv=tdnv[layer],
                vec_norm=float(vec[layer].norm()), **score(txt),
                examples=[(prompts[k][-80:], txt[k]) for k in (0, len(prompts) - 1)],
            ))
            r = runs[-1]
            print(f"  L{layer:3d} a={a:g} TDNV={tdnv[layer]:8.2f} success={r['success_rate']:.3f} "
                  f"(+{r['success_pos']:.2f}/-{r['success_neg']:.2f}) "
                  f"invalid={r['invalid_rate']:.3f} | {txt[0]!r} {txt[-1]!r}")
    steerer.detach()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(
        layers=layers, n_layers_total=len(tdnv) - 1, base=base, runs=runs, n_fit=len(fit),
        n_test=len(test), n_eval=len(prompts), config=vars(args), seconds=time.time() - t0,
    ), indent=1))
    print(f"wrote {out_path} | {time.time() - t0:.0f}s")



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--concepts", default="truth_cities", help="comma list")
    ap.add_argument("--n", type=int, default=1000, help="max samples per class")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-layers", type=int, default=10)
    ap.add_argument("--alphas", default="0.5,1,2,4")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--max-test", type=int, default=0, help="cap on test statements (0 = all)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--tdnv-dir", default=None, help="where the TDNV metrics are (default: --out)")
    args = ap.parse_args()

    model_dir = Path(args.out) / args.model.replace("/", "__")
    todo = [c for c in args.concepts.split(",")
            if not (model_dir / f"steer_{c}" / "metrics.json").exists()]
    print(f"{args.model}: {len(todo)} concepts to steer: {todo}")
    if not todo:
        return
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_model(args.model)
    for concept in todo:
        run_concept(model, tok, concept, args)


if __name__ == "__main__":
    main()
