"""Aggregate the concept sweep and draw the paper figures.

    uv run python notebooks/plot_sweep.py [--outputs outputs] [--preview DIR]

Writes figs/*.pdf and results/u_shape_summary.csv. Reads outputs/<model>/<concept>/metrics.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext, LogLocator, NullLocator

sys.path.insert(0, str(Path(__file__).parent))
from figstyle import CONCEPT_LABEL, FULL, MODELS, MUTED, model_style  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load(outputs: Path) -> dict:
    data = {}
    for model in MODELS:
        mdir = outputs / model.replace("/", "__")
        for f in sorted(mdir.glob("*/metrics.json")):
            if f.parent.name in CONCEPT_LABEL:
                data[(model, f.parent.name)] = json.loads(f.read_text())
    return data


def u_shape(tdnv: list[float]) -> dict:
    """Down-then-up statistics over layers 1..L (layer 0 = embeddings is skipped).

    drop = max TDNV before the minimum / min, rise = TDNV at the last layer / min.
    Counted as U-shaped when drop >= 2, rise >= 1.5 and the minimum is not in the last 10% of depth.
    """
    t = np.array(tdnv[1:], dtype=float)
    L = len(t)
    m = int(np.nanargmin(t))
    drop = float(np.nanmax(t[: m + 1]) / t[m])
    rise = float(t[-1] / t[m])
    return dict(argmin=m + 1, depth=(m + 1) / L, tmin=float(t[m]), drop=drop, rise=rise,
                is_u=bool(drop >= 2 and rise >= 1.5 and (m + 1) / L < 0.9))


def log_axis(ax):
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    if hi / lo > 30:
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
    else:  # under ~1.5 decades: label 1-2-5 steps so the panel has more than one tick
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(NullLocator())


def draw_panel(ax, data, concept, models, rel_depth=True, mark_min=True):
    for model in models:
        r = data.get((model, concept))
        if r is None:
            continue
        t = np.array(r["tdnv"][1:], dtype=float)
        L = len(t)
        x = np.arange(1, L + 1) / L if rel_depth else np.arange(1, L + 1)
        st = model_style(model)
        ax.plot(x, t, color=st["color"], marker=st["marker"], markevery=max(1, L // 8),
                label=st["label"], ms=2.6, lw=0.9)
        if mark_min:
            m = int(np.nanargmin(t))
            ax.axvline(x[m], color=st["color"], lw=0.6, ls="--", alpha=0.45, zorder=0)
    log_axis(ax)
    ax.set_xlabel("Relative Depth (Layer / L)" if rel_depth else "Layer")


def fig_main(data, concepts, models, path):
    n = len(concepts)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(FULL, 1.75 * rows + 0.45), layout="constrained",
                             squeeze=False)
    for i, (ax, c) in enumerate(zip(axes.flat, concepts)):
        draw_panel(ax, data, c, models)
        ax.set_title(f"({chr(97 + i)}) {CONCEPT_LABEL[c]}", loc="left", fontweight="bold")
        if i % cols == 0:
            ax.set_ylabel("TDNV")
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    for ax in axes.flat[1:n]:  # collect models missing from panel (a)
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    order = [MODELS[m][1] for m in models]
    pairs = sorted(zip(handles, labels), key=lambda p: order.index(p[1]))
    ncol = min(len(pairs), 6)
    rows_ = math.ceil(len(pairs) / ncol)
    # matplotlib fills legend columns first; reorder so entries read left to right in model order
    grid = [pairs[r * ncol:(r + 1) * ncol] for r in range(rows_)]
    colmajor = [grid[r][c] for c in range(ncol) for r in range(rows_) if c < len(grid[r])]
    fig.legend(*zip(*colmajor), loc="outside upper center", ncol=ncol, handlelength=1.8,
               columnspacing=1.0)
    fig.savefig(path)
    return fig


def fig_family(data, concept, path):
    fams = {}
    for m, (fam, _, _) in MODELS.items():
        if (m, concept) in data:
            fams.setdefault(fam, []).append(m)
    fams = {k: v for k, v in fams.items()}
    fig, axes = plt.subplots(1, len(fams), figsize=(FULL, 1.9), layout="constrained", squeeze=False)
    for i, (ax, (fam, ms)) in enumerate(zip(axes.flat, fams.items())):
        draw_panel(ax, data, concept, ms, rel_depth=False)
        ax.set_title(f"({chr(97 + i)}) {fam}", loc="left", fontweight="bold")
        ax.legend(loc="upper right", handlelength=1.5)
        if i == 0:
            ax.set_ylabel("TDNV")
    fig.savefig(path)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=str(ROOT / "outputs"))
    ap.add_argument("--preview", default=None, help="also write PNG previews here (not figs/)")
    args = ap.parse_args()

    data = load(Path(args.outputs))
    models = [m for m in MODELS if any(k[0] == m for k in data)]
    concepts = [c for c in CONCEPT_LABEL if any(k[1] == c for k in data)]
    print(f"{len(data)} runs | {len(models)} models | {len(concepts)} concepts")

    rows = []
    for (model, concept), r in data.items():
        s = u_shape(r["tdnv"])
        rows.append(dict(model=model, concept=concept, n_layers=len(r["tdnv"]) - 1, **s,
                         tdnv_l1=r["tdnv"][1], tdnv_last=r["tdnv"][-1],
                         shuffled_at_min=r["tdnv_shuffled"][s["argmin"]],
                         probe_acc_at_min=r["probe_acc"][s["argmin"]]))
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results/u_shape_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["concept"], r["model"])))

    # Rank concepts by how many models show the down-then-up shape, then by median drop*rise.
    score = {}
    for c in concepts:
        rs = [r for r in rows if r["concept"] == c]
        score[c] = (sum(r["is_u"] for r in rs) / len(rs),
                    float(np.median([math.log(r["drop"] * r["rise"]) for r in rs])))
    # Main figure: concepts where most models are U-shaped, ordered by how deep the U is.
    ranked = sorted(concepts, key=lambda c: (score[c][0] >= 0.6, score[c][1]), reverse=True)
    print(f"{'concept':34s} {'U-frac':>6s} {'med log(drop*rise)':>18s}")
    for c in ranked:
        print(f"{c:34s} {score[c][0]:6.2f} {score[c][1]:18.2f}")

    figs = ROOT / "figs"
    figs.mkdir(exist_ok=True)
    out = {
        "tdnv_main": fig_main(data, ranked[:6], models, figs / "tdnv_main.pdf"),
        "tdnv_all_concepts": fig_main(data, concepts, models, figs / "tdnv_all_concepts.pdf"),
        "tdnv_by_family": fig_family(data, ranked[0], figs / "tdnv_by_family.pdf"),
    }
    if args.preview:
        Path(args.preview).mkdir(parents=True, exist_ok=True)
        for name, fig in out.items():
            fig.savefig(Path(args.preview) / f"{name}.png", dpi=200)
    print("wrote", ", ".join(f"figs/{k}.pdf" for k in out))


if __name__ == "__main__":
    main()
