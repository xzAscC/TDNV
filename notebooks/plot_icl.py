"""Figures and summary for the ICL TDNV runs (scripts/run_icl.py).

    uv run python notebooks/plot_icl.py [--outputs outputs/icl] [--k 15]

Reads <outputs>/<model>/<concept>/k<K>/metrics.json. Writes, under figs/icl/:
  tdnv_grid_k<K>.pdf     TDNV vs relative depth at one K, one panel per concept, all models
  tdnv_vs_k_<model>.pdf  TDNV vs relative depth for every K, one panel per concept, one model
  layerwise_<model>.pdf  TDNV with early-exit accuracy of both tasks at one K (as in the paper's
                         Fig. 1), one panel per concept, one model
and results/icl_summary.csv (one row per model, concept and K).
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
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator

sys.path.insert(0, str(Path(__file__).parent))
from figstyle import FULL, INK, MODELS, MUTED  # noqa: E402
from plot_sweep import family_shades, log_axis, u_shape  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LABEL = {
    "truth_cities": "Truth: Cities", "truth_sp_en_trans": "Truth: Translation",
    "truth_larger_than": "Truth: Larger Than", "truth_companies_true_false": "Truth: Companies",
    "truth_common_claim_true_false": "Truth: Common Claims",
    "caa_coordinate-other-ais": "Coordinate w/ AIs", "caa_corrigible-neutral-HHH": "Corrigibility",
    "caa_hallucination": "Hallucination", "caa_myopic-reward": "Myopic Reward",
    "caa_refusal": "Refusal", "caa_survival-instinct": "Survival Instinct",
    "caa_sycophancy": "Sycophancy",
}
TASK_COLOR = {1: "#007E80", 0: "#B77544"}  # task 1 (by the concept), task 0 (flipped)
K_COLOR = {1: "#C6DBEF", 5: "#6BAED6", 10: "#2171B5", 15: "#08306B"}


def load(outputs: Path) -> dict:
    data = {}
    for m in MODELS:
        for c in LABEL:
            for f in sorted((outputs / m.replace("/", "__") / c).glob("k*/metrics.json")):
                data[(m, c, int(f.parent.name[1:]))] = json.loads(f.read_text())
    return data


def depth(tdnv):
    t = np.array(tdnv[1:], dtype=float)
    return np.arange(1, len(t) + 1) / len(t), t


def grid(n, height=1.45):
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(FULL, height * rows + 0.45), layout="constrained",
                             squeeze=False, sharex=True)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes.flat[:n]:
        ax.set_xlim(0, 1.02)
        ax.set_xticks([0, 0.5, 1])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.tick_params(labelsize=6.5)
    for ax in axes.flat[max(0, n - cols):n]:
        ax.set_xlabel("Relative Depth (Layer / L)")
        ax.xaxis.set_tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel("TDNV")
    return fig, axes


def finish_log(ax):
    log_axis(ax)
    if ax.get_ylim()[1] / ax.get_ylim()[0] > 30:
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))


def fig_grid(data, concepts, models, k, path):
    shade = family_shades(models)
    fig, axes = grid(len(concepts))
    for i, (ax, c) in enumerate(zip(axes.flat, concepts)):
        for m in models:
            if (m, c, k) not in data:
                continue
            x, t = depth(data[(m, c, k)]["tdnv"])
            j = int(np.nanargmin(t))
            ax.plot(x, t, color=shade[m], lw=0.7, label=MODELS[m][1], marker=MODELS[m][2],
                    markevery=[j], ms=3.2, markeredgecolor="white", markeredgewidth=0.4)
        finish_log(ax)
        ax.set_title(f"({chr(97 + i)}) {LABEL[c]}", loc="left", fontweight="bold")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=4, fontsize=6.5, columnspacing=1.0)
    fig.savefig(path)
    plt.close(fig)


def fig_vs_k(data, concepts, model, ks, path):
    fig, axes = grid(len(concepts))
    for i, (ax, c) in enumerate(zip(axes.flat, concepts)):
        for k in ks:
            if (model, c, k) in data:
                x, t = depth(data[(model, c, k)]["tdnv"])
                ax.plot(x, t, color=K_COLOR.get(k, INK), lw=0.8, label=f"K = {k}")
        finish_log(ax)
        ax.set_title(f"({chr(97 + i)}) {LABEL[c]}", loc="left", fontweight="bold")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=len(ks), fontsize=6.5,
               title=MODELS[model][1], title_fontsize=7)
    fig.savefig(path)
    plt.close(fig)


def fig_layerwise(data, concepts, model, k, path):
    """TDNV (left, log) and early-exit accuracy of both tasks (right, %), per concept."""
    fig, axes = grid(len(concepts), height=1.55)
    for i, (ax, c) in enumerate(zip(axes.flat, concepts)):
        if (model, c, k) not in data:
            continue
        r = data[(model, c, k)]
        x, t = depth(r["tdnv"])
        ax.plot(x, t, color=INK, lw=0.9)
        finish_log(ax)
        ax2 = ax.twinx()
        for task in (1, 0):
            acc = 100 * np.array(r[f"early_exit_t{task}"][1:])
            ax2.plot(x, acc, color=TASK_COLOR[task], lw=0.8)
        ax2.set_ylim(-3, 103)
        ax2.spines["right"].set_visible(True)
        ax2.tick_params(labelsize=6.5)
        ax2.grid(False)
        if i % 3 == 2 or i == len(concepts) - 1:
            ax2.set_ylabel("Early-Exit Acc. (%)")
        ax.set_title(f"({chr(97 + i)}) {LABEL[c]}", loc="left", fontweight="bold")
    handles = [Line2D([], [], color=INK, lw=0.9, label="TDNV"),
               Line2D([], [], color=TASK_COLOR[1], lw=0.8, label="Early exit, task 1 (concept)"),
               Line2D([], [], color=TASK_COLOR[0], lw=0.8, label="Early exit, task 0 (flipped)")]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, fontsize=6.5,
               title=f"{MODELS[model][1]}, K = {k}", title_fontsize=7)
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=str(ROOT / "outputs" / "icl"))
    ap.add_argument("--k", type=int, default=15, help="K for the grid and layerwise figures")
    ap.add_argument("--figs", default=str(ROOT / "figs" / "icl"))
    ap.add_argument("--summary", default=str(ROOT / "results" / "icl_summary.csv"))
    args = ap.parse_args()

    data = load(Path(args.outputs))
    models = [m for m in MODELS if any(key[0] == m for key in data)]
    concepts = [c for c in LABEL if any(key[1] == c for key in data)]
    ks = sorted({key[2] for key in data})
    print(f"{len(data)} runs | {len(models)} models | {len(concepts)} concepts | K = {ks}")

    rows = []
    for (m, c, k), r in data.items():
        u = u_shape(r["tdnv"])
        rows.append(dict(model=m, concept=c, k=k, n_layers=len(r["tdnv"]) - 1, argmin=u["argmin"],
                         depth=round(u["depth"], 3), tmin=u["tmin"], drop=u["drop"], rise=u["rise"],
                         is_u=u["is_u"], icl_acc_t1=r["early_exit_t1"][-1],
                         icl_acc_t0=r["early_exit_t0"][-1], zero_shot_t1=r["zero_shot_t1"],
                         zero_shot_t0=r["zero_shot_t0"], answer_format_rate=r["answer_format_rate"],
                         n_truncated=r["n_tokens"]["n_truncated"]))
    Path(args.summary).parent.mkdir(exist_ok=True)
    with open(args.summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["concept"], list(MODELS).index(r["model"]), r["k"])))

    figs = Path(args.figs)
    figs.mkdir(parents=True, exist_ok=True)
    fig_grid(data, concepts, models, args.k, figs / f"tdnv_grid_k{args.k}.pdf")
    for m in models:
        tag = m.split("/")[1]
        fig_vs_k(data, concepts, m, ks, figs / f"tdnv_vs_k_{tag}.pdf")
        fig_layerwise(data, concepts, m, args.k, figs / f"layerwise_{tag}.pdf")
    print("wrote", figs, "and", args.summary)


if __name__ == "__main__":
    main()
