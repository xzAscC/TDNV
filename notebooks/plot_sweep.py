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
from matplotlib.colors import to_rgb
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext, LogLocator, NullLocator

sys.path.insert(0, str(Path(__file__).parent))
from figstyle import CONCEPT_LABEL, FAMILY_COLOR, FULL, HALF, MODELS, MUTED, model_style  # noqa: E402

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


def shared_legend(fig, axes, models):
    """One legend above all panels, entries in model-table order, read left to right."""
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    order = [MODELS[m][1] for m in models]
    pairs = sorted(zip(handles, labels), key=lambda p: order.index(p[1]))
    ncol = min(len(pairs), 6)
    rows = math.ceil(len(pairs) / ncol)
    # matplotlib fills legend columns first; reorder so entries read left to right
    grid = [pairs[r * ncol:(r + 1) * ncol] for r in range(rows)]
    colmajor = [grid[r][c] for c in range(ncol) for r in range(rows) if c < len(grid[r])]
    fig.legend(*zip(*colmajor), loc="outside upper center", ncol=ncol, handlelength=1.8,
               columnspacing=1.0)


def fig_main(data, concepts, models, path):
    n = len(concepts)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(FULL, 1.75 * rows + 0.45), layout="constrained",
                             squeeze=False)
    for i, (ax, c) in enumerate(zip(axes.flat, concepts)):
        draw_panel(ax, data, c, models, mark_min=False)  # 11 minimum lines would hide the curves
        ax.set_title(f"({chr(97 + i)}) {CONCEPT_LABEL[c]}", loc="left", fontweight="bold")
        if i % cols == 0:
            ax.set_ylabel("TDNV")
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    shared_legend(fig, axes.flat[:n], models)
    fig.savefig(path)
    return fig


def fig_family(data, concept, path):
    fams = {}
    for m, (fam, _, _) in MODELS.items():
        if (m, concept) in data:
            fams.setdefault(fam, []).append(m)
    fams = {k: v for k, v in fams.items()}
    fig, axes = plt.subplots(1, len(fams), figsize=(FULL, 2.35), layout="constrained", squeeze=False)
    for i, (ax, (fam, ms)) in enumerate(zip(axes.flat, fams.items())):
        draw_panel(ax, data, concept, ms, rel_depth=False)
        ax.set_title(f"({chr(97 + i)}) {fam}", loc="left", fontweight="bold")
        if i == 0:
            ax.set_ylabel("TDNV")
    shared_legend(fig, axes.flat, [m for ms in fams.values() for m in ms])
    fig.savefig(path)
    return fig


def fig_concept(data, concept, models, path):
    """One half-width figure per concept: TDNV vs relative depth, one line per model,
    dashed line at each model's minimum (the layout of the paper's model-comparison panel)."""
    fig, ax = plt.subplots(figsize=(HALF, 2.75), layout="constrained")
    for model in models:
        r = data.get((model, concept))
        if r is None:
            continue
        t = np.array(r["tdnv"][1:], dtype=float)
        L = len(t)
        x = np.arange(1, L + 1) / L
        st = model_style(model)
        m = int(np.nanargmin(t))
        ax.axvline(x[m], color=st["color"], lw=0.7, ls=(0, (3, 2)), alpha=0.55, zorder=1)
        ax.plot(x, t, color=st["color"], lw=1.1, alpha=0.95, zorder=2, label=st["label"],
                marker=st["marker"], markevery=max(1, L // 6), ms=3.2,
                markeredgecolor="white", markeredgewidth=0.4)
    log_axis(ax)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Relative Depth (Layer / L)")
    ax.set_ylabel("TDNV")
    handles, labels = ax.get_legend_handles_labels()
    order = [MODELS[m][1] for m in models]
    pairs = sorted(zip(handles, labels), key=lambda p: order.index(p[1]))
    ncol = 3
    rows = math.ceil(len(pairs) / ncol)
    grid = [pairs[r * ncol:(r + 1) * ncol] for r in range(rows)]
    colmajor = [grid[r][c] for c in range(ncol) for r in range(rows) if c < len(grid[r])]
    fig.legend(*zip(*colmajor), loc="outside upper center", ncol=ncol, fontsize=6.5,
               handlelength=1.6, columnspacing=0.8, labelspacing=0.25, handletextpad=0.4)
    fig.savefig(path)
    return fig


def family_shades(models):
    """Model -> color: one hue per family, lighter for smaller models (MODELS lists them by size)."""
    out = {}
    for fam, base in FAMILY_COLOR.items():
        ms = [m for m in models if MODELS[m][0] == fam]
        rgb = np.array(to_rgb(base))
        for i, m in enumerate(ms):
            t = np.linspace(-0.3, 0.35, len(ms))[i] if len(ms) > 1 else 0.0
            out[m] = tuple(rgb + (1 - rgb) * -t) if t < 0 else tuple(rgb * (1 - t))
    return out


def fig_concept_shades(data, concept, models, path):
    """Full-width single panel: TDNV vs relative depth for every model, one hue per family,
    shade by model size, one marker at each model's minimum. Legend on the right, by family."""
    models = [m for m in models if (m, concept) in data]
    shade = family_shades(models)
    fig, ax = plt.subplots(figsize=(4.1, 1.75), layout="constrained")
    for model in models:
        t = np.array(data[(model, concept)]["tdnv"][1:], dtype=float)
        L = len(t)
        x = np.arange(1, L + 1) / L
        _, label, marker = MODELS[model]
        c = shade[model]
        m = int(np.nanargmin(t))
        ax.plot(x, t, color=c, lw=0.8, alpha=0.9, zorder=2, label=label, marker=marker,
                markevery=[m], ms=4.2, markeredgecolor="white", markeredgewidth=0.5)
    log_axis(ax)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Relative Depth (Layer / L)")
    ax.set_ylabel("TDNV")
    fig.legend(loc="outside right center", ncol=1, fontsize=6.5, handlelength=1.4,
               handletextpad=0.4, labelspacing=0.2, markerscale=0.8)
    fig.savefig(path)
    return fig


def fig_family_panels(data, concept, models, path, title=None, xlabel=True, height=1.75):
    """Main-text figure: one panel per model family, shared log y-axis, TDNV vs relative depth.
    Models in a family share its hue, darker for larger models; one marker at each minimum.
    The appendix stacks one per concept: `title` names the concept above the row (panel titles
    then drop their letters) and `xlabel=False` leaves the axis label to the last row."""
    models = [m for m in models if (m, concept) in data]
    fams = [f for f in FAMILY_COLOR if any(MODELS[m][0] == f for m in models)]
    shade = family_shades(models)
    # Appendix rows use fixed margins (inches) so the stacked rows line up panel for panel.
    fig, axes = plt.subplots(1, len(fams), figsize=(FULL, height), sharey=True, squeeze=False,
                             layout="constrained" if title is None else None)
    if title is not None:
        bottom = 0.42 if xlabel else 0.2
        fig.subplots_adjust(left=0.47 / FULL, right=1 - 0.04 / FULL, wspace=0.12,
                            top=1 - 0.4 / height, bottom=bottom / height)
    for i, (ax, fam) in enumerate(zip(axes.flat, fams)):
        for model in [m for m in models if MODELS[m][0] == fam]:
            t = np.array(data[(model, concept)]["tdnv"][1:], dtype=float)
            L = len(t)
            x = np.arange(1, L + 1) / L
            _, label, marker = MODELS[model]
            m = int(np.nanargmin(t))
            ax.plot(x, t, color=shade[model], lw=0.9, zorder=2, label=label.rsplit("-", 1)[1], marker=marker,
                    markevery=[m], ms=3.6, markeredgecolor="white", markeredgewidth=0.4)
        ax.set_xlim(0, 1.02)
        ax.set_xticks([0, 0.5, 1])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        if title is None:
            ax.set_title(f"({chr(97 + i)}) {fam}", loc="left", fontweight="bold")
        else:
            ax.set_title(fam, loc="left", fontsize=7.5, color=MUTED)
        n_fam = sum(MODELS[m][0] == fam for m in models)
        compact = title is not None and n_fam > 2  # short appendix rows: 2x2 legend
        ax.legend(loc="upper right", fontsize=6.5, handlelength=1.0 if compact else 1.3,
                  handletextpad=0.3 if compact else 0.4, labelspacing=0.15 if compact else 0.2,
                  ncol=2 if compact else 1, columnspacing=0.6, borderaxespad=0.2)
    log_axis(axes.flat[0])
    if axes.flat[0].get_ylim()[1] / axes.flat[0].get_ylim()[0] > 30:  # label every decade
        axes.flat[0].yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    axes.flat[0].set_ylabel("TDNV")
    if xlabel:
        pos = {} if title is None else dict(y=0.03 / height, va="bottom")
        fig.supxlabel("Relative Depth (Layer / L)", fontsize=8, **pos)
    if title is not None:
        fig.suptitle(title, x=0.01, y=1 - 0.03 / height, ha="left", va="top", fontweight="bold",
                     fontsize=8.5)
    fig.savefig(path)
    return fig


def fig_concepts_grid(data, concepts, models, path, cols=3):
    """Appendix grid in the style of fig_concept_shades: one panel per concept, all models,
    one hue per family with shade by size, marker at each model's minimum. The legend fills the
    empty slots after the last panel, or sits above the grid when the last row is full."""
    models = [m for m in models if any((m, c) in data for c in concepts)]
    shade = family_shades(models)
    rows = math.ceil(len(concepts) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(FULL, 1.6 * rows + 0.35), layout="constrained",
                             squeeze=False, sharex=True)
    for i, (ax, c) in enumerate(zip(axes.flat, concepts)):
        for model in models:
            r = data.get((model, c))
            if r is None:
                continue
            t = np.array(r["tdnv"][1:], dtype=float)
            L = len(t)
            x = np.arange(1, L + 1) / L
            _, label, marker = MODELS[model]
            m = int(np.nanargmin(t))
            ax.plot(x, t, color=shade[model], lw=0.7, alpha=0.9, zorder=2, label=label,
                    marker=marker, markevery=[m], ms=3.4, markeredgecolor="white",
                    markeredgewidth=0.4)
        log_axis(ax)
        if ax.get_ylim()[1] / ax.get_ylim()[0] > 30:  # label every decade
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
        ax.set_xlim(0, 1.02)
        ax.set_xticks([0, 0.5, 1])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.tick_params(labelsize=6.5)
        ax.set_title(f"({chr(97 + i)}) {CONCEPT_LABEL[c]}", loc="left", fontweight="bold")
        if i % cols == 0:
            ax.set_ylabel("TDNV")
    n = len(concepts)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes.flat[n - cols:n]:
        ax.set_xlabel("Relative Depth (Layer / L)")
        ax.xaxis.set_tick_params(labelbottom=True)
    # legend in the empty slots after the last panel (outside constrained layout, so it does
    # not resize the grid)
    # legend columns fill top to bottom: Qwen3 + OLMo-3 in the first, Gemma-2 + Gemma-3 in the second
    hl = dict(zip(*axes.flat[0].get_legend_handles_labels()[::-1]))
    fam_order = ["Qwen3", "OLMo-3", "Gemma-2", "Gemma-3"]
    labels = [MODELS[m][1] for f in fam_order for m in models if MODELS[m][0] == f]
    handles = [hl[l] for l in labels]
    if n % cols:
        fig.legend(handles, labels, loc="center",
                   bbox_to_anchor=(0.5 + 0.5 * (n % cols) / cols, 0.5 / rows),
                   ncol=cols - n % cols, fontsize=7, handlelength=1.6, handletextpad=0.4,
                   columnspacing=1.4, labelspacing=0.35)
    else:
        shared_legend(fig, axes.flat[:n], models)
    fig.savefig(path)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=str(ROOT / "outputs"))
    ap.add_argument("--preview", default=None, help="also write PNG previews here (not figs/)")
    ap.add_argument("--families", default="Qwen3,Gemma-2,Gemma-3", help="comma list")
    args = ap.parse_args()

    families = args.families.split(",")
    data = {k: v for k, v in load(Path(args.outputs)).items() if MODELS[k[0]][0] in families}
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
    ranked = sorted(concepts, key=lambda c: (score[c][0] >= 0.8, score[c][1]), reverse=True)
    print(f"{'concept':34s} {'U-frac':>6s} {'med log(drop*rise)':>18s}")
    for c in ranked:
        print(f"{c:34s} {score[c][0]:6.2f} {score[c][1]:18.2f}")

    figs = ROOT / "figs"
    figs.mkdir(exist_ok=True)
    out = {
        "tdnv_main": fig_main(data, ranked[:6], models, figs / "tdnv_main.pdf"),
        "tdnv_all_concepts": fig_main(data, concepts, models, figs / "tdnv_all_concepts.pdf"),
        "tdnv_by_family": fig_family(data, ranked[0], figs / "tdnv_by_family.pdf"),
        "tdnv_by_family_cities": fig_family(data, "truth_cities", figs / "tdnv_by_family_cities.pdf"),
        "tdnv_cities": fig_family_panels(data, "truth_cities", models, figs / "tdnv_cities.pdf"),
    }
    # Appendix: one family-panel row per concept, stacked on one page.
    app = [c for c in CONCEPT_LABEL if c != "truth_cities"]
    (figs / "appendix").mkdir(exist_ok=True)
    for i, c in enumerate(app):
        last = i == len(app) - 1
        out[f"appendix/tdnv_{c}"] = fig_family_panels(
            data, c, models, figs / "appendix" / f"tdnv_{c}.pdf",
            title=f"({chr(97 + i)}) {CONCEPT_LABEL[c]}", xlabel=last, height=1.5 if last else 1.33)
    out |= {
        "tdnv_appendix_concepts": fig_concepts_grid(
            data, [c for c in concepts if c != "truth_cities"], models,
            figs / "tdnv_appendix_concepts.pdf"),
    }
    (figs / "concepts").mkdir(exist_ok=True)
    for c in concepts:
        out[f"concepts/tdnv_{c}"] = fig_concept(data, c, models, figs / "concepts" / f"tdnv_{c}.pdf")
    if args.preview:
        Path(args.preview).mkdir(parents=True, exist_ok=True)
        for name, fig in out.items():
            (Path(args.preview) / name).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(Path(args.preview) / f"{name}.png", dpi=200)
    print("wrote", len(out), "figures:", ", ".join(f"figs/{k}.pdf" for k in out))


if __name__ == "__main__":
    main()
