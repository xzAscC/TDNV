"""TDNV and additive steering on a shared relative-depth axis, per model family, plus a summary
over every (model, concept) pair with steering results.

    uv run python notebooks/plot_steer_grid.py [--concept truth_cities] [--max-invalid 0.1]

Reads outputs/<model>/<concept>/metrics.json (TDNV) and outputs/<model>/steer_<concept>/metrics.json
(scripts/run_steering.py). Writes figs/tdnv_steering.pdf.
(a)-(d) TDNV on --concept, (e)-(h) steering success on --concept, (i) min-TDNV depth vs best-steering
depth, (j) per-model Spearman correlation between -log TDNV and steering success over steered layers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator

sys.path.insert(0, str(Path(__file__).parent))
from figstyle import CONCEPT_LABEL, FAMILY_COLOR, FULL, INK, MODELS, MUTED, RULE  # noqa: E402
from plot_steering import steer_curve  # noqa: E402
from plot_sweep import family_shades, log_axis  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
def spearmanr(a, b) -> float:
    """Spearman correlation with average ranks for ties."""
    def rank(v):
        v = np.asarray(v, dtype=float)
        r = np.empty(len(v))
        r[np.argsort(v, kind="stable")] = np.arange(len(v))
        for u in np.unique(v):  # average ranks of ties
            r[v == u] = r[v == u].mean()
        return r
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


SHORT = {  # concept -> x-tick label in (j), in panel order
    "truth_cities": "Cities", "truth_sp_en_trans": "Transl.", "truth_larger_than": "Larger",
    "truth_companies_true_false": "Compan.", "truth_common_claim_true_false": "Claims",
    "caa_corrigible-neutral-HHH": "Corrig.", "caa_myopic-reward": "Myopic",
}


def load(outputs: Path, families: list[str]) -> dict:
    """(model, concept) -> (TDNV metrics, steering metrics) for every pair that has both,
    models of `families` only."""
    data = {}
    for m in [m for m in MODELS if MODELS[m][0] in families]:
        d = outputs / m.replace("/", "__")
        for c in SHORT:
            t, s = d / c / "metrics.json", d / f"steer_{c}" / "metrics.json"
            if t.exists() and s.exists():
                data[(m, c)] = (json.loads(t.read_text()), json.loads(s.read_text()))
    return data


def pair_stats(t: dict, s: dict, max_invalid: float) -> dict:
    tdnv = np.array(t["tdnv"][1:], dtype=float)  # layers 1..L
    L = len(tdnv)
    layers, succ = steer_curve(s, max_invalid)
    rho = spearmanr(-np.log(tdnv[layers - 1]), succ) if np.ptp(succ) > 0 else np.nan
    return dict(L=L, tdnv=tdnv, k=int(np.nanargmin(tdnv)), layers=layers, succ=succ,
                best=int(layers[np.argmax(succ)]), rho=rho)


def draw_family_rows(top, stats, concept, models, title=None, xlabel=True):
    """TDNV (top row) and steering success (bottom row) per model family on one concept.
    Panels are lettered (a), (b), ... unless `title` is given (appendix rows: the title names the
    concept and panel titles name only the family); `xlabel=False` leaves the axis label out."""
    fams = [f for f in FAMILY_COLOR if any(MODELS[m][0] == f for m in models)]
    shade = family_shades(models)
    axes = top.subplots(2, len(fams), sharex=True, sharey="row")
    for i, fam in enumerate(fams):
        a0, a1 = axes[0, i], axes[1, i]
        for model in [m for m in models if MODELS[m][0] == fam]:
            st = stats[(model, concept)]
            x = np.arange(1, st["L"] + 1) / st["L"]
            _, label, marker = MODELS[model]
            c = shade[model]
            a0.plot(x, st["tdnv"], color=c, lw=0.9, zorder=2, label=label.rsplit("-", 1)[1],
                    marker=marker, markevery=[st["k"]], ms=3.6, markeredgecolor="white",
                    markeredgewidth=0.4)
            xs = st["layers"] / st["L"]
            a1.plot(xs, st["succ"], color=c, lw=0.9, zorder=2, marker=marker, ms=2.4,
                    markeredgecolor="white", markeredgewidth=0.3)
            b = int(np.argmax(st["succ"]))
            a1.plot(xs[b], st["succ"][b], marker="*", color=c, ms=6.5, markeredgecolor="white",
                    markeredgewidth=0.4, zorder=4)
            a1.plot(x[st["k"]], 108, marker="v", color=c, ms=4.5, markeredgecolor="white",
                    markeredgewidth=0.4, zorder=4, clip_on=False)
        if title is None:
            a0.set_title(f"({chr(97 + i)}) {fam}", loc="left", fontweight="bold")
            a1.set_title(f"({chr(97 + len(fams) + i)}) {fam}", loc="left", fontweight="bold", pad=9)
        else:
            a0.set_title(fam, loc="left", fontsize=7.5, color=MUTED)
            a1.set_title("", pad=9)
        compact = sum(MODELS[m][0] == fam for m in models) > 2  # 2x2 legend
        a0.legend(loc="upper right", fontsize=6.5, handlelength=1.0 if compact else 1.3,
                  handletextpad=0.3 if compact else 0.4, labelspacing=0.15 if compact else 0.2,
                  ncol=2 if compact else 1, columnspacing=0.6, borderaxespad=0.2)
    log_axis(axes[0, 0])
    if axes[0, 0].get_ylim()[1] / axes[0, 0].get_ylim()[0] > 30:
        axes[0, 0].yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    axes[0, 0].set_ylabel("TDNV")
    axes[1, 0].set_ylabel("Steering Success (%)")
    axes[1, 0].set_ylim(-3, 103)
    axes[1, 0].set_xlim(0, 1.02)
    axes[1, 0].set_xticks([0, 0.5, 1])
    axes[1, 0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    if title is not None:
        top.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=8.5)
    if xlabel:
        top.supxlabel("Relative Depth (Layer / L)", fontsize=8)
    marks = [Line2D([], [], ls="", marker="v", color=MUTED, ms=4.5, label="Min-TDNV layer"),
             Line2D([], [], ls="", marker="*", color=MUTED, ms=6.5, label="Best steering layer")]
    top.legend(handles=marks, loc="lower right", ncol=2, fontsize=6.5, handletextpad=0.2,
               columnspacing=1.0, borderaxespad=0.1)
    return axes


def fig_steer_grid(data, concept, max_invalid, path):
    stats = {key: pair_stats(*v, max_invalid) for key, v in data.items()}
    models = [m for m in MODELS if (m, concept) in data]
    fams = [f for f in FAMILY_COLOR if any(MODELS[m][0] == f for m in models)]

    fig = plt.figure(figsize=(FULL, 5.0), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[2.9, 2.1])
    draw_family_rows(top, stats, concept, models)

    ai, aj = bottom.subplots(1, 2, width_ratios=[1, 1.45])
    # (i) depth of the min-TDNV layer vs depth of the best steering layer, every pair
    dx = np.array([(st["k"] + 1) / st["L"] for st in stats.values()])
    dy = np.array([st["best"] / st["L"] for st in stats.values()])
    got = np.array([c.startswith("truth_") for _, c in stats])
    ai.plot([0, 1], [0, 1], color=MUTED, lw=0.6, zorder=1)
    ai.plot(dx[got], dy[got], ls="", marker="o", ms=3.6, mfc="none", mec=INK, mew=0.7,
            label="Geometry of Truth")
    ai.plot(dx[~got], dy[~got], ls="", marker="^", ms=3.8, color=INK, mec="white", mew=0.3,
            label="CAA behaviors")
    rho = spearmanr(dx, dy)
    ai.text(0.03, 0.97, f"Spearman $\\rho$ = {rho:.2f}\nmean |Δ| = {np.mean(np.abs(dx - dy)):.2f}\n"
            f"fixed mid-depth: {np.mean(np.abs(0.5 - dy)):.2f}", transform=ai.transAxes,
            va="top", fontsize=6.5, color=INK, linespacing=1.0)
    ai.set_xlim(0, 1)
    ai.set_ylim(0, 1)
    ai.set_aspect("equal")
    ai.grid(axis="x", color=RULE, lw=0.4)
    ai.set_xlabel("Depth of Min-TDNV Layer")
    ai.set_ylabel("Depth of Best Steering Layer")
    n_models = len({m for m, _ in stats})
    n_conc = len({c for _, c in stats})
    ai.set_title(f"({chr(97 + 2 * len(fams))}) {len(stats)} model–concept pairs", loc="left", fontweight="bold")
    ai.legend(loc="lower right", fontsize=6.5, handletextpad=0.2, borderaxespad=0.2)

    # (j) per-model layerwise Spearman correlation, one column per concept with results
    concepts = [c for c in SHORT if any(cc == c for _, cc in stats)]
    rng = np.random.default_rng(0)
    for j, c in enumerate(concepts):
        r = np.array([st["rho"] for (m, cc), st in stats.items() if cc == c])
        r = r[np.isfinite(r)]
        aj.plot(j + rng.uniform(-0.15, 0.15, len(r)), r, ls="", marker="o", ms=3.2, color=INK,
                mec="white", mew=0.3)
        if len(r):
            aj.plot([j - 0.25, j + 0.25], [np.median(r)] * 2, color=INK, lw=1.2)
    aj.axhline(0, color=MUTED, lw=0.5)
    aj.set_xticks(range(len(concepts)), [SHORT[c] for c in concepts])
    aj.set_xlim(-0.6, len(concepts) - 0.4)
    aj.set_ylim(-1.05, 1.05)
    aj.set_ylabel("Spearman ρ(−log TDNV, Steering)")
    aj.set_title(f"({chr(98 + 2 * len(fams))}) Layerwise correlation, one dot per model", loc="left", fontweight="bold")
    fig.savefig(path)
    print(f"(i) {len(stats)} pairs, {n_models} models, {n_conc} concepts; rho={rho:.3f}")
    for c in concepts:
        r = [st["rho"] for (m, cc), st in stats.items() if cc == c]
        print(f"(j) {c}: n={len(r)} median={np.nanmedian(r):.2f}")
    return fig


def fig_appendix_rows(data, concepts, max_invalid, outdir):
    """One TDNV + steering block per concept, stacked in LaTeX into one appendix figure, lettered
    (a), (b), ...; the last block carries the axis label and the marker legend."""
    stats = {key: pair_stats(*v, max_invalid) for key, v in data.items()}
    outdir.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(concepts):
        last = i == len(concepts) - 1
        models = [m for m in MODELS if (m, c) in data]
        fig = plt.figure(figsize=(FULL, 2.85 if last else 2.5), layout="constrained")
        draw_family_rows(fig, stats, c, models, title=f"({chr(97 + i)}) {CONCEPT_LABEL[c]}",
                         xlabel=last)
        if not last:
            fig.legends.clear()
        fig.savefig(outdir / f"steer_{c}.pdf")
        plt.close(fig)


def summary(data, max_invalid):
    """Per concept: medians over models of the layerwise Spearman rho, the depth gap between the
    min-TDNV and best steering layers, and the peak steering success."""
    stats = {key: pair_stats(*v, max_invalid) for key, v in data.items()}
    print(f"{'concept':32s} {'n':>2s} {'med rho':>7s} {'med |gap|':>9s} {'med peak%':>9s}")
    for c in SHORT:
        sts = [st for (m, cc), st in stats.items() if cc == c]
        if not sts:
            continue
        rho = np.nanmedian([st["rho"] for st in sts])
        gap = np.median([abs((st["k"] + 1) / st["L"] - st["best"] / st["L"]) for st in sts])
        peak = np.median([st["succ"].max() for st in sts])
        print(f"{c:32s} {len(sts):2d} {rho:7.2f} {gap:9.2f} {peak:9.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=str(ROOT / "outputs"))
    ap.add_argument("--concept", default="caa_myopic-reward")
    ap.add_argument("--max-invalid", type=float, default=0.1)
    ap.add_argument("--families", default="Qwen3,Gemma-2,Gemma-3", help="comma list")
    args = ap.parse_args()
    path = ROOT / "figs" / "tdnv_steering.pdf"
    data = load(Path(args.outputs), args.families.split(","))
    fig_steer_grid(data, args.concept, args.max_invalid, path)
    print("wrote", path)
    # appendix: behavioral concepts first, then Geometry of Truth
    app = sorted((c for c in CONCEPT_LABEL if c != args.concept), key=lambda c: not c.startswith("caa_"))
    for group in (app[:3], app[3:]):  # two appendix figures of three concepts, one page each
        fig_appendix_rows(data, group, args.max_invalid, ROOT / "figs" / "appendix")
    print("wrote", ", ".join(f"figs/appendix/steer_{c}.pdf" for c in app))
    summary(data, args.max_invalid)


if __name__ == "__main__":
    main()
