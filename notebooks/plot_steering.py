"""TDNV and steering success against relative depth, one line per model.

    uv run python notebooks/plot_steering.py [--concept truth_cities] [--preview DIR]

Reads outputs/<model>/<concept>/metrics.json (TDNV) and outputs/<model>/steer_<concept>/metrics.json
(scripts/run_steering.py). Steering success at a layer is the best success rate over the tested
alphas whose invalid-answer rate is at most --max-invalid. Writes figs/steer_<concept>.pdf.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from figstyle import CONCEPT_LABEL, MODELS  # noqa: E402
from plot_sweep import family_shades, log_axis  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def steer_curve(r: dict, max_invalid: float):
    """(layer indices, success %) with the best valid alpha per layer."""
    best = {}
    for run in r["runs"]:
        if run["invalid_rate"] <= max_invalid:
            best[run["layer"]] = max(best.get(run["layer"], 0.0), run["success_rate"])
    layers = sorted(set(r["layers"]))
    return np.array(layers), np.array([100 * best.get(l, 0.0) for l in layers])


def fig_steer(outputs: Path, concept: str, max_invalid: float, path: Path):
    data = {}
    for m in MODELS:
        d = outputs / m.replace("/", "__")
        if (d / concept / "metrics.json").exists() and (d / f"steer_{concept}" / "metrics.json").exists():
            data[m] = (json.loads((d / concept / "metrics.json").read_text()),
                       json.loads((d / f"steer_{concept}" / "metrics.json").read_text()))
    shade = family_shades(list(data))
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(4.1, 3.2), sharex=True, layout="constrained")
    for m, (t, s) in data.items():
        _, label, marker = MODELS[m]
        tdnv = np.array(t["tdnv"][1:], dtype=float)
        L = len(tdnv)
        x = np.arange(1, L + 1) / L
        k = int(np.nanargmin(tdnv))
        kw = dict(color=shade[m], lw=0.8, alpha=0.9, zorder=2)
        ax0.plot(x, tdnv, label=label, marker=marker, markevery=[k], ms=4.2,
                 markeredgecolor="white", markeredgewidth=0.5, **kw)
        layers, succ = steer_curve(s, max_invalid)
        ax1.plot(layers / L, succ, marker=marker, ms=3.0, markeredgecolor="white",
                 markeredgewidth=0.4, **kw)
        ax1.axvline((k + 1) / L, color=shade[m], lw=0.5, ls=(0, (2, 2)), alpha=0.5, zorder=1)
    log_axis(ax0)
    ax0.set_ylabel("TDNV")
    ax0.set_title(f"(a) TDNV, {CONCEPT_LABEL[concept]}", loc="left", fontweight="bold")
    ax1.set_ylim(-3, 103)
    ax1.set_ylabel("Steering Success (%)")
    ax1.set_xlabel("Relative Depth (Layer / L)")
    ax1.set_xlim(0, 1.02)
    ax1.set_title("(b) Additive Steering", loc="left", fontweight="bold")
    fig.legend(loc="outside right center", ncol=1, fontsize=6.5, handlelength=1.4,
               handletextpad=0.4, labelspacing=0.2, markerscale=0.8)
    fig.savefig(path)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=str(ROOT / "outputs"))
    ap.add_argument("--concept", default="truth_cities")
    ap.add_argument("--max-invalid", type=float, default=0.1)
    ap.add_argument("--preview", default=None, help="also write a PNG preview here (not figs/)")
    args = ap.parse_args()
    path = ROOT / "figs" / f"steer_{args.concept}.pdf"
    fig = fig_steer(Path(args.outputs), args.concept, args.max_invalid, path)
    if args.preview:
        Path(args.preview).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(args.preview) / f"steer_{args.concept}.png", dpi=200)
    print("wrote", path)


if __name__ == "__main__":
    main()
