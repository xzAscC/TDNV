"""Layerwise correlation between -log TDNV and steering success: 10 steered layers vs every layer.

    uv run python notebooks/steer_corr.py [--max-invalid 0.1]

Reads TDNV from outputs/<model>/<concept>/metrics.json, the 10-layer steering runs from
outputs/<model>/steer_<concept>/metrics.json and the every-layer runs (run_steering.py --n-layers 0)
from outputs/steer_all/<model>/steer_<concept>/metrics.json. Steering success at a layer is the best
success over the alphas whose invalid rate is at most --max-invalid (plot_steering.steer_curve).
Per (model, concept) and layer set: Spearman, Pearson and Kendall tau-b between -log TDNV and
success, and the relative-depth gap between the min-TDNV layer and the best steering layer.
Writes results/steer_corr.csv and prints medians over models per concept and per family.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from figstyle import MODELS  # noqa: E402
from plot_steer_grid import SHORT, spearmanr  # noqa: E402
from plot_steering import steer_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def kendall_tau_b(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    da = np.sign(a[:, None] - a[None, :])
    db = np.sign(b[:, None] - b[None, :])
    iu = np.triu_indices(len(a), 1)
    da, db = da[iu], db[iu]
    denom = np.sqrt((da != 0).sum() * (db != 0).sum())
    return float((da * db).sum() / denom) if denom else np.nan


def stats(tdnv: np.ndarray, s: dict, max_invalid: float, only=None) -> dict:
    layers, succ = steer_curve(s, max_invalid)
    if only is not None:
        keep = np.isin(layers, only)
        layers, succ = layers[keep], succ[keep]
    L = len(tdnv) - 1
    x = -np.log(tdnv[layers])
    flat = np.ptp(succ) == 0
    k = 1 + int(np.nanargmin(tdnv[1:]))
    return dict(
        n=len(layers),
        spearman=np.nan if flat else spearmanr(x, succ),
        pearson=np.nan if flat else float(np.corrcoef(x, succ)[0, 1]),
        kendall=np.nan if flat else kendall_tau_b(x, succ),
        gap=abs(k - int(layers[np.argmax(succ)])) / L,
        peak=float(succ.max()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=str(ROOT / "outputs"))
    ap.add_argument("--max-invalid", type=float, default=0.1)
    args = ap.parse_args()
    out = Path(args.outputs)
    rows = []
    for m in MODELS:
        d = m.replace("/", "__")
        for c in SHORT:
            t = out / d / c / "metrics.json"
            s10 = out / d / f"steer_{c}" / "metrics.json"
            sall = out / "steer_all" / d / f"steer_{c}" / "metrics.json"
            if not (t.exists() and s10.exists() and sall.exists()):
                continue
            tdnv = np.array(json.loads(t.read_text())["tdnv"], dtype=float)
            r10, rall = json.loads(s10.read_text()), json.loads(sall.read_text())
            for name, st in (("10", stats(tdnv, r10, args.max_invalid)),
                             ("all", stats(tdnv, rall, args.max_invalid)),
                             ("all@10", stats(tdnv, rall, args.max_invalid, only=r10["layers"]))):
                rows.append(dict(model=m, family=MODELS[m][0], concept=c, layers=name, **st))

    path = ROOT / "results" / "steer_corr.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    print("wrote", path, "|", len(rows) // 3, "(model, concept) pairs")

    def table(key, groups):
        print(f"\n{key:28s} {'n':>2s} | " + " | ".join(
            f"{m:>8s} 10 / all" for m in ("spearman", "pearson", "kendall", "gap")))
        for g in groups:
            sel = [r for r in rows if r[key] == g]
            if not sel:
                continue
            cells = []
            for metric in ("spearman", "pearson", "kendall", "gap"):
                v = [np.nanmedian([r[metric] for r in sel if r["layers"] == n]) for n in ("10", "all")]
                cells.append(f"{v[0]:+8.2f} {v[1]:+5.2f}")
            print(f"{g:28s} {len(sel) // 3:2d} | " + " | ".join(cells))

    table("concept", list(SHORT))
    table("family", list(dict.fromkeys(v[0] for v in MODELS.values())))
    print("\nmedians over (model, concept) pairs; 'all@10' rows in the csv recompute the 10 layers "
          "from the every-layer runs (should match '10')")


if __name__ == "__main__":
    main()
