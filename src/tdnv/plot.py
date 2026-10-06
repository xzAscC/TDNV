from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_report(report: dict, path, title: str = "") -> None:
    layers = np.array(report["layer"])
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))

    ax[0].plot(layers, report["tdnv"], "o-", label="positive vs negative")
    ax[0].plot(layers, report["tdnv_shuffled"], "--", color="gray", label="shuffled labels")
    ax[0].axhline(1.0, color="k", lw=0.6, ls=":")
    ax[0].set_yscale("log")
    ax[0].set(xlabel="layer", ylabel="TDNV", title="TDNV")
    ax[0].legend(fontsize=8)

    var = np.array(report["within_var"])
    ax[1].plot(layers, var.sum(1), "o-", label=r"$\mathrm{var}_0+\mathrm{var}_1$")
    ax[1].plot(layers, report["between_dist2"], "s-", label=r"$\|\mu_0-\mu_1\|^2$")
    ax[1].set_yscale("log")
    ax[1].set(xlabel="layer", title="numerator / denominator")
    ax[1].legend(fontsize=8)

    if report["probe_acc"][0] is not None:
        ax[2].plot(layers, report["probe_acc"], "o-")
        ax[2].set(xlabel="layer", ylabel="held-out acc", title="mean-diff direction", ylim=(0.4, 1.01))

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
