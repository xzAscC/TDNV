"""Layerwise Task-Distance Normalized Variance (TDNV) and related statistics.

TDNV^(l) = sum_{t != t'} (var_t + var_t') / (2 * ||mu_t - mu_t'||^2)
with var_t = mean_i ||h_{i,t} - mu_t||^2  (Eq. in Sec. 3 of the paper).
For two classes this reduces to (var_0 + var_1) / ||mu_0 - mu_1||^2.
"""

from __future__ import annotations

import numpy as np
import torch


def class_stats(h: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-class means [T, d] and within-class variances [T] for features h [N, d]."""
    classes = torch.unique(labels)
    means = torch.stack([h[labels == c].mean(0) for c in classes])
    variances = torch.stack(
        [((h[labels == c] - means[i]) ** 2).sum(-1).mean() for i, c in enumerate(classes)]
    )
    return means, variances


def tdnv(h: torch.Tensor, labels: torch.Tensor) -> dict:
    """TDNV for one layer. h: [N, d], labels: [N] integer class ids."""
    h = h.double()
    means, variances = class_stats(h, labels)
    dist2 = torch.cdist(means, means) ** 2
    num = variances[:, None] + variances[None, :]
    off = ~torch.eye(len(means), dtype=torch.bool, device=h.device)
    value = (num[off] / (2 * dist2[off])).sum()
    return {
        "tdnv": value.item(),
        "within_var": variances.tolist(),
        "between_dist2": dist2[off].tolist(),
    }


def mean_diff_probe_acc(
    h: torch.Tensor, labels: torch.Tensor, train_mask: torch.Tensor
) -> float:
    """Held-out accuracy of the difference-of-means direction (binary labels 0/1).

    The direction and midpoint threshold are fit on train_mask and evaluated on the rest.
    This is the same direction used by difference-of-means steering.
    """
    h = h.double()
    tr, te = train_mask, ~train_mask
    mu1 = h[tr & (labels == 1)].mean(0)
    mu0 = h[tr & (labels == 0)].mean(0)
    w = mu1 - mu0
    b = (w @ (mu1 + mu0)) / 2
    pred = (h[te] @ w > b).long()
    return (pred == labels[te]).double().mean().item()


def layerwise_report(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    n_shuffles: int = 5,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> dict:
    """Compute per-layer TDNV, a shuffled-label baseline, and probe accuracy.

    hidden: [N, L+1, d] last-token states (index 0 = embeddings).
    """
    g = torch.Generator().manual_seed(seed)
    labels = labels.long()
    n = len(labels)
    train_mask = torch.zeros(n, dtype=torch.bool)
    for c in torch.unique(labels):
        idx = torch.nonzero(labels == c).squeeze(1)
        idx = idx[torch.randperm(len(idx), generator=g)]
        train_mask[idx[: len(idx) // 2]] = True
    perms = [torch.randperm(n, generator=g) for _ in range(n_shuffles)]

    labels_d, train_d = labels.to(device), train_mask.to(device)
    out = {"layer": [], "tdnv": [], "tdnv_shuffled": [], "within_var": [],
           "between_dist2": [], "probe_acc": []}
    for layer in range(hidden.shape[1]):
        h = hidden[:, layer].to(device)
        r = tdnv(h, labels_d)
        shuffled = [tdnv(h, labels_d[p.to(device)])["tdnv"] for p in perms]
        out["layer"].append(layer)
        out["tdnv"].append(r["tdnv"])
        out["tdnv_shuffled"].append(float(np.mean(shuffled)))
        out["within_var"].append(r["within_var"])
        out["between_dist2"].append(r["between_dist2"][0])
        out["probe_acc"].append(
            mean_diff_probe_acc(h, labels_d, train_d) if len(torch.unique(labels)) == 2 else None
        )
    return out
