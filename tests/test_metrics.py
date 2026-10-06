import torch

from tdnv.metrics import layerwise_report, tdnv


def test_two_class_closed_form():
    h = torch.tensor([[0.0, 0.0], [2.0, 0.0], [10.0, 0.0], [10.0, 4.0]])
    labels = torch.tensor([0, 0, 1, 1])
    # var_0 = 1, var_1 = 4, mu_0 = (1,0), mu_1 = (10,2) -> dist2 = 85
    assert abs(tdnv(h, labels)["tdnv"] - 5 / 85) < 1e-12


def test_three_classes_matches_definition():
    g = torch.Generator().manual_seed(0)
    h = torch.randn(30, 5, generator=g).double()
    labels = torch.arange(30) % 3
    mus = [h[labels == c].mean(0) for c in range(3)]
    vs = [((h[labels == c] - mus[c]) ** 2).sum(1).mean() for c in range(3)]
    ref = sum((vs[a] + vs[b]) / (2 * ((mus[a] - mus[b]) ** 2).sum())
              for a in range(3) for b in range(3) if a != b)
    assert abs(tdnv(h, labels)["tdnv"] - ref.item()) < 1e-9


def test_separated_vs_shuffled():
    g = torch.Generator().manual_seed(0)
    n, d = 200, 16
    shift = torch.zeros(d)
    shift[0] = 8.0
    h = torch.cat([torch.randn(n, d, generator=g), torch.randn(n, d, generator=g) + shift])
    labels = torch.tensor([0] * n + [1] * n)
    hidden = torch.stack([h, torch.randn(2 * n, d, generator=g)], 1)  # layer 1 = pure noise
    r = layerwise_report(hidden, labels, n_shuffles=3)
    assert r["tdnv"][0] < 1 < r["tdnv_shuffled"][0]
    assert r["probe_acc"][0] > 0.95
    assert r["tdnv"][1] > 1 and r["probe_acc"][1] < 0.7
