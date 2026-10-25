"""clustering / retrieval"""
import numpy as np
import pytest
import torch

from v3.eval import clustering as cl
from v3.eval import retrieval as rt


def blobs(n_cls=20, per=12, dim=32, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(n_cls), per)
    centers = rng.standard_normal((n_cls, dim))
    return (centers[labels] + noise * rng.standard_normal((len(labels), dim))).astype(np.float32), labels


def test_nmi_matches_torchmetrics_with_noise():
    from torchmetrics.clustering import NormalizedMutualInfoScore
    rng = np.random.default_rng(1)
    yt = rng.integers(0, 10, 300)
    yp = np.where(rng.random(300) < 0.7, yt, -1)
    m = cl.cluster_metrics(yt, yp)
    ref = NormalizedMutualInfoScore()(torch.tensor(yp), torch.tensor(yt)).item()
    assert m['nmi'] == pytest.approx(ref, abs=1e-5)
    assert m['noise_frac'] == pytest.approx((yp == -1).mean())


def test_purity_and_acc():
    yt = np.array([0, 0, 0, 1, 1, 1, 2, 2])
    yp = np.array([5, 5, 7, 7, 7, 7, -1, -1])
    m = cl.cluster_metrics(yt, yp)
    assert m['purity'] == pytest.approx((2 + 3) / 6)
    assert m['n_clusters'] == 2
    assert m['acc_ignore_noise'] == pytest.approx(5 / 6)


@pytest.mark.parametrize('reduce', ['pca', 'none', 'umap'])
def test_clustering_eval_separable(reduce):
    f, l = blobs()
    r = cl.clustering_eval(f, l, reduce=reduce, mcs_range=range(2, 5), seed=0)
    assert r['best_nmi'] > 0.95 and r['purity'] > 0.95
    assert set(r['per_mcs']) == {'2', '3', '4'}


def bruteforce_map_at_r(f, l):
    fn = f / np.linalg.norm(f, axis=1, keepdims=True)
    S = fn @ fn.T
    np.fill_diagonal(S, -np.inf)
    aps, r1 = [], []
    for i in range(len(l)):
        R = (l == l[i]).sum() - 1
        if R == 0:
            continue
        o = np.argsort(-S[i], kind='stable')[:R]
        rel = l[o] == l[i]
        aps.append((np.cumsum(rel) / np.arange(1, R + 1) * rel).sum() / R)
        r1.append(rel[0])
    return np.mean(aps), np.mean(r1), len(aps)


def test_retrieval_bruteforce():
    f, l = blobs(noise=1.0, seed=2)
    l = l.copy()
    l[0] = 999         # 单张的类不作查询
    m, r1, nq = bruteforce_map_at_r(f, l)
    r = rt.retrieval_eval(f, l, chunk=37)
    assert r['map_at_r'] == pytest.approx(m, abs=1e-6)
    assert r['recall_at_1'] == pytest.approx(r1, abs=1e-9)
    assert r['n_queries'] == nq
