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


def test_bcubed():
    assert cl.bcubed([0, 0, 1, 1], [5, 5, 7, 7])['f'] == 1.0
    r = cl.bcubed([0, 0, 1, 1], [3, 3, 3, 3])
    assert np.isclose(r['precision'], 0.5) and r['recall'] == 1.0
    assert cl.bcubed([0, 0, -1], [1, 1, -1])['f'] == 1.0


def test_bcubed_noise_and_distractors_are_singletons():
    # all points called noise: every pred cluster is a singleton -> precision 1, recall low
    r = cl.bcubed([0, 0, 1, 1], [-1, -1, -1, -1])
    assert r['precision'] == 1.0 and np.isclose(r['recall'], 0.5)
    # distractors merged into one pred cluster hurt precision
    r = cl.bcubed([0, 0, -1, -1], [1, 1, 2, 2])
    assert np.isclose(r['precision'], (1 + 1 + 0.5 + 0.5) / 4) and r['recall'] == 1.0


def _pca_reduce(monkeypatch):
    real = cl.reduce_features
    monkeypatch.setattr(cl, 'reduce_features',
                        lambda feats, method='umap', *a, **k: real(feats, 'pca' if method == 'umap' else method, *a, **k))


def test_mixed_pool_clustering_keys_and_separable(monkeypatch):
    _pca_reduce(monkeypatch)
    f, l = blobs(n_cls=12, per=10)
    r = cl.mixed_pool_clustering(f, l, n_pools=3, n_artists=5, max_per_artist=8, n_distractors=20, seed=0,
                                 mcs_range=range(3, 6))
    for mode in ('umap', 'none'):
        m = r[mode]
        assert len(m['bcubed_f']) == 3 and len(m['ari']) == 3
        assert m['bcubed_f_mean'] == pytest.approx(np.mean(m['bcubed_f']))
        assert m['ari_std'] == pytest.approx(np.std(m['ari']))
        assert len(m['best_mcs']) == 3
    assert r['none']['bcubed_f_mean'] > 0.8
    assert r['n_artists'] == 5 and r['n_pools'] == 3


def test_mixed_pool_clustering_clamps_small_sets(monkeypatch):
    _pca_reduce(monkeypatch)
    f, l = blobs(n_cls=6, per=6)
    r = cl.mixed_pool_clustering(f, l, n_pools=2, n_artists=100, max_per_artist=40, n_distractors=1000,
                                 mcs_range=range(2, 4))
    assert r['n_artists'] == 3          # A // 2
    assert r['n_distractors'] == 3 * 6  # remaining artists' images only
    assert np.isfinite(r['none']['bcubed_f_mean'])


@pytest.mark.filterwarnings("ignore:n_jobs value 1 overridden")
def test_mixed_pool_clustering_deterministic_and_real_umap():
    f, l = blobs(n_cls=8, per=8)
    kw = dict(n_pools=1, n_artists=4, max_per_artist=8, n_distractors=12, seed=5, mcs_range=range(3, 5))
    a = cl.mixed_pool_clustering(f, l, **kw)
    b = cl.mixed_pool_clustering(f, l, **kw)
    assert a['umap']['bcubed_f'] == b['umap']['bcubed_f']
    assert a['umap']['reduce'] in ('umap', 'pca')
