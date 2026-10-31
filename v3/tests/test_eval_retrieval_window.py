"""time-window retrieval"""
import numpy as np
import pytest

from v3.eval.retrieval import window_retrieval


def test_window_retrieval_excludes_far_same_artist():
    f = np.array([[1, 0], [1, 0.01], [0.9, 0.1], [0, 1.]])
    lab = np.array([0, 0, 0, 1]); t = np.array([2010, 2010.5, 2016, 2010])
    r = window_retrieval(f, lab, t, np.arange(4), window=1.0)
    assert r['map_at_r'][0] == 1.0 and r['r1'][0] == 1.0
    assert np.isnan(r['map_at_r'][2]) and np.isnan(r['map_at_r'][3])   # no in-window positives
    r2 = window_retrieval(f, lab, t, np.arange(4), window=None)
    assert r2['map_at_r'][2] == 1.0


def _brute(f, lab, t, fam, window, query, gallery):
    fn = f / np.linalg.norm(f, axis=1, keepdims=True)
    S = fn @ fn.T
    n = len(lab)
    ap = np.full(n, np.nan); r1 = np.full(n, np.nan)
    for i in np.nonzero(query)[0]:
        same = lab == lab[i]
        near = same if window is None else same & (np.abs(t - t[i]) < window)
        valid = gallery & (fam != fam[i]) & (~same | near)
        valid[i] = False
        rel_all = near & valid
        R = int(rel_all.sum())
        if R == 0:
            continue
        cand = np.nonzero(valid)[0]
        o = cand[np.argsort(-S[i, cand], kind='stable')][:R]
        rel = rel_all[o]
        ap[i] = (np.cumsum(rel) / np.arange(1, R + 1) * rel).sum() / R
        r1[i] = float(rel[0])
    return ap, r1


@pytest.mark.parametrize('window', [1.0, 2.5, None])
def test_window_retrieval_bruteforce_with_masks_and_family(window):
    rng = np.random.default_rng(3)
    n = 120
    lab = rng.integers(0, 8, n)
    t = 2010 + 8 * rng.random(n)
    fam = rng.integers(0, 70, n)
    f = (np.eye(8)[lab] + 0.8 * rng.standard_normal((n, 8))).astype(np.float32)
    query = rng.random(n) < 0.5
    gallery = ~query | (rng.random(n) < 0.3)
    r = window_retrieval(f, lab, t, fam, window=window, query=query, gallery=gallery, chunk=17)
    ap, r1 = _brute(f, lab, t, fam, window, query, gallery)
    np.testing.assert_allclose(r['map_at_r'], ap, atol=1e-6)
    np.testing.assert_allclose(r['r1'], r1, atol=1e-9)
    assert np.isnan(r['map_at_r'][~query]).all()
    assert np.isfinite(ap).sum() > 10
