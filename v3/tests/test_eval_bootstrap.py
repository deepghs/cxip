import numpy as np
from v3.eval import bootstrap as BS

def test_draw_weights_are_multiplicities():
    W = BS.draw_weights(5, 20, 1)
    assert W.shape == (20, 5) and (W.sum(1) == 5).all() and (W >= 0).all()

def test_artist_bootstrap_mean():
    v = np.array([0., 1., 2., 3.])
    r = BS.artist_bootstrap(lambda w: {'m': float((w * v).sum() / w.sum())}, 4, n_boot=2000)
    assert r['m']['point'] == 1.5 and r['m']['lo'] < 1.5 < r['m']['hi'] and r['m']['n_valid'] == 2000

def test_paired_identical_is_zero():
    v = np.random.default_rng(0).random(30)
    f = lambda w: {'m': float((w * v).sum() / w.sum())}
    d = BS.paired_bootstrap(f, f, 30, n_boot=200)
    assert d['m']['point'] == 0 and d['m']['lo'] == 0 and d['m']['hi'] == 0

def test_none_handling():
    r = BS.artist_bootstrap(lambda w: {'x': None, 'y': 1.0}, 3, n_boot=10)
    assert r['x'] == dict(point=None, lo=None, hi=None, n_valid=0) and r['y']['point'] == 1.0

def test_population_weights():
    w, miss = BS.population_weights(np.array([0.5, 0.5, 10]), np.array([2020, 2020, 2020]),
                                    np.array([0.5, 10, 10, 10]), np.array([2020, 2020, 2020, 2010]))
    assert np.isclose(w.mean(), 1) and w[2] > w[0] and np.isclose(miss, 0.25)


# ---------------------------------------------------------------- Task 16 fix round 1 (ruling R22)
import pytest


def _pair_kernel(A=60, seed=3):
    """between-artist pair kernel K (0/1, artist effects + pair noise) as per-artist 2-bin histograms
    Hn[a] = (#pairs (a, a') with K = 0, # with K = 1), each pair counted once on each side"""
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(A)
    K = ((u[:, None] + u[None, :] + 0.5 * rng.standard_normal((A, A))) > 1).astype(float)
    K = np.triu(K, 1)
    K = K + K.T
    off = 1 - np.eye(A)
    return K, off, np.stack([(off * (1 - K)).sum(1), (off * K).sum(1)], 1)


def test_pair_draw_coef_is_linearised_product_weight():
    W = np.array([[0, 2, 1], [1, 1, 1], [3, 0, 0]])
    base = np.array([1.0, 0.5, 2.0])
    C = BS.pair_draw_coef(W, base)
    assert np.array_equal(C, (2 * W - 1) * base)
    assert np.array_equal(BS.pair_draw_coef(np.ones((1, 3)), base)[0], base)        # W = 1 -> base exactly
    # base = 1: pair (a, a') gets (k_a + k_a' - 1) (one-sided count = C @ Hn / 2)
    Hn = np.array([[1.0], [1.0], [0.0]])                                             # one pair (0, 1)
    assert (BS.pair_draw_coef(W) @ Hn / 2)[:, 0].tolist() == [1.0, 1.0, 2.0]


def test_pair_kernel_bootstrap_sd_matches_product_resample_and_doubles_linear():
    K, off, Hn = _pair_kernel()
    A = len(Hn)
    W = BS.draw_weights(A, 2000, 0).astype(np.float64)
    new = BS.pair_draw_hists(W, Hn)
    s_new = new[:, 1] / new.sum(1)
    old = W @ Hn                                                     # pre-R22 linear weight (k_a + k_a') / 2
    s_old = old[:, 1] / old.sum(1)
    P = W[:, :, None] * W[:, None, :] * off                          # brute force: true artist resample k_a k_a'
    s_bf = (P * K).sum((1, 2)) / P.sum((1, 2))
    assert s_new.std() == pytest.approx(s_bf.std(), rel=0.10)
    assert s_new.std() / s_old.std() == pytest.approx(2.0, rel=0.10)
    pt = Hn[:, 1].sum() / Hn.sum()
    assert np.mean(s_new) == pytest.approx(pt, abs=3 * s_new.std() / np.sqrt(2000))


def test_pair_draw_hists_clip_negative_bins_and_curves_stay_finite():
    from v3.eval import timestrata as T
    H = np.zeros((3, T.N_BINS))
    H[0, 9000] = 50.0                      # bin only artist 0 contributes to
    H[:, 8000] = 10.0
    W = np.array([[0, 2, 1], [1, 1, 1]], dtype=np.float64)
    D = BS.pair_draw_hists(W, H)
    assert D[0, 9000] == 0.0 and (D >= 0).all()                      # (2*0 - 1) * 50 < 0 -> clipped
    assert np.array_equal(D[1], H.sum(0))
    pos = np.zeros(T.N_BINS)
    pos[[8500, 9500]] = 3.0
    cm = T.curve_metrics(pos, D[0] / 2, 1.0, 0.0)
    assert np.isfinite(cm['pr_auc']) and np.isfinite(cm['best_f1'])
    assert np.all(np.diff(cm['curve']['recall']) >= 0)


def test_paired_diff_shared_helper():
    pa, pb = {'x': 1.0, 'y': None}, {'x': 1.5, 'y': 2.0}
    da = {'x': np.array([1.0, 2.0, np.nan, 0.0]), 'y': np.zeros(4)}
    db = {'x': np.array([2.0, 1.0, 3.0, 0.5]), 'y': np.ones(4)}
    r = BS.paired_diff(pa, da, pb, db)
    assert r['x']['point'] == 0.5 and r['x']['n_valid'] == 3 and r['x']['p_gt0'] == pytest.approx(2 / 3)
    assert r['y']['point'] is None and r['y']['n_valid'] == 4
    r = BS.paired_diff({'v': 1.0}, {'v': None}, {'v': 2.0}, {'v': np.zeros(3)})
    assert r['v'] == dict(point=1.0, lo=None, hi=None, n_valid=0, p_gt0=None)
