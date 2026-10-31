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
