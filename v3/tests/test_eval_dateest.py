import numpy as np
from v3.eval import dateest as DE


def _toy():
    early, late = np.array([1., 0, 0]), np.array([0., 1, 0])
    f = np.stack([early, early * 1.01, early * 0.99, late, late * 1.01, late * 0.99]) + 1e-3
    t = np.array([2010.0, 2010.2, 2010.4, 2018.0, 2018.2, 2018.4])
    return f, np.zeros(6, int), t, np.arange(6), np.arange(6)


def test_style_dates_uses_similar_images():
    f, lab, t, day, fam = _toy()
    est = DE.style_dates(f, lab, t, day, fam, k=2, min_candidates=2)
    assert abs(est[0] - 2010.3) < 0.15 and abs(est[3] - 2018.3) < 0.15


def test_same_day_and_family_excluded():
    f, lab, t, day, fam = _toy()
    day = np.array([0, 0, 0, 1, 2, 3])
    assert np.isnan(DE.style_dates(f, lab, t, day, fam, k=2, min_candidates=4)[0])   # only 3 candidates left
    est = DE.style_dates(f, lab, t, day, fam, k=2, min_candidates=2)
    assert est[0] > 2015                                                                # same-day early images excluded


def test_block_and_query_gallery():
    f, lab, t, day, fam = _toy()
    block = np.array([[0], [0], [0], [-1], [-1], [-1]])
    assert DE.style_dates(f, lab, t, day, fam, k=2, min_candidates=2, block=block)[0] > 2015
    q = np.array([1, 0, 0, 0, 0, 0], bool)
    est = DE.style_dates(f, lab, t, day, fam, k=2, min_candidates=2, query=q, gallery=~q)
    assert np.isnan(est[1:]).all() and abs(est[0] - 2010.3) < 0.15


def test_conflict_and_baseline():
    est = np.array([2010., 2010., 2010., 2010.]); t = np.array([2010.2, 2009.9, 2010.1, 2016.0])
    s = DE.conflict_scores(est, t, np.zeros(4, int))
    assert s.argmax() == 3
    f, lab, t, day, fam = _toy()
    assert np.allclose(DE.median_baseline(lab, t, day, fam, min_candidates=2)[0], np.median(t[1:]))


def test_family_excluded():
    f, lab, t, day, fam = _toy()
    fam = np.array([0, 0, 0, 1, 2, 3])
    est = DE.style_dates(f, lab, t, day, fam, k=2, min_candidates=2)
    assert est[0] > 2015
    assert np.isnan(DE.style_dates(f, lab, t, day, fam, k=2, min_candidates=4)[0])


def test_empty_candidates_nan_not_crash():
    f, lab, t, day, fam = _toy()
    day = np.zeros(6, int)                       # everything uploaded the same day
    est = DE.style_dates(f, lab, t, day, fam, k=3, min_candidates=1)
    assert np.isnan(est).all()
    assert np.isnan(DE.median_baseline(lab, t, day, fam, min_candidates=1)).all()
    # fewer candidates than k is fine (k clipped), multiple artists handled independently
    lab2 = np.array([0, 0, 0, 1, 1, 1])
    est = DE.style_dates(f, lab2, t, np.arange(6), fam, k=5, min_candidates=2)
    assert np.isfinite(est).all() and abs(est[0] - 2010.3) < 0.15 and abs(est[3] - 2018.3) < 0.15


def test_date_errors_and_per_artist_mean():
    est = np.array([2010., np.nan, 2012., 2013.])
    t = np.array([2011., 2011., 2011., 2011.])
    rel = np.array([True, True, False, True])
    err = DE.date_errors(est, t, rel)
    assert np.allclose(err[[0, 3]], [1., 2.]) and np.isnan(err[[1, 2]]).all()
    lab = np.array([0, 0, 1, 1])
    s, c = DE.per_artist_mean(err, lab, 3)
    assert np.allclose(s, [1., 2., 0.]) and np.allclose(c, [1, 1, 0])
    assert np.isfinite(s).all()
    # conflict_scores: NaN estimate -> NaN score, and does not poison the artist median
    sc = DE.conflict_scores(np.array([2010., np.nan, 2010., 2010.]), np.array([2010.1, 2011., 2010.3, 2015.]), np.zeros(4, int))
    assert np.isnan(sc[1]) and np.isfinite(sc[[0, 2, 3]]).all() and np.nanargmax(sc) == 3
