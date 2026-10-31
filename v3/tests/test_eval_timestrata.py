import numpy as np
from v3.eval import timestrata as T
from v3.eval.pairwise import metrics_exact, threshold_drift

def _toy():
    f = np.eye(4)[[0, 0, 0, 1, 1, 2, 2]] + 0.01 * np.random.default_rng(0).standard_normal((7, 4))
    lab = np.array([0, 0, 0, 1, 1, 2, 2]); t = np.array([2010, 2010.5, 2018, 2012, 2013.5, 2020, 2020.2])
    return f, lab, t

def test_counts_per_stratum():
    f, lab, t = _toy()
    h = T.artist_histograms(f, lab, t, np.arange(7), scale=10, bias=-5)
    P = T.POS_GROUPS
    assert h.pos[0, P.index('<1y')].sum() == 1 and h.pos[0, P.index('>=6y')].sum() == 2
    assert h.pos[1, P.index('1-3y')].sum() == 1 and h.pos[:, P.index('all')].sum() == 5
    assert h.neg[:, 0].sum() == 2 * (21 - 5)

def test_same_family_excluded():
    f, lab, t = _toy()
    fam = np.array([0, 0, 2, 3, 4, 5, 6])
    h = T.artist_histograms(f, lab, t, fam)
    assert h.pos[0, T.POS_GROUPS.index('<1y')].sum() == 0

def test_binned_matches_exact():
    rng = np.random.default_rng(1)
    lab = np.repeat(np.arange(10), 30); f = rng.standard_normal((300, 16)) + 3 * rng.standard_normal((10, 16))[lab]
    h = T.artist_histograms(f, lab, np.full(300, 2020.0), np.arange(300), scale=10, bias=-5)
    z = f / np.linalg.norm(f, axis=1, keepdims=True); iu = np.triu_indices(300, 1)
    ex = metrics_exact(10 * (z @ z.T)[iu] - 5, lab[iu[0]] == lab[iu[1]])
    assert abs(T.stratified_metrics(h)['all']['pr_auc'] - ex['pr_auc']) < 2e-3

def test_empty_stratum_is_none():
    f, lab, t = _toy()
    t = np.full(7, 2020.0)
    assert T.stratified_metrics(T.artist_histograms(f, lab, t, np.arange(7)))['>=6y'] is None

def test_content_groups():
    f, lab, t = _toy()
    chars = T.tag_ids([['a'], ['b'], ['a'], ['a'], [], [], []])
    h = T.artist_histograms(f, lab, t, np.arange(7), chars=chars, copys=T.tag_ids([[]] * 7))
    assert h.neg[:, T.NEG_GROUPS.index('same_char')].sum() == 2 * 2      # (0,3), (2,3)
    assert h.pos[0, T.POS_GROUPS.index('cross_content')].sum() == 2      # (0,1), (1,2)

def test_rows_cols_cross_source():
    f, lab, t = _toy()
    q = np.array([1, 0, 0, 1, 0, 0, 0], bool)
    h = T.artist_histograms(f, lab, t, np.arange(7), rows=q, cols=~q)
    assert h.pos[:, T.POS_GROUPS.index('all')].sum() == 2 + 1          # img0->{1,2}, img3->{4}


# ---- additional coverage -------------------------------------------------------------------------------------------
def test_tag_ids_frequency_order_and_padding():
    ids = T.tag_ids([['x', 'y', 'original'], ['x'], ['x', 'z', 'y', 'w', 'v'], []], k=3)
    assert ids.shape == (4, 3) and ids.dtype == np.int32
    assert ids[1, 0] == ids[0, 0] == ids[2, 0] and (ids[1, 1:] == -1).all()   # 'x' most frequent -> id 0
    assert ids[0, 0] == 0 and (ids[3] == -1).all()
    assert len(set(ids[0, :2])) == 2 and ids[0, 2] == -1                      # 'original' dropped
    assert (ids[2] >= 0).all()                                                # capped at k=3
    assert T.tag_ids([[], []]).shape == (2, 4) and (T.tag_ids([[], []]) == -1).all()

def test_same_copy_and_multi_tag_overlap_counts_once():
    f, lab, t = _toy()
    # img0, img1, img3 all carry the same two char tags; img3 belongs to artist 1 -> neg pairs (0,3), (1,3),
    # each counted once per side even though two tags overlap
    chars = T.tag_ids([['a', 'b'], ['a', 'b'], [], ['a', 'b'], [], [], []])
    copys = T.tag_ids([[], [], ['c'], ['c'], [], [], []])                   # (2,3): artists 0 / 1
    h = T.artist_histograms(f, lab, t, np.arange(7), chars=chars, copys=copys)
    sc, sk = T.NEG_GROUPS.index('same_char'), T.NEG_GROUPS.index('same_copy')
    assert h.neg[:, sc].sum() == 2 * 2                                      # (0,3), (1,3)
    assert h.neg[0, sc].sum() == 2 and h.neg[1, sc].sum() == 2
    assert h.neg[:, sk].sum() == 2 and h.neg[0, sk].sum() == 1 and h.neg[1, sk].sum() == 1
    # (0,1) share chars -> not cross; (0,2), (1,2) share nothing -> cross
    assert h.pos[0, T.POS_GROUPS.index('cross_content')].sum() == 2

def test_chunking_invariant():
    rng = np.random.default_rng(2)
    lab = np.repeat(np.arange(8), 12); f = rng.standard_normal((96, 8)); t = rng.uniform(2000, 2020, 96)
    fam = np.arange(96) // 2 * 1
    chars = T.tag_ids([[f'c{i % 5}'] for i in range(96)]); copys = T.tag_ids([[f'k{i % 7}'] for i in range(96)])
    a = T.artist_histograms(f, lab, t, fam, chars=chars, copys=copys, chunk=7)
    b = T.artist_histograms(f, lab, t, fam, chars=chars, copys=copys, chunk=4096)
    assert (a.pos == b.pos).all() and (a.neg == b.neg).all()

def test_float16_and_nan_time():
    f, lab, t = _toy(); t = t.copy(); t[1] = np.nan
    h = T.artist_histograms(f.astype(np.float16), lab, t, np.arange(7))
    P = T.POS_GROUPS
    assert h.pos[0, P.index('all')].sum() == 3                                # NaN-date pairs: 'all' only
    assert sum(h.pos[0, P.index(s)].sum() for s in T.STRATA) == 1

def test_curve_metrics_none_and_keys():
    z = np.zeros(T.N_BINS)
    assert T.curve_metrics(z, z + 1, 1, 0) is None and T.curve_metrics(z + 1, z, 1, 0) is None
    pos = np.zeros(T.N_BINS); neg = np.zeros(T.N_BINS); pos[12000] = 5; neg[4000] = 100; neg[11000] = 2
    m = T.curve_metrics(pos, neg, 10.0, -5.0)
    assert m['best_f1'] > 0.99 and 'recall@fpr0.001' in m and 'recall@fpr0.0001' in m
    assert m['recall@fpr0.001'] == 1.0                                        # all pos above all neg
    assert abs(m['best_threshold'] - (10 * (-1 + 2 * 12000 / T.N_BINS) - 5)) < 1e-9
    assert not any(isinstance(v, float) and np.isnan(v) for v in m.values())

def test_curve_compatible_with_threshold_drift():
    f, lab, t = _toy()
    h = T.artist_histograms(f, lab, t, np.arange(7), scale=10, bias=-5)
    s = T.stratified_metrics(h)
    d = threshold_drift({'a': s['all'], 'b': s['all']})
    assert d['spread'] == 0.0 and 'max_cross_f1_loss' in d

def test_weights_select_artists_and_zero_weights():
    f, lab, t = _toy()
    h = T.artist_histograms(f, lab, t, np.arange(7))
    only0 = T.stratified_metrics(h, np.array([1.0, 0, 0]))
    assert only0['>=6y'] is not None and only0['1-3y'] is None            # artist 1's only pair is 1-3y
    assert T.stratified_metrics(h, np.zeros(3))['all'] is None
    ones = T.stratified_metrics(h)['all']; w1 = T.stratified_metrics(h, np.ones(3))['all']
    assert ones['pr_auc'] == w1['pr_auc']
    twice = T.stratified_metrics(h, 2 * np.ones(3))['all']
    assert abs(twice['pr_auc'] - ones['pr_auc']) < 1e-12

def test_top_level_content_metrics():
    rng = np.random.default_rng(3)
    n = 120; lab = np.repeat(np.arange(6), 20); f = rng.standard_normal((n, 8)) + 2 * rng.standard_normal((6, 8))[lab]
    chars = T.tag_ids([[f'c{i % 4}'] for i in range(n)]); copys = T.tag_ids([[f'k{i % 6}'] for i in range(n)])
    h = T.artist_histograms(f, lab, np.full(n, 2020.0), np.arange(n), chars=chars, copys=copys, scale=10, bias=-5)
    s = T.stratified_metrics(h)
    for k in ('cross_content_recall', 'same_char_fpr', 'same_copy_fpr'):
        assert 0.0 <= s[k] <= 1.0
    # direct check of same_char_fpr
    z = f / np.linalg.norm(f, axis=1, keepdims=True); iu = np.triu_indices(n, 1)
    sc = 10 * (z @ z.T)[iu] - 5
    m = (lab[iu[0]] != lab[iu[1]]) & (chars[iu[0], 0] == chars[iu[1], 0])
    assert abs(s['same_char_fpr'] - (sc[m] >= s['all']['best_threshold']).mean()) < 0.02
    # no content info -> content metrics None
    h2 = T.artist_histograms(f, lab, np.full(n, 2020.0), np.arange(n))
    s2 = T.stratified_metrics(h2)
    assert s2['cross_content'] is None and s2['cross_content_recall'] is None and s2['same_char_fpr'] is None

def test_drift_spearman():
    rng = np.random.default_rng(4)
    lab = np.repeat([0, 1, 2], [20, 20, 5]); t = np.r_[np.linspace(2000, 2010, 20), np.linspace(2000, 2010, 20),
                                                              np.linspace(2000, 2010, 5)]
    # artist 0: similarity decays with time (drift in one direction); artist 1: random; artist 2: too few images
    d = np.zeros((45, 6)); d[:, 0] = 1
    ang = (t - 2000) / 10 * 1.2
    d[:20, 1] = np.sin(ang[:20]); d[:20, 0] = np.cos(ang[:20])
    d[20:40] = rng.standard_normal((20, 6))
    out = T.drift_spearman(d, lab, t, np.arange(45))
    assert out.shape == (2,) and out[0] < -0.95 and not np.isnan(out).any()
    assert T.drift_spearman(d, lab, t, np.arange(45), min_images=30).shape == (0,)
    assert T.drift_spearman(d, lab, np.full(45, 2000.0), np.arange(45)).shape == (0,)
    # same-family pairs are dropped: all one family -> nothing left
    assert T.drift_spearman(d, lab, t, np.zeros(45, int)).shape == (0,)


def test_fractional_weights_below_one_are_scale_invariant():
    rng = np.random.default_rng(1)
    lab = np.repeat(np.arange(10), 30); f = rng.standard_normal((300, 16)) + 3 * rng.standard_normal((10, 16))[lab]
    h = T.artist_histograms(f, lab, np.full(300, 2020.0), np.arange(300), scale=10, bias=-5)
    ref = T.stratified_metrics(h)['all']
    for c in (1e-4, 1e-9, 1e4):
        m = T.stratified_metrics(h, np.full(10, c))['all']
        for k in ('pr_auc', 'ap', 'roc_auc', 'best_f1', 'recall_at_best', 'precision_at_best', 'recall@fpr0.001',
                  'recall@fpr0.0001'):
            assert abs(m[k] - ref[k]) < 1e-9, (c, k, m[k], ref[k])
        assert m['best_threshold'] == ref['best_threshold']
        assert abs(m['n_pos_pairs'] - c * ref['n_pos_pairs']) < 1e-9 * max(1, c * ref['n_pos_pairs'])  # unscaled counts
    # curve_metrics directly: total pos weight < 1 and total neg weight >> 1
    pos = np.zeros(T.N_BINS); neg = np.zeros(T.N_BINS); pos[9000] = 0.3; neg[3000] = 500.0
    m = T.curve_metrics(pos, neg, 1.0, 0.0)
    assert m['recall_at_best'] == 1.0 and m['best_f1'] == 1.0 and m['n_pos_pairs'] == 0.3
