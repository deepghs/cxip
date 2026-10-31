import numpy as np, pandas as pd
from v3.eval import inject as I

def _df():
    rng = np.random.default_rng(0)
    n = 40
    t = np.r_[np.linspace(2010, 2011.5, 14), np.linspace(2017.5, 2020, 6), np.linspace(2012, 2014, 20)]
    return pd.DataFrame(dict(sample_key=[f'd{i}' for i in range(n)], label=np.repeat([0, 1], 20), t_year=t,
                             family=np.arange(n), date_day=(t * 365).astype(int),
                             characters=[['c1']] * 10 + [[]] * 10 + [['c1']] * 20, copyrights=[[]] * n))

def test_far_period_and_bad_date_rules():
    df = _df()
    fp = I.make_folders(df, 'far_period')
    win = fp[~fp.injected]; inj = fp[fp.injected]
    t = df.set_index('sample_key').t_year
    assert (t[win.sample_key].max() - t[win.sample_key].min()) <= 2
    assert len(inj) >= 1 and (abs(t[inj.sample_key].values - t[win.sample_key].mean()) >= 6).all()
    bd = I.make_folders(df, 'bad_date')
    m = bd[bd.injected]
    assert set(bd.folder) == {0}                                     # artist 1 spans 2y
    assert (abs(m.t_claimed.values - t[m.sample_key].values) >= 4).all()

def test_same_char_injects_other_artists():
    df = _df()
    sc = I.make_folders(df, 'same_char')
    lab = df.set_index('sample_key').label
    assert (lab[sc[sc.injected & (sc.folder == 0)].sample_key] == 1).all()

def test_auroc_perfect_when_outliers_far():
    f = np.r_[np.tile([1., 0], (19, 1)), [[0., 1]]] + 1e-3 * np.random.default_rng(0).standard_normal((20, 2))
    folders = pd.DataFrame(dict(folder=0, sample_key=[f'd{i}' for i in range(20)], injected=[False] * 19 + [True]))
    s = I.knn_outlier_scores(f, folders, {f'd{i}': i for i in range(20)}, k=5)
    assert I.folder_auroc(s, folders)[0] == 1.0

# ---- extra coverage ----

def test_contract_columns_and_clean_rows():
    df = _df()
    for kind in ('same_char', 'far_period', 'bad_date'):
        f = I.make_folders(df, kind)
        assert list(f.columns) == ['folder', 'sample_key', 'injected', 't_claimed']
        assert f.injected.dtype == bool
        clean = f[~f.injected].merge(df[['sample_key', 't_year']], on='sample_key')
        assert np.allclose(clean.t_claimed, clean.t_year)
        assert not f.duplicated(['folder', 'sample_key']).any()


def test_injection_count_and_determinism():
    df = _df()
    f1 = I.make_folders(df, 'same_char', frac=0.1, seed=3)
    f2 = I.make_folders(df, 'same_char', frac=0.1, seed=3)
    pd.testing.assert_frame_equal(f1, f2)
    for fid, g in f1.groupby('folder'):
        n_clean = int((~g.injected).sum())
        assert n_clean == 20
        assert g.injected.sum() == max(1, int(np.floor(0.1 * n_clean + 0.5)))


def test_same_char_skips_artist_without_character():
    df = _df()
    df = pd.concat([df, df.assign(sample_key=[f'e{i}' for i in range(40)], label=2, family=np.arange(40) + 100)],
                   ignore_index=True)
    df['characters'] = [[]] * 20 + [['c1']] * 20 + [['c1']] * 40      # artist 0 has no character tag
    sc = I.make_folders(df, 'same_char')
    assert set(sc.folder) == {1, 2}
    lab = df.set_index('sample_key').label
    for fid in (1, 2):
        inj = sc[sc.injected & (sc.folder == fid)]
        assert len(inj) >= 1 and (lab[inj.sample_key] != fid).all()


def test_same_char_no_candidates_skips_folder():
    df = _df()
    df['characters'] = [['c1']] * 20 + [['c2']] * 20                   # nobody else has the character
    assert len(I.make_folders(df, 'same_char')) == 0


def test_near_style_picks_nearest_artist_one_image_each():
    rng = np.random.default_rng(0)
    cent = np.array([[1., 0, 0], [0.9, 0.3, 0], [0, 0, 1.], [0, 1., 0]])
    feats = np.concatenate([c + 0.02 * rng.standard_normal((20, 3)) for c in cent]).astype(np.float32)
    n = len(feats)
    df = pd.DataFrame(dict(sample_key=[f'k{i}' for i in range(n)], label=np.repeat(np.arange(4), 20),
                           t_year=np.full(n, 2015.0), family=np.arange(n), date_day=np.arange(n),
                           characters=[[]] * n, copyrights=[[]] * n))
    f = I.make_folders(df, 'near_style', frac=0.1, feats=feats)       # 2 injections per folder
    lab = df.set_index('sample_key').label
    g = f[f.folder == 0]
    inj = g[g.injected]
    assert len(g[~g.injected]) == 20 and len(inj) == 2
    ls = lab[inj.sample_key].values
    assert len(set(ls)) == 2 and 0 not in ls and 1 in ls              # one image per other artist, nearest artist included
    z = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    c = z[:20].mean(0); c /= np.linalg.norm(c)
    best1 = 20 + int(np.argmax(z[20:40] @ c))
    assert df.sample_key[best1] in set(inj.sample_key)


def test_bad_date_counts_and_range():
    df = _df()
    df.loc[df.label == 1, 't_year'] = np.linspace(2010, 2018, 20)        # now both span >= 6
    df['date_day'] = (df.t_year * 365).astype(int)
    bd = I.make_folders(df, 'bad_date', frac=0.2, seed=1)
    t = df.set_index('sample_key').t_year
    for fid, g in bd.groupby('folder'):
        tt = df[df.label == fid].t_year
        inj = g[g.injected]
        assert len(inj) == 4
        assert (inj.t_claimed >= tt.min()).all() and (inj.t_claimed <= tt.max()).all()
        assert (abs(inj.t_claimed.values - t[inj.sample_key].values) >= 4).all()
    assert 'stats' in bd.attrs


def test_auroc_nan_handling_and_order():
    folders = pd.DataFrame(dict(folder=[5] * 4 + [2] * 4 + [9] * 4 + [7] * 4,
                                sample_key=[f'x{i}' for i in range(16)],
                                injected=[False, False, False, True,   # 5: perfect
                                          False, False, False, False,  # 2: no injected -> NaN
                                          False, False, False, True,   # 9: injected score NaN -> NaN
                                          False, True, False, False]))  # 7: ties -> 0.5 mixed
    s = np.array([0, 0, 0, 1.,  0, 1, 2, 3.,  0, 1, 2, np.nan,  1, 1, 1, 1.])
    auc = I.folder_auroc(s, folders)
    ids, auc2 = I.folder_auroc(s, folders, return_ids=True)
    assert list(ids) == [2, 5, 7, 9] == list(I.folder_ids(folders))
    assert np.array_equal(auc, auc2, equal_nan=True)
    assert np.isnan(auc[0]) and auc[1] == 1.0 and auc[2] == 0.5 and np.isnan(auc[3])


def test_knn_scores_tiny_folder_nan():
    folders = pd.DataFrame(dict(folder=[0], sample_key=['a'], injected=[False]))
    s = I.knn_outlier_scores(np.ones((1, 2)), folders, {'a': 0})
    assert np.isnan(s).all()


def test_date_conflict_flags_bad_dates_and_small_folder_nan():
    rng = np.random.default_rng(0)
    n = 60
    t = np.linspace(2010, 2020, n)
    f = np.stack([np.cos((t - 2010) / 10 * 1.5), np.sin((t - 2010) / 10 * 1.5)], 1) + 0.01 * rng.standard_normal((n, 2))
    meta = pd.DataFrame(dict(sample_key=[f'm{i}' for i in range(n)], t_year=t, family=np.arange(n),
                             date_day=np.arange(n)))
    df = meta.assign(label=0, characters=[[]] * n, copyrights=[[]] * n)
    bd = I.make_folders(df, 'bad_date', frac=0.1, seed=0)
    s = I.date_conflict_scores(f, bd, meta)
    auc = I.folder_auroc(s, bd)
    assert auc[0] > 0.9
    small = pd.DataFrame(dict(folder=0, sample_key=meta.sample_key[:5], injected=[False] * 4 + [True],
                              t_claimed=t[:5]))
    ss = I.date_conflict_scores(f, small, meta)
    assert np.isnan(ss).all()
    assert np.isnan(I.folder_auroc(ss, small)).all()
