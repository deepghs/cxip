import numpy as np, pandas as pd, pytest
from v3.eval import bench2 as B2

def _set(n_art=12, per=10, seed=0):
    rng = np.random.default_rng(seed)
    lab = np.repeat(np.arange(n_art), per)
    t = 2010 + rng.random(len(lab)) * 10
    df = pd.DataFrame(dict(sample_key=[f'd{i}' for i in range(len(lab))], label=lab, t_year=t, family=np.arange(len(lab)),
                           date_day=(t * 365.25).astype(int), date_source='twitter', characters=[[]] * len(lab),
                           copyrights=[[]] * len(lab), split='dev', source='danbooru'))
    f = rng.standard_normal((len(lab), 8)) + 3 * rng.standard_normal((n_art, 8))[lab]
    return df, f

def test_evaluate_set_shapes_and_self_diff(tmp_path):
    df, f = _set()
    m = B2.ModelFeats('a', {'s': f}, 10.0, -5.0, 0.0)
    r = B2.evaluate_set('s', df, [m, B2.ModelFeats('b', {'s': f}, 10.0, -5.0, 0.0)], 'dev', n_boot=50, inject_dir=tmp_path,
                        clustering_kw=dict(reduces=(None,)))          # fix round 1: skip UMAP (covered in test_eval_clustering)
    x = r['a']['pr_auc/<1y']
    assert x['lo'] <= x['point'] <= x['hi']
    assert r['diff']['b-vs-a']['pr_auc/all']['point'] == 0
    assert 'date_mae' in r['a'] and 'map_at_r/1y' in r['a']
    assert r['composition']['n_artists'] == 12
    assert set(r['clustering']) == {'a', 'b'} and np.isfinite(r['clustering']['a']['none']['bcubed_f_mean'])

def test_split_filter(tmp_path):
    df, f = _set()
    df.loc[df.label < 6, 'split'] = 'test'
    r = B2.evaluate_set('s', df, [B2.ModelFeats('a', {'s': f}, 10.0, -5.0, 0.0)], 'dev', n_boot=10, inject_dir=tmp_path,
                        clustering_kw=dict(reduces=(None,)))
    assert r['composition']['n_artists'] == 6
    # fix round 1: filtered-out rows (different features) must not matter; same as evaluating the pre-filtered set
    out = (df.split == 'test').to_numpy()
    f2 = f.copy()
    f2[out] = np.random.default_rng(9).standard_normal((out.sum(), f.shape[1])) * 5
    r2 = B2.evaluate_set('s', df, [B2.ModelFeats('a', {'s': f2}, 10.0, -5.0, 0.0)], 'dev', n_boot=10,
                         inject_dir=tmp_path, clustering_kw=dict(reduces=(None,)))
    r3 = B2.evaluate_set('s', df[~out].reset_index(drop=True), [B2.ModelFeats('a', {'s': f[~out]}, 10.0, -5.0, 0.0)],
                         'dev', n_boot=10, inject_dir=tmp_path, clustering_kw=dict(reduces=(None,)))
    pts = lambda x: {k: v['point'] for k, v in x['a'].items()}
    assert pts(r) == pts(r2) == pts(r3)
    assert r2['composition'] == r3['composition']


# ---------------------------------------------------------------- additional coverage (beyond the brief's two tests)
from v3.eval import inject


def _sep_set(n_art=10, per=16, seed=0, d=16):
    """well separated artists, 10-year spans, so date / inject metrics have candidates"""
    rng = np.random.default_rng(seed)
    lab = np.repeat(np.arange(n_art), per)
    t = 2010 + np.tile(np.linspace(0, 10, per), n_art) + rng.random(len(lab)) * 0.01
    df = pd.DataFrame(dict(sample_key=[f'd{i}' for i in range(len(lab))], label=lab + 100, t_year=t,
                           family=np.arange(len(lab)), date_day=(t * 365.25).astype(int), date_source='twitter',
                           characters=[[f'c{x}'] for x in rng.integers(0, 3, len(lab))],
                           copyrights=[['original']] * len(lab), split='test', source='danbooru'))
    f = 0.3 * rng.standard_normal((len(lab), d)) + rng.standard_normal((n_art, d))[lab]
    return df, f


def test_weighted_median_matches_repeat():
    v = np.array([3.0, np.nan, 1.0, 2.0, 5.0])
    W = np.array([[1, 4, 0, 2, 1], [0, 0, 0, 0, 0], [2, 1, 1, 0, 0]])
    pt, dr = B2._wmedian_draws(v, W)
    assert pt == 2.5
    assert dr[0] == np.median(np.repeat(v, W[0])[np.isfinite(np.repeat(v, W[0]))])
    assert np.isnan(dr[1]) and dr[2] == 3.0          # [3, 3, 1]


def test_inject_present_and_missing_kinds(tmp_path):
    df, f = _sep_set()
    for kind in ('bad_date', 'same_char', 'far_period'):     # far_period: no artist has >= 10 images in 2 years
        inject.make_folders(df, kind).to_parquet(tmp_path / f'b1_{kind}.parquet')
    m = B2.ModelFeats('a', {'bench2_b1': f}, 10.0, -5.0, 0.0)
    r = B2.evaluate_set('bench2_b1', df, [m], 'test', n_boot=20, inject_dir=tmp_path)
    st = r['inject']
    assert st['bad_date']['status'] == 'ok' and st['near_style']['status'] == 'skipped'
    assert st['far_period'] == dict(st['far_period'], status='skipped', reason='no folders in split')
    assert 'inject_auroc/bad_date' in r['a'] and 'inject_auroc/near_style' not in r['a']
    assert r['a']['inject_auroc/bad_date']['point'] > 0.5
    assert r['a']['inject_auroc/same_char']['n_valid'] == 20
    assert 'clustering' not in r                         # test split: no clustering
    assert r['a']['date_mae']['point'] < r['a']['date_mae_baseline']['point']
    assert r['composition']['n_neg_pairs']['same_char'] > 0
    assert r['a']['same_copy_fpr']['point'] is None      # 'original' is not a shared copyright


def test_upload_only_dates_and_reweight(tmp_path):
    df, f = _sep_set()
    df.loc[df.index % 4 == 0, 'date_source'] = 'danbooru_upload'
    lab = df.label.to_numpy()
    df.loc[lab >= 105, 't_year'] = 2016 + (df.t_year[lab >= 105] - 2010) * 0.05    # five short-career artists
    ref = (np.r_[np.full(30, 10.0), np.full(70, 0.5)], np.r_[np.full(30, 2010.0), np.full(70, 2016.0)])
    m = B2.ModelFeats('a', {'s': f}, 10.0, -5.0, 0.0)
    r = B2.evaluate_set('s', df, [m], 'test', n_boot=20, inject_dir=tmp_path, reweight=ref)
    assert r['composition']['n_date_upload_only'] == 40
    assert r['a']['date_mae_upload_only']['point'] is not None
    assert 'pr_auc/<1y/reweighted' in r['a'] and 'recall@1e-3/<1y/reweighted' not in r['a']
    assert r['reweight']['n_zero_weight'] == 0 and r['reweight']['ref_mass_uncovered'] == 0
    assert r['a']['pr_auc/all/reweighted']['point'] != r['a']['pr_auc/all']['point']


def test_cross_source_pairs_and_threshold_transfer(tmp_path):
    """generic cross_source capability (rows = one source, cols = the other) stays after the B3 removal"""
    df, f = _sep_set()
    df.loc[df.index % 2 == 1, 'source'] = 'pixiv'
    fs = {'xsrc': f, 'bench2_b1': f}
    m = [B2.ModelFeats('a', fs, 10.0, -5.0, 0.3), B2.ModelFeats('b', fs, 10.0, -5.0, 0.3)]
    hs = {}
    r = B2.evaluate_set('xsrc', df.assign(split='dev'), m, 'dev', n_boot=20, inject_dir=tmp_path,
                        cross_source=True, hists_out=hs)
    c = r['composition']
    assert c['n_query'] == 80 and c['n_gallery'] == 80
    assert c['n_pos_pairs']['all'] == 10 * 8 * 8                # pixiv x danbooru of the same artist only
    assert r['clustering'] == {'skipped': 'cross_source set'}
    assert r['diff']['b-vs-a']['map_at_r/1y']['point'] == 0
    h1 = {}
    B2.evaluate_set('bench2_b1', df, m, 'test', n_boot=5, inject_dir=tmp_path, hists_out=h1)
    df2, f2 = _sep_set(seed=1)
    h2 = {}
    B2.evaluate_set('bench2_b2', df2, [B2.ModelFeats('a', {'bench2_b2': f2}, 10.0, -5.0, 0.3)], 'test', n_boot=5,
                    inject_dir=tmp_path, hists_out=h2)
    tt = B2.threshold_transfer({'bench2_b1': h1, 'bench2_b2': h2, 'xsrc': hs, 'test100_0918': h1},
                               {'a': 0.3, 'b': 0.3})
    a = tt['a']
    assert set(a['drift']['best_thresholds']) == {'bench2_b1', 'bench2_b2', 'xsrc', 'test100_0918'}
    g2max = max(a['drift']['cross_f1_loss'][k] for k in ('bench2_b1->bench2_b2', 'bench2_b2->bench2_b1'))
    assert a['max_cross_f1_loss_point'] == g2max
    assert a['max_cross_f1_loss'] == dict(point=g2max, lo=None, hi=None, n_valid=0)     # no per-artist data: no CI
    assert 'b3p1_f1_loss' not in a and a['per_set']['bench2_b1']['default']['f1'] >= 0
    assert a['fpr_transfer']['bench2_b1->test100_0918']['fpr'] <= 1e-3
    assert 'max_cross_f1_loss' not in a['drift'] and 'max_cross_f1_loss_all_sets' in a['drift']
    assert a['fpr_transfer_nominal_fpr'] == 1e-3
    src = a['fpr_transfer']['bench2_b1->bench2_b2']['source_fpr']
    assert 0 <= src <= 1e-3 and src == a['per_set']['bench2_b1']['actual_fpr_at_threshold_at_fpr0.001']
    assert tt['b']['drift'] is not None and tt['b']['max_cross_f1_loss']['point'] is None   # b has only one G2 set
    assert tt['diff']['b-vs-a']['max_cross_f1_loss']['point'] is None


def test_run_with_patched_loader(tmp_path, monkeypatch):
    b1, f1 = _sep_set(seed=0)
    b2, f2 = _sep_set(seed=1)
    leg = b2[['sample_key', 'label']]
    data = {'bench2_b1': b1, 'bench2_b2': b2, 'csip_eval_public': leg}
    monkeypatch.setattr(B2, 'load_set', lambda name: (data[name], None))
    calls = []
    monkeypatch.setattr(B2, 'reference_population', lambda: calls.append(1) or (np.full(10, 9.0), np.full(10, 2011.0)))
    feats = {'bench2_b1': f1, 'bench2_b2': f2, 'csip_eval_public': f2}
    ms = [B2.ModelFeats('a', feats, 10.0, -5.0, 0.3), B2.ModelFeats('b', feats, 10.0, -5.0, 0.3)]
    res = B2.run(ms, split='test', sets=('bench2_b1', 'bench2_b2', 'bench2_x'),
                 legacy=('csip_eval_public', 'test100_0918'), n_boot=10, inject_dir=tmp_path, log=lambda *a: None)
    assert set(res['sets']) == {'bench2_b1', 'bench2_b2'}
    assert set(res['skipped']) == {'bench2_x', 'test100_0918'}
    assert calls == [1] and 'pr_auc/all/reweighted' in res['sets']['bench2_b2']['a']
    assert 'pr_auc/all/reweighted' not in res['sets']['bench2_b1']['a']
    tt = res['threshold_transfer']
    g = tt['a']['max_cross_f1_loss']
    assert g['point'] is not None and g['n_valid'] == 10 and g['lo'] <= g['hi']
    assert g['point'] == tt['a']['max_cross_f1_loss_point']
    assert tt['diff']['b-vs-a']['max_cross_f1_loss'] == dict(point=0.0, lo=0.0, hi=0.0, n_valid=10, p_gt0=0.0)
    assert res['timing']['g2_bootstrap_s'] >= 0
    from v3.scripts.bench2_eval import markdown
    md = markdown(res)
    assert '| pr_auc/<1y |' in md and '| max_cross_f1_loss |' in md and '| pr_auc/<1y_xday |' in md
    assert '| same_char_fpr@fpr1e-3 |' in md and 'b3p1' not in md
    import json
    json.dumps(res)


def test_cli_end_to_end(tmp_path, monkeypatch):
    import json
    from v3.scripts import bench2_eval
    df, f = _sep_set()
    loads = []
    monkeypatch.setattr(B2, 'load_set', lambda name: loads.append(name) or (df, None))
    for m in ('a', 'b'):
        (tmp_path / m).mkdir()
        np.savez(tmp_path / m / 'bench2_b1.npz', feats=f.astype(np.float16), sample_key=df.sample_key.values.astype(str))
    out, md = tmp_path / 'r.json', tmp_path / 'r.md'
    bench2_eval.main(['--model', 'a', str(tmp_path / 'a'), '10', '-5', '0.3', '--model', 'b', str(tmp_path / 'b'), '10',
                      '-5', '0.3', '--split', 'test', '--n-boot', '5', '--sets', 'bench2_b1', 'bench2_b2', '--legacy',
                      '--out', str(out), '--md', str(md)])
    res = json.loads(out.read_text())
    assert res['models'] == ['a', 'b'] and set(res['sets']) == {'bench2_b1'} and 'bench2_b2' in res['skipped']
    assert res['sets']['bench2_b1']['diff']['b-vs-a']['pr_auc/<1y']['point'] == 0
    assert loads == ['bench2_b1']                        # only sets with features, loaded once
    text = md.read_text()
    assert '| metric | a | b | b − a |' in text and '| date_mae |' in text


# ---------------------------------------------------------------- fix round 1
from v3.eval import timestrata as T


def test_content_metrics_at_default_threshold(tmp_path):
    """ruling R14: same_char_fpr / same_copy_fpr / cross_content_recall at ModelFeats.threshold; @best keeps the old ones"""
    df, f = _sep_set()
    df['copyrights'] = [[f'p{x}'] for x in np.random.default_rng(3).integers(0, 4, len(df))]
    thr = -5.0 + 10.0 * 0.2                              # cos 0.2
    ms = [B2.ModelFeats('a', {'s': f}, 10.0, -5.0, thr), B2.ModelFeats('b', {'s': f}, 10.0, -5.0, thr + 6.0)]
    r = B2.evaluate_set('s', df, ms, 'test', n_boot=10, inject_dir=tmp_path)
    h = T.artist_histograms(f, df.label.to_numpy() - 100, df.t_year.to_numpy(), df.family.to_numpy(),
                            T.tag_ids(list(df.characters)), T.tag_ids(list(df.copyrights)), 10.0, -5.0)
    b0 = int((0.2 + 1) / 2 * T.N_BINS)
    for k, x in (('same_char_fpr', h.neg[:, 1]), ('same_copy_fpr', h.neg[:, 2]), ('cross_content_recall', h.pos[:, 5])):
        x = x.sum(0)
        assert x.sum() > 0
        assert r['a'][k]['point'] == pytest.approx(x[b0:].sum() / x.sum(), abs=1e-12)
        assert r['a'][f'{k}@best']['point'] is not None
        assert r['a'][f'{k}@best'] == r['b'][f'{k}@best']    # best-threshold values do not depend on THR
        assert r['a'][f'{k}@fpr1e-3'] == r['b'][f'{k}@fpr1e-3']   # nor do the matched-FPR ones (Task 16 A3)
        bm = B2._fpr_bin(h.neg[:, 0].sum(0).astype(float), 1e-3)
        assert r['a'][f'{k}@fpr1e-3']['point'] == pytest.approx(x[bm:].sum() / x.sum(), abs=1e-12)
        assert r['a'][f'{k}@fpr1e-3']['n_valid'] == 10
    assert 0 < r['a']['same_char_fpr']['point'] < 1
    assert r['b']['same_char_fpr']['point'] < r['a']['same_char_fpr']['point']
    assert r['diff']['b-vs-a']['same_char_fpr']['point'] < 0


def test_empty_set_markdown(tmp_path):
    from v3.scripts.bench2_eval import markdown
    df, f = _sep_set()                                   # all rows are 'test'
    ms = [B2.ModelFeats('a', {'s': f}), B2.ModelFeats('b', {'s': f})]
    r = B2.evaluate_set('s', df, ms, 'dev', n_boot=5, inject_dir=tmp_path)
    assert r['composition']['n_artists'] == 0 and r['diff'] == {'b-vs-a': {}} and r['a'] == {}
    md = markdown(dict(models=['a', 'b'], split='dev', sets={'s': r}))
    assert '| pr_auc/<1y | — | — | — |' in md


def test_na_date_source_is_unreliable(tmp_path):
    df, f = _sep_set()
    df['date_source'] = df['date_source'].astype(object)
    df.loc[df.index % 5 == 0, 'date_source'] = None
    r = B2.evaluate_set('s', df, [B2.ModelFeats('a', {'s': f})], 'test', n_boot=5, inject_dir=tmp_path)
    c = r['composition']
    assert c['n_date_reliable'] == 128 and c['date_source'] == {'twitter': 128, 'NA': 32}


# ---------------------------------------------------------------- final-review fix: G2 excludes B3-P2
def test_g2_sets_exclude_b3p2_by_default_and_are_overridable():
    assert 'bench2_b3p2' not in B2.G2_SETS and 'bench2_b3p1' not in B2.G2_SETS
    hs = {}
    for i, s in enumerate(('bench2_b1', 'bench2_b2', 'bench2_b3p2', 'csip_eval_public')):
        df, f = _sep_set(seed=i)
        hs[s] = B2._legacy_hists(s, df, [B2.ModelFeats('a', {s: f + 0.4 * i}, 10.0, -5.0, 0.3)])
    a = B2.threshold_transfer(hs, {'a': 0.3})['a']
    loss = a['drift']['cross_f1_loss']
    assert a['max_cross_f1_loss_sets'] == ['bench2_b1', 'bench2_b2', 'csip_eval_public']
    g2 = [v for k, v in loss.items() if 'bench2_b3p2' not in k]
    assert a['max_cross_f1_loss']['point'] == a['max_cross_f1_loss_point'] == max(g2)
    w = B2.threshold_transfer(hs, {'a': 0.3}, g2_sets=B2.G2_SETS + ('bench2_b3p2',))['a']
    assert 'bench2_b3p2' in w['max_cross_f1_loss_sets'] and w['max_cross_f1_loss']['point'] == max(loss.values())


def test_cli_default_sets_skip_b3_and_pass_g2_sets(tmp_path, monkeypatch):
    import json
    from v3.scripts import bench2_eval
    assert bench2_eval.SETS == ('bench2_b1', 'bench2_b2')
    df, f = _sep_set()
    df2, f2 = _sep_set(seed=1)
    data = {'bench2_b1': df, 'bench2_b2': df2, 'csip_eval_public': df2[['sample_key', 'label']]}
    loads = []
    monkeypatch.setattr(B2, 'load_set', lambda name: loads.append(name) or (data[name], None))
    monkeypatch.setattr(B2, 'reference_population', lambda: (np.full(10, 9.0), np.full(10, 2011.0)))
    (tmp_path / 'a').mkdir()
    for s, x, d in (('bench2_b1', f, df), ('bench2_b2', f2, df2), ('csip_eval_public', f2, df2)):
        np.savez(tmp_path / 'a' / f'{s}.npz', feats=x.astype(np.float16), sample_key=d.sample_key.values.astype(str))
    base = ['--model', 'a', str(tmp_path / 'a'), '10', '-5', '0.3', '--split', 'test', '--n-boot', '5',
            '--legacy', 'csip_eval_public']
    bench2_eval.main(base + ['--out', str(tmp_path / 'r.json')])
    res = json.loads((tmp_path / 'r.json').read_text())
    assert set(res['sets']) == {'bench2_b1', 'bench2_b2'} and res['skipped'] == {}
    assert sorted(set(loads)) == ['bench2_b1', 'bench2_b2', 'csip_eval_public']       # B3 never read by default
    assert res['threshold_transfer']['a']['max_cross_f1_loss_sets'] == ['bench2_b1', 'bench2_b2', 'csip_eval_public']
    bench2_eval.main(base + ['--g2-sets', 'bench2_b1', 'bench2_b2', '--out', str(tmp_path / 'g.json')])
    t = json.loads((tmp_path / 'g.json').read_text())['threshold_transfer']['a']
    assert t['max_cross_f1_loss_sets'] == ['bench2_b1', 'bench2_b2']
    assert t['max_cross_f1_loss']['point'] == max(t['drift']['cross_f1_loss'][k]
                                                  for k in ('bench2_b1->bench2_b2', 'bench2_b2->bench2_b1'))
    assert t['max_cross_f1_loss']['n_valid'] == 5


# ---------------------------------------------------------------- Task 16: P1-prerequisite metrics, B3 removal
def test_b3_removed_from_benchmark_surface():
    from v3.eval import evalsets
    from v3.scripts import bench2_eval
    assert not any('b3p' in k for k in evalsets.INDEX_SETS)
    assert {'bench2_b1', 'bench2_b2', 'bench2_b1_sealed', 'bench2_b2_sealed'} <= set(evalsets.INDEX_SETS)
    assert not hasattr(B2, 'CROSS_SET') and B2.REWEIGHT_SETS == ('bench2_b2',)
    assert bench2_eval.SETS == ('bench2_b1', 'bench2_b2') and bench2_eval.RUN_LEVEL == ('max_cross_f1_loss',)


def test_xday_keys_and_pair_count_in_evaluate_set(tmp_path):
    df, f = _sep_set()
    df['date_day'] = 20000 + df.index // 2                     # consecutive images of an artist share a day
    r = B2.evaluate_set('s', df, [B2.ModelFeats('a', {'s': f}, 10.0, -5.0, 0.3)], 'test', n_boot=10,
                        inject_dir=tmp_path)
    lab, t, day = df.label.to_numpy(), df.t_year.to_numpy(), df.date_day.to_numpy()
    i, j = np.triu_indices(len(df), 1)
    lt1 = (lab[i] == lab[j]) & (np.abs(t[i] - t[j]) < 1.0)
    n = r['composition']['n_pos_pairs']
    assert n['<1y'] == lt1.sum() and n['<1y_xday'] == (lt1 & (day[i] != day[j])).sum() < n['<1y']
    for k in ('pr_auc/<1y_xday', 'recall@1e-3/<1y_xday', 'recall@1e-4/<1y_xday'):
        assert r['a'][k]['point'] is not None and r['a'][k]['n_valid'] == 10


def test_matched_fpr_content_metrics_hand_computed():
    GP, GN, NB = len(T.POS_GROUPS), len(T.NEG_GROUPS), T.N_BINS
    P, N = T.POS_GROUPS.index, T.NEG_GROUPS.index
    pos = np.zeros((1, GP, NB), np.int64)
    neg = np.zeros((1, GN, NB), np.int64)
    pos[0, P('all'), 13000] = 4
    pos[0, P('cross_content'), 9000] = 1
    pos[0, P('cross_content'), 9500] = 3
    # all-group negatives (each pair stored once per side): 1000 pairs at bin 1000, 1 at 9000, 1 at 12000 -> 1002 pairs.
    # tail(b) = 2/1002 > 1e-3 for b <= 9000 and 1/1002 <= 1e-3 for 9000 < b <= 12000 -> threshold bin 9001
    neg[0, N('all'), 1000] = 2000
    neg[0, N('all'), 9000] = 2
    neg[0, N('all'), 12000] = 2
    neg[0, N('same_char'), 8000] = 3
    neg[0, N('same_char'), 10000] = 1
    neg[0, N('same_copy'), 5000] = 2
    neg[0, N('same_copy'), 11000] = 2
    h = T.ArtistHists(pos, neg, 10.0, -5.0)
    assert B2._fpr_bin(neg[0, N('all')].astype(float), 1e-3) == 9001
    pt, dr = B2._strat_draws(h, np.ones((3, 1), np.int32), np.ones(1), threshold=0.0)
    want = {'same_char_fpr@fpr1e-3': 0.25, 'same_copy_fpr@fpr1e-3': 0.5, 'cross_content_recall@fpr1e-3': 0.75,
            'overall_fpr@fpr1e-3': 1 / 1002, 'same_char_ratio@fpr1e-3': 0.25 * 1002}
    for k, v in want.items():
        assert pt[k] == pytest.approx(v, rel=1e-12), k
        assert np.allclose(dr[k], v, rtol=1e-12), k
    # empty content group -> NaN; no negatives at all -> NaN everywhere
    neg2 = neg.copy()
    neg2[0, N('same_char')] = 0
    pt2, _ = B2._strat_draws(T.ArtistHists(pos, neg2, 10.0, -5.0), np.ones((1, 1), np.int32), np.ones(1))
    assert np.isnan(pt2['same_char_fpr@fpr1e-3']) and np.isnan(pt2['same_char_ratio@fpr1e-3'])
    assert pt2['same_copy_fpr@fpr1e-3'] == pytest.approx(0.5)


def _g2_inputs(tmp_path, n_boot=40, models=('a', 'b')):
    """B1 / B2 through evaluate_set (per-artist 'all' histograms + W into boot_out), csip_eval_public via _legacy_hists"""
    rng = np.random.default_rng(7)
    hs, boot = {}, {}
    for i, s in enumerate(('bench2_b1', 'bench2_b2', 'csip_eval_public')):
        df, f = _set(n_art=12 + 2 * i, per=10, seed=i)
        df = df.assign(split='test')
        noise = rng.standard_normal(f.shape)
        ms = [B2.ModelFeats(m, {s: f + k * 0.8 * noise}, 10.0, -5.0, 0.3) for k, m in enumerate(models)]
        if s == 'csip_eval_public':
            hs[s] = B2._legacy_hists(s, df, ms)
        else:
            hs[s], boot[s] = {}, {}
            B2.evaluate_set(s, df, ms, 'test', n_boot=n_boot, inject_dir=tmp_path, hists_out=hs[s], boot_out=boot[s])
    return hs, boot


def test_boot_out_holds_shared_W_and_per_artist_all_histograms(tmp_path):
    hs, boot = _g2_inputs(tmp_path)
    from v3.eval import bootstrap
    b = boot['bench2_b1']
    assert (b['W'] == bootstrap.draw_weights(12, 40, 0)).all()
    for m in ('a', 'b'):
        assert b['pos'][m].shape == (12, T.N_BINS) and b['neg'][m].shape == (12, T.N_BINS)
        assert (b['pos'][m].sum(0) == hs['bench2_b1'][m].pos[0, T.POS_GROUPS.index('all')]).all()
        assert (b['neg'][m].sum(0) == hs['bench2_b1'][m].neg[0, T.NEG_GROUPS.index('all')]).all()


def test_g2_bootstrap_ci_paired_diff_and_brute_force_draw(tmp_path):
    from v3.eval import pairwise
    hs, boot = _g2_inputs(tmp_path)
    tt = B2.threshold_transfer(hs, {'a': 0.3, 'b': 0.3}, boot=boot)
    for m in ('a', 'b'):
        g = tt[m]['max_cross_f1_loss']
        assert set(g) == {'point', 'lo', 'hi', 'n_valid'} and g['n_valid'] == 40
        cl = tt[m]['drift']['cross_f1_loss']
        assert g['point'] == tt[m]['max_cross_f1_loss_point'] == max(cl.values())   # 3 sets, all G2
        assert g['lo'] <= g['hi'] and g['lo'] < g['hi']
        assert tt[m]['max_cross_f1_loss_boot_sets'] == ['bench2_b1', 'bench2_b2']
    d = tt['diff']['b-vs-a']['max_cross_f1_loss']
    assert d['point'] == pytest.approx(tt['b']['max_cross_f1_loss']['point'] - tt['a']['max_cross_f1_loss']['point'])
    assert 0.0 <= d['p_gt0'] <= 1.0 and d['n_valid'] == 40 and d['lo'] <= d['hi']
    # brute force: draw k = per-artist 'all' histograms of B1 / B2; positives weighted by W[k], negatives by the
    # linearised product weight (2 W[k] - 1) (R22: pair (a, a') -> k_a + k_a' - 1), clipped at 0; csip_eval_public fixed
    pt, draws, bs = B2.g2_loss_draws(hs, boot, 'b', B2.G2_SETS)
    assert bs == ['bench2_b1', 'bench2_b2']
    assert pt == tt['b']['max_cross_f1_loss_point'] and draws.shape == (40,)
    for k in (0, 17, 39):
        cs = {}
        for s in ('bench2_b1', 'bench2_b2'):
            w = boot[s]['W'][k].astype(float)
            cs[s] = T.curve_metrics(w @ boot[s]['pos']['b'], np.maximum((2 * w - 1) @ boot[s]['neg']['b'], 0) / 2,
                                    10.0, -5.0)
        hp = hs['csip_eval_public']['b']
        cs['csip_eval_public'] = T.curve_metrics(hp.pos[0, T.POS_GROUPS.index('all')].astype(float),
                                                 hp.neg[0, 0].astype(float) / 2, 10.0, -5.0)
        assert draws[k] == pytest.approx(max(pairwise.threshold_drift(cs)['cross_f1_loss'].values()), abs=1e-12)
    # identical models -> zero paired diff; boot for only one set -> the other G2 sets held fixed
    hs2, boot2 = _g2_inputs(tmp_path, n_boot=15, models=('a', 'c'))
    for s in hs2:
        hs2[s]['c'] = hs2[s]['a']
    for s in boot2:
        boot2[s]['pos']['c'], boot2[s]['neg']['c'] = boot2[s]['pos']['a'], boot2[s]['neg']['a']
    t2 = B2.threshold_transfer(hs2, {}, boot={'bench2_b1': boot2['bench2_b1']})
    assert t2['diff']['c-vs-a']['max_cross_f1_loss'] == dict(point=0.0, lo=0.0, hi=0.0, n_valid=15, p_gt0=0.0)
    assert t2['a']['max_cross_f1_loss_boot_sets'] == ['bench2_b1'] and t2['a']['max_cross_f1_loss']['n_valid'] == 15


def test_cli_refuses_to_overwrite_outputs(tmp_path):
    from v3.scripts import bench2_eval
    out = tmp_path / 'r.json'
    out.write_text('{}')
    with pytest.raises(SystemExit):
        bench2_eval.main(['--model', 'a', str(tmp_path), '1', '0', '0', '--out', str(out)])
    assert out.read_text() == '{}'


# ---------------------------------------------------------------- Task 16 fix round 1 (ruling R22)
def _hist(seed=0):
    df, f = _sep_set(seed=seed)
    f = f + 1.5 * np.random.default_rng(seed + 5).standard_normal(f.shape)      # overlapping artists: pr_auc < 1
    lab = B2._codes(df['label'].to_numpy())
    chars = T.tag_ids(B2._tag_lists(df, 'characters'))
    copys = T.tag_ids(B2._tag_lists(df, 'copyrights'))
    return T.artist_histograms(f, lab, df['t_year'].to_numpy(), df['family'].to_numpy(), chars, copys, 10.0, -5.0,
                               day=B2._codes(df['date_day'].to_numpy()))


def _same(a, b):
    return (np.isnan(a) and np.isnan(b)) or a == b


def test_strat_draws_all_ones_W_reproduce_point_bitwise():
    h = _hist()
    A = h.pos.shape[0]
    ones = np.ones((3, A), dtype=np.int32)
    pt, dr = B2._strat_draws(h, ones, np.ones(A), threshold=0.3)
    for k, v in pt.items():
        assert all(_same(x, v) for x in dr[k]), k
    rw = np.random.default_rng(1).random(A) + 0.5                       # reweight base: same to rounding
    pt, dr = B2._strat_draws(h, ones, rw, suffix='/reweighted', full=False)
    for k, v in pt.items():
        assert dr[k] == pytest.approx(np.full(3, v), rel=1e-12, abs=1e-15, nan_ok=True), k


def test_strat_draw_negatives_use_linearised_product_weight():
    h = _hist()
    A, GP, NB = h.pos.shape
    W = B2.bootstrap.draw_weights(A, 5, 3)
    assert h.pos[:, T.POS_GROUPS.index('cross_content')].sum() > 0 and h.neg[:, T.NEG_GROUPS.index('same_char')].sum() > 0
    base = np.ones(A)
    pt, dr = B2._strat_draws(h, W, base, threshold=0.3)
    k = 2
    w = W[k].astype(np.float64)
    p = w @ h.pos.reshape(A, -1).astype(np.float64)
    n = np.maximum((2 * w - 1) @ h.neg.reshape(A, -1).astype(np.float64), 0)
    sm = T.stratified_metrics(T.ArtistHists(p.reshape(1, GP, NB), n.reshape(1, -1, NB), h.scale, h.bias))
    assert dr['pr_auc/all'][k] == pytest.approx(sm['all']['pr_auc'], abs=1e-12)
    assert dr['pr_auc/<1y_xday'][k] == pytest.approx(sm['<1y_xday']['pr_auc'], abs=1e-12)
    assert dr['same_char_fpr@best'][k] == pytest.approx(sm['same_char_fpr'], abs=1e-12)
    mf = B2._matched_fpr(p.reshape(GP, NB), n.reshape(-1, NB))
    for key, v in mf.items():
        assert _same(dr[key][k], v) or dr[key][k] == pytest.approx(v, abs=1e-12), key
    # old linear weight W @ neg would give a different draw
    n_old = w @ h.neg.reshape(A, -1).astype(np.float64)
    sm_old = T.stratified_metrics(T.ArtistHists(p.reshape(1, GP, NB), n_old.reshape(1, -1, NB), h.scale, h.bias))
    assert sm_old['all']['pr_auc'] != pytest.approx(dr['pr_auc/all'][k], abs=1e-9)


def test_g2_draws_all_ones_W_reproduce_point_and_no_vacuous_assert(tmp_path):
    hs, boot = _g2_inputs(tmp_path, n_boot=4)
    for s in boot:
        boot[s]['W'] = np.ones_like(boot[s]['W'])
    for m in ('a', 'b'):
        pt, draws, bs = B2.g2_loss_draws(hs, boot, m, B2.G2_SETS)
        assert bs == ['bench2_b1', 'bench2_b2'] and (draws == pt).all()
    import inspect
    assert '1e-9' not in inspect.getsource(B2.threshold_transfer)


def test_shared_paired_diff_helper_used():
    import inspect
    assert 'bootstrap.paired_diff' in inspect.getsource(B2._paired)
    assert 'bootstrap.paired_diff' in inspect.getsource(B2._diff_entry)
    d = B2._diff_entry(1.0, np.array([1.0, 2.0]), 1.5, np.array([2.0, 1.0]))
    assert d['point'] == 0.5 and d['n_valid'] == 2 and d['p_gt0'] == 0.5


def test_markdown_run_level_header_claims_ci_only_for_dict():
    from v3.scripts.bench2_eval import markdown
    new = dict(models=['a'], split='test', sets={},
               threshold_transfer={'a': dict(max_cross_f1_loss=dict(point=0.05, lo=0.04, hi=0.06, n_valid=10))})
    old = dict(models=['a'], split='test', sets={}, threshold_transfer={'a': dict(max_cross_f1_loss=0.05)})
    assert 'G2 CI = artist bootstrap' in markdown(new)
    assert 'G2 CI = artist bootstrap' not in markdown(old) and 'no CI' in markdown(old)
