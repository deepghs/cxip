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
    df, f = _sep_set()
    df.loc[df.index % 2 == 1, 'source'] = 'pixiv'
    fs = {'bench2_b3p1': f, 'bench2_b1': f}
    m = [B2.ModelFeats('a', fs, 10.0, -5.0, 0.3), B2.ModelFeats('b', fs, 10.0, -5.0, 0.3)]
    hs = {}
    r = B2.evaluate_set('bench2_b3p1', df.assign(split='dev'), m, 'dev', n_boot=20, inject_dir=tmp_path,
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
    tt = B2.threshold_transfer({'bench2_b1': h1, 'bench2_b2': h2, 'bench2_b3p1': hs, 'test100_0918': h1},
                               {'a': 0.3, 'b': 0.3})
    a = tt['a']
    assert set(a['drift']['best_thresholds']) == {'bench2_b1', 'bench2_b2', 'test100_0918'}
    assert a['max_cross_f1_loss'] == max(a['drift']['cross_f1_loss'][k] for k in ('bench2_b1->bench2_b2', 'bench2_b2->bench2_b1'))
    assert a['b3p1_f1_loss'] is not None and a['per_set']['bench2_b1']['default']['f1'] >= 0
    assert a['fpr_transfer']['bench2_b1->test100_0918']['fpr'] <= 1e-3
    assert 'max_cross_f1_loss' not in a['drift'] and 'max_cross_f1_loss_all_sets' in a['drift']
    assert a['fpr_transfer_nominal_fpr'] == 1e-3
    src = a['fpr_transfer']['bench2_b1->bench2_b2']['source_fpr']
    assert 0 <= src <= 1e-3 and src == a['per_set']['bench2_b1']['actual_fpr_at_threshold_at_fpr0.001']
    assert tt['b']['drift'] is not None and tt['b']['max_cross_f1_loss'] is None   # b has only one G2 set


def test_run_with_patched_loader(tmp_path, monkeypatch):
    b1, f1 = _sep_set(seed=0)
    b2, f2 = _sep_set(seed=1)
    p3 = b1.iloc[::2].assign(source='pixiv', sample_key=lambda d: 'p' + d.sample_key.str[1:])
    f3 = f1[::2] + 0.05
    leg = b2[['sample_key', 'label']]
    data = {'bench2_b1': b1, 'bench2_b2': b2, 'bench2_b3p1': p3, 'csip_eval_public': leg}
    monkeypatch.setattr(B2, 'load_set', lambda name: (data[name], None))
    calls = []
    monkeypatch.setattr(B2, 'reference_population', lambda: calls.append(1) or (np.full(10, 9.0), np.full(10, 2011.0)))
    feats = {'bench2_b1': f1, 'bench2_b2': f2, 'bench2_b3p1': f3, 'csip_eval_public': f2}
    res = B2.run([B2.ModelFeats('a', feats, 10.0, -5.0, 0.3)], split='test',
                 sets=('bench2_b1', 'bench2_b2', 'bench2_b3p1', 'bench2_b3p2'),
                 legacy=('csip_eval_public', 'test100_0918'), n_boot=10, inject_dir=tmp_path, log=lambda *a: None)
    assert set(res['sets']) == {'bench2_b1', 'bench2_b2', 'bench2_b3p1'}
    assert set(res['skipped']) == {'bench2_b3p2', 'test100_0918'}
    assert calls == [1] and 'pr_auc/all/reweighted' in res['sets']['bench2_b2']['a']
    assert res['sets']['bench2_b3p1']['composition']['n_query'] == 80
    assert res['threshold_transfer']['a']['max_cross_f1_loss'] is not None
    from v3.scripts.bench2_eval import markdown
    md = markdown(res)
    assert '| pr_auc/<1y |' in md and '| max_cross_f1_loss |' in md
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
    assert a['max_cross_f1_loss'] == max(g2)
    w = B2.threshold_transfer(hs, {'a': 0.3}, g2_sets=B2.G2_SETS + ('bench2_b3p2',))['a']
    assert 'bench2_b3p2' in w['max_cross_f1_loss_sets'] and w['max_cross_f1_loss'] == max(loss.values())


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
    assert t['max_cross_f1_loss'] == max(t['drift']['cross_f1_loss'][k]
                                         for k in ('bench2_b1->bench2_b2', 'bench2_b2->bench2_b1'))
