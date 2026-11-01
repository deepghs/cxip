from v3.scripts.bench2_tables import max_cross_loss, render, transfer_table


def test_render_main_and_clustering_tables():
    v = lambda p, lo=None, hi=None: dict(point=p, lo=lo, hi=hi)
    res = dict(split='dev', models=['v3_release', 'kaloscope'], sets={'bench2_b1': dict(
        composition=dict(n_artists=3, n_images=30),
        v3_release={'pr_auc/<1y': v(0.8, 0.7, 0.9), 'same_char_fpr': v(0.008, 0.006, 0.01)},
        kaloscope={'pr_auc/<1y': v(0.75, 0.65, 0.85)},
        diff={'kaloscope-vs-v3_release': {'pr_auc/<1y': v(-0.05, -0.1, -0.01)}},
        clustering={m: dict(n_pools=5, umap=dict(bcubed_f_mean=0.8, bcubed_f_std=0.01), none={})
                    for m in ('v3_release', 'kaloscope')})})
    md = render(res)
    assert '#### bench2_b1 (dev): 3 个画师, 30 张图' in md
    assert '| pr_auc/<1y | 0.800 [0.700, 0.900] | 0.750 [0.650, 0.850] | -0.0500 [-0.100, -0.0100] |' in md
    assert '| same_char_fpr | 0.0080 [0.0060, 0.0100] | — | — |' in md
    assert '| 聚类 (5 个混合池, 均值 ± 池间 std) | v3 发布版 | Kaloscope |' in md
    assert '| umap bcubed_f | 0.800 ± 0.010 | 0.800 ± 0.010 |' in md and '| none ari | — | — |' in md
    assert 'date_mae' not in md                                          # rows absent for every model are skipped


def test_transfer_table_ignores_excluded_sets():
    tt = {'drift': {'cross_f1_loss': {'a->b': 0.05, 'b->a': 0.07, 'a->bad': 0.6, 'bad->b': 0.5}},
          'per_set': {s: dict(best_f1=0.7, default=dict(f1=0.68, fpr=0.0008)) for s in ('a', 'b', 'bad')}}
    res = dict(split='test', models=['m'], threshold_transfer={'m': tt})
    assert max_cross_loss(res, ['a', 'b']) == {'m': (0.07, 'b->a')}
    md = transfer_table(res, ['a', 'b'])
    assert '| 最大跨集 F1 损失 (test; a, b) | 0.070 (b->a) |' in md
    assert '| 默认阈值 F1 / best F1，a | 0.680 / 0.700 (FPR 0.0008) |' in md and 'bad' not in md


def test_regate_g2_drops_b3p2_and_b3_sets():
    from v3.scripts import bench2_tables as BT
    cl = {'bench2_b1->bench2_b2': 0.05, 'bench2_b2->bench2_b1': 0.02, 'bench2_b1->bench2_b3p2': 0.5,
          'bench2_b3p2->bench2_b1': 0.4, 'csip_eval_public->bench2_b1': 0.07, 'bench2_b1->csip_eval_public': 0.01,
          'bench2_b1->test100_0918': 0.9}
    per = {s: {'best_f1': 0.8, 'default': dict(f1=0.7, fpr=1e-3)}
           for s in ('bench2_b1', 'bench2_b2', 'bench2_b3p2', 'csip_eval_public', 'test100_0918')}
    res = dict(models=['a'], split='test', sets={'bench2_b1': {}, 'bench2_b3p1': {}, 'bench2_b3p2': {}},
               threshold_transfer={'a': dict(max_cross_f1_loss=0.5, b3p1_f1_loss=0.6, per_set=per,
                                             drift=dict(cross_f1_loss=cl))})
    r = BT.regate_g2(res)
    t = r['threshold_transfer']['a']
    assert t['max_cross_f1_loss'] == 0.07 and t['max_cross_f1_loss_pair'] == 'csip_eval_public->bench2_b1'
    assert t['max_cross_f1_loss_sets'] == ['bench2_b1', 'bench2_b2', 'csip_eval_public']
    assert t['b3p1_f1_loss'] is None and set(r['sets']) == {'bench2_b1'}
    assert res['threshold_transfer']['a']['max_cross_f1_loss'] == 0.5           # input untouched
    assert BT.regate_g2(res, ['bench2_b1', 'bench2_b3p2'])['threshold_transfer']['a']['max_cross_f1_loss'] == 0.5


# ---------------------------------------------------------------- Task 16
def test_render_skips_b3_sets_and_shows_new_rows():
    v = lambda p, lo=None, hi=None: dict(point=p, lo=lo, hi=hi)
    S = lambda: dict(composition=dict(n_artists=3, n_images=30),
                     a={'pr_auc/<1y_xday': v(0.7, 0.6, 0.8), 'same_char_fpr@fpr1e-3': v(0.004, 0.003, 0.005),
                        'same_char_ratio@fpr1e-3': v(4.5, 3.0, 6.0), 'cross_content_recall@fpr1e-3': v(0.5)})
    res = dict(split='test', models=['a'], sets={'bench2_b1': S(), 'bench2_b3p1': S(), 'bench2_b3p2': S()})
    md = render(res)
    assert '#### bench2_b1' in md and 'b3p' not in md
    assert '| pr_auc/<1y_xday | 0.700 [0.600, 0.800] |' in md
    assert '| same_char_fpr@fpr1e-3 | 0.0040 [0.0030, 0.0050] |' in md
    assert '| same_char_ratio@fpr1e-3 | 4.500 [3.000, 6.000] |' in md and '| cross_content_recall@fpr1e-3 | 0.500 |' in md


def test_g2_table_with_ci_and_paired_diff():
    from v3.scripts.bench2_tables import g2_table
    g = lambda p, lo, hi: dict(point=p, lo=lo, hi=hi, n_valid=1000)
    tt = {'v3_release': dict(max_cross_f1_loss=g(0.077, 0.05, 0.11)), 'kaloscope': dict(max_cross_f1_loss=g(0.058, 0.04, 0.08)),
          'diff': {'kaloscope-vs-v3_release': {'max_cross_f1_loss': dict(point=-0.019, lo=-0.04, hi=0.003, n_valid=1000,
                                                                         p_gt0=0.04)}}}
    res = dict(split='test', models=['v3_release', 'kaloscope'], threshold_transfer=tt)
    md = g2_table(res)
    assert '| 指标 | v3 发布版 | Kaloscope | Kaloscope − v3 发布版 |' in md
    assert ('| G2 最大跨集 F1 损失 (test) | 0.0770 [0.0500, 0.110] | 0.0580 [0.0400, 0.0800] | '
            '-0.0190 [-0.0400, +0.0030] p>0=0.04 |') in md
    # old JSON (float, no CI) still renders
    old = dict(split='dev', models=['a'], threshold_transfer={'a': dict(max_cross_f1_loss=0.05)})
    assert '| G2 最大跨集 F1 损失 (dev) | 0.0500 |' in g2_table(old)


def test_regate_g2_skips_diff_entry():
    from v3.scripts import bench2_tables as BT
    cl = {'bench2_b1->bench2_b2': 0.05, 'bench2_b2->bench2_b1': 0.02}
    res = dict(models=['a'], split='test', sets={}, threshold_transfer={
        'a': dict(max_cross_f1_loss=0.05, per_set={'bench2_b1': {}, 'bench2_b2': {}}, drift=dict(cross_f1_loss=cl)),
        'diff': {}})
    r = BT.regate_g2(res)
    assert r['threshold_transfer']['a']['max_cross_f1_loss'] == 0.05 and r['threshold_transfer']['diff'] == {}
