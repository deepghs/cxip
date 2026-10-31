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
