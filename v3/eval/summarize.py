"""把评测 json 汇总成 markdown (扩展 experiments/common/summarize.py; 兼容旧 pair_eval.py 输出和 bias_eval 输出).

python -m v3.eval.summarize /data/neko/cxip_v3/eval/results/*.json /data/neko/cxip_repro/results/hf_*.json [--out x.md]

识别三种 json: run_suite / pair_eval ({'model', 'datasets': {...}}), run_bias ({'model', 'kind', ...}),
旧 bias_eval ({model: {'content': {'I'}, 'char': {'I'}}}).
"""
from __future__ import annotations

import argparse
import json
import math


def _f(x, nd=4):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return '-'
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return str(x)
    return f'{x:.{nd}f}'


def _table(head, rows):
    L = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    L += ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows]
    return L


def load(paths):
    suites, biases = [], []
    for p in paths:
        with open(p) as f:
            r = json.load(f)
        if 'datasets' in r:
            r['_path'] = p
            suites.append(r)
        elif 'kind' in r and ('content' in r or 'style' in r):
            biases.append(dict(model=r.get('model') or p, **{k: r[k] for k in ('content', 'char', 'style', 'style_ce')
                                                                if k in r}))
        else:   # 旧 bias_eval: {model: {content, char}}
            for m, v in r.items():
                if isinstance(v, dict) and ('content' in v or 'char' in v):
                    biases.append(dict(model=m, **v))
    return suites, biases


def summarize(paths) -> str:
    suites, biases = load(paths)
    L = []
    if suites:
        L += ['## 两两配对 / zero-shot / 检索', '']
        rows = []
        for r in suites:
            for name, d in r['datasets'].items():
                zs1 = d.get('zero_shot_ref1', {}).get('top1')
                zs5 = d.get('zero_shot_ref5', {}).get('top1')
                rt = d.get('retrieval', {})
                rows.append([r['model'], name, d.get('n_images'), d.get('n_classes'), _f(d.get('pr_auc')),
                             _f(d.get('roc_auc')), _f(d.get('best_f1')), _f(d.get('best_threshold'), 3),
                             _f(d.get('f1_plateau_width'), 3), _f(zs1, 3), _f(zs5, 3), _f(rt.get('map_at_r')),
                             _f(rt.get('recall_at_1'), 3)])
        L += _table(['model', 'dataset', 'images', 'classes', 'PR-AUC', 'ROC-AUC', 'best F1', 'threshold',
                     'F1 plateau', 'zs top1 (1 ref)', 'zs top1 (5 ref)', 'mAP@R', 'R@1'], rows)
        rows = []
        for r in suites:
            for name, d in r['datasets'].items():
                for g, x in d.get('groups', {}).items():
                    rows.append([r['model'], name, g, x.get('n_images'), x.get('n_classes'), _f(x.get('pr_auc')),
                                 _f(x.get('best_f1')), _f(x.get('zero_shot_ref1', {}).get('top1'), 3),
                                 _f(x.get('retrieval', {}).get('map_at_r'))])
        if rows:
            L += ['', '## 开放集分组 (new_franchise / hard, 骨干见过 / 没见过)', '']
            L += _table(['model', 'dataset', 'group', 'images', 'classes', 'PR-AUC', 'best F1', 'zs top1 (1 ref)',
                         'mAP@R'], rows)
        rows = []
        for r in suites:
            for name, d in r['datasets'].items():
                for cfg, byA in d.get('ood', {}).items():
                    for A, x in byA.items():
                        rows.append([r['model'], cfg, A, _f(x.get('closed_top1'), 3), _f(x.get('ood_auroc')),
                                     _f(x.get('ood_f1_best')), _f(x.get('open_best_f1_macro')),
                                     _f(x.get('open_tau_f1_macro')), _f(x.get('tau_best'), 2)])
        if rows:
            L += ['', '## OOD zero-shot (N 已知 + X 未知, A 锚点, 20 次重复均值)', '']
            L += _table(['model', 'N_X', 'A', 'closed top1', 'OOD AUROC', 'OOD F1 (best tau)',
                         'open macro F1 (best tau)', 'open macro F1 (tau)', 'best tau'], rows)
        rows = []
        for r in suites:
            for name, d in r['datasets'].items():
                c = d.get('clustering')
                if c:
                    rows.append([r['model'], name, _f(c.get('nmi')), _f(c.get('ari')), _f(c.get('purity')),
                                 _f(c.get('noise_frac'), 3), c.get('best_mcs'), c.get('n_clusters'), c.get('reduce')])
        if rows:
            L += ['', '## 聚类 (降维 + HDBSCAN, 按 NMI 选 min_cluster_size)', '']
            L += _table(['model', 'dataset', 'NMI', 'ARI', 'purity', 'noise', 'mcs', 'clusters', 'reduce'], rows)
        rows = []
        for r in suites:
            for name, d in r['datasets'].items():
                c = d.get('crosschar')
                if c:
                    rows.append([r['model'], name, _f(c.get('pr_auc_balanced')), _f(c.get('pr_auc_balanced_crosschar')),
                                 _f(c.get('roc_auc_hardneg')), _f(c.get('pr_auc_hardneg')),
                                 _f(c.get('hardneg_fpr')), _f(c.get('randneg_fpr')),
                                 c.get('n_pos_pairs'), c.get('n_hardneg_pairs')])
        if rows:
            L += ['', '## CSIP 跨角色 (正 = 同画师, 负 = 同角色不同画师)', '']
            L += _table(['model', 'dataset', 'PR-AUC (平衡)', 'PR-AUC (平衡, 正样本限不同角色)', 'ROC-AUC',
                         'PR-AUC (不平衡)', 'hard neg FPR @thr', 'random neg FPR @thr', 'pos pairs',
                         'hard neg pairs'], rows)
        rows = []
        for r in suites:
            t = r.get('threshold_drift')
            if t:
                rows.append([r['model'], _f(t.get('spread'), 3), _f(t.get('std'), 3), _f(t.get('max_cross_f1_loss')),
                             _f(t.get('shared_threshold'), 3), _f(t.get('shared_threshold_mean_f1_loss'))])
        if rows:
            L += ['', '## 最佳阈值的跨数据集漂移', '']
            L += _table(['model', 'spread', 'std', 'max F1 loss (A 的阈值用在 B)', 'shared thr',
                         'mean F1 loss @ shared'], rows)
    if biases:
        L += ['', '## 偏置探针 (I = 1 - L(f(x), y) / L(ε, y); < 0.05 无可辨识信息)', '']
        rows = [[b['model'], _f(b.get('content', {}).get('I')), _f(b.get('char', {}).get('I')),
                 _f(b.get('style', {}).get('I')), _f(b.get('style_ce', {}).get('I'))] for b in biases]
        L += _table(['model', 'content I_c', 'char I_char', 'style I_style (回归)', 'style I_style (分类)'], rows)
    return '\n'.join(L) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--out')
    a = ap.parse_args()
    md = summarize(a.paths)
    if a.out:
        with open(a.out, 'w') as f:
            f.write(md)
    print(md)


if __name__ == '__main__':
    main()
