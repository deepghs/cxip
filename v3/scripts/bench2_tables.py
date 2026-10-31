"""benchmark v2 结果 JSON (v3.scripts.bench2_eval --out) -> 报告用的 markdown 表 (中文表头, 指标比 bench2_eval 的
markdown 全). 每个 B 集合一张主表 (各模型 point [lo, hi] + 相对第一个模型的配对 diff), dev 结果另有聚类表.

阈值迁移表: 在指定集合之间取 threshold_transfer[模型]['drift']['cross_f1_loss'] 的逐对最大值 (可去掉无效集合,
如 B3-P2), 并列出各集合的默认阈值 F1 / best F1 / 默认阈值下的 FPR.

G2 修正 (--g2fix): 不重跑评测, 用已有 JSON 里逐对的 cross_f1_loss 在 G2 集合 (默认 bench2.G2_SETS, 不含 B3-P2)
之间重算 max_cross_f1_loss, 去掉无效的 B3 集合 (Pixiv 来源无效, 报告 §6.1), 按 bench2_eval 的 markdown 格式写新文件.

用法: python -m v3.scripts.bench2_tables results_test.json [bench2_b1 bench2_b2 ...]
      python -m v3.scripts.bench2_tables results_test.json --transfer bench2_b1 bench2_b2 csip_eval_public
      python -m v3.scripts.bench2_tables results_test.json --g2fix results_test_g2fix.md [G2 集合 ...]
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

NAMES = {'v3_release': 'v3 发布版', 'v3_x4': 'v3 x4', 'kaloscope': 'Kaloscope'}
ROWS = ('pr_auc/<1y', 'pr_auc/1-3y', 'pr_auc/3-6y', 'pr_auc/>=6y', 'pr_auc/all', 'pr_auc/<1y/reweighted',
        'pr_auc/1-3y/reweighted', 'pr_auc/>=6y/reweighted', 'recall@1e-3/<1y', 'recall@1e-3/1-3y', 'recall@1e-3/>=6y',
        'map_at_r/1y', 'r1/1y', 'map_at_r/any', 'ap/<1y', 'drift_spearman_median', 'date_mae', 'date_hit1y',
        'date_mae_content', 'date_mae_baseline', 'date_mae_upload_only', 'inject_auroc/near_style',
        'inject_auroc/same_char', 'inject_auroc/far_period', 'inject_auroc/bad_date', 'cross_content_recall',
        'same_char_fpr', 'same_copy_fpr')


def _num(x, signed=False) -> str:
    if x is None:
        return '—'
    s = '+' if signed else ''
    return f'{x:{s}.4f}' if abs(x) < 0.1 else f'{x:{s}.3f}'


def _cell(v, signed=False) -> str:
    if not v or v.get('point') is None:
        return '—'
    s = _num(v['point'], signed)
    if v.get('lo') is not None:
        s += f' [{_num(v["lo"], signed)}, {_num(v["hi"], signed)}]'
    return s


def _name(m: str) -> str:
    return NAMES.get(m, m)


def render(res: dict, sets=None) -> str:
    ms = res['models']
    ref = ms[0]
    out = []
    for s in (sets or list(res['sets'])):
        if s not in res['sets']:
            continue
        S = res['sets'][s]
        c = S['composition']
        out += [f'#### {s} ({res["split"]}): {c["n_artists"]} 个画师, {c["n_images"]} 张图', '']
        head = [_name(m) for m in ms] + [f'{_name(m)} − {_name(ref)}' for m in ms[1:]]
        out += ['| 指标 | ' + ' | '.join(head) + ' |', '|' + '-|' * (len(head) + 1)]
        for k in ROWS:
            if not any(k in S.get(m, {}) for m in ms):
                continue
            row = [_cell(S[m].get(k)) for m in ms]
            row += [_cell(S.get('diff', {}).get(f'{m}-vs-{ref}', {}).get(k), True) for m in ms[1:]]
            out.append(f'| {k} | ' + ' | '.join(row) + ' |')
        out.append('')
        cl = S.get('clustering')
        if cl and any(isinstance(cl.get(m), dict) and 'umap' in cl[m] for m in ms):
            n_pools = next(cl[m]['n_pools'] for m in ms if isinstance(cl.get(m), dict) and 'n_pools' in cl[m])
            out += [f'| 聚类 ({n_pools} 个混合池, 均值 ± 池间 std) | ' + ' | '.join(_name(m) for m in ms) + ' |',
                    '|' + '-|' * (len(ms) + 1)]
            for mode in ('umap', 'none'):
                for k in ('bcubed_f', 'ari', 'nmi'):
                    vals = []
                    for m in ms:
                        e = (cl.get(m) or {}).get(mode)
                        vals.append(f'{e[k + "_mean"]:.3f} ± {e[k + "_std"]:.3f}' if e and k + '_mean' in e else '—')
                    out.append(f'| {mode} {k} | ' + ' | '.join(vals) + ' |')
            out.append('')
    return '\n'.join(out)


def max_cross_loss(res: dict, sets) -> dict:
    """{模型: (最大 F1 损失, '源->目标')}, 只看 sets 内部的有序对"""
    out = {}
    for m in res['models']:
        cl = ((res['threshold_transfer'].get(m) or {}).get('drift') or {}).get('cross_f1_loss', {})
        pairs = [(v, k) for k, v in cl.items() if k.split('->')[0] in sets and k.split('->')[1] in sets]
        out[m] = max(pairs) if pairs else (None, None)
    return out


def transfer_table(res: dict, sets) -> str:
    ms = res['models']
    out = ['| 指标 | ' + ' | '.join(_name(m) for m in ms) + ' |', '|' + '-|' * (len(ms) + 1)]
    mx = max_cross_loss(res, sets)
    out.append(f'| 最大跨集 F1 损失 ({res["split"]}; {", ".join(sets)}) | ' + ' | '.join(
        f'{v:.3f} ({k})' if v is not None else '—' for v, k in (mx[m] for m in ms)) + ' |')
    for st in sets:
        cells = []
        for m in ms:
            e = (res['threshold_transfer'].get(m) or {}).get('per_set', {}).get(st)
            d = (e or {}).get('default')
            cells.append(f"{d['f1']:.3f} / {e['best_f1']:.3f} (FPR {d['fpr']:.4f})" if d else '—')
        out.append(f'| 默认阈值 F1 / best F1，{st} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(out)


B3_SETS = ('bench2_b3p1', 'bench2_b3p2')


def regate_g2(res: dict, g2_sets=None) -> dict:
    """res 的深拷贝: max_cross_f1_loss 只在 g2_sets 之间重算, 去掉 B3 集合, b3p1_f1_loss 置 None (无效)"""
    from v3.eval import bench2
    g2_sets = tuple(g2_sets or bench2.G2_SETS)
    out = copy.deepcopy(res)
    out['sets'] = {k: v for k, v in out['sets'].items() if k not in B3_SETS}
    mx = max_cross_loss(res, g2_sets)
    for m, t in out.get('threshold_transfer', {}).items():
        have = set((t.get('per_set') or {}))
        t['max_cross_f1_loss'] = mx[m][0]
        t['max_cross_f1_loss_pair'] = mx[m][1]
        t['max_cross_f1_loss_sets'] = [s for s in g2_sets if s in have]
        t['b3p1_f1_loss'] = None
    return out


def g2fix_markdown(res: dict, g2_sets=None) -> str:
    from v3.scripts.bench2_eval import markdown
    r = regate_g2(res, g2_sets)
    sets = next(iter(r['threshold_transfer'].values()), {}).get('max_cross_f1_loss_sets', [])
    pairs = ', '.join(f'{_name(m)}: {t["max_cross_f1_loss_pair"]}' for m, t in r['threshold_transfer'].items())
    note = (f'> G2 修正版 (由已有 JSON 重算, 未重跑评测): max_cross_f1_loss 只在 {", ".join(sets)} 之间取最大; '
            f'B3 集合 (Pixiv 来源无效) 已去掉, b3p1_f1_loss 无效. 原 .md 的 G2 数值含 B3-P2, 已作废. 最差对: {pairs}')
    return note + '\n\n' + markdown(r)


if __name__ == '__main__':
    res = json.load(open(sys.argv[1]))
    args = sys.argv[2:]
    if args[:1] == ['--g2fix']:
        out = Path(args[1])
        if out.exists() or out.is_symlink():
            sys.exit(f'refusing to overwrite existing {out}')
        out.write_text(g2fix_markdown(res, args[2:] or None) + '\n')
        print(out.read_text())
    elif args[:1] == ['--transfer']:
        print(transfer_table(res, args[1:]))
    else:
        print(render(res, args or None))
