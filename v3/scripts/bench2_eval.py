"""benchmark v2 CLI: 读缓存特征 (csip_full_eval.load_aligned), 跑 bench2.run, 写 JSON 并打印 markdown 表.

用法:
  python -m v3.scripts.bench2_eval --model REF FEATS_DIR SCALE BIAS THR [--model NAME FEATS_DIR SCALE BIAS THR ...] \
      [--split dev|test] [--n-boot 1000] [--sets bench2_b1 bench2_b2] [--g2-sets ...] --out res.json [--md res.md]
第一个 --model 是参照模型 (diff 列 = 其他模型 - 参照). 某集合缺特征文件 (FEATS_DIR/<集合>.npz) 时该集合跳过并注明.
markdown: 每个 B 集合一张表, 行 = 主指标, 列 = 各模型 (point [lo, hi]) + 每个非参照模型的 diff 列;
最后一张表是运行级的阈值迁移指标 (G2 的 max_cross_f1_loss, 按画师 bootstrap 的 CI + 配对差; 旧 JSON 只有点估计,
表头相应注明无 CI).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from v3.eval import bench2

MAIN = ('pr_auc/<1y', 'pr_auc/<1y_xday', 'pr_auc/1-3y', 'pr_auc/>=6y', 'map_at_r/1y', 'ap/<1y', 'date_mae',
        'inject_auroc/bad_date', 'same_char_fpr', 'same_char_fpr@fpr1e-3', 'same_char_ratio@fpr1e-3')
RUN_LEVEL = ('max_cross_f1_loss',)
# v4 只有 B1 / B2 (Pixiv / B3 已去掉, 报告 §6.1)
SETS = ('bench2_b1', 'bench2_b2')
LEGACY = ('test100_0918', 'csip_eval_public', 'csip_unseen_v1')


def _num(x, signed=False) -> str:
    if x is None:
        return '—'
    s = '+' if signed else ''
    return f'{x:{s}.4f}' if abs(x) < 0.1 else f'{x:{s}.3f}'


def _cell(v: dict | None, signed=False) -> str:
    if not v or v.get('point') is None:
        return '—'
    s = _num(v['point'], signed)
    if v.get('lo') is not None:
        s += f' [{_num(v["lo"], signed)}, {_num(v["hi"], signed)}]'
    if v.get('p_gt0') is not None:
        s += f' p>0={v["p_gt0"]:.2f}'
    return s


def _as_stat(v) -> dict | None:
    """运行级指标: 新 JSON 为 {point, lo, hi, n_valid}, 旧 JSON 为浮点数"""
    if v is None or isinstance(v, dict):
        return v
    return dict(point=float(v))


def run_level_cells(tt: dict, names: list, k: str) -> list[str]:
    """运行级指标 k 的一行: 各模型 point [lo, hi] + 每个非参照模型相对 names[0] 的配对差 (p>0=...).
    旧 JSON (浮点, 无 'diff') 的差列只给点估计之差. bench2_tables.g2_table 共用此函数, 两处格式一致"""
    ref = names[0]
    vals = [_as_stat(tt.get(m, {}).get(k)) for m in names]
    row = [_cell(v) for v in vals]
    for m, v in zip(names[1:], vals[1:]):
        d = tt.get('diff', {}).get(f'{m}-vs-{ref}', {}).get(k)
        if d is None and v and vals[0] and v.get('point') is not None and vals[0].get('point') is not None:
            d = dict(point=v['point'] - vals[0]['point'])               # 旧 JSON: 只有点估计之差
        row.append(_cell(d, signed=True))
    return row


def has_ci(tt: dict, names: list, k: str = 'max_cross_f1_loss') -> bool:
    """新 JSON: 运行级指标是 {point, lo, hi, n_valid} (按画师 bootstrap); 旧 JSON 是浮点数"""
    return any(isinstance(tt.get(m, {}).get(k), dict) for m in names)


def markdown(res: dict) -> str:
    names = res['models']
    ref = names[0]
    head = names + [f'{m} − {ref}' for m in names[1:]]
    lines = []
    for s, r in res['sets'].items():
        c = r['composition']
        lines += [f'### {s} ({res["split"]}): {c["n_artists"]} artists, {c["n_images"]} images', '',
                  '| metric | ' + ' | '.join(head) + ' |', '|' + '-|' * (len(head) + 1)]
        for k in MAIN:
            row = [_cell(r.get(m, {}).get(k)) for m in names]
            row += [_cell(r.get('diff', {}).get(f'{m}-vs-{ref}', {}).get(k), signed=True) for m in names[1:]]
            lines.append(f'| {k} | ' + ' | '.join(row) + ' |')
        skipped = [k for k, st in r.get('inject', {}).items() if st.get('status') != 'ok']
        if skipped:
            lines.append(f'\ninject skipped: {", ".join(skipped)}')
        lines.append('')
    tt = res.get('threshold_transfer', {})
    if tt:
        note = ('G2 CI = artist bootstrap of B1 / B2, csip_eval_public fixed' if has_ci(tt, names)
                else 'old JSON: point estimates only, no CI')
        lines += [f'### threshold transfer (run level; {note})', '',
                  '| metric | ' + ' | '.join(head) + ' |', '|' + '-|' * (len(head) + 1)]
        for k in RUN_LEVEL:
            lines.append(f'| {k} | ' + ' | '.join(run_level_cells(tt, names, k)) + ' |')
        lines.append('')
    if res.get('skipped'):
        lines.append('skipped: ' + '; '.join(f'{k} ({v})' for k, v in res['skipped'].items()))
    return '\n'.join(lines)


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', nargs=5, action='append', required=True,
                    metavar=('NAME', 'FEATS_DIR', 'SCALE', 'BIAS', 'THR'))
    ap.add_argument('--split', choices=('dev', 'test'), default='dev')
    ap.add_argument('--n-boot', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sets', nargs='+', default=list(SETS))
    ap.add_argument('--legacy', nargs='*', default=list(LEGACY))
    ap.add_argument('--g2-sets', nargs='+', default=list(bench2.G2_SETS),
                    help='G2 口径 (max_cross_f1_loss) 包含的集合')
    ap.add_argument('--out', required=True)
    ap.add_argument('--md')
    a = ap.parse_args(argv)
    taken = [x for x in (a.out, a.md) if x and (Path(x).exists() or Path(x).is_symlink())]
    if taken:
        sys.exit(f'refusing to overwrite existing {taken}; write new files')
    from v3.scripts.csip_full_eval import load_aligned

    need = list(dict.fromkeys(list(a.sets) + a.legacy))
    have = {name: {s for s in need if (Path(fdir) / f'{s}.npz').exists()} for name, fdir, *_ in a.model}
    dfs = {s: bench2.load_set(s)[0] for s in need if any(s in h for h in have.values())}   # 只读有特征的集合, 只读一次
    models = []
    for name, fdir, scale, bias, thr in a.model:
        feats = {s: load_aligned(fdir, s, dfs[s]) for s in need if s in have[name]}
        miss = [s for s in need if s not in feats]
        if miss:
            print(f'[bench2_eval] {name}: no features for {miss}')
        models.append(bench2.ModelFeats(name, feats, float(scale), float(bias), float(thr)))
    res = bench2.run(models, split=a.split, sets=tuple(a.sets), legacy=tuple(a.legacy), n_boot=a.n_boot, seed=a.seed,
                     dfs=dfs, g2_sets=tuple(a.g2_sets))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=_json_default))
    md = markdown(res)
    print(md)
    if a.md:
        Path(a.md).write_text(md + '\n')


if __name__ == '__main__':
    main()
