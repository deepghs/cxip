"""benchmark v2 CLI: 读缓存特征 (csip_full_eval.load_aligned), 跑 bench2.run, 写 JSON 并打印 markdown 表.

用法:
  python -m v3.scripts.bench2_eval --model REF FEATS_DIR SCALE BIAS THR [--model NAME FEATS_DIR SCALE BIAS THR ...] \
      [--split dev|test] [--n-boot 1000] --out res.json [--md res.md]
第一个 --model 是参照模型 (diff 列 = 其他模型 - 参照). 某集合缺特征文件 (FEATS_DIR/<集合>.npz) 时该集合跳过并注明.
markdown: 每个 B 集合一张表, 行 = 主指标, 列 = 各模型 (point [lo, hi]) + 每个非参照模型的 diff 列;
最后一张表是运行级的阈值迁移指标 (max_cross_f1_loss 等, 只有点估计).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from v3.eval import bench2

MAIN = ('pr_auc/<1y', 'pr_auc/1-3y', 'pr_auc/>=6y', 'map_at_r/1y', 'ap/<1y', 'date_mae', 'inject_auroc/bad_date',
        'same_char_fpr')
RUN_LEVEL = ('max_cross_f1_loss', 'b3p1_f1_loss')
SETS = ('bench2_b1', 'bench2_b2', 'bench2_b3p1', 'bench2_b3p2')
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
        lines += ['### threshold transfer (run level, point only)', '',
                  '| metric | ' + ' | '.join(head) + ' |', '|' + '-|' * (len(head) + 1)]
        for k in RUN_LEVEL:
            vals = [tt.get(m, {}).get(k) for m in names]
            row = [_num(v) for v in vals]
            row += [_num(v - vals[0], signed=True) if v is not None and vals[0] is not None else '—' for v in vals[1:]]
            lines.append(f'| {k} | ' + ' | '.join(row) + ' |')
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
    ap.add_argument('--out', required=True)
    ap.add_argument('--md')
    a = ap.parse_args(argv)
    from v3.scripts.csip_full_eval import load_aligned

    need = list(dict.fromkeys(
        [s for name in a.sets for s in (('bench2_b1', name) if name == bench2.CROSS_SET else (name,))] + a.legacy))
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
                     dfs=dfs)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=_json_default))
    md = markdown(res)
    print(md)
    if a.md:
        Path(a.md).write_text(md + '\n')


if __name__ == '__main__':
    main()
