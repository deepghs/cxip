"""汇总 eval_exp 写出的 eval/suite.json (CCIP 与 CSIP 通用), 打印 markdown 表。

用法: python -m v3.scripts.summarize_suite /data/neko/cxip_v3/exps/csip_p2 [--sub eval_native]
CCIP 列: openset FULL/FAST PR-AUC + 分组 + I_style; CSIP 列: 各测试集 PR-AUC + I_content / I_char。
"""
import argparse
import json
from pathlib import Path

HF_CSIP_BIAS = '/data/neko/cxip_repro/results/bias_hf.json'
HF_CSIP_PR = {  # test100_0918 / csip_eval_public (cxip_repro/results/hf_csip*_*.json)
    'HF CSIP v1': (0.7455, 0.7727), 'HF CSIP v2': (0.7919, 0.8414), 'HF CSIP v2.1': (0.7445, 0.8723)}


def fmt(v, nd=4):
    return '' if v is None else f'{v:.{nd}f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--sub', default='eval')
    args = ap.parse_args()
    rows = []
    for d in sorted(Path(args.root).iterdir()):
        f = d / args.sub / 'suite.json'
        if d.name.startswith('_') or not f.exists():
            continue
        s = json.loads(f.read_text())
        rows.append((d.name, s))
    if not rows:
        print('no suite.json')
        return
    kind = rows[0][1].get('bias', {}).get('kind') or ('csip' if 'test100_0918' in rows[0][1]['datasets'] else 'ccip')
    if kind == 'csip':
        print('| 实验 | test100_0918 | test100 | csip_eval_public | I_content | I_char |')
        print('|-|-|-|-|-|-|')
        hb = json.loads(Path(HF_CSIP_BIAS).read_text())
        for name, key in (('HF CSIP v1', 'dghstmp/AnimeCSIP'), ('HF CSIP v2', 'dghstmp/AnimeCSIP-v2'),
                          ('HF CSIP v2.1', 'dghstmp/AnimeCSIP-v2.1')):
            t, p = HF_CSIP_PR[name]
            print(f'| {name} | {t:.4f} | (泄漏) | {p:.4f} | {hb[key]["content"]["I"]:.3f} | {hb[key]["char"]["I"]:.3f} |')
        for name, s in rows:
            ds, b = s['datasets'], s.get('bias', {})
            pr = [fmt(ds.get(k, {}).get('pr_auc')) for k in ('test100_0918', 'test100', 'csip_eval_public')]
            print(f'| {name} | ' + ' | '.join(pr) + f' | {fmt(b.get("content", {}).get("I"), 3)} | {fmt(b.get("char", {}).get("I"), 3)} |')
    else:
        print('| 实验 | FULL | FAST | 新作品 | 同作品难例 | 骨干见过 | 骨干没见过 | I_style |')
        print('|-|-|-|-|-|-|-|-|')
        for name, s in rows:
            full, fast = s['datasets'].get('openset_full', {}), s['datasets'].get('openset_fast', {})
            g = lambda k: fmt((full.get('groups') or {}).get(k, {}).get('pr_auc'))
            b = s.get('bias', {})
            print(f'| {name} | {fmt(full.get("pr_auc"))} | {fmt(fast.get("pr_auc"))} | {g("new_franchise")} | '
                  f'{g("hard")} | {g("backbone_seen")} | {g("backbone_unseen")} | '
                  f'{fmt(b.get("style", {}).get("I"), 3)} |')


if __name__ == '__main__':
    main()
