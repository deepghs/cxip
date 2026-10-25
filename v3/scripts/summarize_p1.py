"""汇总消融: 每个实验最后一次评测 + 吞吐 + 逐层权重变化 (C′ 事后分析)。

python -m v3.scripts.summarize_p1 /data/neko/cxip_v3/exps/ccip_p1 [--md out.md]
"""
import argparse
import json
from pathlib import Path

import numpy as np


def last_jsonl(path):
    if not path.exists():
        return {}
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    return json.loads(lines[-1]) if lines else {}


def wc_profile(wc, depth=24, bins=((0, 8), (8, 16), (16, 24))):
    """按 block 分段的平均相对权重变化, 以及变化最大的 block。"""
    per_block = np.zeros(depth)
    for k, v in wc.items():
        if k.startswith('block') and not k.endswith('.norm'):
            per_block[int(k[5:7])] += v / 2  # attn 与 mlp 平均
    seg = [per_block[a:b].mean() for a, b in bins]
    return seg, int(per_block.argmax()), per_block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--md')
    args = ap.parse_args()
    rows = []
    for d in sorted(Path(args.root).iterdir()):
        if not d.is_dir() or d.name.startswith('_'):
            continue
        ev = last_jsonl(d / 'eval.jsonl')
        tr = last_jsonl(d / 'train.jsonl')
        status = 'DONE' if (d / 'DONE').exists() else 'FAILED' if (d / 'FAILED').exists() else 'running'
        wc = json.loads((d / 'weight_change.json').read_text()) if (d / 'weight_change.json').exists() else {}
        seg, top, _ = wc_profile(wc) if wc else ([np.nan] * 3, -1, None)
        rows.append(dict(name=d.name, status=status, samples=ev.get('samples', tr.get('samples', 0)),
                         pr_auc=ev.get('fast/pr_auc', np.nan), f1=ev.get('fast/best_f1', np.nan),
                         zs=ev.get('fast/zs_top1', np.nan), img_s=tr.get('img_s', np.nan),
                         wc_total=wc.get('_total', np.nan), wc_seg=seg, wc_top=top))
    head = '| 实验 | 状态 | 样本 | FAST PR-AUC | best F1 | zero-shot top1 | img/s | ΔW 总 | ΔW 0-7 / 8-15 / 16-23 (×1e-3) | ΔW 最大 block |'
    lines = [head, '|' + '-|' * 10]
    base = next((r['pr_auc'] for r in rows if r['name'] == 'base'), np.nan)
    for r in sorted(rows, key=lambda r: -np.nan_to_num(r['pr_auc'], nan=-1)):
        delta = f' ({r["pr_auc"] - base:+.4f})' if r['name'] != 'base' and not np.isnan(base) and not np.isnan(r['pr_auc']) else ''
        seg = ' / '.join(f'{x * 1e3:.2f}' for x in r['wc_seg'])
        lines.append(f'| {r["name"]} | {r["status"]} | {r["samples"]} | {r["pr_auc"]:.4f}{delta} | {r["f1"]:.4f} | '
                     f'{r["zs"]:.4f} | {r["img_s"]:.0f} | {r["wc_total"]:.2e} | {seg} | {r["wc_top"]} |')
    out = '\n'.join(lines)
    print(out)
    if args.md:
        Path(args.md).write_text(out + '\n')


if __name__ == '__main__':
    main()
