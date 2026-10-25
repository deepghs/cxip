"""C′ 离线 CKA: 每个实验末尾的逐层特征 (layer_feats.pt) 与原始 AnimeDINOv3 同图特征的线性 CKA。

用法: python -m v3.scripts.cka_report /data/neko/cxip_v3/exps/ccip_p1 [ccip|csip]
CKA 越低说明该层表征被微调改动越多; 与 ΔW (学习率支配) 互补。
"""
import sys
from pathlib import Path

import torch

from v3.analysis import linear_cka

LAYERS = (4, 8, 12, 16, 20, 22, 24)


def main(root, tag=None):
    root = Path(root)
    tag = tag or ('csip' if 'csip' in root.name else 'ccip')
    ref = torch.load(f'/data/neko/cxip_v3/analysis/ref_frozen_{tag}_layer_feats.pt')
    print(f'| 实验 | 特征 | ' + ' | '.join(f'L{l}' for l in LAYERS) + ' |')
    print('|-|-|' + '-|' * len(LAYERS))
    for d in sorted(root.iterdir()):
        f = d / 'layer_feats.pt'
        if d.name.startswith('_') or not f.exists():
            continue
        feats = torch.load(f)
        for kind in ('cls', 'mean'):
            vals = [linear_cka(feats[kind][:, l - 1].float(), ref[kind][:, l - 1].float()) for l in LAYERS]
            print(f'| {d.name} | {kind} | ' + ' | '.join(f'{v:.3f}' for v in vals) + ' |')


if __name__ == '__main__':
    main(*sys.argv[1:])
