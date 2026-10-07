"""用 2025-09-18 对话日志里记录的 test_split 规模 (182,810 张; 前 100 类 22,613 张) 作为指纹, 在 witha / isek 各种组合里找原始划分
python repro/data/match_split_0918.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from match_witha_split import VARIANTS, build_labels, load_meta, split_and_sub, fingerprint  # noqa: E402

TARGET = (182810, 22613)
metas = {img: load_meta(img) for img in ('witha', 'wsfull')}
res = []
for name, (img, tr, te, multi, k) in VARIANTS.items():
    train, test = build_labels(metas[img], tr, te, multi, k)
    merged, test_split, sub = split_and_sub(train, test)
    fp = (len(test_split), len(sub))
    res.append((abs(fp[0] - TARGET[0]) + abs(fp[1] - TARGET[1]), name, fp, fingerprint(sub)))
    print(('MATCH ' if fp == TARGET else '      ') + f'{name}: test={fp[0]} test100={fp[1]} pairs={fingerprint(sub)[1]}', flush=True)
print('closest:', *sorted(res)[:5], sep='\n  ')
