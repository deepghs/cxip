"""
切分换成 danbooru-artist-split-v2 后, 训练集里多了一部分之前没提特征的图片。
1) python assemble_feats_v2.py need   -> 写出缺特征的图片列表 extra_labels.json (交给 extract_ccip_feats.py --label-file)
2) python assemble_feats_v2.py build  -> 合并旧特征和新特征, 写出只包含 v2 train_split 的 train_ccip_feats_v2/
   (与 extract_ccip_feats.py 输出格式相同, scripts/char_labels/ccip_cls.py --feat-npy 直接读)
可选第二个参数指定其他切分目录 (例如 danbooru-artist-split-0918) 和其中的 train split 文件名:
   python assemble_feats_v2.py build danbooru-artist-split-0918 train_split_avail.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get('REPRO_ROOT', '/data/neko/cxip_repro')) / 'data'
SPLIT = DATA / 'danbooru-artist-split-v2'
OLD = DATA / 'danbooru-w640-ws-full-artist/train_ccip_feats'
EXTRA = DATA / 'danbooru-artist-split-v2/train_ccip_feats_extra'
OUT = DATA / 'danbooru-artist-split-v2/train_ccip_feats'


def index(feat_dir):
    idx = {}
    for files in sorted(Path(feat_dir).glob('files_r*.json')):
        arr = np.load(feat_dir / files.name.replace('files_', 'feats_').replace('.json', '.npy'), mmap_mode='r')
        with open(files) as f:
            for row, name in enumerate(json.load(f)):
                idx[name] = (arr, row)
    return idx


def main(cmd, split=None, train_file='train_split.json'):
    global SPLIT, OUT
    if split is not None:
        SPLIT = DATA / split
        OUT = SPLIT / 'train_ccip_feats'
    with open(SPLIT / train_file) as f:
        train = json.load(f)
    if cmd == 'need':
        have = {**index(OLD), **(index(EXTRA) if EXTRA.exists() else {})}
        extra = {k: v for k, v in train.items() if k not in have}
        with open(SPLIT / 'extra_labels.json', 'w') as f:
            json.dump(extra, f)
        print(f'{len(extra)} images need features -> {SPLIT / "extra_labels.json"}')
        return

    idx = {**index(OLD), **(index(EXTRA) if EXTRA.exists() else {})}
    if split is not None and (SPLIT / 'train_ccip_feats_extra').exists():
        idx.update(index(SPLIT / 'train_ccip_feats_extra'))
    names = sorted(train)
    missing = [n for n in names if n not in idx]
    assert not missing, f'{len(missing)} images still without features, e.g. {missing[:3]}'
    OUT.mkdir(parents=True, exist_ok=True)
    world = 4
    for r in range(world):
        part = names[r::world]
        feats = np.lib.format.open_memmap(OUT / f'feats_r{r}.npy', mode='w+', dtype=np.float16, shape=(len(part), 768))
        for start in range(0, len(part), 65536):
            chunk = part[start:start + 65536]
            feats[start:start + len(chunk)] = np.stack([idx[n][0][idx[n][1]] for n in chunk])
        feats.flush()
        with open(OUT / f'files_r{r}.json', 'w') as f:
            json.dump(part, f)
    print(f'{len(names)} features -> {OUT}')


if __name__ == '__main__':
    main(*sys.argv[1:])
