"""
读取合并后的 CCIP 数据集 (data/ccip_merged): tar 分片 + index.parquet 随机访问, 不需要解包
    ds = CCIPMerged('/data/neko/cxip_repro/data/ccip_merged', split='train')
    img, label = ds[0]
python experiments/ccip_merge/reader.py   # 随机抽样校验: 能解码, 偏移正确, 每类训练图 >= 2
"""
import io
import json
import os
import random
import sys
from pathlib import Path

import pandas as pd
from PIL import Image


class CCIPMerged:
    def __init__(self, root, split='train'):
        self.root = Path(root)
        idx = pd.read_parquet(self.root / 'index.parquet')
        self.idx = idx[idx['split'] == split].reset_index(drop=True)
        self.classes = json.load(open(self.root / 'classes.json'))
        self._fh = {}

    def __len__(self):
        return len(self.idx)

    def read_bytes(self, i):
        r = self.idx.iloc[i]
        fh = self._fh.get(r['shard'])
        if fh is None:  # 每个 worker 进程各自打开
            fh = self._fh[r['shard']] = open(self.root / 'shards' / r['shard'], 'rb')
        fh.seek(int(r['offset']))
        return fh.read(int(r['size']))

    def __getitem__(self, i):
        return Image.open(io.BytesIO(self.read_bytes(i))).convert('RGB'), int(self.idx.iloc[i]['label'])


if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get('REPRO_ROOT', '/data/neko/cxip_repro'), 'data/ccip_merged')
    for split in ['train', 'val', 'test']:
        ds = CCIPMerged(root, split)
        rs = random.Random(0)
        for i in rs.sample(range(len(ds)), 2000):
            img, lb = ds[i]
            assert img.size[0] > 0 and 0 <= lb < len(ds.classes)
        print(split, len(ds), 'images,', ds.idx['label'].nunique(), 'classes, 2000 random reads ok')
    tr = CCIPMerged(root, 'train').idx.groupby('label').size()
    assert tr.min() >= 2, tr.min()
    print('train images per class: min', tr.min(), 'median', int(tr.median()), 'max', tr.max())
