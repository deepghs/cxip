"""
把合并后的 CCIP 数据集 (data/ccip_merged) 转成 neko-data 格式 (github.com/IrisRainbowNeko/neko-data >= 0.3.0)
    python experiments/ccip_merge/to_neko_data.py [--src data/ccip_merged] [--out data/ccip_merged_nd]
    python experiments/ccip_merge/to_neko_data.py verify [--out ...]

- 一个目录下 train/val/test 三个 split, 共用 manifest.json
- 同一角色 (label) 的图连续写入同一个分片 (group_key='label'), 类别顺序按 seed 打乱
- 每个样本: 原始图片字节 + json (label, name, zid, source, orig_path, orig_cid, crop, width, height) + txt (角色名)
训练时:
    RainbowLabeledImageSource.from_manifest(out/'manifest.json', split='train',
        group_key='label', group_shuffle=2048, group_chunk_size=16, cache_root=...)
"""
import argparse
import io
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image

from neko_data.build.job import BuildJob, build_dataset
from neko_data.contract.records import NormalizedSample

ROOT = Path(os.environ.get('REPRO_ROOT', '/data/neko/cxip_repro'))
DATASET_ID = 'ccip-merged'
VERSION = 'v1'


def _clean(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if hasattr(v, 'item'):
        v = v.item()
    return v


def iter_samples(src, split, seed=0):
    src = Path(src)
    idx = pd.read_parquet(src / 'index.parquet')
    idx = idx[idx['split'] == split]
    names = {c['label']: c['name'] for c in json.load(open(src / 'classes.json'))}
    labels = sorted(idx['label'].unique().tolist())
    random.Random(f'{seed}-{split}').shuffle(labels)
    groups = dict(tuple(idx.groupby('label', sort=False)))
    handles = {}
    t0, n = time.time(), 0
    try:
        for label in labels:
            for r in groups[label].itertuples(index=False):
                fh = handles.get(r.shard)
                if fh is None:
                    fh = handles[r.shard] = open(src / 'shards' / r.shard, 'rb')
                fh.seek(int(r.offset))
                data = fh.read(int(r.size))
                with Image.open(io.BytesIO(data)) as im:
                    width, height = im.size
                name = names[int(label)]
                yield NormalizedSample(
                    sample_key=Path(r.key).stem,
                    image=data,
                    image_extension=r.ext,
                    split=split,
                    source_id=f'ccip_merged/{r.source}',
                    caption=name,
                    metadata={
                        'label': int(label), 'name': name, 'zid': _clean(r.zid), 'source': r.source,
                        'orig_path': r.orig_path, 'orig_cid': _clean(r.orig_cid), 'crop': bool(r.crop),
                    },
                    width=width, height=height,
                )
                n += 1
                if n % 50000 == 0:
                    print(f'{split} {n}/{len(idx)} {n / (time.time() - t0):.0f} img/s', flush=True)
    finally:
        for fh in handles.values():
            fh.close()


def build(src, out, max_shard_size, seed, splits):
    src, out = Path(src), Path(out)
    classes = json.load(open(src / 'classes.json'))
    out.mkdir(parents=True, exist_ok=True)
    for name in ['classes.json', 'variants.json', 'report.md']:
        if (src / name).exists():
            shutil.copy2(src / name, out / name)
    meta = {
        'task': 'ccip', 'num_classes': len(classes), 'label_key': 'label', 'group_key': 'label',
        'classes_file': 'classes.json', 'source': 'zerochan-character-w640-ws-m50-800k + character_similarity v1_pruned/v2',
        'seed': seed,
    }
    for split in splits:
        m = build_dataset(BuildJob(
            source=iter_samples(src, split, seed), output_dir=out, dataset_id=DATASET_ID, version=VERSION,
            split=split, max_shard_size=max_shard_size, group_key='label', overwrite=True,
            skip_invalid_samples=False, metadata=meta,
        ))
        print(split, 'done', m.splits, flush=True)


def verify(out):
    from neko_data.integrations.rainbow import RainbowLabeledImageSource
    from neko_data.runtime import RuntimeContext
    out = Path(out)
    for split in ['train', 'val', 'test']:
        src = RainbowLabeledImageSource.from_manifest(
            out / 'manifest.json', runtime=RuntimeContext(), split=split, cache_strategy='disabled',
            prefetch_workers=0, group_shuffle=256, group_chunk_size=16,
        )
        labels, t0 = [], time.time()
        for i, item in enumerate(src):
            if i < 200:
                item['image'].convert('RGB')
            labels.append(item['label'])
            if len(labels) >= 20000:
                break
        runs = sum(1 for i in range(len(labels)) if i == 0 or labels[i] != labels[i - 1])
        print(f'{split}: {len(src)} samples, num_classes={src.num_classes}, first {len(labels)} items '
              f'{len(set(labels))} labels in {runs} runs ({len(labels) / runs:.1f}/run), '
              f'{len(labels) / (time.time() - t0):.0f} it/s')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', nargs='?', default='build', choices=['build', 'verify'])
    ap.add_argument('--src', default=ROOT / 'data/ccip_merged')
    ap.add_argument('--out', default=ROOT / 'data/ccip_merged_nd')
    ap.add_argument('--max-shard-size', default='1GiB')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--splits', default='train,val,test')
    a = ap.parse_args()
    if a.cmd == 'build':
        build(a.src, a.out, a.max_shard_size, a.seed, a.splits.split(','))
    else:
        verify(a.out)
    sys.exit(0)
