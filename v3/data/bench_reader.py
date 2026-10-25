"""Decode throughput of IndexedTarDataset + PKBatchSampler with a multi-worker DataLoader.

    python -m v3.data.bench_reader --root /data/neko/cxip_repro/data/ccip_merged_nd \
        --index /data/neko/cxip_v3/data/ccip-merged-v1-index/train.parquet --workers 32
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from v3.data.indexed_tar import IndexedTarDataset, list_collate
from v3.data.pk_sampler import PKBatchSampler


class ResizeToTensor:
    """Pad-free bicubic resize to res x res + uint8 CHW tensor (cheap stand-in for real aug)."""

    def __init__(self, res):
        self.res = res

    def __call__(self, im):
        from PIL import Image
        im = im.resize((self.res, self.res), Image.BICUBIC)
        return torch.from_numpy(np.array(im)).permute(2, 0, 1)


def run(root, index, workers, P, K, n_batches, res, warmup, jpeg_draft=None):
    tf = ResizeToTensor(res) if res else None
    ds = IndexedTarDataset(root, index, transform=tf, jpeg_draft=jpeg_draft)
    sampler = PKBatchSampler(ds.labels, P=P, K=K, seed=int(time.time()), num_batches=n_batches + warmup)
    dl = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=workers, collate_fn=list_collate,
                                     prefetch_factor=4, persistent_workers=False)
    n, t0 = 0, None
    nbytes = 0
    for b, batch in enumerate(dl):
        if b == warmup:
            t0, n = time.time(), 0
        n += len(batch['label'])
        if t0 is not None:
            nbytes += int(ds.sizes[batch['index'].numpy()].sum())
    dt = time.time() - t0
    return {'root': str(root), 'index': str(index), 'workers': workers, 'batch': P * K, 'res': res,
            'jpeg_draft': jpeg_draft,
            'images': n, 'seconds': round(dt, 2), 'img_per_s': round(n / dt, 1),
            'MB_per_s': round(nbytes / dt / 1e6, 1)}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--index', required=True)
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--P', type=int, default=64)
    ap.add_argument('--K', type=int, default=4)
    ap.add_argument('--batches', type=int, default=200)
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--res', type=int, nargs='+', default=[0, 256])
    ap.add_argument('--jpeg-draft', action='store_true', help='jpeg_draft = res')
    a = ap.parse_args()
    for res in a.res:
        print(json.dumps(run(a.root, a.index, a.workers, a.P, a.K, a.batches, res, a.warmup,
                             res if a.jpeg_draft and res else None)), flush=True)
