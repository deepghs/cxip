"""M0-0.6 冻结特征评测: 不同输入分辨率下 AnimeDINOv3 原始特征的角色区分能力 (frozen baseline)。

顺序流式读 neko-data webdataset 的 test 分片, 按类抽样 (max_classes × max_per_class),
提取 CLS / mean(patch) / 若干中间层 CLS, 输出 PR-AUC、best-F1、zero-shot top1、留一 kNN top1, 并缓存特征。

python -m v3.bench.frozen_eval --res 256 384 512 --taps 12 18
"""
import argparse
import io
import json
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from v3.bench.bench_backbone import MODEL_PATH
from v3.models.backbone import AnimeDINOBackbone
from v3.transforms import EvalTransform

DATA = Path('/data/neko/cxip_repro/data/ccip_merged_nd')
OUT = Path('/data/neko/cxip_v3/features/frozen_eval')


def sample_split(split, max_classes, max_per_class, seed=0):
    meta = pd.concat([pd.read_parquet(p) for p in sorted((DATA / 'metadata' / split).glob('*.parquet'))])
    rng = np.random.RandomState(seed)
    counts = meta['label'].value_counts()
    cand = counts[counts >= 2].index.values
    keep = set(rng.choice(cand, min(max_classes, len(cand)), replace=False).tolist())
    sub = meta[meta['label'].isin(keep)]
    sub = sub.groupby('label', group_keys=False).apply(lambda g: g.sample(min(len(g), max_per_class), random_state=seed))
    return sub.reset_index(drop=True)


def load_images(split, sub):
    want = {f'{k}.{e}': i for i, (k, e) in enumerate(zip(sub['sample_key'], sub['image_extension']))}
    images = [None] * len(sub)
    for tar_path in sorted((DATA / 'wds' / split).glob('*.tar')):
        with tarfile.open(tar_path) as tf:
            for m in tf:
                i = want.get(m.name)
                if i is not None:
                    images[i] = tf.extractfile(m).read()
    assert all(x is not None for x in images), 'missing images'
    return images


def pair_metrics(z, labels):
    from sklearn.metrics import average_precision_score, precision_recall_curve
    z = F.normalize(z.cuda().float(), dim=-1)
    y = torch.as_tensor(labels, device=z.device)
    s = z @ z.t()
    iu = torch.triu_indices(len(z), len(z), 1, device=z.device)
    sc = s[iu[0], iu[1]].cpu().numpy()
    tg = (y[iu[0]] == y[iu[1]]).cpu().numpy()
    ap = float(average_precision_score(tg, sc))
    p, r, _ = precision_recall_curve(tg, sc)
    f1 = float(np.max(2 * p * r / np.clip(p + r, 1e-12, None)))
    labels = np.asarray(labels)
    _, first = np.unique(labels, return_index=True)
    rest = np.setdiff1d(np.arange(len(z)), first)
    zs = float((labels[first][(z[rest] @ z[first].t()).argmax(1).cpu().numpy()] == labels[rest]).mean())
    s.fill_diagonal_(-2)
    knn = float((labels[s.argmax(1).cpu().numpy()] == labels).mean())
    return dict(pr_auc=round(ap, 4), best_f1=round(f1, 4), zs_top1=round(zs, 4), knn_top1=round(knn, 4))


@torch.no_grad()
def extract(model, images, res, bs, taps):
    tf = EvalTransform(res)
    feats = {'cls': [], 'mean': [], **{f'cls_l{t}': [] for t in taps}, **{f'mean_l{t}': [] for t in taps}}
    for i in range(0, len(images), bs):
        x = torch.stack([tf(Image.open(io.BytesIO(b))) for b in images[i:i + bs]]).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(x)
        feats['cls'].append(out['cls'].float().cpu())
        feats['mean'].append(out['patches'].float().mean(1).cpu())
        for t in taps:
            h = out['taps'][t].float()
            feats[f'cls_l{t}'].append(h[:, 0].cpu())
            feats[f'mean_l{t}'].append(h[:, 1:].mean(1).cpu())
    return {k: torch.cat(v) for k, v in feats.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test')
    ap.add_argument('--res', type=int, nargs='+', default=[256, 384, 512])
    ap.add_argument('--taps', type=int, nargs='*', default=[12, 18])
    ap.add_argument('--max-classes', type=int, default=1500)
    ap.add_argument('--max-per-class', type=int, default=6)
    ap.add_argument('--bs', type=int, default=64)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True

    t0 = time.time()
    sub = sample_split(args.split, args.max_classes, args.max_per_class)
    images = load_images(args.split, sub)
    sub.drop(columns=['metadata_json']).to_parquet(OUT / f'{args.split}_subset.parquet')
    print(f'{len(sub)} images / {sub.label.nunique()} classes loaded in {time.time() - t0:.0f}s', flush=True)

    model = AnimeDINOBackbone(MODEL_PATH, tap_layers=args.taps).cuda().eval()
    labels = sub['label'].values
    for res in args.res:
        t0 = time.time()
        feats = extract(model, images, res, args.bs, args.taps)
        torch.save({k: v.half() for k, v in feats.items()}, OUT / f'{args.split}_r{res}.pt')
        feats['cls+mean'] = torch.cat([F.normalize(feats['cls'], dim=-1), F.normalize(feats['mean'], dim=-1)], 1)
        for name, z in feats.items():
            rec = dict(split=args.split, res=res, feat=name, n=len(z), classes=int(sub.label.nunique()),
                       **pair_metrics(z, labels))
            print(json.dumps(rec), flush=True)
            with open(OUT / 'results.jsonl', 'a') as f:
                f.write(json.dumps(rec) + '\n')
        print(f'res {res} done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
