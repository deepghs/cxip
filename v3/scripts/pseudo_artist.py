"""用 CSIP 模型给 CCIP 训练图聚类出 "伪画师", 供 CCIP 采样器做 "同画风、不同角色" 的交叉成组 (data.cross_key)。

CCIP 的同类正样本大多同画风, 训练越久画风泄漏越高 (l12b 0.140 → l12b_x4 0.150); CSIP 用 "同角色不同画师"
的交叉成组让负例内容相同, 这里是它的镜像: 让负例画风相同, 直接惩罚靠画风认角色。

两步 (可分开跑):
  extract: 每张 GPU 一个进程, native_full 协议 (短边 ≤640 整图, 按尺寸分桶) 提取 CSIP 向量 → <out>/feats.npy (fp16, 已归一化)
  cluster: 球面 k-means (torch, 单卡) → <out>/train.parquet = 原索引 + pseudo_artist (int) + pseudo_artist_cos

python -m v3.scripts.pseudo_artist extract --csip-exp <exp> --gpus 4,5,6,7
python -m v3.scripts.pseudo_artist cluster --k 8192 --gpu 4
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

CCIP_ROOT = '/data/neko/cxip_repro/data/ccip_merged_nd'
INDEX = '/data/neko/cxip_v3/data/ccip-merged-v1-index/index/train.parquet'
OUT = '/data/neko/cxip_v3/data/ccip-pseudo-artist-v1'


def _collate(batch):
    return [b['image'] for b in batch], [b['index'] for b in batch]


def _extract_worker(rank, gpus, args):
    from torch.utils.data import DataLoader

    from v3.data.indexed_tar import IndexedTarDataset
    from v3.eval_exp import load_exp_model
    from v3.transforms import NativeFullEval

    torch.cuda.set_device(int(gpus[rank]))
    torch.set_num_threads(4)
    df = pd.read_parquet(args.index)
    rows = np.array_split(np.arange(len(df)), len(gpus))[rank]
    out = Path(args.out) / f'feats_part{rank}.npy'
    if out.exists():
        return
    model, _ = load_exp_model(Path(args.csip_exp))
    model = model.cuda().eval().to(torch.bfloat16)
    ds = IndexedTarDataset(args.root, df.iloc[rows].reset_index(drop=True), transform=NativeFullEval(short_side=640))
    dl = DataLoader(ds, batch_size=args.bs, num_workers=args.workers, collate_fn=_collate, prefetch_factor=4)
    feats = np.zeros((len(rows), model.matryoshka[0]), np.float16)
    budget = args.bs * 24 * 24  # 每次前向的 token 上限 = bs 张 384² 图
    t0 = time.time()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for n, (xs, idx) in enumerate(dl):
            buckets = {}
            for x, i in zip(xs, idx):
                buckets.setdefault(tuple(x.shape), []).append((x, i))
            for shape, items in buckets.items():
                step = max(1, budget // ((shape[1] // 16) * (shape[2] // 16)))
                for k in range(0, len(items), step):
                    sel = items[k:k + step]
                    e = model(torch.stack([x for x, _ in sel]).cuda(non_blocking=True))['embedding']
                    feats[[i for _, i in sel]] = F.normalize(e.float(), dim=-1).cpu().numpy().astype(np.float16)
            if rank == 0 and n % 200 == 0:
                done = (n + 1) * args.bs
                print(f'[{rank}] {done}/{len(rows)} {done / (time.time() - t0):.0f} img/s', flush=True)
    np.save(str(out) + '.tmp.npy', feats)
    os.replace(str(out) + '.tmp.npy', out)


def extract(args):
    import torch.multiprocessing as mp
    gpus = args.gpus.split(',')
    Path(args.out).mkdir(parents=True, exist_ok=True)
    mp.spawn(_extract_worker, args=(gpus, args), nprocs=len(gpus), join=True)
    feats = np.concatenate([np.load(Path(args.out) / f'feats_part{r}.npy') for r in range(len(gpus))])
    np.save(Path(args.out) / 'feats.npy', feats)
    for r in range(len(gpus)):
        (Path(args.out) / f'feats_part{r}.npy').unlink()
    (Path(args.out) / 'extract_info.json').write_text(json.dumps(dict(
        csip_exp=args.csip_exp, index=args.index, protocol='native_full short_side=640', n=len(feats),
        dim=int(feats.shape[1]), time=time.strftime('%Y-%m-%d %H:%M:%S')), indent=1))
    print('feats', feats.shape)


@torch.no_grad()
def spherical_kmeans(x: torch.Tensor, k: int, iters: int = 25, seed: int = 0, chunk: int = 65536):
    """x: (N, D) 已归一化 (fp16, GPU)。返回 (assign (N,), cos-to-centroid (N,), centroids (k, D))。"""
    g = torch.Generator(device='cpu').manual_seed(seed)
    c = x[torch.randperm(len(x), generator=g)[:k].to(x.device)].float()
    for it in range(iters):
        assign = torch.empty(len(x), dtype=torch.long, device=x.device)
        cos = torch.empty(len(x), device=x.device)
        for s in range(0, len(x), chunk):
            v, a = (x[s:s + chunk].float() @ c.T).max(1)
            assign[s:s + chunk], cos[s:s + chunk] = a, v
        new = torch.zeros_like(c).index_add_(0, assign, x.float())
        cnt = torch.bincount(assign, minlength=k)
        empty = cnt == 0
        if empty.any():  # 空簇: 换成离自己中心最远的点
            far = cos.argsort()[:int(empty.sum())]
            new[empty] = x[far].float()
        c = F.normalize(new, dim=-1)
        print(f'iter {it} mean cos {cos.mean().item():.4f} empty {int(empty.sum())}', flush=True)
    return assign.cpu().numpy(), cos.cpu().numpy(), c


def cluster(args):
    out = Path(args.out)
    df = pd.read_parquet(args.index)
    feats = np.load(out / 'feats.npy')
    assert len(feats) == len(df), (len(feats), len(df))
    torch.cuda.set_device(int(args.gpu))
    assign, cos, _ = spherical_kmeans(torch.from_numpy(feats).cuda(), args.k, iters=args.iters)
    df['pseudo_artist'] = assign.astype(np.int64)
    df['pseudo_artist_cos'] = cos.astype(np.float32)
    df.to_parquet(out / 'train.parquet', index=False)
    # 诊断: 每簇图片数 / 角色数; 能形成交叉组 (≥2 个角色) 的簇占多少
    g = df.groupby('pseudo_artist')
    n_img, n_cls = g.size(), g['label'].nunique()
    stats = dict(k=args.k, n=len(df), img_per_cluster=np.percentile(n_img, [5, 25, 50, 75, 95]).tolist(),
                 classes_per_cluster=np.percentile(n_cls, [5, 25, 50, 75, 95]).tolist(),
                 frac_clusters_multi_class=float((n_cls >= 2).mean()),
                 frac_images_in_multi_class=float(n_img[n_cls >= 2].sum() / len(df)),
                 mean_cos=float(cos.mean()))
    info = json.loads((out / 'extract_info.json').read_text())
    (out / 'build_info.json').write_text(json.dumps(dict(info, cluster=stats, base_index=args.index), indent=1))
    print(json.dumps(stats, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['extract', 'cluster'])
    ap.add_argument('--csip-exp')
    ap.add_argument('--index', default=INDEX)
    ap.add_argument('--root', default=CCIP_ROOT)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--gpus', default='4,5,6,7')
    ap.add_argument('--gpu', default='4')
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--k', type=int, default=8192)
    ap.add_argument('--iters', type=int, default=25)
    args = ap.parse_args()
    extract(args) if args.stage == 'extract' else cluster(args)


if __name__ == '__main__':
    main()
