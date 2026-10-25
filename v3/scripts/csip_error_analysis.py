"""CSIP 错误分析: 在 test100_0918 (Danbooru 元数据齐全) 上按配对属性拆解正/负样本对的得分与错误率。

正对 (同画师): 按年份差 / 媒介组合 (彩色·黑白·漫画·草图) / 是否同角色 分桶 → 召回率 @ best-F1 阈值;
负对 (不同画师): 按是否同角色 / 同作品 / 同媒介 / 年份差分桶 → 误报率 @ 同一阈值;
每个画师: 自身正对的 AP 近似 (该画师的图作查询的 mAP) 与画师属性 (图数、年份跨度、黑白占比、角色数) 的关系。
用法: python -m v3.scripts.csip_error_analysis <feats.npz> [--out json]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from v3.eval.runner import load_set

META = '/data/neko/cxip_v3/data/danbooru-artist-v1/index/test100_0918.parquet'
MEDIUM = [('comic', {'comic', '4koma'}), ('sketch', {'sketch', 'lineart'}), ('mono', {'monochrome', 'greyscale'})]


def medium(general):
    tags = set((general or '').split(', '))
    for name, s in MEDIUM:
        if tags & s:
            return name
    return 'color'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('feats')
    ap.add_argument('--out')
    ap.add_argument('--bins', type=int, default=4096)
    args = ap.parse_args()
    d = np.load(args.feats)
    df, _ = load_set('test100_0918')
    df = df.set_index(df['sample_key'].astype(str)).loc[d['sample_key']].reset_index(drop=True)
    df['danbooru_id'] = df['path'].map(lambda p: int(Path(p).stem))
    meta = pd.read_parquet(META, columns=['danbooru_id', 'year', 'general', 'characters', 'copyrights'])
    df = df.merge(meta, on='danbooru_id', how='left')
    assert df['general'].notna().mean() > 0.95
    # 少数图 (139 张) 元数据缺年份: 用该画师的中位年份
    df['year'] = df['year'].fillna(df.groupby('label')['year'].transform('median')).fillna(df['year'].median())
    n = len(df)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    z = torch.nn.functional.normalize(torch.tensor(d['feats'], dtype=torch.float32, device=dev), dim=-1)
    art = torch.tensor(df['label'].values, device=dev)
    year = torch.tensor(df['year'].values, device=dev)
    med_names = ['color', 'mono', 'comic', 'sketch']
    med = torch.tensor(df['general'].map(medium).map(med_names.index).values, device=dev)
    single = lambda c: c[0] if c is not None and len(c) == 1 else None  # noqa: E731
    char = torch.tensor(pd.factorize(df['characters'].map(single))[0], device=dev)       # -1 = 无/多角色
    copy = torch.tensor(pd.factorize(df['copyrights'].map(lambda c: c[0] if c is not None and len(c) else None))[0],
                        device=dev)
    # 属性组合 → 桶 id; 每桶一个分数直方图 (正/负分开)
    ygap_edges = torch.tensor([0, 1, 3, 6, 100], device=dev)  # 0 / 1-2 / 3-5 / 6+ 年
    yg_names = ['0', '1-2', '3-5', '6+']
    B = args.bins
    hist = {}

    def add(key, scores):
        h = hist.setdefault(key, torch.zeros(B, dtype=torch.float64, device=dev))
        h += torch.bincount(((scores + 1) / 2 * (B - 1)).long().clamp(0, B - 1), minlength=B).double()

    chunk = 2048
    for s in range(0, n, chunk):
        r = torch.arange(s, min(n, s + chunk), device=dev)
        sim = z[r] @ z.T
        upper = torch.arange(n, device=dev)[None] > r[:, None]
        same = art[r, None] == art[None]
        yg = torch.bucketize((year[r, None] - year[None]).abs(), ygap_edges, right=True) - 1
        mpair = med[r, None] * 4 + med[None]
        mpair = torch.minimum(mpair, med[r, None] + med[None] * 4)  # 无序对
        same_char = (char[r, None] == char[None]) & (char[r, None] >= 0)
        same_copy = (copy[r, None] == copy[None]) & (copy[r, None] >= 0)
        for lab, m0 in (('pos', same & upper), ('neg', ~same & upper)):
            add((lab, 'all'), sim[m0])
            for g in range(4):
                add((lab, 'ygap', yg_names[g]), sim[m0 & (yg == g)])
            for a in range(4):
                for b in range(a, 4):
                    add((lab, 'medium', f'{med_names[a]}-{med_names[b]}'), sim[m0 & (mpair == a * 4 + b)])
            add((lab, 'char', 'same'), sim[m0 & same_char])
            add((lab, 'char', 'diff'), sim[m0 & ~same_char])
            add((lab, 'copyright', 'same'), sim[m0 & same_copy])
            add((lab, 'copyright', 'diff'), sim[m0 & ~same_copy])
    # best-F1 阈值 (总体)
    pos, neg = hist[('pos', 'all')], hist[('neg', 'all')]
    tp, fp = pos.flip(0).cumsum(0), neg.flip(0).cumsum(0)
    prec, rec = tp / (tp + fp).clamp_min(1), tp / pos.sum()
    f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-12)
    bi = int(f1.argmax())
    thr_bin = B - 1 - bi  # 分数 ≥ 该 bin 判为同画师
    thr = thr_bin / (B - 1) * 2 - 1
    ap_all = float((prec * pos.flip(0)).sum() / pos.sum())
    res = dict(feats=args.feats, n=n, ap=ap_all, best_f1=float(f1[bi]), thr_cos=thr, buckets={})
    for key, h in sorted(hist.items(), key=lambda kv: str(kv[0])):
        lab, *rest = key
        tot = float(h.sum())
        if tot == 0:
            continue
        above = float(h[thr_bin:].sum())
        centers = torch.linspace(-1, 1, B, device=dev, dtype=torch.float64)
        mean = float((h * centers).sum() / tot)
        name = '/'.join(rest)
        res['buckets'].setdefault(name, {})[lab] = dict(n=tot, rate_above=above / tot, mean_cos=mean)
    # 每个画师: 以其图为查询的 mAP (检索全集) 与属性
    rows = []
    for a in torch.unique(art).tolist():
        q = (art == a).nonzero().squeeze(1)
        sim = z[q] @ z.T
        sim[torch.arange(len(q)), q] = -2
        order = sim.argsort(dim=1, descending=True)
        rel = (art[order] == a).float()[:, :-1]
        cum = rel.cumsum(1)
        prec_k = cum / torch.arange(1, rel.shape[1] + 1, device=dev)
        ap_q = (prec_k * rel).sum(1) / rel.sum(1).clamp_min(1)
        sub = df[df['label'] == a]
        rows.append(dict(artist=sub['label_name'].iloc[0], n=len(sub), map=float(ap_q.mean()),
                         year_span=int(sub['year'].max() - sub['year'].min()),
                         mono_frac=float((sub['general'].map(medium) != 'color').mean()),
                         n_chars=int(sub['characters'].map(single).nunique()),
                         top_copy_frac=float(sub['copyrights'].map(lambda c: c[0] if c is not None and len(c) else None)
                                             .value_counts(normalize=True).iloc[0]) if sub['copyrights'].notna().any() else 0.0))
    art_df = pd.DataFrame(rows).sort_values('map')
    res['artists'] = art_df.to_dict('records')
    corr = {c: float(art_df['map'].corr(art_df[c], method='spearman')) for c in
            ('n', 'year_span', 'mono_frac', 'n_chars', 'top_copy_frac')}
    res['artist_corr_spearman'] = corr
    print(json.dumps({k: v for k, v in res.items() if k != 'artists'}, indent=1))
    print(art_df.head(15).to_string())
    print(art_df.tail(5).to_string())
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
