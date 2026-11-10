"""画师画风随时间的变化: 变化快慢的分布 + 自适应画风分期 (按向量变化切, 不按日历切)。CPU。

输入: 评测集 (Danbooru 来源, 日期取 nyanko 2026-07 快照 created_at) + 若干模型的缓存特征 (--model NAME DIR)。
另加 fusion = 各模型 L2 归一化后拼接 (cos = 各模型 cos 的均值)。

1. 保持度曲线: 同画师图片对按时间差分桶, r(g) = (桶内平均 cos - 跨画师平均 cos) / (近期 (<0.5 年) 平均 cos - 跨画师平均 cos)。
   tau = r 降到 0.5 的时间差 (对数时间上线性插值; 跨度内没降到 0.5 记为 > 跨度, 删失)。
2. 分期: 按日期排序, 二分切分 (使段内平方误差下降最多的切点, 每段 >= MIN_SEG 张),
   两段的归一化交叉相似度 rho = (cross - neg) / (sqrt(within_A * within_B) - neg) < RHO 才接受切分, 递归;
   然后合并 rho >= RHO 的相邻段。rho 就是 "两段画风有多像" (1 = 分不出, 0 = 像不同画师)。
   复现: 段 A < B < C 中 rho(A, C) >= RHO 且 rho(A, C) > rho(A, B) + 0.1 记为 "画风回归"。
   媒介: 段内黑白/漫画/草图占比 >= 0.6 记为媒介段。
3. 接触表 (--sheets): 每画师一张, 每行一个分期, 均匀取样 + 日期; 画师匿名 (A01...), 名字映射另存 json。
用法: CUDA_VISIBLE_DEVICES= python -m v3.scripts.style_drift test100_0918 --model v3x4 <dir> --model kal <dir> \\
          --out-dir /data/neko/cxip_v3/eval/gap/drift [--sheets 8]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

from v3.eval.runner import load_set
from v3.scripts.csip_full_eval import load_aligned

SNAP = '/scratch/anime_tagger/dl/nyanko_danbooru2026/posts-snapshot.parquet'
GAP_EDGES = np.array([0, 0.25, 0.5, 1, 2, 3, 4, 6, 10, 30])
MIN_SEG = 8
RHOS = (0.5, 0.6, 0.7, 0.8)
NONCOLOR = {'monochrome', 'greyscale', 'comic', '4koma', 'sketch', 'lineart'}


def load_meta(name):
    df, _ = load_set(name)
    if 'id' not in df:
        df['id'] = df['path'].map(lambda p: int(Path(p).stem))
    t = pq.read_table(SNAP, columns=['id', 'created_at', 'tag_string_general', 'rating'],
                      filters=[('id', 'in', df['id'].tolist())]).to_pandas()
    df = df.drop(columns=['created_at', 'rating'], errors='ignore').merge(t, on='id', how='left')
    ts = pd.to_datetime(df['created_at'], utc=True)
    df['t'] = ts.dt.year + (ts.dt.dayofyear - 1) / 365.25
    df['t'] = df['t'].fillna(df.groupby('label')['t'].transform('median'))
    df['noncolor'] = df['tag_string_general'].fillna('').str.split().map(lambda s: bool(NONCOLOR & set(s)))
    return df


def norm(f):
    return f / np.linalg.norm(f, axis=1, keepdims=True)


def retention(S, t, neg):
    iu = np.triu_indices(len(t), 1)
    g, s = np.abs(t[iu[0]] - t[iu[1]]), S[iu]
    near = s[g < 0.5].mean() if (g < 0.5).sum() >= 10 else s[g < 1].mean() if (g < 1).sum() >= 10 else np.nan
    b = np.digitize(g, GAP_EDGES) - 1
    r = np.full(len(GAP_EDGES) - 1, np.nan)
    for k in range(len(r)):
        m = b == k
        if m.sum() >= 10:
            r[k] = (s[m].mean() - neg) / (near - neg)
    return r, near


def tau_from(r, span):
    centers = np.sqrt(np.maximum(GAP_EDGES[:-1], 0.125) * GAP_EDGES[1:])
    ok = ~np.isnan(r)
    c, v = centers[ok], r[ok]
    for k in range(1, len(v)):
        if v[k] < 0.5 <= v[k - 1]:
            w = (v[k - 1] - 0.5) / (v[k - 1] - v[k])
            return float(np.exp(np.log(c[k - 1]) + w * (np.log(c[k]) - np.log(c[k - 1])))), False
    if len(v) and v[0] < 0.5:
        return float(c[0]), False
    return float(span), True     # 删失: 跨度内没掉到 0.5


def rho(S, A, B, neg):
    wa = S[np.ix_(A, A)][np.triu_indices(len(A), 1)].mean()
    wb = S[np.ix_(B, B)][np.triu_indices(len(B), 1)].mean()
    cr = S[np.ix_(A, B)].mean()
    return float((cr - neg) / (np.sqrt(max(wa - neg, 1e-6) * max(wb - neg, 1e-6))))


def best_split(Z, lo, hi):
    """段 [lo, hi) 内使平方误差下降最多的切点 (Z 已按时间排序, 单位向量)"""
    cs = np.cumsum(Z[lo:hi], axis=0)
    tot = cs[-1]
    n = hi - lo
    best, arg = -1, None
    for k in range(MIN_SEG, n - MIN_SEG + 1):
        left, right = cs[k - 1], tot - cs[k - 1]
        gain = (left @ left) / k + (right @ right) / (n - k) - (tot @ tot) / n
        if gain > best:
            best, arg = gain, lo + k
    return arg


def segment(Z, S, neg, thr):
    segs, stack = [], [(0, len(Z))]
    while stack:
        lo, hi = stack.pop()
        k = best_split(Z, lo, hi) if hi - lo >= 2 * MIN_SEG else None
        if k is not None and rho(S, np.arange(lo, k), np.arange(k, hi), neg) < thr:
            stack += [(k, hi), (lo, k)]
        else:
            segs.append((lo, hi))
    segs.sort()
    merged = [segs[0]]
    for s in segs[1:]:
        a = merged[-1]
        if rho(S, np.arange(*a), np.arange(*s), neg) >= thr:
            merged[-1] = (a[0], s[1])
        else:
            merged.append(s)
    return merged


def analyze_artist(Z, t, noncolor, neg):
    S = Z @ Z.T
    r, near = retention(S, t, neg)
    tau, cens = tau_from(r, t.max() - t.min())
    out = dict(n=len(t), span=float(t.max() - t.min()), near=float(near), neg=float(neg), retention=r.tolist(),
               tau=tau, censored=cens, segs={})
    for thr in RHOS:
        sg = segment(Z, S, neg, thr)
        idx = [np.arange(*s) for s in sg]
        rec = any(rho(S, idx[a], idx[c], neg) >= thr and
                  rho(S, idx[a], idx[c], neg) > rho(S, idx[a], idx[b], neg) + 0.1
                  for a in range(len(idx)) for b in range(a + 1, len(idx)) for c in range(b + 1, len(idx)))
        out['segs'][str(thr)] = dict(bounds=[list(map(int, s)) for s in sg],
                                     years=[float(t[s[1] - 1] - t[s[0]]) for s in sg],
                                     medium_seg=[bool(noncolor[s[0]:s[1]].mean() >= 0.6) for s in sg],
                                     recurrence=bool(rec))
    out['modes'] = modes(Z, S, t, noncolor, neg)
    return out


def modes(Z, S, t, noncolor, neg, thr=0.7):
    """不看时间把画师的图分成 2 类 (平均连接, cos); 两类够大且 rho < thr 时, 用时间重叠判断是 "分期" 还是 "并存画法"。
    overlap = 两类各自 10-90% 时间区间的交集 / 较短区间长度 (0 = 时间上完全分开, 1 = 同时期并存)。"""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    if len(t) < 2 * MIN_SEG:
        return None
    D = np.clip(1 - S, 0, 2)
    np.fill_diagonal(D, 0)
    lab = fcluster(linkage(squareform(D, checks=False), 'average'), 2, 'maxclust')
    A, B = np.flatnonzero(lab == 1), np.flatnonzero(lab == 2)
    if min(len(A), len(B)) < max(MIN_SEG, 0.1 * len(t)):
        return dict(split=False)
    r = rho(S, A, B, neg)
    qa, qb = np.quantile(t[A], [.1, .9]), np.quantile(t[B], [.1, .9])
    inter = max(0.0, min(qa[1], qb[1]) - max(qa[0], qb[0]))
    overlap = inter / max(min(qa[1] - qa[0], qb[1] - qb[0]), 0.25)
    return dict(split=bool(r < thr), rho=r, overlap=float(min(overlap, 1.0)), sizes=[len(A), len(B)],
                noncolor_diff=float(abs(noncolor[A].mean() - noncolor[B].mean())))


def same_seg_pairs(res, thr):
    """按分期定义正样本时保留的同画师对比例, 以及 365 天窗口规则的比例 (对照)"""
    keep = tot = 0
    for a in res.values():
        sizes = np.array([b - a_ for a_, b in a['segs'][str(thr)]['bounds']])
        keep += (sizes * (sizes - 1) // 2).sum()
        tot += a['n'] * (a['n'] - 1) // 2
    return keep / tot


def contact_sheet(df, rows, seg, path, title, per_row=10, th=150):
    W, H = per_row * th, len(seg) * (th + 22) + 30
    im = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(im)
    d.text((6, 6), title, fill='black')
    y = 30
    for lo, hi in seg:
        safe = [p for p in range(lo, hi) if df['rating'].iloc[rows[p]] in ('g', 's')]   # 只展示全年龄 / sensitive
        pick = np.array(safe)[np.linspace(0, len(safe) - 1, min(per_row, len(safe))).round().astype(int)] if safe else []
        d.text((6, y), f'{df.t.iloc[rows[lo]]:.1f} - {df.t.iloc[rows[hi - 1]]:.1f}  (n={hi - lo}, shown: rating g/s only)',
               fill='black')
        for k, p in enumerate(pick):
            r = df.iloc[rows[p]]
            try:
                x = Image.open(r['path']).convert('RGB')
            except Exception:
                continue
            x.thumbnail((th, th))
            im.paste(x, (k * th + (th - x.width) // 2, y + 18))
            d.text((k * th + 3, y + 18), f'{r.t:.1f}', fill=(255, 0, 0))
        y += th + 22
    im.save(path, quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('set')
    ap.add_argument('--model', nargs=2, action='append', metavar=('NAME', 'FEATS_DIR'), required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--sheets', type=int, default=0, help='每类 (快 / 慢 / 回归 / 媒介) 各出几张接触表')
    ap.add_argument('--sheet-model', default='fusion')
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_meta(args.set)
    Z = {n: norm(load_aligned(d, args.set, df)) for n, d in args.model}
    Z['fusion'] = norm(np.concatenate([Z[n] for n, _ in args.model], 1))
    lab = df['label'].values
    summary, per = {}, {}
    for name, z in Z.items():
        cent = np.stack([z[lab == a].mean(0) for a in np.unique(lab)])
        res = {}
        for i, a in enumerate(np.unique(lab)):
            rows = np.flatnonzero(lab == a)
            rows = rows[np.argsort(df['t'].values[rows], kind='stable')]
            others = np.delete(cent, i, 0)
            neg = float((z[rows] @ others.T).mean())     # 该画师图对其他画师的平均 cos (按画师均值近似)
            res[int(a)] = analyze_artist(z[rows], df['t'].values[rows], df['noncolor'].values[rows], neg)
            res[int(a)]['rows'] = rows.tolist()
        per[name] = res
        tau = pd.DataFrame([(v['tau'], v['censored'], v['span']) for v in res.values()], columns=['tau', 'cens', 'span'])
        s = dict(n_artists=len(res),
                 tau_quantiles_uncensored=tau[~tau.cens].tau.quantile([.1, .25, .5, .75, .9]).round(2).to_dict(),
                 frac_censored=float(tau.cens.mean()),
                 tau_bins={k: float(v) for k, v in pd.cut(tau.tau.where(~tau.cens, 99), [0, 1, 2, 4, 8, 100],
                                                         labels=['<1y', '1-2y', '2-4y', '4-8y', '>=8y or censored']
                                                         ).value_counts(normalize=True).sort_index().items()},
                 retention_median=np.nanmedian(np.array([v['retention'] for v in res.values()]), 0).round(3).tolist(),
                 seg={})
        for thr in RHOS:
            k = str(thr)
            nseg = np.array([len(v['segs'][k]['bounds']) for v in res.values()])
            yrs = np.concatenate([v['segs'][k]['years'] for v in res.values()])
            s['seg'][k] = dict(segs_per_artist=pd.Series(nseg).value_counts().sort_index().to_dict(),
                               mean_segs=float(nseg.mean()), seg_years_median=float(np.median(yrs)),
                               recurrence_frac=float(np.mean([v['segs'][k]['recurrence'] for v in res.values()])),
                               medium_seg_frac=float(np.mean(np.concatenate([v['segs'][k]['medium_seg']
                                                                             for v in res.values()]))),
                               same_seg_pair_frac=float(same_seg_pairs(res, thr)))
        md = [v['modes'] for v in res.values() if v['modes'] and v['modes'].get('split')]
        s['modes_rho07'] = dict(frac_artists_split=len(md) / len(res),
                                temporal=float(np.mean([m['overlap'] < 0.3 for m in md])) if md else None,
                                concurrent=float(np.mean([m['overlap'] >= 0.7 for m in md])) if md else None,
                                concurrent_medium_driven=float(np.mean([m['noncolor_diff'] >= 0.5 for m in md
                                                                        if m['overlap'] >= 0.7])) if md else None)
        R = np.array([v['retention'] for v in res.values()])
        s['retention_spread'] = {f'{GAP_EDGES[k]}-{GAP_EDGES[k + 1]}y': pd.Series(R[:, k]).dropna()
                                 .quantile([.1, .25, .5, .75, .9]).round(3).tolist() for k in (3, 5, 6, 7)}
        summary[name] = s
    # 模型间一致性: tau 的 Spearman, 分期边界的 ARI
    from sklearn.metrics import adjusted_rand_score
    names = list(Z)
    agree = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = per[names[i]], per[names[j]]
            ta = pd.Series({k: v['tau'] for k, v in a.items()})
            tb = pd.Series({k: v['tau'] for k, v in b.items()})
            aris = []
            for k in a:
                la = np.concatenate([[q] * (h - l) for q, (l, h) in enumerate(a[k]['segs']['0.7']['bounds'])])
                lb = np.concatenate([[q] * (h - l) for q, (l, h) in enumerate(b[k]['segs']['0.7']['bounds'])])
                if len(set(la)) > 1 or len(set(lb)) > 1:
                    aris.append(adjusted_rand_score(la, lb))
            agree[f'{names[i]}~{names[j]}'] = dict(tau_spearman=float(ta.corr(tb, method='spearman')),
                                                    seg_ari_rho07_mean=float(np.mean(aris)) if aris else None,
                                                    n_artists_segmented=len(aris))
    summary['agreement'] = agree
    # 365 天窗口规则保留的同画师对比例 (对照)
    keep = tot = 0
    for a in per['fusion'].values():
        t = df['t'].values[a['rows']]
        g = np.abs(t[:, None] - t[None])[np.triu_indices(len(t), 1)]
        keep += (g <= 1).sum()
        tot += len(g)
    summary['window365_pair_frac'] = keep / tot
    (out / f'{args.set}_summary.json').write_text(json.dumps(summary, indent=1, default=float))
    (out / f'{args.set}_per_artist.json').write_text(json.dumps(per, default=float))
    print(json.dumps(summary, indent=1, default=float))
    if args.sheets:
        res = per[args.sheet_model]
        k = '0.7'
        names_map = {}
        cand = {
            'fast': sorted([a for a, v in res.items() if not v['censored'] and v['span'] >= 4], key=lambda a: res[a]['tau']),
            'slow': sorted([a for a, v in res.items() if v['censored'] and v['span'] >= 8], key=lambda a: -res[a]['span']),
            'recur': [a for a, v in res.items() if v['segs'][k]['recurrence']],
            'medium': [a for a, v in res.items() if any(v['segs'][k]['medium_seg'])],
        }
        sd = out / f'{args.set}_sheets'
        sd.mkdir(exist_ok=True)
        done = set()
        for kind, lst in cand.items():
            for a in [x for x in lst if x not in done][:args.sheets]:
                done.add(a)
                code = f'A{len(names_map) + 1:02d}'
                names_map[code] = df.loc[df['label'] == a, 'label_name'].iloc[0]
                v = res[a]
                title = (f'{code} [{kind}] tau={v["tau"]:.1f}y{"+" if v["censored"] else ""} span={v["span"]:.1f}y '
                         f'segs(rho<0.7)={len(v["segs"][k]["bounds"])} recurrence={v["segs"][k]["recurrence"]}')
                contact_sheet(df, v['rows'], v['segs'][k]['bounds'], sd / f'{code}_{kind}.jpg', title)
        (sd / 'names_private.json').write_text(json.dumps(names_map, indent=1))
        print('sheets ->', sd, len(names_map))


if __name__ == '__main__':
    main()
