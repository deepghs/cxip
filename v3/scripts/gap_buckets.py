"""CSIP v3 vs Kaloscope: 在任一 Danbooru 来源的评测集上按配对属性分桶比较 (精确计算全部图片对, CPU)。

每个模型用自己在该集合上的 best-F1 logit 阈值; 正对按年份差 / 同角色 / 同作品 / 媒介组合分桶给召回, 负对给误报率。
另给: 两模型逐查询 1-NN 对错的重叠, 每画师 mAP 的相关, 以及两模型 logit 相加 (late fusion) 的 PR-AUC / F1。
元数据: test100_0918 用 danbooru-artist-v1 index (year / general / characters / copyrights);
        其余集合 (index.parquet 带 Danbooru id) 从 nyanko 2026-07 快照取 created_at 年份与 tag_string_*。
用法: CUDA_VISIBLE_DEVICES= python -m v3.scripts.gap_buckets csip_unseen_v1 \\
          --model v3 <feats_dir> 18.25 -9.875 --model kal <feats_dir> 38.2264 -18.0963 [--out json]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from v3.eval import pairwise
from v3.eval.runner import load_set
from v3.scripts.csip_full_eval import load_aligned

SNAP = '/scratch/anime_tagger/dl/nyanko_danbooru2026/posts-snapshot.parquet'
META0918 = '/data/neko/cxip_v3/data/danbooru-artist-v1/index/test100_0918.parquet'
MEDIUM = [('comic', {'comic', '4koma'}), ('sketch', {'sketch', 'lineart'}), ('mono', {'monochrome', 'greyscale'})]
MED_NAMES = ['color', 'mono', 'comic', 'sketch']
YG_EDGES, YG_NAMES = np.array([0, 1, 3, 6, 100]), ['0', '1-2', '3-5', '6+']


def medium(tags):
    for name, s in MEDIUM:
        if tags & s:
            return name
    return 'color'


def load_meta(name):
    df, _ = load_set(name)
    if name == 'test100_0918':
        df['id'] = df['path'].map(lambda p: int(Path(p).stem))
        m = pd.read_parquet(META0918, columns=['danbooru_id', 'year', 'general', 'characters', 'copyrights'])
        df = df.merge(m.rename(columns={'danbooru_id': 'id'}), on='id', how='left')
        df['gen'] = df['general'].map(lambda g: set((g or '').split(', ')))
        one = lambda c: c[0] if c is not None and len(c) == 1 else None  # noqa: E731
        df['char'] = df['characters'].map(one)
        df['copy'] = df['copyrights'].map(lambda c: c[0] if c is not None and len(c) else None)
    else:
        t = pq.read_table(SNAP, columns=['id', 'created_at', 'tag_string_general', 'tag_string_character',
                                         'tag_string_copyright'], filters=[('id', 'in', df['id'].tolist())]).to_pandas()
        df = df.drop(columns=['created_at'], errors='ignore').merge(t, on='id', how='left')
        df['year'] = pd.to_datetime(df['created_at'], utc=True).dt.year
        df['gen'] = df['tag_string_general'].fillna('').str.split().map(set)
        df['char'] = df['tag_string_character'].fillna('').str.split().map(lambda c: c[0] if len(c) == 1 else None)
        df['copy'] = df['tag_string_copyright'].fillna('').str.split().map(lambda c: c[0] if len(c) else None)
    df['year'] = df['year'].fillna(df.groupby('label')['year'].transform('median')).fillna(df['year'].median())
    df['med'] = df['gen'].map(medium)
    return df


def pair_masks(df):
    n = len(df)
    iu = np.triu_indices(n, 1)
    lab = df['label'].values
    same = lab[iu[0]] == lab[iu[1]]
    yg = np.digitize(np.abs(df['year'].values[iu[0]] - df['year'].values[iu[1]]), YG_EDGES) - 1
    ch = pd.factorize(df['char'])[0]
    cp = pd.factorize(df['copy'])[0]
    md = df['med'].map(MED_NAMES.index).values
    a, b = np.minimum(md[iu[0]], md[iu[1]]), np.maximum(md[iu[0]], md[iu[1]])
    buckets = {'all': np.ones(len(same), bool)}
    for g, nm in enumerate(YG_NAMES):
        buckets[f'ygap/{nm}'] = yg == g
    sc = (ch[iu[0]] == ch[iu[1]]) & (ch[iu[0]] >= 0)
    buckets['char/same'], buckets['char/diff'] = sc, ~sc
    sp = (cp[iu[0]] == cp[iu[1]]) & (cp[iu[0]] >= 0)
    buckets['copyright/same'], buckets['copyright/diff'] = sp, ~sp
    for i in range(4):
        for j in range(i, 4):
            buckets[f'medium/{MED_NAMES[i]}-{MED_NAMES[j]}'] = (a == i) & (b == j)
    return iu, same, buckets


def logits(f, scale, bias):
    z = torch.nn.functional.normalize(torch.from_numpy(f), dim=-1)
    return (scale * (z @ z.T) + bias).numpy()


def best_thr(s, y):
    m = pairwise.metrics_exact(s, y)
    return m['best_threshold'], m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('set')
    ap.add_argument('--model', nargs=4, action='append', metavar=('NAME', 'FEATS_DIR', 'SCALE', 'BIAS'), required=True)
    ap.add_argument('--out')
    args = ap.parse_args()
    df = load_meta(args.set)
    iu, same, buckets = pair_masks(df)
    lab = df['label'].values
    out = dict(set=args.set, n=len(df), n_pos=int(same.sum()),
               year_span_per_artist=df.groupby('label')['year'].agg(lambda y: y.max() - y.min()).describe().to_dict(),
               pos_share={k: float(m[same].mean()) for k, m in buckets.items()}, models={})
    L, nn_ok = {}, {}
    for name, fd, sc, b in args.model:
        f = load_aligned(fd, args.set, df)
        S = logits(f, float(sc), float(b))
        s = S[iu]
        thr, m = best_thr(s, same)
        r = dict(pr_auc=m['pr_auc'], best_f1=m['best_f1'], thr=thr, buckets={})
        for k, msk in buckets.items():
            p, q = msk & same, msk & ~same
            r['buckets'][k] = dict(n_pos=int(p.sum()), recall=float((s[p] > thr).mean()) if p.any() else None,
                                   n_neg=int(q.sum()), fpr=float((s[q] > thr).mean()) if q.any() else None)
        np.fill_diagonal(S, -np.inf)
        nn_ok[name] = lab[S.argmax(1)] == lab
        r['nn1_acc'] = float(nn_ok[name].mean())
        L[name] = S
        out['models'][name] = r
    names = [m[0] for m in args.model]
    if len(names) == 2:
        a, b = (nn_ok[x] for x in names)
        out['nn1_overlap'] = dict(both_right=float((a & b).mean()), only_first=float((a & ~b).mean()),
                                  only_second=float((~a & b).mean()), both_wrong=float((~a & ~b).mean()))
        Sf = L[names[0]] + L[names[1]]
        fused = Sf[iu]
        thr, m = best_thr(fused, same)
        out['fusion'] = dict(pr_auc=m['pr_auc'], best_f1=m['best_f1'], nn1_acc=float((lab[Sf.argmax(1)] == lab).mean()),
                             buckets={k: dict(recall=float((fused[msk & same] > thr).mean()) if (msk & same).any() else None,
                                              fpr=float((fused[msk & ~same] > thr).mean()) if (msk & ~same).any() else None)
                                      for k, msk in buckets.items()})
    print(json.dumps({k: v for k, v in out.items() if k != 'models'}, indent=1, default=float)[:1500])
    keys = [k for k in buckets if out['models'][names[0]]['buckets'][k]['n_pos'] >= 200]
    print(f'{"bucket":22s} {"pos%":>6s} ' + ' '.join(f'{n + " rec":>9s}' for n in names) + '  ' +
          ' '.join(f'{n + " fpr":>9s}' for n in names))
    for k in keys:
        bs = [out['models'][n]['buckets'][k] for n in names]
        print(f'{k:22s} {out["pos_share"][k] * 100:5.1f}% ' + ' '.join(f'{x["recall"]:9.3f}' for x in bs) + '  ' +
              ' '.join(f'{x["fpr"]:9.5f}' if x['fpr'] is not None else f'{"-":>9s}' for x in bs))
    if 'fusion' in out:
        print('fusion recall: ' + ' '.join(f'{k}={out["fusion"]["buckets"][k]["recall"]:.3f}' for k in keys))
    for n in names:
        r = out['models'][n]
        print(n, 'pr_auc %.4f best_f1 %.4f thr %.3f nn1 %.4f' % (r['pr_auc'], r['best_f1'], r['thr'], r['nn1_acc']))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
