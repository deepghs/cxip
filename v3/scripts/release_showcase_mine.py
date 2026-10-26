"""README 效果展示图的选例 (CPU 读图 + GPU 算分): 结果写到 /data/neko/cxip_v3/release/_analysis/showcase_v1/。

pairs    难例对 (D). 每个模型用自己在该测试集上的 best-F1 阈值判定; "旧版错" = 所有旧版都错, 图中显示最新旧版的分数。
         CCIP (openset_full): lookalike  同作品不同角色, 旧版误报, v3 正确
                              crossstyle 同角色, 旧版漏判, v3 正确
                              fail_pos / fail_neg  v3 仍然判错 (漏判 / 误报)
         CSIP (test100_0918, 仅 rating=g): ygap6 同画师相隔 6 年以上 / medium 同画师彩图 vs 黑白·漫画·草图
                              (旧版全部漏判, v3 正确), samechar 不同画师同一角色 (旧版全部误报, v3 正确), fail_pos / fail_neg
grid     画师 x 角色对照 (A): test100_0918 rating=g 单角色图, 找 3 角色 x 3 画师每格都有图的组合 (4 x 4 不存在)
gridfeat 用发布的 HF 模型给选定的网格图打分 (picks.json 由人工审阅 sheet 后写入)
cluster  自动分组样本 (E): 固定种子抽 8 类 x 6 张
所有入选图片都要 imgutils anime_rating == 'safe' (Danbooru 图还要求 rating=g); 选例后人工逐张查看 contact sheet。
用法: python -m v3.scripts.release_showcase_mine {pairs,grid,gridfeat,cluster}
"""
import argparse
import itertools
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from v3.eval.runner import load_set
from v3.scripts.csip_error_analysis import META, medium
from v3.scripts.plot_release_details import Pairs
from v3.scripts.release_analysis import OUT as ANA, load_feats, models

OUT = ANA / 'showcase_v1'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
_rating_cache = {}


def is_safe(key, read):
    """imgutils anime_rating, 结果按 sample_key 缓存到 OUT/rating.json"""
    if key not in _rating_cache:
        from imgutils.validate import anime_rating
        _rating_cache[key] = str(anime_rating(read())[0])
    return _rating_cache[key] == 'safe'


def load_rating_cache():
    p = OUT / 'rating.json'
    if p.exists():
        _rating_cache.update(json.loads(p.read_text()))


def save_rating_cache():
    (OUT / 'rating.json').write_text(json.dumps(_rating_cache))


def task_data(task):
    """返回 df (含元数据), read_image, {model: (feats, scale, bias, thr)}, 新模型名, 最新旧版名"""
    s = 'openset_full' if task == 'ccip' else 'test100_0918'
    df, read = load_set(s)
    ana = np.load(ANA / f'{task}.npz')
    meta = json.loads((ANA / f'{task}.json').read_text())
    M = {}
    for name, by_set, erase in models(task):
        fd, scale, bias = by_set[s]
        M[name] = (load_feats(fd, s, df, erase).to(DEV), scale, bias, Pairs(ana, meta, f'{s}/{name}').best_thr)
    if task == 'csip':
        df['danbooru_id'] = df['path'].map(lambda p: int(Path(p).stem))
        m = pd.read_parquet(META, columns=['danbooru_id', 'rating', 'year', 'general', 'characters', 'copyrights'])
        df = df.merge(m, on='danbooru_id', how='left')
        df['medium'] = df['general'].map(medium)
        df['char1'] = df['characters'].map(lambda c: c[0] if c is not None and len(c) == 1 else None)
    names = list(M)
    return df, read, M, names[-1], names[-2]


def mine_pairs(task, k_keep=400, chunk=1024):
    df, read, M, new, last_old = task_data(task)
    olds = [m for m in M if m != new]
    n = len(df)
    pool = np.arange(n) if task == 'ccip' else np.nonzero((df['rating'] == 'g').values)[0]
    idx = torch.as_tensor(pool, device=DEV)
    lab = torch.as_tensor(df['label'].values[pool], device=DEV)
    if task == 'ccip':
        grp = torch.as_tensor(pd.factorize(df['series'])[0][pool], device=DEV)
    else:
        year = torch.as_tensor(df['year'].fillna(-100).values[pool], device=DEV)
        med = torch.as_tensor(df['medium'].map(['color', 'mono', 'comic', 'sketch'].index).values[pool], device=DEV)
        ch = torch.as_tensor(pd.factorize(df['char1'])[0][pool], device=DEV)
    F = {m: M[m][0][idx] for m in M}
    cands = {}

    def keep(cat, score, mask, r):
        score = torch.where(mask, score, torch.full_like(score, -1e9))
        v, j = score.flatten().topk(min(k_keep, score.numel()))
        ok = v > -1e8
        rows, cols = r[j[ok] // score.shape[1]], j[ok] % score.shape[1]
        cands.setdefault(cat, []).append(torch.stack([v[ok], rows.float(), cols.float()], 1).cpu())

    N = len(pool)
    for s0 in range(0, N, chunk):
        r = torch.arange(s0, min(N, s0 + chunk), device=DEV)
        L = {m: (F[m][r] @ F[m].T) * M[m][1] + M[m][2] for m in M}
        mrg = {m: L[m] - M[m][3] for m in M}          # > 0: 判为同类
        upper = torch.arange(N, device=DEV)[None] > r[:, None]
        same = (lab[r, None] == lab[None]) & upper
        diff = (lab[r, None] != lab[None]) & upper
        old_pos_min = torch.stack([mrg[m] for m in olds]).min(0).values   # 所有旧版都判同类 ⇔ min > 0
        old_pos_max = torch.stack([mrg[m] for m in olds]).max(0).values   # 所有旧版都判不同 ⇔ max < 0
        v3 = mrg[new]
        miss_fixed = same & (old_pos_max < 0) & (v3 > 0)
        fp_fixed = diff & (old_pos_min > 0) & (v3 < 0)
        if task == 'ccip':
            keep('lookalike', old_pos_min - v3, fp_fixed & (grp[r, None] == grp[None]) & (grp[r, None] >= 0), r)
            keep('crossstyle', v3 - old_pos_max, miss_fixed, r)
        else:
            keep('ygap6', v3 - old_pos_max, miss_fixed & ((year[r, None] - year[None]).abs() >= 6)
                 & (year[r, None] > 0) & (year[None] > 0), r)
            keep('medium', v3 - old_pos_max, miss_fixed & (med[r, None] != med[None])
                 & ((med[r, None] == 0) | (med[None] == 0)), r)
            keep('samechar', old_pos_min - v3, fp_fixed & (ch[r, None] == ch[None]) & (ch[r, None] >= 0), r)
        keep('fail_pos', -v3, same & (v3 < 0), r)
        keep('fail_neg', v3, diff & (v3 > 0), r)

    out = {}
    load_rating_cache()
    for cat, parts in cands.items():
        c = torch.cat(parts)
        c = c[c[:, 0].argsort(descending=True)]
        used_img, used_cls, picked = set(), set(), []
        for v, a, b in c.tolist():
            a, b = int(pool[int(a)]), int(pool[int(b)])
            cls = tuple(sorted((int(df['label'].iat[a]), int(df['label'].iat[b]))))
            if a in used_img or b in used_img or cls in used_cls:
                continue
            if not (is_safe(str(df['sample_key'].iat[a]), lambda: read(a))
                    and is_safe(str(df['sample_key'].iat[b]), lambda: read(b))):
                continue
            used_img |= {a, b}
            used_cls.add(cls)
            rec = dict(a=a, b=b, key_a=str(df['sample_key'].iat[a]), key_b=str(df['sample_key'].iat[b]),
                       label_a=int(df['label'].iat[a]), label_b=int(df['label'].iat[b]))
            for m in M:
                rec[m] = float((M[m][0][a] @ M[m][0][b]) * M[m][1] + M[m][2])
            if task == 'ccip':
                rec.update(name_a=str(df['name'].iat[a]), name_b=str(df['name'].iat[b]), series=str(df['series'].iat[a]))
            else:
                rec.update(year_a=float(df['year'].iat[a]), year_b=float(df['year'].iat[b]),
                           medium_a=df['medium'].iat[a], medium_b=df['medium'].iat[b],
                           char_a=df['char1'].iat[a], char_b=df['char1'].iat[b])
            picked.append(rec)
            if len(picked) >= 30:
                break
        out[cat] = picked
        save_rating_cache()
        print(task, cat, len(picked), flush=True)
    thr = {m: float(M[m][3]) for m in M}
    (OUT / f'pairs_{task}.json').write_text(json.dumps(dict(thr=thr, new=new, last_old=last_old, pairs=out),
                                                       indent=1, ensure_ascii=False))
    for cat, ps in out.items():
        contact_pairs(read, ps, OUT / f'sheet_{task}_{cat}.jpg')


def thumb(img, size=200):
    from PIL import Image
    img = img.copy()
    img.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new('RGB', (size, size), 'white')
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def contact_pairs(read, pairs, path, size=200):
    """审阅用: 每对一格 (左右两张), 左上角写序号"""
    from PIL import Image, ImageDraw
    cols = 5
    rows = (len(pairs) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * (2 * size + 20), rows * (size + 10)), 'white')
    dr = ImageDraw.Draw(sheet)
    with ThreadPoolExecutor(16) as ex:
        ims = list(ex.map(lambda p: (thumb(read(p['a']), size), thumb(read(p['b']), size)), pairs))
    for i, (ta, tb) in enumerate(ims):
        x, y = (i % cols) * (2 * size + 20), (i // cols) * (size + 10)
        sheet.paste(ta, (x, y))
        sheet.paste(tb, (x + size, y))
        dr.rectangle([x, y, x + 26, y + 16], fill='black')
        dr.text((x + 3, y + 2), str(i), fill='white')
    sheet.save(path, quality=88)


def mine_grid(n_char=3, n_art=3, top_chars=25):
    df, read, M, new, _ = task_data('csip')
    load_rating_cache()
    g = df[(df['rating'] == 'g') & df['char1'].notna()]
    ct = g.groupby(['char1', 'label']).size().unstack(fill_value=0)
    chars = ct.sum(1).sort_values(ascending=False).index[:top_chars]
    combos = []
    for cs in itertools.combinations(chars, n_char):
        arts = ct.columns[(ct.loc[list(cs)] > 0).all(0)]
        if len(arts) >= n_art:
            combos.append((len(arts), cs, list(map(int, arts))))
    combos.sort(key=lambda t: -t[0])
    print('combos with >=%d artists:' % n_art, len(combos))
    for k, cs, arts in combos[:10]:
        print(k, cs, arts)
    # 第一个组合: 每格列出 rating=g 且 safe 的候选图
    k, cs, arts = combos[0]
    cells = {}
    for c in cs:
        for a in arts:
            ids = g.index[(g['char1'] == c) & (g['label'] == a)].tolist()
            ok = [int(i) for i in ids if is_safe(str(df['sample_key'].iat[i]), lambda i=i: read(i))]
            cells[f'{c}|{a}'] = ok
    save_rating_cache()
    (OUT / 'grid_candidates.json').write_text(json.dumps(dict(chars=list(cs), artists=arts, cells=cells,
                                                              combos=[(k, list(c), a) for k, c, a in combos[:30]]),
                                                         indent=1))
    # 审阅 sheet: 行 = (角色, 画师), 每行最多 8 张候选
    from PIL import Image, ImageDraw
    size, rows = 160, [(key, v) for key, v in cells.items() if v]
    sheet = Image.new('RGB', (8 * size + 220, len(rows) * size), 'white')
    dr = ImageDraw.Draw(sheet)
    for r, (key, ids) in enumerate(rows):
        dr.text((4, r * size + 4), key, fill='black')
        for j, i in enumerate(ids[:8]):
            sheet.paste(thumb(read(i), size), (220 + j * size, r * size))
            dr.text((220 + j * size + 3, r * size + 3), str(j), fill='red')
    sheet.save(OUT / 'sheet_grid.jpg', quality=88)


def mine_cluster(n_cls=8, per=6, seed=0):
    load_rating_cache()
    res = {}
    for task in ('ccip', 'csip'):
        df, read, M, new, _ = task_data(task)
        pool = df if task == 'ccip' else df[df['rating'] == 'g']
        cnt = pool['label'].value_counts()
        cls = cnt.index[cnt >= 2 * per].values
        rng = np.random.default_rng(seed)
        picked = []
        for c in rng.permutation(cls):
            ids = rng.permutation(pool.index[pool['label'] == c].values)
            ok = []
            for i in ids:
                if is_safe(str(df['sample_key'].iat[i]), lambda i=i: read(int(i))):
                    ok.append(int(i))
                if len(ok) == per:
                    break
            if len(ok) == per:
                picked += ok
            if len(picked) == n_cls * per:
                break
        save_rating_cache()
        picked = rng.permutation(picked)
        f = M[new][0][torch.as_tensor(picked, device=DEV)].float().cpu().numpy()
        import hdbscan
        import umap
        x = umap.UMAP(n_neighbors=11, n_components=35, min_dist=0.0, metric='cosine', random_state=seed).fit_transform(f)
        pred = hdbscan.HDBSCAN(min_cluster_size=2).fit_predict(x)
        y = df['label'].values[picked]
        from sklearn.metrics import adjusted_rand_score
        print(task, 'clusters', len(set(pred) - {-1}), 'noise', int((pred < 0).sum()), 'ARI', adjusted_rand_score(y, pred))
        res[task] = dict(idx=[int(i) for i in picked], keys=[str(df['sample_key'].iat[i]) for i in picked],
                         label=[int(v) for v in y], pred=[int(v) for v in pred])
    (OUT / 'cluster.json').write_text(json.dumps(res, indent=1))


def grid_feats():
    """用发布的 HF 模型 (本地目录, transformers 加载) 给对照网格的 9 张图打分, 写 grid.json"""
    from transformers import AutoImageProcessor, AutoModel
    picks = json.loads((OUT / 'picks.json').read_text())
    cand = json.loads((OUT / 'grid_candidates.json').read_text())
    df, read = load_set('test100_0918')
    keys = list(picks['grid_cells'])
    idx = [cand['cells'][k][picks['grid_cells'][k]] for k in keys]
    imgs = [read(i) for i in idx]
    a = keys.index(picks['grid_anchor'])
    res = dict(chars=cand['chars'], artists=cand['artists'], keys=keys, idx=idx,
               sample_keys=[str(df['sample_key'].iat[i]) for i in idx], anchor=picks['grid_anchor'])
    for repo in ('AnimeCCIP-v3', 'AnimeCSIP-v3'):
        d = ANA.parent / repo
        model = AutoModel.from_pretrained(d, trust_remote_code=True, dtype=torch.bfloat16).to(DEV).eval()
        proc = AutoImageProcessor.from_pretrained(d, trust_remote_code=True)
        with torch.no_grad():
            f = model.get_image_features(proc(imgs)['pixel_values'])
            res[repo] = dict(logits=model.compute_logits(f[a:a + 1], f)[0].float().cpu().tolist(),
                             threshold=float(model.config.threshold))
        del model
        torch.cuda.empty_cache()
        print(repo, np.round(res[repo]['logits'], 2).tolist(), flush=True)
    (OUT / 'grid.json').write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['pairs', 'grid', 'gridfeat', 'cluster'])
    ap.add_argument('--task', choices=['ccip', 'csip'], nargs='*', default=['ccip', 'csip'])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.what == 'pairs':
        for t in args.task:
            mine_pairs(t)
    elif args.what == 'grid':
        mine_grid()
    elif args.what == 'gridfeat':
        grid_feats()
    else:
        mine_cluster()


if __name__ == '__main__':
    main()
