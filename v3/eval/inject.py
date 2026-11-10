"""清洗注入基准: 构造 5% 注入离群的画师文件夹 (4 类), 打分 (kNN 离群分 / date_conflict), 逐文件夹 AUROC.

文件夹: id = 画师 label (int). 一个 B 集合的 index (dev + test 一起) 同一张图可出现在多个文件夹
(作为 A 的原图, 或作为别的文件夹的注入), 所以结果表按 (folder, sample_key) 唯一, 不按 sample_key.

make_folders(df, kind, frac, seed, feats) -> DataFrame[folder, sample_key, injected, t_claimed]
  行顺序: 按 folder 升序, 每个文件夹内先原图 (df 内顺序) 后注入图; 每个文件夹注入数 =
  max(1, floor(frac * 原图数 + 0.5)) (四舍五入, 0.5 向上). 某类的候选不够时能注入多少算多少 (不到 1 张则整个画师跳过),
  统计放在 result.attrs['stats'] (n_folders / n_skipped / n_short / n_injected).
  df 需要列: sample_key, label, t_year, characters (same_char); feats 与 df 行对齐 (near_style).
  同 seed 结果完全一致 (每个文件夹的随机流只由 (seed, kind, label) 决定, 与其他画师无关).
  - near_style: 注入 = 其他画师里离 A 的单位特征质心 (均值后 L2 归一) 最近的图, 每画师至多 1 张 (取离质心最近的那张),
                画师按这张图的余弦从高到低选.
  - same_char : 注入 = 其他画师里带 A 最常见角色 (并列取字典序小者) 的图, 随机抽. A 没有角色标签则跳过.
  - far_period: 跨度 >= 7 年的画师; 文件夹 = 图数最多的 2 年窗口 (起点取某张图, 并列取最早) 内的图, 须 >= 10 张;
                窗口中心 = 窗口内图的平均日期; 注入 = A 中距中心 >= 6 年的图 (随机抽).
  - bad_date  : 跨度 >= 6 年的画师; 文件夹 = A 的全部图; 随机顺序逐张试, 把 t_claimed 改成距真值 >= 4 年且在
                [t_min, t_max] 内的均匀随机值, 该图无法满足就换下一张. 其余图 t_claimed = t_year.
打分 (分数越大越像离群, AUROC 的正类 = injected):
  knn_outlier_scores: -(到同文件夹其他图的 top-k 余弦均值); 文件夹只有 1 张图 -> NaN.
  date_conflict_scores: style_dates(label=文件夹, 日期=t_claimed, day=date_day, family) -> conflict_scores.
    默认 k=5, temp=10, min_candidates=10; 文件夹太小 (候选 < 10, 含同日 / 同作品族被排除的) 的图得到 NaN.
  folder_auroc: 每个文件夹一个 AUROC (midrank, 平局算 0.5); NaN 分数的行被剔除; 没有有效注入行或没有有效原图行
    的文件夹为 NaN. 返回按 folder_ids(folders) (升序唯一 id) 排列的 ndarray; return_ids=True 时返回 (ids, auroc).
    指标 = 按画师加权的宏平均, 由调用方 (bench2) 用权重算, 且只对非 NaN 文件夹.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from v3.eval import dateest

KINDS = ('near_style', 'same_char', 'far_period', 'bad_date')
COLS = ['folder', 'sample_key', 'injected', 't_claimed']
_EPS = 1e-6                                              # 年; 防止浮点把 "恰好 4 年" 判成 3.9999999


def _n_inject(n: int, frac: float) -> int:
    return max(1, int(np.floor(frac * n + 0.5)))


def _rng(seed: int, kind: str, label) -> np.random.Generator:
    return np.random.default_rng([int(seed), KINDS.index(kind), int(label)])


def _unit(f) -> np.ndarray:
    f = np.asarray(f, dtype=np.float64)
    return f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)


def _rows(folder, keys, injected, t_claimed) -> pd.DataFrame:
    return pd.DataFrame({'folder': folder, 'sample_key': list(keys), 'injected': np.asarray(injected, dtype=bool),
                         't_claimed': np.asarray(t_claimed, dtype=np.float64)})


def _bad_date_value(t: float, t_min: float, t_max: float, rng) -> float | None:
    """[t_min, t - 4] ∪ [t + 4, t_max] 上均匀取一个值; 空则 None"""
    segs = []
    if t - 4 - _EPS >= t_min:
        segs.append((t_min, t - 4 - _EPS))
    if t + 4 + _EPS <= t_max:
        segs.append((t + 4 + _EPS, t_max))
    if not segs:
        return None
    w = np.array([hi - lo for lo, hi in segs]) + 1e-12
    lo, hi = segs[rng.choice(len(segs), p=w / w.sum())]
    return float(rng.uniform(lo, hi))


def make_folders(df: pd.DataFrame, kind: str, frac: float = 0.05, seed: int = 0,
                 feats: np.ndarray | None = None) -> pd.DataFrame:
    if kind not in KINDS:
        raise ValueError(f'unknown kind {kind!r}; expected one of {KINDS}')
    df = df.reset_index(drop=True)
    labels = df['label'].to_numpy()
    keys = df['sample_key'].to_numpy()
    t = df['t_year'].to_numpy(dtype=np.float64)
    uniq = np.unique(labels)
    by_art = {a: np.flatnonzero(labels == a) for a in uniq}
    out, n_skipped, n_short = [], 0, 0

    if kind == 'near_style':
        if feats is None:
            raise ValueError('near_style needs feats')
        feats = np.asarray(feats)
        if len(feats) != len(df):
            raise ValueError('feats must be aligned with df rows')
        z = _unit(feats)
        order = np.argsort(labels, kind='stable')
        sl = labels[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sl)) + 1]
        seg_label = sl[starts]
        seg_id = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(sl)]))
    elif kind == 'same_char':
        chars = [list(c) if c is not None else [] for c in df['characters']]
        char_rows: dict = {}
        for i, cs in enumerate(chars):
            for ch in set(cs):
                char_rows.setdefault(ch, []).append(i)
        char_rows = {ch: np.array(v, dtype=np.int64) for ch, v in char_rows.items()}

    for a in uniq:
        idx = by_art[a]
        n_inj = _n_inject(len(idx), frac)
        rng = _rng(seed, kind, a)

        if kind == 'near_style':
            c = z[idx].mean(0)
            c = c / max(np.linalg.norm(c), 1e-12)
            s = z[order] @ c
            seg_max = np.maximum.reduceat(s, starts)
            first = np.flatnonzero(s == seg_max[seg_id])                    # 每段 argmax (并列取最前)
            _, u = np.unique(seg_id[first], return_index=True)
            best = first[u]                                                  # 每个画师一个位置 (in order)
            keep = seg_label != a
            cand_pos, cand_s = best[keep], seg_max[keep]
            top = np.lexsort((order[cand_pos], -cand_s))[:n_inj]             # 余弦降序, 并列按行号
            inj = order[cand_pos[top]]
            clean_idx, t_cl = idx, t[idx]
            inj_t = t[inj]

        elif kind == 'same_char':
            cnt = Counter(ch for i in idx for ch in set(chars[i]))
            if not cnt:
                n_skipped += 1
                continue
            top_char = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            pool = char_rows[top_char]
            pool = pool[labels[pool] != a]
            inj = rng.permutation(pool)[:n_inj] if len(pool) else pool
            clean_idx, t_cl = idx, t[idx]
            inj_t = t[inj]

        elif kind == 'far_period':
            ta = t[idx]
            if ta.max() - ta.min() < 7:
                n_skipped += 1
                continue
            srt = np.sort(ta)
            hi = np.searchsorted(srt, srt + 2.0, side='right')
            counts = hi - np.arange(len(srt))
            j = int(np.argmax(counts))                                       # 并列取最早
            if counts[j] < 10:
                n_skipped += 1
                continue
            lo_t, hi_t = srt[j], srt[j] + 2.0
            clean_idx = idx[(ta >= lo_t) & (ta <= hi_t)]
            center = t[clean_idx].mean()
            n_inj = _n_inject(len(clean_idx), frac)
            pool = idx[np.abs(ta - center) >= 6]
            inj = rng.permutation(pool)[:n_inj] if len(pool) else pool
            t_cl, inj_t = t[clean_idx], t[inj]

        else:                                                                # bad_date
            ta = t[idx]
            if ta.max() - ta.min() < 6:
                n_skipped += 1
                continue
            t_min, t_max = float(ta.min()), float(ta.max())
            inj_l, inj_t_l = [], []
            for i in rng.permutation(idx):
                if len(inj_l) >= n_inj:
                    break
                v = _bad_date_value(float(t[i]), t_min, t_max, rng)
                if v is not None:
                    inj_l.append(i)
                    inj_t_l.append(v)
            inj, inj_t = np.array(inj_l, dtype=np.int64), np.array(inj_t_l)
            clean_idx = idx
            t_cl = t[idx].copy()
            if len(inj):                                                     # 注入图的 t_claimed 改了, 它不再算 "原图"
                pos = {int(i): k for k, i in enumerate(idx)}
                drop = np.zeros(len(idx), bool)
                drop[[pos[int(i)] for i in inj]] = True
                clean_idx, t_cl = idx[~drop], t_cl[~drop]

        if len(inj) == 0:
            n_skipped += 1
            continue
        if len(inj) < n_inj:
            n_short += 1
        out.append(_rows(int(a) if np.issubdtype(type(a), np.integer) else a,
                         np.r_[keys[clean_idx], keys[inj]],
                         np.r_[np.zeros(len(clean_idx), bool), np.ones(len(inj), bool)],
                         np.r_[t_cl, inj_t]))

    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        {'folder': pd.Series(dtype='int64'), 'sample_key': pd.Series(dtype=object),
         'injected': pd.Series(dtype=bool), 't_claimed': pd.Series(dtype='float64')})
    res.attrs['stats'] = dict(kind=kind, n_folders=len(out), n_skipped=n_skipped, n_short=n_short,
                              n_injected=int(res.injected.sum()))
    return res


def _groups(folders: pd.DataFrame):
    f = folders['folder'].to_numpy()
    order = np.argsort(f, kind='stable')
    cuts = np.flatnonzero(f[order][1:] != f[order][:-1]) + 1 if len(f) else []
    return np.split(order, cuts) if len(f) else []


def folder_ids(folders: pd.DataFrame) -> np.ndarray:
    return np.unique(folders['folder'].to_numpy())


def knn_outlier_scores(feats, folders: pd.DataFrame, key_to_row: dict, k: int = 10) -> np.ndarray:
    feats = np.asarray(feats)
    rows = np.fromiter((key_to_row[s] for s in folders['sample_key']), dtype=np.int64, count=len(folders))
    z = _unit(feats[rows])
    out = np.full(len(folders), np.nan)
    for g in _groups(folders):
        kk = min(int(k), len(g) - 1)
        if kk < 1:
            continue
        S = z[g] @ z[g].T
        np.fill_diagonal(S, -np.inf)
        top = -np.sort(-S, axis=1)[:, :kk]
        out[g] = -top.mean(1)
    return out


def date_conflict_scores(feats, folders: pd.DataFrame, meta: pd.DataFrame) -> np.ndarray:
    """feats 与 meta 行对齐; meta 需要 sample_key, date_day, family"""
    pos = {k: i for i, k in enumerate(meta['sample_key'])}
    rows = np.fromiter((pos[s] for s in folders['sample_key']), dtype=np.int64, count=len(folders))
    f = np.asarray(feats)[rows]
    lab = folders['folder'].to_numpy()
    tc = folders['t_claimed'].to_numpy(dtype=np.float64)
    day = meta['date_day'].to_numpy()[rows]
    fam = meta['family'].to_numpy()[rows]
    est = dateest.style_dates(f, lab, tc, day, fam)
    return dateest.conflict_scores(est, tc, lab)


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    r = rankdata(np.r_[pos, neg])
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def folder_auroc(scores, folders: pd.DataFrame, return_ids: bool = False):
    scores = np.asarray(scores, dtype=np.float64)
    inj = folders['injected'].to_numpy(dtype=bool)
    f = folders['folder'].to_numpy()
    ids = np.unique(f)
    out = np.full(len(ids), np.nan)
    for g in _groups(folders):
        ok = np.isfinite(scores[g])
        gi = g[ok]
        out[np.searchsorted(ids, f[g[0]])] = _auc(scores[gi][inj[gi]], scores[gi][~inj[gi]])
    return (ids, out) if return_ids else out
