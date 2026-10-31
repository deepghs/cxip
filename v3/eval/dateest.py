"""年代估计 (benchmark v2 的 date MAE、清洗的 date_conflict 分数、之后 era 工具的 style_year).

同画师内, 用其他图的风格近邻估计这张图的日期: 候选 = 同画师、在 gallery 内、不是自己、上传日不同、
作品族不同 (可选再排除共享任一标签 id 的图), 估计 = top-k 余弦近邻日期的加权平均, 权重 exp(temp * cos).
候选数 < min_candidates 的图得到 NaN (不崩溃, 且 NaN 不会进入和 / 计数, 见 per_artist_mean).
逐画师处理: 每个画师内一个 n_a x n_a 的余弦矩阵 (B1 每人 <= 80 张), 画师内向量化.

加权 bootstrap 的约定: per_artist_mean 返回 (每画师的和, 每画师的有效个数), 调用方用权重 w 算
MAE = sum(w * sum) / sum(w * cnt).
"""
from __future__ import annotations

import sys

import numpy as np


def _as_mask(m, n: int, default: bool) -> np.ndarray:
    return np.full(n, default, dtype=bool) if m is None else np.asarray(m, dtype=bool)


def _unit(f) -> np.ndarray:
    f = np.asarray(f, dtype=np.float64)
    return f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)


def _candidates(idx, day, family, block, gallery) -> np.ndarray:
    """(n, n) bool: [i, j] = idx[j] 可作为 idx[i] 的候选 (同画师已由 idx 保证)"""
    d, fam = day[idx], family[idx]
    ok = (d[:, None] != d[None]) & (fam[:, None] != fam[None]) & gallery[idx][None]
    np.fill_diagonal(ok, False)
    if block is not None:
        b = block[idx]                                   # (n, kb); 负数 = 空位, 不参与排除
        for u in range(b.shape[1]):
            for v in range(b.shape[1]):
                ok &= ~((b[:, u][:, None] == b[:, v][None]) & (b[:, u][:, None] >= 0))
    return ok


def _by_artist(labels):
    labels = np.asarray(labels)
    order = np.argsort(labels, kind='stable')
    cuts = np.flatnonzero(np.diff(labels[order])) + 1
    return np.split(order, cuts) if len(order) else []


def style_dates(feats, labels, t_year, day, family, k=5, temp=10.0, min_candidates=10,
                block=None, query=None, gallery=None) -> np.ndarray:
    labels, t_year = np.asarray(labels), np.asarray(t_year, dtype=np.float64)
    day, family = np.asarray(day), np.asarray(family)
    n = len(labels)
    query, gallery = _as_mask(query, n, True), _as_mask(gallery, n, True)
    block = None if block is None else np.asarray(block).reshape(n, -1)
    z = _unit(feats)
    est = np.full(n, np.nan)
    need = max(int(min_candidates), 1)
    for idx in _by_artist(labels):
        if not query[idx].any():
            continue
        ok = _candidates(idx, day, family, block, gallery)
        S = np.where(ok, z[idx] @ z[idx].T, -np.inf)
        kk = min(int(k), len(idx) - 1)
        if kk < 1:
            continue
        nn = np.argsort(-S, axis=1, kind='stable')[:, :kk]
        s = np.take_along_axis(S, nn, 1)                 # 不合格候选为 -inf, 权重 0
        w = np.exp(temp * (s - np.max(s, axis=1, keepdims=True).clip(min=-1e300)))
        w = np.where(np.isfinite(s), w, 0.0)
        with np.errstate(invalid='ignore', divide='ignore'):
            e = (w * t_year[idx][nn]).sum(1) / w.sum(1)
        good = query[idx] & (ok.sum(1) >= need) & (w.sum(1) > 0)
        est[idx[good]] = e[good]
    return est


def median_baseline(labels, t_year, day, family, min_candidates=10, query=None, gallery=None) -> np.ndarray:
    labels, t_year = np.asarray(labels), np.asarray(t_year, dtype=np.float64)
    day, family = np.asarray(day), np.asarray(family)
    n = len(labels)
    query, gallery = _as_mask(query, n, True), _as_mask(gallery, n, True)
    out = np.full(n, np.nan)
    need = max(int(min_candidates), 1)
    for idx in _by_artist(labels):
        ok = _candidates(idx, day, family, None, gallery)
        for i in np.flatnonzero(query[idx] & (ok.sum(1) >= need)):
            out[idx[i]] = np.median(t_year[idx][ok[i]])
    return out


def conflict_scores(est, t_claimed, labels, floor=0.25) -> np.ndarray:
    """|t - est| / max(同画师 median|t - est|, floor); est 为 NaN 时为 NaN (且不参与画师中位数)"""
    est, t_claimed, labels = (np.asarray(x) for x in (est, t_claimed, labels))
    err = np.abs(t_claimed.astype(np.float64) - est.astype(np.float64))
    out = np.full(len(err), np.nan)
    for idx in _by_artist(labels):
        e = err[idx]
        fin = np.isfinite(e)
        if fin.any():
            out[idx[fin]] = e[fin] / max(float(np.median(e[fin])), floor)
    return out


def date_errors(est, t_true, reliable) -> np.ndarray:
    err = np.abs(np.asarray(est, dtype=np.float64) - np.asarray(t_true, dtype=np.float64))
    return np.where(np.asarray(reliable, dtype=bool) & np.isfinite(err), err, np.nan)


def per_artist_mean(values, labels, n_artists) -> tuple[np.ndarray, np.ndarray]:
    """(每画师的和, 每画师的有效 (非 NaN) 个数); labels 取值 0..n_artists-1"""
    v = np.asarray(values, dtype=np.float64)
    fin = np.isfinite(v)
    lab = np.asarray(labels)[fin].astype(np.int64)
    s = np.bincount(lab, weights=v[fin], minlength=n_artists)[:n_artists]
    c = np.bincount(lab, minlength=n_artists)[:n_artists].astype(np.float64)
    return s, c


FEATS = {'v3_release': '/data/neko/cxip_v3/eval/release_csip_v3/feats',
         'v3_x4_raw': '/data/neko/cxip_v3/exps/csip_p2/dft_r640_tag001_x4/eval_native/feats',
         'kaloscope': '/data/neko/cxip_v3/eval/kaloscope3/proj256/feats'}


def check_reference(set_name: str = 'test100_0918') -> dict:
    """复现 spec §0.3 的参考值 (一次性检查): 预期 v3_release 0.95, kaloscope 0.89, baseline 2.51, 21,710 张"""
    import pandas as pd
    from v3.scripts.csip_full_eval import load_aligned
    from v3.scripts.style_drift import load_meta
    df = load_meta(set_name)
    lab, t = df['label'].values, df['t'].values
    day = pd.to_datetime(df['created_at'], utc=True).dt.floor('D').values
    fam = np.arange(len(df))
    sel = np.zeros(len(df), bool)
    for idx in _by_artist(lab):
        sel[idx] = t[idx].max() - t[idx].min() >= 3
    out = {'baseline': np.abs(median_baseline(lab, t, day, fam, query=sel, gallery=sel) - t)}
    for name, path in FEATS.items():
        est = style_dates(load_aligned(path, set_name, df), lab, t, day, fam, query=sel, gallery=sel)
        out[name] = np.abs(est - t)
    n = int(np.isfinite(out['baseline']).sum())
    res = {'n_images': n}
    print('images', n)
    for name, e in out.items():
        e = e[np.isfinite(e)]
        res[name] = float(e.mean())
        print(f'{name:12s} n={len(e)} MAE {e.mean():.3f}y  median {np.median(e):.3f}y  within 1y {np.mean(e <= 1):.3f}')
    return res


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'check':
        check_reference()
    else:
        sys.exit('usage: python -m v3.eval.dateest check')
