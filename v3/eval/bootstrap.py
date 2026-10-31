"""按画师 bootstrap (benchmark v2 的 95% CI): 重采样画师, 不是图.

`draw_weights` 给出每次重采样的画师重数 (多项分布, 行和 = 画师数); 统计量写成 `stat(w) -> dict` 的形式,
w 为长度 n_artists 的权重向量 (重数 x 可选基础权重, 可以含 0), 调用方自己做 sum(w * x) / sum(w * n).
`artist_bootstrap` 给单模型指标的 CI, `paired_bootstrap` 对 b - a 在同一次重采样下做差 (配对),
`population_weights` 把评测集的画师分布 (跨度 x 出道年份) 重加权到参考总体分布.
抽样结果为 None / 非有限值 (NaN, inf) 的丢弃并计入 n_valid, 不让 NaN 进入分位数.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

LO_Q, HI_Q = 2.5, 97.5


def draw_weights(n_artists: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n_artists) int32: 每行是 n_artists 次有放回抽样得到的各画师重数, 行和 = n_artists"""
    rng = np.random.default_rng(seed)
    p = np.full(n_artists, 1.0 / n_artists)
    return rng.multinomial(n_artists, p, size=n_boot).astype(np.int32)


def _valid(x) -> bool:
    return x is not None and bool(np.isfinite(x))


def _base(base_w, n_artists: int) -> np.ndarray:
    if base_w is None:
        return np.ones(n_artists, dtype=np.float64)
    base_w = np.asarray(base_w, dtype=np.float64)
    if base_w.shape != (n_artists,):
        raise ValueError(f'base_w shape {base_w.shape} != ({n_artists},)')
    return base_w


def _keys(*dicts) -> list:
    keys: list = []
    for d in dicts:
        for k in d:
            if k not in keys:
                keys.append(k)
    return keys


def summarize(point: dict, draws: list[dict]) -> dict[str, dict]:
    """每个键 -> {point, lo, hi, n_valid}; lo / hi = 抽样值的 2.5 / 97.5 分位, None / 非有限的抽样丢弃;
    point 为 None (或非有限) 时 lo, hi 也为 None"""
    out = {}
    for k in _keys(point, *draws):
        vals = np.array([float(d[k]) for d in draws if _valid(d.get(k))], dtype=np.float64)
        p = point.get(k)
        p = float(p) if _valid(p) else None
        if p is None or len(vals) == 0:
            lo = hi = None
        else:
            lo, hi = (float(q) for q in np.percentile(vals, [LO_Q, HI_Q]))
        out[k] = dict(point=p, lo=lo, hi=hi, n_valid=int(len(vals)))
    return out


def artist_bootstrap(stat: Callable[[np.ndarray], dict], n_artists: int, n_boot: int = 1000, seed: int = 0,
                     base_w: np.ndarray | None = None) -> dict:
    """point = stat(base_w 或全 1); 每次抽样 stat(W[b] * base_w)"""
    base = _base(base_w, n_artists)
    W = draw_weights(n_artists, n_boot, seed)
    return summarize(stat(base.copy()), [stat(W[b] * base) for b in range(n_boot)])


def paired_bootstrap(stat_a: Callable[[np.ndarray], dict], stat_b: Callable[[np.ndarray], dict],
                     n_artists: int, n_boot: int = 1000, seed: int = 0,
                     base_w: np.ndarray | None = None) -> dict:
    """b - a, 两个统计量每次抽样看到同一个重采样. 输出 {point, lo, hi, n_valid, p_gt0}, p_gt0 = 有效抽样中差 > 0 的比例"""
    base = _base(base_w, n_artists)
    W = draw_weights(n_artists, n_boot, seed)

    def diff(da: dict, db: dict) -> dict:
        return {k: (db[k] - da[k]) if _valid(da.get(k)) and _valid(db.get(k)) else None
                for k in _keys(da, db)}

    pt = diff(stat_a(base.copy()), stat_b(base.copy()))
    draws = []
    for b in range(n_boot):
        w = W[b] * base
        draws.append(diff(stat_a(w), stat_b(w)))
    out = summarize(pt, draws)
    for k, v in out.items():
        vals = np.array([d[k] for d in draws if _valid(d.get(k))], dtype=np.float64)
        v['p_gt0'] = float((vals > 0).mean()) if len(vals) else None
    return out


def population_weights(span: np.ndarray, debut_year: np.ndarray, ref_span: np.ndarray, ref_year: np.ndarray,
                       span_edges=(1, 3, 6), year_edges=(2013, 2018, 2022)) -> tuple[np.ndarray, float]:
    """把集合的画师分布 (跨度 x 出道年份的格) 重加权到参考分布: 权重 = p_ref(格) / p_set(格), 归一化到均值 1.

    格 = np.digitize(span, span_edges) x np.digitize(year, year_edges). 集合里有画师而参考里没有的格
    (p_ref = 0) 权重为 0 (归一化前). span / year 非有限的画师权重为 0, 参考里非有限的行不计入 p_ref.
    返回 (每画师权重, 集合没有覆盖到的格在参考分布里的质量).
    """
    n_y = len(year_edges) + 1
    n_cells = (len(span_edges) + 1) * n_y

    def cells(s, y):
        s, y = np.asarray(s, dtype=np.float64), np.asarray(y, dtype=np.float64)
        ok = np.isfinite(s) & np.isfinite(y)
        c = np.digitize(s, span_edges) * n_y + np.digitize(y, year_edges)
        return c, ok

    c_set, ok_set = cells(span, debut_year)
    c_ref, ok_ref = cells(ref_span, ref_year)
    cnt_set = np.bincount(c_set[ok_set], minlength=n_cells).astype(np.float64)
    cnt_ref = np.bincount(c_ref[ok_ref], minlength=n_cells).astype(np.float64)
    p_set = cnt_set / max(cnt_set.sum(), 1.0)
    p_ref = cnt_ref / max(cnt_ref.sum(), 1.0)
    miss = float(p_ref[cnt_set == 0].sum())
    ratio = np.divide(p_ref, p_set, out=np.zeros(n_cells), where=p_set > 0)
    w = np.where(ok_set, ratio[c_set], 0.0)
    m = w.mean() if len(w) else 0.0
    return (w / m if m > 0 else w), miss
