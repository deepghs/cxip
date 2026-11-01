"""按画师 bootstrap (benchmark v2 的 95% CI): 重采样画师, 不是图.

`draw_weights` 给出每次重采样的画师重数 (多项分布, 行和 = 画师数); 统计量写成 `stat(w) -> dict` 的形式,
w 为长度 n_artists 的权重向量 (重数 x 可选基础权重, 可以含 0), 调用方自己做 sum(w * x) / sum(w * n).
`artist_bootstrap` 给单模型指标的 CI, `paired_bootstrap` 对 b - a 在同一次重采样下做差 (配对),
`population_weights` 把评测集的画师分布 (跨度 x 出道年份) 重加权到参考总体分布.
抽样结果为 None / 非有限值 (NaN, inf) 的丢弃并计入 n_valid, 不让 NaN 进入分位数.

画师对 (pair) 统计量的抽样权重 (裁定 R22): 同画师正对 (a, a) 的真实重采样权重是 k_a (乘基础权重), 线性, 照用 W·base.
不同画师负对 (a, a') 的真实重采样权重是乘积 k_a·k_a'; 旧做法给 (k_a + k_a') / 2 (线性平均), 它的偏差项是乘积
线性化 (k_a − 1) + (k_a' − 1) 的一半, 所以负对一侧统计量的 bootstrap SD 被低估一半 (CI 窄约 2 倍). 现在负对用乘积的
线性化 (Hájek) 权重 `pair_draw_coef`: 对 "每对在两侧画师各计一次" 的每画师直方图 Hn, 单侧计数的抽样直方图 =
((W − 1)·base) @ Hn + (base @ Hn) / 2 = ((2W − 1)·base) @ Hn / 2; base = 1 时每对权重 = k_a + k_a' − 1.
W = 1 时与点估计完全相同. 单个箱可能为负, `pair_draw_hists` 截到 0.
`paired_diff` 是所有配对差 (b − a, 同一个 W 下逐次相减) 的共用汇总: {point, lo, hi, n_valid, p_gt0}.
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


def pair_draw_coef(W: np.ndarray, base_w: np.ndarray | None = None) -> np.ndarray:
    """不同画师对的抽样系数 (2W − 1)·base (n_boot, A): 对两侧各计一次的每画师直方图 H, coef @ H 是两侧计数口径的
    抽样直方图 (/ 2 得单侧计数); 每对 (a, a') 的权重 = 乘积 k_a·k_a' 的线性化 (base = 1 时 k_a + k_a' − 1). W = 1 时 = base"""
    W = np.asarray(W, dtype=np.float64)
    base = _base(base_w, W.shape[-1])
    return (2.0 * W - 1.0) * base


def pair_draw_hists(W: np.ndarray, H: np.ndarray, base_w: np.ndarray | None = None) -> np.ndarray:
    """max(pair_draw_coef(W, base) @ H, 0): 不同画师对的抽样直方图 (两侧计数口径, 与 base @ H 同一口径).
    线性化权重可为负 (k = 0 的画师), 个别箱会 < 0, 截到 0 后尾部 / 曲线都单调、有限"""
    return np.maximum(pair_draw_coef(W, base_w) @ H, 0.0)


def paired_diff(pa: dict, da: dict, pb: dict, db: dict) -> dict:
    """b − a 的配对差. pa / pb: {键: 点估计 (可为 None)}; da / db: {键: (n_boot,) 同一组 W 下的抽样值 (或 None)}.
    键 = pa 与 pb 共有的键. 输出每键 {point, lo, hi, n_valid, p_gt0}; 任一侧点估计无效则 point = None;
    任一侧没有抽样 (或长度不同) 则 lo = hi = None, n_valid = 0, p_gt0 = None"""
    out = {}
    for k in [k for k in pa if k in pb]:
        a, b = pa[k], pb[k]
        pt = (float(b) - float(a)) if _valid(a) and _valid(b) else None
        xa, xb = da.get(k), db.get(k)
        if xa is None or xb is None or len(xa) != len(xb):
            out[k] = dict(point=pt, lo=None, hi=None, n_valid=0, p_gt0=None)
            continue
        dd = np.asarray(xb, dtype=np.float64) - np.asarray(xa, dtype=np.float64)
        v = dd[np.isfinite(dd)]
        r = summarize({'v': pt}, [{'v': x} for x in dd])['v']
        r['p_gt0'] = float((v > 0).mean()) if len(v) else None
        out[k] = r
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

    pa, pb = stat_a(base.copy()), stat_b(base.copy())
    keys = _keys(pa, pb)
    da, db = {k: np.full(n_boot, np.nan) for k in keys}, {k: np.full(n_boot, np.nan) for k in keys}
    for b in range(n_boot):
        w = W[b] * base
        ra, rb = stat_a(w), stat_b(w)
        for k in keys:
            if _valid(ra.get(k)) and _valid(rb.get(k)):
                da[k][b], db[k][b] = ra[k], rb[k]
    return paired_diff({k: pa.get(k) for k in keys}, da, {k: pb.get(k) for k in keys}, db)


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
