"""benchmark v2 编排: 多模型、共享的按画师 bootstrap 抽样、阈值迁移, 输出 JSON 友好的 dict.

evaluate_set(name, df, models, split, ...) 在一个 B 集合 (按 split 过滤后) 上, 对每个模型只算一次:
  ArtistHists (timestrata), 逐查询时间窗检索值 (retrieval.window_retrieval), 逐图年代误差 (dateest), 逐文件夹
  注入 AUROC (inject), 画师内 drift Spearman; 之后每次 bootstrap 抽样只是这些量的按画师加权和.
  所有模型共用同一个 draw_weights(A, n_boot, seed) 矩阵, 所以 'diff' 是配对差 (同一次重采样下 b - a).
  直方图类指标的抽样走批量路径: W (chunk, A) @ pos (A, G*B) 得到每次抽样的加权直方图, 再交给
  stratified_metrics (单 "画师" 的 ArtistHists), 不对每次抽样重做 (A, G, B) 的加权和.
输出: {model: {metric: {point, lo, hi, n_valid}}, 'diff': {f'{m}-vs-{ref}': {metric: {point, lo, hi, n_valid, p_gt0}}},
       'composition': {...}, 'inject': {kind: 状态}, 'clustering': {model: ...} (仅 split == 'dev'), 'reweight': {...}}.
metric 键名见下方 "metric 键" 的说明 (固定). None / NaN 原样进 bootstrap.summarize, 由它丢弃并计入 n_valid.
boot_out (可选) 收到该集合的抽样矩阵 W 和每模型按画师的 all 组正 / 负直方图, 供 threshold_transfer 做 G2 的 bootstrap.

threshold_transfer(sets, boot=...) 用各集合 all 组曲线做阈值漂移 (pairwise.threshold_drift), G2 口径的 max_cross_f1_loss
({point, lo, hi, n_valid}: 按画师重采样 boot 里的集合 (B1 / B2, 用 evaluate_set 的同一个 W), 其他 G2 集合
(csip_eval_public) 固定; 'diff' 下为相对参照模型的配对差), 默认阈值下的 F1, 源集合 FPR = 1e-3 阈值迁移到目标集合的实际 FPR.
run(models, split, ...) 读集合 (runner.load_set), B2 另报总体重加权 (参考 = danbooru-artist-v2 中 n_local >= 30 的画师),
legacy 集合只进阈值迁移.

Pixiv / B3 (bench2_b3p1 / bench2_b3p2) 已从 v4 去掉 (P0 检查点决定, 2026-11-01): s1 来源是 AI 图语料库
(docs/csip_v4_p0_baseline_zh.md §6.1). evaluate_set 的 cross_source (rows / cols 两个来源) 是与来源无关的通用能力, 保留.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from v3.eval import bootstrap, dateest, inject, pairwise, retrieval
from v3.eval import timestrata as T

V4 = Path('/scratch/cxip_v4')
BENCH = V4 / 'data/bench2_v1'
V2_ARTISTS = V4 / 'data/danbooru-artist-v2-work/artists.parquet'     # artist, n_local, t_first, span, ...
UPLOAD_SOURCE = 'danbooru_upload'                                     # 兜底日期 (不作年代真值)
# G2 口径: B1 + B2 + csip_eval_public (v4 不用 Pixiv / B3, 见 docs/csip_v4_p0_baseline_zh.md §6.1)
G2_SETS = ('bench2_b1', 'bench2_b2', 'csip_eval_public')
REWEIGHT_SETS = ('bench2_b2',)
RESERVED = ('diff', 'clustering', 'composition', 'inject', 'reweight', 'name', 'split')
FPRS = (('1e-3', 'recall@fpr0.001'), ('1e-4', 'recall@fpr0.0001'))
DATE_KW = dict(k=5, temp=10.0, min_candidates=10)
_P_ALL = T.POS_GROUPS.index('all')
_P_CROSS = T.POS_GROUPS.index('cross_content')
_N_ALL = T.NEG_GROUPS.index('all')
_N_CHAR = T.NEG_GROUPS.index('same_char')
_N_COPY = T.NEG_GROUPS.index('same_copy')
CONTENT_KEYS = ('cross_content_recall', 'same_char_fpr', 'same_copy_fpr')
MATCH_FPR, MATCH_TAG = 1e-3, '@fpr1e-3'          # 内容指标的统一总体 FPR (Task 16 A3)

# metric 键 (每模型):
#   pr_auc/<g>, recall@1e-3/<g>, recall@1e-4/<g>  g in timestrata.POS_GROUPS
#       ('<1y','1-3y','3-6y','>=6y','all','cross_content','<1y_xday'); '<1y_xday' = <1y 中原发日 date_day 不同的对 (R20)
#   ap/<1y (= µAP), ap/all
#   cross_content_recall, same_char_fpr, same_copy_fpr      默认阈值 (ModelFeats.threshold) 下, ruling R14
#   cross_content_recall@best, same_char_fpr@best, same_copy_fpr@best   all 组最佳阈值下 (stratified_metrics)
#   cross_content_recall@fpr1e-3, same_char_fpr@fpr1e-3, same_copy_fpr@fpr1e-3   统一总体 FPR 下: 阈值箱 = all 组负对
#       尾部比例 <= 1e-3 的最低箱 (同 _fpr_bin); same_char_ratio@fpr1e-3 = same_char_fpr@fpr1e-3 / 该箱的实际总体 FPR
#       (overall_fpr@fpr1e-3). 每次抽样各自定箱
#   pr_auc/<g>/reweighted (仅传入 reweight 时)
#   map_at_r/1y, r1/1y, map_at_r/any;  drift_spearman_median
#   date_mae, date_hit1y, date_mae_content, date_mae_baseline, date_mae_upload_only;  inject_auroc/<kind>


@dataclass
class ModelFeats:
    name: str
    feats: dict = field(default_factory=dict)    # 集合名 -> (N, D), 与 load_set(集合名) 的行对齐 (过滤前)
    scale: float = 1.0
    bias: float = 0.0
    threshold: float = 0.0                       # 默认阈值 (score 单位)


# ----------------------------------------------------------------------------------------------- 抽样工具
def _f(x) -> float:
    return float(x) if x is not None and np.isfinite(x) else float('nan')


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    num, den = np.asarray(num, np.float64), np.asarray(den, np.float64)
    return np.divide(num, den, out=np.full(np.broadcast(num, den).shape, np.nan), where=den > 0)


def _mean_stat(s: np.ndarray, c: np.ndarray, W: np.ndarray, base: np.ndarray):
    """按画师的 (和, 个数) -> (point, (n_boot,) 抽样值): Σ w·s / Σ w·c"""
    Wb = W * base
    return float(_ratio(base @ s, base @ c)), _ratio(Wb @ s, Wb @ c)


def _strat_scalars(sm: dict, out: dict, suffix: str = '', full: bool = True) -> None:
    for g in T.POS_GROUPS:
        m = sm[g]
        out[f'pr_auc/{g}{suffix}'] = _f(m['pr_auc']) if m else float('nan')
        if full:
            for short, k in FPRS:
                out[f'recall@{short}/{g}'] = _f(m[k]) if m else float('nan')
    if full:
        for g in ('<1y', 'all'):
            out[f'ap/{g}'] = _f(sm[g]['ap']) if sm[g] else float('nan')
        for k in ('cross_content_recall', 'same_char_fpr', 'same_copy_fpr'):
            out[k] = _f(sm[k])


def _tail(x: np.ndarray, b: int) -> float:
    tot = x.sum()
    return float(x[b:].sum() / tot) if tot > 0 else float('nan')


def _strat_draws(h: T.ArtistHists, W: np.ndarray, base: np.ndarray, suffix: str = '', full: bool = True,
                 threshold: float | None = None, chunk_elems: int = 1 << 25):
    """point + 每次抽样的 stratified_metrics 标量. W: (n_boot, A) 重数; 抽样权重 = W * base.
    每画师直方图只转一次 float64, 抽样按块做矩阵乘得到加权直方图, 再按单 "画师" 的 ArtistHists 调 stratified_metrics.
    内容指标 (ruling R14): cross_content_recall / same_char_fpr / same_copy_fpr 取默认阈值 threshold (score 单位,
    预测为正 = 所在箱及以上); stratified_metrics 在 all 组最佳阈值下的值改名为 '<键>@best'. threshold=None 时前者为 NaN.
    另有统一总体 FPR 的 '<键>@fpr1e-3' (阈值箱 = _fpr_bin(all 组负对, 1e-3), 每次抽样各自定), 以及
    same_char_ratio@fpr1e-3 = same_char_fpr@fpr1e-3 / overall_fpr@fpr1e-3 (该箱的实际总体 FPR); 无负对时为 NaN."""
    A, GP, NB = h.pos.shape
    GN = h.neg.shape[1]
    Hp = h.pos.reshape(A, GP * NB).astype(np.float64)
    Hn = h.neg.reshape(A, GN * NB).astype(np.float64)
    b0 = _bin_of(threshold, h.scale, h.bias) if threshold is not None else None

    def one(p, n):
        out = {}
        sm = T.stratified_metrics(T.ArtistHists(p.reshape(1, GP, NB), n.reshape(1, GN, NB), h.scale, h.bias))
        _strat_scalars(sm, out, suffix, full)
        if full:
            for k in CONTENT_KEYS:
                out[f'{k}@best'] = out.pop(k)
            pp, nn = p.reshape(GP, NB), n.reshape(GN, NB)
            vals = ((pp[_P_CROSS], nn[_N_CHAR], nn[_N_COPY]) if b0 is not None else (None,) * 3)
            for k, x in zip(CONTENT_KEYS, vals):
                out[k] = _tail(x, b0) if x is not None else float('nan')
            out.update(_matched_fpr(pp, nn))
        return out

    point = one(base @ Hp, base @ Hn)
    draws = {k: np.full(len(W), np.nan) for k in point}
    Wb = W * base
    step = max(1, chunk_elems // (GP * NB))
    for c0 in range(0, len(W), step):
        P, N = Wb[c0:c0 + step] @ Hp, Wb[c0:c0 + step] @ Hn
        for d in range(len(P)):
            for k, v in one(P[d], N[d]).items():
                draws[k][c0 + d] = v
    return point, draws


def _matched_fpr(pp: np.ndarray, nn: np.ndarray) -> dict:
    """pp: (GP, B) / nn: (GN, B) 加权直方图 -> 统一总体 FPR (MATCH_FPR) 下的内容指标"""
    nan = float('nan')
    bm = _fpr_bin(nn[_N_ALL], MATCH_FPR)
    if bm is None:
        act = scf = scp = ccr = nan
    else:
        act = _tail(nn[_N_ALL], bm)
        ccr, scf, scp = _tail(pp[_P_CROSS], bm), _tail(nn[_N_CHAR], bm), _tail(nn[_N_COPY], bm)
    ratio = scf / act if np.isfinite(scf) and np.isfinite(act) and act > 0 else nan
    t = MATCH_TAG
    return {f'cross_content_recall{t}': ccr, f'same_char_fpr{t}': scf, f'same_copy_fpr{t}': scp,
            f'same_char_ratio{t}': ratio, f'overall_fpr{t}': act}


def _wmedian_draws(v: np.ndarray, W: np.ndarray):
    """每画师一个值 v (NaN = 不合格) 的中位数; 抽样 = 按整数重数展开后的中位数 (偶数取中间两个的均值)"""
    ok = np.isfinite(v)
    vals = np.sort(v[ok])
    point = float(np.median(vals)) if len(vals) else float('nan')
    draws = np.full(len(W), np.nan)
    if not len(vals):
        return point, draws
    Ws = W[:, ok][:, np.argsort(v[ok], kind='stable')].astype(np.int64)
    cum = np.cumsum(Ws, axis=1)
    for b in range(len(W)):
        tot = int(cum[b, -1])
        if tot == 0:
            continue
        lo = vals[np.searchsorted(cum[b], (tot - 1) // 2, side='right')]
        hi = vals[np.searchsorted(cum[b], tot // 2, side='right')]
        draws[b] = 0.5 * (lo + hi)
    return point, draws


def _to_dicts(draws: dict, n: int) -> list[dict]:
    keys = list(draws)
    return [{k: draws[k][b] for k in keys} for b in range(n)]


def _summarize(point: dict, draws: dict, n_boot: int) -> dict:
    return bootstrap.summarize(point, _to_dicts(draws, n_boot))


def _paired(pa: dict, da: dict, pb: dict, db: dict, n_boot: int) -> dict:
    """b - a, 抽样是同一个 W 下的逐次差; 输出 summarize 的 {point, lo, hi, n_valid} 加 p_gt0"""
    keys = [k for k in pa if k in pb]
    pt = {k: pb[k] - pa[k] for k in keys}
    dd = {k: db[k] - da[k] for k in keys}
    out = _summarize(pt, dd, n_boot)
    for k in keys:
        v = dd[k][np.isfinite(dd[k])]
        out[k]['p_gt0'] = float((v > 0).mean()) if len(v) else None
    return out


# ----------------------------------------------------------------------------------------------- 单集合
def _codes(x) -> np.ndarray:
    return pd.factorize(pd.Series(x), sort=True)[0].astype(np.int64)


def _family_codes(df: pd.DataFrame) -> np.ndarray:
    """作品族编号; 有 source 列时按 (source, family) 编号, 避免 Danbooru / Pixiv 的 id 撞号"""
    if 'source' in df.columns:
        return df.groupby(['source', 'family'], sort=False).ngroup().to_numpy(np.int64)
    return _codes(df['family'].to_numpy())


def _tag_lists(df: pd.DataFrame, col: str) -> list:
    if col not in df.columns:
        return [[] for _ in range(len(df))]
    return [list(x) if x is not None and not (isinstance(x, float) and math.isnan(x)) else [] for x in df[col]]


def _drift_per_artist(feats, lab, t, fam, A: int) -> np.ndarray:
    out = np.full(A, np.nan)
    order = np.argsort(lab, kind='stable')
    cuts = np.flatnonzero(np.diff(lab[order])) + 1
    for idx in np.split(order, cuts) if len(order) else []:
        r = T.drift_spearman(feats[idx], lab[idx], t[idx], fam[idx])
        if len(r):
            out[lab[idx[0]]] = r[0]
    return out


def _inject_path(inject_dir: Path, name: str, kind: str) -> Path | None:
    names = [name] + ([name[len('bench2_'):]] if name.startswith('bench2_') else [])
    for n in names:
        p = Path(inject_dir) / f'{n}_{kind}.parquet'
        if p.exists():
            return p
    return None


def _load_inject(inject_dir, name, df_full: pd.DataFrame, split_labels: np.ndarray):
    """{kind: (folders, status)}; 缺文件的类 folders = None 并注明 skipped. 只保留本 split 画师的文件夹"""
    keys = set(df_full['sample_key'])
    out = {}
    for kind in inject.KINDS:
        p = _inject_path(inject_dir, name, kind)
        if p is None:
            out[kind] = (None, dict(status='skipped', reason='missing file'))
            continue
        fo = pd.read_parquet(p)
        n0 = len(fo)
        fo = fo[fo['folder'].isin(split_labels)]
        n_split = len(fo)
        fo = fo[fo['sample_key'].isin(keys)].reset_index(drop=True)
        st = dict(status='ok', file=p.name, n_rows=n_split, n_rows_other_split=n0 - n_split,
                  n_rows_unknown_key=n_split - len(fo), n_folders=int(fo['folder'].nunique()))
        if len(fo) == 0:
            out[kind] = (None, dict(st, status='skipped', reason='no folders in split'))
        else:
            out[kind] = (fo, st)
    return out


def _collapse(h: T.ArtistHists) -> T.ArtistHists:
    return T.ArtistHists(h.pos.sum(0, keepdims=True), h.neg.sum(0, keepdims=True), h.scale, h.bias)


def evaluate_set(name: str, df: pd.DataFrame, models: list[ModelFeats], split: str, n_boot: int = 1000, seed: int = 0,
                 reweight: tuple | None = None, inject_dir=BENCH / 'inject', *, cross_source: bool = False,
                 clustering_kw: dict | None = None, hists_out: dict | None = None, boot_out: dict | None = None) -> dict:
    """reweight = (ref_span, ref_debut_year) 参考总体 (画师级); cross_source: 只用 source == 'pixiv' (rows / 查询) x
    source == 'danbooru' (cols / 库) 的图对 (通用的跨来源能力; 原用于已取消的 B3-P1).
    hists_out: 若给出, 写入 {model: 按画师求和后的 ArtistHists} (供 threshold_transfer).
    boot_out: 若给出, 写入 W (本集合所有模型共用的 (n_boot, A) 抽样矩阵), pos / neg: {model: (A, N_BINS) int64}
    = 每画师 all 组正对 / 负对直方图 (负对在两侧画师各计一次), 供 threshold_transfer 做 G2 的 bootstrap.
    A == 0 时 boot_out 不写."""
    for m in models:
        assert m.name not in RESERVED, f'model name {m.name!r} is reserved'
    df_full = df.reset_index(drop=True)
    df_full = df_full.assign(family=_family_codes(df_full))
    keep = (df_full['split'] == split).to_numpy() if 'split' in df_full.columns else np.ones(len(df_full), bool)
    d = df_full[keep].reset_index(drop=True)
    for m in models:
        assert len(m.feats[name]) == len(df_full), f'{m.name}: feats for {name} not aligned with df'

    raw_lab = d['label'].to_numpy()
    uniq, lab = np.unique(raw_lab, return_inverse=True)
    lab = lab.astype(np.int64)
    A, n = len(uniq), len(d)
    t = d['t_year'].to_numpy(np.float64)
    fam = d['family'].to_numpy(np.int64)
    day = d['date_day'].to_numpy()
    if 'date_source' in d.columns:
        src_s = d['date_source']
        reliable = (src_s.notna() & (src_s != UPLOAD_SOURCE)).to_numpy()   # NA 来源不可靠, 不当真值
        src = src_s.astype(object).where(src_s.notna(), 'NA').astype(str).to_numpy()
    else:
        reliable, src = np.zeros(n, bool), np.full(n, 'NA')
    if cross_source:
        rows = (d['source'] == 'pixiv').to_numpy()
        cols = (d['source'] == 'danbooru').to_numpy()
    else:
        rows = cols = None
    q_mask = np.ones(n, bool) if rows is None else rows
    g_mask = np.ones(n, bool) if cols is None else cols
    chars = T.tag_ids(_tag_lists(d, 'characters'))
    copys = T.tag_ids(_tag_lists(d, 'copyrights'))
    block = np.concatenate([chars, np.where(copys >= 0, copys + chars.max(initial=-1) + 1, -1)], axis=1)

    comp = dict(n_artists=int(A), n_images=int(n), n_query=int(q_mask.sum()), n_gallery=int(g_mask.sum()),
                n_query_artists=int(len(np.unique(lab[q_mask]))), n_date_reliable=int((reliable & q_mask).sum()),
                n_date_upload_only=int((~reliable & q_mask).sum()),
                date_source={k: int(v) for k, v in pd.Series(src[q_mask]).value_counts().items()},
                cross_source=bool(cross_source))
    res: dict = dict(name=name, split=split, composition=comp)
    if A == 0:
        res.update({m.name: {} for m in models}, diff={f'{m.name}-vs-{models[0].name}': {} for m in models[1:]},
                   inject={})
        return res

    W = bootstrap.draw_weights(A, n_boot, seed)                       # 所有模型共用
    if boot_out is not None:
        boot_out.update(W=W, pos={}, neg={})
    day_codes = _codes(day)                                          # 原发日编码 (NaN -> -1 = 未知), 供 <1y_xday
    ones = np.ones(A)
    base_rw = None
    if reweight is not None:
        span = pd.Series(t).groupby(lab).agg(lambda x: x.max() - x.min()).reindex(range(A)).to_numpy()
        debut = pd.Series(t).groupby(lab).min().reindex(range(A)).to_numpy()
        base_rw, miss = bootstrap.population_weights(span, debut, *reweight)
        res['reweight'] = dict(ref_mass_uncovered=float(miss), n_zero_weight=int((base_rw == 0).sum()))

    # 与模型无关: 中位数基线
    base_est = dateest.median_baseline(lab, t, day, fam, DATE_KW['min_candidates'], query=q_mask, gallery=g_mask)
    s_base, c_base = dateest.per_artist_mean(dateest.date_errors(base_est, t, reliable & q_mask), lab, A)

    inj = _load_inject(inject_dir, name, df_full, uniq)
    res['inject'] = {k: st for k, (_, st) in inj.items()}
    key_to_row = {k: i for i, k in enumerate(df_full['sample_key'])}

    points, draws, clus = {}, {}, {}
    for mi, m in enumerate(models):
        f_full = np.asarray(m.feats[name])
        f = f_full[keep]
        pt, dr = {}, {}
        # 1) 时间分层直方图
        h = T.artist_histograms(f, lab, t, fam, chars, copys, m.scale, m.bias, rows=rows, cols=cols, day=day_codes)
        if mi == 0:
            comp['n_pos_pairs'] = {g: int(h.pos[:, i].sum()) for i, g in enumerate(T.POS_GROUPS)}
            comp['n_neg_pairs'] = {g: int(h.neg[:, i].sum() // 2) for i, g in enumerate(T.NEG_GROUPS)}
        p, x = _strat_draws(h, W, ones, threshold=m.threshold)
        pt.update(p); dr.update(x)
        if base_rw is not None:
            p, x = _strat_draws(h, W, base_rw, suffix='/reweighted', full=False)
            pt.update(p); dr.update(x)
        if hists_out is not None:
            hists_out[m.name] = _collapse(h)
        if boot_out is not None:
            boot_out['pos'][m.name] = h.pos[:, _P_ALL].copy()
            boot_out['neg'][m.name] = h.neg[:, _N_ALL].copy()
        del h
        # 2) 时间窗检索
        for win, tag in ((1.0, '1y'), (None, 'any')):
            r = retrieval.window_retrieval(f, lab, t, fam, window=win, query=q_mask, gallery=g_mask)
            keys = ('map_at_r', 'r1') if tag == '1y' else ('map_at_r',)
            for k in keys:
                pt[f'{k}/{tag}'], dr[f'{k}/{tag}'] = _mean_stat(*dateest.per_artist_mean(r[k], lab, A), W, ones)
        # 3) 画师内 drift
        dq = q_mask if cross_source else np.ones(n, bool)
        dv = _drift_per_artist(f[dq], lab[dq], t[dq], fam[dq], A)
        pt['drift_spearman_median'], dr['drift_spearman_median'] = _wmedian_draws(dv, W)
        # 4) 年代估计: 真值只用可靠日期的图, 候选来自 gallery 的全部图
        est = dateest.style_dates(f, lab, t, day, fam, **DATE_KW, query=q_mask, gallery=g_mask)
        err = dateest.date_errors(est, t, reliable & q_mask)
        pt['date_mae'], dr['date_mae'] = _mean_stat(*dateest.per_artist_mean(err, lab, A), W, ones)
        hit = np.where(np.isfinite(err), (err <= 1.0).astype(np.float64), np.nan)
        pt['date_hit1y'], dr['date_hit1y'] = _mean_stat(*dateest.per_artist_mean(hit, lab, A), W, ones)
        est_c = dateest.style_dates(f, lab, t, day, fam, **DATE_KW, block=block, query=q_mask, gallery=g_mask)
        err_c = dateest.date_errors(est_c, t, reliable & q_mask)
        pt['date_mae_content'], dr['date_mae_content'] = _mean_stat(*dateest.per_artist_mean(err_c, lab, A), W, ones)
        pt['date_mae_baseline'], dr['date_mae_baseline'] = _mean_stat(s_base, c_base, W, ones)
        err_u = dateest.date_errors(est, t, ~reliable & q_mask)
        pt['date_mae_upload_only'], dr['date_mae_upload_only'] = _mean_stat(*dateest.per_artist_mean(err_u, lab, A),
                                                                            W, ones)
        # 5) 注入 AUROC (文件夹 = 画师; 打分可用过滤前的全部行, 注入图可能来自另一个 split)
        for kind, (fo, _) in inj.items():
            if fo is None:
                continue
            if kind == 'bad_date':
                sc = inject.date_conflict_scores(f_full, fo, df_full)
            else:
                sc = inject.knn_outlier_scores(f_full, fo, key_to_row)
            ids, au = inject.folder_auroc(sc, fo, return_ids=True)
            v = np.full(A, np.nan)
            v[np.searchsorted(uniq, ids)] = au
            fin = np.isfinite(v)
            pt[f'inject_auroc/{kind}'], dr[f'inject_auroc/{kind}'] = _mean_stat(np.where(fin, v, 0.0),
                                                                              fin.astype(np.float64), W, ones)
        # 6) 聚类 (仅 dev, 不做 bootstrap)
        if split == 'dev' and not cross_source:
            from v3.eval import clustering
            try:
                clus[m.name] = clustering.mixed_pool_clustering(f, lab, seed=seed, **(clustering_kw or {}))
            except ValueError as e:
                clus[m.name] = dict(skipped=str(e))
        points[m.name], draws[m.name] = pt, dr
        res[m.name] = _summarize(pt, dr, n_boot)

    ref = models[0].name
    res['diff'] = {f'{m.name}-vs-{ref}': _paired(points[ref], draws[ref], points[m.name], draws[m.name], n_boot)
                   for m in models[1:]}
    if split == 'dev':
        res['clustering'] = clus if not cross_source else dict(skipped='cross_source set')
    return res


# ----------------------------------------------------------------------------------------------- 阈值迁移
def _all_counts(h: T.ArtistHists):
    return h.pos[:, _P_ALL].sum(0).astype(np.float64), h.neg[:, _N_ALL].sum(0).astype(np.float64) / 2


def _bin_of(score: float, scale: float, bias: float) -> int:
    import torch
    return int(pairwise._bin(torch.tensor([(float(score) - bias) / scale]), T.N_BINS)[0])


def _fpr_bin(neg: np.ndarray, fpr: float) -> int | None:
    """FPR <= fpr 的最低阈值箱 (阈值 = 箱下沿, 预测为正 = 箱 >= b)"""
    tot = neg.sum()
    if tot <= 0:
        return None
    tail = np.cumsum(neg[::-1])[::-1] / tot                 # tail[b] = FPR(阈值 = 箱 b 下沿)
    ok = np.flatnonzero(tail <= fpr)
    return int(ok[0]) if len(ok) else None


def _g2_max_loss(curves: dict) -> float | None:
    """{集合: curve_metrics} (>= 2 个) -> 集合之间逐对 F1 损失的最大值 (pairwise.threshold_drift)"""
    if len(curves) < 2:
        return None
    return float(max(pairwise.threshold_drift(curves)['cross_f1_loss'].values()))


def g2_loss_draws(sets: dict, boot: dict | None, model: str, g2_sets=G2_SETS, chunk: int = 64):
    """G2 的 max_cross_f1_loss 的点估计和按画师 bootstrap 抽样值. 返回 (point, draws 或 None, 被重采样的集合).
    sets: {集合: {模型: ArtistHists}} (按画师求和后的单行版本即可); boot: {集合: evaluate_set 的 boot_out}.
    G2 集合中 boot 里有该模型每画师直方图的 (B1 / B2) 每次抽样按 W[b] 重新加权求和 (负对 / 2), 其余 (csip_eval_public)
    固定为点估计的曲线, 只算一次. 每次抽样: 各集合 T.curve_metrics -> pairwise.threshold_drift -> G2 集合之间的最大损失,
    与点估计完全同一条计算路径. 某次抽样里某个集合的曲线为空 (无正 / 负对) 时该次为 NaN. 各集合的 W 行数须相同."""
    curves = {}
    for s in g2_sets:
        if s in sets and model in sets[s]:
            h = sets[s][model]
            cm = T.curve_metrics(*_all_counts(h), h.scale, h.bias)
            if cm is not None:
                curves[s] = cm
    point = _g2_max_loss(curves)
    bs = [s for s in curves if s in (boot or {}) and model in boot[s].get('pos', {})]
    if point is None or not bs:
        return point, None, bs
    n_boot = {len(boot[s]['W']) for s in bs}
    assert len(n_boot) == 1, f'W of {bs} differ in n_boot: {n_boot}'
    nb = n_boot.pop()
    mats = {s: (boot[s]['W'].astype(np.float64), boot[s]['pos'][model].astype(np.float64),
                boot[s]['neg'][model].astype(np.float64) / 2) for s in bs}
    draws = np.full(nb, np.nan)
    for c0 in range(0, nb, chunk):
        P = {s: (W[c0:c0 + chunk] @ Hp, W[c0:c0 + chunk] @ Hn) for s, (W, Hp, Hn) in mats.items()}
        for d in range(min(chunk, nb - c0)):
            cs = dict(curves)
            for s in bs:
                h = sets[s][model]
                cs[s] = T.curve_metrics(P[s][0][d], P[s][1][d], h.scale, h.bias)
            if all(c is not None for c in cs.values()):
                draws[c0 + d] = _g2_max_loss(cs)
    return point, draws, bs


def _diff_entry(pa, da, pb, db) -> dict:
    """b - a 的配对差 (抽样为同一组 W 下的逐次差): {point, lo, hi, n_valid, p_gt0}"""
    pt = (pb - pa) if pa is not None and pb is not None else None
    if da is None or db is None or len(da) != len(db):
        return dict(point=pt, lo=None, hi=None, n_valid=0, p_gt0=None)
    dd = db - da
    out = bootstrap.summarize({'v': pt}, [{'v': x} for x in dd])['v']
    v = dd[np.isfinite(dd)]
    out['p_gt0'] = float((v > 0).mean()) if len(v) else None
    return out


def threshold_transfer(sets: dict, thresholds: dict | None = None, fpr: float = 1e-3, g2_sets=G2_SETS,
                       boot: dict | None = None, ref: str | None = None) -> dict:
    """sets: {集合名: {模型名: ArtistHists}} (可以是按画师求和后的单行版本); thresholds: {模型名: 默认阈值 (score)};
    boot: {集合名: evaluate_set 的 boot_out} (可缺); ref: 配对差的参照模型 (默认第一个模型).
    返回 {模型名: {per_set, drift (threshold_drift, 全部集合), max_cross_f1_loss {point, lo, hi, n_valid} (仅 G2 集合之间;
    CI 来自 g2_loss_draws, 无每画师数据时 lo / hi = None, n_valid = 0), max_cross_f1_loss_point (同 point, 旧的浮点字段),
    max_cross_f1_loss_sets, max_cross_f1_loss_boot_sets (被重采样的集合), fpr_transfer {'s->t': {fpr, recall,
    source_fpr (源集合在所选箱的实际 FPR)}}, fpr_transfer_nominal_fpr},
    'diff': {f'{m}-vs-{ref}': {'max_cross_f1_loss': {point, lo, hi, n_valid, p_gt0}}}};
    drift 里全部集合的最大损失改名为 max_cross_f1_loss_all_sets, 以免与 G2 口径的 max_cross_f1_loss 混淆;
    g2_sets: G2 口径包含的集合 (默认 G2_SETS)"""
    thresholds = thresholds or {}
    models = []
    for hs in sets.values():
        models += [m for m in hs if m not in models]
    assert 'diff' not in models, "model name 'diff' is reserved"
    out = {}
    g2_pt, g2_dr = {}, {}
    for m in models:
        curves, counts, per = {}, {}, {}
        for s, hs in sets.items():
            if m not in hs:
                continue
            h = hs[m]
            pos, neg = _all_counts(h)
            cm = T.curve_metrics(pos, neg, h.scale, h.bias)
            if cm is None:
                per[s] = None
                continue
            curves[s], counts[s] = cm, (pos, neg)
            e = {k: cm[k] for k in ('pr_auc', 'ap', 'best_f1', 'best_threshold', 'n_pos_pairs', 'n_neg_pairs')}
            b = _fpr_bin(neg, fpr)
            e[f'threshold_at_fpr{fpr:g}'] = (float(h.scale * T._edge_cos(b) + h.bias) if b is not None else None)
            e[f'actual_fpr_at_threshold_at_fpr{fpr:g}'] = float(neg[b:].sum() / neg.sum()) if b is not None else None
            thr = thresholds.get(m)
            if thr is not None:
                at = pairwise.metrics_at_threshold(pairwise.PairHist(pos, neg, h.scale, h.bias), thr)
                bb = _bin_of(thr, h.scale, h.bias)
                e['default'] = dict(at, fpr=float(neg[bb:].sum() / neg.sum()))
            per[s] = e
        res = dict(per_set=per)
        res['drift'] = pairwise.threshold_drift(curves) if len(curves) >= 2 else None
        if res['drift'] and 'max_cross_f1_loss' in res['drift']:      # 全部集合 (含 legacy) 的最大值; G2 口径见下
            res['drift']['max_cross_f1_loss_all_sets'] = res['drift'].pop('max_cross_f1_loss')
        g2 = [s for s in g2_sets if s in curves]
        loss = (res['drift'] or {}).get('cross_f1_loss', {})
        g2_loss = [v for k, v in loss.items() if k.split('->')[0] in g2 and k.split('->')[1] in g2]
        point = float(max(g2_loss)) if g2_loss else None
        p, dr, bs = g2_loss_draws(sets, boot, m, g2_sets)
        assert (p is None) == (point is None) and (p is None or abs(p - point) <= 1e-9), (m, p, point)
        res['max_cross_f1_loss'] = (bootstrap.summarize({'v': point}, [{'v': x} for x in dr])['v'] if dr is not None
                                    else dict(point=point, lo=None, hi=None, n_valid=0))
        res['max_cross_f1_loss_point'] = point
        res['max_cross_f1_loss_sets'] = g2
        res['max_cross_f1_loss_boot_sets'] = bs
        g2_pt[m], g2_dr[m] = point, dr
        tr = {}
        for s in counts:
            b = _fpr_bin(counts[s][1], fpr)
            if b is None:
                continue
            src_fpr = float(counts[s][1][b:].sum() / counts[s][1].sum())
            for t_ in counts:
                if t_ == s:
                    continue
                pos, neg = counts[t_]
                tr[f'{s}->{t_}'] = dict(fpr=float(neg[b:].sum() / neg.sum()), recall=float(pos[b:].sum() / pos.sum()),
                                        source_fpr=src_fpr)
        res['fpr_transfer'] = tr
        res['fpr_transfer_nominal_fpr'] = fpr       # 源集合阈值 = FPR <= 该值的最低箱; 实际源 FPR 见各项 source_fpr
        out[m] = res
    if models:
        ref = ref if ref in models else models[0]
        out['diff'] = {f'{m}-vs-{ref}': {'max_cross_f1_loss': _diff_entry(g2_pt[ref], g2_dr[ref], g2_pt[m], g2_dr[m])}
                       for m in models if m != ref}
    return out


# ----------------------------------------------------------------------------------------------- 全流程
def reference_population(path=V2_ARTISTS, min_local: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """B2 重加权的参考分布: danbooru-artist-v2 中 n_local (= n_artist_local) >= min_local 的画师的
    (span 年, 首图年份). 只在 run() 里调用 (测试 monkeypatch 掉)."""
    from v3.data import dates
    a = pd.read_parquet(path, columns=['n_local', 't_first', 'span'])
    a = a[a['n_local'] >= min_local]
    return a['span'].to_numpy(np.float64), dates.to_year(pd.to_datetime(a['t_first'], utc=True))


def load_set(name):
    from v3.eval import runner
    return runner.load_set(name)


def _legacy_hists(name: str, df: pd.DataFrame, models: list[ModelFeats]) -> dict:
    lab = _codes(df['label'].to_numpy())
    t = np.full(len(df), np.nan)
    fam = np.arange(len(df))
    out = {}
    for m in models:
        h = T.artist_histograms(np.asarray(m.feats[name]), lab, t, fam, scale=m.scale, bias=m.bias)
        out[m.name] = _collapse(h)
    return out


def run(models: list[ModelFeats], split: str = 'dev', sets=('bench2_b1', 'bench2_b2'),
        legacy=('test100_0918', 'csip_eval_public', 'csip_unseen_v1'), n_boot: int = 1000, seed: int = 0,
        inject_dir=BENCH / 'inject', log=print, dfs: dict | None = None, g2_sets=G2_SETS) -> dict:
    """dfs: 可选的已读取集合 {集合名: df} (CLI 读特征时已经读过); 缺的集合用 load_set 读;
    g2_sets: 传给 threshold_transfer 的 G2 口径集合. 参照模型 = models[0].
    out['timing']['g2_bootstrap_s']: threshold_transfer 的总耗时 (主要是 G2 的 bootstrap)."""
    import time
    dfs = dfs or {}

    def get(s):
        return dfs[s] if s in dfs else load_set(s)[0]

    out = dict(split=split, n_boot=n_boot, seed=seed, models=[m.name for m in models], sets={}, skipped={})
    hists: dict = {}
    boot: dict = {}
    ref_pop = None
    for name in sets:
        miss = [f'{m.name}:{name}' for m in models if name not in m.feats]
        if miss:
            out['skipped'][name] = f'missing feats {miss}'
            log(f'[bench2] skip {name}: missing feats {miss}')
            continue
        df = get(name)
        rw = None
        if name in REWEIGHT_SETS:
            if ref_pop is None:
                ref_pop = reference_population()
            rw = ref_pop
        hists[name], boot[name] = {}, {}
        log(f'[bench2] {name}: {len(df)} rows')
        out['sets'][name] = evaluate_set(name, df, models, split, n_boot=n_boot, seed=seed, reweight=rw,
                                         inject_dir=inject_dir, hists_out=hists[name], boot_out=boot[name])
    for name in legacy:
        miss = [m.name for m in models if name not in m.feats]
        if miss:
            out['skipped'][name] = f'missing feats {miss}'
            continue
        df = get(name)
        hists[name] = _legacy_hists(name, df, models)
    t0 = time.perf_counter()
    out['threshold_transfer'] = threshold_transfer(hists, {m.name: m.threshold for m in models}, g2_sets=g2_sets,
                                                   boot=boot, ref=models[0].name)
    out['timing'] = dict(g2_bootstrap_s=round(time.perf_counter() - t0, 3))
    log(f'[bench2] threshold_transfer (G2 bootstrap, n_boot={n_boot}): {out["timing"]["g2_bootstrap_s"]:.1f} s')
    return out
