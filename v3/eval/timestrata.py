"""时间分层验证 (benchmark v2 核心): 每画师的 cos 直方图 + 加权曲线指标.

对每个画师 a 保存两组 cos 直方图 (cos 空间 [-1, 1] 上 N_BINS 个等宽箱):
  pos[a, g]: 画师 a 的同画师正样本对, g = 原发日期差 |Δt| 分层 (<1y / 1-3y / 3-6y / >=6y), all, cross_content
             (两张图的角色标签、作品标签都不重叠; 没有传入任何内容标签时该组为空);
  neg[a, g]: 画师 a 参与的不同画师负样本对, g = all / same_char / same_copy (有共同角色 / 作品标签).
             每个负对在两侧画师各计一次 (所以聚合时 Σ w·neg / 2).
同一作品族 (family) 的对既不算正也不算负; 自身对排除. 任何 PR 类指标 / 每次 bootstrap 抽样都是这些直方图的
加权和 (权重 = 每画师被抽中的次数或总体重加权), 不需要再碰图片.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from v3.eval.pairwise import _bin, _curves_from_counts, _normalize, _summarize, downsample_curve

STRATA = ('<1y', '1-3y', '3-6y', '>=6y')
EDGES = (1.0, 3.0, 6.0)
POS_GROUPS = STRATA + ('all', 'cross_content')
NEG_GROUPS = ('all', 'same_char', 'same_copy')
N_BINS = 1 << 14

_P_ALL, _P_CROSS = POS_GROUPS.index('all'), POS_GROUPS.index('cross_content')
_N_CHAR, _N_COPY = NEG_GROUPS.index('same_char'), NEG_GROUPS.index('same_copy')


@dataclass
class ArtistHists:
    pos: np.ndarray        # (A, len(POS_GROUPS), N_BINS) int64
    neg: np.ndarray        # (A, len(NEG_GROUPS), N_BINS) int64, 每个负对在两侧画师各计一次
    scale: float = 1.0
    bias: float = 0.0


def tag_ids(lists: Sequence[list], k: int = 4, drop=('original',)) -> np.ndarray:
    """每图最多 k 个标签 id (按全集频次从高到低取, 同频按字典序), 不足用 -1 补齐. drop 里的标签忽略."""
    drop = set(drop)
    sets = [sorted({t for t in tags if t not in drop}) for tags in lists]
    cnt = Counter(t for tags in sets for t in tags)
    vocab = {t: i for i, (t, _) in enumerate(sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])))}
    out = np.full((len(sets), k), -1, np.int32)
    for n, tags in enumerate(sets):
        ids = sorted(vocab[t] for t in tags)[:k]
        out[n, :len(ids)] = ids
    return out


def _overlap(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a: (r, k), b: (c, k) 标签 id (-1 = 空) -> (r, c) 布尔: 两图有共同的有效标签 (k × k 次逐元素比较)"""
    m = torch.zeros(len(a), len(b), dtype=torch.bool, device=a.device)
    for i in range(a.shape[1]):
        ai = a[:, i:i + 1]
        valid = ai >= 0
        if not bool(valid.any()):
            continue
        for j in range(b.shape[1]):
            m |= (ai == b[None, :, j]) & valid
    return m


def _overlap_pairs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """逐对版: a, b: (n, k) -> (n,) 布尔"""
    m = torch.zeros(len(a), dtype=torch.bool, device=a.device)
    for i in range(a.shape[1]):
        for j in range(b.shape[1]):
            m |= (a[:, i] == b[:, j]) & (a[:, i] >= 0)
    return m


def _tags(x, n, k_default, device):
    if x is None:
        return None
    x = torch.as_tensor(np.asarray(x), dtype=torch.int64, device=device)
    assert x.ndim == 2 and len(x) == n, 'tag id arrays must be (N, k)'
    return x


@torch.no_grad()
def artist_histograms(feats, labels, t_year, family, chars=None, copys=None, scale=1.0, bias=0.0, rows=None, cols=None,
                      chunk=2048, device='cpu') -> ArtistHists:
    """rows × cols 的所有图对 (布尔掩码, 默认全部) 的每画师直方图. 特征先 (转 float32 后) L2 归一化.
    rows 与 cols 都为全部时只计 i < j; 否则计 rows 中的 i 与 cols 中的 j 的全部 (i, j) 对 (i != j), 例如 B3-P1 用
    rows = Pixiv 查询, cols = Danbooru 库. labels 为 0..A-1 的连续整数. t_year 为 NaN 的正对只进 all / cross_content."""
    n = len(labels)
    dev = torch.device(device)
    z = _normalize(feats).to(dev)
    lab = torch.as_tensor(np.asarray(labels), dtype=torch.int64, device=dev)
    fam = torch.as_tensor(np.asarray(family), dtype=torch.int64, device=dev)
    t = torch.as_tensor(np.asarray(t_year), dtype=torch.float64, device=dev)
    chars_t, copys_t = _tags(chars, n, 4, dev), _tags(copys, n, 4, dev)
    have_content = chars_t is not None or copys_t is not None
    rows = np.ones(n, bool) if rows is None else np.asarray(rows, bool)
    cols = np.ones(n, bool) if cols is None else np.asarray(cols, bool)
    full = bool(rows.all() and cols.all())
    ri = torch.as_tensor(np.nonzero(rows)[0], device=dev)
    ci = torch.as_tensor(np.nonzero(cols)[0], device=dev)
    A, NB = int(lab.max()) + 1 if n else 0, N_BINS
    GP, GN = len(POS_GROUPS), len(NEG_GROUPS)
    pos = torch.zeros(A * GP * NB, dtype=torch.int64, device=dev)
    neg = torch.zeros(A, GN, NB, dtype=torch.int64, device=dev)
    neg_flat = neg.view(-1)
    edges = torch.tensor(EDGES, dtype=torch.float64, device=dev)
    chunk = max(1, int(chunk))
    cb = max(256, (1 << 24) // chunk)                      # 列块宽: 单个 (r, cb) 块 <= 16M 个元素
    trash = A * NB

    def add(flat, key):
        if len(key):
            flat.index_add_(0, key, torch.ones(len(key), dtype=torch.int64, device=dev))

    for r0 in range(0, len(ri), chunk):
        rr = ri[r0:r0 + chunk]
        zr, lr, fr = z[rr], lab[rr], fam[rr]
        for c0 in range(0, len(ci), cb):
            cc = ci[c0:c0 + cb]
            if full and int(cc[-1]) <= int(rr[0]):         # ci / rr 升序: 该列块没有 j > i
                continue
            bins = _bin(zr @ z[cc].T, NB)
            lc = lab[cc]
            valid = fr[:, None] != fam[cc][None, :]
            valid &= (rr[:, None] < cc[None, :]) if full else (rr[:, None] != cc[None, :])
            same = lr[:, None] == lc[None, :]
            # 负对: 全部 -- 无效位置落到垃圾箱, 不需要布尔索引
            nm = valid & ~same
            for side in (lr[:, None], lc[None, :]):
                key = torch.where(nm, side * NB + bins, torch.full_like(bins, trash))
                bc = torch.bincount(key.reshape(-1), minlength=trash + 1)
                neg[:, 0] += bc[:trash].view(A, NB)
            if have_content:
                for g, tg in ((_N_CHAR, chars_t), (_N_COPY, copys_t)):
                    if tg is None:
                        continue
                    ii, jj = torch.nonzero(nm & _overlap(tg[rr], tg[cc]), as_tuple=True)
                    b = bins[ii, jj]
                    add(neg_flat, lr[ii] * (GN * NB) + g * NB + b)
                    add(neg_flat, lc[jj] * (GN * NB) + g * NB + b)
            # 正对: 同画师、不同作品族
            ii, jj = torch.nonzero(valid & same, as_tuple=True)
            if len(ii) == 0:
                continue
            b = bins[ii, jj]
            a = lr[ii]
            gap = (t[rr[ii]] - t[cc[jj]]).abs()
            st = torch.bucketize(gap, edges, right=True)
            st[torch.isnan(gap)] = -1
            keys = [a * (GP * NB) + _P_ALL * NB + b]
            ok = st >= 0
            keys.append(a[ok] * (GP * NB) + st[ok] * NB + b[ok])
            if have_content:
                ov = torch.zeros(len(ii), dtype=torch.bool, device=dev)
                for tg in (chars_t, copys_t):
                    if tg is not None:
                        ov |= _overlap_pairs(tg[rr[ii]], tg[cc[jj]])
                keys.append(a[~ov] * (GP * NB) + _P_CROSS * NB + b[~ov])
            add(pos, torch.cat(keys))
    return ArtistHists(pos.view(A, GP, NB).cpu().numpy(), neg.cpu().numpy(), float(scale), float(bias))


def _edge_cos(b):
    return -1.0 + 2.0 * np.asarray(b) / N_BINS


def curve_metrics(pos: np.ndarray, neg: np.ndarray, scale: float, bias: float, fprs=(1e-3, 1e-4)) -> dict | None:
    """pos / neg: (N_BINS,) 每箱的 (可加权的) 正 / 负对数. 同箱视为同分, 阈值 = 箱下沿 (score 单位).
    pos 或 neg 总数为 0 时返回 None. recall@fpr{f:g}: FPR <= f 的最大召回."""
    pos = np.asarray(pos, np.float64)
    neg = np.asarray(neg, np.float64)
    n_pos, n_neg = float(pos.sum()), float(neg.sum())
    if not (n_pos > 0 and n_neg > 0):
        return None
    nz = np.nonzero((pos + neg) > 0)[0][::-1]                # 从高到低
    tp, fp, P, R, F1, FPR = _curves_from_counts(pos[nz], neg[nz], n_pos, n_neg)
    thr_cos = _edge_cos(nz)
    thr_score = scale * thr_cos + bias
    out = dict(n_pos_pairs=n_pos, n_neg_pairs=n_neg)
    out.update(_summarize(tp, fp, P, R, F1, FPR, thr_score, thr_cos))
    for f in fprs:
        ok = FPR <= f
        out[f'recall@fpr{f:g}'] = float(R[ok].max()) if ok.any() else 0.0
    out['curve'] = downsample_curve(thr_score, P, R, F1)
    return out


def _weighted_sum(x: np.ndarray, w: np.ndarray, block: int = 64) -> np.ndarray:
    """Σ_a w[a] · x[a] (x: (A, ...) int64 -> float64), 分块转换以免复制整个数组; 跳过权重为 0 的画师"""
    out = np.zeros(x.shape[1:], np.float64)
    nzi = np.nonzero(w != 0)[0]
    for s in range(0, len(nzi), block):
        idx = nzi[s:s + block]
        out += np.tensordot(w[idx], x[idx].astype(np.float64), axes=(0, 0))
    return out


def stratified_metrics(h: ArtistHists, w: np.ndarray | None = None) -> dict:
    """键为 POS_GROUPS 中的每一组: curve_metrics(Σ w·pos[:, g], Σ w·neg[:, all] / 2), 空则 None.
    另有 cross_content_recall / same_char_fpr / same_copy_fpr (全部 None 若 all 组为空或分母为 0),
    都在 all 组的最佳阈值下计算 (预测为正 = score >= best_threshold)."""
    A = h.pos.shape[0]
    w = np.ones(A) if w is None else np.asarray(w, np.float64)
    assert w.shape == (A,)
    pos = _weighted_sum(h.pos, w)                            # (G, B)
    neg = _weighted_sum(h.neg, w)                            # (3, B)
    neg_all = neg[0] / 2
    out = {g: curve_metrics(pos[i], neg_all, h.scale, h.bias) for i, g in enumerate(POS_GROUPS)}
    out.update(cross_content_recall=None, same_char_fpr=None, same_copy_fpr=None)
    m = out['all']
    if m is None:
        return out
    b = int(np.rint((((m['best_threshold'] - h.bias) / h.scale) + 1.0) / 2.0 * N_BINS))    # 最佳阈值所在箱
    b = min(max(b, 0), N_BINS - 1)

    def tail(x):
        tot = x.sum()
        return float(x[b:].sum() / tot) if tot > 0 else None
    out['cross_content_recall'] = tail(pos[_P_CROSS])
    out['same_char_fpr'] = tail(neg[_N_CHAR])
    out['same_copy_fpr'] = tail(neg[_N_COPY])
    return out


def drift_spearman(feats, labels, t_year, family, min_images: int = 10, min_span: float = 3.0) -> np.ndarray:
    """每个合格画师 (有效日期的图 >= min_images 张且日期跨度 >= min_span 年) 一个值: 画师内 (不同作品族) 图对的
    cos 与 |Δt| 的 Spearman. 值为 NaN (cos 或 |Δt| 为常数) 的画师不计入."""
    from scipy.stats import spearmanr
    z = _normalize(feats)
    lab = np.asarray(labels)
    t = np.asarray(t_year, np.float64)
    fam = np.asarray(family)
    ok = ~np.isnan(t)
    order = np.argsort(lab, kind='stable')
    bounds = np.r_[0, np.nonzero(np.diff(lab[order]))[0] + 1, len(order)]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        idx = idx[ok[idx]]
        if len(idx) < min_images or t[idx].max() - t[idx].min() < min_span:
            continue
        iu = np.triu_indices(len(idx), 1)
        keep = fam[idx][iu[0]] != fam[idx][iu[1]]
        i, j = idx[iu[0][keep]], idx[iu[1][keep]]
        if len(i) < 3:
            continue
        cos = (z[i] * z[j]).sum(-1).numpy()
        r = spearmanr(cos, np.abs(t[i] - t[j]))[0]
        if np.isfinite(r):
            out.append(float(r))
    return np.asarray(out, np.float64)
