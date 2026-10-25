"""两两配对评测 (模型无关: 输入特征矩阵 + 类标签).

score(i, j) = scale * cos(f_i, f_j) + bias (scale > 0, 单调变换), 同类为正样本, 全部 i < j 的对.
大集合 (2.5 万张 = 3.1 亿对) 不排序: 在 cos 空间 [-1, 1] 上做 2^20 个等宽箱的直方图 (全部对) +
逐类枚举正样本对的直方图, 负样本 = 全部 - 正样本; PR / ROC / F1 曲线按箱边界计算 (同箱视为同分).
箱宽 1.9e-6 (cos), 与精确排序的差异 < 1e-5 (见 v3/tests/test_eval_pairwise.py). 小集合可用 exact=True.

指标定义与 experiments/common/pair_eval.py 一致: pr_auc = trapz((R=0, P=1) + PR 曲线) (卡片上的 "PR Curve AUC"),
ap = sum dR * P, roc_auc = trapz(TPR, FPR) (同分按 0.5 计), best F1 及阈值 (score 单位, 预测为正 = score >= 阈值).
新增: F1 平台宽度 (包含最佳阈值、F1 >= 0.98 * max 的连续阈值区间, score 和 cos 两种单位)、固定阈值 P/R/F1、
阈值漂移 (多个数据集最佳阈值的离散程度, 及用 A 的最佳阈值在 B 上损失多少 F1)、CSIP 跨角色协议.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

N_BINS = 1 << 20


def _normalize(feats) -> torch.Tensor:
    f = torch.as_tensor(np.asarray(feats) if not torch.is_tensor(feats) else feats).float()
    return torch.nn.functional.normalize(f, dim=-1)


def _bin(cos: torch.Tensor, n_bins: int) -> torch.Tensor:
    return ((cos.clamp(-1, 1) + 1) * (n_bins / 2)).long().clamp_(0, n_bins - 1)


@dataclass
class PairHist:
    """cos 空间直方图: pos[b], neg[b] 为 cos 落在 [edge_b, edge_{b+1}) 的正/负样本对数"""
    pos: np.ndarray
    neg: np.ndarray
    scale: float = 1.0
    bias: float = 0.0

    @property
    def n_bins(self):
        return len(self.pos)

    def edges_cos(self):
        return np.linspace(-1.0, 1.0, self.n_bins + 1)

    def to_score(self, cos):
        return self.scale * np.asarray(cos) + self.bias

    def to_cos(self, score):
        return (np.asarray(score) - self.bias) / self.scale


def positive_pair_cos(f: torch.Tensor, labels: np.ndarray):
    """逐类枚举 i<j 的同类对的 cos (正样本对很少, 不需要全矩阵)"""
    out = []
    order = np.argsort(labels, kind='stable')
    lab = labels[order]
    bounds = np.r_[0, np.nonzero(np.diff(lab))[0] + 1, len(lab)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 2:
            continue
        idx = torch.as_tensor(order[a:b])
        g = f[idx]
        s = g @ g.T
        iu = torch.triu_indices(len(idx), len(idx), 1)
        out.append(s[iu[0], iu[1]])
    return torch.cat(out) if out else torch.zeros(0)


def pair_histogram(feats, labels, scale=1.0, bias=0.0, n_bins=N_BINS, chunk=2048, threads=None) -> PairHist:
    """全部 i<j 对的 cos 直方图 (分块矩阵乘 + bincount), 负样本 = 全部 - 正样本.
    先按标签排序, 每个行块的正样本只在 [r0, 该块最后一类的结尾) 的列窗口内; 正样本的 cos 取自与全部对
    同一个矩阵块 (保证分箱一致, 否则浮点误差会让少数正样本落到相邻箱, 稀疏的高分区出现负的负样本数)"""
    if threads:
        torch.set_num_threads(threads)
    labels = np.asarray(labels)
    order = np.argsort(labels, kind='stable')
    f = _normalize(feats)[torch.as_tensor(order)]
    lab = labels[order]
    _, lab_id = np.unique(lab, return_inverse=True)
    grp_end = np.searchsorted(lab_id, lab_id, side='right')    # 每张图所属类在排序后序列中的结尾
    lab_t = torch.as_tensor(lab_id)
    n = len(f)
    total = torch.zeros(n_bins, dtype=torch.int64)
    pos = torch.zeros(n_bins, dtype=torch.int64)
    with torch.no_grad():
        for r0 in range(0, n, chunk):
            r1 = min(n, r0 + chunk)
            s = f[r0:r1] @ f[r0:].T                       # (r1-r0, n-r0)
            # 对角块只要上三角 (j > i)
            blk = s[:, :r1 - r0]
            iu = torch.triu_indices(r1 - r0, r1 - r0, 1)
            parts = [blk[iu[0], iu[1]]]
            if r1 < n:
                parts.append(s[:, r1 - r0:].reshape(-1))
            for p in parts:
                total += torch.bincount(_bin(p, n_bins), minlength=n_bins)
            w = int(grp_end[r1 - 1]) - r0
            m = lab_t[r0:r1, None] == lab_t[None, r0:r0 + w]
            m &= torch.arange(r0, r0 + w)[None] > torch.arange(r0, r1)[:, None]
            pv = s[:, :w][m]
            if len(pv):
                pos += torch.bincount(_bin(pv, n_bins), minlength=n_bins)
    neg = total - pos
    assert int(neg.min()) >= 0
    return PairHist(pos.numpy(), neg.numpy(), float(scale), float(bias))


def exact_pair_scores(feats, labels, scale=1.0, bias=0.0):
    """小集合: 全部 i<j 对的 score 与是否同类 (与 pair_eval.pair_scores 相同)"""
    f = _normalize(feats)
    sim = (f @ f.T) * scale + bias
    iu = torch.triu_indices(len(f), len(f), 1)
    c = torch.as_tensor(np.asarray(labels))
    return sim[iu[0], iu[1]].numpy(), (c[iu[0]] == c[iu[1]]).numpy()


# ------------------------------------------------------------------------------------------------- curves / metrics
def _curves_from_counts(tp_step, fp_step, n_pos, n_neg):
    """tp_step/fp_step 为按阈值从高到低每个台阶新增的正/负数; 返回累计 tp, fp, P, R, F1, FPR"""
    tp = np.cumsum(tp_step, dtype=np.float64)
    fp = np.cumsum(fp_step, dtype=np.float64)
    P = tp / np.maximum(tp + fp, 1e-300)
    R = tp / max(n_pos, 1)
    F1 = 2 * P * R / np.maximum(P + R, 1e-12)
    FPR = fp / max(n_neg, 1)
    return tp, fp, P, R, F1, FPR


def _summarize(tp, fp, P, R, F1, FPR, thr_score, thr_cos=None, plateau_rel=0.98):
    best = int(np.argmax(F1))
    ap = float(np.sum(np.diff(np.r_[0.0, R]) * P))
    pr_auc = float(np.trapezoid(np.r_[1.0, P], np.r_[0.0, R]))
    roc = float(np.trapezoid(np.r_[0.0, R], np.r_[0.0, FPR]))
    out = dict(pr_auc=pr_auc, ap=ap, roc_auc=roc, best_f1=float(F1[best]), best_threshold=float(thr_score[best]),
               precision_at_best=float(P[best]), recall_at_best=float(R[best]))
    out.update(f1_plateau(F1, thr_score, best, plateau_rel, thr_cos))
    return out


def f1_plateau(F1, thr_score, best=None, rel=0.98, thr_cos=None):
    """包含最佳阈值、F1 >= rel * max 的连续区间 (阈值按从高到低排列). 宽度 = 区间两端阈值之差"""
    F1 = np.asarray(F1)
    best = int(np.argmax(F1)) if best is None else best
    ok = F1 >= rel * F1[best]
    lo = best
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    hi = best
    while hi < len(F1) - 1 and ok[hi + 1]:
        hi += 1
    out = dict(f1_plateau_rel=rel, f1_plateau_lo=float(thr_score[hi]), f1_plateau_hi=float(thr_score[lo]),
               f1_plateau_width=float(thr_score[lo] - thr_score[hi]))
    if thr_cos is not None:
        out['f1_plateau_width_cos'] = float(thr_cos[lo] - thr_cos[hi])
    return out


def metrics_from_hist(h: PairHist, plateau_rel=0.98, curve_points=256) -> dict:
    """从直方图得到全部指标; 台阶 = 非空箱 (阈值取该箱下沿)"""
    nz = np.nonzero((h.pos + h.neg) > 0)[0][::-1]       # 从高到低
    tp_step, fp_step = h.pos[nz], h.neg[nz]
    n_pos, n_neg = int(h.pos.sum()), int(h.neg.sum())
    tp, fp, P, R, F1, FPR = _curves_from_counts(tp_step, fp_step, n_pos, n_neg)
    thr_cos = h.edges_cos()[nz]
    thr_score = h.to_score(thr_cos)
    out = dict(n_pairs=n_pos + n_neg, n_pos_pairs=n_pos)
    out.update(_summarize(tp, fp, P, R, F1, FPR, thr_score, thr_cos, plateau_rel))
    if curve_points:
        out['curve'] = downsample_curve(thr_score, P, R, F1, curve_points)
    return out


def metrics_exact(scores, labels, plateau_rel=0.98, curve_points=0) -> dict:
    """精确版 (排序), 与 pair_eval.pr_metrics 数值一致"""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    order = np.argsort(-scores, kind='stable')
    s, y = scores[order], labels[order]
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    cy = np.cumsum(y)
    tp_cum = cy[last].astype(np.float64)
    fp_cum = (last + 1) - tp_cum
    tp_step = np.diff(np.r_[0.0, tp_cum])
    fp_step = np.diff(np.r_[0.0, fp_cum])
    n_pos = int(y.sum())
    tp, fp, P, R, F1, FPR = _curves_from_counts(tp_step, fp_step, n_pos, len(y) - n_pos)
    out = dict(n_pairs=len(y), n_pos_pairs=n_pos)
    out.update(_summarize(tp, fp, P, R, F1, FPR, s[last], None, plateau_rel))
    if curve_points:
        out['curve'] = downsample_curve(s[last], P, R, F1, curve_points)
    return out


def downsample_curve(thr, P, R, F1, k=256):
    """按召回率均匀取 k 个点, 存 json 用 (画 PR / F1-阈值曲线、算漂移)"""
    if len(thr) <= k:
        idx = np.arange(len(thr))
    else:
        idx = np.unique(np.searchsorted(R, np.linspace(R[0], R[-1], k)).clip(0, len(R) - 1))
        idx = np.unique(np.r_[idx, int(np.argmax(F1))])
    return dict(threshold=np.round(thr[idx], 5).tolist(), precision=np.round(P[idx], 6).tolist(),
                recall=np.round(R[idx], 6).tolist(), f1=np.round(F1[idx], 6).tolist())


def metrics_at_threshold(h: PairHist, threshold: float) -> dict:
    """固定阈值 (score 单位) 下的 P / R / F1: 预测为正 = score >= threshold"""
    b = int(_bin(torch.tensor([float(h.to_cos(threshold))]), h.n_bins)[0])
    tp, fp = float(h.pos[b:].sum()), float(h.neg[b:].sum())
    n_pos = float(h.pos.sum())
    P = tp / max(tp + fp, 1e-300)
    R = tp / max(n_pos, 1)
    return dict(threshold=float(threshold), precision=P, recall=R, f1=2 * P * R / max(P + R, 1e-12))


def pairwise_eval(feats, labels, scale=1.0, bias=0.0, exact=None, plateau_rel=0.98, fixed_thresholds=(),
                  curve_points=256, n_bins=N_BINS, chunk=2048) -> dict:
    """主入口. exact=None: <= 4000 张用精确排序, 否则直方图"""
    n = len(labels)
    if exact is None:
        exact = n <= 4000
    labels = np.asarray(labels)
    if exact:
        s, y = exact_pair_scores(feats, labels, scale, bias)
        out = metrics_exact(s, y, plateau_rel, curve_points)
        h = None
    else:
        h = pair_histogram(feats, labels, scale, bias, n_bins, chunk)
        out = metrics_from_hist(h, plateau_rel, curve_points)
    out.update(n_images=int(n), n_classes=int(len(np.unique(labels))), scale=float(scale), bias=float(bias),
               method='exact' if exact else f'hist{n_bins}')
    out['f1_plateau_width_cos'] = out['f1_plateau_width'] / float(scale)
    if fixed_thresholds:
        if h is None:
            h = pair_histogram(feats, labels, scale, bias, n_bins, chunk)
        out['fixed'] = [metrics_at_threshold(h, t) for t in fixed_thresholds]
    return out


# ------------------------------------------------------------------------------------------------- threshold drift
def threshold_drift(results: dict) -> dict:
    """results: {数据集: pairwise_eval 输出}. 返回最佳阈值的离散程度, 以及把 A 的最佳阈值用在 B 上的 F1 损失
    (用各自存的 curve 插值; 损失 = 1 - F1_B(thr_A) / maxF1_B)"""
    names = list(results)
    thr = {k: results[k]['best_threshold'] for k in names}
    v = np.array(list(thr.values()))
    out = dict(best_thresholds=thr, spread=float(v.max() - v.min()) if len(v) else 0.0, std=float(v.std()),
               mean=float(v.mean()) if len(v) else 0.0)
    loss = {}
    for b in names:
        c = results[b].get('curve')
        if not c:
            continue
        t = np.array(c['threshold'])[::-1]
        f = np.array(c['f1'])[::-1]
        for a in names:
            if a == b:
                continue
            f_ab = float(np.interp(thr[a], t, f))
            loss[f'{a}->{b}'] = 1 - f_ab / max(results[b]['best_f1'], 1e-12)
    if loss:
        out['cross_f1_loss'] = loss
        out['max_cross_f1_loss'] = max(loss.values())
    # 共用一个阈值 (各数据集最佳阈值的中位数) 时的平均 F1 损失
    if loss:
        med = float(np.median(v))
        ls = []
        for b in names:
            c = results[b]['curve']
            ls.append(1 - float(np.interp(med, np.array(c['threshold'])[::-1], np.array(c['f1'])[::-1]))
                      / max(results[b]['best_f1'], 1e-12))
        out['shared_threshold'] = med
        out['shared_threshold_mean_f1_loss'] = float(np.mean(ls))
    return out


# --------------------------------------------------------------------------------------------- CSIP cross-character
def _group_pairs(labels: np.ndarray):
    """同标签 (>= 0) 的全部 i<j 对的下标"""
    ii, jj = [], []
    order = np.argsort(labels, kind='stable')
    lab = labels[order]
    bounds = np.r_[0, np.nonzero(np.diff(lab))[0] + 1, len(lab)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        if lab[a] < 0 or b - a < 2:
            continue
        idx = order[a:b]
        iu = np.triu_indices(b - a, 1)
        ii.append(idx[iu[0]])
        jj.append(idx[iu[1]])
    if not ii:
        return np.zeros(0, int), np.zeros(0, int)
    i, j = np.concatenate(ii), np.concatenate(jj)
    return np.minimum(i, j), np.maximum(i, j)


def balanced_pr_auc(pos_scores, neg_scores) -> float:
    """负样本按 n_pos / n_neg 重新加权 (等价于正负各占一半) 后的 PR 曲线 AUC (trapz, 同 pr_auc 的定义).
    跨角色协议里同画师对比困难负样本多两个数量级, 不加权的 PR-AUC 接近 1, 区分不了模型"""
    s = np.r_[pos_scores, neg_scores].astype(np.float64)
    y = np.r_[np.ones(len(pos_scores), bool), np.zeros(len(neg_scores), bool)]
    w = len(pos_scores) / max(len(neg_scores), 1)
    order = np.argsort(-s, kind='stable')
    s, y = s[order], y[order]
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    tp = np.cumsum(y)[last].astype(np.float64)
    fp = (last + 1) - tp
    P = tp / np.maximum(tp + w * fp, 1e-300)
    R = tp / max(len(pos_scores), 1)
    return float(np.trapezoid(np.r_[1.0, P], np.r_[0.0, R]))


def csip_cross_character(feats, style_labels, char_labels, scale=1.0, bias=0.0, threshold=None) -> dict:
    """画风模型的跨角色协议: 正样本 = 同画师对; 困难负样本 = 同角色、不同画师的对.
    char_labels < 0 表示角色未知 (不进负样本, 也不算 "不同角色" 的正样本).
    - pr_auc_hardneg / roc_auc_hardneg: 正 = 全部同画师对, 负 = 同角色不同画师 (正负比约 100:1, PR-AUC 饱和);
    - pr_auc_balanced: 同上但负样本加权到与正样本等量 (主指标);
    - *_crosschar: 正样本只取确定是不同角色的同画师对 (只能靠画风);
    - threshold (score 单位, 一般用全部对上的最佳阈值) 下: 困难负样本误判为同画师的比例 hardneg_fpr,
      以及随机负样本 (不同画师, 不限角色) 的误判比例 randneg_fpr 作对照.
    还给出各类对的平均 score."""
    f = _normalize(feats)
    st = np.asarray(style_labels)
    ch = np.asarray(char_labels)
    pi, pj = _group_pairs(st)
    ni, nj = _group_pairs(ch)
    keep = st[ni] != st[nj]
    ni, nj = ni[keep], nj[keep]

    def sc(i, j):
        if len(i) == 0:
            return np.zeros(0)
        out = []
        for a in range(0, len(i), 1 << 22):
            out.append(((f[i[a:a + (1 << 22)]] * f[j[a:a + (1 << 22)]]).sum(-1) * scale + bias).numpy())
        return np.concatenate(out)
    ps, ns = sc(pi, pj), sc(ni, nj)
    cross = (ch[pi] >= 0) & (ch[pj] >= 0) & (ch[pi] != ch[pj])
    res = dict(n_pos_pairs=int(len(ps)), n_hardneg_pairs=int(len(ns)), n_pos_crosschar=int(cross.sum()),
               mean_score_pos=float(ps.mean()) if len(ps) else math.nan,
               mean_score_hardneg=float(ns.mean()) if len(ns) else math.nan)
    if len(ps) and len(ns):
        m = metrics_exact(np.r_[ps, ns], np.r_[np.ones(len(ps)), np.zeros(len(ns))])
        res.update(pr_auc_hardneg=m['pr_auc'], roc_auc_hardneg=m['roc_auc'], best_f1_hardneg=m['best_f1'],
                   pr_auc_balanced=balanced_pr_auc(ps, ns))
        if cross.any():
            m = metrics_exact(np.r_[ps[cross], ns], np.r_[np.ones(int(cross.sum())), np.zeros(len(ns))])
            res.update(pr_auc_crosschar=m['pr_auc'], roc_auc_crosschar=m['roc_auc'],
                       pr_auc_balanced_crosschar=balanced_pr_auc(ps[cross], ns))
        res['pos_neg_ratio'] = len(ps) / len(ns)
    if threshold is not None and len(ns):
        rng = np.random.default_rng(0)
        k = min(1_000_000, len(st) * (len(st) - 1) // 2)
        ri, rj = rng.integers(0, len(st), k), rng.integers(0, len(st), k)
        ok = st[ri] != st[rj]
        rs = sc(ri[ok], rj[ok])
        res.update(threshold=float(threshold), hardneg_fpr=float((ns >= threshold).mean()),
                   randneg_fpr=float((rs >= threshold).mean()), pos_recall=float((ps >= threshold).mean()),
                   mean_score_randneg=float(rs.mean()))
    return res
