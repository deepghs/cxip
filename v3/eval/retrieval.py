"""检索指标: 每张图作查询, 其余全部图作库 (排除自身), cos 排序.
- mAP@R (Musgrave et al. 2020 "A Metric Learning Reality Check"): R = 库中同类图数, AP@R = (1/R) Σ_{i<=R} P(i)·rel(i);
- R@1 (= precision@1), 以及 R@k (前 k 个里至少一个同类) 可选.
只有 1 张图的类不作查询 (R=0). 分块在 CPU 上做 (float32 矩阵乘 + topk), 2.5 万张约数秒.
"""
from __future__ import annotations

import numpy as np
import torch


def retrieval_eval(feats, labels, ks=(1, 5), chunk=1024) -> dict:
    f = torch.nn.functional.normalize(torch.as_tensor(np.asarray(feats)).float(), dim=-1)
    labels = np.asarray(labels)
    _, lab = np.unique(labels, return_inverse=True)
    lab_t = torch.as_tensor(lab)
    counts = np.bincount(lab)
    R_all = counts[lab] - 1
    q_ok = R_all > 0
    kmax = int(max(R_all.max(), max(ks)))
    kmax = min(kmax, len(f) - 1)
    ap_sum, n_q = 0.0, 0
    hit = {k: 0 for k in ks}
    with torch.no_grad():
        for r0 in range(0, len(f), chunk):
            r1 = min(len(f), r0 + chunk)
            idx = np.nonzero(q_ok[r0:r1])[0] + r0
            if len(idx) == 0:
                continue
            it = torch.as_tensor(idx)
            s = f[it] @ f.T
            s[torch.arange(len(idx)), it] = -float('inf')
            # 只取到本块最大的 R 和 max(ks)
            kk = int(min(max(R_all[idx].max(), max(ks)), len(f) - 1))
            top = s.topk(kk, dim=1).indices
            rel = (lab_t[top] == lab_t[it][:, None]).float()          # (q, kk)
            R = torch.as_tensor(R_all[idx]).float()
            pos = torch.arange(1, kk + 1).float()
            mask = pos[None] <= R[:, None]
            prec = rel.cumsum(1) / pos[None]
            ap = (prec * rel * mask).sum(1) / R
            ap_sum += float(ap.sum())
            n_q += len(idx)
            for k in ks:
                hit[k] += int((rel[:, :k].sum(1) > 0).sum())
    out = dict(map_at_r=ap_sum / max(n_q, 1), n_queries=int(n_q), n_images=int(len(f)),
               n_classes=int(len(counts)))
    for k in ks:
        out[f'recall_at_{k}'] = hit[k] / max(n_q, 1)
    return out


def _codes(x):
    return np.unique(np.asarray(x), return_inverse=True)[1].reshape(-1)


def window_retrieval(feats, labels, t_year, family, window=1.0, query=None, gallery=None, chunk=1024) -> dict:
    """时间窗检索 (benchmark v2 零样本检索口径), 返回逐查询数组 (长度 N, 非查询 / 无相关项为 NaN).
    查询 i 的库 = gallery 中除自身、同 family、以及 "同画师且 |Δt| >= window" 之外的全部图;
    相关项 = 同画师且 |Δt| < window (window=None: 同画师任意时期都相关). R = 该查询的相关项数,
    map_at_r 的定义同 retrieval_eval (前 R 个里 precision@i 在相关命中处求和再除以 R); r1 = 首位是否相关.
    """
    f = torch.nn.functional.normalize(torch.as_tensor(np.asarray(feats)).float(), dim=-1)
    n = len(f)
    lab = torch.as_tensor(_codes(labels))
    fam = torch.as_tensor(_codes(family))
    t = torch.as_tensor(np.asarray(t_year, dtype=np.float64))
    q_mask = np.ones(n, bool) if query is None else np.asarray(query, bool)
    g_mask = np.ones(n, bool) if gallery is None else np.asarray(gallery, bool)
    g_idx = torch.as_tensor(np.nonzero(g_mask)[0])
    fg, lab_g, fam_g, t_g = f[g_idx], lab[g_idx], fam[g_idx], t[g_idx]
    G = len(g_idx)
    ap_out = np.full(n, np.nan)
    r1_out = np.full(n, np.nan)
    with torch.no_grad():
        for r0 in range(0, n, chunk):
            qi = np.nonzero(q_mask[r0:min(n, r0 + chunk)])[0] + r0
            if len(qi) == 0 or G == 0:
                continue
            it = torch.as_tensor(qi)
            same = lab[it][:, None] == lab_g[None]                                    # (q, G)
            near = same if window is None else same & ((t[it][:, None] - t_g[None]).abs() < window)
            valid = (fam[it][:, None] != fam_g[None]) & (~same | near)               # 同 family (含自身) 与窗外同画师被移出库
            rel_m = near & valid
            R = rel_m.sum(1)
            ok = R > 0
            if not ok.any():
                continue
            s = f[it] @ fg.T
            s = s.masked_fill(~valid, -float('inf'))
            kk = int(min(R.max().item(), G))
            top = s.topk(kk, dim=1).indices
            rel = torch.gather(rel_m, 1, top).float()                                 # (q, kk)
            Rf = R.clamp(min=1).float()
            pos = torch.arange(1, kk + 1).float()
            prec = rel.cumsum(1) / pos[None]
            ap = (prec * rel * (pos[None] <= Rf[:, None])).sum(1) / Rf
            ok_np = ok.numpy()
            ap_out[qi[ok_np]] = ap.numpy()[ok_np]
            r1_out[qi[ok_np]] = rel[:, 0].numpy()[ok_np]
    return dict(map_at_r=ap_out, r1=r1_out)
