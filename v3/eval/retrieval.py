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
