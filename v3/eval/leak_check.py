"""开放集划分的泄漏复核.

- name_leak_report(): 不需要特征. 剔除类的名字 (去掉限定词后的 norm_name) 与留在训练里的类重名的情况
  (同名不同作品是正常的, 同名同作品说明合并漏了 —— 这类剔除类应该视为 "见过");
- class_leak_report(): 给定特征 (任意模型, 推荐 HF AnimeCCIP 或骨干), 计算每个剔除类中心与全部训练类中心的最大 cos,
  列出最近的训练类; 高于阈值的说明训练集里很可能有同一个角色 (合并漏掉的别名 / 不同语言的名字).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def class_centroids(feats, labels):
    f = torch.nn.functional.normalize(torch.as_tensor(np.asarray(feats)).float(), dim=-1)
    labels = np.asarray(labels)
    cls, inv = np.unique(labels, return_inverse=True)
    C = torch.zeros(len(cls), f.shape[1])
    C.index_add_(0, torch.as_tensor(inv), f)
    return cls, torch.nn.functional.normalize(C, dim=-1)


def class_leak_report(heldout_feats, heldout_labels, train_feats, train_labels, names=None, threshold=0.85,
                      top=3) -> pd.DataFrame:
    """每个剔除类: 与训练类中心的最大 cos (以及 top 个最近训练类), 和自身类内平均 cos (参考尺度).
    names: {label: 名字} 便于人工查看. 返回按 max_cos 降序的表, 列 leak = max_cos >= threshold"""
    hc, H = class_centroids(heldout_feats, heldout_labels)
    tc, T = class_centroids(train_feats, train_labels)
    S = H @ T.T
    val, idx = S.topk(min(top, len(tc)), dim=1)
    hf = torch.nn.functional.normalize(torch.as_tensor(np.asarray(heldout_feats)).float(), dim=-1)
    hl = np.asarray(heldout_labels)
    intra = []
    for i, c in enumerate(hc):
        g = hf[torch.as_tensor(np.nonzero(hl == c)[0])]
        intra.append(float((g @ H[i]).mean()))
    rows = []
    for i, c in enumerate(hc):
        r = dict(label=c, max_cos=float(val[i, 0]), nearest=[tc[j] for j in idx[i].tolist()],
                 nearest_cos=[round(float(v), 4) for v in val[i]], intra_to_centroid=intra[i])
        if names:
            r['name'] = names.get(c)
            r['nearest_names'] = [names.get(tc[j]) for j in idx[i].tolist()]
        rows.append(r)
    df = pd.DataFrame(rows).sort_values('max_cos', ascending=False).reset_index(drop=True)
    df['leak'] = df['max_cos'] >= threshold
    return df


def name_leak_report(class_meta: pd.DataFrame, heldout_labels) -> pd.DataFrame:
    """class_meta: ccip_openset_v1/class_meta.parquet. 返回剔除类中与训练类同名 (norm_name; FGO 式 "职阶 (真名)" 用真名)
    的行. same_franchise: 同名且 franchise 有交集 (疑似合并漏掉的同一角色); dup_train_labels: 同名且 franchise 有交集
    或任一方没有 franchise 信息的训练类 (保守起见建议一并从训练删掉, build_ccip_openset 里作为 name_dup 剔除)"""
    from .ccip_meta import GENERIC_NAMES, norm_key, norm_name, split_quals

    def key(name, series, series_keys):
        base, quals = split_quals(name)
        base = norm_name(base)
        if base in GENERIC_NAMES:
            # FGO 式 "职阶 (真名)" 用真名; "职阶 (作品名)" (如 Rider (Fate/stay night)) 不是名字, 不参与
            if not quals:
                return ''
            q = quals[0]
            known = {norm_key(series or '')} | {norm_key(k) for k in (series_keys if series_keys is not None else [])}
            if norm_key(q) in known or 'fate' in norm_key(q):
                return ''
            base = norm_name(q)
        return base
    held = set(int(x) for x in heldout_labels)
    m = class_meta.copy()
    m['base'] = [key(n, s, k) for n, s, k in zip(m['name'], m['series'], m['series_keys'])]
    tr = m[~m['label'].isin(held)]
    by = {}
    for lab, b, fr in zip(tr['label'], tr['base'], tr['franchises']):
        by.setdefault(b, []).append((lab, set(fr) if fr is not None else set()))
    rows = []
    for lab, name, b, fr in zip(m['label'], m['name'], m['base'], m['franchises']):
        if lab not in held or not b or b not in by:
            continue
        frs = set(fr) if fr is not None else set()
        hits = by[b]
        same = [h[0] for h in hits if frs & h[1]]
        unk = [h[0] for h in hits if not (frs and h[1])]
        rows.append(dict(label=lab, name=name, train_labels=[h[0] for h in hits], same_franchise=bool(same),
                         dup_train_labels=same + unk))
    return pd.DataFrame(rows, columns=['label', 'name', 'train_labels', 'same_franchise', 'dup_train_labels'])
