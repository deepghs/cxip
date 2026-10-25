"""聚类评测 (伪标注用途): 降维 + HDBSCAN, 扫 min_cluster_size, 报告最佳 NMI / ARI 及簇纯度.

默认与 scripts/eval/eval_ccip.py 一致: UMAP(n_neighbors=11, n_components=35, min_dist=0, metric='cosine') +
hdbscan.HDBSCAN(min_cluster_size in 2..7). NMI 同 torchmetrics NormalizedMutualInfoScore (= sklearn arithmetic 平均),
噪声点 (-1) 整体当作一个簇 (旧脚本的做法); ARI 同 sklearn. 另给:
- purity: 非噪声点中, 每个预测簇里最多的真类占比 (按点加权), 以及 noise_frac;
- acc_ignore_noise / acc_noise_as_class: 匈牙利匹配的聚类准确率 (旧脚本 clustering_accuracy);
- nmi_noise_singleton: 噪声点各自成簇时的 NMI (不奖励"全扔进噪声").
UMAP 不可用或 reduce='pca' 时用 PCA(35) (L2 归一化后), reduce='none' 直接在 L2 归一化特征上聚类. seed 控制 UMAP / PCA.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def reduce_features(feats, method='umap', n_components=35, n_neighbors=11, min_dist=0.0, seed=0, n_jobs=None):
    X = np.asarray(feats, dtype=np.float32)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    if method == 'umap':
        try:
            import umap
        except ImportError:
            method = 'pca'
        else:
            kw = dict(n_neighbors=n_neighbors, n_components=n_components, min_dist=min_dist, metric='cosine')
            if seed is not None:
                kw['random_state'] = seed
            if n_jobs is not None:
                kw['n_jobs'] = n_jobs
            return umap.UMAP(**kw).fit_transform(X), 'umap'
    if method == 'pca':
        from sklearn.decomposition import PCA
        k = min(n_components, X.shape[1], len(X))
        return PCA(k, random_state=seed).fit_transform(X), 'pca'
    if method == 'none':
        return X, 'none'
    raise ValueError(method)


def hdbscan_labels(X, min_cluster_size, n_jobs=32):
    try:
        import hdbscan
        return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, core_dist_n_jobs=n_jobs).fit_predict(X)
    except ImportError:
        from sklearn.cluster import HDBSCAN
        return HDBSCAN(min_cluster_size=min_cluster_size, n_jobs=n_jobs).fit_predict(X)


def clustering_accuracy(y_true, y_pred, ignore_noise=True):
    """旧脚本的匈牙利匹配准确率 (稀疏重编号后再建表, 结果相同)"""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if ignore_noise:
        m = y_pred != -1
        y_true, y_pred = y_true[m], y_pred[m]
        if len(y_pred) == 0:
            return 0.0
    elif np.any(y_pred == -1):
        y_pred = y_pred.copy()
        y_pred[y_pred == -1] = y_pred.max() + 1
    _, t = np.unique(y_true, return_inverse=True)
    _, p = np.unique(y_pred, return_inverse=True)
    D = max(p.max(), t.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    np.add.at(w, (p, t), 1)
    r, c = linear_sum_assignment(w.max() - w)
    return float(w[r, c].sum() / len(y_pred))


def cluster_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    noise = y_pred == -1
    out = dict(n_clusters=int(len(np.unique(y_pred[~noise]))), noise_frac=float(noise.mean()),
               nmi=float(normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')),
               ari=float(adjusted_rand_score(y_true, y_pred)))
    single = y_pred.copy()
    single[noise] = y_pred.max() + 1 + np.arange(noise.sum())
    out['nmi_noise_singleton'] = float(normalized_mutual_info_score(y_true, single, average_method='arithmetic'))
    if (~noise).any():
        yt, yp = y_true[~noise], y_pred[~noise]
        _, p = np.unique(yp, return_inverse=True)
        _, t = np.unique(yt, return_inverse=True)
        tab = np.zeros((p.max() + 1, t.max() + 1), dtype=np.int64)
        np.add.at(tab, (p, t), 1)
        out['purity'] = float(tab.max(1).sum() / len(yp))
        out['inverse_purity'] = float(tab.max(0).sum() / len(yp))   # 每个真类被最大的那个簇覆盖的比例
    else:
        out['purity'] = out['inverse_purity'] = 0.0
    out['acc_ignore_noise'] = clustering_accuracy(y_true, y_pred, True)
    out['acc_noise_as_class'] = clustering_accuracy(y_true, y_pred, False)
    return out


def clustering_eval(feats, labels, reduce='umap', mcs_range=range(2, 8), seed=0, n_components=35, n_neighbors=11,
                    min_dist=0.0, select='nmi', n_jobs=32) -> dict:
    """扫 min_cluster_size, 返回每个 mcs 的指标和按 select (nmi) 选出的最佳"""
    t0 = time.time()
    X, used = reduce_features(feats, reduce, n_components, n_neighbors, min_dist, seed)
    t_reduce = time.time() - t0
    per = {}
    for mcs in mcs_range:
        per[int(mcs)] = cluster_metrics(labels, hdbscan_labels(X, int(mcs), n_jobs))
    best_mcs = max(per, key=lambda k: per[k][select])
    out = dict(reduce=used, n_components=int(X.shape[1]), seed=seed, n_images=int(len(labels)),
               n_classes=int(len(np.unique(labels))), best_mcs=best_mcs, select=select,
               **{k: v for k, v in per[best_mcs].items()},
               best_nmi=max(v['nmi'] for v in per.values()), best_ari=max(v['ari'] for v in per.values()),
               per_mcs={str(k): v for k, v in per.items()}, seconds_reduce=t_reduce, seconds=time.time() - t0)
    return out
