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
import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def reduce_features(feats, method='umap', n_components=35, n_neighbors=11, min_dist=0.0, seed=0, n_jobs=None):
    X = np.asarray(feats, dtype=np.float32)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    if method is None:      # reduce=None: 直接在 L2 归一化特征上 HDBSCAN
        method = 'none'
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
                    min_dist=0.0, select='nmi', n_jobs=32, return_pred=False) -> dict:
    """扫 min_cluster_size, 返回每个 mcs 的指标和按 select (nmi) 选出的最佳. return_pred=True 时多给最佳 mcs 的预测标签 pred_labels"""
    t0 = time.time()
    X, used = reduce_features(feats, reduce, n_components, n_neighbors, min_dist, seed)
    t_reduce = time.time() - t0
    per, preds = {}, {}
    for mcs in mcs_range:
        preds[int(mcs)] = hdbscan_labels(X, int(mcs), n_jobs)
        per[int(mcs)] = cluster_metrics(labels, preds[int(mcs)])
    best_mcs = max(per, key=lambda k: per[k][select])
    out = dict(reduce=used, n_components=int(X.shape[1]), seed=seed, n_images=int(len(labels)),
               n_classes=int(len(np.unique(labels))), best_mcs=best_mcs, select=select,
               **{k: v for k, v in per[best_mcs].items()},
               best_nmi=max(v['nmi'] for v in per.values()), best_ari=max(v['ari'] for v in per.values()),
               per_mcs={str(k): v for k, v in per.items()}, seconds_reduce=t_reduce, seconds=time.time() - t0)
    if return_pred:
        out['pred_labels'] = preds[best_mcs]
    return out


def _singletonize(y):
    """-1 (噪声 / 干扰图) 各自成单例类"""
    y = np.asarray(y).copy()
    m = y == -1
    y[m] = y.max() + 1 + np.arange(int(m.sum()))
    return y


def bcubed(y_true, y_pred) -> dict:
    """BCubed precision / recall / F. 预测噪声 (-1) 视为各自独立的单例簇; y_true 为 -1 的干扰图也是单例真类."""
    yt, yp = _singletonize(y_true), _singletonize(y_pred)
    if len(yt) == 0:
        return dict(precision=0.0, recall=0.0, f=0.0)
    _, t = np.unique(yt, return_inverse=True)
    _, p = np.unique(yp, return_inverse=True)
    t, p = t.reshape(-1), p.reshape(-1)
    pair = p.astype(np.int64) * (t.max() + 1) + t
    _, pair_inv, pair_cnt = np.unique(pair, return_inverse=True, return_counts=True)
    both = pair_cnt[pair_inv.reshape(-1)]                  # 同预测簇且同真类的点数 (含自身)
    prec = float(np.mean(both / np.bincount(p)[p]))
    rec = float(np.mean(both / np.bincount(t)[t]))
    f = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return dict(precision=prec, recall=rec, f=f)


def mixed_pool_clustering(feats, labels, n_pools=5, n_artists=100, max_per_artist=40, n_distractors=1000, seed=0,
                          mcs_range=range(2, 8), reduces=('umap', None), n_jobs=32, **eval_kw) -> dict:
    """混合池聚类 (benchmark v2): 每个池抽 n_artists 个画师 (每人至多 max_per_artist 张) 作真类, 再从其余画师抽
    n_distractors 张图作干扰 (真标签 -1, 即单例). 小集合时 n_artists 钳到 min(n_artists, A//2), 干扰从其余画师取.
    每个 reduce 模式 ('umap' / None=直接 HDBSCAN) 调 clustering_eval 扫 mcs, 用 nmi 选最佳 mcs, 报该 mcs 下
    bcubed_f / ari 在各池的均值与标准差 (np.std, ddof=0; 聚类不做 bootstrap, 用池间标准差表示波动).
    nmi / ari 的选择与计算里, 干扰图各自编独立单例标签 (与 bcubed 对 -1 的处理一致).
    返回 {mode_key: {...}, n_pools, n_artists, n_distractors, ...}; mode_key = 'umap' / 'none'.
    """
    labels = np.asarray(labels)
    arts = np.unique(labels)
    na = int(min(n_artists, len(arts) // 2))
    if na < 1:
        raise ValueError('need at least 2 artists')
    modes = {('none' if r is None else r): r for r in reduces}
    acc = {k: dict(bcubed_f=[], ari=[], nmi=[], best_mcs=[], noise_frac=[]) for k in modes}
    n_img, n_dis = [], []
    for pool in range(n_pools):
        rng = np.random.default_rng([seed, pool])
        pick = rng.permutation(arts)[:na]
        rest = np.isin(labels, pick, invert=True)
        idx = []
        for a in pick:
            ia = np.nonzero(labels == a)[0]
            idx.append(rng.permutation(ia)[:max_per_artist])
        idx = np.concatenate(idx)
        di = rng.permutation(np.nonzero(rest)[0])[:n_distractors]
        sel = np.concatenate([idx, di])
        _, y = np.unique(labels[idx], return_inverse=True)
        y_true = np.concatenate([y.reshape(-1), np.full(len(di), -1)])
        y_eval = _singletonize(y_true)
        n_img.append(len(sel)); n_dis.append(len(di))
        for key, r in modes.items():
            with warnings.catch_warnings():     # 单例标签多时 sklearn 会提示 "类数超过样本数一半"
                warnings.filterwarnings('ignore', message='The number of unique classes is greater')
                e = clustering_eval(np.asarray(feats)[sel], y_eval, reduce=r, mcs_range=mcs_range, seed=seed,
                                    n_jobs=n_jobs, return_pred=True, **eval_kw)
            a = acc[key]
            a['bcubed_f'].append(bcubed(y_true, e['pred_labels'])['f'])
            a['ari'].append(e['ari']); a['nmi'].append(e['nmi'])
            a['best_mcs'].append(e['best_mcs']); a['noise_frac'].append(e['noise_frac'])
            a['reduce'] = e['reduce']
    out = dict(n_pools=int(n_pools), n_artists=na, n_distractors=int(min(n_dis)) if n_dis else 0,
               n_images=[int(x) for x in n_img], seed=seed)
    for key, a in acc.items():
        out[key] = dict(a, **{f'{m}_{s}': float(fn(a[m])) for m in ('bcubed_f', 'ari', 'nmi')
                              for s, fn in (('mean', np.mean), ('std', np.std))})
    return out
