"""README 聚类可视化: 测试集随机多类 (CCIP 200 角色 / CSIP 100 画师) 的 2D t-SNE, 其中 24 类着色, 其余作灰色背景。

只取 24 类时, 15-NN 同类率 ≈ 99%, kNN 图近乎断开, UMAP 会把每类压成极小一团并随意推远 (画面稀疏, 新旧看不出差别);
类多了邻域才会相互竞争, 放错位置的图才看得到。
结果: /data/neko/cxip_v3/release/_analysis/tsne_v1.npz  ({task}/y, {task}/hl, {task}/{model})
用法: python -m v3.scripts.release_tsne
"""
import numpy as np
from sklearn.manifold import TSNE

from v3.eval.runner import load_set
from v3.scripts.release_analysis import OUT, load_feats, models

# (task, 测试集, 对比的模型, 类数, 每类最多张数)
SPEC = [('ccip', 'openset_full', ('AnimeCCIP', 'AnimeCCIP-v3'), 200, 40),
        ('csip', 'test100_0918', ('AnimeCSIP-v2.1', 'AnimeCSIP-v3'), 100, 60)]


def main():
    res = {}
    for task, s, keep, ncls, cap in SPEC:
        df, _ = load_set(s)
        y = df['label'].values
        rng = np.random.default_rng(1)
        cls, cnt = np.unique(y, return_counts=True)
        pool = cls[cnt >= 20]
        pick = rng.choice(pool, min(ncls, len(pool)), replace=False)
        sub = np.concatenate([rng.permutation(np.nonzero(y == c)[0])[:cap] for c in pick])
        res[f'{task}/y'] = np.searchsorted(np.sort(pick), y[sub])
        res[f'{task}/hl'] = np.searchsorted(np.sort(pick), pick[:24])
        for name, by_set, erase in models(task):
            if name not in keep:
                continue
            f = load_feats(by_set[s][0], s, df, erase)[sub].numpy()
            res[f'{task}/{name}'] = TSNE(2, perplexity=30, metric='cosine', init='pca', random_state=0,
                                         n_jobs=64).fit_transform(f).astype(np.float32)
            print(task, name, len(sub), flush=True)
    np.savez(OUT / 'tsne_v1.npz', **res)


if __name__ == '__main__':
    main()
