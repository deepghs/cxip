import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from pathlib import Path
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN, OPTICS
import hdbscan
import json
import pandas as pd
from tqdm import tqdm
from torchmetrics.clustering import NormalizedMutualInfoScore, AdjustedRandScore
from sklearn.metrics import adjusted_rand_score
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environment
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import f1_score, precision_recall_curve

def clustering_accuracy(y_true, y_pred, ignore_noise=True):
    """
    Clustering Accuracy (ACC) with Hungarian algorithm.
    Specially handles DBSCAN noise (-1).
    
    Args:
        y_true: array-like, shape (n_samples,)
        y_pred: array-like, shape (n_samples,)
        ignore_noise: bool, whether to ignore noise points (-1 in y_pred)
    
    Returns:
        acc: float
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_pred.size == y_true.size
    
    if ignore_noise:
        mask = y_pred != -1
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    
    # shift -1 labels to a new cluster if not ignoring noise
    if not ignore_noise and np.any(y_pred == -1):
        y_pred = y_pred.copy()
        y_pred[y_pred == -1] = y_pred.max() + 1
    
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size

def find_best_f1_threshold(y_true, y_probas):
    """
    通过遍历所有可能的阈值，找到使F1分数最大化的最佳阈值。

    参数:
    y_true (array-like): 真实的标签 (0 或 1)。
    y_probas (array-like): 模型预测为正类 (1) 的概率。

    返回:
    tuple: (最佳阈值, 对应的最大F1分数)
    """
    # 使用precision_recall_curve可以更高效地获取所有阈值和对应的precision, recall
    # precision_recall_curve返回的阈值数量比precision和recall少一个
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probas)

    # 计算每个阈值下的F1分数
    # 我们需要忽略最后一个precision和recall值，因为它们对应没有阈值的情况
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    f1_scores = f1_scores[:-1]  # 移除最后一个元素
    
    # 防止除以零的情况
    # f1_scores = np.divide(2 * precisions[:-1] * recalls[:-1], precisions[:-1] + recalls[:-1], 
    #                       out=np.zeros_like(precisions[:-1]), where=(precisions[:-1] + recalls[:-1]) != 0)


    # 找到最大F1分数对应的索引
    best_f1_idx = np.argmax(f1_scores)

    # 获取最佳阈值和最大F1分数
    best_threshold = thresholds[best_f1_idx]
    max_f1_score = f1_scores[best_f1_idx]

    return best_threshold, max_f1_score, thresholds, f1_scores

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

emb_dir = Path('feat/')
label_file="/nfs/danbooru-w640-ws-full-artist/test_split.json"
with open(label_file, 'r') as f:
    label_dict = json.load(f)

df = pd.DataFrame(list(label_dict.items()), columns=["name", "label"])
print(df.shape)

N = 50

cats = pd.Categorical(df[df.columns[-1]])
df['ids'] = cats.codes  # 得到类别id
df_topN = df[df['ids'] < N]
print(df_topN.shape, df_topN.iloc[0]["name"])

embs = np.empty((df_topN.shape[0], 768))
for i, name in enumerate(tqdm(df_topN["name"])):
    file = emb_dir / (name+".pt")
    emb = torch.load(file, map_location='cpu').numpy()
    embs[i] = emb / np.linalg.norm(emb)

logit_bias = -8.2377
logit_scale = 14.51934
logit_max = logit_scale + logit_bias+0.1

pred = embs @ embs.T
pred = pred * logit_scale + logit_bias


y_true = np.array(df_topN['ids'])
y_true = (y_true[:, None] == y_true[None, :]).astype(int).reshape(-1)
y_pred = pred.reshape(-1)

# 2. 找到最佳阈值
best_thresh, max_f1, all_thresholds, all_f1_scores = find_best_f1_threshold(y_true, y_pred)

print(f"找到的最佳阈值是: {best_thresh:.4f}")
print(f"在此阈值下的最大F1分数是: {max_f1:.4f}")

# 3. 使用找到的最佳阈值进行预测
y_pred_best = (y_pred >= best_thresh).astype(int)
f1_best_manual = f1_score(y_true, y_pred_best)
print(f"使用最佳阈值 {best_thresh:.4f} 应用于原始分数，得到的F1分数是: {f1_best_manual:.4f}")

# 4. 可视化F1分数随阈值的变化
plt.figure(figsize=(10, 6))
# 注意：thresholds 数组的长度是 f1_scores 长度 + 1
plt.plot(all_thresholds, all_f1_scores, label='F1 Score')
plt.scatter(best_thresh, max_f1, color='red', zorder=5, label=f'Best Threshold ({best_thresh:.2f})')
plt.xlabel("Threshold (on original scores)")
plt.ylabel("F1 Score")
plt.title("F1 Score vs. Threshold on Raw Scores")
plt.legend()
plt.grid(True)
plt.savefig('f1_scores.png', dpi=300, bbox_inches='tight')
plt.close()  # Close the figure to free memory



# D = 1 - sigmoid(pred)
D = logit_max - pred

# TSNE visualization of embedding space using precomputed distance matrix
if True:
    print("Running t-SNE visualization with precomputed distance matrix...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, 
                metric='precomputed', init='random')
    tsne_results = tsne.fit_transform(D)

    # Create visualization
    plt.figure(figsize=(12, 8))

    # Generate colors for any number of classes
    if N <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:N]
    elif N <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[:N]
    else:
        # For more than 20 classes, use hsv colormap to generate distinct colors
        colors = plt.cm.hsv(np.linspace(0, 1, N))

    labels = df_topN['ids'].values
    unique_labels = np.unique(labels)

    # Plot each class with different color
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1], 
                    c=[colors[label]], label=f'Class {label}', alpha=0.7, s=50)

    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.title('t-SNE Visualization with Precomputed Distance Matrix (Top N Classes)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    plt.savefig('tsne_embedding_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

    print(f"t-SNE plot saved as 'tsne_embedding_visualization.png'")
    print(f"Embedding shape: {embs.shape}")
    print(f"t-SNE results shape: {tsne_results.shape}")

0/0

# print(D[:10,:10])


# print(eps, min_samples)
# db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
# db = OPTICS(eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=64)
db = hdbscan.HDBSCAN(min_cluster_size=20, core_dist_n_jobs=64)
y_pred = db.fit_predict(D)
y_true = np.array(df_topN['ids'])

n_clusters = len(set(y_pred))
print("N clusters:", n_clusters)

print("ACC (ignore noise):", clustering_accuracy(y_true, y_pred, ignore_noise=True))
print("ACC (treat noise as class):", clustering_accuracy(y_true, y_pred, ignore_noise=False))

print(y_pred[:20])
print(y_true[:20])

nmi = NormalizedMutualInfoScore()(torch.tensor(y_pred), torch.tensor(y_true))
print("NMI:", nmi.item())

ari = adjusted_rand_score(y_pred, y_true)
print("ARI:", ari)