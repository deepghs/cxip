import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from pathlib import Path
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN, OPTICS
import hdbscan
import umap
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
from sklearn.metrics import accuracy_score, precision_recall_curve

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
df_topN = df[df['ids'] < N].reset_index(drop=True)
print(df_topN.shape, df_topN.iloc[0]["name"])

embs = np.empty((df_topN.shape[0], 768))
for i, name in enumerate(tqdm(df_topN["name"])):
    file = emb_dir / (name+".pt")
    emb = torch.load(file, map_location='cpu').numpy()
    embs[i] = emb / np.linalg.norm(emb)

logit_bias = -8.2377
logit_scale = 14.51934
logit_max = logit_scale + logit_bias+0.1


anchors = np.empty((N, 768))

for cls in range(N):
    # 获取满足条件的最小序号
    print(df_topN[df_topN['ids'] == cls])
    matched_indices = df_topN[df_topN['ids'] == cls].index
    if len(matched_indices) > 0:
        min_index = matched_indices[1]  # 获取第一个（最小的）索引
        print(f"类别 {cls} 的最小序号: {min_index}")
    print(df_topN.iloc[min_index])
    anchors[cls] = embs[min_index]

pred = embs @ anchors.T

y_pred = np.argmax(pred, axis=1)
y_true = df_topN['ids']

print(y_pred[:20])
print(y_true[:20])

print(accuracy_score(y_true, y_pred))

