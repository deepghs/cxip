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
from sklearn.metrics import f1_score, precision_recall_curve
from metrics_visualization import generate_all_visualizations
import os

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

def load_data(emb_dir):
    cls_map = {}
    embs = []
    labels = []
    filenames = []
    for file in sorted(Path(emb_dir).glob('*.pt')):
        emb = torch.load(file, map_location='cpu').float().numpy()
        embs.append(emb / np.linalg.norm(emb))
        cls_name = file.stem.split('_')[0]
        if cls_name not in cls_map:
            cls_map[cls_name] = len(cls_map)
        labels.append(cls_map[cls_name])
        filenames.append(file.stem)  # 保存文件名（不含.pt后缀）
    embs = np.array(embs)
    _, labels = np.unique(np.array(labels), return_inverse=True)
    return embs, labels, filenames

embs, labels, filenames = load_data('feat_fscls_zero/')
unique_labels = np.unique(labels)

logit_bias = -14.9218397
logit_scale = 29.7230533
logit_max = logit_scale + logit_bias+0.1

# pred = embs @ embs.T

from transformers import AutoImageProcessor, AutoModel
model = AutoModel.from_pretrained("deeplol/char_fscls_v1.0", trust_remote_code=True).cuda()
embs_t = torch.tensor(embs).cuda()
m, dim = embs_t.shape
x = embs_t.unsqueeze(1).expand(m, m, dim)  # [m, m, dim]
y = embs_t.unsqueeze(0).expand(m, m, dim)  # [m, m, dim]
x = torch.cat([x, y], dim=-1)
x = model.cls_model(x).squeeze(-1)
pred = x.sigmoid().detach().cpu().numpy()

logit_max = 1.0

# logit_scale = 14.51934
# logit_max = logit_scale + 0.1

# pred = embs @ embs.T
# pred = pred * logit_scale 

print(pred[:10,:10])


# >>> save FN images
image_dir = Path('/nfs/zerochan-character-w640-ws-m50-full/small_val_folder')

def save_fn_images(pred, labels, filenames, threshold, image_dir, output_dir, max_pairs=100):
    """
    Save False Negative (FN) image pairs.
    FN: samples with same label but predicted as different (score < threshold).
    """
    import shutil
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find FN pairs
    fn_pairs = []
    n = len(labels)
    for i in range(n):
        for j in range(i+1, n):
            # Same label (should be same class)
            if labels[i] == labels[j]:
                # But predicted as different (score below threshold)
                if pred[i, j] < threshold:
                    fn_pairs.append((i, j, pred[i, j]))
    
    # Sort by score (lowest first - worst FN cases)
    fn_pairs.sort(key=lambda x: x[2])
    fn_pairs = fn_pairs[:max_pairs]
    
    print(f"Found {len(fn_pairs)} FN pairs, saving top {min(len(fn_pairs), max_pairs)}...")
    
    # Save images
    for idx, (i, j, score) in enumerate(fn_pairs):
        img_name_i = filenames[i]  # 已经去掉.pt后缀
        img_name_j = filenames[j]
        
        # Try different image extensions
        for ext in ['.jpg', '.png', '.jpeg', '.webp']:
            src_path_i = image_dir / f"{img_name_i}{ext}"
            src_path_j = image_dir / f"{img_name_j}{ext}"
            
            if src_path_i.exists() and src_path_j.exists():
                # Copy images with informative names
                dst_name_i = f"FN_{idx:04d}_a_score{score:.4f}_{img_name_i}{ext}"
                dst_name_j = f"FN_{idx:04d}_b_score{score:.4f}_{img_name_j}{ext}"
                
                shutil.copy(src_path_i, output_dir / dst_name_i)
                shutil.copy(src_path_j, output_dir / dst_name_j)
                break
    
    print(f"FN images saved to {output_dir}")

# <<< save FN images



# 计算最佳阈值来判断两个embedding是否同一类
print("=" * 50)
print("开始计算最佳分类阈值...")

# 1. 构建ground truth：如果两个样本标签相同则为1，否则为0
labels_expanded = labels[:, None]  # (N, 1)
y_true = (labels_expanded == labels_expanded.T).astype(int)  # (N, N)

# 2. 将预测矩阵和ground truth展平成一维数组
# 注意：只取上三角矩阵（不包括对角线）来避免重复和自相似性
mask = np.triu(np.ones_like(pred, dtype=bool), k=1)  # 上三角矩阵，不包括对角线
y_true_flat = y_true[mask]  # 真实标签
y_pred_flat = pred[mask]   # 预测得分

print(f"总共有 {len(y_true_flat)} 个样本对")
print(f"同类样本对数量: {np.sum(y_true_flat)}")
print(f"不同类样本对数量: {len(y_true_flat) - np.sum(y_true_flat)}")

# 3. 使用新的可视化模块生成所有图表和NPY
os.makedirs('evaluation_ccip', exist_ok=True)
results = generate_all_visualizations(
    y_true=y_true_flat,
    y_scores=y_pred_flat,
    output_prefix='evaluation_fscls_zero/',
    dpi=300
)

# 使用最佳阈值保存FN图像
best_threshold = results['best_threshold']
print(f"\nUsing best threshold {best_threshold:.4f} to save FN images...")
save_fn_images(
    pred=pred,
    labels=labels,
    filenames=filenames,
    threshold=best_threshold,
    image_dir=image_dir,
    output_dir='evaluation_fscls_zero/fn_images',
    max_pairs=100
)

# D = 1 - sigmoid(pred)
D = logit_max - pred


# TSNE visualization of embedding space using precomputed distance matrix
if True:
    N=len(unique_labels)

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

    # Plot each class with different color
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1], 
                    c=[colors[label]], label=f'Class {label}', alpha=0.7, s=50)

    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.title('t-SNE Visualization with Precomputed Distance Matrix (Top N Classes)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-small')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    plt.savefig('tsne_fscls_zero.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

    print(f"t-SNE plot saved as 'tsne_embedding_visualization.png'")
    print(f"Embedding shape: {embs.shape}")
    print(f"t-SNE results shape: {tsne_results.shape}")



# print(D[:10,:10])


umap_model = umap.UMAP(
    n_neighbors=11,        # 控制局部结构保留，越小越关注局部
    min_dist=0.0,          # 越小簇越紧密，常用 0.0 ~ 0.3
    n_components=35,        # 先降到 2D 可视化，如果聚类可以先降到 10~50
    metric='cosine'        # 如果是嵌入向量，cosine 效果更好
)
X_umap = umap_model.fit_transform(embs)

# print(eps, min_samples)
# db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
# db = OPTICS(eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=64)

for min_cluster_size in range(2, 8):
    print(min_cluster_size)
    db = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, core_dist_n_jobs=64)
    y_pred = db.fit_predict(X_umap)
    y_true = np.array(labels)

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