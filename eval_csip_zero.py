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
from sklearn.metrics import accuracy_score, precision_recall_curve, precision_score, recall_score, f1_score, classification_report
import argparse

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

def calculate_top_k_accuracy(y_true, y_pred_scores, k=5):
    """
    计算Top-K准确率
    sklearn没有直接提供top-k accuracy，所以需要自己实现
    """
    if k > y_pred_scores.shape[1]:
        k = y_pred_scores.shape[1]
    
    # 更高效的向量化实现
    top_k_predictions = np.argsort(y_pred_scores, axis=1)[:, -k:]
    
    # 向量化检查：每行检查真实标签是否在top-k中
    y_true = np.array(y_true)
    correct = np.any(top_k_predictions == y_true[:, np.newaxis], axis=1)
    
    return correct.mean()

def softmax(logits, axis=-1):
    # logits -> 概率
    e = np.exp(logits - np.max(logits, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

def normalized_entropy(logits, axis=-1, eps=1e-12, base=None):
    """
    计算归一化熵
    参数:
        logits: ndarray, 模型输出 (未归一化)
        axis:   沿哪个维度计算熵
        eps:    防止log(0)
        base:   对数底数, None=自然对数, 2=比特
    """
    probs = softmax(logits, axis=axis)
    probs = np.clip(probs, eps, 1.0)
    log_fn = np.log if base is None else (lambda x: np.log(x) / np.log(base))

    H = -np.sum(probs * log_fn(probs), axis=axis)  # 信息熵
    K = logits.shape[axis]                         # 类别数
    H_max = log_fn(K)                              # 最大熵
    return H / H_max


parser = argparse.ArgumentParser()
parser.add_argument('--emb_dir', type=str, default='feat/')
parser.add_argument('--N', type=int, default=60)
parser.add_argument('--seen', type=int, default=None)
parser.add_argument('--ood_thr', type=float, default=-1.3)
parser.add_argument('--A', type=int, default=1, help='每个类别使用的anchor数量')
args = parser.parse_args()

emb_dir = Path('feat/')
label_file="/nfs/danbooru-w640-ws-full-artist/test_split.json"
with open(label_file, 'r') as f:
    label_dict = json.load(f)

df = pd.DataFrame(list(label_dict.items()), columns=["name", "label"])
print(df.shape)

logit_bias = -8.2377
logit_scale = 14.51934
logit_max = logit_scale + logit_bias+0.1


def compute_metrics_with_random_anchors(embs_topN, labels_topN, seen, A=1):
    """计算使用随机选择anchor的各种指标
    
    Args:
        A: 每个类别使用的anchor数量
    """
    # 向量化收集anchor和类别信息
    anchor_indices_list = []  # 存储所有选中的anchor索引
    anchor_class_mapping = []  # 记录每个anchor属于哪个类别
    
    # 预计算每个类别的样本索引
    class_indices = [np.where(labels_topN == cls)[0] for cls in range(seen)]
    
    # 为每个类别选择anchor
    for cls, matched_indices in enumerate(class_indices):
        if len(matched_indices) > 0:
            # 随机选择A个索引（如果样本数不足A个，则全部选择）
            num_anchors = min(A, len(matched_indices))
            random_indices = np.random.choice(matched_indices, size=num_anchors, replace=False)
            
            # 批量添加索引和类别映射
            anchor_indices_list.extend(random_indices)
            anchor_class_mapping.extend([cls] * num_anchors)
        else:
            print(f"警告：类别 {cls} 没有匹配的数据")
            # 如果没有匹配数据，添加一个特殊标记索引（用于后续处理）
            anchor_indices_list.append(-1)  # 使用-1表示无效索引
            anchor_class_mapping.append(cls)
    
    # 向量化构建anchor矩阵
    anchor_indices_array = np.array(anchor_indices_list)
    anchor_class_mapping = np.array(anchor_class_mapping)
    
    # 处理有效的anchor索引
    valid_mask = anchor_indices_array >= 0
    all_anchors = np.zeros((len(anchor_indices_array), embs_topN.shape[1]))  # shape: (total_anchors, 768)
    
    # 使用向量化索引获取有效的anchor
    if np.any(valid_mask):
        all_anchors[valid_mask] = embs_topN[anchor_indices_array[valid_mask]]
    
    # 计算所有embedding与所有anchor的相似度分数
    similarity_scores = embs_topN @ all_anchors.T  # shape: (num_samples, total_anchors)
    
    # 向量化计算每个类别的平均分数
    # 创建one-hot编码矩阵：每行表示一个anchor，每列表示一个类别
    one_hot_matrix = np.zeros((len(all_anchors), seen))  # shape: (total_anchors, seen)
    one_hot_matrix[np.arange(len(anchor_class_mapping)), anchor_class_mapping] = 1
    
    # 计算每个类别的分数和: (num_samples, total_anchors) @ (total_anchors, seen) = (num_samples, seen)
    class_score_sums = similarity_scores @ one_hot_matrix
    
    # 计算每个类别的anchor数量: shape (seen,)
    anchor_counts = one_hot_matrix.sum(axis=0)
    
    # 避免除以0，将anchor数量为0的类别设置为1（对应的分数和也是0）
    anchor_counts = np.maximum(anchor_counts, 1)
    
    # 计算平均分数: (num_samples, seen) / (seen,) = (num_samples, seen)
    pred_scores = class_score_sums / anchor_counts[np.newaxis, :]
    
    # 应用logit scale和bias
    pred_scores = pred_scores * logit_scale + logit_bias
    scores = np.max(pred_scores, axis=1)

    y_pred = np.argmax(pred_scores, axis=1)
    ood = scores < args.ood_thr
    y_pred[ood] = seen
    pred_scores = np.hstack([pred_scores, ood[:, None]*logit_scale + logit_bias])

    labels_seen = np.clip(labels_topN, None, seen)
    y_true = labels_seen
    
    # 计算Top-1和Top-5准确率
    top1_accuracy = accuracy_score(y_true, y_pred)
    top5_accuracy = calculate_top_k_accuracy(y_true, pred_scores, k=5)
    
    # 计算其他指标（基于top1预测）
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        'top1_accuracy': top1_accuracy,
        'top5_accuracy': top5_accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted
    }

def main(embs, labels, N, A, seen):
    embs_topN = embs[labels < N]
    labels_topN = labels[labels < N]

    # 设置随机种子以确保可重复性
    np.random.seed(42)

    # 运行20次实验并计算各种指标的平均值
    num_experiments = 20
    all_results = []

    print(f"开始进行20次随机anchor选择实验 (每个类别使用{A}个anchor)...")
    for i in tqdm(range(num_experiments), desc="计算各种指标"):
        metrics = compute_metrics_with_random_anchors(embs_topN, labels_topN, seen, A=A)
        all_results.append(metrics)
        print(f"实验 {i+1}: Top1={metrics['top1_accuracy']:.4f}, Top5={metrics['top5_accuracy']:.4f}, "
              f"P_macro={metrics['precision_macro']:.4f}, R_macro={metrics['recall_macro']:.4f}, "
              f"F1_macro={metrics['f1_macro']:.4f}")

    # 将结果转换为numpy数组以便计算统计信息
    results_dict = {}
    for key in all_results[0].keys():
        results_dict[key] = np.array([result[key] for result in all_results])

    # 计算每个指标的统计信息
    print(f"\n=== 20次实验的统计结果 (类别数={seen}, 每类anchor数={A}) ===")
    print(f"{'指标':<18} {'平均值':<7} {'标准差':<7} {'最小值':<7} {'最大值':<7}")
    print("-" * 60)

    for metric_name, values in results_dict.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        # 格式化指标名称
        display_name = {
            'top1_accuracy': 'Top-1 Accuracy',
            'top5_accuracy': 'Top-5 Accuracy',
            'precision_macro': 'Precision (Macro)',
            'recall_macro': 'Recall (Macro)',
            'f1_macro': 'F1 (Macro)',
            'precision_weighted': 'Precision (Weighted)',
            'recall_weighted': 'Recall (Weighted)',
            'f1_weighted': 'F1 (Weighted)'
        }.get(metric_name, metric_name)
        
        print(f"{display_name:<20} {mean_val:<10.4f} {std_val:<10.4f} {min_val:<10.4f} {max_val:<10.4f}")

    # 额外显示macro平均的主要指标汇总
    print(f"\n=== 主要指标汇总 (Macro Average) ===")
    print(f"Top-1 Accuracy: {np.mean(results_dict['top1_accuracy']):.4f} ± {np.std(results_dict['top1_accuracy']):.4f}")
    print(f"Top-5 Accuracy: {np.mean(results_dict['top5_accuracy']):.4f} ± {np.std(results_dict['top5_accuracy']):.4f}")
    print(f"Precision: {np.mean(results_dict['precision_macro']):.4f} ± {np.std(results_dict['precision_macro']):.4f}")
    print(f"Recall: {np.mean(results_dict['recall_macro']):.4f} ± {np.std(results_dict['recall_macro']):.4f}")
    print(f"F1-Score: {np.mean(results_dict['f1_macro']):.4f} ± {np.std(results_dict['f1_macro']):.4f}")

    return (
        np.mean(results_dict['top1_accuracy']),
        np.mean(results_dict['top5_accuracy']),
        np.mean(results_dict['precision_macro']),
        np.mean(results_dict['recall_macro']),
        np.mean(results_dict['f1_macro'])
    )

# 保持原有的数据读取和标签构建方式
cats = pd.Categorical(df[df.columns[-1]])
df['ids'] = cats.codes  # 得到类别id

# 准备数据
N = 100  # 保持原有的设置作为默认最大类别数
df_topN = df[df['ids'] < N].reset_index(drop=True)
print(df_topN.shape, df_topN.iloc[0]["name"])

embs = np.empty((df_topN.shape[0], 768))
for i, name in enumerate(tqdm(df_topN["name"])):
    file = emb_dir / (name+".pt")
    emb = torch.load(file, map_location='cpu').numpy()
    embs[i] = emb / np.linalg.norm(emb)

# 将数据转换为与ccip_zero相同的格式
labels = df_topN['ids'].values

res = ""

for N,seen in [(40, 20),(60, 40),(100, 60)]:
    for A in [1,3,5]:
        metrics = ','.join(map(str, main(embs, labels, N, A, seen)))
        res += f"csip,{N},{A},{metrics}\n"

with open('csip_zero_ood_metrics.csv', 'w') as f:
    f.write(res)

