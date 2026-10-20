# Metrics Visualization 模块使用说明

这个模块提供了一套完整的二分类模型评估可视化工具，包括F1分数、精确率、召回率、PR曲线等多种可视化方法。

## 功能概览

### 1. F1分数 vs 阈值曲线
绘制不同阈值下的F1分数，帮助找到最优分类阈值。

### 2. 分数分布直方图
展示同类样本和不同类样本的分数分布，直观显示模型的区分能力。

### 3. PR曲线（Precision-Recall Curve）
绘制精确率-召回率曲线，并计算PR AUC值。

### 4. 精确率和召回率 vs 阈值曲线
在同一张图中展示精确率、召回率和F1分数随阈值的变化。

### 5. 指标NPY导出
将不同阈值下的精确率、召回率保存为NPY文件（float16格式），便于快速加载和进一步分析。

## 快速开始

### 方式一：一键生成所有图表（推荐）

```python
from metrics_visualization import generate_all_visualizations

results = generate_all_visualizations(
    y_true=y_true_flat,        # 真实标签 (0 or 1)
    y_scores=y_pred_flat,      # 预测分数
    best_thresh=best_thresh,   # 可选：最佳阈值
    output_prefix='',          # 输出文件前缀
    dpi=300                    # 图像分辨率
)

print(f"最佳阈值: {results['best_threshold']}")
print(f"最大F1分数: {results['max_f1']}")
print(f"PR AUC: {results['auc_score']}")
```

这将生成以下文件：
- `f1_scores_vs_threshold.png` - F1分数vs阈值曲线
- `score_distribution.png` - 分数分布直方图
- `pr_curve.png` - PR曲线
- `precision_recall_vs_threshold.png` - 精确率/召回率vs阈值曲线
- `metrics_vs_threshold.npy` - 指标NPY文件（float16格式，shape: [N, 3]）

### 方式二：单独调用各个函数

```python
from metrics_visualization import (
    plot_f1_vs_threshold,
    plot_score_distribution,
    plot_pr_curve,
    plot_precision_recall_vs_threshold,
    save_metrics_to_npy
)

# 1. 绘制F1 vs 阈值
thresholds, f1_scores = plot_f1_vs_threshold(
    y_true, y_scores,
    best_thresh=0.5,
    save_path='f1_plot.png'
)

# 2. 绘制分数分布
stats = plot_score_distribution(
    y_true, y_scores,
    best_thresh=0.5,
    save_path='distribution.png'
)

# 3. 绘制PR曲线
precisions, recalls, thresholds, auc_score = plot_pr_curve(
    y_true, y_scores,
    save_path='pr_curve.png'
)

# 4. 绘制精确率/召回率 vs 阈值
precisions, recalls, thresholds = plot_precision_recall_vs_threshold(
    y_true, y_scores,
    best_thresh=0.5,
    save_path='pr_vs_threshold.png'
)

# 5. 保存指标到NPY (float16) - 全量存储，不采样
metrics_array = save_metrics_to_npy(
    y_true, y_scores,
    save_path='metrics.npy'
)
# metrics_array shape: (N, 3) where N is number of unique thresholds
# [:, 0] = thresholds, [:, 1] = precision, [:, 2] = recall
```

## 函数详细说明

### `plot_f1_vs_threshold(y_true, y_scores, best_thresh=None, max_f1=None, save_path, dpi=300)`

绘制F1分数随阈值变化的曲线。

**参数：**
- `y_true`: 真实二分类标签 (0 or 1)
- `y_scores`: 预测分数
- `best_thresh`: 最佳阈值（可选，如不提供会自动计算）
- `max_f1`: 最大F1分数（可选）
- `save_path`: 保存路径
- `dpi`: 图像分辨率

**返回：**
- `thresholds`: 阈值数组
- `f1_scores`: F1分数数组

### `plot_score_distribution(y_true, y_scores, best_thresh=None, save_path, dpi=300, bins=50)`

绘制同类vs不同类样本的分数分布直方图。

**参数：**
- `y_true`: 真实二分类标签 (0 or 1)
- `y_scores`: 预测分数
- `best_thresh`: 最佳阈值标记线（可选）
- `save_path`: 保存路径
- `dpi`: 图像分辨率
- `bins`: 直方图bins数量

**返回：**
- `stats`: 包含分布统计信息的字典

### `plot_pr_curve(y_true, y_scores, save_path, dpi=300)`

绘制Precision-Recall曲线。

**参数：**
- `y_true`: 真实二分类标签 (0 or 1)
- `y_scores`: 预测分数
- `save_path`: 保存路径
- `dpi`: 图像分辨率

**返回：**
- `precisions`: 精确率数组
- `recalls`: 召回率数组
- `thresholds`: 阈值数组
- `auc_score`: PR曲线下面积

### `plot_precision_recall_vs_threshold(y_true, y_scores, best_thresh=None, save_path, dpi=300)`

绘制精确率、召回率和F1分数随阈值变化的曲线。

**参数：**
- `y_true`: 真实二分类标签 (0 or 1)
- `y_scores`: 预测分数
- `best_thresh`: 最佳阈值标记线（可选）
- `save_path`: 保存路径
- `dpi`: 图像分辨率

**返回：**
- `precisions`: 精确率数组
- `recalls`: 召回率数组
- `thresholds`: 阈值数组

### `save_metrics_to_npy(y_true, y_scores, save_path)`

保存完整PR曲线的阈值、精确率和召回率到NPY文件（float16格式），**不做采样，存储全量数据**。

**参数：**
- `y_true`: 真实二分类标签 (0 or 1)
- `y_scores`: 预测分数
- `save_path`: NPY保存路径

**返回：**
- `metrics_array`: numpy数组，shape为 (N, 3)，其中N是唯一阈值的数量

**数组格式：**
```python
# metrics_array shape: (N, 3)
# 列0: threshold values (所有唯一阈值)
# 列1: precision values
# 列2: recall values

# 使用示例：
import numpy as np
metrics = np.load('metrics_vs_threshold.npy')
thresholds = metrics[:, 0]  # 全量阈值
precisions = metrics[:, 1]  # 对应的精确率
recalls = metrics[:, 2]     # 对应的召回率

print(f"Total unique thresholds: {len(thresholds)}")
```

**优点：**
- 使用float16节省存储空间（相比float32节省50%，相比float64节省75%）
- 保存完整的PR曲线，无信息损失
- 加载速度快，适合大规模数据
- 易于与NumPy/PyTorch集成
- 可用于后续精确分析和可视化

### `generate_all_visualizations(y_true, y_scores, best_thresh=None, output_prefix='', dpi=300)`

一次性生成所有可视化图表和NPY文件（全量数据，不采样）。

**参数：**
- `y_true`: 真实二分类标签 (0 or 1)
- `y_scores`: 预测分数
- `best_thresh`: 最佳阈值（可选）
- `output_prefix`: 输出文件名前缀
- `dpi`: 图像分辨率

**返回：**
- `results`: 包含所有计算结果的字典
  - `best_threshold`: 最佳阈值
  - `max_f1`: 最大F1分数
  - `auc_score`: PR AUC分数
  - `metrics_array`: 指标数组 (N, 3)，N为唯一阈值数量

## 在eval_ccip.py中的应用

在 `scripts/eval/eval_ccip.py` 中，原来的绘图代码已被替换为：

```python
from metrics_visualization import generate_all_visualizations

# 生成所有图表和NPY（全量数据）
results = generate_all_visualizations(
    y_true=y_true_flat,
    y_scores=y_pred_flat,
    output_prefix='visualizations/',
    dpi=300
)
```

这将替代原来的手动绘图代码，生成更完整的评估报告，包括完整的PR曲线数据。

## 示例代码

查看 `scripts/eval/example_metrics_visualization.py` 获取完整的使用示例。

## 依赖项

- numpy
- matplotlib
- sklearn
- pandas

## 输出示例

### 1. F1 Score vs Threshold
显示不同阈值下的F1分数，红点标记最佳阈值。

### 2. Score Distribution
直方图显示同类（绿色）和不同类（红色）样本的分数分布，红色虚线为最佳阈值。

### 3. PR Curve
标准的Precision-Recall曲线，显示PR AUC值。

### 4. Precision/Recall/F1 vs Threshold
三条曲线显示精确率（蓝色）、召回率（绿色）、F1分数（粉色）随阈值的变化。

### 5. NPY数组文件
包含均匀采样的阈值及对应的precision、recall值，以float16格式存储，便于快速加载和进一步分析。

## 注意事项

1. 输入的 `y_true` 必须是二分类标签（0或1）
2. `y_scores` 可以是任意范围的分数，不必归一化到[0,1]
3. 所有函数都使用非交互式后端（Agg），适合在服务器环境运行
4. 建议使用 `generate_all_visualizations()` 函数一次性生成所有报告
5. NPY文件中的阈值是在分数范围内均匀采样的
6. 使用float16格式存储可节省约50%的磁盘空间（相比float32）
7. float16精度足够用于阈值、precision、recall的存储（精度约为3-4位有效数字）

## 许可

MIT License

