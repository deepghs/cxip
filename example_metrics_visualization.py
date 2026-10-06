"""
Example Usage of metrics_visualization module

This script demonstrates how to use the metrics visualization functions.
"""

import numpy as np
from metrics_visualization import (
    plot_f1_vs_threshold,
    plot_score_distribution,
    plot_pr_curve,
    plot_precision_recall_vs_threshold,
    save_metrics_to_npy,
    generate_all_visualizations
)

# Example 1: Generate sample data
print("Example 1: Using sample data")
print("-" * 50)

# Create sample data
np.random.seed(42)
n_positive = 500
n_negative = 500

# Positive class: higher scores
y_true_pos = np.ones(n_positive)
y_scores_pos = np.random.normal(0.7, 0.15, n_positive)

# Negative class: lower scores  
y_true_neg = np.zeros(n_negative)
y_scores_neg = np.random.normal(0.3, 0.15, n_negative)

# Combine
y_true = np.concatenate([y_true_pos, y_true_neg])
y_scores = np.concatenate([y_scores_pos, y_scores_neg])

# Clip scores to [0, 1] range for demonstration
y_scores = np.clip(y_scores, 0, 1)

print(f"Sample data created: {len(y_true)} samples")
print(f"Positive samples: {np.sum(y_true)}")
print(f"Negative samples: {len(y_true) - np.sum(y_true)}")
print()

# Example 2: Generate all visualizations at once (recommended)
print("Example 2: Generate all visualizations at once")
print("-" * 50)
results = generate_all_visualizations(
    y_true=y_true,
    y_scores=y_scores,
    output_prefix='example_',
    dpi=300
)

print(f"\nBest threshold: {results['best_threshold']:.4f}")
print(f"Max F1 score: {results['max_f1']:.4f}")
print(f"PR AUC: {results['auc_score']:.4f}")
print()

# Example 3: Use individual functions (for more control)
print("Example 3: Using individual functions")
print("-" * 50)

# Plot F1 vs Threshold
thresholds, f1_scores = plot_f1_vs_threshold(
    y_true, y_scores,
    save_path='individual_f1_vs_threshold.png'
)
print(f"F1 vs Threshold plot created with {len(thresholds)} thresholds")

# Plot score distribution
stats = plot_score_distribution(
    y_true, y_scores,
    best_thresh=0.5,
    save_path='individual_score_distribution.png'
)
print(f"Score distribution stats:")
print(f"  Same class mean: {stats['same_class_mean']:.4f}")
print(f"  Different class mean: {stats['diff_class_mean']:.4f}")

# Plot PR curve
precisions, recalls, thresholds, auc_score = plot_pr_curve(
    y_true, y_scores,
    save_path='individual_pr_curve.png'
)
print(f"PR curve created with AUC: {auc_score:.4f}")

# Plot Precision/Recall vs Threshold
precisions, recalls, thresholds = plot_precision_recall_vs_threshold(
    y_true, y_scores,
    best_thresh=0.5,
    save_path='individual_pr_vs_threshold.png'
)
print(f"Precision/Recall vs Threshold plot created")

# Save metrics to NPY (float16) - full curve without sampling
metrics_array = save_metrics_to_npy(
    y_true, y_scores,
    save_path='individual_metrics.npy'
)
print(f"Metrics NPY saved with shape {metrics_array.shape}")
print(f"  Column 0: thresholds (all unique values)")
print(f"  Column 1: precision")
print(f"  Column 2: recall")
print()

print("=" * 50)
print("All examples completed successfully!")
print("=" * 50)

