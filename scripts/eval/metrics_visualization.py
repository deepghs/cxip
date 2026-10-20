"""
Metrics Visualization Utilities for Binary Classification

This module provides functions to visualize and analyze binary classification metrics,
including F1 score, precision, recall, and score distributions.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import pandas as pd


def plot_f1_vs_threshold(y_true, y_scores, best_thresh=None, max_f1=None, 
                         save_path='f1_scores_vs_threshold.png', dpi=300):
    """
    Plot F1 score as a function of threshold.
    
    Args:
        y_true (array-like): True binary labels (0 or 1)
        y_scores (array-like): Predicted scores
        best_thresh (float, optional): Best threshold to highlight
        max_f1 (float, optional): Maximum F1 score to highlight
        save_path (str): Path to save the plot
        dpi (int): DPI for saved figure
    
    Returns:
        tuple: (thresholds, f1_scores) arrays
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    
    # Calculate F1 scores for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    f1_scores = f1_scores[:-1]  # Remove last element to match thresholds length
    
    # If best_thresh not provided, find it
    if best_thresh is None:
        best_f1_idx = np.argmax(f1_scores)
        best_thresh = thresholds[best_f1_idx]
        max_f1 = f1_scores[best_f1_idx]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(thresholds, f1_scores, label='F1 Score', linewidth=2, color='#2E86DE')
    plt.scatter(best_thresh, max_f1, color='red', zorder=5, s=100, 
                label=f'Best Threshold ({best_thresh:.3f})')
    plt.axvline(x=best_thresh, color='red', linestyle='--', alpha=0.7)
    plt.xlabel("Threshold", fontsize=12)
    plt.ylabel("F1 Score", fontsize=12)
    plt.title("F1 Score vs. Threshold for Same Class Classification", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"F1 vs Threshold plot saved to '{save_path}'")
    return thresholds, f1_scores


def plot_score_distribution(y_true, y_scores, best_thresh=None, 
                            save_path='score_distribution.png', dpi=300, bins=50):
    """
    Plot score distribution for same class vs different class pairs.
    
    Args:
        y_true (array-like): True binary labels (0 or 1)
        y_scores (array-like): Predicted scores
        best_thresh (float, optional): Best threshold to highlight
        save_path (str): Path to save the plot
        dpi (int): DPI for saved figure
        bins (int): Number of histogram bins
    
    Returns:
        dict: Statistics about the score distributions
    """
    same_class_scores = y_scores[y_true == 1]
    diff_class_scores = y_scores[y_true == 0]
    
    # Calculate statistics
    stats = {
        'same_class_mean': np.mean(same_class_scores),
        'same_class_std': np.std(same_class_scores),
        'diff_class_mean': np.mean(diff_class_scores),
        'diff_class_std': np.std(diff_class_scores),
    }
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.hist(diff_class_scores, bins=bins, alpha=0.7, 
             label=f'Different class (n={len(diff_class_scores)})', 
             density=True, color='#EE5A6F')
    plt.hist(same_class_scores, bins=bins, alpha=0.7, 
             label=f'Same class (n={len(same_class_scores)})', 
             density=True, color='#26DE81')
    
    if best_thresh is not None:
        plt.axvline(x=best_thresh, color='red', linestyle='--', linewidth=2, 
                   label=f'Best threshold ({best_thresh:.3f})')
    
    plt.xlabel('Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Same Class vs Different Class Score Distribution', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Score distribution plot saved to '{save_path}'")
    return stats


def plot_pr_curve(y_true, y_scores, save_path='pr_curve.png', dpi=300):
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true (array-like): True binary labels (0 or 1)
        y_scores (array-like): Predicted scores
        save_path (str): Path to save the plot
        dpi (int): DPI for saved figure
    
    Returns:
        tuple: (precisions, recalls, thresholds, auc_score)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    
    # Calculate AUC (Area Under Curve)
    auc_score = auc(recalls, precisions)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    plt.plot(recalls, precisions, linewidth=2, color='#2E86DE', 
             label=f'PR Curve (AUC = {auc_score:.3f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"PR curve saved to '{save_path}'")
    return precisions, recalls, thresholds, auc_score


def plot_precision_recall_vs_threshold(y_true, y_scores, best_thresh=None,
                                       save_path='precision_recall_vs_threshold.png', dpi=300):
    """
    Plot Precision and Recall as functions of threshold.
    
    Args:
        y_true (array-like): True binary labels (0 or 1)
        y_scores (array-like): Predicted scores
        best_thresh (float, optional): Best threshold to highlight
        save_path (str): Path to save the plot
        dpi (int): DPI for saved figure
    
    Returns:
        tuple: (precisions, recalls, thresholds)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    
    # Remove last element to match thresholds length
    precisions = precisions[:-1]
    recalls = recalls[:-1]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(thresholds, precisions, linewidth=2, color='#2E86DE', label='Precision')
    plt.plot(thresholds, recalls, linewidth=2, color='#26DE81', label='Recall')
    
    # Calculate F1 for reference
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    plt.plot(thresholds, f1_scores, linewidth=2, color='#FDA7DF', 
             label='F1 Score', linestyle='--')
    
    if best_thresh is not None:
        plt.axvline(x=best_thresh, color='red', linestyle='--', linewidth=2, 
                   alpha=0.7, label=f'Best threshold ({best_thresh:.3f})')
    
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Precision, Recall, and F1 Score vs. Threshold', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([thresholds.min(), thresholds.max()])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Precision/Recall vs Threshold plot saved to '{save_path}'")
    return precisions, recalls, thresholds


def save_metrics_to_npy(y_true, y_scores, save_path='metrics_vs_threshold.npy'):
    """
    Save all Precision, Recall and Threshold values to NPY file as float16.
    Uses the full precision-recall curve without sampling.
    
    Args:
        y_true (array-like): True binary labels (0 or 1)
        y_scores (array-like): Predicted scores
        save_path (str): Path to save the NPY file
    
    Returns:
        np.ndarray: Array of shape (N, 3) containing [threshold, precision, recall]
                    where N is the number of unique thresholds
    """
    # Get all precision, recall, threshold values from sklearn
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    
    # precision_recall_curve returns one more precision/recall than thresholds
    # Remove the last precision/recall to match thresholds length
    precisions = precisions[:-1]
    recalls = recalls[:-1]
    
    # Stack into (N, 3) array: [threshold, precision, recall]
    metrics_array = np.stack([
        thresholds,
        precisions,
        recalls
    ], axis=1).astype(np.float16)
    
    # Save to NPY
    np.save(save_path, metrics_array)
    
    print(f"Metrics saved to NPY (float16): '{save_path}'")
    print(f"Array shape: {metrics_array.shape}")
    print(f"Total {len(thresholds)} thresholds (full curve, no sampling)")
    print(f"Threshold range: [{thresholds.min():.4f}, {thresholds.max():.4f}]")
    
    return metrics_array


def generate_all_visualizations(y_true, y_scores, best_thresh=None, 
                                output_prefix='', dpi=300):
    """
    Generate all visualizations and save metrics to NPY in one call.
    
    Args:
        y_true (array-like): True binary labels (0 or 1)
        y_scores (array-like): Predicted scores
        best_thresh (float, optional): Best threshold. If None, will be computed.
        output_prefix (str): Prefix for output files
        dpi (int): DPI for saved figures
    
    Returns:
        dict: Dictionary containing all computed metrics and statistics
    """
    print("=" * 50)
    print("Generating all visualizations...")
    print("=" * 50)
    
    # Compute best threshold if not provided
    if best_thresh is None:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        f1_scores = f1_scores[:-1]
        best_f1_idx = np.argmax(f1_scores)
        best_thresh = thresholds[best_f1_idx]
        max_f1 = f1_scores[best_f1_idx]
    else:
        # Calculate max_f1 for the given threshold
        y_pred = (y_scores >= best_thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        max_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    results = {
        'best_threshold': best_thresh,
        'max_f1': max_f1
    }
    
    # 1. F1 vs Threshold
    thresholds, f1_scores = plot_f1_vs_threshold(
        y_true, y_scores, best_thresh, max_f1,
        save_path=f'{output_prefix}f1_scores_vs_threshold.png', dpi=dpi
    )
    results['thresholds'] = thresholds
    results['f1_scores'] = f1_scores
    
    # 2. Score Distribution
    stats = plot_score_distribution(
        y_true, y_scores, best_thresh,
        save_path=f'{output_prefix}score_distribution.png', dpi=dpi
    )
    results['distribution_stats'] = stats
    
    # 3. PR Curve
    precisions, recalls, _, auc_score = plot_pr_curve(
        y_true, y_scores,
        save_path=f'{output_prefix}pr_curve.png', dpi=dpi
    )
    results['auc_score'] = auc_score
    
    # 4. Precision/Recall vs Threshold
    precisions, recalls, thresholds = plot_precision_recall_vs_threshold(
        y_true, y_scores, best_thresh,
        save_path=f'{output_prefix}precision_recall_vs_threshold.png', dpi=dpi
    )
    
    # 5. Save to NPY (float16) - full curve without sampling
    metrics_array = save_metrics_to_npy(
        y_true, y_scores,
        save_path=f'{output_prefix}metrics_vs_threshold.npy'
    )
    results['metrics_array'] = metrics_array
    
    print("=" * 50)
    print("All visualizations completed!")
    print("=" * 50)
    
    return results


if __name__ == '__main__':
    # Example usage
    print("Example usage of metrics_visualization module:")
    print("\nfrom metrics_visualization import generate_all_visualizations")
    print("\nresults = generate_all_visualizations(")
    print("    y_true=y_true_flat,")
    print("    y_scores=y_pred_flat,")
    print("    best_thresh=best_thresh,")
    print("    output_prefix='output/',")
    print("    n_thresholds=100")
    print(")")
    print("\n# Load saved metrics:")
    print("import numpy as np")
    print("metrics = np.load('metrics_vs_threshold.npy')")
    print("# metrics shape: (n_thresholds, 3)")
    print("# metrics[:, 0] = thresholds")
    print("# metrics[:, 1] = precision values")
    print("# metrics[:, 2] = recall values")

