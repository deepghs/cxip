"""
Test script to verify NPY saving functionality
"""

import numpy as np
from metrics_visualization import save_metrics_to_npy

# Create simple test data
np.random.seed(42)
n_samples = 100

# Positive class: higher scores
y_true_pos = np.ones(n_samples // 2)
y_scores_pos = np.random.uniform(0.6, 1.0, n_samples // 2)

# Negative class: lower scores
y_true_neg = np.zeros(n_samples // 2)
y_scores_neg = np.random.uniform(0.0, 0.5, n_samples // 2)

# Combine
y_true = np.concatenate([y_true_pos, y_true_neg])
y_scores = np.concatenate([y_scores_pos, y_scores_neg])

print("=" * 50)
print("Testing save_metrics_to_npy function")
print("=" * 50)

# Save metrics
metrics_array = save_metrics_to_npy(
    y_true, y_scores,
    n_thresholds=50,
    save_path='test_metrics.npy'
)

print(f"\nSaved array info:")
print(f"  Shape: {metrics_array.shape}")
print(f"  Dtype: {metrics_array.dtype}")
print(f"  Size: {metrics_array.nbytes} bytes")

# Load and verify
print("\n" + "=" * 50)
print("Loading and verifying saved data")
print("=" * 50)

loaded_metrics = np.load('test_metrics.npy')
print(f"Loaded array shape: {loaded_metrics.shape}")
print(f"Loaded array dtype: {loaded_metrics.dtype}")

# Display first few rows
print(f"\nFirst 5 rows:")
print(f"{'Threshold':>12} {'Precision':>12} {'Recall':>12}")
print("-" * 40)
for i in range(5):
    print(f"{loaded_metrics[i, 0]:12.6f} {loaded_metrics[i, 1]:12.6f} {loaded_metrics[i, 2]:12.6f}")

print(f"\nLast 5 rows:")
print(f"{'Threshold':>12} {'Precision':>12} {'Recall':>12}")
print("-" * 40)
for i in range(-5, 0):
    print(f"{loaded_metrics[i, 0]:12.6f} {loaded_metrics[i, 1]:12.6f} {loaded_metrics[i, 2]:12.6f}")

# Verify data integrity
print("\n" + "=" * 50)
print("Verifying data integrity")
print("=" * 50)

assert np.array_equal(metrics_array, loaded_metrics), "Data mismatch!"
print("✓ Saved and loaded data match perfectly")

# Check value ranges
thresholds = loaded_metrics[:, 0]
precisions = loaded_metrics[:, 1]
recalls = loaded_metrics[:, 2]

print(f"\nThreshold range: [{thresholds.min():.6f}, {thresholds.max():.6f}]")
print(f"Precision range: [{precisions.min():.6f}, {precisions.max():.6f}]")
print(f"Recall range: [{recalls.min():.6f}, {recalls.max():.6f}]")

assert np.all((precisions >= 0) & (precisions <= 1)), "Precision out of range!"
assert np.all((recalls >= 0) & (recalls <= 1)), "Recall out of range!"
print("✓ All precision and recall values are in valid range [0, 1]")

# Check file size comparison
import os
file_size = os.path.getsize('test_metrics.npy')
theoretical_float32_size = metrics_array.shape[0] * metrics_array.shape[1] * 4  # 4 bytes per float32
theoretical_float64_size = metrics_array.shape[0] * metrics_array.shape[1] * 8  # 8 bytes per float64
savings_vs_float32 = (1 - file_size / theoretical_float32_size) * 100
savings_vs_float64 = (1 - file_size / theoretical_float64_size) * 100

print(f"\nFile size comparison:")
print(f"  Actual (float16): {file_size} bytes")
print(f"  Theoretical float32: {theoretical_float32_size} bytes")
print(f"  Theoretical float64: {theoretical_float64_size} bytes")
print(f"  Savings vs float32: {savings_vs_float32:.1f}%")
print(f"  Savings vs float64: {savings_vs_float64:.1f}%")

print("\n" + "=" * 50)
print("All tests passed! ✓")
print("=" * 50)

# Cleanup
os.remove('test_metrics.npy')
print("\nTest file cleaned up.")
