import torch

def cohens_d(sim: torch.Tensor, label: torch.Tensor) -> float:
    """
    Compute Cohen's d between positive and negative groups.

    Args:
        sim (Tensor): shape (L,), predicted scores
        label (Tensor): shape (L,), binary labels (0 or 1)

    Returns:
        d (float): Cohen's d value
    """
    assert sim.shape == label.shape
    assert label.min() >= 0 and label.max() <= 1

    # Positive and negative groups
    sim_pos = sim[label == 1]
    sim_neg = sim[label == 0]

    # Means
    mean_pos = sim_pos.mean()
    mean_neg = sim_neg.mean()

    # Variances
    var_pos = sim_pos.var(unbiased=True)
    var_neg = sim_neg.var(unbiased=True)

    # Pooled standard deviation
    n_pos = sim_pos.numel()
    n_neg = sim_neg.numel()
    pooled_std = torch.sqrt(
        ((n_pos - 1) * var_pos + (n_neg - 1) * var_neg) / (n_pos + n_neg - 2)
    )

    d = (mean_pos - mean_neg) / (pooled_std + 1e-8)
    return d
