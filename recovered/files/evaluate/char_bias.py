#!/usr/bin/env python3
"""Evaluate the content leakage of style embeddings using a probe MLP."""

from __future__ import annotations

import argparse
import copy
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import pickle
from tqdm import tqdm
from loss import LogitsKLLoss


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class StyleCharDataset(Dataset):
    """Dataset wrapping style embeddings and character embeddings."""

    def __init__(self, style_path: str, char_path: str, shuffle=False, indices=None):
        """
        Args:
            style_path: Path to style embedding directory
            char_path: Path to character embedding directory
            shuffle: If True, randomly pair style and char (for baseline)
            indices: Optional list of indices to use (for train/val split)
        """
        all_style_files = sorted(list(Path(style_path).glob('*.pt')))
        all_char_files = sorted(list(Path(char_path).glob('*.pt')))
        
        if indices is not None:
            self.style_list = [all_style_files[i] for i in indices]
            self.char_list = [all_char_files[i] for i in indices]
        else:
            self.style_list = all_style_files
            self.char_list = all_char_files

        self.shuffle = shuffle

    def __len__(self) -> int:
        return len(self.style_list)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.shuffle:
            index_c = random.randint(0, len(self.style_list) - 1)
        else:
            index_c = index

        style = torch.load(self.style_list[index], map_location='cpu')
        char = torch.load(self.char_list[index_c], map_location='cpu')
        return style, char


class ProbeMLP(nn.Module):
    """A two-layer MLP probe that predicts character logits from style embeddings."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.layer1(inputs))
        logits = self.layer2(hidden)
        return logits

class ProbeModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.style_probe = ProbeMLP(input_dim, output_dim, hidden_dim)
        self.char_probe = ProbeMLP(input_dim, output_dim, hidden_dim)

    def forward(self, style: torch.Tensor, char: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits_style = self.style_probe(style)
        logits_char = self.char_probe(char)
        return logits_style, logits_char

@dataclass
class ProbeMetrics:
    train_loss: float
    val_loss: float
    test_loss: float


def train_probe(
    model: ProbeModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    device: torch.device,
    lr: float,
) -> ProbeMetrics:
    criterion = LogitsKLLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    train_loss = float("inf")

    for epoch in tqdm(range(epochs)):
        train_loss = _train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        val_loss = _evaluate(model, val_loader, criterion, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    test_loss = _evaluate(model, test_loader, criterion, device)
    return ProbeMetrics(train_loss=train_loss, val_loss=best_val_loss, test_loss=test_loss)


def _train_one_epoch(
    model: ProbeModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    sample_count = 0

    for style, char in dataloader:
        style = style.float().to(device)
        char = char.float().to(device)
        optimizer.zero_grad(set_to_none=True)

        logits_style, logits_char = model(style, char)
        loss = criterion(logits_style, logits_char)

        loss.backward()
        optimizer.step()

        batch_size = style.size(0)
        running_loss += loss.item() * batch_size
        sample_count += batch_size

    return running_loss / max(sample_count, 1)


@torch.no_grad()
def _evaluate(
    model: ProbeModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    running_loss = 0.0
    sample_count = 0

    for style, char in dataloader:
        style = style.float().to(device)
        char = char.float().to(device)
        logits_style, logits_char = model(style, char)
        loss = criterion(logits_style, logits_char)
        batch_size = style.size(0)
        running_loss += loss.item() * batch_size
        sample_count += batch_size

    return running_loss / max(sample_count, 1)


def split_dataset_by_character(
    style_path: Path,
    val_ratio: float,
    seed: int,
) -> Tuple[list, list]:
    """
    Split dataset by character, ensuring each character contributes to both train and val.
    
    Args:
        style_path: Path to style embedding directory
        val_ratio: Ratio of validation samples per character
        seed: Random seed
    
    Returns:
        train_indices: List of indices for training set
        val_indices: List of indices for validation set
    """
    rng = random.Random(seed)
    
    # Group files by character name
    all_files = sorted(list(style_path.glob('*.pt')))
    char_groups = {}
    
    for idx, filepath in enumerate(all_files):
        # Extract character name from filename: {char_name}_{...}.pt
        char_name = filepath.stem.split('_')[0]
        if char_name not in char_groups:
            char_groups[char_name] = []
        char_groups[char_name].append(idx)
    
    train_indices = []
    val_indices = []
    
    # For each character, split its samples into train/val
    for char_name, indices in char_groups.items():
        rng.shuffle(indices)
        n_samples = len(indices)
        n_val = max(1, int(n_samples * val_ratio))  # At least 1 sample for val
        n_train = n_samples - n_val
        
        if n_train < 1:
            # If character has too few samples, put all in train
            train_indices.extend(indices)
            print(f"Warning: Character '{char_name}' has only {n_samples} samples, all used for training")
        else:
            train_indices.extend(indices[:n_train])
            val_indices.extend(indices[n_train:])
            print(f"Character '{char_name}': {n_train} train, {n_val} val samples")
    
    return sorted(train_indices), sorted(val_indices)


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def summarize_results(
    real_metrics: ProbeMetrics,
    baseline_losses: torch.Tensor,
) -> None:
    baseline_mean = baseline_losses.mean().item()
    baseline_std = baseline_losses.std(unbiased=False).item()
    ratio = real_metrics.test_loss / baseline_mean

    print("\n==============================")
    print("   Style Embedding Probe Summary")
    print("==============================")
    print(f"Real probe test loss     : {real_metrics.test_loss:.6f}")
    print(f"Baseline loss (mean±std) : {baseline_mean:.6f} ± {baseline_std:.6f}")
    print(f"Loss ratio (real/base)   : {ratio:.4f}")

    if ratio <= 0.5:
        print("✅ The probe finds strong content leakage in style embeddings.")
    elif ratio <= 0.9:
        print("⚠️  Moderate leakage detected. Further disentangling recommended.")
    else:
        print("✅ Style embeddings appear content-agnostic relative to baseline.")


def evaluate_style_content_bias(args: argparse.Namespace) -> None:
    style_path = Path(args.style_path)
    char_path = Path(args.char_path)
    if not style_path.exists():
        raise FileNotFoundError(f"Style embedding file not found: {style_path}")
    if not char_path.exists():
        raise FileNotFoundError(f"Character embedding file not found: {char_path}")

    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Split dataset by character (stratified split)
    print("\n=== Splitting dataset by character ===")
    train_indices, val_indices = split_dataset_by_character(style_path, args.val_ratio, args.seed)
    print(f"\nTotal: {len(train_indices)} train, {len(val_indices)} val samples")
    
    # Create datasets with non-overlapping indices
    train_ds = StyleCharDataset(style_path, char_path, shuffle=False, indices=train_indices)
    val_ds = StyleCharDataset(style_path, char_path, shuffle=False, indices=val_indices)
    test_ds = val_ds  # Use same data for val and test
    
    train_loader = build_dataloader(train_ds, args.batch_size, True, args.num_workers, device)
    val_loader = build_dataloader(val_ds, args.batch_size, False, args.num_workers, device)
    test_loader = build_dataloader(test_ds, args.batch_size, False, args.num_workers, device)

    model = ProbeModel(
        input_dim=768,
        output_dim=100,
        hidden_dim=args.hidden_dim,
    ).to(device)

    real_metrics = train_probe(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        device=device,
        lr=args.lr,
    )

    # Baseline: shuffled pairing with same train/val split
    print("\n=== Training baseline models (shuffled pairing) ===")
    baseline_losses = []
    for run_idx in range(args.baseline_repeats):
        print(f"\nBaseline run {run_idx + 1}/{args.baseline_repeats}")
        # Use same split but shuffle pairing
        train_ds_b = StyleCharDataset(style_path, char_path, shuffle=True, indices=train_indices)
        val_ds_b = StyleCharDataset(style_path, char_path, shuffle=True, indices=val_indices)
        test_ds_b = val_ds_b  # Use same data for val and test
        
        train_loader_b = build_dataloader(train_ds_b, args.batch_size, True, args.num_workers, device)
        val_loader_b = build_dataloader(val_ds_b, args.batch_size, False, args.num_workers, device)
        test_loader_b = build_dataloader(test_ds_b, args.batch_size, False, args.num_workers, device)

        baseline_model = ProbeModel(
            input_dim=768,
            output_dim=100,
            hidden_dim=args.hidden_dim,
        ).to(device)

        baseline_metrics = train_probe(
            model=baseline_model,
            train_loader=train_loader_b,
            val_loader=val_loader_b,
            test_loader=test_loader_b,
            epochs=args.epochs,
            device=device,
            lr=args.lr,
        )
        baseline_losses.append(baseline_metrics.test_loss)

    summarize_results(real_metrics, torch.tensor(baseline_losses))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a probe to quantify content leakage in style embeddings.",
    )
    parser.add_argument("--style-path", type=str, default="/data/cxip/feats/feat_csip_v2.1-char", help="Path to style embedding npy.")
    parser.add_argument("--char-path", type=str, default="/data/cxip/feats/feat_ccip_zero", help="Path to character embedding npy.")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Probe hidden layer width.")
    parser.add_argument("--batch-size", type=int, default=512, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=2e-4, help="AdamW learning rate.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio per character.")
    parser.add_argument("--baseline-repeats", type=int, default=5, help="Number of shuffled baselines.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker threads.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_style_content_bias(args)


if __name__ == "__main__":
    main()

