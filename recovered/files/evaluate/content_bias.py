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


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class StyleContentDataset(Dataset):
    """Dataset wrapping style embeddings and content probabilities."""

    def __init__(self, style_path: str, content_path: str, shuffle=False):
        self.style_list = list(Path(style_path).glob('*.pt'))
        with open(content_path, "rb") as f:
            self.content_dict = pickle.load(f)

        if shuffle:
            random.shuffle(self.style_list)

    def __len__(self) -> int:
        return len(self.style_list)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        style = torch.load(self.style_list[index], map_location='cpu')
        content = self.content_dict[self.style_list[index].stem]
        return style, content


class ProbeMLP(nn.Module):
    """A two-layer MLP probe that predicts content logits from style embeddings."""

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


@dataclass
class ProbeMetrics:
    train_loss: float
    val_loss: float
    test_loss: float


def train_probe(
    model: ProbeMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    device: torch.device,
    lr: float,
) -> ProbeMetrics:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    train_loss = float("inf")

    for epoch in range(epochs):
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
    model: ProbeMLP,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    sample_count = 0

    for style, content in dataloader:
        style = style.to(device)
        content = content.to(device)
        optimizer.zero_grad(set_to_none=True)

        logits = model(style)
        loss = criterion(logits, content)

        loss.backward()
        optimizer.step()

        batch_size = style.size(0)
        running_loss += loss.item() * batch_size
        sample_count += batch_size

    return running_loss / max(sample_count, 1)


@torch.no_grad()
def _evaluate(
    model: ProbeMLP,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    running_loss = 0.0
    sample_count = 0

    for style, content in dataloader:
        style = style.to(device)
        content = content.to(device)
        logits = model(style)
        loss = criterion(logits, content)
        batch_size = style.size(0)
        running_loss += loss.item() * batch_size
        sample_count += batch_size

    return running_loss / max(sample_count, 1)


def split_dataset(
    dataset: StyleContentDataset,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[Dataset, Dataset, Dataset]:
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    total_len = len(dataset)
    val_len = int(total_len * val_ratio)
    test_len = int(total_len * test_ratio)
    train_len = total_len - val_len - test_len

    if min(train_len, val_len, test_len) <= 0:
        raise ValueError("Dataset too small for the requested splits.")

    return random_split(
        dataset,
        lengths=[train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(seed),
    )


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


def load_arrays(style_path: Path, content_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    style = np.load(style_path)
    content = np.load(content_path)
    return style, content


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
    content_path = Path(args.content_path)
    if not style_path.exists():
        raise FileNotFoundError(f"Style embedding file not found: {style_path}")
    if not content_path.exists():
        raise FileNotFoundError(f"Content probability file not found: {content_path}")

    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = StyleContentDataset(style_path, content_path)

    train_ds, val_ds, test_ds = split_dataset(dataset, args.val_ratio, args.test_ratio, args.seed)
    train_loader = build_dataloader(train_ds, args.batch_size, True, args.num_workers, device)
    val_loader = build_dataloader(val_ds, args.batch_size, False, args.num_workers, device)
    test_loader = build_dataloader(test_ds, args.batch_size, False, args.num_workers, device)

    model = ProbeMLP(
        input_dim=768,
        output_dim=320,
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

    baseline_losses = []
    for run_idx in range(args.baseline_repeats):
        baseline_dataset = StyleContentDataset(style_path, content_path, shuffle=True)
        train_ds_b, val_ds_b, test_ds_b = split_dataset(
            baseline_dataset, args.val_ratio, args.test_ratio, args.seed + run_idx + 1
        )
        train_loader_b = build_dataloader(train_ds_b, args.batch_size, True, args.num_workers, device)
        val_loader_b = build_dataloader(val_ds_b, args.batch_size, False, args.num_workers, device)
        test_loader_b = build_dataloader(test_ds_b, args.batch_size, False, args.num_workers, device)

        baseline_model = ProbeMLP(
            input_dim=baseline_dataset.style.shape[1],
            output_dim=baseline_dataset.content.shape[1],
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
    parser.add_argument("--style-path", type=str, default="/data/cxip/feat_csip_dbv4", help="Path to style embedding npy.")
    parser.add_argument("--content-path", type=str, default="/nfs/danbooru-wdtagger-v4-w640-ws-30k/train/probs.pkl", help="Path to content probability npy.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Probe hidden layer width.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument("--baseline-repeats", type=int, default=5, help="Number of shuffled baselines.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker threads.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_style_content_bias(args)


if __name__ == "__main__":
    main()

