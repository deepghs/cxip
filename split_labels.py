#!/usr/bin/env python3
"""
Merge two label JSON files and split by class into train/test by randomly sampling classes.

Inputs are JSON files with mapping: {"id.webp": "class_name", ...}

Outputs:
- train_split.json: mapping of id -> class for training set
- test_split.json: mapping of id -> class for test set

Usage example:
  python3 /data/cxip/split_labels.py \
    --inputs /nfs/danbooru-w640-ws-full-artist/train_labels.json \
            /nfs/danbooru-w640-ws-full-artist/test_labels.json \
    --output_dir /data/cxip/label_splits \
    --n_test_classes 800 \
    --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Tuple


def read_labels_json(json_path: str) -> Dict[str, str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"File {json_path} must contain a JSON object mapping id -> class")
    # Basic validation
    for key, value in data.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(
                f"Invalid entry in {json_path}: keys and values must be strings, got {type(key)}->{type(value)}"
            )
    return data


def merge_label_maps(label_maps: List[Dict[str, str]]) -> Tuple[Dict[str, str], int]:
    merged: Dict[str, str] = {}
    conflict_count: int = 0
    for label_map in label_maps:
        for image_id, class_name in label_map.items():
            if image_id in merged and merged[image_id] != class_name:
                # Prefer the first occurrence deterministically; count conflict and skip
                conflict_count += 1
                continue
            merged.setdefault(image_id, class_name)
    return merged, conflict_count


def split_by_random_classes(
    merged_labels: Dict[str, str], n_test_classes: int, seed: int
) -> Tuple[Dict[str, str], Dict[str, str], List[str]]:
    all_classes = sorted({cls for cls in merged_labels.values()})
    if n_test_classes <= 0:
        raise ValueError("n_test_classes must be > 0")
    if len(all_classes) < n_test_classes:
        raise ValueError(
            f"Requested {n_test_classes} test classes, but only {len(all_classes)} unique classes were found"
        )
    rng = random.Random(seed)
    test_classes = set(rng.sample(all_classes, n_test_classes))

    test_split: Dict[str, str] = {}
    train_split: Dict[str, str] = {}
    for image_id, class_name in merged_labels.items():
        if class_name in test_classes:
            test_split[image_id] = class_name
        else:
            train_split[image_id] = class_name

    return train_split, test_split, sorted(test_classes)


def save_json(path: str, data: Dict[str, str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge two label JSON files and split by random classes into train/test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Two JSON files to merge (order determines precedence on conflicts)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to write train/test split JSON files",
    )
    parser.add_argument(
        "--n_test_classes",
        type=int,
        default=800,
        help="Number of classes to sample for the test set",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for class sampling",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if len(args.inputs) < 2:
        print("[ERROR] Please provide at least two input JSON files via --inputs", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Read inputs
    label_maps: List[Dict[str, str]] = []
    for path in args.inputs:
        try:
            label_maps.append(read_labels_json(path))
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Failed to read {path}: {exc}", file=sys.stderr)
            sys.exit(1)

    # Merge with precedence by input order
    merged_labels, conflict_count = merge_label_maps(label_maps)

    # Split by classes
    try:
        train_split, test_split, test_classes = split_by_random_classes(
            merged_labels, args.n_test_classes, args.seed
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Split failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # Save outputs
    train_path = os.path.join(args.output_dir, "train_split.json")
    test_path = os.path.join(args.output_dir, "test_split.json")
    save_json(train_path, train_split)
    save_json(test_path, test_split)

    # Summary
    total_images = len(merged_labels)
    total_classes = len({c for c in merged_labels.values()})
    print(
        json.dumps(
            {
                "total_images": total_images,
                "total_unique_classes": total_classes,
                "conflicting_labels_skipped": conflict_count,
                "n_test_classes": len(test_classes),
                "test_images": len(test_split),
                "train_images": len(train_split),
                "output_dir": os.path.abspath(args.output_dir),
                "train_split_path": os.path.abspath(train_path),
                "test_split_path": os.path.abspath(test_path),
                "seed": args.seed,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

