
#!/usr/bin/env python3
"""
Memory-aware clustering utility for CCIP features.

This script mirrors the workflow in `eval_ccip.py` while adding memmap-based
feature staging so that multi-million embeddings can be clustered without
exhausting system RAM.
"""

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import hdbscan
import numpy as np
import torch
import umap
from tqdm import tqdm


DEFAULT_FEAT_DIR = Path("/nfs/danbooru-w640-ws-full-artist/train_ccip_feats")
DEFAULT_OUTPUT_DIR = Path("/nfs/danbooru-w640-ws-full-artist")
DEFAULT_MEMMAP_PATH = Path("/tmp/ccip_feat_matrix.dat")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster CCIP features with UMAP + HDBSCAN (memory friendly)."
    )
    parser.add_argument("--feat-dir", type=Path, default=DEFAULT_FEAT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--file-pattern", default="*.pt", help="Glob to match feature files.")
    parser.add_argument("--feat-npy", type=Path, default=None,
                        help="repro/data/extract_ccip_feats.py 的输出目录 (feats_r*.npy + files_r*.json)，"
                             "代替每张图一个 .pt 的 --feat-dir")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of samples.")
    parser.add_argument("--sample-rate", type=float, default=1.0,
                        help="Random subsampling probability in (0,1].")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--memmap-path", type=Path, default=DEFAULT_MEMMAP_PATH)
    parser.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--chunk-size", type=int, default=8192,
                        help="How many files to stream before flushing progress log.")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Skip L2 normalization of each feature vector.")
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.0)
    parser.add_argument("--umap-components", type=int, default=35)
    parser.add_argument("--umap-metric", default="cosine")
    parser.add_argument("--umap-jobs", type=int, default=os.cpu_count() or 8)
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--cluster-metric", default="euclidean")
    parser.add_argument("--cluster-selection-eps", type=float, default=0.0)
    parser.add_argument("--core-dist-jobs", type=int, default=os.cpu_count() or 8)
    parser.add_argument("--out-name", default=None,
                        help="Override output json name. Defaults to timestamped file.")
    args = parser.parse_args()
    if not (0.0 < args.sample_rate <= 1.0):
        raise ValueError("sample-rate must be in (0, 1].")
    return args


def list_feature_files(feat_dir: Path, pattern: str) -> List[Path]:
    if not feat_dir.exists():
        raise FileNotFoundError(f"Feature directory {feat_dir} does not exist.")
    return sorted(feat_dir.glob(pattern))


def resolve_feature_tensor(obj: torch.Tensor) -> torch.Tensor:
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        for key in ("feat", "feature", "embedding", "emb", "rep", "vector"):
            if key in obj:
                return obj[key]
    raise ValueError("Unsupported feature container. Expect tensor or dict with feature key.")


def to_numpy(feature: torch.Tensor, normalize: bool) -> np.ndarray:
    array = feature.detach().float().cpu().numpy()
    if normalize:
        norm = np.linalg.norm(array)
        if norm > 0:
            array = array / norm
    return array


def maybe_subsample(files: Sequence[Path], sample_rate: float, seed: int) -> List[Path]:
    if sample_rate >= 1.0:
        return list(files)
    rng = np.random.default_rng(seed)
    return [path for path in files if rng.random() < sample_rate]


def stream_to_memmap(
    files: Sequence[Path],
    memmap_path: Path,
    dtype: str,
    normalize: bool,
    chunk_size: int,
    base_dir: Path,
) -> Tuple[np.memmap, List[str], dict]:
    if not files:
        raise RuntimeError("No feature files found to cluster.")

    first = resolve_feature_tensor(torch.load(files[0], map_location="cpu"))
    feature_dim = int(first.numel())
    dtype_np = np.float32 if dtype == "fp32" else np.float16

    memmap_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.memmap(
        filename=memmap_path,
        mode="w+",
        dtype=dtype_np,
        shape=(len(files), feature_dim),
    )

    filenames: List[str] = []
    for idx, file_path in enumerate(tqdm(files, desc="Loading features", unit="feat")):
        tensor = resolve_feature_tensor(torch.load(file_path, map_location="cpu"))
        flat = to_numpy(tensor.reshape(-1), normalize=normalize).astype(dtype_np, copy=False)
        if flat.shape[0] != feature_dim:
            raise RuntimeError(f"Feature dimension mismatch for {file_path}. "
                               f"Expected {feature_dim}, got {flat.shape[0]}")
        matrix[idx] = flat
        try:
            rel_path = file_path.relative_to(base_dir)
            filenames.append(rel_path.as_posix())
        except ValueError:
            filenames.append(file_path.name)
        if (idx + 1) % chunk_size == 0:
            matrix.flush()
    matrix.flush()
    metadata = {
        "feature_count": len(files),
        "feature_dim": feature_dim,
        "dtype": str(dtype_np),
        "normalized": normalize,
    }
    readonly = np.memmap(
        filename=memmap_path,
        mode="r",
        dtype=dtype_np,
        shape=(len(files), feature_dim),
    )
    return readonly, filenames, metadata


def load_npy_shards(
    feat_dir: Path,
    memmap_path: Path,
    dtype: str,
    normalize: bool,
    sample_rate: float,
    seed: int,
    limit,
) -> Tuple[np.memmap, List[str], dict]:
    """读取 extract_ccip_feats.py 的分片, 与 stream_to_memmap 一样写入 memmap; 文件名记为 {id}.pt 以兼容 clusters_to_char_labels.py"""
    shards = sorted(feat_dir.glob("files_r*.json"))
    if not shards:
        raise RuntimeError(f"No files_r*.json in {feat_dir}")
    names, arrays = [], []
    for shard in shards:
        with shard.open("r", encoding="utf-8") as fp:
            names.extend(f"{name}.pt" for name in json.load(fp))
        arrays.append(np.load(feat_dir / shard.name.replace("files_", "feats_").replace(".json", ".npy"), mmap_mode="r"))
    order = np.argsort(names, kind="stable")  # 与 --feat-dir 的 sorted(glob) 顺序一致
    if sample_rate < 1.0:
        rng = np.random.default_rng(seed)
        order = order[rng.random(len(order)) < sample_rate]
    if limit is not None:
        order = order[:limit]
    offsets = np.cumsum([0] + [len(a) for a in arrays])
    dtype_np = np.float32 if dtype == "fp32" else np.float16
    feature_dim = arrays[0].shape[1]
    memmap_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.memmap(filename=memmap_path, mode="w+", dtype=dtype_np, shape=(len(order), feature_dim))
    for start in tqdm(range(0, len(order), 65536), desc="Loading features", unit="chunk"):
        idx = order[start:start + 65536]
        shard_idx = np.searchsorted(offsets, idx, side="right") - 1
        chunk = np.empty((len(idx), feature_dim), dtype=np.float32)
        for s in np.unique(shard_idx):
            m = shard_idx == s
            local = idx[m] - offsets[s]
            sort = np.argsort(local)
            rows = np.empty((len(local), feature_dim), dtype=np.float32)
            rows[sort] = arrays[s][local[sort]]
            chunk[m] = rows
        if normalize:
            chunk /= np.maximum(np.linalg.norm(chunk, axis=1, keepdims=True), 1e-12)
        matrix[start:start + len(idx)] = chunk.astype(dtype_np, copy=False)
    matrix.flush()
    metadata = {"feature_count": len(order), "feature_dim": int(feature_dim), "dtype": str(dtype_np),
                "normalized": normalize}
    readonly = np.memmap(filename=memmap_path, mode="r", dtype=dtype_np, shape=(len(order), feature_dim))
    return readonly, [names[i] for i in order], metadata


def run_umap(matrix: np.memmap, args: argparse.Namespace) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        n_components=args.umap_components,
        metric=args.umap_metric,
        low_memory=True,
        random_state=args.seed,
        n_jobs=args.umap_jobs,
        verbose=True,
    )
    return reducer.fit_transform(matrix)


def run_hdbscan(embedding: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric=args.cluster_metric,
        core_dist_n_jobs=args.core_dist_jobs,
        cluster_selection_epsilon=args.cluster_selection_eps,
    )
    return clusterer.fit_predict(embedding)


def serialize_results(
    args: argparse.Namespace,
    filenames: Sequence[str],
    labels: Sequence[int],
    metadata: dict,
    embedding_shape: Tuple[int, int],
) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_name = args.out_name or f"ccip_clusters_{timestamp}.json"
    out_path = args.output_dir / out_name

    payload = {
        "meta": {
            **metadata,
            "umap": {
                "n_neighbors": args.umap_neighbors,
                "min_dist": args.umap_min_dist,
                "n_components": args.umap_components,
                "metric": args.umap_metric,
            },
            "hdbscan": {
                "min_cluster_size": args.min_cluster_size,
                "min_samples": args.min_samples,
                "metric": args.cluster_metric,
                "cluster_selection_epsilon": args.cluster_selection_eps,
            },
            "embedding_shape": embedding_shape,
            "noise_label": -1,
        },
        "files": list(filenames),
        "labels": [int(x) for x in labels],
    }

    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False)

    return out_path


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.feat_npy is not None:
        matrix, filenames, metadata = load_npy_shards(
            feat_dir=args.feat_npy,
            memmap_path=args.memmap_path,
            dtype=args.dtype,
            normalize=not args.no_normalize,
            sample_rate=args.sample_rate,
            seed=args.seed,
            limit=args.limit,
        )
    else:
        files = list_feature_files(args.feat_dir, args.file_pattern)
        files = maybe_subsample(files, args.sample_rate, args.seed)
        if args.limit is not None:
            files = files[: args.limit]

        matrix, filenames, metadata = stream_to_memmap(
            files=files,
            memmap_path=args.memmap_path,
            dtype=args.dtype,
            normalize=not args.no_normalize,
            chunk_size=args.chunk_size,
            base_dir=args.feat_dir,
        )

    embedding = run_umap(matrix, args)
    labels = run_hdbscan(embedding, args)
    out_path = serialize_results(
        args=args,
        filenames=filenames,
        labels=labels,
        metadata=metadata,
        embedding_shape=tuple(map(int, embedding.shape)),
    )
    print(f"Cluster assignments written to {out_path}")


if __name__ == "__main__":
    main()
