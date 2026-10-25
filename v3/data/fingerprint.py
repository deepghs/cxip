"""Perceptual fingerprints (dHash) for near-duplicate matching.

Used to map the numbered class folders of csip_eval (deepghs/csip_eval, folder names are
1..122, file names are not danbooru md5s) onto danbooru artists, and as a reusable
dedup key for later dataset versions.

    python -m v3.data.fingerprint danbooru --out <work>/fingerprints   # resumable, parts of 100k
    python -m v3.data.fingerprint folder --root <csip_eval> --out <work>/csip_eval_fp.parquet
"""
from __future__ import annotations

import argparse
import io
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

DANBOORU_ROOT = Path('/data/neko/cxip_repro/data/danbooru-w640-ws-full-artist')


def _to_gray(im: Image.Image) -> Image.Image:
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
        im = im.convert('RGBA')
        bg = Image.new('RGBA', im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        im = bg
    return im.convert('L')


def dhash_bytes(gray: Image.Image, w: int, h: int) -> bytes:
    a = np.asarray(gray.resize((w + 1, h), Image.BOX), dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).ravel()).tobytes()


def fingerprint_bytes(data: bytes):
    im = Image.open(io.BytesIO(data))
    width, height = im.size
    if im.format == 'JPEG':
        im.draft('L', (256, 256))
    gray = _to_gray(im)
    return width, height, dhash_bytes(gray, 8, 8), dhash_bytes(gray, 16, 16)


def _work_ids(ids):
    out = []
    for i in ids:
        path = DANBOORU_ROOT / 'train' / f'{i}.webp'
        try:
            data = path.read_bytes()
            w, h, d64, d256 = fingerprint_bytes(data)
            out.append((i, len(data), w, h, d64, d256, None))
        except Exception as e:  # noqa: BLE001 - recorded per sample
            out.append((i, None, None, None, None, None, repr(e)[:300]))
    return out


def _work_paths(paths):
    out = []
    for p in paths:
        try:
            data = Path(p).read_bytes()
            w, h, d64, d256 = fingerprint_bytes(data)
            out.append((p, len(data), w, h, d64, d256, None))
        except Exception as e:  # noqa: BLE001
            out.append((p, None, None, None, None, None, repr(e)[:300]))
    return out


COLS = ['nbytes', 'width', 'height', 'dhash64', 'dhash256', 'error']


def run_danbooru(out: Path, workers: int, part_size: int, chunk: int):
    out.mkdir(parents=True, exist_ok=True)
    ids = np.sort(pd.read_parquet(DANBOORU_ROOT / 'index.parquet', columns=['id'])['id'].unique())
    n_parts = (len(ids) + part_size - 1) // part_size
    print(f'{len(ids)} ids, {n_parts} parts', flush=True)
    t0 = time.time()
    done = 0
    with Pool(workers) as pool:
        for p in range(n_parts):
            dst = out / f'part-{p:04d}.parquet'
            if dst.exists():
                continue
            part = ids[p * part_size:(p + 1) * part_size].tolist()
            chunks = [part[i:i + chunk] for i in range(0, len(part), chunk)]
            rows = [r for res in pool.imap_unordered(_work_ids, chunks) for r in res]
            df = pd.DataFrame(rows, columns=['id'] + COLS).sort_values('id')
            tmp = dst.with_name(f'.{dst.name}.tmp')
            df.to_parquet(tmp, index=False)
            os.replace(tmp, dst)
            done += len(part)
            el = time.time() - t0
            print(f'part {p + 1}/{n_parts} errors={df.error.notna().sum()} {done / el:.0f} img/s '
                  f'elapsed {el / 60:.1f} min', flush=True)
    print('done', flush=True)


def run_folder(root: Path, out: Path, workers: int):
    paths = sorted(str(p) for p in root.rglob('*') if p.is_file())
    with Pool(workers) as pool:
        rows = [r for res in pool.imap(_work_paths, [paths[i:i + 64] for i in range(0, len(paths), 64)])
                for r in res]
    df = pd.DataFrame(rows, columns=['path'] + COLS)
    df['rel'] = [os.path.relpath(p, root) for p in df.path]
    df['cls'] = [r.split('/')[0] for r in df.rel]
    df.to_parquet(out, index=False)
    print(f'{len(df)} files, errors={df.error.notna().sum()} -> {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['danbooru', 'folder'])
    ap.add_argument('--root', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--part-size', type=int, default=100_000)
    ap.add_argument('--chunk', type=int, default=256)
    a = ap.parse_args()
    if a.cmd == 'danbooru':
        run_danbooru(a.out, a.workers, a.part_size, a.chunk)
    else:
        run_folder(a.root, a.out, a.workers)
