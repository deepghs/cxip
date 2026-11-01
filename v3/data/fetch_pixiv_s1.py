"""Download the pixiv s1 webp tars (hub dataset ``pixiv_ai_s1-webp-4Mpixel``) used by pixiv-artist-v1 and bench B3.

Unused in v4: the s1 source is an AI-image corpus (see docs/csip_v4_p0_baseline_zh.md §6.1). Pixiv was dropped
from CSIP v4 at the P0 checkpoint (2026-11-01); the s1 / pixiv-artist-v1 data it produced has been deleted.
Kept only as a record of what was run.

    python -m v3.data.fetch_pixiv_s1 index    # all images/NNNN.json (small)
    python -m v3.data.fetch_pixiv_s1 table    # merge the jsons -> raw/image_table.parquet (+ id-set check)
    python -m v3.data.fetch_pixiv_s1 tars     # all images/NNNN.tar, sha256-verified, resumable

Outputs under ``PIXIV_RAW``: ``images/NNNN.{json,tar}``, ``done.jsonl`` (one row per verified tar), ``image_table.parquet``.

Auth: the hub token is read from ``~/.config/cxip/env.sh`` and handed to curl only through ``--config -`` on stdin
(never argv / files / logs). The file listing uses huggingface_hub (``HfApi.dataset_info``) in-process, which gives
names, sizes and LFS sha256 in one call; downloads use curl (the hub 403s python urllib). Concurrency is a thread pool
(<= 8 curls; no fork pools, forked workers inherit curl stdin pipes and deadlock).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

V4 = Path('/scratch/cxip_v4')
PIXIV_RAW = V4 / 'data/pixiv-s1'
REPO = 'deepghs/pixiv_ai_s1-webp-4Mpixel'
ENV_SH = Path('~/.config/cxip/env.sh').expanduser()
EXIST_IDS = Path('/scratch/pixiv_probe/exist_ids.json')
COLS = ['image_id', 'tar', 'offset', 'size', 'sha256']
MAX_WORKERS = 8
_NAME_RE = re.compile(r'(\d+)\.webp')


# ----------------------------------------------------------------------------- pure functions

def parse_tar_index(json_path: Path) -> pd.DataFrame:
    """One tar's json -> DataFrame[image_id:int64, tar:str, offset:int64, size:int64, sha256:str]; webp members only."""
    json_path = Path(json_path)
    files = json.loads(json_path.read_text())['files']
    tar = str(json_path.with_suffix('.tar'))
    rows = []
    for name, m in files.items():
        mt = _NAME_RE.fullmatch(name)
        if mt:
            rows.append((int(mt.group(1)), tar, int(m['offset']), int(m['size']), m['sha256']))
    df = pd.DataFrame(rows, columns=COLS)
    return df.astype({'image_id': 'int64', 'offset': 'int64', 'size': 'int64'})


def image_table(raw: Path = PIXIV_RAW, rebuild: bool = False) -> pd.DataFrame:
    """Merge every ``raw/images/*.json``; cached in ``raw/image_table.parquet`` (``rebuild=True`` ignores the cache)."""
    raw = Path(raw)
    cache = raw / 'image_table.parquet'
    if cache.exists() and not rebuild:
        return pd.read_parquet(cache)
    jsons = sorted((raw / 'images').glob('*.json'))
    if not jsons:
        raise FileNotFoundError(f'no json index under {raw / "images"}; run `index` first')
    df = pd.concat([parse_tar_index(p) for p in jsons], ignore_index=True)
    tmp = cache.with_name(cache.name + '.tmp')
    df.to_parquet(tmp, index=False)
    os.replace(tmp, cache)
    return df


def verify_sha256(path: Path, expected: str, chunk: int = 1 << 24) -> bool:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest() == expected.lower()


# ----------------------------------------------------------------------------- network

def read_env(path: Path = ENV_SH) -> dict:
    """HF_ENDPOINT / HF_TOKEN as exported by ``env.sh`` (sourced by bash: values may reference other variables).
    Never logged."""
    r = subprocess.run(['bash', '-c', '. "$1" >/dev/null 2>&1; printf "%s\\0%s" "$HF_ENDPOINT" "$HF_TOKEN"', 'bash', str(path)],
                       capture_output=True, check=True)
    ep, tok = r.stdout.decode().split('\0')
    if not ep or not tok:
        raise RuntimeError(f'HF_ENDPOINT / HF_TOKEN not set by {path}')
    return dict(HF_ENDPOINT=ep, HF_TOKEN=tok)


def list_remote(env: dict) -> dict:
    """{name: (size, lfs_sha256 or None)} for everything under images/ (flat names like 0000.tar)."""
    from huggingface_hub import HfApi
    info = HfApi(endpoint=env['HF_ENDPOINT'], token=env['HF_TOKEN']).dataset_info(REPO, files_metadata=True)
    out = {}
    for s in info.siblings:
        if s.rfilename.startswith('images/') and s.rfilename.count('/') == 1:
            out[s.rfilename[len('images/'):]] = (s.size, s.lfs.sha256 if s.lfs else None)
    return out


def curl_get(env: dict, name: str, dest: Path, timeout_s: int = 7200) -> None:
    """curl -fL -C - download of images/{name} into dest (resumes a partial dest). Token only via stdin."""
    url = f"{env['HF_ENDPOINT'].rstrip('/')}/datasets/{REPO}/resolve/main/images/{name}"
    cfg = f'header = "Authorization: Bearer {env["HF_TOKEN"]}"\n'
    cmd = ['curl', '-fL', '-C', '-', '--retry', '5', '--retry-delay', '3', '--retry-all-errors',
           '--connect-timeout', '30', '--speed-time', '120', '--speed-limit', '1000',
           '-sS', '--config', '-', '-o', str(dest), url]
    r = subprocess.run(cmd, input=cfg.encode(), capture_output=True, timeout=timeout_s)
    if r.returncode != 0:
        err = r.stderr.decode(errors='replace').replace(env['HF_TOKEN'], '***')[-400:]
        raise RuntimeError(f'curl {name} exit {r.returncode}: {err}')


def _log(msg: str) -> None:
    print(time.strftime('%F %T'), msg, flush=True)


def _fetch_one(env: dict, name: str, final: Path, expected_size: int | None, expected_sha: str | None,
               tries: int = 3) -> tuple[int, float]:
    """Download -> (verify) -> atomic rename. Returns (bytes, seconds). Raises after ``tries`` failed attempts."""
    part = final.with_name(final.name + '.part')
    t0 = time.time()
    err = None
    for k in range(tries):
        try:
            if not (part.exists() and expected_size and part.stat().st_size == expected_size):
                curl_get(env, name, part)       # a complete .part would answer 416, so skip curl in that case
            size = part.stat().st_size
            if expected_size is not None and size != expected_size:
                raise RuntimeError(f'size {size} != {expected_size}')
            if expected_sha and not verify_sha256(part, expected_sha):
                part.unlink()                   # our own partial file; corrupt -> restart from scratch
                raise RuntimeError('sha256 mismatch')
            os.replace(part, final)
            return size, time.time() - t0
        except Exception as e:                  # noqa: BLE001
            err = e
            _log(f'WARN {name} attempt {k + 1}/{tries}: {e}')
            time.sleep(5 * (k + 1))
    raise RuntimeError(f'{name} failed after {tries} attempts: {err}')


# ----------------------------------------------------------------------------- subcommands

def cmd_index(raw: Path) -> int:
    env = read_env()
    (raw / 'images').mkdir(parents=True, exist_ok=True)
    remote = list_remote(env)
    names = sorted(n for n in remote if n.endswith('.json'))
    todo = [n for n in names if not ((raw / 'images' / n).exists() and (raw / 'images' / n).stat().st_size == remote[n][0])]
    _log(f'index: {len(names)} json, {len(todo)} to fetch')
    failed = 0
    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        futs = {ex.submit(_fetch_one, env, n, raw / 'images' / n, remote[n][0], None): n for n in todo}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                f.result()
            except Exception as e:              # noqa: BLE001
                failed += 1
                _log(f'FAIL {futs[f]}: {e}')
            if i % 200 == 0:
                _log(f'index: {i}/{len(todo)}')
    _log(f'index done, failed={failed}')
    return 1 if failed else 0


def _done_names(done: Path) -> set:
    if not done.exists():
        return set()
    return {json.loads(l)['name'] for l in done.read_text().splitlines() if l.strip()}


def cmd_tars(raw: Path) -> int:
    env = read_env()
    img = raw / 'images'
    img.mkdir(parents=True, exist_ok=True)
    done_path = raw / 'done.jsonl'
    remote = list_remote(env)
    tars = sorted(n for n in remote if n.endswith('.tar'))
    if not all((img / (n[:-4] + '.json')).exists() for n in tars):
        _log('missing json index; run `index` first')
        return 2
    done = _done_names(done_path)
    todo = []
    for n in tars:
        p = img / n
        if n in done and p.exists() and p.stat().st_size == remote[n][0]:
            continue
        todo.append(n)
    _log(f'tars: {len(tars)} total, {len(tars) - len(todo)} already done, {len(todo)} to fetch, workers={MAX_WORKERS}')
    lock = threading.Lock()
    stats = dict(bytes=0, n=0, failed=0)
    t_start = time.time()

    def work(n: str):
        meta = json.loads((img / (n[:-4] + '.json')).read_text())
        sha = meta.get('hash_lfs') or remote[n][1]
        size = meta.get('filesize') or remote[n][0]
        if not sha:
            raise RuntimeError(f'{n}: no expected sha256 available')
        nb, sec = _fetch_one(env, n, img / n, size, sha)
        row = dict(name=n, size=nb, sha256=sha, seconds=round(sec, 1), ts=int(time.time()))
        with lock:
            with open(done_path, 'a') as f:
                f.write(json.dumps(row) + '\n')
            stats['bytes'] += nb
            stats['n'] += 1
            el = time.time() - t_start
            _log(f'OK {n} {nb / 1e6:.0f}MB in {sec:.0f}s | {stats["n"]}/{len(todo)} done | '
                 f'avg {stats["bytes"] / el / 1e6:.0f} MB/s ({el / 60:.1f} min)')

    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        futs = {ex.submit(work, n): n for n in todo}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:              # noqa: BLE001
                stats['failed'] += 1
                _log(f'FAIL {futs[f]}: {e}')
    total_done = len(_done_names(done_path))
    _log(f'tars finished: done.jsonl={total_done}/{len(tars)}, failed this run={stats["failed"]}')
    return 1 if stats['failed'] else 0


def cmd_table(raw: Path, rebuild: bool, compare: Path | None) -> int:
    df = image_table(raw, rebuild=rebuild)
    _log(f'image_table: {len(df)} rows, {df.tar.nunique()} tars, unique ids {df.image_id.nunique()}')
    rc = 0
    if compare is not None and compare.exists():
        ref = set(json.loads(compare.read_text()))
        got = set(df.image_id.tolist())
        same = ref == got
        _log(f'compare with {compare.name}: ref={len(ref)} table={len(got)} only_ref={len(ref - got)} '
             f'only_table={len(got - ref)} identical={same}')
        rc = 0 if same else 1
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('cmd', choices=['index', 'tars', 'table'])
    ap.add_argument('--raw', type=Path, default=PIXIV_RAW)
    ap.add_argument('--rebuild', action='store_true', help='table: ignore the cached parquet')
    ap.add_argument('--compare', type=Path, default=EXIST_IDS, help='table: json list of ids to compare with')
    a = ap.parse_args(argv)
    if a.cmd == 'index':
        return cmd_index(a.raw)
    if a.cmd == 'tars':
        return cmd_tars(a.raw)
    return cmd_table(a.raw, a.rebuild, a.compare)


if __name__ == '__main__':
    sys.exit(main())
