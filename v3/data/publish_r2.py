"""Publish a local neko-data dataset directory to R2 with rclone and verify it.

    python -m v3.data.publish_r2 upload --local /data/neko/cxip_v3/data/danbooru-artist-v1 \
        --remote r2:sizigi-deepghs/datasets/khub/cxip-danbooru-artist/v1
    python -m v3.data.publish_r2 verify --local ... --remote ...

upload: ``rclone copy --checksum --immutable --s3-no-check-bucket`` (never deletes anything remote or
local), data files first and ``manifest.json`` last, so a manifest only appears once its shards exist.
verify: (1) remote object list == local file list with equal sizes, (2) ``rclone check --one-way``
(size + MD5) local vs remote, (3) sha256 of every local tar / parquet == manifest, (4) one shard read
back fully from R2 (streamed) -> sha256 == manifest, tar member count == num_samples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# --s3-no-head: R2 answers the post-upload 'HEAD ?versionId=' of rclone 1.60 with 501 NotImplemented
# (the PUT itself succeeds; Content-MD5 is verified server side, and verify() re-checks MD5s).
RCLONE = ['rclone', '--s3-no-check-bucket', '--s3-no-head']


def _run(args, **kw):
    print('+', ' '.join(args), flush=True)
    return subprocess.run(args, check=True, **kw)


def upload(local: Path, remote: str, transfers: int = 16, log: str | None = None):
    common = [*RCLONE, 'copy', '--checksum', '--immutable', '--transfers', str(transfers), '--checkers', '32',
              '--s3-chunk-size', '64M', '--s3-upload-concurrency', '4', '--stats', '60s', '--stats-one-line',
              '--retries', '5', '--low-level-retries', '20']
    if log:
        common += ['--log-file', log, '--log-level', 'INFO']
    # --filter (not --include + --exclude, whose relative order rclone does not guarantee)
    _run([*common, '--filter', '- .*', '--filter', '- /manifest.json', '--filter', '+ **', str(local), remote])
    _run([*common, '--filter', '+ /manifest.json', '--filter', '- **', str(local), remote])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(16 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify(local: Path, remote: str, read_back: int = 1, workers: int = 16, seed: int = 0):
    ok = True
    # (1) listing
    out = subprocess.run([*RCLONE, 'lsjson', '-R', '--files-only', remote], check=True, capture_output=True,
                         text=True).stdout
    remote_files = {o['Path']: o['Size'] for o in json.loads(out)}
    local_files = {p.relative_to(local).as_posix(): p.stat().st_size for p in local.rglob('*')
                   if p.is_file() and not p.name.startswith('.')}
    missing = sorted(set(local_files) - set(remote_files))
    extra = sorted(set(remote_files) - set(local_files))
    size_bad = sorted(k for k in set(local_files) & set(remote_files) if local_files[k] != remote_files[k])
    print(f'(1) local {len(local_files)} files {sum(local_files.values()) / 2**30:.2f} GiB; remote '
          f'{len(remote_files)} files {sum(remote_files.values()) / 2**30:.2f} GiB; missing {len(missing)} '
          f'extra {len(extra)} size mismatch {len(size_bad)}', flush=True)
    if missing or size_bad:
        ok = False
        print('   missing:', missing[:10], 'size mismatch:', size_bad[:10])
    # (2) hash check (MD5 stored by rclone for multipart uploads)
    r = subprocess.run([*RCLONE, 'check', '--one-way', '--checkers', '32', str(local), remote],
                       capture_output=True, text=True)
    tail = [l for l in r.stderr.splitlines() if 'differences' in l or 'matching' in l or 'ERROR' in l][-6:]
    print('(2) rclone check exit', r.returncode, *tail, sep='\n    ', flush=True)
    ok &= r.returncode == 0
    # (3) local sha256 vs manifest
    manifest = json.load(open(local / 'manifest.json'))
    items = []
    for s in manifest['shards']:
        items.append((s['path'], s['sha256']))
        if s.get('metadata_path'):
            items.append((s['metadata_path'], s['metadata_sha256']))
    with ThreadPoolExecutor(workers) as ex:
        got = list(ex.map(lambda it: _sha256(local / it[0]), items))
    bad = [p for (p, want), g in zip(items, got) if g != want]
    print(f'(3) sha256 local vs manifest: {len(items) - len(bad)}/{len(items)} OK', flush=True)
    ok &= not bad
    # (4) read back shards from R2
    rng = random.Random(seed)
    for s in rng.sample(manifest['shards'], read_back):
        proc = subprocess.Popen([*RCLONE, 'cat', f"{remote}/{s['path']}"], stdout=subprocess.PIPE)
        h = hashlib.sha256()

        class _Tee:
            def read(self, n=-1):
                b = proc.stdout.read(n)
                h.update(b)
                return b

        n_img = 0
        with tarfile.open(fileobj=_Tee(), mode='r|') as tf:
            for m in tf:
                if m.name.rsplit('.', 1)[-1] in {'webp', 'jpg', 'jpeg', 'png'}:
                    n_img += 1
        while True:  # drain tar padding
            b = proc.stdout.read(1 << 20)
            if not b:
                break
            h.update(b)
        proc.wait()
        good = h.hexdigest() == s['sha256'] and n_img == s['num_samples']
        print(f"(4) read back {s['path']}: sha256 {'OK' if h.hexdigest() == s['sha256'] else 'BAD'}, "
              f"{n_img} images (manifest {s['num_samples']})", flush=True)
        ok &= good
    print('ALL OK' if ok else 'VERIFY FAILED', flush=True)
    return ok


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['upload', 'verify'])
    ap.add_argument('--local', type=Path, required=True)
    ap.add_argument('--remote', required=True)
    ap.add_argument('--transfers', type=int, default=16)
    ap.add_argument('--log')
    ap.add_argument('--read-back', type=int, default=1)
    a = ap.parse_args()
    if a.cmd == 'upload':
        upload(a.local, a.remote, a.transfers, a.log)
    else:
        sys.exit(0 if verify(a.local, a.remote, a.read_back) else 1)
