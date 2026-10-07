"""
把 neko-data 格式的合并 CCIP 数据集上传到 KohakuHub (hub.deepghs.org) deepghs/character_similarity 的子目录
    source env.sh; python repro/ccip_merge/upload_khub.py [--src data/ccip_merged_nd] [--prefix ccip_merged_v1]
分批提交 (每批约 4GB), 远端已有同大小文件跳过, 可断点重跑; manifest.json 最后提交
"""
import argparse
import os
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

ap = argparse.ArgumentParser()
ap.add_argument('--src', default='/data/neko/cxip_repro/data/ccip_merged_nd')
ap.add_argument('--repo', default='deepghs/character_similarity')
ap.add_argument('--prefix', default='ccip_merged_v1')
ap.add_argument('--batch-bytes', type=float, default=4e9)
a = ap.parse_args()

api = HfApi(endpoint=os.environ['KHUB_ENDPOINT'], token=os.environ['KHUB_TOKEN'])
src = Path(a.src)
remote = {f.path: getattr(f, 'size', None)
          for f in api.list_repo_tree(a.repo, repo_type='dataset', path_in_repo=a.prefix, recursive=True)} \
    if any(f.path == a.prefix for f in api.list_repo_tree(a.repo, repo_type='dataset')) else {}
files = sorted(p for p in src.rglob('*') if p.is_file() and not p.name.startswith('.'))
files = [p for p in files if p.name != 'manifest.json'] + [src / 'manifest.json']
todo = [p for p in files if remote.get(f'{a.prefix}/{p.relative_to(src).as_posix()}') != p.stat().st_size]
print(f'{len(files)} files, {len(todo)} to upload, {sum(p.stat().st_size for p in todo) / 1e9:.1f} GB', flush=True)

batches, cur, size = [], [], 0
for p in todo:
    if p.name == 'manifest.json':
        continue
    if cur and size + p.stat().st_size > a.batch_bytes:
        batches.append(cur)
        cur, size = [], 0
    cur.append(p)
    size += p.stat().st_size
if cur:
    batches.append(cur)
if src / 'manifest.json' in todo:
    batches.append([src / 'manifest.json'])

for i, batch in enumerate(batches):
    ops = [CommitOperationAdd(path_in_repo=f'{a.prefix}/{p.relative_to(src).as_posix()}', path_or_fileobj=str(p))
           for p in batch]
    for attempt in range(5):
        try:
            t0 = time.time()
            api.create_commit(a.repo, ops, repo_type='dataset',
                              commit_message=f'ccip_merged_v1 (neko-data format): batch {i + 1}/{len(batches)}')
            gb = sum(p.stat().st_size for p in batch) / 1e9
            print(f'batch {i + 1}/{len(batches)} {len(batch)} files {gb:.1f} GB {gb / (time.time() - t0) * 1e3:.0f} MB/s',
                  flush=True)
            break
        except Exception as e:
            print(f'batch {i + 1} attempt {attempt} failed: {e!r}'[:500], flush=True)
            time.sleep(30 * (attempt + 1))
    else:
        raise SystemExit(f'batch {i + 1} failed')
print('done', flush=True)
