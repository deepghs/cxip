"""校验 KohakuHub deepghs/character_similarity: 原文件 sha256 与 HF 一致, ccip_merged_v1 分片 sha256 与 manifest 一致"""
import json
import os
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO = 'deepghs/character_similarity'
hf = HfApi(token=os.environ.get('HF_TOKEN'))
kh = HfApi(endpoint=os.environ['KHUB_ENDPOINT'], token=os.environ['KHUB_TOKEN'])
if len(sys.argv) > 1:
    kh.create_commit(REPO, [CommitOperationAdd('README.md', sys.argv[1])], repo_type='dataset',
                     commit_message='README: add merged_v1 (neko-data format, ccip_merged_v1/)')
print('khub source:', kh.dataset_info(REPO).__dict__.get('_source', '?'))
remote = {f.path: f for f in kh.list_repo_tree(REPO, repo_type='dataset', recursive=True) if hasattr(f, 'size')}
bad = 0
for s in hf.dataset_info(REPO, files_metadata=True).siblings:
    r = remote.get(s.rfilename)
    ok = r is not None and r.size == s.size and (not s.lfs or (r.lfs and r.lfs.sha256 == s.lfs.sha256))
    bad += not ok and s.rfilename != 'README.md'
    print('hf-mirror', 'OK ' if ok else 'BAD', s.rfilename)
src = Path('/data/neko/cxip_repro/data/ccip_merged_nd')
m = json.load(open(src / 'manifest.json'))
n = 0
for sh in m['shards']:
    for path, sha in [(sh['path'], sh['sha256']), (sh.get('metadata_path'), sh.get('metadata_sha256'))]:
        if not path:
            continue
        r = remote.get(f'ccip_merged_v1/{path}')
        got = r and (r.lfs.sha256 if r.lfs else None)
        if r is None or (got and got != sha):
            bad += 1
            print('BAD', path, got, sha)
        n += 1
local = {p.relative_to(src).as_posix() for p in src.rglob('*') if p.is_file()}
missing = [p for p in local if f'ccip_merged_v1/{p}' not in remote]
print(f'ccip_merged_v1: {n} manifest files checked, {len(local)} local files, missing remote: {missing}')
print('ALL OK' if bad == 0 and not missing else f'{bad} BAD')
