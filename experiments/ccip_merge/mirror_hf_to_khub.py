"""
把 HF 上 deepghs/character_similarity 原样镜像为 KohakuHub 本地仓库 (khub 上原来只是 HF 回源, 不可写)
    source env.sh; python experiments/ccip_merge/mirror_hf_to_khub.py
已下载的同名文件做硬链接复用, 缺的从 HF 下载; LFS 文件按 HF 的 sha256 校验; 远端已有同大小文件跳过
"""
import hashlib
import os
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

REPO = 'deepghs/character_similarity'
ROOT = Path('/data/neko/cxip_repro/data/zips')
MIRROR = ROOT / 'character_similarity_mirror'
KNOWN = [ROOT / 'character_similarity', ROOT]

hf = HfApi(token=os.environ.get('HF_TOKEN'))
kh = HfApi(endpoint=os.environ['KHUB_ENDPOINT'], token=os.environ['KHUB_TOKEN'])

info = hf.dataset_info(REPO, files_metadata=True)
print('hf sha', info.sha, flush=True)
MIRROR.mkdir(parents=True, exist_ok=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(64 << 20), b''):
            h.update(b)
    return h.hexdigest()


files = []
for s in info.siblings:
    dst = MIRROR / s.rfilename
    if not dst.exists():
        for d in KNOWN:
            if (d / s.rfilename).is_file() and (d / s.rfilename).stat().st_size == s.size:
                os.link(d / s.rfilename, dst)
                break
        else:
            hf_hub_download(REPO, s.rfilename, repo_type='dataset', revision=info.sha, local_dir=MIRROR)
    assert dst.stat().st_size == s.size, (s.rfilename, dst.stat().st_size, s.size)
    if s.lfs:
        got = sha256(dst)
        assert got == s.lfs.sha256, (s.rfilename, got, s.lfs.sha256)
    print('ok', s.rfilename, s.size, flush=True)
    files.append(s.rfilename)

kh.create_repo(REPO, repo_type='dataset', private=False, exist_ok=True)
remote = {f.path: getattr(f, 'size', None) for f in kh.list_repo_tree(REPO, repo_type='dataset', recursive=True)}
small = [f for f in files if (MIRROR / f).stat().st_size < 100 << 20]
big = [f for f in files if f not in small]
for batch in [small] + [[f] for f in big]:
    batch = [f for f in batch if remote.get(f) != (MIRROR / f).stat().st_size]
    if not batch:
        continue
    ops = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=str(MIRROR / f)) for f in batch]
    for attempt in range(5):
        try:
            t0 = time.time()
            kh.create_commit(REPO, ops, repo_type='dataset',
                             commit_message=f'mirror huggingface.co/datasets/{REPO}@{info.sha[:8]}: {", ".join(batch)}'[:200])
            print('uploaded', batch, f'{time.time() - t0:.0f}s', flush=True)
            break
        except Exception as e:
            print('failed', batch, repr(e)[:400], flush=True)
            time.sleep(30 * (attempt + 1))
    else:
        raise SystemExit('upload failed')
print('mirror done', flush=True)
