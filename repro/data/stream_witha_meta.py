"""
从 hub.deepghs.org deepghs/danbooru-wdtagger-v4-w640-ws-full-witha (WD v4 加了画师标签的版本, 推测是原
/nfs/danbooru-w640-ws-full-artist 的标签来源) 流式读取每个 tar, 只保留样本的 json (id / artist_tags / character_tags),
图片直接丢弃, 不落盘。每个 tar 输出一个小 json: data/witha_meta/{split}_{nnnnn}.json, 可断点续跑。

python repro/data/stream_witha_meta.py --jobs 8
"""
import argparse
import json
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import requests
from huggingface_hub import HfApi

REPO = 'deepghs/danbooru-wdtagger-v4-w640-ws-full-witha'
OUT = Path(os.environ.get('REPRO_ROOT', '/data/neko/cxip_repro')) / 'data/witha_meta'


def stream_one(path):
    split, name = path.split('/')
    out = OUT / f'{split}_{name[:-4]}.json'
    if out.exists():
        return path, -1
    url = f"{os.environ['KHUB_ENDPOINT']}/datasets/{REPO}/resolve/main/{path}"
    headers = {'Authorization': f"Bearer {os.environ['KHUB_TOKEN']}"}
    rows = []
    with requests.get(url, headers=headers, stream=True, timeout=600) as r:
        r.raise_for_status()
        r.raw.decode_content = True
        with tarfile.open(fileobj=r.raw, mode='r|') as tf:
            for m in tf:
                if m.name.endswith('.json'):
                    d = json.load(tf.extractfile(m))
                    rows.append({'id': d['id'], 'artist_tags': d.get('artist_tags', []),
                                 'character_tags': d.get('character_tags', [])})
    tmp = out.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(rows, f)
    tmp.rename(out)
    return path, len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    api = HfApi(endpoint=os.environ['KHUB_ENDPOINT'], token=os.environ['KHUB_TOKEN'])
    # 仓库里每个文件在子目录 danbooru-wdtagger-v4-w640-ws-full-witha/ 下还有一份重复, 只读顶层
    tars = sorted(f.path for f in api.list_repo_tree(REPO, repo_type='dataset', recursive=True)
                  if f.path.endswith('.tar') and f.path.count('/') == 1)
    print(f'{len(tars)} tars', flush=True)
    done = 0
    with ProcessPoolExecutor(args.jobs) as ex:
        futs = [ex.submit(stream_one, t) for t in tars]
        for fut in as_completed(futs):
            path, n = fut.result()
            done += 1
            print(f'[{done}/{len(tars)}] {path}: {n}', flush=True)


if __name__ == '__main__':
    main()
