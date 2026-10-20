"""
重建 CSIP 训练/测试集 danbooru-w640-ws-full-artist (原 /nfs/danbooru-w640-ws-full-artist 已丢失)

- 图片: animetimm/danbooru-wdtagger-v4-w640-ws-full (webdataset, {id}.webp)
- 画师: isek-ai/danbooru-tags-2024 (按 post id), 缺失的用 deepghs/danbooru_newest_index 补
- 只保留单画师图片, 画师图片数 >= --min-count (默认 30, 按 HF 卡片 test_split_100 的 10360 张反推)
- train_labels.json = wd 的 train split, test_labels.json = wd 的 val+test split, 之后跑原版 scripts/data/split_labels.py (800 test 类, seed 42)
  和 scripts/tagger/csip_sub.py 的逻辑 (test_split_100.json)
- 图片解包到 {out}/train/{id}.webp (原配置 train/test 都用 img_root=.../train)

用法: python experiments/data_prep/build_danbooru_artist.py --out data/danbooru-w640-ws-full-artist --workers 48
"""
import argparse
import glob
import json
import os
import subprocess
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download, hf_hub_download

WD_REPO = 'animetimm/danbooru-wdtagger-v4-w640-ws-full'


def list_tar(path):
    out = subprocess.run(['tar', 'tf', path], capture_output=True, text=True, check=True).stdout.split()
    return path, [int(n[:-5]) for n in out if n.endswith('.webp')]


def extract_tar(args):
    path, ids, dst = args
    ids = set(ids)
    n = 0
    with tarfile.open(path) as tf:
        for m in tf:
            if m.name.endswith('.webp') and int(m.name[:-5]) in ids:
                target = os.path.join(dst, m.name)
                if not os.path.exists(target):
                    with open(target + '.tmp', 'wb') as f:
                        f.write(tf.extractfile(m).read())
                    os.replace(target + '.tmp', target)
                n += 1
    return path, n


def load_artists():
    d = snapshot_download('isek-ai/danbooru-tags-2024', repo_type='dataset', allow_patterns=['data/*.parquet'])
    df = pd.concat([pq.read_table(f, columns=['id', 'artist', 'tag_count_artist']).to_pandas()
                    for f in sorted(glob.glob(f'{d}/data/*.parquet'))])
    df = df.rename(columns={'tag_count_artist': 'n_artist'})
    df['artist'] = df['artist'].str.replace(' ', '_')  # isek 把 tag 里的 _ 换成了空格, 多画师用 ', ' 分隔
    newest = pq.read_table(hf_hub_download('deepghs/danbooru_newest_index', 'records.parquet', repo_type='dataset'),
                           columns=['id', 'tag_string_artist', 'tag_count_artist']).to_pandas()
    newest = newest.rename(columns={'tag_string_artist': 'artist', 'tag_count_artist': 'n_artist'})
    newest = newest[~newest['id'].isin(df['id'])]
    return pd.concat([df, newest], ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('data/danbooru-w640-ws-full-artist'))
    parser.add_argument('--min-count', type=int, default=30)
    parser.add_argument('--n-test-classes', type=int, default=800)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--no-extract', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    root = snapshot_download(WD_REPO, repo_type='dataset', allow_patterns=['meta.json', 'README.md'])
    tars = sorted(glob.glob(f'{root}/*/*.tar'))
    if len(tars) < 237:  # 只有元数据时, 先补全 tar (缓存里一般已有)
        root = snapshot_download(WD_REPO, repo_type='dataset')
        tars = sorted(glob.glob(f'{root}/*/*.tar'))
    print(f'{len(tars)} tars')

    with ProcessPoolExecutor(args.workers) as ex:
        tar_ids = dict(ex.map(list_tar, tars))
    rows = [(i, p, Path(p).parent.name) for p, ids in tar_ids.items() for i in ids]
    wd = pd.DataFrame(rows, columns=['id', 'tar', 'split'])
    print(wd['split'].value_counts().to_dict())

    art = load_artists()
    wd = wd.merge(art, on='id', how='left')
    print(f'no artist info: {wd["artist"].isna().sum()}')
    wd = wd[wd['n_artist'] == 1]
    counts = wd['artist'].value_counts()
    keep = counts[counts >= args.min_count].index
    wd = wd[wd['artist'].isin(keep)].sort_values('id')
    print(f'single-artist >= {args.min_count}: {len(wd)} images, {len(keep)} artists')
    wd[['id', 'artist', 'split', 'tar']].to_parquet(args.out / 'index.parquet', index=False)

    key = wd['id'].astype(str) + '.webp'
    train_labels = dict(zip(key[wd['split'] == 'train'], wd['artist'][wd['split'] == 'train']))
    test_labels = dict(zip(key[wd['split'] != 'train'], wd['artist'][wd['split'] != 'train']))
    for name, d in [('train_labels.json', train_labels), ('test_labels.json', test_labels)]:
        with open(args.out / name, 'w') as f:
            json.dump(d, f, ensure_ascii=False)

    repo = Path(__file__).resolve().parents[2]
    subprocess.run(['python', str(repo / 'scripts/data/split_labels.py'), '--inputs', str(args.out / 'train_labels.json'),
                    str(args.out / 'test_labels.json'), '--output_dir', str(args.out),
                    '--n_test_classes', str(args.n_test_classes), '--seed', str(args.seed)], check=True)

    # scripts/tagger/csip_sub.py: pd.Categorical 按类名排序, 取前 100 类
    with open(args.out / 'test_split.json') as f:
        test = json.load(f)
    df = pd.DataFrame(list(test.items()), columns=['name', 'label'])
    df['ids'] = pd.Categorical(df['label']).codes
    sub = df[df['ids'] < 100].set_index('name')['label'].to_dict()
    with open(args.out / 'test_split_100.json', 'w') as f:
        json.dump(sub, f, ensure_ascii=False, indent=2)
    n = pd.Series(list(sub.values())).value_counts().values
    print(f'test_split_100: {len(sub)} images, pos pairs {(n * (n - 1) // 2).sum()}, all pairs {len(sub) * (len(sub) - 1) // 2}')

    if not args.no_extract:
        dst = args.out / 'train'
        dst.mkdir(exist_ok=True)
        jobs = [(t, g['id'].tolist(), str(dst)) for t, g in wd.groupby('tar')]
        total = 0
        with ProcessPoolExecutor(args.workers) as ex:
            for i, (t, n) in enumerate(ex.map(extract_tar, jobs)):
                total += n
                if i % 20 == 0:
                    print(f'extracted {i + 1}/{len(jobs)} tars, {total} images', flush=True)
        print(f'extracted {total} images -> {dst}')


if __name__ == '__main__':
    main()
