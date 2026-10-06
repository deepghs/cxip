"""
用 witha 的画师标签重建原 train_labels.json / test_labels.json, 按原 split_labels.py (seed 42, 800 个测试画师)
和 csip_sub.py (按画师名排序取前 100 个) 重新切分, 和 HF CSIP v2.1 卡片上的测试集指纹对比:
  10,360 张图, 1,672,116 个正样本对 (sum n_i*(n_i-1)/2)
原脚本的筛选条件未知, 这里枚举几种合理的组合; 指纹完全一致的组合即为原始划分。

python repro/data/match_witha_split.py            # 搜索
python repro/data/match_witha_split.py --write <variant>   # 写出该组合的 label 文件
"""
import argparse
import itertools
import json
import os
import random
from collections import Counter
from pathlib import Path

DATA = Path(os.environ.get('REPRO_ROOT', '/data/neko/cxip_repro')) / 'data'
TARGET = (10360, 1672116)


def load_meta(images='witha'):
    """images='witha': witha 全部图片和它的 split; 'wsfull': 只取 animetimm ws-full 里的图片, 用 ws-full 的 split
    (原目录名 danbooru-w640-ws-full-artist, 图片可能来自 ws-full)"""
    rows = {}
    for p in sorted((DATA / 'witha_meta').glob('*.json')):
        split = p.name.split('_')[0]
        with open(p) as f:
            for d in json.load(f):
                rows[d['id']] = (split, d['artist_tags'])
    if images == 'wsfull':
        with open(DATA / 'wsfull_id_split.json') as f:
            wsfull = {int(k): v for k, v in json.load(f).items()}
        rows = {i: (wsfull[i], a) for i, (_, a) in rows.items() if i in wsfull}
    return rows


def build_labels(rows, train_splits, test_splits, multi, min_count):
    '''返回 (train_labels, test_labels), 与原 /nfs/danbooru-w640-ws-full-artist/{train,test}_labels.json 同格式'''
    labels = {}
    for i, (split, artists) in rows.items():
        if split not in train_splits and split not in test_splits:
            continue
        if not artists:
            continue
        if len(artists) > 1:
            if multi == 'drop':
                continue
            artist = artists[0]
        else:
            artist = artists[0]
        labels[i] = (split, artist)
    counts = Counter(a for _, a in labels.values())
    train, test = {}, {}
    for i, (split, artist) in labels.items():
        if counts[artist] < min_count:
            continue
        (train if split in train_splits else test)[f'{i}.webp'] = artist
    return train, test


def split_and_sub(train, test, n_test=800, seed=42, n_sub=100):
    '''原 split_labels.py + csip_sub.py 的逻辑'''
    merged = {**train, **test}
    all_classes = sorted(set(merged.values()))
    test_classes = set(random.Random(seed).sample(all_classes, n_test))
    test_split = {k: v for k, v in merged.items() if v in test_classes}
    sub_classes = set(sorted(test_classes)[:n_sub])  # pd.Categorical 的 codes 按类别名排序
    sub = {k: v for k, v in test_split.items() if v in sub_classes}
    return merged, test_split, sub


def fingerprint(sub):
    c = Counter(sub.values())
    return len(sub), sum(n * (n - 1) // 2 for n in c.values())


VARIANTS = {
    f'img={img}|tr={"+".join(tr)}|te={"+".join(te)}|multi={multi}|min={k}': (img, tr, te, multi, k)
    for img, (tr, te), multi, k in itertools.product(
        ['witha', 'wsfull'],
        [(('train',), ('test',)), (('train',), ('test', 'val')), (('train', 'val'), ('test',))],
        ['drop', 'first'],
        [1, 2, 5, 10, 20, 30, 50, 100],
    )
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', default=None, help='写出该组合的 label 文件到 --out')
    parser.add_argument('--out', type=Path, default=DATA / 'danbooru-w640-ws-full-artist-witha')
    args = parser.parse_args()

    metas = {img: load_meta(img) for img in sorted({v[0] for v in VARIANTS.values()})}
    print({img: len(rows) for img, rows in metas.items()}, flush=True)
    if args.write:
        img, tr, te, multi, k = VARIANTS[args.write]
        rows = metas[img]
        train, test = build_labels(rows, tr, te, multi, k)
        merged, test_split, sub = split_and_sub(train, test)
        test_classes = set(test_split.values())
        train_split = {i: a for i, a in merged.items() if a not in test_classes}
        args.out.mkdir(parents=True, exist_ok=True)
        for name, d in [('train_labels', train), ('test_labels', test), ('train_split', train_split),
                        ('test_split', test_split), ('test_split_100', sub)]:
            with open(args.out / f'{name}.json', 'w') as f:
                json.dump(d, f, ensure_ascii=False)
        print(args.write, 'fingerprint', fingerprint(sub), 'target', TARGET, '->', args.out)
        return

    results = []
    for name, (img, tr, te, multi, k) in VARIANTS.items():
        train, test = build_labels(metas[img], tr, te, multi, k)
        merged, _, sub = split_and_sub(train, test)
        fp = fingerprint(sub)
        results.append((fp == TARGET, abs(fp[0] - TARGET[0]), name, len(merged), len(set(merged.values())), fp))
        print(('MATCH ' if fp == TARGET else '      ') + f'{name}: images={len(merged)} artists={len(set(merged.values()))} '
              f'test100={fp}', flush=True)
    best = sorted(results, key=lambda r: (not r[0], r[1]))[:5]
    print('closest:', *best, sep='\n  ')


if __name__ == '__main__':
    main()
