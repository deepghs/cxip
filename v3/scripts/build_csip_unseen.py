"""csip_unseen_v1: 两边都没训练过的画师组成的 CSIP 测试集, 用于和第三方模型 (Kaloscope 3.0) 公平对比。

test100_0918 / csip_eval_public 的画师几乎都在 Kaloscope 的 44,129 个训练画师里 (97/100, 86/88), 对它不是留出集。
这里从 Danbooru 2026-07 快照 (nyanko-devs/danbooru2026 posts-snapshot) 选画师:
  - 单画师帖, 未删除, jpg/png/webp, 不带 ai-generated / ai-assisted
  - 画师不在 Kaloscope class_mapping.csv (= 其训练集画师), 也不在 cxip-danbooru-artist/v1 classes.json (CSIP v3 数据全集)
  - 作品族去重: 每个 (画师, parent_id 或自身 id) 只留一张, 去重后 >= 30 张
  - 随机抽 100 位画师 (seed 0), 每人最多 40 张
图片从本地 danbooru-full-4mpixel-v1 shard 读 (train_index: id -> shard), 与 test100 一致缩放到短边 <= 640, webp q95。
输出 /data/neko/cxip_v3/data/csip_unseen_v1/{images/<label>/<id>.webp, index.parquet, manifest.json}。
用法: python -m v3.scripts.build_csip_unseen
"""
import csv
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

SNAP = '/scratch/anime_tagger/dl/nyanko_danbooru2026/posts-snapshot.parquet'
SHARDS = Path('/scratch/anime_tagger/shards')
KAL_CLASSES = '/data/neko/ext_models/kaloscope3_preview/hf/v1-artist-classifier/class_mapping.csv'
V3_CLASSES = '/data/neko/cxip_v3/data/danbooru-artist-v1/classes.json'
OUT = Path('/data/neko/cxip_v3/data/csip_unseen_v1')
N_ARTISTS, MIN_IMGS, MAX_IMGS, SEED = 100, 30, 40, 0
AI_TAGS = {'ai-generated', 'ai-assisted'}


def candidates():
    t = pq.read_table(SNAP, columns=['id', 'tag_string_artist', 'tag_count_artist', 'is_deleted', 'file_ext', 'parent_id',
                                     'tag_string', 'rating', 'created_at']).to_pandas()
    t = t[(t.tag_count_artist == 1) & (t.is_deleted.fillna(False) == False) & t.file_ext.isin(['jpg', 'jpeg', 'png', 'webp'])]
    t = t[~t.tag_string.str.split().map(lambda s: bool(AI_TAGS & set(s)))]
    local = pq.read_table(SHARDS / 'train_index.parquet').to_pandas()
    local = local[local.ds == 0].set_index('id')['shard']
    t = t[t.id.isin(local.index)].copy()
    t['shard'] = local.reindex(t.id).values
    kal = {r['class_name'].strip().lower() for r in csv.DictReader(open(KAL_CLASSES))}
    v3 = {str(c['name']).strip().lower() for c in json.load(open(V3_CLASSES))}
    a = t.tag_string_artist.str.lower()
    t = t[~a.isin(kal) & ~a.isin(v3)]
    t['family'] = t.parent_id.fillna(t.id).astype(np.int64)
    t = t.sample(frac=1.0, random_state=SEED).drop_duplicates(['tag_string_artist', 'family'])
    n = t.tag_string_artist.value_counts()
    return t[t.tag_string_artist.isin(n[n >= MIN_IMGS].index)], len(kal), len(v3)


def _tar_size(field):
    if field[0] & 0x80:                                   # base-256 (GNU, > 8 GiB members)
        return int.from_bytes(field[1:], 'big')
    return int(field.split(b'\0', 1)[0].strip() or b'0', 8)


def read_members(shard, ids):
    """只读 512 字节头逐个跳 (pread + FADV_RANDOM): tarfile 的小读 + 内核预读会把整个 1 GB shard 读一遍。"""
    path = SHARDS / 'danbooru-full-4mpixel-v1/wds/train' / f'train-{int(shard):06d}.tar'
    want = {f'{i}.webp': i for i in ids}
    out = {}
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        off, end, long_name = 0, os.fstat(fd).st_size, None
        while off + 512 <= end and len(out) < len(want):
            h = os.pread(fd, 512, off)
            if not h.strip(b'\0'):
                break
            size, kind = _tar_size(h[124:136]), h[156:157]
            data_off = off + 512
            if kind == b'L':                               # GNU 长文件名: 数据是下一个成员的名字
                long_name = os.pread(fd, size, data_off).split(b'\0', 1)[0].decode()
            elif kind in (b'x', b'g'):                     # pax 头, 跳过
                pass
            else:
                name = long_name or h[:100].split(b'\0', 1)[0].decode()
                long_name = None
                if name in want:
                    out[want[name]] = os.pread(fd, size, data_off)
            off = data_off + (size + 511) // 512 * 512
    finally:
        os.close(fd)
    return out


def save(row, data):
    im = Image.open(io.BytesIO(data))
    im.load()
    w, h = im.size
    if min(w, h) > 640:
        s = 640 / min(w, h)
        im = im.resize((round(w * s), round(h * s)), Image.BICUBIC)
    p = OUT / 'images' / f'{row.label:03d}' / f'{row.id}.webp'
    p.parent.mkdir(parents=True, exist_ok=True)
    im.save(p, 'WEBP', quality=95)     # 透明通道保留, 两个模型各自按白底合成
    return str(p)


def main():
    if (OUT / 'manifest.json').exists():
        sys.exit(f'{OUT} already exists; bump the version instead of overwriting')
    t, n_kal, n_v3 = candidates()
    pool = sorted(t.tag_string_artist.unique())
    rng = np.random.default_rng(SEED)
    artists = sorted(rng.choice(pool, N_ARTISTS, replace=False).tolist())
    sel = (t[t.tag_string_artist.isin(artists)].groupby('tag_string_artist', group_keys=False)
           .apply(lambda g: g.head(MAX_IMGS)).reset_index(drop=True))
    sel['label'] = sel.tag_string_artist.map({a: i for i, a in enumerate(artists)})
    print(f'pool {len(pool)} artists; selected {len(artists)} artists, {len(sel)} images', flush=True)
    data = {}
    with ThreadPoolExecutor(32) as ex:
        for got in ex.map(lambda kv: read_members(kv[0], kv[1].id.tolist()), sel.groupby('shard')):
            data.update(got)
    missing = sel[~sel.id.isin(data)]
    print('missing from shards:', len(missing), flush=True)
    sel = sel[sel.id.isin(data)].reset_index(drop=True)
    with ThreadPoolExecutor(32) as ex:
        sel['path'] = list(ex.map(lambda r: save(r, data[r.id]), sel.itertuples()))
    sel['sample_key'] = sel.path.map(lambda p: str(Path(p).relative_to(OUT)))
    sel = sel.rename(columns={'tag_string_artist': 'label_name'})
    sel[['path', 'label_name', 'label', 'sample_key', 'id', 'family', 'rating', 'created_at']].to_parquet(OUT / 'index.parquet')
    per = sel.groupby('label').size()
    manifest = dict(version='v1', seed=SEED, snapshot='nyanko-devs/danbooru2026 posts-snapshot (2026-07-13)',
                    images_from='danbooru-full-4mpixel-v1 (local shards)', n_artists=int(sel.label.nunique()),
                    n_images=len(sel), per_artist=dict(min=int(per.min()), median=float(per.median()), max=int(per.max())),
                    candidate_pool=len(pool), excluded_kaloscope_classes=n_kal, excluded_v3_artists=n_v3,
                    rules=['single-artist, not deleted, jpg/png/webp, no ai-generated/ai-assisted',
                           'artist not in Kaloscope3.0-preview class_mapping.csv (its training artists)',
                           'artist not in cxip-danbooru-artist/v1 classes.json (CSIP v3 data)',
                           'one image per (artist, parent_id-or-id) family', f'>= {MIN_IMGS} images after dedupe',
                           f'{N_ARTISTS} artists sampled, <= {MAX_IMGS} images each', 'short side <= 640, webp q95'],
                    rating_counts=sel.rating.value_counts().to_dict())
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=1, default=str))
    print(json.dumps(manifest, indent=1, default=str))


if __name__ == '__main__':
    main()
