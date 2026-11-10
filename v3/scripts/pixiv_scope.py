"""方案 (a) 的规模粗估: 真实 Pixiv 图 (只用 pixiv_index 元数据, 不用 s1 语料), 只打印计数.

方法:
  - pixiv_index tables/{1..4} 的 artworks (去重), 只留 type == illust 且 illust_ai_type != 2 (非 AI);
  - 去掉已经在 Danbooru 快照里出现过的作品 (按 pixiv_id), 余下 = "不在 Danbooru 上的作品";
  - Pixiv 账号 -> Danbooru 画师映射: pixiv_meta.shared_table + map_accounts (在过滤后的 Danbooru 帖子上);
  - 三个账号范围: 已映射到任一 Danbooru 画师的账号; 纯 Pixiv 账号 (>= 30 件不在 Danbooru 的作品且从没有作品上过
    Danbooru); B1 画师 (bench2 splits.json 的 b1, 仅内部使用, 不打印) 的账号 = B3-P1 候选池;
  - 每个范围打印: 账号数, 作品数, 全部页数的图数, 每件作品至多 2 页的图数, 只取第 1 页的图数 (不含 ugoira).
下载体积是另行按每张缩放后大小估出来的, 本脚本不算.
用法: CUDA_VISIBLE_DEVICES= python -m v3.scripts.pixiv_scope
"""
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from v3.data.build_danbooru_artist_v2 import filter_posts
from v3.data.pixiv_meta import PIXIV_INDEX, SNAP, map_accounts, shared_table

SPLITS = '/scratch/cxip_v4/data/bench2_v1/splits.json'


def main():
    parts = [PIXIV_INDEX / 'tables' / str(i) for i in (1, 2, 3, 4)]
    aw = pd.concat([pd.read_parquet(p / 'artworks.parquet',
                                    columns=['id', 'type', 'illust_ai_type', 'user_id', 'created_at'])
                    for p in parts]).drop_duplicates('id')
    print('index artworks', len(aw), 'types', aw.type.value_counts().to_dict(),
          'ai', aw.illust_ai_type.value_counts().to_dict())
    im = pd.concat([pd.read_parquet(p / 'images.parquet', columns=['id', 'artwork_id', 'order', 'is_ugoira', 'url'])
                    for p in parts]).drop_duplicates('id')
    print('index images', len(im), 'with url', int(im.url.notna().sum()))
    snap = pq.read_table(SNAP, columns=['id', 'pixiv_id', 'tag_string_artist', 'tag_count_artist', 'is_deleted',
                                        'file_ext', 'tag_string', 'parent_id']).to_pandas()
    snap_ids = set(snap.pixiv_id.dropna().astype('int64'))
    filt = filter_posts(snap)
    filt = filt[filt.pixiv_id.notna()]
    mp = map_accounts(shared_table(filt, aw))
    ok = aw[(aw.type == 'illust') & (aw.illust_ai_type != 2)].copy()
    ok['on_db'] = ok.id.isin(snap_ids)
    free = ok[~ok.on_db]
    users_on_db = set(aw.loc[aw.id.isin(snap_ids), 'user_id'])
    nw = free.groupby('user_id').size()
    pure = nw[(nw >= 30) & ~nw.index.isin(users_on_db)].index
    mapped = set(mp.user_id)
    b1 = set(json.load(open(SPLITS))['b1'])
    b1_users = set(mp[mp.artist.isin(b1)].user_id)
    img = im[~im.is_ugoira.astype(bool)]

    def scope(users, tag):
        w = free[free.user_id.isin(users)]
        i = img[img.artwork_id.isin(set(w.id))]
        per = i.groupby('artwork_id').size()
        p2 = int(np.minimum(per, 2).sum())
        print(f'{tag}: accounts {len(set(w.user_id))}, works {len(w)}, images all pages {len(i)}, '
              f'<=2 pages/work {p2}, page-1 only {int((per > 0).sum())}')

    scope(mapped, 'mapped accounts (any Danbooru artist), works not on Danbooru')
    scope(set(pure), 'pure-Pixiv accounts (>=30 works not on Danbooru)')
    scope(b1_users, 'accounts of B1 artists (B3-P1 pool)')
    print('mapped accounts', len(mapped), 'artists', mp.artist.nunique(),
          'B1 artists with account', mp[mp.artist.isin(b1)].artist.nunique())


if __name__ == '__main__':
    main()
