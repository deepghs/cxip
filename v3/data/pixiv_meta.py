"""Pixiv s1 image -> artwork -> account tables for CSIP v4 (feeds the benchmark splits and pixiv-artist-v1).

Unused in v4: the s1 source is an AI-image corpus (see docs/csip_v4_p0_baseline_zh.md §6.1). Pixiv was dropped
from CSIP v4 at the P0 checkpoint (2026-11-01); the s1 / pixiv-artist-v1 data it produced has been deleted.
Kept only as a record of what was run.

works.parquet    one row per usable s1 image (type illust, illust_ai_type != 2, not ugoira).
accounts.parquet one row per account that owns at least one usable s1 image; maps the account to a Danbooru artist
                 (current tag name) when the artworks they share on Danbooru point to exactly one artist.

    python -m v3.data.pixiv_meta prepare
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from v3.data import dates
from v3.data.build_danbooru_artist_v2 import filter_posts
from v3.data.fetch_pixiv_s1 import image_table

V4 = Path('/scratch/cxip_v4')
SNAP = '/scratch/anime_tagger/dl/nyanko_danbooru2026/posts-snapshot.parquet'
PIXIV_INDEX = Path('/scratch/pixiv_probe/pixiv_index')
WORK = V4 / 'data/pixiv-artist-v1-work'
MIN_SHARED = 3
_COMIC = {'漫画', '4コマ'}
_MONO = {'モノクロ', '白黒'}


def pixiv_medium(tags) -> str:
    """'comic' if a tag is exactly 漫画 / 4コマ, else 'mono' if exactly モノクロ / 白黒, else 'color'."""
    s = set() if tags is None else set(tags)
    if s & _COMIC:
        return 'comic'
    return 'mono' if s & _MONO else 'color'


def shared_table(posts: pd.DataFrame, artworks: pd.DataFrame) -> pd.DataFrame:
    """DataFrame[user_id, artist, n_shared]: distinct artworks per (account, danbooru artist).

    ``posts``: filtered Danbooru posts (pixiv_id, tag_string_artist); ``artworks``: pixiv index (id, user_id)."""
    p = posts[['pixiv_id', 'tag_string_artist']].dropna()
    p = p[p.tag_string_artist.str.len() > 0].astype({'pixiv_id': 'int64'})
    m = p.merge(artworks[['id', 'user_id']].rename(columns={'id': 'pixiv_id'}), on='pixiv_id', how='inner')
    g = m.drop_duplicates(['user_id', 'tag_string_artist', 'pixiv_id']).groupby(['user_id', 'tag_string_artist']).size()
    return g.rename('n_shared').reset_index().rename(columns={'tag_string_artist': 'artist'})


def map_accounts(shared: pd.DataFrame, min_shared: int = MIN_SHARED) -> pd.DataFrame:
    """Accounts with exactly one artist having n_shared >= min_shared -> DataFrame[user_id, artist].
    Several accounts may map to one artist; an account with two qualifying artists is not mapped."""
    q = shared[shared.n_shared >= min_shared]
    cnt = q.groupby('user_id').artist.transform('size')
    return q[cnt == 1][['user_id', 'artist']].reset_index(drop=True)


def build_tables(img: pd.DataFrame, images: pd.DataFrame, artworks: pd.DataFrame, snap_pixiv_ids: set,
                 shared: pd.DataFrame, min_shared: int = MIN_SHARED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(works, accounts).

    img: s1 image table (image_id, tar, offset, size, sha256); images: pixiv index images (id, artwork_id, order,
    is_ugoira); artworks: FULL pixiv index (id, user_id, type, illust_ai_type, x_restrict, created_at, tags - tags
    only needed for s1 artworks); snap_pixiv_ids: every pixiv_id on any Danbooru snapshot post."""
    snap_ids = pd.Series(sorted(snap_pixiv_ids), dtype='int64')
    w = img.merge(images[['id', 'artwork_id', 'order', 'is_ugoira']].rename(columns={'id': 'image_id'}), on='image_id')
    w = w[~w.is_ugoira.astype(bool)]
    w = w.merge(artworks[['id', 'user_id', 'type', 'illust_ai_type', 'x_restrict', 'created_at', 'tags']]
                .rename(columns={'id': 'artwork_id'}), on='artwork_id')
    w = w[(w.type == 'illust') & (w.illust_ai_type != 2)].copy()
    ts = pd.to_datetime(w.created_at, unit='s', utc=True)
    ok = ts.notna().to_numpy()
    w['t_year'] = dates.to_year(ts)                                       # NaN where the date is missing
    w['date_day'] = pd.array(np.where(ok, dates.to_day(ts.fillna(pd.Timestamp('2000-01-01', tz='UTC'))), -1), dtype='Int64')
    w.loc[~ok, 'date_day'] = pd.NA
    w['date_source'] = 'pixiv'
    w['medium'] = [pixiv_medium(t) for t in w.tags]
    w['on_danbooru'] = w.artwork_id.isin(set(snap_ids)).to_numpy()
    w = w.rename(columns={'order': 'page'})
    works = w[['image_id', 'artwork_id', 'page', 'user_id', 't_year', 'date_day', 'date_source', 'x_restrict', 'medium',
               'tar', 'offset', 'size', 'on_danbooru']].sort_values('image_id').reset_index(drop=True)

    # ---- accounts
    free = works[~works.on_danbooru]
    g = free.groupby('user_id').agg(n_works=('artwork_id', 'nunique'), n_images=('image_id', 'size'),
                                    t_first=('t_year', 'min'), t_last=('t_year', 'max'))
    acc = pd.DataFrame({'user_id': np.sort(works.user_id.unique())}).merge(g.reset_index(), on='user_id', how='left')
    acc[['n_works', 'n_images']] = acc[['n_works', 'n_images']].fillna(0).astype('int64')
    acc['span'] = acc.t_last - acc.t_first
    mp = map_accounts(shared, min_shared)
    acc = acc.merge(mp, on='user_id', how='left')
    # n_shared: the mapped artist's count; for unmapped accounts the best candidate's (0 if none)
    sh = shared.merge(mp.assign(_m=True), on=['user_id', 'artist'], how='left')
    sh['_m'] = sh._m.fillna(False).astype(bool)
    best = sh.sort_values(['user_id', '_m', 'n_shared'], ascending=[True, False, False]).drop_duplicates('user_id')
    acc = acc.merge(best[['user_id', 'n_shared']], on='user_id', how='left')
    acc['n_shared'] = acc.n_shared.fillna(0).astype('int64')
    acc['artist'] = acc.artist.astype(object).where(acc.artist.notna(), None)
    on_db_users = set(artworks.loc[artworks.id.isin(snap_ids), 'user_id'])
    acc['pixiv_only'] = ~acc.user_id.isin(on_db_users)
    acc = acc[['user_id', 'artist', 'n_shared', 'n_works', 'n_images', 't_first', 't_last', 'span', 'pixiv_only']]
    return works, acc.reset_index(drop=True)


def _load_index(img_ids: np.ndarray):
    parts = [PIXIV_INDEX / 'tables' / str(i) for i in (1, 2, 3, 4)]
    images = pd.concat([pd.read_parquet(p / 'images.parquet', columns=['id', 'artwork_id', 'order', 'is_ugoira'])
                        for p in parts]).drop_duplicates('id')
    images = images[images.id.isin(img_ids)]
    cols = ['id', 'type', 'illust_ai_type', 'x_restrict', 'user_id', 'created_at']
    aw = pd.concat([pd.read_parquet(p / 'artworks.parquet', columns=cols) for p in parts]).drop_duplicates('id')
    need = set(images.artwork_id)
    tags = pd.concat([pd.read_parquet(p / 'artworks.parquet', columns=['id', 'tags']).pipe(lambda d: d[d.id.isin(need)])
                      for p in parts]).drop_duplicates('id')
    aw = aw.merge(tags, on='id', how='left')
    return images, aw


def prepare(out: Path = WORK):
    out = Path(out)
    if (out / 'works.parquet').exists() or (out / 'accounts.parquet').exists() or (out / 'manifest.json').exists():
        sys.exit(f'{out} already has output; bump the version instead of overwriting')
    t0 = time.time()
    img = image_table()
    images, aw = _load_index(img.image_id.to_numpy())
    print(f's1 images {len(img)}; in pixiv index {len(images)}; artworks (full index) {len(aw)} ({time.time() - t0:.0f}s)', flush=True)

    snap = pq.read_table(SNAP, columns=['id', 'pixiv_id', 'tag_string_artist', 'tag_count_artist', 'is_deleted',
                                        'file_ext', 'tag_string', 'parent_id']).to_pandas()
    pid = snap.pixiv_id.dropna().astype('int64')
    snap_ids = set(pid.unique().tolist())
    filt = filter_posts(snap)
    filt = filt[filt.pixiv_id.notna()]
    shared = shared_table(filt, aw)
    print(f'snapshot {len(snap)} posts, {len(snap_ids)} pixiv ids; filtered posts with pixiv_id {len(filt)}; '
          f'shared (account, artist) pairs {len(shared)} ({time.time() - t0:.0f}s)', flush=True)

    works, acc = build_tables(img, images, aw, snap_ids, shared)
    stats = _stats(img, images, works, acc, shared)
    out.mkdir(parents=True, exist_ok=True)
    works.to_parquet(out / 'works.parquet', index=False)
    acc.to_parquet(out / 'accounts.parquet', index=False)
    (out / 'prepare_stats.json').write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1), f'\ndone in {time.time() - t0:.0f}s', flush=True)


def _stats(img, images, works, acc, shared) -> dict:
    mp = map_accounts(shared)
    m = acc[acc.artist.notna()]
    po = acc.pixiv_only
    # cross-check: s1-only definition (no *usable s1* work of the account is on Danbooru)
    s1_only = ~acc.user_id.isin(set(works.loc[works.on_danbooru, 'user_id']))
    return dict(
        s1_images=int(len(img)), s1_images_in_index=int(len(images)), usable_images=int(len(works)),
        usable_artworks=int(works.artwork_id.nunique()), accounts=int(len(acc)),
        works_on_danbooru_images=int(works.on_danbooru.sum()), works_missing_date=int(works.t_year.isna().sum()),
        t_year_min=float(works.t_year.min()), t_year_max=float(works.t_year.max()),
        medium=works.medium.value_counts().to_dict(),
        mapped_accounts_all=int(len(mp)), mapped_artists_all=int(mp.artist.nunique()),
        mapped_accounts=int(len(m)), mapped_artists=int(m.artist.nunique()),
        pixiv_only_accounts=int(po.sum()),
        pixiv_only_n_works_ge30=int((po & (acc.n_works >= 30)).sum()),
        pixiv_only_n_works_ge30_s1only_def=int((s1_only & (acc.n_works >= 30)).sum()),
        pixiv_only_s1only_def=int(s1_only.sum()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prepare'])
    ap.add_argument('--out', default=str(WORK))
    a = ap.parse_args(argv)
    prepare(Path(a.out))


if __name__ == '__main__':
    main()
