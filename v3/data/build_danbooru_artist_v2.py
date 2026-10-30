"""Danbooru2026 per-post table for CSIP v4 (danbooru-artist-v2 prepare step).

Reads the 2026 Danbooru snapshot, keeps single-artist / not deleted / jpg|png|webp / non-AI posts that have a local
image, keeps one post per (artist, work family), corrects the date to the earliest plausible original-publication
date (twitter snowflake / pixiv index), maps 2024 (v1) artist tags to their current names by majority of shared post
ids, and flags the evaluation artists. Output: WORK/{posts.parquet, artists.parquet, prepare_stats.json}.

    python -m v3.data.build_danbooru_artist_v2 prepare
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from v3.data import dates

V4 = Path('/scratch/cxip_v4')
SNAP = '/scratch/anime_tagger/dl/nyanko_danbooru2026/posts-snapshot.parquet'
SHARDS = Path('/scratch/anime_tagger/shards')
SPLIT_0918 = Path('/data/neko/cxip_repro/data/danbooru-artist-split-0918')
PIXIV_INDEX = Path('/scratch/pixiv_probe/pixiv_index')
V1_PLAN = Path('/data/neko/cxip_v3/data/danbooru-artist-v1-work/plan.parquet')
KAL_CLASSES = '/data/neko/ext_models/kaloscope3_preview/hf/v1-artist-classifier/class_mapping.csv'
UNSEEN_V1 = Path('/data/neko/cxip_v3/data/csip_unseen_v1/index.parquet')
WORK = V4 / 'data/danbooru-artist-v2-work'
MIN_LOCAL = 10
AI_TAGS = {'ai-generated', 'ai-assisted'}
EXTS = ['jpg', 'jpeg', 'png', 'webp']
MEDIUM = [('comic', {'comic', '4koma'}), ('sketch', {'sketch', 'lineart'}), ('mono', {'monochrome', 'greyscale'})]
FLAGS = ['eval_0918_test', 'eval_test100', 'eval_public', 'eval_unseen_v1', 'in_v1', 'in_kaloscope']
EVAL_FLAGS = ['eval_0918_test', 'eval_test100', 'eval_public', 'eval_unseen_v1']


def filter_posts(snap: pd.DataFrame) -> pd.DataFrame:
    """Single artist, not deleted, jpg/jpeg/png/webp, no ai-generated / ai-assisted. Adds ``family`` (parent_id or id)."""
    t = snap[(snap.tag_count_artist == 1) & ~snap.is_deleted.astype('boolean').fillna(False).to_numpy(bool) & snap.file_ext.isin(EXTS)]
    ai = t.tag_string.fillna('').str.split().map(lambda s: bool(AI_TAGS.intersection(s)))
    t = t[~ai.to_numpy()].copy()
    t['family'] = t.parent_id.fillna(t.id).astype(np.int64)
    return t


def dedupe_families(posts: pd.DataFrame) -> pd.DataFrame:
    """One post per (artist, family): the smallest id."""
    return posts.sort_values('id').drop_duplicates(['artist', 'family'], keep='first')


def v1_name_map(v1: pd.DataFrame, snap: pd.DataFrame) -> dict[str, str]:
    """v1 artist name -> current name by majority over shared post ids (ties: most votes, then smallest name).
    A v1 name with no shared post (or no artist tag left on them) maps to itself."""
    cur = snap[['id', 'tag_string_artist']].drop_duplicates('id')
    m = v1[['id', 'artist']].drop_duplicates().merge(cur, on='id', how='inner')
    m['cur'] = m.tag_string_artist.fillna('').str.split()
    m = m.explode('cur').dropna(subset=['cur'])
    cnt = m.groupby(['artist', 'cur']).size().rename('n').reset_index().sort_values(['artist', 'n', 'cur'],
                                                                                    ascending=[True, False, True])
    best = cnt.drop_duplicates('artist').set_index('artist')['cur']
    return {a: best.get(a, a) for a in v1.artist.unique()}


def medium_of(general: set[str]) -> str:
    for name, s in MEDIUM:
        if general & s:
            return name
    return 'color'


def _pixiv_index() -> tuple[np.ndarray, np.ndarray]:
    a = pd.concat([pd.read_parquet(PIXIV_INDEX / 'tables' / str(i) / 'artworks.parquet', columns=['id', 'created_at'])
                   for i in (1, 2, 3, 4)]).drop_duplicates('id')
    return a.id.to_numpy(np.int64), a.created_at.to_numpy()


def prepare():
    if (WORK / 'posts.parquet').exists() or (WORK / 'manifest.json').exists():
        sys.exit(f'{WORK} already has prepared output; bump the version instead of overwriting')
    t0 = time.time()
    cols = ['id', 'created_at', 'source', 'pixiv_id', 'rating', 'image_width', 'image_height', 'tag_string',
            'tag_string_general', 'tag_string_character', 'tag_string_copyright', 'tag_string_artist',
            'tag_count_artist', 'is_deleted', 'file_ext', 'parent_id']
    snap = pq.read_table(SNAP, columns=cols).to_pandas()
    print(f'snapshot {len(snap)} rows ({time.time() - t0:.0f}s)', flush=True)

    # ---- v1 names -> current names
    v1 = pd.read_parquet(V1_PLAN, columns=['id', 'artist', 'split_0918', 'artist_0918_test', 'csip_eval_public_artist'])
    nm = v1_name_map(v1[['id', 'artist']], snap[snap.id.isin(v1.id)])
    renamed = {a: c for a, c in nm.items() if a != c}
    inv = Counter(nm.values())
    many = {c: [a for a, x in nm.items() if x == c] for c, n in inv.items() if n > 1}
    print(f'v1 names {len(nm)}; renamed {len(renamed)}; many-to-one targets {len(many)} '
          f'({sum(len(v) for v in many.values())} v1 names)', flush=True)
    for c, v in list(many.items())[:15]:
        print('   merge:', v, '->', c)

    def cur_names(mask_artists: pd.Series) -> set[str]:
        return {nm[a] for a in mask_artists.unique()}

    f_0918 = cur_names(v1[v1.artist_0918_test].artist)
    # test_split.json lists a few test artists absent from the v1 plan (no image in the 2024 index): map them by post id too
    ts = json.load(open(SPLIT_0918 / 'test_split.json'))
    ts = pd.DataFrame({'id': [int(k.split('.')[0]) for k in ts], 'artist': list(ts.values())})
    extra = v1_name_map(ts, snap[snap.id.isin(ts.id)])
    n_extra = len(set(extra.values()) - f_0918)
    f_0918 |= set(extra.values())
    print(f'0918 test: {len(f_0918)} current names (plan {len(cur_names(v1[v1.artist_0918_test].artist))}, '
          f'+{n_extra} from test_split.json not in the plan)', flush=True)
    f_100 = cur_names(v1[v1.split_0918 == 'test100'].artist)
    f_pub = cur_names(v1[v1.csip_eval_public_artist].artist)
    f_unseen = set(pd.read_parquet(UNSEEN_V1, columns=['label_name']).label_name)
    in_v1 = set(nm.values())
    kal = {r['class_name'].strip().lower() for r in csv.DictReader(open(KAL_CLASSES))}
    pre = {}
    for a, c in nm.items():
        pre.setdefault(c, []).append(a)

    # ---- filter, local images, family dedupe
    t = filter_posts(snap).rename(columns={'tag_string_artist': 'artist'})
    n_filtered = len(t)
    n_snapshot = dedupe_families(t).artist.value_counts()
    local = pq.read_table(SHARDS / 'train_index.parquet').to_pandas().sort_values(['id', 'ds']).drop_duplicates('id')
    t = t.merge(local, on='id', how='inner')                 # an id in both ds keeps ds 0 (lowest); none observed
    n_local_posts = len(t)
    t = dedupe_families(t)
    n_dedup = len(t)
    t['n_artist_local'] = t.groupby('artist').id.transform('size')
    print(f'filtered {n_filtered}; with local image {n_local_posts}; family-deduped {n_dedup}; '
          f'artists {t.artist.nunique()} ({time.time() - t0:.0f}s)', flush=True)

    # ---- flags (artist level)
    def flag_frame(names: pd.Index) -> pd.DataFrame:
        s = pd.Series(names, index=names)
        f = pd.DataFrame({'eval_0918_test': s.isin(f_0918), 'eval_test100': s.isin(f_100), 'eval_public': s.isin(f_pub),
                          'eval_unseen_v1': s.isin(f_unseen), 'in_v1': s.isin(in_v1)})
        f['in_kaloscope'] = [c.strip().lower() in kal or any(a.strip().lower() in kal for a in pre.get(c, []))
                             for c in names]
        return f

    names = pd.Index(t.artist.unique(), name='artist')
    fr = flag_frame(names)
    for c in FLAGS:
        t[c] = t.artist.map(fr[c])
    missing = {k: sorted(s - set(names)) for k, s in [('eval_0918_test', f_0918), ('eval_test100', f_100),
                                                       ('eval_public', f_pub), ('eval_unseen_v1', f_unseen)]}
    WORK.mkdir(parents=True, exist_ok=True)
    keep = (t.n_artist_local >= MIN_LOCAL) | t[EVAL_FLAGS].any(axis=1)     # eval artists are never cut by n_local
    t = t[keep].copy()
    if any(missing.values()):
        json.dump(missing, open(WORK / 'missing_eval_artists.json', 'w'), indent=1)
        print({k: len(v) for k, v in missing.items()}, 'eval artists without local posts -> missing_eval_artists.json')

    # ---- dates
    ii, it = _pixiv_index()
    ptime, psrc = dates.pixiv_time(t.pixiv_id, ii, it)
    d = dates.correct_dates(t.created_at, [(dates.twitter_time(t.source), 'twitter'), (ptime, psrc)])
    t['date'] = pd.to_datetime(d.date, utc=True)
    t['date_source'] = d.date_source
    assert t.date.notna().all()
    t['t_year'] = dates.to_year(t.date)
    t['date_day'] = dates.to_day(t.date)
    t['upload_day'] = dates.to_day(t.created_at)

    t['characters'] = t.tag_string_character.fillna('').str.split()
    t['copyrights'] = t.tag_string_copyright.fillna('').str.split()
    t['general'] = t.tag_string_general.fillna('')
    t['medium'] = [medium_of(set(g.split())) for g in t.general]
    t = t.rename(columns={'image_width': 'orig_width', 'image_height': 'orig_height'})
    out_cols = (['id', 'artist', 'family', 'ds', 'shard', 'date', 'date_source', 't_year', 'date_day', 'upload_day',
                 'characters', 'copyrights', 'general', 'medium', 'rating', 'orig_width', 'orig_height', 'pixiv_id',
                 'n_artist_local'] + FLAGS)
    t['pixiv_id'] = pd.to_numeric(t.pixiv_id, errors='coerce').astype('Int64')
    posts = t[out_cols].sort_values(['artist', 'id']).reset_index(drop=True)

    g = posts.groupby('artist')
    art = pd.DataFrame({'n_local': g.size(), 't_first': g.date.min(), 't_last': g.date.max()})
    art['n_snapshot'] = n_snapshot.reindex(art.index).fillna(0).astype(int)
    art['span'] = (art.t_last - art.t_first).dt.days / 365.25
    art = art.join(g[FLAGS].first()).reset_index()[['artist', 'n_snapshot', 'n_local', 't_first', 't_last', 'span'] + FLAGS]

    posts.to_parquet(WORK / 'posts.parquet', index=False)
    art.to_parquet(WORK / 'artists.parquet', index=False)
    stats = {'n_snapshot_rows': len(snap), 'n_filtered': n_filtered, 'n_local_posts': n_local_posts,
             'n_family_deduped': n_dedup, 'n_posts': len(posts), 'n_artists': len(art),
             'n_artists_ge30': int((art.n_local >= 30).sum()), 'n_artists_10_29': int(art.n_local.between(10, 29).sum()),
             'n_artists_lt10_eval': int((art.n_local < MIN_LOCAL).sum()),
             'flag_artists': {c: int(art[c].sum()) for c in FLAGS},
             'missing_eval': {k: len(v) for k, v in missing.items()},
             'date_source_share': posts.date_source.value_counts(normalize=True).round(4).to_dict(),
             'date_source_n': posts.date_source.value_counts().to_dict(),
             'n_0918_extra_from_test_split': n_extra, 'n_v1_names': len(nm), 'n_v1_renamed': len(renamed), 'n_many_to_one_targets': len(many),
             'n_many_to_one_v1_names': sum(len(v) for v in many.values()),
             'runtime_s': round(time.time() - t0)}
    json.dump(stats, open(WORK / 'prepare_stats.json', 'w'), indent=1)
    print(json.dumps(stats, indent=1), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prepare'])
    args = ap.parse_args()
    prepare()
