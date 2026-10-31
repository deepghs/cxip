"""Danbooru2026 per-post table for CSIP v4 (danbooru-artist-v2 prepare step).

Reads the 2026 Danbooru snapshot, keeps single-artist / not deleted / jpg|png|webp / non-AI posts that have a local
image, keeps one post per (artist, work family), corrects the date to the earliest plausible original-publication
date (twitter snowflake / pixiv index), maps 2024 (v1) artist tags to their current names by majority of shared post
ids, and flags the evaluation artists. Output: WORK/{posts.parquet, artists.parquet, prepare_stats.json}.
Then resizes the images (short side <= 640, webp q95) into one part tar per source shard and packs the
``cxip-danbooru-artist/v2`` neko-data dataset (single split ``all``, artists contiguous, labels = name-sorted
artists, stream order = artists shuffled with ``seed``, ids ascending within an artist).

    python -m v3.data.build_danbooru_artist_v2 prepare
    python -m v3.data.build_danbooru_artist_v2 jobs                    # WORK/jobs.parquet (src_tar, member, key)
    python -m v3.data.build_danbooru_artist_v2 resize --workers 160    # WORK/resized/*.tar + *.parquet (resumable)
    python -m v3.data.build_danbooru_artist_v2 build                   # neko-data build (resumable)
    python -m v3.data.build_danbooru_artist_v2 verify
    python -m v3.data.build_index danbooru_v2                          # OUT/index/{all,train_ge30,train_ge10}.parquet
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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
OUT = V4 / 'data/danbooru-artist-v2'
RESIZED = WORK / 'resized'
DS_DIRS = {0: 'danbooru-full-4mpixel-v1', 1: 'danbooru-newest-4mpixel-v1'}
DATASET_ID = 'cxip-danbooru-artist'
VERSION = 'v2'
SPLIT = 'all'
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


# ---------------------------------------------------------------------------------------------
# resize jobs -> parts -> plan -> neko-data build
# ---------------------------------------------------------------------------------------------

PLAN_COLS = ['id', 'artist', 'family', 'date', 'date_source', 't_year', 'date_day', 'upload_day', 'characters',
             'copyrights', 'general', 'medium', 'rating', 'orig_width', 'orig_height', 'n_artist_local'] + FLAGS


def src_tar(ds: int, shard: int) -> str:
    return str(SHARDS / DS_DIRS[int(ds)] / 'wds/train' / f'train-{int(shard):06d}.tar')


def make_jobs(posts: pd.DataFrame) -> pd.DataFrame:
    """(src_tar, member, key) per post: member ``{id}.webp`` in its local shard, output key ``{id}``."""
    j = pd.DataFrame({'src_tar': [src_tar(d, s) for d, s in zip(posts.ds, posts.shard)],
                      'member': posts.id.astype(str) + '.webp', 'key': posts.id.astype(str),
                      'id': posts.id.astype('int64')})
    return j.sort_values(['src_tar', 'id']).reset_index(drop=True)


def make_plan(posts: pd.DataFrame, parts: pd.DataFrame, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join posts with the resized parts. Labels = index in the name-sorted list of ALL artists in ``posts``;
    stream order = artists shuffled (``seed``), ids ascending within an artist. Posts whose resize failed go to the
    returned error table; every post must have a parts row."""
    artists = sorted(posts.artist.unique().tolist())
    label_of = {a: i for i, a in enumerate(artists)}
    order = list(artists)
    random.Random(f'{seed}-{SPLIT}').shuffle(order)
    rank = {a: i for i, a in enumerate(order)}
    pp = parts[['key', 'part', 'offset', 'size', 'width', 'height', 'error']]
    assert not pp.key.duplicated().any(), 'duplicate keys in parts'
    m = posts.assign(key=posts.id.astype(str)).merge(pp, on='key', how='left', indicator=True)
    assert (m._merge == 'both').all(), f'{int((m._merge != "both").sum())} posts have no resize result'
    m['label'] = m.artist.map(label_of).astype('int64')
    bad = m.error.notna()
    errors = m.loc[bad, ['id', 'artist', 'part', 'error']].sort_values('id').reset_index(drop=True)
    plan = m[~bad].assign(_rank=lambda d: d.artist.map(rank)).sort_values(['_rank', 'id'])
    plan = plan.drop(columns=['_rank', '_merge', 'key', 'error']).reset_index(drop=True)
    for c in ['offset', 'size', 'width', 'height']:
        plan[c] = plan[c].astype('int64')
    return plan, errors


def jobs():
    p = WORK / 'jobs.parquet'
    if p.exists():
        print(f'{p} exists ({len(pd.read_parquet(p, columns=["key"]))} rows); keeping it', flush=True)
        return
    j = make_jobs(pd.read_parquet(WORK / 'posts.parquet', columns=['id', 'ds', 'shard']))
    missing = [t for t in j.src_tar.unique() if not os.path.exists(t)]
    assert not missing, f'{len(missing)} source tars missing, e.g. {missing[:3]}'
    tmp = p.with_name(p.name + '.tmp')
    j.to_parquet(tmp, index=False)
    os.replace(tmp, p)
    print(f'jobs: {len(j)} images in {j.src_tar.nunique()} source tars -> {p}', flush=True)


def resize(workers: int = 160, limit_tars: int = 0):
    from v3.data.resize_parts import run_jobs
    j = pd.read_parquet(WORK / 'jobs.parquet')
    if limit_tars:
        keep = sorted(j.src_tar.unique())[:limit_tars]
        j = j[j.src_tar.isin(keep)]
    t0 = time.time()
    parts = run_jobs(j, RESIZED, workers)
    n_err = int(parts.error.notna().sum())
    print(f'resize done: {parts.part.nunique()} parts, {len(parts)} rows, errors {n_err} '
          f'({n_err / max(len(parts), 1):.4%}), {time.time() - t0:.0f}s', flush=True)
    if n_err:
        print(parts[parts.error.notna()].error.str.slice(0, 60).value_counts().head(10).to_string(), flush=True)


def load_parts(j: pd.DataFrame) -> pd.DataFrame:
    from v3.data.resize_parts import part_name
    names = [part_name(t) for t in sorted(j.src_tar.unique())]
    todo = [n for n in names if not ((RESIZED / n).exists() and (RESIZED / n).with_suffix('.parquet').exists())]
    if todo:
        sys.exit(f'{len(todo)} resize parts not finished (e.g. {todo[:3]}); run `resize` first')
    with ThreadPoolExecutor(32) as ex:
        dfs = list(ex.map(lambda n: pd.read_parquet((RESIZED / n).with_suffix('.parquet')).assign(part=n), names))
    return pd.concat(dfs, ignore_index=True)


def _completed_samples(out: Path) -> int:
    journal = out / 'reports' / f'build-journal-{SPLIT}.jsonl'
    if not journal.exists():
        return 0
    return sum(json.loads(l)['num_samples'] for l in journal.read_text().splitlines() if l.strip())


def _pread(part: str, offset: int, size: int, root: Path | None = None) -> bytes:
    fd = os.open((root or RESIZED) / part, os.O_RDONLY)
    try:
        return os.pread(fd, size, offset)
    finally:
        os.close(fd)


def _md(r: dict) -> dict:
    md = {'label': int(r['label']), 'artist': r['artist'], 'danbooru_id': int(r['id']), 'family': int(r['family']),
          'date': pd.Timestamp(r['date']).isoformat(), 'date_source': r['date_source'], 't_year': float(r['t_year']),
          'date_day': int(r['date_day']), 'upload_day': int(r['upload_day']),
          'characters': [str(x) for x in r['characters']], 'copyrights': [str(x) for x in r['copyrights']],
          'general': r['general'], 'medium': r['medium'], 'rating': r['rating'],
          'orig_width': None if pd.isna(r['orig_width']) else int(r['orig_width']),
          'orig_height': None if pd.isna(r['orig_height']) else int(r['orig_height']),
          'n_artist_local': int(r['n_artist_local'])}
    md.update({c: bool(r[c]) for c in FLAGS})
    return md


def iter_samples(plan: pd.DataFrame, out: Path, threads: int = 32, window: int = 4096):
    """Ordered NormalizedSample stream; part preads run in a thread pool with a bounded window. On resume the first
    ``completed`` samples (already in the build journal) are yielded as cheap placeholders that neko-data skips."""
    from neko_data.contract.records import NormalizedSample
    completed = _completed_samples(Path(out))
    print(f'stream: {len(plan)} samples, {completed} already in journal', flush=True)
    for _ in range(completed):
        yield None
    rows = plan.iloc[completed:].to_dict('records')

    def sample(r, data):
        return NormalizedSample(sample_key=str(r['id']), image=data, image_extension='webp', split=SPLIT,
                                source_id='danbooru2026-4mpixel-w640', caption=r['artist'], metadata=_md(r),
                                width=int(r['width']), height=int(r['height']))

    t0, n = time.time(), 0
    with ThreadPoolExecutor(threads) as ex:
        it = iter(rows)
        futures = []
        for r in it:
            futures.append((r, ex.submit(_pread, r['part'], r['offset'], r['size'])))
            if len(futures) >= window:
                break
        head = 0
        while head < len(futures):
            r, fut = futures[head]
            futures[head] = None
            head += 1
            nxt = next(it, None)
            if nxt is not None:
                futures.append((nxt, ex.submit(_pread, nxt['part'], nxt['offset'], nxt['size'])))
            yield sample(r, fut.result())
            n += 1
            if n % 100_000 == 0:
                el = time.time() - t0
                print(f'{completed + n}/{len(plan)} {n / el:.0f} img/s elapsed {el / 60:.1f} min', flush=True)
            if head > 65536:
                futures = futures[head:]
                head = 0


def _write_parquet(df: pd.DataFrame, path: Path):
    tmp = path.with_name(path.name + '.tmp')
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def build(max_shard_size: str = '1GiB', seed: int = 0, threads: int = 32, out: Path | None = None):
    from neko_data.build.job import BuildJob, build_dataset
    out = out or OUT
    if (out / 'manifest.json').exists():
        sys.exit(f'{out} already has manifest.json; bump the version instead of overwriting')
    plan_path = WORK / 'plan.parquet'
    posts = pd.read_parquet(WORK / 'posts.parquet', columns=PLAN_COLS)
    if plan_path.exists():
        plan = pd.read_parquet(plan_path)
        print(f'plan exists: {len(plan)} samples (resume)', flush=True)
    else:
        parts = load_parts(pd.read_parquet(WORK / 'jobs.parquet', columns=['src_tar']))
        plan, errors = make_plan(posts, parts, seed)
        errors.to_csv(WORK / 'resize_errors.csv', index=False)
        _write_parquet(plan, plan_path)
        print(f'plan: {len(plan)} samples, {plan.label.nunique()} artists; resize errors skipped {len(errors)} '
              f'-> resize_errors.csv', flush=True)
    out.mkdir(parents=True, exist_ok=True)
    artists = sorted(posts.artist.unique().tolist())
    if not (out / 'classes.json').exists():
        n = plan.groupby('label').size()
        fl = posts.groupby('artist')[FLAGS].first()
        nl = posts.groupby('artist').n_artist_local.first()
        classes = [{'label': i, 'name': a, 'n': int(n.get(i, 0)), 'n_artist_local': int(nl[a]),
                    **{c: bool(fl.loc[a, c]) for c in FLAGS}} for i, a in enumerate(artists)]
        (out / 'classes.json').write_text(json.dumps(classes, ensure_ascii=False) + '\n', encoding='utf-8')
    del posts
    meta = {
        'task': 'csip', 'num_classes': len(artists), 'label_key': 'label', 'group_key': 'label',
        'classes_file': 'classes.json', 'seed': seed,
        'source': 'nyanko-devs/danbooru2026 posts snapshot + local danbooru-{full,newest}-4mpixel-v1 shards; '
                  'single-artist non-AI posts, one per work family, >= 10 local posts per artist (+ all eval-flagged '
                  'artists); images resized to short side <= 640 (BICUBIC, never upscaled), webp q95, alpha kept',
        'split_columns': FLAGS,
        'date': 'earliest plausible original-publication date (twitter snowflake / pixiv index / upload), ISO UTC',
    }
    m = build_dataset(BuildJob(
        source=iter_samples(plan, out=out, threads=threads), output_dir=out, dataset_id=DATASET_ID, version=VERSION,
        split=SPLIT, max_shard_size=max_shard_size, group_key='label', resume=True, skip_invalid_samples=False,
        metadata=meta,
    ))
    print('done', m.metadata.get('split_stats'), flush=True)


def verify(out: Path | None = None, n_check: int = 2000):
    """Manifest vs files, sample count vs successfully resized plan rows, order, artist contiguity, and
    byte-exact comparison of ``n_check`` random samples against the resized parts."""
    from v3.data.resize_parts import iter_tar_members
    out = out or OUT
    m = json.load(open(out / 'manifest.json'))
    shards = [s for s in m['shards'] if s['split'] == SPLIT]
    plan = pd.read_parquet(WORK / 'plan.parquet', columns=['id', 'label', 'part', 'offset', 'size'])
    parts = load_parts(pd.read_parquet(WORK / 'jobs.parquet', columns=['src_tar']))
    n_ok = int(parts.error.isna().sum())
    total = sum(s['num_samples'] for s in shards)
    print(f'{len(shards)} shards, {total} samples (plan {len(plan)}, resized OK {n_ok}, resize errors '
          f'{len(parts) - n_ok}), {sum(s["size_bytes"] for s in shards) / 2**30:.1f} GiB', flush=True)
    assert total == len(plan) == n_ok, 'sample count mismatch'
    for s in shards:
        assert (out / s['path']).stat().st_size == s['size_bytes'], s['path']
    meta = pd.concat([pd.read_parquet(out / s['metadata_path'], columns=['sample_key', 'label', 'shard_path'])
                      for s in shards], ignore_index=True)
    assert meta.sample_key.astype('int64').tolist() == plan.id.tolist(), 'order mismatch vs plan'
    assert (meta.label.to_numpy() == plan.label.to_numpy()).all(), 'label mismatch vs plan'
    runs = int(meta.label.ne(meta.label.shift()).sum())
    per_label = meta.groupby('label').shard_path.nunique()
    print(f'labels {meta.label.nunique()} (classes.json {len(json.load(open(out / "classes.json")))}), '
          f'label runs {runs}, labels split across shards {(per_label > 1).sum()}', flush=True)
    assert runs == meta.label.nunique(), 'artists not contiguous'
    assert (per_label == 1).all(), 'artist split across shards'
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(len(plan), size=min(n_check, len(plan)), replace=False))
    sel = meta.iloc[pick].assign(part=plan.part.to_numpy()[pick], offset=plan.offset.to_numpy()[pick],
                                 size=plan['size'].to_numpy()[pick])
    by_name = {Path(s['path']).name: s['path'] for s in shards}
    n = 0
    for shard_path, g in sel.groupby('shard_path'):
        path = by_name[Path(shard_path).name]
        members = {name: (o, sz) for name, o, sz in iter_tar_members(out / path)}
        for r in g.itertuples():
            o, sz = members[f'{r.sample_key}.webp']
            assert _pread(path, o, sz, out) == _pread(r.part, int(r.offset), int(r.size)), r.sample_key
            n += 1
    print(f'byte-exact checks OK ({n} samples)', flush=True)
    print('VERIFY OK', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prepare', 'jobs', 'resize', 'build', 'verify'])
    ap.add_argument('--workers', type=int, default=160)
    ap.add_argument('--limit-tars', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=32)
    ap.add_argument('--max-shard-size', default='1GiB')
    a = ap.parse_args()
    if a.cmd == 'prepare':
        prepare()
    elif a.cmd == 'jobs':
        jobs()
    elif a.cmd == 'resize':
        resize(a.workers, a.limit_tars)
    elif a.cmd == 'build':
        build(a.max_shard_size, a.seed, a.threads)
    else:
        verify()
