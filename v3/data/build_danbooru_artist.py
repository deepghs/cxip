"""CSIP danbooru artist dataset (cxip-danbooru-artist/v1) in neko-data format.

Pipeline (all outputs under /data/neko/cxip_v3/data/; work files in danbooru-artist-v1-work/):

    python -m v3.data.fingerprint danbooru --out <work>/fingerprints             # dHash + true image size
    python -m v3.data.fingerprint folder --root <csip_eval> --out <work>/csip_eval_fp.parquet
    python -m v3.data.build_danbooru_artist match-csip-eval                         # csip_eval class -> artist
    python -m v3.data.build_danbooru_artist prepare                                 # DuckDB join -> plan.parquet
    python -m v3.data.build_danbooru_artist build                                   # neko-data build (resumable)
    python -m v3.data.build_danbooru_artist verify

Layout follows ccip_merged_nd: one neko-data split ``all`` (every image, artists contiguous and
never split across shards, artist order shuffled with ``seed``), ``classes.json`` (label -> artist),
original webp bytes. Train/test membership is stored per sample (split_rebuilt, split_0918,
artist_0918_test, csip_eval_public_artist) so any split can be derived from the index without
repacking.

Per-sample metadata (tar json + parquet sidecar; list fields only in metadata_json / json):
    label, artist, danbooru_id, characters[list], copyrights[list], year, rating, general (isek string,
    as-is), orig_width, orig_height (danbooru post size), score, fav_count, split_rebuilt, split_0918,
    artist_0918_test, csip_eval_public_artist, n_characters, n_copyrights, dhash64 (hex)
    + neko-data fields width/height (stored image size), image_sha256, ...
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path('/data/neko/cxip_repro/data/danbooru-w640-ws-full-artist')
SPLIT_0918 = Path('/data/neko/cxip_repro/data/danbooru-artist-split-0918')
CSIP_EVAL = Path('/data/neko/cxip_repro/data/csip_eval')
ISEK = '/data/.hf/hub/datasets--isek-ai--danbooru-tags-2024/snapshots/*/data/*.parquet'
OUT = Path('/data/neko/cxip_v3/data/danbooru-artist-v1')
WORK = Path('/data/neko/cxip_v3/data/danbooru-artist-v1-work')
DATASET_ID = 'cxip-danbooru-artist'
VERSION = 'v1'
SPLIT = 'all'


# ---------------------------------------------------------------------------------------------
# csip_eval -> artist matching (folders are numbered; match images by dHash)
# ---------------------------------------------------------------------------------------------

def _u64(col) -> np.ndarray:
    return np.frombuffer(b''.join(col), dtype='>u8').astype(np.uint64)


def _hamming256(a: bytes, b: bytes) -> int:
    return int(np.unpackbits(np.frombuffer(a, np.uint8) ^ np.frombuffer(b, np.uint8)).sum())


def _match_chunk(args):
    q, db, thr = args
    out = []
    for qi, h in q:
        d = np.bitwise_count(db ^ h)
        idx = np.nonzero(d <= thr)[0]
        out.extend((qi, int(j), int(d[j])) for j in idx)
    return out


def match_csip_eval(thr64: int = 10, thr256: int = 24, min_votes: int = 2, min_frac: float = 0.5,
                    workers: int = 32):
    from multiprocessing import Pool
    fp = pd.concat([pd.read_parquet(p) for p in sorted((WORK / 'fingerprints').glob('part-*.parquet'))])
    fp = fp[fp.error.isna()].reset_index(drop=True)
    idx = pd.read_parquet(SRC / 'index.parquet').drop_duplicates('id')
    artist_of = dict(zip(idx.id, idx.artist))
    ev = pd.read_parquet(WORK / 'csip_eval_fp.parquet')
    ev = ev[ev.error.isna()].reset_index(drop=True)
    db = _u64(fp.dhash64.tolist())
    q = _u64(ev.dhash64.tolist())
    # perfectly flat images hash to 0 / all-ones and match everything: skip them
    flat = (q == 0) | (q == np.uint64(2 ** 64 - 1))
    items = [(i, q[i]) for i in range(len(q)) if not flat[i]]
    chunks = [(items[i:i + 32], db, thr64) for i in range(0, len(items), 32)]
    with Pool(workers) as pool:
        pairs = [p for res in pool.imap_unordered(_match_chunk, chunks) for p in res]
    pairs = pd.DataFrame(pairs, columns=['ev', 'db', 'd64'])
    pairs['d256'] = [_hamming256(ev.dhash256[e], fp.dhash256[d]) for e, d in zip(pairs.ev, pairs.db)]
    pairs['id'] = fp.id.values[pairs.db]
    pairs['artist'] = pairs.id.map(artist_of)
    pairs['cls'] = ev.cls.values[pairs.ev]
    good = pairs[pairs.d256 <= thr256].sort_values(['ev', 'd256']).drop_duplicates('ev')
    rows = []
    for cls, g in ev.groupby('cls'):
        m = good[good.cls == cls]
        vc = m.artist.value_counts()
        top, votes = (vc.index[0], int(vc.iloc[0])) if len(vc) else (None, 0)
        ok = votes >= min_votes and votes >= min_frac * len(m)
        rows.append({'cls': cls, 'n_images': len(g), 'n_matched_images': len(m), 'top_artist': top,
                     'votes': votes, 'second': vc.index[1] if len(vc) > 1 else None,
                     'second_votes': int(vc.iloc[1]) if len(vc) > 1 else 0,
                     'artist': top if ok else None})
    res = pd.DataFrame(rows).sort_values('cls', key=lambda s: s.astype(int))
    res.to_csv(WORK / 'csip_eval_artist_match.csv', index=False)
    good.to_parquet(WORK / 'csip_eval_image_match.parquet', index=False)
    print(res.to_string())
    print(f'classes matched: {res.artist.notna().sum()}/{len(res)}; images matched {len(good)}/{len(ev)}; '
          f'distinct artists {res.artist.nunique()}')


# ---------------------------------------------------------------------------------------------
# metadata join -> plan.parquet (stream order)
# ---------------------------------------------------------------------------------------------

def _split_list(s):
    if s is None or (isinstance(s, float) and np.isnan(s)) or not s:
        return []
    return [t.strip().replace(' ', '_') for t in s.split(', ') if t.strip()]


def prepare(seed: int = 0):
    import duckdb
    idx = pd.read_parquet(SRC / 'index.parquet')
    n_raw = len(idx)
    idx = idx.drop_duplicates(['id']).rename(columns={'split': 'split_rebuilt'})[['id', 'artist', 'split_rebuilt']]
    print(f'index rows {n_raw} -> {len(idx)} unique ids, {idx.artist.nunique()} artists', flush=True)

    con = duckdb.connect()
    con.register('idx', idx)
    # isek contains a few duplicate ids (re-scraped rows): keep the most recently updated one
    meta = con.sql(f"""
        with isek as (
            select id, created_at, rating, "character", copyright, artist as isek_artist, general,
                   image_width, image_height, score, fav_count,
                   row_number() over (partition by id order by updated_at desc) rn
            from read_parquet('{ISEK}')
            where id in (select id from idx)
        )
        select idx.id, idx.artist, idx.split_rebuilt, isek.created_at, isek.rating, isek."character",
               isek.copyright, isek.isek_artist, isek.general, isek.image_width, isek.image_height,
               isek.score, isek.fav_count
        from idx left join (select * from isek where rn = 1) isek on idx.id = isek.id
    """).df()
    print(f'joined {len(meta)}; isek coverage {meta.created_at.notna().mean():.4f}', flush=True)
    meta['characters'] = [_split_list(s) for s in meta['character']]
    meta['copyrights'] = [_split_list(s) for s in meta['copyright']]
    meta['year'] = pd.to_numeric(meta.created_at.str.slice(0, 4), errors='coerce').astype('Int64')
    isek_art = meta.isek_artist.fillna('').str.replace(' ', '_')
    meta['artist_agree'] = [a in s.split(',_') for a, s in zip(meta.artist, isek_art)]
    print(f'isek artist agrees with index artist: {meta.artist_agree.mean():.4f}', flush=True)

    # 0918 split, by image id
    s0918 = {}
    for name, tag in [('train_split.json', 'train'), ('test_split.json', 'test'), ('test_split_100.json', 'test100')]:
        for k in json.load(open(SPLIT_0918 / name)):
            s0918[int(k.split('.')[0])] = tag  # later files override: test100 > test > train
    meta['split_0918'] = meta.id.map(s0918).fillna('none')
    test_artists_0918 = set(json.load(open(SPLIT_0918 / 'test_split.json')).values())
    meta['artist_0918_test'] = meta.artist.isin(test_artists_0918)

    # csip_eval_public
    match = pd.read_csv(WORK / 'csip_eval_artist_match.csv')
    eval_artists = set(match.artist.dropna())
    meta['csip_eval_public_artist'] = meta.artist.isin(eval_artists)
    csip_cls = {a: int(c) for c, a in zip(match.cls, match.artist) if isinstance(a, str)}

    # true stored image size + dHash from the fingerprint pass
    fp = pd.concat([pd.read_parquet(p, columns=['id', 'nbytes', 'width', 'height', 'dhash64', 'error'])
                    for p in sorted((WORK / 'fingerprints').glob('part-*.parquet'))])
    bad = fp[fp.error.notna()]
    if len(bad):
        print(f'WARNING {len(bad)} undecodable images, dropped: {bad.id.tolist()[:20]}', flush=True)
        bad.to_csv(WORK / 'undecodable.csv', index=False)
    fp = fp[fp.error.isna()]
    fp['dhash64'] = [b.hex() for b in fp.dhash64]
    meta = meta.merge(fp.drop(columns=['error']), on='id', how='inner')

    # labels: artists sorted by name; stream order: artists shuffled (seed), ids ascending within
    artists = sorted(meta.artist.unique().tolist())
    label_of = {a: i for i, a in enumerate(artists)}
    meta['label'] = meta.artist.map(label_of).astype('int64')
    order = list(artists)
    random.Random(f'{seed}-{SPLIT}').shuffle(order)
    rank = {a: i for i, a in enumerate(order)}
    meta['_rank'] = meta.artist.map(rank)
    meta = meta.sort_values(['_rank', 'id']).drop(columns=['_rank', 'character', 'copyright', 'created_at',
                                                           'isek_artist']).reset_index(drop=True)
    meta = meta.rename(columns={'image_width': 'orig_width', 'image_height': 'orig_height'})
    WORK.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(WORK / 'plan.parquet', index=False)

    classes = []
    g = meta.groupby('label')
    sizes = g.size()
    n_by = meta.groupby(['label', 'split_rebuilt']).size().unstack(fill_value=0)
    n_0918 = meta.groupby(['label', 'split_0918']).size().unstack(fill_value=0)
    flags = g[['artist_0918_test', 'csip_eval_public_artist']].first()
    for lab, a in enumerate(artists):
        classes.append({
            'label': lab, 'name': a, 'n': int(sizes[lab]),
            'n_split_rebuilt': {k: int(v) for k, v in n_by.loc[lab].items() if v},
            'n_split_0918': {k: int(v) for k, v in n_0918.loc[lab].items() if v},
            'artist_0918_test': bool(flags.loc[lab, 'artist_0918_test']),
            'csip_eval_public_artist': bool(flags.loc[lab, 'csip_eval_public_artist']),
            'csip_eval_class': csip_cls.get(a),
        })
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'classes.json').write_text(json.dumps(classes, ensure_ascii=False) + '\n', encoding='utf-8')
    stats = {
        'num_samples': len(meta), 'num_classes': len(artists),
        'frac_with_characters': float((meta.characters.str.len() > 0).mean()),
        'frac_with_copyrights': float((meta.copyrights.str.len() > 0).mean()),
        'frac_with_non_original_copyright': float(meta.copyrights.map(lambda c: any(x != 'original' for x in c)).mean()),
        'frac_with_year': float(meta.year.notna().mean()),
        'frac_isek_joined': float(meta.rating.notna().mean()),
        'isek_artist_agree': float(meta.artist_agree.mean()),
        'split_rebuilt': meta.split_rebuilt.value_counts().to_dict(),
        'split_0918': meta.split_0918.value_counts().to_dict(),
        'artist_0918_test_artists': int(flags.artist_0918_test.sum()),
        'artist_0918_test_images': int(meta.artist_0918_test.sum()),
        'csip_eval_public_artists': int(flags.csip_eval_public_artist.sum()),
        'csip_eval_public_images': int(meta.csip_eval_public_artist.sum()),
        'total_image_bytes': int(meta.nbytes.sum()),
        'index_rows_raw': n_raw,
        'extra_files_not_in_index': None,
    }
    (WORK / 'prepare_stats.json').write_text(json.dumps(stats, indent=2, default=int) + '\n')
    print(json.dumps(stats, indent=2, default=int), flush=True)


# ---------------------------------------------------------------------------------------------
# neko-data build
# ---------------------------------------------------------------------------------------------

def _completed_samples(out: Path) -> int:
    journal = out / 'reports' / f'build-journal-{SPLIT}.jsonl'
    if not journal.exists():
        return 0
    return sum(json.loads(l)['num_samples'] for l in journal.read_text().splitlines() if l.strip())


def iter_samples(plan: str | os.PathLike = WORK / 'plan.parquet', out: str | os.PathLike = OUT,
                 threads: int = 32, window: int = 4096, limit: int = 0):
    """Ordered stream of NormalizedSample; file reads run in a thread pool with a bounded window.

    On resume, neko-data skips the first ``completed`` samples of the stream; we yield cheap
    placeholders for them instead of reading their bytes.
    """
    from neko_data.contract.records import NormalizedSample

    meta = pd.read_parquet(plan)
    if limit:
        meta = meta.iloc[:limit]
    completed = _completed_samples(Path(out))
    print(f'stream: {len(meta)} samples, {completed} already in journal', flush=True)
    for _ in range(completed):
        yield None
    rows = meta.iloc[completed:].to_dict('records')

    def read(r):
        with open(SRC / 'train' / f"{r['id']}.webp", 'rb') as f:
            return f.read()

    def sample(r, data):
        md = {
            'label': int(r['label']), 'artist': r['artist'], 'danbooru_id': int(r['id']),
            'characters': list(r['characters']), 'copyrights': list(r['copyrights']),
            'n_characters': len(r['characters']), 'n_copyrights': len(r['copyrights']),
            'year': None if pd.isna(r['year']) else int(r['year']),
            'rating': r['rating'], 'general': r['general'],
            'orig_width': None if pd.isna(r['orig_width']) else int(r['orig_width']),
            'orig_height': None if pd.isna(r['orig_height']) else int(r['orig_height']),
            'score': None if pd.isna(r['score']) else int(r['score']),
            'fav_count': None if pd.isna(r['fav_count']) else int(r['fav_count']),
            'split_rebuilt': r['split_rebuilt'], 'split_0918': r['split_0918'],
            'artist_0918_test': bool(r['artist_0918_test']),
            'csip_eval_public_artist': bool(r['csip_eval_public_artist']),
            'dhash64': r['dhash64'],
        }
        return NormalizedSample(
            sample_key=str(r['id']), image=data, image_extension='webp', split=SPLIT,
            source_id='danbooru-w640-ws-full', caption=r['artist'], metadata=md,
            width=int(r['width']), height=int(r['height']),
        )

    t0, n = time.time(), 0
    with ThreadPoolExecutor(threads) as ex:
        futures = []
        it = iter(rows)
        for r in it:
            futures.append((r, ex.submit(read, r)))
            if len(futures) >= window:
                break
        head = 0
        while head < len(futures):
            r, fut = futures[head]
            futures[head] = None
            head += 1
            nxt = next(it, None)
            if nxt is not None:
                futures.append((nxt, ex.submit(read, nxt)))
            yield sample(r, fut.result())
            n += 1
            if n % 100_000 == 0:
                el = time.time() - t0
                print(f'{completed + n}/{len(meta)} {n / el:.0f} img/s elapsed {el / 60:.1f} min', flush=True)
            if head > 65536:  # keep the list from growing forever
                futures = futures[head:]
                head = 0


def build(max_shard_size: str = '1GiB', seed: int = 0, limit: int = 0, out: Path = OUT):
    from neko_data.build.job import BuildJob, build_dataset
    out.mkdir(parents=True, exist_ok=True)
    classes = json.load(open(OUT / 'classes.json'))
    if out != OUT:
        (out / 'classes.json').write_text((OUT / 'classes.json').read_text())
    stats = json.load(open(WORK / 'prepare_stats.json'))
    meta = {
        'task': 'csip', 'num_classes': len(classes), 'label_key': 'label', 'group_key': 'label',
        'classes_file': 'classes.json', 'seed': seed,
        'source': 'animetimm/danbooru-wdtagger-v4-w640-ws-full images (webp bytes as-is) + '
                  'isek-ai/danbooru-tags-2024 tags; artists = single-artist posts, >= 30 images/artist',
        'split_columns': ['split_rebuilt', 'split_0918', 'artist_0918_test', 'csip_eval_public_artist'],
        'coverage': {k: stats[k] for k in ['frac_with_characters', 'frac_with_copyrights', 'frac_with_year']},
    }
    m = build_dataset(BuildJob(
        source=iter_samples(out=out, limit=limit), output_dir=out, dataset_id=DATASET_ID, version=VERSION,
        split=SPLIT, max_shard_size=max_shard_size, group_key='label', resume=True,
        skip_invalid_samples=False, metadata=meta,
    ))
    print('done', m.metadata.get('split_stats'), flush=True)


def verify(out: Path = OUT, n_check: int = 2000):
    """Manifest vs files, sample counts vs plan, artist contiguity, random byte-exact checks."""
    import hashlib
    import tarfile
    m = json.load(open(out / 'manifest.json'))
    shards = [s for s in m['shards'] if s['split'] == SPLIT]
    plan = pd.read_parquet(WORK / 'plan.parquet', columns=['id', 'label', 'nbytes'])
    total = sum(s['num_samples'] for s in shards)
    print(f'{len(shards)} shards, {total} samples (plan {len(plan)}), '
          f'{sum(s["size_bytes"] for s in shards) / 2**30:.1f} GiB')
    assert total == len(plan), 'sample count mismatch'
    for s in shards:
        p = out / s['path']
        assert p.stat().st_size == s['size_bytes'], s['path']
    meta = pd.concat([pd.read_parquet(out / s['metadata_path'], columns=['sample_key', 'label', 'shard_path'])
                      for s in shards], ignore_index=True)
    assert meta.sample_key.astype('int64').tolist() == plan.id.tolist(), 'order mismatch vs plan'
    # every label lives in exactly one shard
    per_label = meta.groupby('label').shard_path.nunique()
    print(f'labels {meta.label.nunique()}, split across shards: {(per_label > 1).sum()}')
    rng = np.random.default_rng(0)
    for s in rng.choice(len(shards), size=min(5, len(shards)), replace=False):
        sh = shards[s]
        with tarfile.open(out / sh['path']) as tf:
            members = [mm for mm in tf.getmembers() if mm.name.endswith('.webp')]
            for mm in rng.choice(members, size=min(n_check // 5, len(members)), replace=False):
                data = tf.extractfile(mm).read()
                src = (SRC / 'train' / mm.name).read_bytes()
                assert data == src, mm.name
    print('byte-exact spot checks OK')
    print('VERIFY OK')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['match-csip-eval', 'prepare', 'build', 'verify'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--max-shard-size', default='1GiB')
    a = ap.parse_args()
    if a.cmd == 'match-csip-eval':
        match_csip_eval()
    elif a.cmd == 'prepare':
        prepare(a.seed)
    elif a.cmd == 'build':
        build(a.max_shard_size, a.seed, a.limit, a.out)
    else:
        verify(a.out)
    sys.exit(0)
