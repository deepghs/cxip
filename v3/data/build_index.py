"""Random-access tar offset indexes for neko-data datasets.

One parquet per split: ``<index_dir>/<split>.parquet`` (normally ``<dataset_root>/index/``), columns

    sample_key  str     neko-data sample key (tar member stem)
    shard       str     tar path relative to the dataset root, e.g. wds/train/train-000012.tar
    offset      int64   byte offset of the IMAGE member's data inside the tar
    size        int64   image bytes
    ext         str     image extension (webp/jpg/png/...)
    label       int64
    + metadata columns from the parquet sidecars (+ list columns parsed from metadata_json)

Built by sequentially scanning tar headers (``TarInfo.offset_data``) and joining with the
per-shard parquet sidecars. Usage:

    python -m v3.data.build_index ccip       # ccip_merged_nd -> /data/neko/cxip_v3/data/ccip-merged-v1-index/index/
    python -m v3.data.build_index danbooru   # danbooru-artist-v1 -> <root>/index/all.parquet
    python -m v3.data.build_index danbooru_v2   # danbooru-artist-v2 -> <root>/index/{all,train_ge30,train_ge10}.parquet
    python -m v3.data.build_index pixiv_v1   # pixiv-artist-v1 -> <root>/index/{all,train}.parquet
    python -m v3.data.build_index generic --root R --split S --out O
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tarfile
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

IMAGE_EXTS = {'jpg', 'jpeg', 'png', 'webp', 'avif', 'gif', 'bmp', 'tif', 'tiff'}
CCIP_ROOT = Path('/data/neko/cxip_repro/data/ccip_merged_nd')
CCIP_INDEX = Path('/data/neko/cxip_v3/data/ccip-merged-v1-index/index')
DANBOORU_ROOT = Path('/data/neko/cxip_v3/data/danbooru-artist-v1')
DANBOORU_V2_ROOT = Path('/scratch/cxip_v4/data/danbooru-artist-v2')
PIXIV_V1_ROOT = Path('/scratch/cxip_v4/data/pixiv-artist-v1')
BENCH2_SPLITS = Path('/scratch/cxip_v4/data/bench2_v1/splits.json')
EVAL_FLAGS_V2 = ['eval_0918_test', 'eval_public', 'eval_unseen_v1']
ISEK = '/data/.hf/hub/datasets--isek-ai--danbooru-tags-2024/snapshots/*/data/*.parquet'
INDEX_COLS = ['sample_key', 'shard', 'offset', 'size', 'ext', 'label']
# sidecar columns that duplicate index columns or are not useful for training
DROP_COLS = {'shard_path', 'image_extension', 'caption', 'metadata_json', 'source_id', 'split'}


def scan_tar(path: str | os.PathLike) -> list[tuple[str, str, int, int]]:
    """(sample_key, ext, offset_data, size) for every image member, in tar order (headers only)."""
    out = []
    with tarfile.open(path, mode='r:') as tf:
        for m in tf:
            if not m.isfile():
                continue
            stem, dot, ext = m.name.rpartition('.')
            if dot and ext.lower() in IMAGE_EXTS:
                out.append((stem, ext.lower(), m.offset_data, m.size))
    return out


def _index_shard(args):
    root, shard, list_cols = args
    rows = scan_tar(Path(root) / shard['path'])
    idx = pd.DataFrame(rows, columns=['sample_key', 'ext', 'offset', 'size'])
    idx.insert(1, 'shard', shard['path'])
    if len(idx) != shard['num_samples']:
        raise RuntimeError(f"{shard['path']}: {len(idx)} images in tar, manifest says {shard['num_samples']}")
    if idx.sample_key.duplicated().any():
        raise RuntimeError(f"{shard['path']}: duplicate sample keys")
    import pyarrow as pa
    import pyarrow.parquet as pq
    # keep integer columns with nulls as nullable Int64 (not float64)
    meta = pq.read_table(Path(root) / shard['metadata_path']).to_pandas(
        types_mapper={pa.int64(): pd.Int64Dtype()}.get)
    if list_cols:
        parsed = [json.loads(s) for s in meta['metadata_json']]
        for c in list_cols:
            meta[c] = [p.get(c) for p in parsed]
    meta = meta.drop(columns=[c for c in DROP_COLS if c in meta.columns])
    out = idx.merge(meta, on='sample_key', how='left', validate='one_to_one')
    if out['label'].isna().any():
        raise RuntimeError(f"{shard['path']}: samples without sidecar metadata")
    out['label'] = out['label'].astype('int64')
    out['offset'] = out['offset'].astype('int64')
    out['size'] = out['size'].astype('int64')
    return out


def build_split_index(root: Path, split: str, list_cols=(), workers: int = 16) -> pd.DataFrame:
    manifest = json.load(open(root / 'manifest.json'))
    shards = [s for s in manifest['shards'] if s['split'] == split]
    shards.sort(key=lambda s: s['path'])
    t0 = time.time()
    with Pool(min(workers, len(shards))) as pool:
        parts = pool.map(_index_shard, [(str(root), s, list(list_cols)) for s in shards])
    df = pd.concat(parts, ignore_index=True)
    for c in df.columns:  # nullable Int64 only where there really are nulls
        if isinstance(df[c].dtype, pd.Int64Dtype) and not df[c].isna().any():
            df[c] = df[c].astype('int64')
    cols = INDEX_COLS + [c for c in df.columns if c not in INDEX_COLS]
    df = df[cols]
    print(f'{split}: {len(df)} samples from {len(shards)} shards in {time.time() - t0:.1f}s', flush=True)
    return df


def write_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.tmp')
    df.to_parquet(tmp, index=False, compression='zstd')
    os.replace(tmp, path)
    print(f'wrote {path} ({path.stat().st_size / 2**20:.1f} MiB)', flush=True)


# ---------------------------------------------------------------------------------------------
# CCIP series (franchise) derivation
# ---------------------------------------------------------------------------------------------

def _fold(s: str) -> str:
    import unicodedata
    return ''.join(ch for ch in unicodedata.normalize('NFKD', str(s)) if not unicodedata.combining(ch)).lower()


def alnum(s: str) -> str:
    return re.sub(r'[^0-9a-z]', '', _fold(s))


def folder_series(s: str) -> str:
    return re.sub(r'_+', '_', re.sub(r'[^0-9a-z]+', '_', _fold(s))).strip('_')


# folder names / zerochan suffixes that are not a franchise
GENERIC = {'female', 'male', 'character', 'characters', 'others', 'other', 'original', 'vtuber', 'unknown',
           'oc', 'misc', 'human', 'mascot', 'cosplay', 'genderswap', 'gender_swap'}
CHAR_LOOKUP_WEIGHT = 1.5  # canonical danbooru copyright of the character beats folder names on ties


V1_ALIAS = {'fgo': 'fate/grand_order', 'genshin': 'genshin_impact', 'arknights': 'arknights',
            'azurlane': 'azur_lane', 'bluearchive': 'blue_archive', 'girlsfrontline': "girls'_frontline",
            'neuralcloud': "girls'_frontline_neural_cloud"}
PAREN = re.compile(r'^(.*?)\s*\(([^()]*(?:\([^()]*\))?[^()]*)\)\s*$')


def danbooru_tag_maps():
    """character-tag alnum key -> top copyright; copyright alnum key -> canonical copyright tag."""
    import duckdb
    con = duckdb.connect()
    pairs = con.sql(f"""
        with posts as (select string_split("character", ', ') chars, string_split(copyright, ', ') cps
                       from read_parquet('{ISEK}')
                       where tag_count_character between 1 and 3 and copyright is not null),
             y as (select chars, unnest(cps) cp from posts),
             x as (select unnest(chars) ch, cp from y)
        select ch, cp, count(*) n from x group by 1, 2
    """).df()
    pairs['ch'] = pairs.ch.str.replace(' ', '_')
    pairs['cp'] = pairs.cp.str.replace(' ', '_')
    pairs = pairs.sort_values(['ch', 'cp'], kind='stable').reset_index(drop=True)  # deterministic ties
    cp_total = pairs.groupby('cp').n.sum()
    copyright_key = {}
    for cp, n in cp_total.sort_values(kind='stable').items():  # most frequent wins on key collisions
        copyright_key[alnum(cp)] = cp
    p = pairs[pairs.cp != 'original'].sort_values(['ch', 'n'], ascending=[True, False], kind='stable')
    tot = pairs.groupby('ch').n.sum()
    top = p.drop_duplicates('ch')
    char_series = {}
    char_n = {}
    for ch, cp, n in zip(top.ch, top.cp, top.n):
        if n >= 3 and n >= 0.3 * tot[ch]:
            k = alnum(ch)
            if k not in char_n or n > char_n[k]:
                char_series[k], char_n[k] = cp, n
    # fallback for names without the danbooru qualifier: "emilia" -> "emilia_(re:zero)" when one
    # qualified tag dominates the base name
    base = p.assign(base=p.ch.str.replace(r'_\([^()]*\)$', '', regex=True))
    base = base[base.base != base.ch].groupby(['base', 'cp']).n.sum().reset_index()
    base_tot = base.groupby('base').n.sum()
    base = base.sort_values(['base', 'n'], ascending=[True, False], kind='stable').drop_duplicates('base')
    for b, cp, n in zip(base.base, base.cp, base.n):
        k = alnum(b)
        if k not in char_series and n >= 10 and n >= 0.6 * base_tot[b]:
            char_series[k] = cp
    return char_series, copyright_key


def ccip_class_series(classes, char_series, copyright_key):
    """Majority vote (weighted by member image count) over per-member series candidates."""
    def canon_copyright(s):
        return copyright_key.get(alnum(s))

    def generic(s):
        return folder_series(s) in GENERIC

    out = {}
    for c in classes:
        votes = Counter()
        for m in c['members']:
            src, key, n = m['source'], m['key'], max(int(m.get('n', 1)), 1)
            cands = []  # (series, weight)
            if src == 'zc800k':
                cands.append((char_series.get(alnum(key)), CHAR_LOOKUP_WEIGHT))
                mm = PAREN.match(key)
                if mm and not generic(mm.group(2)):
                    cands.append((canon_copyright(mm.group(2)), 1.0))
                if mm:
                    cands.append((char_series.get(alnum(mm.group(1))), CHAR_LOOKUP_WEIGHT))
            elif src == 'v2':
                parts = key.split('/')
                top = parts[0]
                if not generic(top):
                    cands.append((canon_copyright(top) or folder_series(top), 1.0))
                cands.append((char_series.get(alnum(parts[-1])), CHAR_LOOKUP_WEIGHT))
            elif src == 'v1':
                top, _, ch = key.partition('/')
                cands.append((V1_ALIAS.get(top), 1.0))
                cands.append((char_series.get(alnum(ch)), CHAR_LOOKUP_WEIGHT))
            for cand, wt in cands:
                if cand:
                    votes[cand] += n * wt
        if not votes:
            mm = PAREN.match(c['name'])
            cand = char_series.get(alnum(c['name'])) or (
                canon_copyright(mm.group(2)) if mm and not generic(mm.group(2)) else None)
            if cand:
                votes[cand] += 1
        # deterministic: highest vote, then name
        out[c['label']] = min(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0] if votes else None
    return out


def build_ccip(out_dir: Path = CCIP_INDEX, root: Path = CCIP_ROOT, workers: int = 16):
    classes = json.load(open(root / 'classes.json'))
    char_series, copyright_key = danbooru_tag_maps()
    series = ccip_class_series(classes, char_series, copyright_key)
    names = {c['label']: c['name'] for c in classes}
    n_cls = sum(1 for v in series.values() if v)
    print(f'series derived for {n_cls}/{len(classes)} classes, {len(set(v for v in series.values() if v))} series')
    series_codes = {s: i for i, s in enumerate(sorted({v for v in series.values() if v}))}
    stats = {}
    for split in ['train', 'val', 'test']:
        df = build_split_index(root, split, workers=workers)
        df['series'] = df.label.map(series)
        df['series_id'] = df.series.map(series_codes).fillna(-1).astype('int64')
        assert (df.name == df.label.map(names)).all()
        write_parquet(df, out_dir / f'{split}.parquet')
        stats[split] = {'samples': len(df), 'classes': int(df.label.nunique()),
                        'frac_samples_with_series': float(df.series.notna().mean()),
                        'frac_classes_with_series': float(df.groupby('label').series.first().notna().mean())}
    # unmerged variants (gender swap / form change / ...) as explicit hard-negative class pairs
    key_to_label = {(m['source'], m['key']): c['label'] for c in classes for m in c['members']}
    name_to_label = {c['name']: c['label'] for c in classes}
    rows = []
    for v in json.load(open(root / 'variants.json')):
        la = key_to_label.get((v['sources'][0], v['a']), name_to_label.get(v['a']))
        lb = key_to_label.get((v['sources'][1], v['b']), name_to_label.get(v['b']))
        rows.append({'a': v['a'], 'b': v['b'], 'label_a': la, 'label_b': lb, 'merged': v['merged'],
                     'category': v['category'], 'visual': v.get('visual')})
    vp = pd.DataFrame(rows)
    vp['label_a'] = vp.label_a.astype('Int64')
    vp['label_b'] = vp.label_b.astype('Int64')
    write_parquet(vp, out_dir / 'variant_pairs.parquet')
    cs = pd.DataFrame({'label': list(series), 'name': [names[k] for k in series],
                       'series': list(series.values())})
    cs['series_id'] = cs.series.map(series_codes).fillna(-1).astype('int64')
    write_parquet(cs, out_dir / 'class_series.parquet')
    stats['series'] = {'classes_with_series': n_cls, 'num_series': len(series_codes),
                       'variant_pairs_unmerged_resolved': int(((~vp.merged) & vp.label_a.notna()
                                                               & vp.label_b.notna()).sum())}
    stats['source_root'] = str(root)
    (out_dir / 'index_stats.json').write_text(json.dumps(stats, indent=2) + '\n')
    (out_dir / 'README.md').write_text(CCIP_README)
    print(json.dumps(stats, indent=2))


CCIP_README = """# ccip-merged v1 random-access index

Offsets into the tar shards of `datasets/khub/cxip-ccip-merged/v1` (local copy:
/data/neko/cxip_repro/data/ccip_merged_nd). `shard` paths are relative to the dataset root.

- `{train,val,test}.parquet`: sample_key, shard, offset, size, ext, label, name, source, orig_cid, zid,
  orig_path, crop, width, height, image_sha256, series (danbooru-style copyright tag, class-level
  majority vote over members; null if unknown), series_id (-1 if unknown)
- `class_series.parquet`: label, name, series, series_id
- `variant_pairs.parquet`: variant pairs from variants.json with labels; `merged == False` rows are
  distinct classes that look alike (gender swap, form change, ...) -> hard negatives
- built by `v3/data/build_index.py ccip` (cxip repo)

```python
from v3.data.indexed_tar import IndexedTarDataset
ds = IndexedTarDataset('/data/neko/cxip_repro/data/ccip_merged_nd',
                       '/data/neko/cxip_v3/data/ccip-merged-v1-index/index/train.parquet', columns=['series_id'])
```
"""


def build_danbooru(root: Path = DANBOORU_ROOT, workers: int = 32):
    df = build_split_index(root, 'all', list_cols=('characters', 'copyrights'), workers=workers)
    df['danbooru_id'] = df['danbooru_id'].astype('int64')
    write_parquet(df, root / 'index' / 'all.parquet')
    # convenience split for CSIP training (design doc 2.2): drop the 0918 test artists and the
    # csip_eval_public artists entirely; labels stay global (see classes.json)
    train = df[~df.artist_0918_test & ~df.csip_eval_public_artist]
    write_parquet(train, root / 'index' / 'train.parquet')
    test100 = df[df.split_0918 == 'test100']
    write_parquet(test100, root / 'index' / 'test100_0918.parquet')
    stats = {
        'all': {'samples': len(df), 'classes': int(df.label.nunique())},
        'train': {'samples': len(train), 'classes': int(train.label.nunique())},
        'test100_0918': {'samples': len(test100), 'classes': int(test100.label.nunique())},
        'frac_with_characters': float((df.characters.str.len() > 0).mean()),
        'frac_with_copyrights': float((df.copyrights.str.len() > 0).mean()),
        'frac_with_year': float(df.year.notna().mean()),
    }
    (root / 'index' / 'index_stats.json').write_text(json.dumps(stats, indent=2) + '\n')
    print(json.dumps(stats, indent=2))


def danbooru_v2_train(df: pd.DataFrame, exclude: set[str], min_n: int) -> pd.DataFrame:
    """Training rows: artist not in the benchmark exclusion list and >= ``min_n`` local posts. Asserts the result is
    disjoint from the exclusion list and holds no 0918-test / public / unseen-v1 evaluation artist."""
    t = df[~df.artist.isin(exclude) & (df.n_artist_local >= min_n)]
    assert not t.artist.isin(exclude).any(), 'train index intersects exclude_danbooru_artists'
    leak = t[t[EVAL_FLAGS_V2].any(axis=1)]
    assert leak.empty, f'{leak.artist.nunique()} evaluation-flagged artists would enter training'
    return t


def build_danbooru_v2(root: Path = DANBOORU_V2_ROOT, splits: Path = BENCH2_SPLITS, workers: int = 32):
    out = root / 'index'
    if (out / 'index_stats.json').exists():
        raise SystemExit(f'{out} already built (index_stats.json exists); not overwriting')
    exclude = set(json.load(open(splits))['exclude_danbooru_artists'])
    df = build_split_index(root, 'all', list_cols=('characters', 'copyrights'), workers=workers)
    df['danbooru_id'] = df['danbooru_id'].astype('int64')
    write_parquet(df, out / 'all.parquet')
    stats = {'all': {'samples': len(df), 'classes': int(df.label.nunique())},
             'exclude_danbooru_artists': len(exclude),
             'excluded_present': int(df.artist.isin(exclude).groupby(df.artist).first().sum())}
    for name, min_n in [('train_ge30', 30), ('train_ge10', 10)]:
        t = danbooru_v2_train(df, exclude, min_n)
        write_parquet(t, out / f'{name}.parquet')
        stats[name] = {'samples': len(t), 'classes': int(t.label.nunique()), 'min_n_artist_local': min_n}
    stats['source_root'] = str(root)
    stats['splits'] = str(splits)
    (out / 'index_stats.json').write_text(json.dumps(stats, indent=2) + '\n')
    print(json.dumps(stats, indent=2))


def build_pixiv_v1(root: Path = PIXIV_V1_ROOT, splits: Path = BENCH2_SPLITS, workers: int = 32):
    """pixiv-artist-v1 -> index/all.parquet (every image) and index/train.parquet (eval_excluded == False). Asserts
    the training rows hold no account of ``exclude_pixiv_users`` (ints; ``b3p2`` keys are strings elsewhere)."""
    out = root / 'index'
    if (out / 'index_stats.json').exists():
        raise SystemExit(f'{out} already built (index_stats.json exists); not overwriting')
    exclude = {int(u) for u in json.load(open(splits))['exclude_pixiv_users']}
    df = build_split_index(root, 'all', workers=workers)
    df['eval_excluded'] = df.eval_excluded.astype(bool)
    df['user_id'] = df.user_id.astype('int64')
    train = df[~df.eval_excluded]
    assert not set(train.user_id) & exclude, 'train index intersects exclude_pixiv_users'
    assert df[df.user_id.isin(exclude)].eval_excluded.all(), 'exclude_pixiv_users row not flagged eval_excluded'
    write_parquet(df, out / 'all.parquet')
    write_parquet(train, out / 'train.parquet')
    pure = df.artist.str.startswith('pixiv:')
    stats = {'all': {'samples': len(df), 'classes': int(df.label.nunique()),
                     'classes_mapped': int(df[~pure].label.nunique()), 'classes_pure_pixiv': int(df[pure].label.nunique())},
             'train': {'samples': len(train), 'classes': int(train.label.nunique()),
                       'classes_mapped': int(train[~train.artist.str.startswith('pixiv:')].label.nunique()),
                       'classes_pure_pixiv': int(train[train.artist.str.startswith('pixiv:')].label.nunique())},
             'eval_excluded': {'samples': int(df.eval_excluded.sum()), 'accounts': int(df[df.eval_excluded].user_id.nunique())},
             'exclude_pixiv_users': len(exclude), 'source_root': str(root), 'splits': str(splits)}
    (out / 'index_stats.json').write_text(json.dumps(stats, indent=2) + '\n')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['ccip', 'danbooru', 'danbooru_v2', 'pixiv_v1', 'generic'])
    ap.add_argument('--root', type=Path)
    ap.add_argument('--split')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    if a.cmd == 'ccip':
        build_ccip(a.out or CCIP_INDEX, a.root or CCIP_ROOT, a.workers)
    elif a.cmd == 'danbooru':
        build_danbooru(a.root or DANBOORU_ROOT, a.workers)
    elif a.cmd == 'danbooru_v2':
        build_danbooru_v2(a.root or DANBOORU_V2_ROOT, workers=max(a.workers, 32))
    elif a.cmd == 'pixiv_v1':
        build_pixiv_v1(a.root or PIXIV_V1_ROOT, workers=max(a.workers, 32))
    else:
        write_parquet(build_split_index(a.root, a.split, workers=a.workers), a.out)
