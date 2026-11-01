"""Benchmark v2 evaluation sets B1 / B2 / B3-P1 / B3-P2 (CSIP v4 P0): pick images, write index parquets, make the
cleaning-injection folders.

The B3-P1 / B3-P2 sets (bench2_b3p1 / bench2_b3p2 and their _sealed dirs) are unused in v4: the s1 source is an
AI-image corpus (see docs/csip_v4_p0_baseline_zh.md §6.1). Pixiv was dropped from CSIP v4 at the P0 checkpoint
(2026-11-01); the B3 index parquets still on disk point into deleted tars and are not registered in
v3.eval.evalsets.INDEX_SETS. The B1 / B2 parts are in use.

The sets are index parquets that point into the existing danbooru-artist-v2 / pixiv-artist-v1 tars
(path = tar, offset = data offset inside the tar, size); no image is copied.

  B1    = splits.json b1 artists, Danbooru images only, <= 80 per artist
  B2    = splits.json b2 artists, Danbooru images, <= 60 per artist
  B3-P2 = splits.json b3p2 Pixiv-only accounts, one page per artwork (smallest page), <= 60 per account
  B3-P1 = b3p1_accounts (Pixiv accounts of B1 artists), works not on Danbooru (pixiv-artist-v1 holds only those),
          one page per artwork, <= 40 per B1 artist (pooled over the artist's accounts); label / label_name / split are
          those of the matching B1 artist, so B1 + B3-P1 concatenate.
Images per artist are picked evenly along time (spread_pick on t_year after a seeded shuffle that only breaks ties).
BENCH/{set} holds dev + test rows only, BENCH/{set}_sealed the sealed rows; each has index.parquet + manifest.json.

    python -m v3.data.bench2_sets build
    python -m v3.data.bench2_sets inject     # BENCH/inject/bench2_{b1,b2}_{same_char,far_period,bad_date}.parquet
    python -m v3.data.bench2_sets inject-near --feats FEATS_DIR
        # BENCH/inject/bench2_{b1,b2}_near_style.parquet (+ manifest_near_style.json); FEATS_DIR/bench2_{b1,b2}.npz are
        # features of a model that is not evaluated (CSIP v2.1), read with csip_full_eval.load_aligned
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

V4 = Path('/scratch/cxip_v4')
BENCH = V4 / 'data/bench2_v1'
DV2 = V4 / 'data/danbooru-artist-v2'
PX1 = V4 / 'data/pixiv-artist-v1'
SEED = 0
CAPS = dict(b1=80, b2=60, b3p1=40, b3p2=60)
SETS = ('b1', 'b2', 'b3p1', 'b3p2')
REGULAR = ('dev', 'test')
FEW_IMAGES = 30
PIXIV_FAMILY_OFFSET = 10 ** 10            # Pixiv family = artwork_id + offset, never equal to a Danbooru post id
INJECT_SETS = ('b1', 'b2')
INJECT_KINDS = ('same_char', 'far_period', 'bad_date')   # near_style needs features: make_inject_near
INJECT_FRAC = 0.05
LIST_COLS = ('characters', 'copyrights')
COLUMNS = ['path', 'tar', 'offset', 'size', 'sample_key', 'label', 'label_name', 'id', 'source', 'split', 'family',
           't_year', 'date_day', 'date_source', 'characters', 'copyrights', 'medium', 'rating', 'user_id']
DV2_COLS = ['sample_key', 'shard', 'offset', 'size', 'danbooru_id', 'artist', 't_year', 'date_day', 'date_source',
            'family', 'medium', 'rating', 'characters', 'copyrights']
PX1_COLS = ['sample_key', 'shard', 'offset', 'size', 'artwork_id', 'user_id', 'page', 't_year', 'date_day',
            'date_source', 'medium', 'x_restrict', 'family']
RULES = [
    'B1: splits.json b1 artists; Danbooru images only (danbooru-artist-v2 index/all); <= 80 per artist',
    'B2: splits.json b2 artists; Danbooru images (danbooru-artist-v2 index/all); <= 60 per artist',
    'B3-P2: splits.json b3p2 Pixiv-only accounts (pixiv-artist-v1 index/all); one page per artwork (smallest page); '
    '<= 60 per account; label_name = pixiv:{user_id}',
    'B3-P1: b3p1_accounts (Pixiv accounts of B1 artists); pixiv-artist-v1 holds only works not on Danbooru; one page '
    'per artwork (smallest page); <= 40 per B1 artist pooled over its accounts; label, label_name and split of the '
    'matching B1 artist (B1 dev/test labels in b3p1, B1 sealed labels in b3p1_sealed)',
    'artists with fewer images than the cap keep all of them (B1 / B2 artists with < 30 images in v2 all are counted)',
    'per-artist pick: rows sorted by sample_key, shuffled with the seed (only breaks ties), then spread_pick on '
    't_year: positions floor(linspace(0, n-1, cap) + 0.5) of the time-sorted rows',
    'regular directory = dev + test rows, *_sealed directory = sealed rows; label = 0..n-1 by sorted label_name '
    'within each directory',
    'path = tar = absolute tar path; offset = data offset inside the tar; size = bytes',
    'Pixiv rows: characters / copyrights = empty lists (no tags); rating = g if x_restrict == 0 else e; '
    'family = artwork_id + 1e10 so it never equals a Danbooru family; id = image id (sample_key p{id})',
    'Danbooru rows: sample_key = d{danbooru_id}, id = danbooru_id, family = parent_id or own id (v2), user_id = -1',
]


# ---------------------------------------------------------------------------------------------- picking
def spread_pick(t_year: np.ndarray, n: int) -> np.ndarray:
    """Indices (into t_year) of up to n items spread evenly along time: stable time sort, then positions
    floor(linspace(0, len - 1, n) + 0.5), deduplicated. n >= len returns every index (in time order)."""
    t = np.asarray(t_year, dtype=np.float64)
    m = len(t)
    if n <= 0 or m == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(t, kind='stable')
    if n >= m:
        return order.astype(np.int64)
    pos = np.unique(np.floor(np.linspace(0, m - 1, n) + 0.5).astype(np.int64))
    return order[pos].astype(np.int64)


def pick_per_artist(df: pd.DataFrame, key: str, cap: int, seed: int = SEED) -> pd.DataFrame:
    """At most ``cap`` rows per ``key`` value, spread along t_year. Rows are first put in a fixed order (key,
    sample_key) and shuffled with the seed, so the result does not depend on the input order and ties in t_year are
    broken at random. Output sorted by (key, t_year, sample_key)."""
    d = df.sort_values([key, 'sample_key'], kind='stable').reset_index(drop=True)
    d = d.iloc[np.random.default_rng(seed).permutation(len(d))].reset_index(drop=True)
    t = d['t_year'].to_numpy(np.float64)
    keep = [g[spread_pick(t[g], cap)] for g in d.groupby(key, sort=True).indices.values()]
    idx = np.concatenate(keep) if keep else np.zeros(0, np.int64)
    return d.iloc[idx].sort_values([key, 't_year', 'sample_key'], kind='stable').reset_index(drop=True)


def first_page_per_work(df: pd.DataFrame) -> pd.DataFrame:
    """One row per artwork_id: the smallest page."""
    return df.sort_values(['artwork_id', 'page'], kind='stable').drop_duplicates('artwork_id').reset_index(drop=True)


# ---------------------------------------------------------------------------------------------- standard rows
def _tags(col) -> list:
    return [[str(v) for v in x] if x is not None else [] for x in col]


def std_danbooru(d: pd.DataFrame, root) -> pd.DataFrame:
    """danbooru-artist-v2 index rows -> standard columns (no label / split yet)."""
    tar = [str(Path(root) / s) for s in d['shard']]
    ids = d['danbooru_id'].to_numpy(np.int64)
    return pd.DataFrame(dict(
        path=tar, tar=tar, offset=d['offset'].to_numpy(np.int64), size=d['size'].to_numpy(np.int64),
        sample_key=[f'd{i}' for i in ids], label_name=d['artist'].astype(str).to_numpy(), id=ids, source='danbooru',
        family=d['family'].to_numpy(np.int64), t_year=d['t_year'].to_numpy(np.float64),
        date_day=d['date_day'].to_numpy(np.int64), date_source=d['date_source'].astype(str).to_numpy(),
        characters=_tags(d['characters']), copyrights=_tags(d['copyrights']), medium=d['medium'].to_numpy(),
        rating=d['rating'].to_numpy(), user_id=np.full(len(d), -1, np.int64)))


def std_pixiv(p: pd.DataFrame, root, label_names: pd.Series) -> pd.DataFrame:
    """pixiv-artist-v1 index rows -> standard columns; label_names is aligned with p by position."""
    tar = [str(Path(root) / s) for s in p['shard']]
    keys = p['sample_key'].astype(str)
    if not keys.str.match(r'^p\d+$').all():
        raise ValueError('pixiv sample_key must look like p{image_id}')
    return pd.DataFrame(dict(
        path=tar, tar=tar, offset=p['offset'].to_numpy(np.int64), size=p['size'].to_numpy(np.int64),
        sample_key=keys.to_numpy(), label_name=np.asarray(label_names, dtype=object),
        id=keys.str[1:].astype(np.int64).to_numpy(), source='pixiv',
        family=p['artwork_id'].to_numpy(np.int64) + PIXIV_FAMILY_OFFSET, t_year=p['t_year'].to_numpy(np.float64),
        date_day=p['date_day'].to_numpy(np.int64), date_source=p['date_source'].astype(str).to_numpy(),
        characters=[[] for _ in range(len(p))], copyrights=[[] for _ in range(len(p))],
        medium=p['medium'].to_numpy(), rating=np.where(p['x_restrict'].to_numpy() == 0, 'g', 'e'),
        user_id=p['user_id'].to_numpy(np.int64)))


def _order(df: pd.DataFrame) -> pd.DataFrame:
    return df[[c for c in COLUMNS if c in df.columns] + [c for c in df.columns if c not in COLUMNS]]


def split_and_label(df: pd.DataFrame, split_of: dict, label_of: tuple | None = None):
    """split = split_of[label_name]. Returns (regular = dev + test rows, sealed rows). label = 0..n-1 by sorted
    label_name within each part, or label_of = (regular map, sealed map) from label_name (B3-P1 uses B1's maps)."""
    sp = df['label_name'].map(split_of)
    if sp.isna().any():
        raise ValueError(f'{int(sp.isna().sum())} rows whose label_name has no split')
    bad = set(sp) - set(REGULAR) - {'sealed'}
    if bad:
        raise ValueError(f'unknown splits {bad}')
    df = df.assign(split=sp.to_numpy())
    out = []
    for j, part in enumerate((df[df['split'].isin(REGULAR)], df[df['split'] == 'sealed'])):
        if label_of is None:
            lm = {n: i for i, n in enumerate(sorted(part['label_name'].unique()))}
        else:
            lm = label_of[j]
        lab = part['label_name'].map(lm)
        if lab.isna().any():
            raise ValueError(f'{int(lab.isna().sum())} rows whose label_name has no label')
        out.append(_order(part.assign(label=lab.astype(np.int64).to_numpy()).reset_index(drop=True)))
    return out[0], out[1]


def label_map(df: pd.DataFrame) -> dict:
    return dict(zip(df['label_name'], df['label']))


# ---------------------------------------------------------------------------------------------- writing / summary
def write_set(df: pd.DataFrame, out_dir, manifest: dict) -> None:
    out_dir = Path(out_dir)
    if (out_dir / 'manifest.json').exists() or (out_dir / 'index.parquet').exists():
        raise FileExistsError(f'{out_dir} already written')
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _order(df.reset_index(drop=True))
    tbl = pa.Table.from_pandas(df.drop(columns=[c for c in LIST_COLS if c in df.columns]), preserve_index=False)
    for c in LIST_COLS:
        if c in df.columns:
            arr = pa.array([list(x) for x in df[c]], type=pa.list_(pa.string()))
            tbl = tbl.append_column(c, arr)
    tbl = tbl.select(list(df.columns))
    pq.write_table(tbl, out_dir / 'index.parquet')
    with open(out_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)


QS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


def _q(x) -> dict:
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    return {str(q): (round(float(np.quantile(x, q)), 3) if len(x) else None) for q in QS}


def summarize(df: pd.DataFrame, n_source: dict | None = None) -> dict:
    """Per split: artists, images, per-artist image counts / span (selected images) / t_year quantiles, date_source
    shares; n_source = {label_name: images available} adds how many artists have < FEW_IMAGES available."""
    out = {}
    for sp in sorted(set(df['split'])):
        d = df[df['split'] == sp]
        g = d.groupby('label_name')['t_year']
        n_img = g.size()
        span = g.max() - g.min()
        r = dict(n_artists=int(len(n_img)), n_images=int(len(d)), images_per_artist=_q(n_img),
                 n_artists_lt30_selected=int((n_img < FEW_IMAGES).sum()), span_years=_q(span), t_year=_q(d['t_year']),
                 date_source={k: round(float(v), 4) for k, v in d['date_source'].value_counts(normalize=True).items()},
                 source={k: int(v) for k, v in d['source'].value_counts().items()})
        if n_source is not None:
            avail = np.array([n_source.get(a, 0) for a in n_img.index])
            r['n_artists_lt30_available'] = int((avail < FEW_IMAGES).sum())
        out[sp] = r
    return out


# ---------------------------------------------------------------------------------------------- build
def _set_dirs(root: Path = BENCH) -> list:
    return [root / s for s in SETS] + [root / f'{s}_sealed' for s in SETS]


def _read(path: Path, cols: list, field: str, values) -> pd.DataFrame:
    vals = pa.array(sorted(values), type=pa.string() if field == 'artist' else pa.int64())
    return pq.read_table(path, columns=cols, filters=pc.field(field).isin(vals)).to_pandas()


def build(root: Path = BENCH) -> dict:
    done = [str(p) for p in _set_dirs(root) if (p / 'manifest.json').exists()]
    if done:
        sys.exit(f'already built: {done}; bump the version instead of overwriting')
    t0 = time.time()
    splits = json.load(open(root / 'splits.json'))
    b1, b2 = splits['b1'], splits['b2']
    b3p2 = {f'pixiv:{int(u)}': s for u, s in splits['b3p2'].items()}
    acc = {int(u): a for u, a in splits['b3p1_accounts'].items()}
    if not set(acc.values()) <= set(b1):
        raise ValueError('b3p1 account mapped to a non-B1 artist')

    dv = _read(DV2 / 'index/all.parquet', DV2_COLS, 'artist', set(b1) | set(b2))
    avail_d = dv['artist'].value_counts().to_dict()
    print(f'danbooru rows for B1 + B2 artists: {len(dv)} ({time.time() - t0:.0f}s)', flush=True)
    px = _read(PX1 / 'index/all.parquet', PX1_COLS, 'user_id', {int(u) for u in splits['b3p2']} | set(acc))
    print(f'pixiv rows for B3 accounts: {len(px)} ({time.time() - t0:.0f}s)', flush=True)

    out, extra = {}, {}
    for name, sp in (('b1', b1), ('b2', b2)):
        rows = std_danbooru(dv[dv['artist'].isin(set(sp))], DV2)
        missing = sorted(set(sp) - set(rows['label_name']))
        out[name] = split_and_label(pick_per_artist(rows, 'label_name', CAPS[name]), sp)
        extra[name] = dict(n_artists_listed=len(sp), n_artists_without_images=len(missing),
                           n_rows_available=int(len(rows)), n_source=avail_d)

    p2 = first_page_per_work(px[px['user_id'].isin({int(u) for u in splits['b3p2']})])
    rows = std_pixiv(p2, PX1, 'pixiv:' + p2['user_id'].astype(str))
    out['b3p2'] = split_and_label(pick_per_artist(rows, 'label_name', CAPS['b3p2']), b3p2)
    extra['b3p2'] = dict(n_artists_listed=len(b3p2),
                         n_artists_without_images=len(set(b3p2) - set(rows['label_name'])),
                         n_rows_available=int(len(rows)), n_source=rows['label_name'].value_counts().to_dict())

    p1 = first_page_per_work(px[px['user_id'].isin(set(acc))])
    rows = std_pixiv(p1, PX1, p1['user_id'].map(acc))
    maps = (label_map(out['b1'][0]), label_map(out['b1'][1]))
    out['b3p1'] = split_and_label(pick_per_artist(rows, 'label_name', CAPS['b3p1']), b1, label_of=maps)
    extra['b3p1'] = dict(n_accounts_listed=len(acc), n_accounts_with_works=int(rows['user_id'].nunique()),
                         n_b1_artists_listed=len(set(acc.values())), n_rows_available=int(len(rows)),
                         n_source=rows['label_name'].value_counts().to_dict())

    _check(out)
    summary = {}
    for name in SETS:
        reg, sealed = out[name]
        n_src = extra[name].pop('n_source')
        info = dict(set=name, seed=SEED, cap_per_artist=CAPS[name], rules=RULES, **extra[name])
        for part, d in ((name, reg), (f'{name}_sealed', sealed)):
            m = dict(info, directory=part, splits=summarize(d, n_src), n_images=int(len(d)),
                     n_artists=int(d['label_name'].nunique()))
            write_set(d, root / part, m)
            summary[part] = m
            print(f'{part}: {len(d)} images, {d["label_name"].nunique()} artists, splits '
                  f'{d["split"].value_counts().to_dict()}', flush=True)
    for name in SETS:
        print(f'{name} total (all splits): {len(out[name][0]) + len(out[name][1])} images '
              f'over {out[name][0]["label_name"].nunique() + out[name][1]["label_name"].nunique()} artists')
    print(f'done ({time.time() - t0:.0f}s)')
    return summary


def _check(out: dict) -> None:
    for name, (reg, sealed) in out.items():
        assert set(reg['split']) <= set(REGULAR) and set(sealed['split']) <= {'sealed'}, name
        assert not set(reg['label_name']) & set(sealed['label_name']), name
        assert not reg['sample_key'].duplicated().any() and not sealed['sample_key'].duplicated().any(), name
        assert set(reg['source']) | set(sealed['source']) <= {'danbooru', 'pixiv'}, name
        cap = CAPS[name]
        for d in (reg, sealed):
            assert (d.groupby('label_name').size() <= cap).all(), name
            assert (d.groupby('label_name')['label'].nunique() == 1).all(), name
            if name != 'b3p1':
                assert sorted(d['label'].unique()) == list(range(d['label_name'].nunique())), name
    for j in (0, 1):                     # B3-P1 rows carry the matching B1 artist's label / split
        b1, p1 = out['b1'][j], out['b3p1'][j]
        m = p1.merge(b1.drop_duplicates('label_name')[['label_name', 'label', 'split']], on='label_name',
                     suffixes=('', '_b1'))
        assert len(m) == len(p1) and (m['label'] == m['label_b1']).all() and (m['split'] == m['split_b1']).all()
        p1_fam, b1_fam = p1['family'].to_numpy(), b1['family'].to_numpy()
        assert p1_fam.min(initial=PIXIV_FAMILY_OFFSET) >= PIXIV_FAMILY_OFFSET > b1_fam.max(initial=0)
        assert not set(p1['sample_key']) & set(b1['sample_key'])


# ---------------------------------------------------------------------------------------------- inject
def make_inject(root: Path = BENCH) -> dict:
    from v3.eval import evalsets, inject
    out_dir = root / 'inject'
    if (out_dir / 'manifest.json').exists():
        sys.exit(f'{out_dir} already has a manifest; bump the version instead of overwriting')
    _check_free([out_dir / f'bench2_{s}_{kind}.parquet' for s in INJECT_SETS for kind in INJECT_KINDS])
    out_dir.mkdir(parents=True, exist_ok=True)
    man = dict(seed=SEED, frac=INJECT_FRAC, kinds=list(INJECT_KINDS), splits=list(REGULAR),
               note='folders over the dev + test rows of each set (sealed never used); near_style is made by '
                    'inject-near; folder = set label; bench2 keeps the folders of the evaluated split', files={})
    for s in INJECT_SETS:
        df, _ = evalsets.load_index_set(f'bench2_{s}')
        assert set(df['split']) <= set(REGULAR), f'bench2_{s} holds non dev/test rows'
        split_of = df.drop_duplicates('label').set_index('label')['split']
        for kind in INJECT_KINDS:
            p = out_dir / f'bench2_{s}_{kind}.parquet'
            fo = inject.make_folders(df, kind, frac=INJECT_FRAC, seed=SEED)
            man['files'][p.name] = _write_folders(fo, split_of, p)
    with open(out_dir / 'manifest.json', 'w') as f:
        json.dump(man, f, indent=1)
    return man


def _check_free(paths) -> None:
    """Refuse up front (before any features are loaded or folders built) if an output file already exists."""
    taken = [str(p) for p in paths if Path(p).exists()]
    if taken:
        raise FileExistsError(f'output already exists: {taken}; bump the version instead of overwriting')


def _write_folders(fo: pd.DataFrame, split_of: pd.Series, p: Path) -> dict:
    """Write one folder table (refuses to overwrite); returns its manifest record."""
    if p.exists():
        raise FileExistsError(p)
    st = {k: (v if isinstance(v, str) else int(v)) for k, v in fo.attrs.get('stats', {}).items()}
    fs = fo.drop_duplicates('folder')['folder'].map(split_of).value_counts().to_dict()
    fo = fo.reset_index(drop=True)
    fo.attrs = {}
    fo.to_parquet(p, index=False)
    rec = dict(rows=int(len(fo)), injected=int(fo['injected'].sum()), folders=int(fo['folder'].nunique()),
               folders_by_split={k: int(v) for k, v in fs.items()}, stats=st)
    print(f'{p.name}: {rec}', flush=True)
    return rec


def make_inject_near(feats_dir, root: Path = BENCH) -> dict:
    """near_style folders (spec 5.3 (a)): injected = the other artists' images nearest to each artist's feature
    centroid, mined with features of a model that is not evaluated (FEATS_DIR/bench2_{s}.npz, aligned to the set by
    sample_key).
    Same rows / seed / frac as make_inject: dev + test rows together, so (as for same_char) an injected image can come
    from an artist of the other split. Writes its own manifest_near_style.json (the inject manifest is left as is)."""
    from v3.eval import evalsets, inject
    from v3.scripts.csip_full_eval import load_aligned
    out_dir = root / 'inject'
    man_p = out_dir / 'manifest_near_style.json'
    if man_p.exists():
        sys.exit(f'{man_p} already exists; bump the version instead of overwriting')
    _check_free([out_dir / f'bench2_{s}_near_style.parquet' for s in INJECT_SETS])
    out_dir.mkdir(parents=True, exist_ok=True)
    man = dict(seed=SEED, frac=INJECT_FRAC, kinds=['near_style'], splits=list(REGULAR), feats_dir=str(feats_dir),
               note='folders over the dev + test rows of each set (sealed never used); injected images mined with the '
                    'features in feats_dir (a model that is not evaluated); folder = set label; bench2 keeps the '
                    'folders of the evaluated split', files={})
    for s in INJECT_SETS:
        name = f'bench2_{s}'
        df, _ = evalsets.load_index_set(name)
        assert set(df['split']) <= set(REGULAR), f'{name} holds non dev/test rows'
        feats = load_aligned(feats_dir, name, df)
        assert len(feats) == len(df) and np.isfinite(feats).all(), name
        split_of = df.drop_duplicates('label').set_index('label')['split']
        fo = inject.make_folders(df, 'near_style', frac=INJECT_FRAC, seed=SEED, feats=feats)
        p = out_dir / f'{name}_near_style.parquet'
        man['files'][p.name] = _write_folders(fo, split_of, p)
    with open(man_p, 'w') as f:
        json.dump(man, f, indent=1)
    return man


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'build':
        build()
    elif cmd == 'inject':
        make_inject()
    elif cmd == 'inject-near':
        import argparse
        ap = argparse.ArgumentParser(prog='bench2_sets inject-near')
        ap.add_argument('--feats', required=True, help='dir with bench2_b1.npz / bench2_b2.npz (runner npz format)')
        make_inject_near(ap.parse_args(sys.argv[2:]).feats)
    else:
        sys.exit('usage: python -m v3.data.bench2_sets build|inject|inject-near --feats DIR')
