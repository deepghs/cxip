"""pixiv-artist-v1: Pixiv works that are not on Danbooru, labelled by artist (CSIP v4 training data).

Source: the s1 Pixiv tars (``PIXIV_RAW``) plus the work / account tables from ``v3/data/pixiv_meta.py``. Kept: works not
on Danbooru of (a) accounts mapped to a current Danbooru artist name and (b) pure-Pixiv accounts (no Danbooru
counterpart) with >= 30 works. Class name = mapped Danbooru artist name, or ``pixiv:{user_id}``. Accounts linked to the
benchmark (``exclude_pixiv_users`` in splits.json) stay in the dataset (``eval_excluded`` = True, B3 needs them) but are
left out of ``index/train.parquet``. Images are resized (short side <= 640, webp q95) into one part tar per source tar
and packed as the ``cxip-pixiv-artist/v1`` neko-data dataset (single split ``all``, label = index in the name-sorted
class list, stream order = classes shuffled with ``seed``, image ids ascending within a class).

    python -m v3.data.build_pixiv_artist prepare
    python -m v3.data.build_pixiv_artist jobs                      # WORK/jobs.parquet (src_tar, member, key)
    python -m v3.data.build_pixiv_artist resize --workers 48       # WORK/resized/*.tar + *.parquet (resumable)
    python -m v3.data.build_pixiv_artist build                     # neko-data build (resumable)
    python -m v3.data.build_pixiv_artist verify
    python -m v3.data.build_index pixiv_v1                         # OUT/index/{all,train}.parquet
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

V4 = Path('/scratch/cxip_v4')
PIXIV_RAW = V4 / 'data/pixiv-s1'
WORK = V4 / 'data/pixiv-artist-v1-work'
OUT = V4 / 'data/pixiv-artist-v1'
RESIZED = WORK / 'resized'
BENCH2_SPLITS = V4 / 'data/bench2_v1/splits.json'
DATASET_ID = 'cxip-pixiv-artist'
VERSION = 'v1'
SPLIT = 'all'
MIN_PURE_WORKS = 30
N_B3P2 = 300                       # pure-Pixiv accounts held out as B3-P2 (spec value)


def select_works(works: pd.DataFrame, accounts: pd.DataFrame, exclude_users: set) -> pd.DataFrame:
    """Works not on Danbooru of mapped accounts (``artist`` not null) and pure-Pixiv accounts with >= 30 works.
    Adds ``artist`` (class name: mapped name or ``pixiv:{user_id}``) and ``eval_excluded`` (account in
    ``exclude_users``; such rows are kept). Row order of ``works`` is preserved."""
    mapped = accounts.artist.notna()
    pure = accounts.pixiv_only.astype(bool) & (accounts.n_works >= MIN_PURE_WORKS) & ~mapped
    acc = accounts.loc[mapped | pure, ['user_id', 'artist']].copy()
    acc['artist'] = [a if isinstance(a, str) else f'pixiv:{u}' for a, u in zip(acc.artist, acc.user_id)]
    s = works[~works.on_danbooru.astype(bool)].merge(acc, on='user_id', how='inner')
    ex = {int(u) for u in exclude_users}
    s['eval_excluded'] = s.user_id.astype('int64').isin(ex)
    return s.reset_index(drop=True)


def _exclude_users(splits: Path = BENCH2_SPLITS) -> set[int]:
    return {int(u) for u in json.load(open(splits))['exclude_pixiv_users']}


def prepare():
    if (WORK / 'selected.parquet').exists() or (WORK / 'manifest.json').exists():
        sys.exit(f'{WORK} already has prepared output (selected.parquet); bump the version instead of overwriting')
    t0 = time.time()
    works = pd.read_parquet(WORK / 'works.parquet')
    accounts = pd.read_parquet(WORK / 'accounts.parquet')
    exclude = _exclude_users()
    s = select_works(works, accounts, exclude)
    assert len(s), 'no works selected'
    assert not s.image_id.duplicated().any(), 'duplicate image ids'

    # every source tar must be downloaded and sha256-verified
    done = {json.loads(l)['name'] for l in open(PIXIV_RAW / 'done.jsonl') if l.strip()}
    need = sorted({Path(t).name for t in s.tar.unique()})
    missing = [n for n in need if n not in done or not (PIXIV_RAW / 'images' / n).exists()]
    if missing:
        sys.exit(f'{len(missing)} source tars not downloaded/verified: {missing}')

    s['date_day'] = s.date_day.astype('Int64')
    cols = ['image_id', 'artwork_id', 'page', 'user_id', 'artist', 'eval_excluded', 't_year', 'date_day',
            'date_source', 'medium', 'x_restrict', 'tar', 'offset', 'size']
    s = s[cols].sort_values('image_id').reset_index(drop=True)
    pure = s.artist.str.startswith('pixiv:')
    acc_pure = accounts[accounts.pixiv_only.astype(bool) & (accounts.n_works >= MIN_PURE_WORKS) & accounts.artist.isna()]
    stats = {
        'n_images': len(s), 'n_classes': int(s.artist.nunique()), 'n_accounts': int(s.user_id.nunique()),
        'n_classes_mapped': int(s[~pure].artist.nunique()), 'n_classes_pure': int(s[pure].artist.nunique()),
        'n_accounts_mapped': int(s[~pure].user_id.nunique()), 'n_accounts_pure': int(s[pure].user_id.nunique()),
        'n_pure_eligible_accounts': int(len(acc_pure)),
        'n_images_eval_excluded': int(s.eval_excluded.sum()), 'n_accounts_eval_excluded': int(s[s.eval_excluded].user_id.nunique()),
        'n_classes_eval_excluded_only': int((s.groupby('artist').eval_excluded.all()).sum()),
        'n_classes_train': int(s[~s.eval_excluded].artist.nunique()),
        'n_classes_train_pure': int(s[~s.eval_excluded & pure].artist.nunique()),
        'n_images_train': int((~s.eval_excluded).sum()), 'n_src_tars': len(need),
        'n_exclude_pixiv_users': len(exclude), 'date_day_null': int(s.date_day.isna().sum()),
        'runtime_s': round(time.time() - t0),
    }
    WORK.mkdir(parents=True, exist_ok=True)
    tmp = WORK / 'selected.parquet.tmp'
    s.to_parquet(tmp, index=False)
    os.replace(tmp, WORK / 'selected.parquet')
    json.dump(stats, open(WORK / 'select_stats.json', 'w'), indent=1)
    print(json.dumps(stats, indent=1), flush=True)


# ---------------------------------------------------------------------------------------------
# resize jobs -> parts -> plan -> neko-data build
# ---------------------------------------------------------------------------------------------

def make_jobs(sel: pd.DataFrame) -> pd.DataFrame:
    """(src_tar, member, key) per image: member ``{image_id}.webp`` of its source tar, output key ``{image_id}``."""
    j = pd.DataFrame({'src_tar': sel.tar.astype(str).to_numpy(), 'member': sel.image_id.astype(str) + '.webp',
                      'key': sel.image_id.astype(str), 'id': sel.image_id.astype('int64')})
    return j.sort_values(['src_tar', 'id']).reset_index(drop=True)


def make_plan(sel: pd.DataFrame, parts: pd.DataFrame, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join the selection with the resized parts. Labels = index in the name-sorted list of all class names; stream
    order = classes shuffled (``seed``), image ids ascending within a class. Images whose resize failed go to the
    returned error table; every image must have a parts row."""
    names = sorted(sel.artist.unique().tolist())
    label_of = {a: i for i, a in enumerate(names)}
    order = list(names)
    random.Random(f'{seed}-{SPLIT}').shuffle(order)
    rank = {a: i for i, a in enumerate(order)}
    pp = parts[['key', 'part', 'offset', 'size', 'width', 'height', 'error']].rename(
        columns={'offset': 'p_offset', 'size': 'p_size'})
    assert not pp.key.duplicated().any(), 'duplicate keys in parts'
    m = sel.drop(columns=['tar', 'offset', 'size']).assign(key=sel.image_id.astype(str)).merge(
        pp, on='key', how='left', indicator=True)
    assert (m._merge == 'both').all(), f'{int((m._merge != "both").sum())} images have no resize result'
    m['label'] = m.artist.map(label_of).astype('int64')
    bad = m.error.notna()
    errors = m.loc[bad, ['image_id', 'artist', 'part', 'error']].sort_values('image_id').reset_index(drop=True)
    plan = m[~bad].assign(_rank=lambda d: d.artist.map(rank)).sort_values(['_rank', 'image_id'])
    plan = plan.drop(columns=['_rank', '_merge', 'key', 'error']).rename(
        columns={'p_offset': 'offset', 'p_size': 'size'}).reset_index(drop=True)
    for c in ['offset', 'size', 'width', 'height']:
        plan[c] = plan[c].astype('int64')
    return plan, errors


def jobs():
    p = WORK / 'jobs.parquet'
    if p.exists():
        print(f'{p} exists ({len(pd.read_parquet(p, columns=["key"]))} rows); keeping it', flush=True)
        return
    j = make_jobs(pd.read_parquet(WORK / 'selected.parquet', columns=['image_id', 'tar']))
    missing = [t for t in j.src_tar.unique() if not os.path.exists(t)]
    assert not missing, f'{len(missing)} source tars missing, e.g. {missing[:3]}'
    tmp = p.with_name(p.name + '.tmp')
    j.to_parquet(tmp, index=False)
    os.replace(tmp, p)
    print(f'jobs: {len(j)} images in {j.src_tar.nunique()} source tars -> {p}', flush=True)


def resize(workers: int = 48, limit_tars: int = 0):
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


def _none_if_na(x, f):
    return None if pd.isna(x) else f(x)


def _md(r: dict) -> dict:
    day = _none_if_na(r['date_day'], int)
    return {'label': int(r['label']), 'artist': r['artist'], 'user_id': int(r['user_id']),
            'artwork_id': int(r['artwork_id']), 'family': int(r['artwork_id']), 'page': int(r['page']),
            'date': None if day is None else pd.Timestamp(day * 86400, unit='s', tz='UTC').isoformat(),
            'date_source': r['date_source'], 't_year': _none_if_na(r['t_year'], float), 'date_day': day,
            'medium': r['medium'], 'x_restrict': int(r['x_restrict']), 'eval_excluded': bool(r['eval_excluded'])}


def iter_samples(plan: pd.DataFrame, out: Path, threads: int = 32, window: int = 4096):
    """Ordered NormalizedSample stream; part preads run in a thread pool with a bounded window. On resume the first
    ``completed`` samples (already in the build journal) are yielded as cheap placeholders that neko-data skips."""
    from neko_data.contract.records import NormalizedSample
    from v3.data.build_danbooru_artist_v2 import _completed_samples, _pread
    completed = _completed_samples(Path(out))
    print(f'stream: {len(plan)} samples, {completed} already in journal', flush=True)
    for _ in range(completed):
        yield None
    rows = plan.iloc[completed:].to_dict('records')

    def sample(r, data):
        return NormalizedSample(sample_key=f'p{r["image_id"]}', image=data, image_extension='webp', split=SPLIT,
                                source_id='pixiv-s1-4mpixel-w640', caption=r['artist'], metadata=_md(r),
                                width=int(r['width']), height=int(r['height']))

    t0, n = time.time(), 0
    with ThreadPoolExecutor(threads) as ex:
        it = iter(rows)
        futures = []
        for r in it:
            futures.append((r, ex.submit(_pread, r['part'], r['offset'], r['size'], RESIZED)))
            if len(futures) >= window:
                break
        head = 0
        while head < len(futures):
            r, fut = futures[head]
            futures[head] = None
            head += 1
            nxt = next(it, None)
            if nxt is not None:
                futures.append((nxt, ex.submit(_pread, nxt['part'], nxt['offset'], nxt['size'], RESIZED)))
            yield sample(r, fut.result())
            n += 1
            if n % 100_000 == 0:
                el = time.time() - t0
                print(f'{completed + n}/{len(plan)} {n / el:.0f} img/s elapsed {el / 60:.1f} min', flush=True)
            if head > 65536:
                futures = futures[head:]
                head = 0


def build(max_shard_size: str = '1GiB', seed: int = 0, threads: int = 32, out: Path | None = None):
    from neko_data.build.job import BuildJob, build_dataset
    out = out or OUT
    if (out / 'manifest.json').exists():
        sys.exit(f'{out} already has manifest.json; bump the version instead of overwriting')
    plan_path = WORK / 'plan.parquet'
    sel = pd.read_parquet(WORK / 'selected.parquet')
    if plan_path.exists():
        plan = pd.read_parquet(plan_path)
        print(f'plan exists: {len(plan)} samples (resume)', flush=True)
    else:
        parts = load_parts(pd.read_parquet(WORK / 'jobs.parquet', columns=['src_tar']))
        plan, errors = make_plan(sel, parts, seed)
        errors.to_csv(WORK / 'resize_errors.csv', index=False)
        tmp = plan_path.with_name(plan_path.name + '.tmp')
        plan.to_parquet(tmp, index=False)
        os.replace(tmp, plan_path)
        print(f'plan: {len(plan)} samples, {plan.label.nunique()} classes; resize errors skipped {len(errors)} '
              f'-> resize_errors.csv', flush=True)
    out.mkdir(parents=True, exist_ok=True)
    names = sorted(sel.artist.unique().tolist())
    if not (out / 'classes.json').exists():
        n = plan.groupby('label').size()
        g = sel.groupby('artist')
        n_acc, n_ex = g.user_id.nunique(), g.eval_excluded.sum()
        classes = [{'label': i, 'name': a, 'n': int(n.get(i, 0)), 'n_accounts': int(n_acc[a]),
                    'pure_pixiv': a.startswith('pixiv:'), 'n_eval_excluded': int(n_ex[a])}
                   for i, a in enumerate(names)]
        (out / 'classes.json').write_text(json.dumps(classes, ensure_ascii=False) + '\n', encoding='utf-8')
    del sel
    meta = {
        'task': 'csip', 'num_classes': len(names), 'label_key': 'label', 'group_key': 'label',
        'classes_file': 'classes.json', 'seed': seed,
        'source': 'deepghs/pixiv_ai_s1-webp-4Mpixel works that are not on Danbooru; accounts mapped to a current '
                  'Danbooru artist name (class = that name) or pure-Pixiv accounts with >= 30 works (class = '
                  'pixiv:{user_id}); images resized to short side <= 640 (BICUBIC, never upscaled), webp q95, alpha kept',
        'split_columns': ['eval_excluded'],
        'date': 'pixiv artwork time (exact / monotone interpolation), date_day = days since 1970-01-01 UTC',
    }
    m = build_dataset(BuildJob(
        source=iter_samples(plan, out=out, threads=threads), output_dir=out, dataset_id=DATASET_ID, version=VERSION,
        split=SPLIT, max_shard_size=max_shard_size, group_key='label', resume=True, skip_invalid_samples=False,
        metadata=meta,
    ))
    print('done', m.metadata.get('split_stats'), flush=True)


def check_train_disjoint(df: pd.DataFrame, exclude: set[int], what: str = 'train'):
    """No evaluation-linked account in a training table (Review Focus 4)."""
    hit = set(df.user_id.astype('int64')) & {int(u) for u in exclude}
    assert not hit, f'{what} contains {len(hit)} accounts of exclude_pixiv_users'


def verify(out: Path | None = None, n_check: int = 2000):
    """Manifest vs files, sample count vs successfully resized plan rows, order, class contiguity, training-set
    disjointness from exclude_pixiv_users, pure-Pixiv class budget, byte-exact comparison of random samples."""
    from v3.data.build_danbooru_artist_v2 import _pread
    from v3.data.resize_parts import iter_tar_members
    out = out or OUT
    m = json.load(open(out / 'manifest.json'))
    shards = [s for s in m['shards'] if s['split'] == SPLIT]
    plan = pd.read_parquet(WORK / 'plan.parquet')
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
    assert meta.sample_key.tolist() == ('p' + plan.image_id.astype(str)).tolist(), 'order mismatch vs plan'
    assert (meta.label.to_numpy() == plan.label.to_numpy()).all(), 'label mismatch vs plan'
    runs = int(meta.label.ne(meta.label.shift()).sum())
    per_label = meta.groupby('label').shard_path.nunique()
    n_cls = len(json.load(open(out / 'classes.json')))
    print(f'labels {meta.label.nunique()} (classes.json {n_cls}), label runs {runs}, '
          f'labels split across shards {(per_label > 1).sum()}', flush=True)
    assert runs == meta.label.nunique(), 'classes not contiguous'
    assert (per_label == 1).all(), 'class split across shards'

    # ---- evaluation-account leakage and pure-Pixiv class budget
    exclude = _exclude_users()
    train = plan[~plan.eval_excluded]
    check_train_disjoint(train, exclude, 'plan train rows')
    assert plan[plan.user_id.isin(exclude)].eval_excluded.all(), 'exclude account row not flagged eval_excluded'
    tp = out / 'index' / 'train.parquet'
    if tp.exists():
        ti = pd.read_parquet(tp, columns=['sample_key', 'user_id', 'eval_excluded'])
        check_train_disjoint(ti, exclude, 'index/train.parquet')
        assert not ti.eval_excluded.any()
        assert len(ti) == len(train), 'index/train.parquet row count differs from plan train rows'
        print(f'index/train.parquet: {len(ti)} rows, disjoint from exclude_pixiv_users', flush=True)
    else:
        print('index/train.parquet not built yet (re-run verify after build_index pixiv_v1)', flush=True)
    accounts = pd.read_parquet(WORK / 'accounts.parquet')
    elig = accounts[accounts.pixiv_only.astype(bool) & (accounts.n_works >= MIN_PURE_WORKS) & accounts.artist.isna()]
    n_pure_train = int(train[train.artist.str.startswith('pixiv:')].artist.nunique())
    budget = len(elig) - N_B3P2
    n_b3 = int(elig.user_id.isin(exclude).sum())
    print(f'pure-Pixiv classes in train {n_pure_train}; eligible accounts {len(elig)} (in exclude {n_b3}); '
          f'budget eligible-{N_B3P2} = {budget}', flush=True)
    assert n_b3 >= N_B3P2, 'fewer than 300 B3-P2 accounts among the eligible pure-Pixiv accounts'
    assert n_pure_train <= budget, 'too many pure-Pixiv classes in the training set'
    mapped = plan[~plan.artist.str.startswith('pixiv:')]
    print(f'classes: mapped {mapped.artist.nunique()}, pure-Pixiv {plan.artist.nunique() - mapped.artist.nunique()}; '
          f'eval_excluded images {int(plan.eval_excluded.sum())} of {len(plan)}', flush=True)

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
            assert _pread(path, o, sz, out) == _pread(r.part, int(r.offset), int(r.size), RESIZED), r.sample_key
            n += 1
    print(f'byte-exact checks OK ({n} samples)', flush=True)
    print('VERIFY OK', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prepare', 'jobs', 'resize', 'build', 'verify'])
    ap.add_argument('--workers', type=int, default=48)
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
