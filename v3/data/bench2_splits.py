"""Benchmark v2 artist / account splits and training exclusion lists (CSIP v4 P0).

The B3-P2 split and the Pixiv exclusion lists (b3p2, b3p1_accounts, exclude_pixiv_users) are unused in v4: the s1
source is an AI-image corpus (see docs/csip_v4_p0_baseline_zh.md §6.1). Pixiv was dropped from CSIP v4 at the P0
checkpoint (2026-11-01); the B3 index parquets still on disk point into deleted tars and are not registered in
v3.eval.evalsets.INDEX_SETS. The B1 / B2 parts are in use.

B1  = every 0918 test artist (dev / test / sealed, stratified by career span, test100 forced into test)
B2  = artists unseen by v1, Kaloscope and every public eval set (n_local >= 30), 200 per span stratum, then dev/test/sealed
B3-P2 = pixiv-only accounts (n_works >= 30), sampled and split 100 / 100 / 100
Outputs BENCH/{splits.json, manifest.json}; splits.json contains artist names and stays under /scratch.

    python -m v3.data.bench2_splits
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

V4 = Path('/scratch/cxip_v4')
SNAP = '/scratch/anime_tagger/dl/nyanko_danbooru2026/posts-snapshot.parquet'
V1_PLAN = Path('/data/neko/cxip_v3/data/danbooru-artist-v1-work/plan.parquet')
UNSEEN_V1 = Path('/data/neko/cxip_v3/data/csip_unseen_v1/index.parquet')
SPLIT_0918_TEST = Path('/data/neko/cxip_repro/data/danbooru-artist-split-0918/test_split.json')
WORK = V4 / 'data/danbooru-artist-v2-work'
PIXIV_WORK = V4 / 'data/pixiv-artist-v1-work'
BENCH = V4 / 'data/bench2_v1'
SEED = 0
B1_SIZES = dict(dev=300, test=250)          # sealed = remainder
B1_EDGES = (3, 8)
B2_EDGES = (2, 5)
B2_PER_STRATUM = 200
B2_TOTAL = 600
B2_MIN_LOCAL = 30
B3_EDGES = (2, 4)
B3_TOTAL = 300
B3_MIN_WORKS = 30


def span_bin(span: np.ndarray, edges: tuple) -> np.ndarray:
    """Stratum index of each span: 0 below edges[0], ..., len(edges) at or above edges[-1]."""
    return np.digitize(np.asarray(span, dtype=float), np.asarray(edges, dtype=float)).astype(int)


def assign_splits(keys: np.ndarray, strata: np.ndarray, sizes: dict, seed: int, forced: dict | None = None) -> dict:
    """Stratified split with exact global sizes. Forced keys take their split's slots first; the remaining slots are
    handed out one at a time to the (stratum, split) cell furthest below its proportional share, then the keys of each
    stratum are shuffled (seeded) into their cell quotas. sum(sizes) must equal len(keys)."""
    keys = np.asarray(keys); strata = np.asarray(strata)
    names = list(sizes)
    if sum(sizes.values()) != len(keys):
        raise ValueError(f'sizes sum {sum(sizes.values())} != {len(keys)} keys')
    if len(set(keys.tolist())) != len(keys):
        raise ValueError('duplicate keys')
    forced = forced or {}
    pos = {k: i for i, k in enumerate(keys.tolist())}
    zs = sorted(set(strata.tolist())); zi = {z: i for i, z in enumerate(zs)}
    fixed: dict = {}
    for sp, ks in forced.items():
        if sp not in sizes:
            raise ValueError(f'forced split {sp!r} not in sizes')
        for k in ks:
            if k not in pos:
                raise ValueError(f'forced key {k!r} not in keys')
            if k in fixed and fixed[k] != sp:
                raise ValueError(f'key {k!r} forced into two splits')
            fixed[k] = sp
    n_z = np.array([(strata == z).sum() for z in zs], float)
    ideal = n_z[:, None] * np.array([sizes[s] for s in names], float)[None, :] / len(keys)
    q = np.zeros((len(zs), len(names)), int)
    for k, sp in fixed.items():
        q[zi[strata[pos[k]]], names.index(sp)] += 1
    row_left = n_z.astype(int) - q.sum(1); col_left = np.array([sizes[s] for s in names]) - q.sum(0)
    if (row_left < 0).any() or (col_left < 0).any():
        raise ValueError('forced keys exceed a split size')
    while row_left.sum() > 0:
        gap = np.where((row_left[:, None] > 0) & (col_left[None, :] > 0), ideal - q, -np.inf)
        z, s = np.unravel_index(int(np.argmax(gap)), gap.shape)   # first max: deterministic tie-break
        q[z, s] += 1; row_left[z] -= 1; col_left[s] -= 1
    rng = np.random.default_rng(seed)
    out = dict.fromkeys(keys.tolist())
    for z in zs:
        members = sorted(k for k in keys[strata == z].tolist())
        free = [k for k in members if k not in fixed]
        for k in members:
            if k in fixed:
                out[k] = fixed[k]
        need = q[zi[z]].copy()
        for k in members:
            if k in fixed:
                need[names.index(fixed[k])] -= 1
        order = rng.permutation(len(free))
        slots = [names[j] for j in range(len(names)) for _ in range(need[j])]
        for o, sp in zip(order, slots):
            out[free[o]] = sp
    return out


def stratified_draw(strata: np.ndarray, per: int, total: int, seed: int) -> np.ndarray:
    """Indices: up to ``per`` from each stratum; a stratum with fewer than ``per`` is topped up from the nearest
    strata that still have spare candidates, so that exactly ``total`` are drawn."""
    strata = np.asarray(strata)
    zs = sorted(set(strata.tolist()))
    cap = {z: int((strata == z).sum()) for z in zs}
    take = {z: min(per, cap[z]) for z in zs}
    deficit = total - sum(take.values())
    if deficit > sum(cap.values()) - sum(take.values()):
        raise ValueError('candidate pool smaller than the requested total')
    for z in zs:                       # short strata first take from their neighbours
        short = per - take[z]
        for w in sorted((w for w in zs if w != z), key=lambda w: (abs(w - z), w)):
            if deficit <= 0 or short <= 0 or cap[z] >= per:
                break
            g = min(short, deficit, cap[w] - take[w])
            take[w] += g; short -= g; deficit -= g
    while deficit > 0:                 # still short (e.g. all strata full at ``per`` but total > per * n): spread
        for z in zs:
            if deficit > 0 and cap[z] > take[z]:
                take[z] += 1; deficit -= 1
    rng = np.random.default_rng(seed)
    idx = []
    for z in zs:
        members = np.flatnonzero(strata == z)
        idx.append(np.sort(rng.choice(members, take[z], replace=False)))
    return np.concatenate(idx)


def eval_post_ids() -> dict[str, set]:
    """Snapshot post ids attributed to evaluation artists by the v1 plan, the 0918 split file and csip_unseen_v1."""
    v1 = pd.read_parquet(V1_PLAN, columns=['id', 'artist_0918_test', 'csip_eval_public_artist'])
    plan = set(v1.id[v1.artist_0918_test | v1.csip_eval_public_artist].astype(np.int64))
    split = {int(k.split('.')[0]) for k in json.load(open(SPLIT_0918_TEST))}
    unseen = set(pd.read_parquet(UNSEEN_V1, columns=['id']).id.astype(np.int64))
    return {'plan': plan, 'split0918': split, 'unseen_v1': unseen}


def snapshot_artist_tags(ids: set) -> set[str]:
    """Every current artist tag (whitespace split of tag_string_artist) on the snapshot posts with these ids."""
    t = pq.read_table(SNAP, columns=['id', 'tag_string_artist'],
                      filters=pc.field('id').isin(pa_ids(ids)))
    names: set[str] = set()
    for s in t.column('tag_string_artist').to_pylist():
        if s:
            names.update(s.split())
    return names


def pa_ids(ids: set):
    import pyarrow as pa
    return pa.array(sorted(ids), type=pa.int64())


def main():
    if (BENCH / 'manifest.json').exists():
        sys.exit(f'{BENCH} already has a manifest; bump the version instead of overwriting')
    t0 = time.time()
    art = pd.read_parquet(WORK / 'artists.parquet')
    acc = pd.read_parquet(PIXIV_WORK / 'accounts.parquet')

    # ---- B1
    b1df = art[art.eval_0918_test].sort_values('artist')
    keys = b1df.artist.to_numpy()
    sizes = dict(dev=B1_SIZES['dev'], test=B1_SIZES['test'], sealed=len(keys) - sum(B1_SIZES.values()))
    forced = {'test': set(b1df.artist[b1df.eval_test100])}
    b1 = assign_splits(keys, span_bin(b1df.span.to_numpy(), B1_EDGES), sizes, SEED, forced)
    print('B1 dev {dev} / test {test} / sealed {sealed}'.format(**{k: list(b1.values()).count(k) for k in sizes}))

    # ---- R8 extended exclusion base
    base = set(keys) | set(art.artist[art.eval_public]) | set(art.artist[art.eval_unseen_v1])
    ids = eval_post_ids()
    r8_all = set()
    r8_src = {}
    for k, v in ids.items():
        r8_src[k] = {'ids': len(v)}
        r8_all |= snapshot_artist_tags(v)
        print(f'R8 source {k}: {len(v)} ids ({time.time() - t0:.0f}s)', flush=True)
    r8_extra = r8_all - base
    print(f'R8: {len(r8_all)} snapshot artist tags on eval posts; {len(r8_extra)} beyond the base set of {len(base)}')

    # ---- B2
    pool = art[(art.n_local >= B2_MIN_LOCAL) & ~art.in_v1 & ~art.in_kaloscope & ~art.eval_unseen_v1 & ~art.eval_public
               & ~art.eval_0918_test & ~art.artist.isin(r8_extra | base)].sort_values('artist').reset_index(drop=True)
    st = span_bin(pool.span.to_numpy(), B2_EDGES)
    pool_strata = {int(z): int((st == z).sum()) for z in sorted(set(st.tolist()))}
    sel = pool.iloc[stratified_draw(st, B2_PER_STRATUM, B2_TOTAL, SEED)]
    sst = span_bin(sel.span.to_numpy(), B2_EDGES)
    b2 = assign_splits(sel.artist.to_numpy(), sst, dict(dev=200, test=200, sealed=200), SEED)
    fill = {int(z): int((sst == z).sum()) for z in sorted(set(sst.tolist()))}
    print(f'B2 {len(b2)} artists; pool {len(pool)} by stratum {pool_strata}; drawn by stratum {fill}')

    exclude_art = base | set(b2) | r8_extra
    # ---- B3-P2
    cand = acc[acc.pixiv_only & (acc.n_works >= B3_MIN_WORKS)].sort_values('user_id').reset_index(drop=True)
    assert cand.artist.isna().all(), 'pixiv_only account with a mapped artist'
    cst = span_bin(cand.span.to_numpy(), B3_EDGES)
    pick = assign_splits(cand.user_id.to_numpy(), cst, dict(pick=B3_TOTAL, rest=len(cand) - B3_TOTAL), SEED)
    chosen = cand[cand.user_id.map(pick) == 'pick']
    b3p2 = assign_splits(chosen.user_id.to_numpy(), span_bin(chosen.span.to_numpy(), B3_EDGES),
                         dict(dev=100, test=100, sealed=100), SEED)
    print(f'B3-P2 {len(b3p2)} accounts (pool {len(cand)})')

    # ---- B3-P1 accounts and exclusions
    mapped = acc[acc.artist.notna()]
    p1 = mapped[mapped.artist.isin(set(b1)) & (mapped.n_works >= 1)]
    b3p1 = {str(int(u)): a for u, a in zip(p1.user_id, p1.artist)}
    ex_users = set(mapped.user_id[mapped.artist.isin(exclude_art)].astype(int)) | set(int(u) for u in b3p2)
    print(f'B3-P1 {len(b3p1)} accounts over {p1.artist.nunique()} of {len(b1)} B1 artists')

    # ---- asserts
    assert not set(b1) & set(b2)
    assert len(b1) == len(b1df) and len(b2) == B2_TOTAL and len(b3p2) == B3_TOTAL
    assert exclude_art >= set(b1) | set(b2) | r8_extra
    assert not set(b2) & (base - set(b2)) and not set(b2) & r8_extra
    assert not set(b2) & set(art.artist[art.in_v1 | art.in_kaloscope | art.eval_unseen_v1 | art.eval_public])
    assert all(a in exclude_art for a in b3p1.values())
    assert not {int(u) for u in b3p2} - ex_users
    assert set(b3p1) <= {str(u) for u in mapped.user_id}

    BENCH.mkdir(parents=True, exist_ok=True)
    splits = {'b1': b1, 'b2': b2, 'b3p2': {str(int(u)): s for u, s in b3p2.items()}, 'b3p1_accounts': b3p1,
              'exclude_danbooru_artists': sorted(exclude_art), 'exclude_pixiv_users': sorted(ex_users)}
    json.dump(splits, open(BENCH / 'splits.json', 'w'), ensure_ascii=False)
    count = lambda d: {s: list(d.values()).count(s) for s in ('dev', 'test', 'sealed')}
    manifest = {'seed': SEED, 'b1': count(b1), 'b2': count(b2), 'b3p2': count(b3p2),
                'b2_pool': len(pool), 'b2_pool_by_stratum': pool_strata, 'b2_drawn_by_stratum': fill,
                'b3p2_pool': len(cand), 'b3p1_accounts': len(b3p1), 'b3p1_artists': int(p1.artist.nunique()),
                'r8_sources': r8_src, 'r8_snapshot_tags': len(r8_all), 'base_exclusion': len(base),
                'r8_added_beyond_base': len(r8_extra), 'exclude_danbooru_artists': len(exclude_art),
                'exclude_pixiv_users': len(ex_users), 'edges': {'b1': B1_EDGES, 'b2': B2_EDGES, 'b3': B3_EDGES},
                'seconds': round(time.time() - t0)}
    json.dump(manifest, open(BENCH / 'manifest.json', 'w'), indent=1)
    print(f'exclude_danbooru_artists {len(exclude_art)}; exclude_pixiv_users {len(ex_users)}; manifest written')


if __name__ == '__main__':
    main()
