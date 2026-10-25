"""CCIP 跨画风训练补充集: Danbooru 单角色图 -> CCIP 角色类 (ccip-danbooru-xstyle-v1).

    cd /data/neko/cxip && nice -n 10 python -m v3.data.build_ccip_danbooru_xstyle          (CPU only, ~10 min)

目的: 同一角色的正样本来自不同画师 (Danbooru label = 画师), 减少身份嵌入里的画风泄漏.

流程 (确定性; 无随机数, 只用固定 seed 的哈希排序):
  0. CCIP train/val/test 全部图片算 dHash64 (与 danbooru-artist 的 dhash64 同一实现 v3.data.fingerprint), 缓存到
     OUT/ccip_dhash64.parquet. CCIP 一侧原本只有 image_sha256.
  1. Danbooru train.parquet (已排除 CSIP 测试画师) 只留恰好 1 个 character 标签的图.
  2. 留出 (泄漏) 排除, 宁可多删:
     角色级: 标签被开放集剔除类的 Danbooru 匹配 / 歧义候选 / 松散骨干候选 认领; 基名 (精确、词序颠倒、长音折叠、去空格)
       与任一剔除类的名字候选相同; 词集合包含 (小的一方 >= 2 词); 同 franchise 且共享一个有信息量的名字词;
       角色的 copyright (占比 >= 15%) 属于整组剔除的 franchise (new_franchise); dHash 与剔除类图片重合的票数够多.
       dHash 票: 该角色图片与剔除类图片精确重合 >= 1 张且占其有效票 >= 20%, 或 >= 2 张且 >= 5% (抓不同译名, 如
       isabelle (animal crossing) = Shizue, kumbhira = Kubira).
     图片级: sha256 / dHash (海明距离 <= HAMMING) 与开放集 eval_index、CCIP val/test、旧闭集评测集 (zc50x15 等) 的
       任一图片重合; 图片的任一 copyright 属于整组剔除的 franchise; 与剔除类图片 dHash 精确重合.
     另外去掉 cosplay / 玩家化身 / 职业 / 真马 等不对应固定长相的标签 (generic_role).
  3. 角色 -> CCIP 类 (只接受无歧义的):
       meta_series / meta_unique_name: ccip_openset_v1/class_meta.parquet 的 danbooru_tags (v3.eval.ccip_meta 的匹配器,
         名字+作品 / 名字唯一) 里恰好一个类认领该标签;
       hash_resolved: 多个类认领或基名有多个候选类时, dHash 投票 (与 CCIP train 图片重合) 明确指向其中一个;
       direct_full_name: 规范化后的完整名字 (含作品限定词, 去掉 others 等泛化限定词) 恰好对应一个 CCIP 类;
       hash_vote: 没有名字证据, 但 dHash 投票很强且作品相容 (同一角色的不同译名, 如 anchovy / Anzai Chiyomi);
       variant_of_matched: 未匹配的服装/形态变体标签并入已匹配的同角色标签 (基名相同、作品限定词相同);
     作品冲突 (双方都有能对上 Danbooru 的 franchise 且不相交) 时不匹配; 名字匹配与 dHash 票强烈矛盾时记 new:hash_conflict.
     dHash 投票只用信息量足够的精确匹配 (与名字匹配一致率约 99%; 海明 1/2/3 的一致率迅速降到 50%/10%/2%).
  4. 去掉与 CCIP train 图 sha256 / dHash 精确相同的图 (没有新画风; 异类时是标签冲突)、跨身份重复图、同一角色内的
     dHash 重复图; 未匹配的变体标签按 (基名, 作品限定词) 合并成一个新类; 只留 >= MIN_ARTISTS 个画师的类.
  5. 每角色最多 CAP 张: 画师按 hash(seed, 角色, 画师) 排序轮转, 画师内按 hash(seed, danbooru_id) 排序.
  6. label: 匹配上的用 CCIP label (与 ccip-merged-v1-index 同一 id 空间), 否则 NEW_BASE + k (k = 角色标签排序序号).
  7. series_id: 图片的 copyright (多个时取该角色图里最常见的, 再按全数据集频次、名字) -> CCIP class_series 的 series
     (完全相同, 或 norm_key 唯一对应); 否则 NEW_BASE + j (j = copyright 排序序号); 无 copyright 或只有 original -> -1.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import io
import json
import os
import random
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from v3.data.fingerprint import fingerprint_bytes

DAN_ROOT = Path('/data/neko/cxip_v3/data/danbooru-artist-v1')
DAN_INDEX = DAN_ROOT / 'index/train.parquet'
CCIP_ROOT = Path('/data/neko/cxip_repro/data/ccip_merged_nd')
CCIP_INDEX = Path('/data/neko/cxip_v3/data/ccip-merged-v1-index/index')
OPENSET = Path('/data/neko/cxip_v3/eval/ccip_openset_v1')
OUT = Path('/data/neko/cxip_v3/data/ccip-danbooru-xstyle-v1')

CAP = 400
MIN_ARTISTS = 2
NEW_BASE = 1_000_000
SEED = 20261020
HAMMING = 3            # 图片级近重复: dHash64 海明距离 <= 3 (4 段 16 bit 鸽巢, 精确召回)


# ------------------------------------------------------------------------------------------------- dHash (CCIP)
def _dhash_chunk(args):
    shard, rows = args
    out = []
    fd = os.open(CCIP_ROOT / shard, os.O_RDONLY)
    try:
        for i, off, size in rows:
            try:
                data = os.pread(fd, size, off)
                _, _, d64, _ = fingerprint_bytes(data)
                out.append((i, d64.hex(), None))
            except Exception as e:  # noqa: BLE001 - recorded per sample
                out.append((i, None, repr(e)[:200]))
    finally:
        os.close(fd)
    return out


def ccip_dhash(path: Path, workers: int) -> pd.DataFrame:
    """CCIP train/val/test 每张图的 dHash64 (hex, 与 danbooru-artist 的 dhash64 同格式). 结果缓存为单个 parquet."""
    if path.exists():
        return pd.read_parquet(path)
    cols = ['sample_key', 'label', 'shard', 'offset', 'size']
    df = pd.concat([pd.read_parquet(CCIP_INDEX / f'{s}.parquet', columns=cols).assign(split=s)
                    for s in ('train', 'val', 'test')], ignore_index=True)
    df = df.sort_values(['shard', 'offset']).reset_index(drop=True)
    tasks = []
    for shard, g in df.groupby('shard', sort=True):
        rows = list(zip(g.index.tolist(), g['offset'].tolist(), g['size'].tolist()))
        tasks += [(shard, rows[j:j + 512]) for j in range(0, len(rows), 512)]
    t0 = time.time()
    res = {}
    with Pool(workers) as pool:
        for k, part in enumerate(pool.imap_unordered(_dhash_chunk, tasks, chunksize=1)):
            for i, h, e in part:
                res[i] = (h, e)
            if k % 200 == 0:
                print(f'  dhash {len(res)}/{len(df)} {len(res) / max(time.time() - t0, 1e-6):.0f} img/s', flush=True)
    df['dhash64'] = [res[i][0] for i in range(len(df))]
    df['error'] = [res[i][1] for i in range(len(df))]
    df = df[['sample_key', 'split', 'label', 'dhash64', 'error']]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.tmp')
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    print(f'  dhash done {len(df)} in {time.time() - t0:.0f}s, errors {df.error.notna().sum()}', flush=True)
    return df




# 旧闭集评测集 (与 v3/eval/build_ccip_openset.py 的 EVAL_DIRS 相同), 图片级也不能进训练
CLOSEDSET_ROOT = Path('/data/neko/cxip_repro/data')
CLOSEDSET_DIRS = {'zc50x15': 'zerochan_test_c50x15', 'zc500x10': 'zerochan_test_c500x10', 'test_ccip': 'ccip/test_ccip',
                  'ccip_eval': 'ccip/ccip_eval/eval', 'ccip_test_v1': 'ccip_test_v1'}


def _fp_files(paths):
    import hashlib
    out = []
    for p in paths:
        data = Path(p).read_bytes()
        try:
            d64 = fingerprint_bytes(data)[2].hex()
        except Exception:  # noqa: BLE001 - 非图片文件只记 sha256
            d64 = None
        out.append((p, hashlib.sha256(data).hexdigest(), d64))
    return out


def closedset_fingerprints(workers: int) -> pd.DataFrame:
    paths = sorted(str(f) for d in CLOSEDSET_DIRS.values() for f in (CLOSEDSET_ROOT / d).rglob('*') if f.is_file())
    with Pool(min(workers, 16)) as pool:
        rows = [r for part in pool.map(_fp_files, [paths[i:i + 128] for i in range(0, len(paths), 128)]) for r in part]
    return pd.DataFrame(rows, columns=['path', 'image_sha256', 'dhash64'])


# ----------------------------------------------------------------------------------------------------- hashing utils
M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def hex_u64(hexes) -> np.ndarray:
    return np.frombuffer(bytes.fromhex(''.join(hexes)), dtype='>u8').astype(np.uint64)


def mix64(x: np.ndarray) -> np.ndarray:
    """splitmix64 finalizer (确定性排序键)"""
    with np.errstate(over='ignore'):
        x = x.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def str_u64(strings) -> np.ndarray:
    import hashlib
    return np.array([int.from_bytes(hashlib.md5(s.encode()).digest()[:8], 'little') for s in strings], dtype=np.uint64)


def near_pairs(a: np.ndarray, b: np.ndarray, max_d: int, block: int = 300_000) -> pd.DataFrame:
    """a, b: uint64 dHash. 返回海明距离 <= max_d 的全部 (ia, ib, d). max_d <= 3 时 4 段 16 bit 鸽巢保证精确召回."""
    assert max_d <= 3
    outs = []
    for c in range(4):
        sh = np.uint64(16 * c)
        kb = ((b >> sh) & np.uint64(0xFFFF)).astype(np.int64)
        order = np.argsort(kb, kind='stable')
        kbs = kb[order]
        for s in range(0, len(a), block):
            aa = a[s:s + block]
            ka = ((aa >> sh) & np.uint64(0xFFFF)).astype(np.int64)
            lo = np.searchsorted(kbs, ka, 'left')
            cnt = np.searchsorted(kbs, ka, 'right') - lo
            tot = int(cnt.sum())
            ia = np.repeat(np.arange(len(aa)), cnt)
            offs = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
            ib = order[np.repeat(lo, cnt) + offs]
            d = np.bitwise_count(aa[ia] ^ b[ib])
            k = d <= max_d
            outs.append(np.stack([ia[k] + s, ib[k], d[k].astype(np.int64)], 1))
    p = np.unique(np.concatenate(outs), axis=0)
    return pd.DataFrame(p, columns=['ia', 'ib', 'd'])


# ------------------------------------------------------------------------------------------------------ name utils
# 判断留出泄漏时不当作名字信息的词 (职阶 / 泛称 / 服装 / 常见虚词)
STOP_TOKENS = {
    'no', 'the', 'of', 'and', 'de', 'du', 'da', 'di', 'von', 'van', 'der', 'den', 'la', 'le', 'el', 'al', 'chan', 'kun',
    'san', 'sama', 'tan', 'alter', 'lily', 'santa', 'summer', 'swimsuit', 'school', 'uniform', 'version', 'ver', 'form',
    'mode', 'costume', 'outfit', 'dress', 'ii', 'iii', 'iv', 'jr', 'mr', 'ms', 'mrs', 'dr', 'sir', 'lady', 'lord', 'king',
    'queen', 'prince', 'princess', 'god', 'goddess', 'black', 'white', 'red', 'blue', 'green', 'yellow', 'pink', 'purple',
    'dark', 'light', 'little', 'big', 'young', 'old', 'new', 'mini', 'chibi', 'neo', 'super', 'great', 'grand', 'true',
    'first', 'second', 'third', 'one', 'two', 'three', 'others', 'other', 'character', 'female', 'male', 'girl', 'boy',
    'man', 'woman', 'kid', 'child', 'cat', 'dog', 'fox', 'ver', 'type', 'series', 'game', 'anime', 'manga', 'movie', 'film'}
# 不对应固定长相的标签 (玩家化身 / 可自定义主角 / 职业), 仅在没有被 CCIP 类认领时去掉
GENERIC_ROLE_TAGS = {
    'warrior of light (ff14)', 'adventurer (ff11)', 'monster hunter (character)', 'tarnished (elden ring)',
    'hunter (bloodborne)', 'protagonist (pso2)', 'admiral (kancolle)', 'commander (azur lane)',
    "commander (girls' frontline)", 'sensei (blue archive)', 'trainer (umamusume)', 'producer (idolmaster)',
    'master (vocaloid)'}
GENERIC_ROLE_QUALS = {'ragnarok online', 'cosplay', 'cosplayer', 'racehorse'}   # ragnarok online 的角色标签是职业; racehorse 是真马
GENERIC_ROLE_BASES = {'avatar', 'player', 'player character', 'user', 'reader', 'oc', 'original character', 'self insert'}
GENDER_QUALS = {'female', 'male'}


def name_variants(b: str) -> set:
    from v3.eval.ccip_meta import fuzzy_name
    out = set()
    toks = b.split()
    forms = [b] + ([' '.join(reversed(toks))] if 2 <= len(toks) <= 3 else [])
    for f in forms:
        fz = fuzzy_name(f)
        out |= {f, fz, f.replace(' ', ''), fz.replace(' ', '')}
    return {x for x in out if x}


def informative_tokens(tokens, extra_stop=()) -> set:
    from v3.eval.ccip_meta import GENERIC_NAMES
    gen = {t for g in GENERIC_NAMES for t in g.split()}
    return {t for t in tokens if len(t) >= 3 and t not in STOP_TOKENS and t not in gen and t not in extra_stop}


# ------------------------------------------------------------------------------------------------------------ main
def file_info(p: Path) -> dict:
    st = p.stat()
    return dict(path=str(p), bytes=st.st_size, mtime=datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec='seconds'))


def load_inputs():
    cols = ['sample_key', 'shard', 'offset', 'size', 'ext', 'label', 'danbooru_id', 'dhash64', 'image_sha256', 'width',
            'height', 'n_characters', 'characters', 'copyrights']
    dan = pd.read_parquet(DAN_INDEX, columns=cols)
    n_all = len(dan)
    dan = dan[dan['n_characters'] == 1].reset_index(drop=True)
    dan['char_us'] = dan['characters'].str[0]                     # danbooru 原写法 (下划线)
    dan['tag'] = dan['char_us'].str.replace('_', ' ')             # isek / class_meta 的空格写法
    dan = dan.drop(columns=['characters', 'n_characters'])
    ccip = pd.concat([pd.read_parquet(CCIP_INDEX / f'{s}.parquet', columns=['sample_key', 'label', 'image_sha256'])
                      .assign(split=s) for s in ('train', 'val', 'test')], ignore_index=True)
    return dan, n_all, ccip


def build(out: Path, workers: int):
    from v3.eval import ccip_meta
    from v3.eval.ccip_meta import GENERIC_NAMES, GENERIC_SERIES, norm_key, norm_name, split_quals
    t0 = time.time()
    info = dict(config=dict(cap=CAP, min_artists=MIN_ARTISTS, new_label_base=NEW_BASE, seed=SEED,
                            image_hamming=HAMMING, single_character_only=True),
                inputs={}, counts={}, exclusion={}, matching={}, checks={})
    for p in [DAN_INDEX, CCIP_INDEX / 'train.parquet', CCIP_INDEX / 'val.parquet', CCIP_INDEX / 'test.parquet',
              CCIP_INDEX / 'class_series.parquet', OPENSET / 'class_meta.parquet', OPENSET / 'train_exclude.json',
              OPENSET / 'eval_index.parquet', OPENSET / 'heldout_classes.json']:
        info['inputs'][p.name if p.parent.name != 'index' else f'{p.parent.parent.name}/{p.name}'] = file_info(p)

    # ---------------------------------------------------------------- 0. inputs + CCIP dHash
    dh = ccip_dhash(out / 'ccip_dhash64.parquet', workers)
    dan, n_all, ccip = load_inputs()
    info['inputs']['ccip_dhash64.parquet (computed)'] = file_info(out / 'ccip_dhash64.parquet')
    ccip = ccip.merge(dh[['sample_key', 'dhash64']], on='sample_key', how='left')
    assert ccip['dhash64'].notna().all()
    ex = json.load(open(OPENSET / 'train_exclude.json'))
    H = set(int(x) for x in ex['heldout_labels'])
    eval_keys = set(ex['exclude_sample_keys']) | set(ex['zid_overlap_sample_keys'])
    ev_idx = pd.read_parquet(OPENSET / 'eval_index.parquet', columns=['sample_key', 'label'])
    assert set(ev_idx['sample_key']) <= set(ccip['sample_key']) and set(ev_idx['label']) <= H
    ccip['is_eval'] = ccip['sample_key'].isin(eval_keys)
    ccip['is_valtest'] = ccip['split'].isin(['val', 'test'])
    ccip['protected'] = ccip['is_eval'] | ccip['is_valtest']        # 不能进训练的图片
    cm = pd.read_parquet(OPENSET / 'class_meta.parquet')
    cm_by = cm.set_index('label')
    ccip_train_labels = set(pd.read_parquet(CCIP_INDEX / 'train.parquet', columns=['label'])['label'].unique())
    info['counts']['danbooru_train_images'] = n_all
    info['counts']['single_character_images'] = len(dan)
    info['counts']['single_character_tags'] = int(dan['tag'].nunique())
    print(f'[{time.time() - t0:.0f}s] danbooru {n_all} -> single-character {len(dan)} ({dan.tag.nunique()} tags)', flush=True)

    # ---------------------------------------------------------------- 1. image hashes vs CCIP
    a = hex_u64(dan['dhash64'])
    b = hex_u64(ccip['dhash64'])
    pairs = near_pairs(a, b, HAMMING)
    pairs['clabel'] = ccip['label'].values[pairs['ib']]
    pairs['protected'] = ccip['protected'].values[pairs['ib']]
    pairs['is_eval'] = ccip['is_eval'].values[pairs['ib']]
    # 投票只用信息量足够的精确匹配 (退化哈希如纯色 / 条纹图会大量碰撞)
    dfreq = dan['dhash64'].map(dan['dhash64'].value_counts()).values
    cfreq = ccip['dhash64'].map(ccip['dhash64'].value_counts()).values
    pc = np.bitwise_count(a)
    pairs['informative'] = ((pairs['d'] == 0) & (dfreq[pairs['ia']] <= 3) & (cfreq[pairs['ib']] <= 5)
                            & (pc[pairs['ia']] >= 8) & (pc[pairs['ia']] <= 56))
    print(f'[{time.time() - t0:.0f}s] dHash pairs (d<={HAMMING}) {len(pairs)}, exact informative {int(pairs.informative.sum())}',
          flush=True)
    prot_sha = set(ccip.loc[ccip['protected'], 'image_sha256'])
    img_excl = pd.DataFrame(index=dan.index)
    img_excl['sha256_eval_valtest'] = dan['image_sha256'].isin(prot_sha).values
    hit = pairs[pairs['protected']]
    img_excl['dhash_eval_valtest'] = np.isin(np.arange(len(dan)), hit['ia'].unique())
    img_excl['dhash_eval_valtest_exact'] = np.isin(np.arange(len(dan)), hit.loc[hit['d'] == 0, 'ia'].unique())
    cs_fp = closedset_fingerprints(workers)
    cs_h = cs_fp['dhash64'].dropna()
    cs_pairs = near_pairs(a, hex_u64(cs_h), HAMMING)
    img_excl['closedset_eval'] = (dan['image_sha256'].isin(set(cs_fp['image_sha256'])).values
                                  | np.isin(np.arange(len(dan)), cs_pairs['ia'].unique()))
    info['counts']['closedset_eval_files'] = len(cs_fp)
    info['counts']['closedset_eval_files_with_dhash'] = int(len(cs_h))

    # 每张 danbooru 图的投票: 信息量足够的精确匹配指向的 CCIP 类 (只有一个类时才算一票)
    inf = pairs[pairs['informative']]
    lab_sets = inf.groupby('ia')['clabel'].agg(lambda x: tuple(sorted(set(x))))
    vote_img = pd.DataFrame({'ia': lab_sets.index, 'labels': lab_sets.values})
    vote_img['tag'] = dan['tag'].values[vote_img['ia']]
    vote_img['held'] = [any(l in H for l in ls) for ls in vote_img['labels']]
    vote_img['vote'] = [ls[0] if len(ls) == 1 and ls[0] not in H else -1 for ls in vote_img['labels']]
    tag_votes = {t: collections.Counter(g['vote'][g['vote'] >= 0].tolist()) for t, g in vote_img.groupby('tag')}
    tag_nvote_img = vote_img.groupby('tag').size().to_dict()
    tag_held_img = vote_img[vote_img['held']].groupby('tag').size().to_dict()

    # ---------------------------------------------------------------- 2. tag table + franchises
    db = ccip_meta.Danbooru()
    infos = ccip_meta.class_infos(db)
    info_by = {x.label: x for x in infos}
    tags = dan.groupby('tag').agg(n_img=('label', 'size'), n_art=('label', 'nunique'), char_us=('char_us', 'first'))
    cp_long = dan[['tag', 'copyrights']].explode('copyrights').dropna()
    cp_long['cp'] = cp_long['copyrights'].str.replace('_', ' ')
    cp_cnt = cp_long.groupby(['tag', 'cp']).size().rename('n').reset_index()
    cp_cnt['share'] = cp_cnt['n'] / cp_cnt['tag'].map(tags['n_img'])
    local_cps = cp_cnt[(cp_cnt['share'] >= 0.15) & ~cp_cnt['cp'].isin(['original', 'real life'])].groupby('tag')['cp'].agg(list)

    def cp_keys(cp):
        return {norm_key(cp), norm_key(db.root(cp)), norm_key(split_quals(cp)[0])} - {''}

    tag_fr, tag_fr_keys = {}, {}
    for t in tags.index:
        cps = [c for c, _ in db.char_cp.get(t, [])] + local_cps.get(t, [])
        tag_fr[t] = db.tag_franchises(t) | {norm_key(db.root(c)) for c in local_cps.get(t, [])}
        tag_fr_keys[t] = set().union(*[cp_keys(c) for c in cps]) if cps else set()

    # ---------------------------------------------------------------- 3. leak exclusion (character level)
    held_fr = set(ex['heldout_franchises']) | {norm_key(s) for s in ex['heldout_series']} \
        | {norm_key(split_quals(s)[0]) for s in ex['heldout_series']}
    held_fr -= {''} | GENERIC_SERIES
    held_cm = cm[cm['label'].isin(H)]
    held_claimed = {t for ts in held_cm['danbooru_tags'] for t in ts}
    held_amb = {t for col in ('ambiguous_tags', 'loose_backbone') for ts in held_cm[col] for t in ts}
    held_bases = collections.defaultdict(set)     # base -> labels
    held_tok = []                                 # (label, token set, informative tokens, franchises)
    for lb in sorted(H):
        r = cm_by.loc[lb]
        bases = {c.base for c in info_by[lb].cands}
        base, quals = split_quals(r['name'])
        series_tok = set()
        for s in list(r['series_raw']) + [r['series'] or ''] + list(r['db_series']):
            series_tok |= set(norm_name(s).split())
        known = set(r['series_keys']) | set(r['franchises']) | {norm_key(r['series'] or '')}
        if norm_name(base) not in GENERIC_NAMES:
            bases.add(norm_name(base))
        for q in quals:
            if norm_key(q) not in known and norm_key(q) not in GENERIC_SERIES:
                qb = norm_name(split_quals(q)[0])
                if qb not in GENERIC_NAMES:
                    bases.add(qb)
        bases = {x for x in bases if x and x not in GENERIC_NAMES}
        for x in bases:
            for v in name_variants(x):
                held_bases[v].add(lb)
            held_tok.append((lb, set(x.split()), informative_tokens(x.split(), series_tok), set(r['franchises'])))
    tok_index = collections.defaultdict(list)
    for j, (_, ts, _, _) in enumerate(held_tok):
        for tk in ts:
            tok_index[tk].append(j)

    reasons = {t: [] for t in tags.index}
    for t in tags.index:
        base, quals = split_quals(t)
        b = norm_name(base)
        bt = set(b.split())
        R = reasons[t]
        if t in held_claimed:
            R.append('heldout_claimed_tag')
        if t in held_amb:
            R.append('heldout_ambiguous_tag')
        if tag_fr_keys[t] & held_fr:
            R.append('heldout_franchise')
        if any(v in held_bases for v in name_variants(b)):
            R.append('heldout_bare_name')
        cand = {j for tk in bt for j in tok_index.get(tk, ())}
        for j in sorted(cand):
            lb, ts, itok, fr = held_tok[j]
            small, big = (ts, bt) if len(ts) <= len(bt) else (bt, ts)
            if len(small) >= 2 and small <= big:
                R.append('heldout_name_tokens')
                break
            if fr and (fr & tag_fr[t]) and (itok & bt):
                R.append('heldout_same_franchise_token')
                break
        nh, nv = tag_held_img.get(t, 0), tag_nvote_img.get(t, 0)
        if (nh >= 1 and nh >= 0.2 * nv) or (nh >= 2 and nh >= 0.05 * nv):
            R.append('heldout_hash_votes')
    # 非固定长相的标签
    claimed_all = collections.defaultdict(set)
    for lb, ts in zip(cm['label'], cm['danbooru_tags']):
        for x in ts:
            claimed_all[x].add(int(lb))
    for t in tags.index:
        if t in claimed_all:
            continue
        base, quals = split_quals(t)
        if t in GENERIC_ROLE_TAGS or (set(quals) & GENERIC_ROLE_QUALS) or norm_name(base) in GENERIC_ROLE_BASES:
            reasons[t].append('generic_role')
    tags['reasons'] = [reasons[t] for t in tags.index]
    tags['excluded'] = tags['reasons'].map(len) > 0
    rc = collections.Counter(r for rs in tags['reasons'] for r in rs)
    first = collections.Counter(rs[0] for rs in tags['reasons'] if rs)
    img_by_reason = {r: int(tags.loc[[r in rs for rs in tags['reasons']], 'n_img'].sum()) for r in rc}
    info['exclusion']['characters'] = dict(
        total_excluded_tags=int(tags['excluded'].sum()), total_excluded_images=int(tags.loc[tags.excluded, 'n_img'].sum()),
        tags_by_reason_nonexclusive=dict(rc), images_by_reason_nonexclusive=img_by_reason,
        tags_by_first_reason=dict(first),
        excluded_tags_with_ge2_artists=int((tags.excluded & (tags.n_art >= 2)).sum()))
    print(f'[{time.time() - t0:.0f}s] excluded tags {int(tags.excluded.sum())} ({int(tags.loc[tags.excluded, "n_img"].sum())} imgs): '
          f'{dict(rc)}', flush=True)

    # ---------------------------------------------------------------- 4. image-level exclusion
    cp_held = {}
    for c in cp_long['cp'].unique():
        cp_held[c] = bool(cp_keys(c) & held_fr)
    img_excl['heldout_franchise_copyright'] = [any(cp_held.get(c.replace('_', ' '), False) for c in l)
                                               for l in dan['copyrights']]
    img_excl['character_excluded'] = dan['tag'].map(tags['excluded']).values
    img_excl['heldout_hash_image'] = np.isin(np.arange(len(dan)), vote_img.loc[vote_img['held'], 'ia'])
    leak_cols = ['sha256_eval_valtest', 'dhash_eval_valtest', 'closedset_eval', 'heldout_franchise_copyright',
                 'heldout_hash_image']
    img_leak = img_excl[leak_cols].any(axis=1).values
    keep = ~img_excl['character_excluded'].values & ~img_leak
    info['exclusion']['images'] = dict(
        **{f'{c}': int(img_excl[c].sum()) for c in leak_cols + ['dhash_eval_valtest_exact']},
        image_level_any=int(img_leak.sum()),
        image_level_only_after_character_exclusion=int((img_leak & ~img_excl['character_excluded'].values).sum()),
        by_character_exclusion=int(img_excl['character_excluded'].sum()),
        remaining=int(keep.sum()))
    print(f'[{time.time() - t0:.0f}s] image exclusion {info["exclusion"]["images"]}', flush=True)

    # ---------------------------------------------------------------- 5. mapping tag -> CCIP class
    ccip_base = collections.defaultdict(set)       # 非留出类的名字候选 (变体) -> labels
    ccip_full = collections.defaultdict(set)       # 完整名字 (含非泛化限定词) -> labels

    def full_key(name):
        base, quals = split_quals(name)
        qs = [q for q in quals if norm_key(q) not in GENERIC_SERIES]
        return norm_name(' '.join([base] + qs))

    merged_classes = ccip_meta.load_classes()
    for c in merged_classes:
        lb = int(c['label'])
        if lb in H:
            continue
        for cand in info_by[lb].cands:
            for v in name_variants(cand.base):
                ccip_base[v].add(lb)
        keys = {full_key(c['name'])} | {full_key(m['key']) for m in c['members'] if m['source'] == 'zc800k'}
        for k in keys:
            if len(k.split()) >= 2:
                ccip_full[k].add(lb)
    # 类一侧只用能对上 Danbooru copyright 根的 franchise 键 (其余如 ourokroniich / xenoblade2 不足以否决)
    all_roots = {norm_key(db.root(cp)) for cp in db.cp_count if cp} - {''}
    cls_fr = {int(lb): set(fr) & all_roots for lb, fr in zip(cm['label'], cm['franchises'])}

    def fr_ok(t, lb):
        a_, b_ = tag_fr[t], cls_fr.get(lb, set())
        return not (a_ and b_) or bool(a_ & b_)

    def hash_pick(t, among=None, min_votes=3, min_share=0.7):
        """dHash 投票最多的类 (限定在 among 里), 占该标签全部有效票的比例 >= min_share 才接受"""
        v = tag_votes.get(t)
        if not v:
            return None, 0, 0.0
        tot = sum(v.values())
        cand = [(n, -k, k) for k, n in v.items() if among is None or k in among]
        if not cand:
            return None, 0, 0.0
        n, _, lb = max(cand)
        return (lb if n >= min_votes and n / tot >= min_share else None), n, n / tot

    rows = []
    for t in tags.index[~tags['excluded'].values]:
        b = norm_name(split_quals(t)[0])
        claims = {lb for lb in claimed_all.get(t, set())}
        base_c = set().union(*[ccip_base.get(v, set()) for v in name_variants(b)]) if b else set()
        lb, method, cands = -1, '', sorted(claims or base_c)
        v = tag_votes.get(t, collections.Counter())
        if len(claims) == 1:
            lb = next(iter(claims))
            method = 'meta_' + cm_by.loc[lb, 'match_method']
            if not fr_ok(t, lb):
                lb, method = -1, 'new:franchise_conflict'
        elif len(claims) > 1:
            pick, _, _ = hash_pick(t, among=claims)
            lb, method = (pick, 'hash_resolved_multi') if pick is not None else (-1, 'new:ambiguous_multi_class')
        else:
            fk = full_key(t)
            fc = ccip_full.get(fk, set()) if len(fk.split()) >= 2 else set()
            if len(fc) == 1 and fr_ok(t, next(iter(fc))):
                lb, method = next(iter(fc)), 'direct_full_name'
            else:
                pick, _, _ = hash_pick(t, among=base_c) if base_c else (None, 0, 0)
                if pick is not None and fr_ok(t, pick):
                    lb, method = pick, 'hash_name'
                else:
                    pick, n, share = hash_pick(t, min_votes=5, min_share=0.8)
                    if pick is not None and fr_ok(t, pick):
                        lb, method = pick, 'hash_vote'
                    else:
                        method = 'new:ambiguous_name' if (base_c or fc) else 'new:no_ccip_match'
                        cands = sorted(base_c | fc)
        # 名字匹配与 dHash 强烈矛盾 (>= 5 票, 指向该类的不到一半) -> 视为有歧义
        if lb >= 0 and not method.startswith('hash') and sum(v.values()) >= 5 and v.get(lb, 0) < 0.5 * sum(v.values()):
            cands = sorted({lb} | {k for k, _ in v.most_common(3)})
            lb, method = -1, 'new:hash_conflict'
        rows.append(dict(tag=t, ccip_label=int(lb), method=method, candidates=cands[:20],
                         hash_votes=int(sum(v.values())), hash_top=int(v.most_common(1)[0][0]) if v else -1,
                         hash_top_n=int(v.most_common(1)[0][1]) if v else 0,
                         hash_agree=int(v.get(lb, 0)) if lb >= 0 else 0))
    mp = pd.DataFrame(rows).set_index('tag')
    assert not (set(mp.loc[mp.ccip_label >= 0, 'ccip_label']) & H)

    # ---- 同一角色的服装 / 形态变体标签合并: 基名 (非泛称) 与性别限定词相同, franchise 相交, 且
    #      (a) 最后一个限定词 (作品) 相同: 'fubuki (music pixy) (azur lane)' ~ 'fubuki (azur lane)'; 或
    #      (b) 一方无限定词、另一方的限定词都不是作品名: 'shirakami fubuki (1st costume)' ~ 'shirakami fubuki'
    #      ('haru urara (racehorse)' / 'archer (fate/samurai remnant)' 不会并进 'haru urara (umamusume)' / 'archer (fate)')
    parent = {t: t for t in mp.index}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def is_series_qual(q):
        k = norm_key(q)
        return k in db.cp_by_key or k in all_roots

    by_base = collections.defaultdict(list)
    for t in mp.index:
        base, quals = split_quals(t)
        by_base[(norm_name(base), tuple(sorted(set(quals) & GENDER_QUALS)))].append(t)
    for key, ts in by_base.items():
        if not key[0] or key[0] in GENERIC_NAMES or len(ts) < 2:
            continue
        ts = sorted(ts)
        qs = {t: [q for q in split_quals(t)[1] if q not in GENDER_QUALS] for t in ts}
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                x, y = ts[i], ts[j]
                if not (tag_fr[x] & tag_fr[y]):
                    continue
                qx, qy = qs[x], qs[y]
                same_series = bool(qx) and bool(qy) and qx[-1] == qy[-1]
                main_variant = (not qx and qy and not any(is_series_qual(q) for q in qy)) or \
                               (not qy and qx and not any(is_series_qual(q) for q in qx))
                if same_series or main_variant:
                    parent[find(y)] = find(x)
    roots = pd.Series([find(t) for t in mp.index], index=mp.index)
    rep = pd.Series(mp.index, index=mp.index).groupby(roots).transform('min')   # 组代表 = 组内字典序最小的标签
    mp['group'] = rep.values
    upgraded = 0
    for g, gg in mp.groupby('group'):
        labs = set(gg.loc[gg.ccip_label >= 0, 'ccip_label'])
        if len(labs) != 1:
            continue
        lb = next(iter(labs))
        sel = gg.index[(gg.ccip_label < 0) & gg.method.isin(['new:no_ccip_match', 'new:ambiguous_name'])]
        sel = [t for t in sel if fr_ok(t, lb)]
        mp.loc[sel, 'ccip_label'] = lb
        mp.loc[sel, 'method'] = 'variant_of_matched'
        upgraded += len(sel)
    info['matching']['variant_of_matched_tags'] = upgraded

    # ---------------------------------------------------------------- 6. kept images, dedup, >= 2 artists, labels
    d = dan[keep].copy()
    d = d[d['tag'].isin(mp.index)]
    d['ccip_label'] = d['tag'].map(mp['ccip_label']).values
    d['group'] = d['tag'].map(mp['group']).values
    d['ident'] = np.where(d['ccip_label'] >= 0, 'c' + d['ccip_label'].astype(str), 'g' + d['group'])
    info['counts']['after_exclusion_images'] = len(d)
    # 与 CCIP train 图完全相同 (sha256 或 dHash 精确相同): 同类时没有新画风信息, 异类时是标签冲突 -> 都去掉
    ccip_tr = ccip[ccip['split'] == 'train']
    dup_tr = d['image_sha256'].isin(set(ccip_tr['image_sha256'])) | d['dhash64'].isin(set(ccip_tr['dhash64']))
    tr_lab = ccip_tr.groupby('dhash64')['label'].agg(set)
    same = [lb >= 0 and lb in tr_lab.get(h, ()) for lb, h in zip(d.loc[dup_tr, 'ccip_label'], d.loc[dup_tr, 'dhash64'])]
    info['counts']['dup_with_ccip_train_removed'] = int(dup_tr.sum())
    info['counts']['dup_with_ccip_train_removed_same_label'] = int(np.sum(same))
    d = d[~dup_tr.values]
    # 同一张图 (dHash 精确相同) 出现在不同身份下 -> 全部去掉
    nid = d.groupby('dhash64')['ident'].transform('nunique')
    info['counts']['cross_identity_dhash_dup_removed'] = int((nid > 1).sum())
    d = d[(nid <= 1).values]
    n0 = len(d)
    d = d.sort_values('danbooru_id').drop_duplicates(['ident', 'dhash64'], keep='first')
    info['counts']['dedup_same_identity_dhash_removed'] = n0 - len(d)
    na = d.groupby('ident')['label'].nunique()
    good = set(na.index[na >= MIN_ARTISTS])
    info['counts']['identities_before_min_artists'] = int(len(na))
    info['counts']['identities_dropped_lt_min_artists'] = int((na < MIN_ARTISTS).sum())
    info['counts']['images_dropped_lt_min_artists'] = int((~d['ident'].isin(good)).sum())
    d = d[d['ident'].isin(good)].copy()
    # 新类 id: NEW_BASE + k, k = 变体组代表标签 (组内字典序最小) 的排序序号
    new_groups = sorted(d.loc[d['ccip_label'] < 0, 'group'].unique())
    new_id = {g: NEW_BASE + k for k, g in enumerate(new_groups)}
    d['out_label'] = np.where(d['ccip_label'] >= 0, d['ccip_label'], d['group'].map(new_id).fillna(-1)).astype(np.int64)
    info['counts']['before_cap_images'] = len(d)
    info['counts']['before_cap_classes'] = int(d['out_label'].nunique())

    # ---------------------------------------------------------------- 7. cap: 画师轮转
    seed = np.uint64(SEED)
    lab_h = mix64(d['out_label'].values.astype(np.uint64) ^ seed)
    d['_ka'] = mix64(lab_h ^ mix64(d['label'].values.astype(np.uint64) + seed))      # 画师顺序
    d['_ki'] = mix64(d['danbooru_id'].values.astype(np.uint64) ^ mix64(seed))         # 画师内顺序
    d = d.sort_values(['out_label', 'label', '_ki'])
    d['_r'] = d.groupby(['out_label', 'label']).cumcount()
    d = d.sort_values(['out_label', '_r', '_ka', '_ki'])
    d['_pos'] = d.groupby('out_label').cumcount()
    d = d[d['_pos'] < CAP].copy()
    info['counts']['final_images'] = len(d)
    info['counts']['final_classes'] = int(d['out_label'].nunique())

    # ---------------------------------------------------------------- 8. series_id
    cs = pd.read_parquet(CCIP_INDEX / 'class_series.parquet')
    ser_id = cs.dropna(subset=['series']).groupby('series')['series_id'].first().to_dict()
    ser_by_key = collections.defaultdict(set)
    for s, i in ser_id.items():
        ser_by_key[norm_key(s)].add(i)
    lab_series = cs.set_index('label')['series'].to_dict()
    glob_cp = collections.Counter(c for l in dan['copyrights'] for c in l)
    char_cp = collections.Counter()
    for lab, l in zip(d['out_label'].values, d['copyrights'].values):
        for c in l:
            char_cp[(lab, c)] += 1
    chosen = []
    for lab, l in zip(d['out_label'].values, d['copyrights'].values):
        l = [c for c in l if c != 'original']
        if not l:
            chosen.append('')
            continue
        pref = lab_series.get(lab) if lab < NEW_BASE else None
        chosen.append(min(l, key=lambda c: (c != pref, -char_cp[(lab, c)], -glob_cp[c], c)))
    d['copyright'] = chosen

    def map_cp(c):
        if not c:
            return -1, 'none'
        if c in ser_id:
            return int(ser_id[c]), 'exact'
        k = ser_by_key.get(norm_key(c), set())
        if len(k) == 1:
            return int(next(iter(k))), 'norm_key'
        return None, 'new'
    cmap = {c: map_cp(c) for c in sorted(set(chosen))}
    new_cps = sorted(c for c, (i, _) in cmap.items() if i is None)
    new_sid = {c: NEW_BASE + j for j, c in enumerate(new_cps)}
    d['series_id'] = [cmap[c][0] if cmap[c][0] is not None else new_sid[c] for c in d['copyright']]
    d['series_id'] = d['series_id'].astype(np.int64)
    sm = collections.Counter(cmap[c][1] for c in d['copyright'])
    info['matching']['series_id_images_by_method'] = dict(sm)
    info['matching']['series_new_ids'] = len(new_cps)
    mm = d[d['out_label'] < NEW_BASE]
    cls_sid = cs.set_index('label')['series_id']
    info['matching']['matched_images_series_equal_ccip_class'] = float((mm['series_id'].values == mm['out_label'].map(cls_sid).values).mean())

    # ---------------------------------------------------------------- 9. output
    res = pd.DataFrame({
        'sample_key': 'dan_' + d['sample_key'].astype(str),
        'shard': [str(DAN_ROOT / s) for s in d['shard']],
        'offset': d['offset'].astype(np.int64).values, 'size': d['size'].astype(np.int64).values, 'ext': d['ext'].values,
        'label': d['out_label'].astype(np.int64).values, 'series_id': d['series_id'].values,
        'artist_id': d['label'].astype(np.int64).values, 'character': d['char_us'].values,
        'ccip_matched': (d['out_label'] < NEW_BASE).values,
        'match_method': d['tag'].map(mp['method']).values, 'source': 'danbooru',
        'danbooru_id': d['danbooru_id'].astype(np.int64).values, 'dhash64': d['dhash64'].values,
        'image_sha256': d['image_sha256'].values, 'copyright': d['copyright'].values,
        'width': d['width'].astype(np.int64).values, 'height': d['height'].astype(np.int64).values})
    res = res.sort_values(['label', 'danbooru_id']).reset_index(drop=True)
    (out / 'index').mkdir(parents=True, exist_ok=True)

    # 角色级审计表 (全部单角色标签, 含被排除的)
    names = cm_by['name'].to_dict()
    ct = tags[['char_us', 'n_img', 'n_art', 'reasons']].join(mp, how='left')
    ct['method'] = ct['method'].fillna('excluded')
    ct['ccip_label'] = ct['ccip_label'].fillna(-1).astype(np.int64)
    ct['ccip_name'] = ct['ccip_label'].map(names)
    ct['candidates'] = [list(x) if isinstance(x, list) else [] for x in ct['candidates']]
    ct['candidate_names'] = [[names.get(c, '') for c in x[:5]] for x in ct['candidates']]
    kept_stats = res.groupby('character').agg(n_out=('label', 'size'), out_label=('label', 'first'),
                                              n_out_artists=('artist_id', 'nunique'))
    ct = ct.join(kept_stats, on='char_us')
    ct['n_out'] = ct['n_out'].fillna(0).astype(np.int64)
    ct['n_out_artists'] = ct['n_out_artists'].fillna(0).astype(np.int64)
    ct['out_label'] = ct['out_label'].fillna(-1).astype(np.int64)
    ct['franchises'] = [sorted(tag_fr[t]) for t in ct.index]
    for c in ('hash_votes', 'hash_top', 'hash_top_n', 'hash_agree'):
        ct[c] = ct[c].fillna(0 if c != 'hash_top' else -1).astype(np.int64)
    ct = ct.reset_index().rename(columns={'tag': 'character_tag', 'char_us': 'character'})
    ct = ct[['character', 'character_tag', 'n_img', 'n_art', 'reasons', 'method', 'ccip_label', 'ccip_name', 'candidates',
             'candidate_names', 'group', 'franchises', 'hash_votes', 'hash_top', 'hash_top_n', 'hash_agree', 'out_label',
             'n_out', 'n_out_artists']]
    return res, ct, mp, tags, info, dict(ccip=ccip, pairs=pairs, H=H, cm_by=cm_by, ccip_train_labels=ccip_train_labels,
                                         eval_keys=eval_keys, t0=t0)


# ---------------------------------------------------------------------------------------------------------- checks
def read_check(res: pd.DataFrame, n: int = 24) -> dict:
    """绝对 shard 路径直接 pread + tar 头里的成员名 + PIL 解码; 再用 IndexedTarDataset (任意 root) 读几张"""
    from PIL import Image
    idx = sorted(random.Random(SEED).sample(range(len(res)), min(n, len(res))))
    ok, bad = 0, []
    for i in idx:
        r = res.iloc[i]
        with open(r['shard'], 'rb') as f:
            f.seek(int(r['offset']) - 512)
            hdr = f.read(512)
            data = f.read(int(r['size']))
        member = hdr[:100].rstrip(b'\0').decode()
        im = Image.open(io.BytesIO(data))
        im.load()
        good = (len(data) == r['size'] and member == f"{r['sample_key'][4:]}.{r['ext']}"
                and im.size == (r['width'], r['height']))
        ok += good
        if not good:
            bad.append(dict(sample_key=r['sample_key'], member=member, size=im.size))
    from v3.data.indexed_tar import IndexedTarDataset
    ds = IndexedTarDataset('/nonexistent-root', res.iloc[idx[:4]].reset_index(drop=True), columns=['artist_id', 'series_id'])
    items = [ds[i] for i in range(len(ds))]
    ds.close()
    return dict(n=len(idx), ok=int(ok), bad=bad[:5],
                indexed_tar_dataset_ok=all(it['image'].size[0] > 0 for it in items),
                examples=[dict(sample_key=res.iloc[i]['sample_key'], character=res.iloc[i]['character'],
                               shard=res.iloc[i]['shard'], offset=int(res.iloc[i]['offset'])) for i in idx[:3]])


def run_checks(res, ct, mp, info, aux):
    ccip, H, cm_by = aux['ccip'], aux['H'], aux['cm_by']
    chk = {}
    lab = res['label'].values
    matched = res['ccip_matched'].values
    chk['matched_labels_in_ccip_train'] = bool(set(lab[matched]) <= aux['ccip_train_labels'])
    chk['new_labels_ge_base'] = bool((lab[~matched] >= NEW_BASE).all())
    chk['matched_labels_lt_base'] = bool((lab[matched] < NEW_BASE).all())
    chk['ccip_max_label'] = int(max(aux['ccip_train_labels']))
    chk['labels_overlap_heldout'] = int(len(set(lab) & H))
    chk['sample_key_unique'] = bool(res['sample_key'].is_unique)
    chk['sample_key_overlap_ccip'] = int(res['sample_key'].isin(set(ccip['sample_key'])).sum())
    chk['sample_key_overlap_eval_exclude'] = int(res['sample_key'].isin(aux['eval_keys']).sum()
                                                 + res['sample_key'].str[4:].isin(aux['eval_keys']).sum())
    prot = ccip[ccip['protected']]
    chk['sha256_overlap_eval_valtest'] = int(res['image_sha256'].isin(set(prot['image_sha256'])).sum())
    p = near_pairs(hex_u64(res['dhash64']), hex_u64(prot['dhash64']), HAMMING)
    chk[f'dhash_overlap_eval_valtest_d<={HAMMING}'] = int(p['ia'].nunique()) if len(p) else 0
    ev = ccip[ccip['is_eval']]
    p = near_pairs(hex_u64(res['dhash64']), hex_u64(ev['dhash64']), HAMMING)
    chk[f'dhash_overlap_openset_eval_d<={HAMMING}'] = int(p['ia'].nunique()) if len(p) else 0
    tr = ccip[ccip['split'] == 'train']
    chk['sha256_dup_with_ccip_train_images'] = int(res['image_sha256'].isin(set(tr['image_sha256'])).sum())
    chk['dhash_exact_dup_with_ccip_train_images'] = int(res['dhash64'].isin(set(tr['dhash64'])).sum())
    cs_fp = closedset_fingerprints(16)
    p = near_pairs(hex_u64(res['dhash64']), hex_u64(cs_fp['dhash64'].dropna()), HAMMING)
    chk[f'closedset_eval_overlap_sha256_or_dhash_d<={HAMMING}'] = int(
        len(set(p['ia']) | set(np.nonzero(res['image_sha256'].isin(set(cs_fp['image_sha256'])).values)[0])))
    chk['dhash_shared_across_labels'] = int((res.groupby('dhash64')['label'].transform('nunique') > 1).sum())
    chk['danbooru_id_unique'] = bool(res['danbooru_id'].is_unique)
    chk['dtypes'] = {c: str(t) for c, t in res.dtypes.items()}
    assert chk['matched_labels_in_ccip_train'] and chk['new_labels_ge_base'] and chk['matched_labels_lt_base']
    assert chk['ccip_max_label'] < NEW_BASE and chk['labels_overlap_heldout'] == 0
    assert chk['sample_key_unique'] and chk['sample_key_overlap_ccip'] == 0 and chk['sample_key_overlap_eval_exclude'] == 0
    assert chk['sha256_overlap_eval_valtest'] == 0 and chk[f'dhash_overlap_eval_valtest_d<={HAMMING}'] == 0
    assert chk[f'closedset_eval_overlap_sha256_or_dhash_d<={HAMMING}'] == 0
    assert chk['sha256_dup_with_ccip_train_images'] == 0 and chk['dhash_exact_dup_with_ccip_train_images'] == 0
    assert chk['dhash_shared_across_labels'] == 0
    # 分布
    g = res.groupby('label').agg(n=('sample_key', 'size'), na=('artist_id', 'nunique'), m=('ccip_matched', 'first'))
    q = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]

    def dist(x):
        return {f'p{int(k * 100)}': float(v) for k, v in x.quantile(q).items()} | {'mean': float(x.mean())}
    chk['artists_per_class'] = dist(g['na'])
    chk['artists_per_class_matched'] = dist(g.loc[g.m, 'na'])
    chk['artists_per_class_new'] = dist(g.loc[~g.m, 'na'])
    chk['images_per_class'] = dist(g['n'])
    bins = [2, 3, 5, 10, 20, 50, 100, 200, 400, 10 ** 9]
    chk['classes_by_artist_count'] = {f'[{bins[i]},{bins[i + 1]})': int(((g.na >= bins[i]) & (g.na < bins[i + 1])).sum())
                                      for i in range(len(bins) - 1)}
    chk['images_per_artist_within_class_max_share'] = dist(
        res.groupby(['label', 'artist_id']).size().groupby(level=0).max() / g['n'])
    chk['ccip_train_classes_covered'] = int(g.m.sum())
    chk['ccip_train_classes_covered_frac'] = float(g.m.sum() / len(aux['ccip_train_labels'] - H))
    chk['read_check'] = read_check(res)
    assert chk['read_check']['ok'] == chk['read_check']['n'] and chk['read_check']['indexed_tar_dataset_ok']
    return chk


def spot_checks(ct: pd.DataFrame, n: int = 30) -> dict:
    rng = random.Random(SEED)
    k = ct[ct['n_out'] > 0]
    m = k[k['ccip_label'] >= 0]
    m_idx = sorted(rng.sample(list(m.index), min(n, len(m))))
    matched = [f"{r.character}  ->  [{r.ccip_label}] {r.ccip_name}   ({r.method}; hash {r.hash_agree}/{r.hash_votes})"
               for r in m.loc[m_idx].itertuples()]
    u = k[(k['ccip_label'] < 0) & (k['candidates'].map(len) > 0)]
    u_idx = sorted(rng.sample(list(u.index), min(n, len(u))))
    unmatched = [f"{r.character}  -x-  {list(r.candidate_names)[:3]}   ({r.method}; hash top n={r.hash_top_n}/{r.hash_votes})"
                 for r in u.loc[u_idx].itertuples()]
    hv = k[k['method'].isin(['hash_vote', 'hash_name'])].sort_values('n_img', ascending=False).head(15)
    hash_only = [f"{r.character}  ->  {r.ccip_name}   ({r.method}; hash {r.hash_agree}/{r.hash_votes})" for r in hv.itertuples()]
    return dict(matched_random=matched, unmatched_similar_random=unmatched, hash_matched_top=hash_only)


# ------------------------------------------------------------------------------------------------------------ main
def write_readme(out: Path, res: pd.DataFrame, info: dict):
    c, e, m, k = info['counts'], info['exclusion'], info['matching'], info['checks']
    ce, ie = e['characters'], e['images']
    mc = m['final_classes_by_method']
    L = ['# ccip-danbooru-xstyle-v1', '',
         'CCIP 训练补充集: Danbooru 单角色图, 角色作为身份类、画师 (`artist_id`) 各不相同 -> 跨画风正样本, 减少身份嵌入里的画风泄漏。',
         f'由 `/data/neko/cxip/v3/data/build_ccip_danbooru_xstyle.py` 生成 ({info["timestamp"]}), 规则见脚本文件头。', '',
         '## 文件', '',
         '| 文件 | 内容 |', '|-|-|',
         '| `index/train.parquet` | 训练索引 (见下). `shard` 是绝对路径, `IndexedTarDataset` 的 root 随便给 |',
         '| `index/characters.parquet` | 全部单角色 Danbooru 标签的审计表: 排除原因、匹配方法、CCIP 候选类、dHash 投票、最终 label/张数 |',
         '| `ccip_dhash64.parquet` | CCIP train/val/test 全部 1.14M 图的 dHash64 (与 danbooru-artist 同一实现), 泄漏检查用 |',
         '| `build_info.json` | 输入文件 (大小/mtime)、每步计数、排除/匹配统计、检查结果、抽查样例 |', '',
         '## 列', '',
         '`sample_key` (`dan_` + danbooru id), `shard` (绝对路径), `offset`, `size`, `ext`, `label` (int64; < 1e6 = CCIP '
         'ccip-merged-v1-index 的 label, >= 1,000,000 = 新角色), `series_id` (CCIP class_series 的 id 或 >= 1e6 新作品, -1 未知), '
         '`artist_id` (danbooru-artist 的画师 label), `character` (danbooru 标签), `ccip_matched`, `match_method`, `source`=danbooru, '
         '`danbooru_id`, `dhash64`, `image_sha256`, `copyright` (定 series_id 用的 copyright), `width`, `height`。', '',
         '```python',
         'from v3.data.indexed_tar import IndexedTarDataset',
         "df = pd.read_parquet('/data/neko/cxip_v3/data/ccip-danbooru-xstyle-v1/index/train.parquet')",
         "ds = IndexedTarDataset('/', df, columns=['series_id', 'artist_id'])   # shard 是绝对路径",
         '# 与 CCIP 合用: CCIP train 先按 train_exclude.json 去掉留出类, 再与本表 concat, 然后统一重映射 label;',
         '# PKBatchSampler(avoid_same=artist_id) 可让同类 K 张来自不同画师 (CCIP 部分 artist_id 置 -1)',
         '```', '',
         '## 统计', '',
         f'- 最终 **{c["final_images"]:,} 张 / {c["final_classes"]:,} 类**: 匹配到 CCIP 的 {mc["matched_classes"]:,} 类 '
         f'({mc["matched_images"]:,} 张), 新角色 {mc["new_classes"]:,} 类 ({mc["new_images"]:,} 张)。'
         f'覆盖 CCIP 训练类 {k["ccip_train_classes_covered"]:,} 个 ({100 * k["ccip_train_classes_covered_frac"]:.1f}% 的非留出类)。',
         f'- 每类画师数: 中位数 {k["artists_per_class"]["p50"]:.0f}, p10 {k["artists_per_class"]["p10"]:.0f}, '
         f'p90 {k["artists_per_class"]["p90"]:.0f}, 均值 {k["artists_per_class"]["mean"]:.1f} '
         f'(匹配类中位数 {k["artists_per_class_matched"]["p50"]:.0f}, 新类 {k["artists_per_class_new"]["p50"]:.0f}); '
         f'每类张数中位数 {k["images_per_class"]["p50"]:.0f}; 类内单个画师最大占比中位数 '
         f'{k["images_per_artist_within_class_max_share"]["p50"]:.2f}',
         f'- 画师数分箱: {k["classes_by_artist_count"]}', '',
         '| 步骤 | 张 |', '|-|-|',
         f'| danbooru-artist train | {c["danbooru_train_images"]:,} |',
         f'| 恰好 1 个角色标签 | {c["single_character_images"]:,} ({c["single_character_tags"]:,} 个标签) |',
         f'| 角色级排除 | -{ie["by_character_exclusion"]:,} ({ce["total_excluded_tags"]:,} 个标签) |',
         f'| 图片级排除 (角色级之外) | -{ie["image_level_only_after_character_exclusion"]:,} |',
         f'| 与 CCIP train 图相同 (sha256 / dHash 精确) | -{c["dup_with_ccip_train_removed"]:,} '
         f'(其中同类 {c["dup_with_ccip_train_removed_same_label"]:,}) |',
         f'| 同一张图出现在不同类 | -{c["cross_identity_dhash_dup_removed"]:,} |',
         f'| 同类 dHash 重复 | -{c["dedup_same_identity_dhash_removed"]:,} |',
         f'| 画师 < {MIN_ARTISTS} 的类 | -{c["images_dropped_lt_min_artists"]:,} ({c["identities_dropped_lt_min_artists"]:,} 类) |',
         f'| 每类上限 {CAP} 前 | {c["before_cap_images"]:,} ({c["before_cap_classes"]:,} 类) |',
         f'| 最终 | {c["final_images"]:,} |', '',
         '### 留出 (泄漏) 排除', '',
         '角色级 (按原因, 不互斥; 标签数 / 张数):', '']
    for r_, n_ in sorted(ce['tags_by_reason_nonexclusive'].items(), key=lambda x: -x[1]):
        L.append(f'- `{r_}`: {n_} / {ce["images_by_reason_nonexclusive"][r_]:,}')
    L += ['', '图片级 (在全部单角色图上统计, 不互斥): ' + ', '.join(
        f'`{x}` {ie[x]:,}' for x in ('sha256_eval_valtest', 'dhash_eval_valtest', 'dhash_eval_valtest_exact', 'closedset_eval',
                                     'heldout_franchise_copyright', 'heldout_hash_image')), '',
          '(`*_eval_valtest` = 开放集 eval_index 全部图 + CCIP val/test; `closedset_eval` = 旧闭集评测集 zc50x15 / zc500x10 / '
          'test_ccip / ccip_eval / ccip_test_v1 的文件)', '',
          f'检查: label 与 heldout_labels 交集 {k["labels_overlap_heldout"]}; 与开放集评测图 / CCIP val/test 的 sha256 重合 '
          f'{k["sha256_overlap_eval_valtest"]}, dHash (海明 <= {HAMMING}) 重合 {k[f"dhash_overlap_eval_valtest_d<={HAMMING}"]}; '
          f'与旧闭集评测集重合 {k[f"closedset_eval_overlap_sha256_or_dhash_d<={HAMMING}"]}; 与 CCIP train 图 sha256 / dHash 相同 '
          f'{k["sha256_dup_with_ccip_train_images"]} / {k["dhash_exact_dup_with_ccip_train_images"]}; '
          f'随机 {k["read_check"]["n"]} 张按绝对路径读取 + PIL 解码 {k["read_check"]["ok"]} 张成功。', '',
          '### 匹配', '',
          '| match_method | 类 (标签) | 张 |', '|-|-|-|']
    for meth, r_ in sorted(m['final_by_method'].items(), key=lambda x: -x[1]['images']):
        L.append(f'| {meth} | {r_["tags"]:,} | {r_["images"]:,} |')
    L += ['', f'- dHash 精确匹配 (信息量足够的哈希) 与名字匹配的一致率: ' + ', '.join(
        f'{k_} {v_["vote_share"]:.3f} ({v_["tags_with_votes"]} 个有票的标签)' for k_, v_ in m['hash_agreement'].items()),
          '- `new:*` 是没有无歧义 CCIP 对应的角色 (新 id); 其中 `new:ambiguous_name` / `new:hash_conflict` / '
          '`new:ambiguous_multi_class` / `new:franchise_conflict` 在 CCIP 里有同名候选, 可能是 CCIP 已有角色的重复类 '
          '(假负例风险), 见 characters.parquet 的 candidates。', '',
          '## 注意', '',
          '- 图片级检查只能抓到同一张图 (sha256 / dHash64 近重复); CCIP 的裁剪图 (crop=True、v2 头像) 与原图的 dHash 不同, '
          '查不到; 角色级排除 (名字/作品/dHash 投票) 是主要防线。dHash 海明 <= 3 的匹配里 d>=2 大多是误报 (宁可多删)。',
          '- 名字完全不同且没有共享图片的同一角色 (罕见译名) 可能漏网; 建议用 GPU 提特征后跑 '
          '`v3.eval.leak_check.class_leak_report` 复核 (本表新类 vs 开放集剔除类)。',
          '- 匹配沿用 `ccip_openset_v1/class_meta.parquet` 的作品/franchise 推断, 个别 CCIP 类的作品推断有误 (如 Horo -> ookami game), '
          '这时作品冲突 -> 不匹配 (新 id), 而不是硬并。CCIP 自身有重复类 (Asuna (ALO) / Yuuki Asuna 等), 这些标签记为 new:hash_conflict。',
          '- series_id: 匹配类优先取 CCIP 类自己的 series (若在该图的 copyright 里), 否则取该类图里最常见的 copyright; '
          f'匹配类的图片 series_id 与 CCIP 类一致的比例 {m["matched_images_series_equal_ccip_class"]:.3f}。', '']
    (out / 'README.md').write_text('\n'.join(L) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--workers', type=int, default=48)
    a = ap.parse_args()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    res, ct, mp, tags, info, aux = build(a.out, a.workers)
    t0 = aux['t0']
    # 最终按方法统计
    fin = res.groupby('match_method').agg(images=('sample_key', 'size'), labels=('label', 'nunique'),
                                          tags=('character', 'nunique'))
    info['matching']['final_by_method'] = {k: {c: int(v) for c, v in r.items()} for k, r in fin.iterrows()}
    info['matching']['final_classes_by_method'] = dict(
        matched_classes=int(res.loc[res.ccip_matched, 'label'].nunique()), matched_images=int(res.ccip_matched.sum()),
        new_classes=int(res.loc[~res.ccip_matched, 'label'].nunique()), new_images=int((~res.ccip_matched).sum()))
    agree = {}
    m = mp[(mp.ccip_label >= 0) & (mp.hash_votes > 0)]
    for meth, g in m.groupby('method'):
        agree[meth] = dict(tags_with_votes=int(len(g)), vote_share=float(g.hash_agree.sum() / g.hash_votes.sum()),
                           top_is_label=float((g.hash_top == g.ccip_label).mean()))
    info['matching']['hash_agreement'] = agree
    info['matching']['all_tags_by_method'] = {k: int(v) for k, v in mp['method'].value_counts().items()}
    print(f'[{time.time() - t0:.0f}s] checks ...', flush=True)
    info['checks'] = run_checks(res, ct, mp, info, aux)
    info['spot_checks'] = spot_checks(ct)
    info['timestamp'] = datetime.datetime.now().isoformat(timespec='seconds')
    info['outputs'] = {'index/train.parquet': len(res), 'index/characters.parquet': len(ct)}
    out = a.out
    res.to_parquet(out / 'index/train.parquet', index=False)
    ct.to_parquet(out / 'index/characters.parquet', index=False)
    json.dump(info, open(out / 'build_info.json', 'w'), indent=1, ensure_ascii=False, default=str)
    write_readme(out, res, info)
    print(json.dumps({k: info[k] for k in ('counts', 'exclusion', 'matching')}, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({k: v for k, v in info['checks'].items() if k != 'dtypes'}, indent=1, ensure_ascii=False, default=str))
    for k, v in info['spot_checks'].items():
        print(f'== {k}')
        for x in v:
            print('  ' + x)
    print(f'[{time.time() - t0:.0f}s] done -> {out}', flush=True)


if __name__ == '__main__':
    main()
