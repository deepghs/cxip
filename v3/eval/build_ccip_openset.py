"""CCIP 开放集划分 (v3 任务 0.7): 从 ccip_merged_nd 整类剔除约 1,000 个角色做开放集评测.

python -m v3.eval.build_ccip_openset            (在 /data/neko/cxip 下运行, CPU, 约 2 分钟)

划分规则 (固定 seed):
- 候选类: 全部 split 合计 >= 5 张图; 有作品信息; 不在旧闭集评测集 (zc500x10 / zc50x15 / test_ccip / ccip_eval /
  ccip_test_v1) 里 (这些集合因此保持闭集有效); 不是 variants.json 中未合并的变体对 (性转等, 视为同一身份的近亲).
- new_franchise 子集 (约 30%): 作品图按 "类 - franchise 键" 连通分量划分, 选中等大小 (总类数 4-40) 的分量整组剔除,
  分量里的全部类 (含不足 5 张、不进评测的小类) 都从训练里删掉. 一半分量从 "含骨干见过角色" 的分量里抽.
- hard 子集 (其余): 在保留于训练的大作品 (主 franchise >= 12 类) 内按组抽类, 每个作品最多剔除 15% / 30 类,
  同作品的其余角色 (长相相近) 留在训练里. 抽样按类大小分箱拉平、对骨干见过的类加权, 使 seen 约占 1/4.
- 剔除的类的全部图片 (train/val/test) 进评测; 与剔除图片同 zerochan id 的其他类训练图 (多人图的其他人物裁剪) 也从训练删.
- FULL: 全部候选剔除类, 每类最多 40 张 (按来源轮转抽样); FAST: 约 300 类 x <=10 张, 只取同作品至少 3 个剔除类的作品组
  (同作品困难负样本). FULL 的图片字节打包进 images/full-*.tar.
- OOD 协议: (N 已知, X 未知) in {(20,20),(40,20),(60,40),(500,500)}, A in {1,3,5} (锚点嵌套: A=1 ⊂ 3 ⊂ 5),
  20 次重复; (500,500) 按作品成组抽类, 组内一半已知一半未知 (未知类大多有同作品的已知类, 困难 OOD).
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from v3.eval import ccip_meta, leak_check
from v3.eval.tario import PackWriter, read_at, scan_tar

REPRO = ccip_meta.REPRO
MERGED = ccip_meta.MERGED
OUT = Path('/data/neko/cxip_v3/eval/ccip_openset_v1')
SEED = 20261020
EVAL_DIRS = {'zc50x15': 'zerochan_test_c50x15', 'zc500x10': 'zerochan_test_c500x10', 'test_ccip': 'ccip/test_ccip',
             'ccip_eval': 'ccip/ccip_eval/eval', 'ccip_test_v1': 'ccip_test_v1'}
SIZE_BINS = [5, 10, 20, 50, 150, 10 ** 9]


# --------------------------------------------------------------------------------------------------------- loading
def load_samples() -> pd.DataFrame:
    cols = ['sample_key', 'label', 'name', 'source', 'zid', 'split', 'shard_path', 'image_extension', 'orig_path', 'crop',
            'width', 'height']
    dfs = [pd.read_parquet(p, columns=cols) for p in sorted(MERGED.glob('metadata/*/*.parquet'))]
    return pd.concat(dfs, ignore_index=True)


def eval_set_labels(samples: pd.DataFrame) -> dict:
    """旧闭集评测集 -> 合并类 label (zc 文件夹名对 zc800k 成员; 其余按文件名里的 zerochan id 经合并前 images.parquet 对回)"""
    classes = ccip_meta.load_classes()
    pre = json.load(open(ccip_meta.PREMERGE / 'classes.json'))
    key2label = {(m['source'], m['key']): c['label'] for c in classes for m in c['members']}
    cid2label = {c['cid']: key2label.get((c['source'], c['key'])) for c in pre}
    img = pd.read_parquet(ccip_meta.PREMERGE / 'images.parquet', columns=['cid', 'zid'])
    img['label'] = img['cid'].map(cid2label)
    zid2label = img.dropna().groupby('zid')['label'].agg(lambda x: collections.Counter(x).most_common(1)[0][0])
    out = {}
    for name, d in EVAL_DIRS.items():
        labels, miss = set(), 0
        for p in (REPRO / 'data' / d).rglob('*'):
            if not p.is_file():
                continue
            lb = key2label.get(('zc800k', p.parent.name)) if name.startswith('zc') else None
            if lb is None:
                m = re.match(r'(\d+)', p.name)
                lb = zid2label.get(int(m.group(1))) if m else None
            if lb is None:
                miss += 1
            else:
                labels.add(int(lb))
        out[name] = dict(labels=sorted(labels), unmapped_files=miss)
    return out


def variant_labels() -> set:
    classes = ccip_meta.load_classes()
    key2label = {m['key']: c['label'] for c in classes for m in c['members']}
    out = set()
    for v in json.load(open(MERGED / 'variants.json')):
        if v.get('merged'):
            continue
        la, lb = key2label.get(v['a']), key2label.get(v['b'])
        if la is not None and lb is not None and la != lb:
            out |= {la, lb}
    return out


def franchise_components(meta: pd.DataFrame) -> np.ndarray:
    """类与其全部 franchise 键做并查集; 无作品信息的类单独成分量. 返回每类的分量 id"""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for lb, frs in zip(meta['label'], meta['franchises']):
        node = ('c', lb)
        find(node)
        for f in frs:
            parent[find(('f', f))] = find(node)
    roots = [find(('c', lb)) for lb in meta['label']]
    _, comp = np.unique([str(r) for r in roots], return_inverse=True)
    return comp


# ------------------------------------------------------------------------------------------------------- selection
def weighted_sample(rng, items, weights, k):
    """Efraimidis-Spirakis 加权无放回抽样"""
    if k >= len(items):
        return list(items)
    w = np.asarray(weights, dtype=np.float64)
    keys = rng.random(len(items)) ** (1.0 / w)
    return [items[i] for i in np.argsort(-keys)[:k]]


def size_bin(n):
    return int(np.searchsorted(SIZE_BINS, n, side='right') - 1)


def select(meta: pd.DataFrame, excluded: set, n_target=1000, frac_new=0.30, seed=SEED, seen_weight=None):
    rng = np.random.default_rng(seed)
    meta = meta.copy()
    meta['comp'] = franchise_components(meta)
    meta['eligible'] = (meta['n_total'] >= 5) & (meta['franchises'].map(len) > 0) & ~meta['label'].isin(excluded)
    meta['size_bin'] = meta['n_total'].map(size_bin)
    comp_n = meta.groupby('comp').size()
    comp_elig = meta.groupby('comp')['eligible'].sum()
    comp_bad = meta[meta['label'].isin(excluded)].groupby('comp').size()  # 含旧评测集/变体类的分量不能整组剔除
    comp_seen = meta[meta['eligible'] & (meta['backbone_seen'] == 'seen')].groupby('comp').size()

    # ---- new_franchise: 中等大小分量整组剔除
    cands = [c for c in comp_n.index if 4 <= comp_n[c] <= 40 and comp_elig[c] >= 3 and c not in comp_bad.index]
    with_seen = [c for c in cands if comp_seen.get(c, 0) > 0]
    without = [c for c in cands if comp_seen.get(c, 0) == 0]
    rng.shuffle(with_seen)
    rng.shuffle(without)
    new_comps, n_new, i, j = [], 0, 0, 0
    target_new = int(round(n_target * frac_new))
    while n_new < target_new and (i < len(with_seen) or j < len(without)):
        take_seen = (i < len(with_seen)) and (len(new_comps) % 2 == 0 or j >= len(without))
        c = with_seen[i] if take_seen else without[j]
        i, j = (i + 1, j) if take_seen else (i, j + 1)
        if n_new + comp_elig[c] > target_new * 1.08:
            continue
        new_comps.append(c)
        n_new += int(comp_elig[c])
    new_set = set(new_comps)
    meta['subset'] = ''
    meta.loc[meta['comp'].isin(new_set), 'subset'] = 'new_franchise'

    # ---- hard: 保留于训练的大作品内按组抽类
    # 作品按 sqrt(候选类数) 加权无放回抽, 每个作品剔除一组 g = clip(0.8*sqrt(候选数), 3, 配额) 个类,
    # 配额 = min(30, 15% 的作品类数), 直到凑满; 组内按类大小分箱拉平、骨干见过的类加权
    pool = meta[meta['eligible'] & ~meta['comp'].isin(new_set) & (meta['franchise'] != '')]
    pf_n = meta[~meta['comp'].isin(new_set)].groupby('franchise').size()
    groups = {f: g for f, g in pool.groupby('franchise') if f and pf_n[f] >= 20}
    quota = {f: min(30, int(0.15 * pf_n[f]), len(g)) for f, g in groups.items()}
    gsize = {f: int(np.clip(round(0.8 * np.sqrt(len(g))), 3, max(quota[f], 3))) for f, g in groups.items() if quota[f] >= 3}
    fs = sorted(gsize)
    n_hard = n_target - n_new
    chosen_f = weighted_sample(rng, fs, [np.sqrt(len(groups[f])) for f in fs], len(fs))  # 加权随机排列
    plan, tot = [], 0
    for f in chosen_f:
        if tot >= n_hard:
            break
        g = min(gsize[f], max(3, n_hard - tot))
        plan.append((f, g))
        tot += g
    sel_pool = pd.concat([groups[f] for f, _ in plan])
    bin_freq = sel_pool['size_bin'].value_counts(normalize=True)
    if seen_weight is None:  # 使骨干见过的类在 hard 子集里约占 1/4 (目标 odds 按 35% 设, 分箱拉平会稀释一些)
        p_seen = float((sel_pool['backbone_seen'] == 'seen').mean())
        seen_weight = max(1.0, (0.35 / 0.65) / max(p_seen / (1 - p_seen), 1e-6))
    hard = []
    for f, k in sorted(plan):
        g = groups[f]
        w = (1.0 / np.sqrt(g['size_bin'].map(bin_freq).values)) * np.where(g['backbone_seen'] == 'seen', seen_weight, 1.0)
        hard += weighted_sample(rng, list(g['label']), w, k)
    meta.loc[meta['label'].isin(set(hard)), 'subset'] = 'hard'
    alloc = dict(plan)
    return meta, dict(new_franchise_components=len(new_comps), seen_weight=float(seen_weight),
                      hard_franchises=len(plan), hard_group_sizes=dict(collections.Counter(k for _, k in plan)))


def pick_images(samples: pd.DataFrame, labels, k: int, seed: int) -> pd.DataFrame:
    """每类最多 k 张: 按 (来源, split) 轮转、组内随机, 让人物全身图 (zc/v1) 和 v2 都有"""
    rng = np.random.default_rng(seed)
    out = []
    sub = samples[samples['label'].isin(set(labels))]
    for lb, g in sub.groupby('label', sort=True):
        buckets = []
        for _, gg in g.groupby(['source', 'split'], sort=True):
            buckets.append(list(gg['sample_key'].values[rng.permutation(len(gg))]))
        order = []
        while any(buckets):
            for b in buckets:
                if b:
                    order.append(b.pop())
        out += [(lb, key, r) for r, key in enumerate(order[:k])]
    return pd.DataFrame(out, columns=['label', 'sample_key', 'rank'])


def select_fast(meta_h: pd.DataFrame, n_classes=300, frac_new=0.3, max_per_group=8, seed=SEED):
    """FAST: 同作品至少 3 个剔除类的作品组, 每组最多 8 类; new_franchise / hard 按比例"""
    rng = np.random.default_rng(seed + 1)
    chosen = []
    for subset, target in (('new_franchise', int(n_classes * frac_new)), ('hard', n_classes - int(n_classes * frac_new))):
        g = meta_h[meta_h['subset'] == subset]
        key = 'comp' if subset == 'new_franchise' else 'franchise'
        groups = [list(x['label']) for _, x in g.groupby(key) if len(x) >= 3]
        rng.shuffle(groups)
        got = []
        for grp in groups:
            if len(got) >= target:
                break
            take = list(rng.permutation(grp))[:max_per_group]
            got += take[:max(3, min(len(take), target - len(got)))]
        chosen += got
    return chosen


# ------------------------------------------------------------------------------------------------------------- OOD
def build_ood(meta_h: pd.DataFrame, full_imgs: pd.DataFrame, seed=SEED, repeats=20,
              configs=((20, 20), (40, 20), (60, 40), (500, 500)), anchors=(1, 3, 5)):
    rng = np.random.default_rng(seed + 2)
    keys_by_label = {lb: list(g.sort_values('rank')['sample_key']) for lb, g in full_imgs.groupby('label')}
    n_img = {lb: len(v) for lb, v in keys_by_label.items()}
    amax = max(anchors)
    known_ok = sorted(lb for lb in keys_by_label if n_img[lb] >= amax + 1)
    all_lb = sorted(keys_by_label)
    fr = meta_h.set_index('label')['franchise'].to_dict()
    protos = []
    for n_known, n_unk in configs:
        reps = []
        for r in range(repeats):
            if n_known + n_unk <= 200:
                known = sorted(int(x) for x in rng.choice(known_ok, n_known, replace=False))
                rest = sorted(set(all_lb) - set(known))
                unknown = sorted(int(x) for x in rng.choice(rest, n_unk, replace=False))
            else:  # 按作品成组: 打乱作品组, 组内一半已知一半未知
                groups = collections.defaultdict(list)
                for lb in all_lb:
                    groups[fr.get(lb, '') or f'_{lb}'].append(lb)
                order = list(groups)
                rng.shuffle(order)
                known, unknown = [], []
                for gname in order:
                    g = list(rng.permutation(groups[gname]))
                    gk = [x for x in g if x in set(known_ok)]
                    h = (len(g) + 1) // 2
                    for x in g:
                        if len(known) < n_known and x in gk and sum(1 for y in g if y in known) < h:
                            known.append(int(x))
                    for x in g:
                        if x not in known and len(unknown) < n_unk:
                            unknown.append(int(x))
                    if len(known) >= n_known and len(unknown) >= n_unk:
                        break
                # 组内配不满时从剩余的补
                if len(known) < n_known:
                    extra = [x for x in known_ok if x not in known and x not in unknown]
                    known += [int(x) for x in rng.choice(extra, n_known - len(known), replace=False)]
                if len(unknown) < n_unk:
                    extra = [x for x in all_lb if x not in known and x not in unknown]
                    unknown += [int(x) for x in rng.choice(extra, min(len(extra), n_unk - len(unknown)), replace=False)]
                known, unknown = sorted(known), sorted(unknown)
            anchors_r = {str(lb): [keys_by_label[lb][i] for i in rng.choice(n_img[lb], amax, replace=False)]
                         for lb in known}
            reps.append(dict(known=known, unknown=unknown, anchors=anchors_r))
        protos.append(dict(N=n_known, X=n_unk, repeats=reps))
    return dict(version='ccip_ood_v1', eval_set='full', seed=seed, anchors=list(anchors),
                note='anchors[label] 有 max(A) 个 sample_key, A=k 时取前 k 个; queries = 已知类的非锚点图 + 未知类全部图 '
                     '(都限于 FULL 集); 已知类分数 = A 个锚点 cos 的均值; max 分数 < tau 判未知',
                protocols=protos)


# ------------------------------------------------------------------------------------------------------------ main
def locate(samples: pd.DataFrame) -> pd.DataFrame:
    """补上每个样本图片在原 tar 里的 (offset, size)"""
    need = samples[['split', 'shard_path']].drop_duplicates()
    rows = []
    for split, shard in need.itertuples(index=False):
        for name, off, size in scan_tar(MERGED / 'wds' / split / shard):
            key, ext = name.rsplit('.', 1)
            if ext in ('json', 'txt'):
                continue
            rows.append((split, shard, key, off, size))
    loc = pd.DataFrame(rows, columns=['split', 'shard_path', 'sample_key', 'offset', 'size'])
    return samples.merge(loc, on=['split', 'shard_path', 'sample_key'], how='left')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--n-classes', type=int, default=1000)
    ap.add_argument('--full-k', type=int, default=40)
    ap.add_argument('--fast-k', type=int, default=10)
    ap.add_argument('--no-pack', action='store_true')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    meta, aux, _ = ccip_meta.build_class_meta(out / 'class_meta.parquet')
    samples = load_samples()
    ev = eval_set_labels(samples)
    var = variant_labels()
    excluded = set(var)
    for v in ev.values():
        excluded |= set(v['labels'])
    meta_s, sel_info = select(meta, excluded, n_target=args.n_classes)
    # 与评测类同名 (同 franchise 或 franchise 未知) 的训练类也剔除 (不评测), 防止合并漏掉的同一角色留在训练里
    sel = meta_s[meta_s['subset'] != '']
    nd = leak_check.name_leak_report(meta_s, sel.loc[sel['eligible'], 'label'])
    dup = sorted({x for v in nd['dup_train_labels'] for x in v} - set(sel['label']) - excluded)
    meta_s.loc[meta_s['label'].isin(dup), 'subset'] = 'name_dup'
    sel_info['name_dup_classes'] = len(dup)
    held = meta_s[meta_s['subset'] != ''].copy()   # 剔除的类 (含 new_franchise 分量里不足 5 张的小类、name_dup)
    held['eval'] = held['eligible'] & held['subset'].isin(['new_franchise', 'hard'])
    held_eval = held[held['eval']]
    print(f"held-out classes {len(held)}, eval classes {len(held_eval)}: {held_eval['subset'].value_counts().to_dict()}")

    # ---- 图片
    full = pick_images(samples, held_eval['label'], args.full_k, SEED)
    fast_labels = select_fast(held_eval)
    fast = full[full['label'].isin(set(fast_labels)) & (full['rank'] < args.fast_k)]
    hs = samples[samples['label'].isin(set(held['label']))].copy()
    hs = locate(hs)
    assert hs['offset'].notna().all()
    cm = held.set_index('label')
    for col in ('series', 'franchise', 'backbone_seen', 'subset'):
        hs[col] = hs['label'].map(cm[col])
    hs['eval_class'] = hs['label'].map(cm['eval'])
    hs['in_full'] = hs['sample_key'].isin(set(full['sample_key']))
    hs['in_fast'] = hs['sample_key'].isin(set(fast['sample_key']))
    hs['full_rank'] = hs['sample_key'].map(full.set_index('sample_key')['rank']).fillna(-1).astype(int)
    hs['shard_path'] = 'wds/' + hs['split'] + '/' + hs['shard_path']
    hs = hs.rename(columns={'split': 'orig_split', 'image_extension': 'ext'})

    # zid 重叠: 同一 zerochan 图里其他人物的训练图
    held_zids = set(hs['zid'])
    overlap = samples[~samples['label'].isin(set(held['label'])) & samples['zid'].isin(held_zids)]

    # ---- 打包 FULL 图片
    if not args.no_pack:
        pw = PackWriter(out / 'images', 'full')
        locs = {}
        sub = hs[hs['in_full']].sort_values(['label', 'full_rank'])
        for r in sub.itertuples(index=False):
            data = read_at(MERGED / r.shard_path, int(r.offset), int(r.size))
            locs[r.sample_key] = pw.write(f'{r.sample_key}.{r.ext}', data)
        pw.close()
        hs['pack_tar'] = hs['sample_key'].map(lambda k: f'images/{locs[k][0]}' if k in locs else None)
        hs['pack_offset'] = hs['sample_key'].map(lambda k: locs[k][1] if k in locs else -1)
    cols = ['sample_key', 'label', 'name', 'series', 'franchise', 'backbone_seen', 'subset', 'eval_class', 'in_full',
            'in_fast', 'full_rank', 'orig_split', 'source', 'zid', 'crop', 'width', 'height', 'ext', 'shard_path',
            'offset', 'size', 'orig_path'] + (['pack_tar', 'pack_offset'] if not args.no_pack else [])
    hs = hs[cols].sort_values(['label', 'full_rank', 'sample_key']).reset_index(drop=True)
    hs.to_parquet(out / 'eval_index.parquet')
    hs[hs['in_full']].reset_index(drop=True).to_parquet(out / 'full.parquet')
    hs[hs['in_fast']].reset_index(drop=True).to_parquet(out / 'fast.parquet')

    # ---- json
    def cls_rec(r):
        return dict(label=int(r.label), name=r.name, series=r.series, franchise=r.franchise,
                    franchises=list(r.franchises), backbone_seen=r.backbone_seen, backbone_tags=list(r.backbone_tags),
                    danbooru_tags=list(r.danbooru_tags)[:5], match_method=r.match_method, subset=r.subset,
                    eval=bool(r.eval), in_fast=int(r.label) in set(fast_labels),
                    n_images={'train': int(r.n_train), 'val': int(r.n_val), 'test': int(r.n_test)}, sources=r.sources)
    held_json = dict(version='ccip_openset_v1', seed=SEED, dataset='ccip_merged_nd (cxip-ccip-merged/v1)',
                     classes=[cls_rec(r) for r in held.sort_values('label').itertuples(index=False)])
    json.dump(held_json, open(out / 'heldout_classes.json', 'w'), ensure_ascii=False, indent=0)
    comp_fr = sorted({f for frs in held.loc[held['subset'] == 'new_franchise', 'franchises'] for f in frs})
    excl = dict(version='ccip_openset_v1', seed=SEED,
                note='训练 (v2 数据集) 必须删掉: heldout_labels 的全部图片 (所有 split) + zid_overlap_sample_keys. '
                     'heldout_franchises 下的所有类都已在 heldout_labels 里 (按 franchise 连通分量整组剔除).',
                heldout_labels=sorted(int(x) for x in held['label']),
                heldout_franchises=comp_fr,
                heldout_series=sorted({s for s in held.loc[held['subset'] == 'new_franchise', 'series'] if s}),
                exclude_sample_keys=sorted(hs['sample_key']),
                zid_overlap_sample_keys=sorted(overlap['sample_key']),
                counts=dict(classes=len(held), images=int(len(hs)), zid_overlap=int(len(overlap)),
                            images_by_split=hs['orig_split'].value_counts().to_dict()))
    json.dump(excl, open(out / 'train_exclude.json', 'w'), ensure_ascii=False)

    ood = build_ood(held_eval.assign(comp=held_eval['comp']), full)
    json.dump(ood, open(out / 'ood_protocol.json', 'w'))
    json.dump(dict(eval_sets=ev, variant_labels=sorted(var), selection=sel_info, aux={k: v for k, v in aux.items()
                                                                                        if k != 'series_alias'}),
              open(out / 'build_info.json', 'w'), indent=1, ensure_ascii=False)
    write_readme(out, meta_s, held, hs, ev, sel_info, aux, overlap, samples, ood)
    print('done', out)


def _pct(a, b):
    return f'{100 * a / max(b, 1):.1f}%'


def write_readme(out, meta_s, held, hs, ev, sel_info, aux, overlap, samples, ood):
    he = held[held['eval']]
    full = hs[hs['in_full']]
    fast = hs[hs['in_fast']]
    L = ['# CCIP 开放集评测 v1 (ccip_openset_v1)', '',
         '由 `v3/eval/build_ccip_openset.py` 生成 (seed %d)。数据源: `ccip_merged_nd` (= R2 `datasets/khub/cxip-ccip-merged/v1`)。' % SEED,
         '划分规则见脚本文件头; 类别元数据 (作品、Danbooru 匹配) 见 `v3/eval/ccip_meta.py`。', '',
         '## 文件', '',
         '| 文件 | 内容 |', '|-|-|',
         '| `heldout_classes.json` | 剔除的类: label、名字、作品、franchise、backbone_seen、subset、是否进评测 (eval)、是否在 FAST |',
         '| `train_exclude.json` | 训练要删掉的: 剔除类 label、整组剔除的 franchise 键、全部样本 key (所有 split)、zid 重叠的其他类训练图 |',
         '| `eval_index.parquet` | 剔除类的每一张图: sample_key, label, series, backbone_seen, subset, in_full/in_fast, 原 tar 位置 (shard_path, offset, size), 打包位置 (pack_tar, pack_offset) |',
         '| `full.parquet` / `fast.parquet` | FULL / FAST 评测集 (eval_index 的子集) |',
         '| `images/full-*.tar` | FULL 图片字节 (FAST 是其子集); `pack_offset` 处直接读 `size` 字节 |',
         '| `ood_protocol.json` | OOD zero-shot 协议 (已知/未知类与锚点, 引用 FULL 的 sample_key) |',
         '| `class_meta.parquet` | 全部 33,173 类的元数据 (作品、franchise、Danbooru 匹配、backbone_seen) |',
         '| `build_info.json` | 旧评测集映射到的类、变体类、抽样参数 |', '',
         '读取: `v3.eval.evalsets.load_ccip_openset("full")` 返回 (DataFrame, 读图函数)。', '']
    L += ['## 统计', '',
          '| | 类 | 图片 | seen | unseen | unknown |', '|-|-|-|-|-|-|']

    def row(name, df_cls, n_img):
        vc = df_cls['backbone_seen'].value_counts()
        return f"| {name} | {len(df_cls)} | {n_img} | {vc.get('seen', 0)} | {vc.get('unseen', 0)} | {vc.get('unknown', 0)} |"
    L.append(row('剔除 (训练里删掉)', held, len(hs)))
    L.append(row('评测类 (>= 5 张)', he, int(hs['eval_class'].sum())))
    for sub in ('new_franchise', 'hard'):
        d = he[he['subset'] == sub]
        L.append(row(f'  {sub}', d, int(hs[hs['eval_class'] & (hs['subset'] == sub)].shape[0])))
    fl = held[held['label'].isin(set(full['label']))]
    L.append(row('FULL', fl, len(full)))
    for sub in ('new_franchise', 'hard'):
        d = fl[fl['subset'] == sub]
        L.append(row(f'  FULL {sub}', d, int((full['subset'] == sub).sum())))
    fa = held[held['label'].isin(set(fast['label']))]
    L.append(row('FAST', fa, len(fast)))
    for sub in ('new_franchise', 'hard'):
        d = fa[fa['subset'] == sub]
        L.append(row(f'  FAST {sub}', d, int((fast['subset'] == sub).sum())))
    npairs = len(full) * (len(full) - 1) // 2
    pos = int((full.groupby('label').size() * (full.groupby('label').size() - 1) // 2).sum())
    L += ['', f'- FULL: {len(full)} 张 / {npairs:,} 对 (正样本对 {pos:,}); 每类图片数 中位数 {int(full.groupby("label").size().median())}',
          f'- FAST: {len(fast)} 张 / {len(fast) * (len(fast) - 1) // 2:,} 对; 作品组数 {fast.groupby("franchise").ngroups}',
          f'- 剔除图片按原 split: {hs["orig_split"].value_counts().to_dict()}; 来源: {hs["source"].value_counts().to_dict()}',
          f'- zid 重叠 (同一 zerochan 图里其他人物的训练图, 也要从训练删): {len(overlap)} 张',
          f'- new_franchise: {sel_info["new_franchise_components"]} 个 franchise 分量整组剔除 (含不足 5 张不进评测的类 '
          f'{int(((~held["eval"]) & (held["subset"] == "new_franchise")).sum())} 个); hard: 分布在 {sel_info["hard_franchises"]} 个作品, 每作品最多剔除 15% / 30 类',
          f'- name_dup: 与评测类同名 (franchise 相同或未知) 的训练类 {sel_info.get("name_dup_classes", 0)} 个也剔除 (不评测), '
          f'见 `v3.eval.leak_check.name_leak_report`',
          f'- 骨干见过 (seen) 的类在 hard 抽样中加权 x{sel_info["seen_weight"]:.2f}', '']
    # series 统计
    tr_left = meta_s[~meta_s['label'].isin(set(held['label']))]
    L += ['### 作品', '',
          f'- 全部类有作品信息的比例: {_pct((meta_s["series"] != "").sum(), len(meta_s))} '
          f'(>= 5 张的类: {_pct(((meta_s["series"] != "") & (meta_s["n_total"] >= 5)).sum(), (meta_s["n_total"] >= 5).sum())})',
          f'- 来源: Danbooru copyright {_pct((meta_s["db_series"].map(len) > 0).sum(), len(meta_s))}, '
          f'仅成员目录/限定词 {_pct(((meta_s["db_series"].map(len) == 0) & (meta_s["series"] != "")).sum(), len(meta_s))}',
          f'- 评测类涉及 {he["franchise"].nunique()} 个主 franchise; FAST 的作品组: '
          + ', '.join(f'{k} ({v})' for k, v in fa['franchise'].value_counts().head(15).items()) + ' ...', '',
          '整组剔除的 franchise 分量 (new_franchise; 代表作品名: 评测类 / 剔除类):', '',
          ', '.join(f"{g['series'].value_counts().index[0]} ({int(g['eval'].sum())}/{len(g)})"
                    for _, g in sorted(held[held['subset'] == 'new_franchise'].groupby('comp'), key=lambda x: -len(x[1]))), '',
          'hard 子集最多的作品: ' + ', '.join(f'{k} ({v})' for k, v in he[he['subset'] == 'hard']['franchise'].value_counts().head(25).items()), '']
    # Danbooru
    vc = meta_s['backbone_seen'].value_counts()
    mm = meta_s['match_method'].value_counts()
    L += ['### Danbooru 角色标签匹配 (骨干是否见过)', '',
          f'- 骨干 character 标签 {aux["n_backbone_character_tags"]} 个, 其中 {aux["n_backbone_tags_matched"]} 个 '
          f'({_pct(aux["n_backbone_tags_matched"], aux["n_backbone_character_tags"])}) 对上了本数据集的某个类',
          f'- 全部 {len(meta_s)} 类: seen {vc.get("seen", 0)}, unseen {vc.get("unseen", 0)}, unknown {vc.get("unknown", 0)}',
          f'- 匹配方法: 名字+作品 (high) {mm.get("series", 0)}, 名字唯一 (medium) {mm.get("unique_name", 0)}, '
          f'有歧义 {mm.get("ambiguous", 0)}, 同名不同作品 {mm.get("series_mismatch", 0)}, 无同名标签 {mm.get("none", 0)}; '
          f'有 Danbooru 匹配的类 {_pct((meta_s["danbooru_tags"].map(len) > 0).sum(), len(meta_s))}',
          f'- unknown 的原因: 名字与骨干标签松散重合 {int((meta_s["unknown_reason"] == "loose_name_match").sum())}, '
          f'未匹配但热门 (zerochan 图数 >= {aux["zc_popularity_threshold"]:.0f} 或 v1/v2 成员 >= {aux["member_popular"]} 张) '
          f'{int((meta_s["unknown_reason"] == "popular_unmatched").sum())}',
          '- unseen = 匹配到非骨干标签, 或没有任何同名/近名骨干标签且不热门. 评测时 seen / unseen 分开报告, unknown 只进总体.', '']
    ex = meta_s[(meta_s['backbone_seen'] == 'unknown')].sample(min(12, (meta_s['backbone_seen'] == 'unknown').sum()), random_state=0)
    L += ['不确定 (unknown) 的例子:', '']
    for r in ex.itertuples(index=False):
        L.append(f'- {r.name} [{r.series}]: {r.unknown_reason}; 近似骨干标签 {list(r.loose_backbone)[:3]}')
    ex = meta_s[meta_s['match_method'] == 'unique_name'].sample(8, random_state=1)
    L += ['', '中等可信 (只靠名字唯一) 的匹配例子:', '']
    for r in ex.itertuples(index=False):
        L.append(f'- {r.name} -> {list(r.danbooru_tags)[:2]} ({r.backbone_seen})')
    # 旧评测集
    L += ['', '## 旧闭集评测集', '',
          '旧评测集里出现的类全部排除在候选之外 (含这些类的 franchise 分量也不整组剔除), 所以剔除后它们仍然是闭集评测, 可继续与 HF AnimeCCIP 对比:', '',
          '| 评测集 | 映射到的类 | 未映射文件 | 与剔除类重合 | 仍有效 |', '|-|-|-|-|-|']
    hl = set(held['label'])
    for k, v in ev.items():
        inter = len(set(v['labels']) & hl)
        L.append(f"| {k} | {len(v['labels'])} | {v['unmapped_files']} | {inter} | {'是' if inter == 0 else '否'} |")
    zc_test = samples[samples['split'] == 'test']
    L += ['', f'- ccip_merged 的 val/test (zc800k 划分) 中属于剔除类的图片随剔除类一起进开放集; 剩下的 val/test 仍是闭集: '
              f'test {int((~zc_test["label"].isin(hl)).sum())} 张 / {zc_test.loc[~zc_test["label"].isin(hl), "label"].nunique()} 类 '
              f'(原 {len(zc_test)} / {zc_test["label"].nunique()})',
          '- CSIP 评测集 (test100_0918、csip_eval_public) 不受影响。', '',
          '## OOD zero-shot 协议', '',
          '`ood_protocol.json`: ' + ', '.join(f"(N={p['N']}, X={p['X']})" for p in ood['protocols'])
          + f', 每个 {len(ood["protocols"][0]["repeats"])} 次重复, A in {ood["anchors"]} (锚点嵌套)。 '
          '(500, 500) 按作品成组抽类、组内一半已知一半未知。用 `v3.eval.zeroshot.ood_protocol_eval` 计算。', '',
          '## 注意', '',
          '- 名字匹配不到的同一角色 (完全不同的译名) 可能仍以别的类留在训练里; 建议主会话用 GPU 提完 CCIP 特征后跑 '
          '`v3.eval.leak_check.class_leak_report` 复核 (剔除类中心与训练类中心的最大相似度).',
          '- v2 来源只有 train split, 图片大多是头像/半身; zc800k 是单人全身图。FULL 按来源轮转抽图。', '']
    (out / 'README.md').write_text('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
