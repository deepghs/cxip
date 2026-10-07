"""
合并 CCIP 训练集第 3 步: 按 codex 判为 "same" 的候选对合并类别, 去重, 打包成 tar 分片 + 随机访问索引
python repro/ccip_merge/merge.py plan   # 只算合并/去重/划分, 写报告 (不读图片)
python repro/ccip_merge/merge.py pack   # 读三个来源的原图字节 (不重编码), 写 data/ccip_merged/shards/*.tar + index.parquet

输出 data/ccip_merged/:
  index.parquet   key, split, label, shard, offset, size, ext, zid, source, orig_path, orig_cid
                  (offset/size 是图片在 tar 里的字节位置, 直接 seek 读取)
  classes.json    label -> name, members (原始类别), 各 split 图片数
  variants.json   codex 判为 variant 的类别对 (同一人物的不同版本, 未合并)
  report.md       统计
"""
import collections
import glob
import io
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import pandas as pd

R = Path(os.environ.get('REPRO_ROOT', '/data/neko/cxip_repro'))
SRC = R / 'data/ccip_merge'
OUT = R / 'data/ccip_merged'
ZC = R / 'data/zerochan_m50_800k'
CS = R / 'data/zips/character_similarity'
EVAL_DIRS = ['zerochan_test_c50x15', 'zerochan_test_c500x10', 'ccip/test_ccip', 'ccip/ccip_eval/eval', 'ccip_test_v1']
MIN_TRAIN = 2
DROP_CATEGORIES = {'group_or_species', 'generic_or_collective', 'multiple_characters', 'uncertain'}
# flag_classes.py 的误报 (视觉复核确认是单一角色): v2 的 Azur_Lane/Yukikaze 文件夹里其实是舰 C 雪风
KEEP_FLAGGED = {'Yukikaze (Kantai Collection)'}
SHARD_BYTES = 1 << 30
PRIORITY = {'zc800k': 0, 'v1': 1, 'v2': 2}


def load_verdicts():
    res = {}
    for f in sorted(glob.glob(str(SRC / 'judge/batch_*.json'))):
        for x in json.load(open(f))['results']:
            res[x['id']] = x
    return res


def union_find(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        parent[find(a)] = find(b)
    return [find(i) for i in range(n)]


def plan():
    classes = json.load(open(SRC / 'classes.json'))
    cands = json.load(open(SRC / 'candidates.json'))
    verdicts = load_verdicts()
    missing = [i for i in range(len(cands)) if i not in verdicts]
    print(f'{len(cands)} candidates, {len(verdicts)} judged, {len(missing)} missing')
    same = [(cands[i]['a'], cands[i]['b']) for i, v in verdicts.items() if v['verdict'] == 'same']
    variants = [dict(a=classes[cands[i]['a']]['key'], b=classes[cands[i]['b']]['key'], reason=v['reason'],
                     sources=[classes[cands[i]['a']]['source'], classes[cands[i]['b']]['source']])
                for i, v in verdicts.items() if v['verdict'] == 'variant']
    root = union_find(len(classes), same)

    img = pd.read_parquet(SRC / 'images.parquet')
    img['root'] = img['cid'].map(lambda c: root[c])
    img['crop'] = img['path'].str.contains(r'_person\d+\.', regex=True)
    img['prio'] = img['source'].map(PRIORITY) * 2 + img['crop'].astype(int)
    n0 = len(img)

    # 评测/验证/测试图片的 zid 不进训练
    held = set(img.loc[img['split'] != 'train', 'zid'])
    for d in EVAL_DIRS:
        for p in (R / 'data' / d).rglob('*'):
            m = re.match(r'(\d+)', p.name)
            if p.is_file() and m:
                held.add(int(m.group(1)))
    drop_held = (img['split'] == 'train') & img['zid'].isin(held)
    # 整图出现在多个合并类里 = 多人图, 去掉 (人物裁剪保留)
    roots_per_zid = img.groupby('zid')['root'].nunique()
    multi = img['zid'].map(roots_per_zid) > 1
    drop_multi = multi & ~img['crop']
    img = img[~drop_held & ~drop_multi]
    # 同一合并类内同一 zid 只留一张
    n1 = len(img)
    img = img.sort_values(['root', 'zid', 'prio']).drop_duplicates(['root', 'zid', 'split'])
    n_dup = n1 - len(img)
    # 训练图片太少的类去掉 (val/test 也跟着去掉)
    ntrain = img[img['split'] == 'train'].groupby('root').size()
    keep_roots = set(ntrain[ntrain >= MIN_TRAIN].index)
    n_small = (~img['root'].isin(keep_roots)).sum()
    img = img[img['root'].isin(keep_roots)]

    # flag_classes.py 标出的非单一角色类别 (基于上一次 plan 的 label, 用成员集合对回来)
    dropped = []
    flag_files = sorted(glob.glob(str(OUT / 'flags/batch_*.json')))
    if flag_files:
        # flags 的 id 是 flag_classes.py 运行时的 label (筛查前那次 plan 的 classes.json, 已备份为 classes_preflag.json)
        prev = json.load(open(OUT / 'classes_preflag.json'))
        key_root = {(c['source'], c['key']): root[c['cid']] for c in classes}
        for f in flag_files:
            for x in json.load(open(f))['flagged']:
                pc = prev[x['id']]
                cat = x['category']
                if cat == 'non_character' and 'real' in x['reason'].lower():
                    continue  # 真人的同人图仍是单一身份
                if (cat in DROP_CATEGORIES or cat == 'non_character') and pc['name'] not in KEEP_FLAGGED:
                    r = key_root[(pc['members'][0]['source'], pc['members'][0]['key'])]
                    if r in keep_roots:
                        keep_roots.discard(r)
                        dropped.append(dict(name=pc['name'], category=cat, reason=x['reason'], n=pc['n']))
        n_flag = (~img['root'].isin(keep_roots)).sum()
        img = img[img['root'].isin(keep_roots)]
        json.dump(dropped, open(OUT / 'dropped_flagged.json', 'w'), ensure_ascii=False)
    else:
        n_flag = 0

    # 最终标签: 按训练图片数降序编号
    order = img[img['split'] == 'train'].groupby('root').size().sort_values(ascending=False).index
    label_of = {r: i for i, r in enumerate(order)}
    img['label'] = img['root'].map(label_of)
    members = collections.defaultdict(list)
    for c in classes:
        if root[c['cid']] in label_of:
            members[label_of[root[c['cid']]]].append(c)
    out_classes = []
    sp = img.groupby(['label', 'split']).size().unstack(fill_value=0)
    for lb in range(len(order)):
        ms = sorted(members[lb], key=lambda c: (PRIORITY[c['source']], -c['n']))
        nm = ms[0]['name'] + (f" ({ms[0]['context']})" if ms[0]['source'] != 'zc800k' and ms[0]['context'] else '')
        out_classes.append(dict(label=lb, name=nm, members=[dict(source=c['source'], key=c['key'], n=c['n']) for c in ms],
                                n={k: int(v) for k, v in sp.loc[lb].items() if v}))
    OUT.mkdir(parents=True, exist_ok=True)
    img = img.drop(columns=['root', 'prio']).rename(columns={'cid': 'orig_cid', 'path': 'orig_path'})
    img.to_parquet(OUT / 'plan.parquet')
    json.dump(out_classes, open(OUT / 'classes.json', 'w'), ensure_ascii=False)
    json.dump(variants, open(OUT / 'variants.json', 'w'), ensure_ascii=False)

    sizes = collections.Counter(len(c['members']) for c in out_classes)
    big = sorted(out_classes, key=lambda c: -len(c['members']))[:15]
    src_cnt = img.groupby(['split', 'source']).size().unstack(fill_value=0)
    vc = collections.Counter(v['verdict'] for v in verdicts.values())
    lines = [
        '# CCIP 合并数据集 (zc800k + character_similarity v1_pruned + v2)', '',
        f'- 原始: {len(classes)} 个类别 ({collections.Counter(c["source"] for c in classes)}), {n0} 张图',
        f'- 候选对 {len(cands)}, codex ({"gpt-6-astra"}) 判定: {dict(vc)}, 未判定 {len(missing)}',
        f'- 合并后类别 (训练图 >= {MIN_TRAIN}): {len(out_classes)}; 每类成员数分布: {dict(sorted(sizes.items()))}',
        f'- 去掉: 与 val/test/评测集重合的训练图 {int(drop_held.sum())}, 多人整图 {int(drop_multi.sum())}, '
        f'类内重复 {n_dup}, 训练图不足的类的图 {int(n_small)}',
        f'- 去掉 codex 标出的非单一角色类别 {len(dropped)} 个 ({dict(collections.Counter(d["category"] for d in dropped))}), '
        f'{int(n_flag)} 张图 (见 dropped_flagged.json; 视觉复核: data/ccip_merge/visual_check.parquet)',
        f'- 最终图片: {len(img)}', '', '| split | ' + ' | '.join(src_cnt.columns) + ' | total |', '|---' * (len(src_cnt.columns) + 2) + '|',
    ]
    for s, row in src_cnt.iterrows():
        lines.append(f'| {s} | ' + ' | '.join(str(int(v)) for v in row) + f' | {int(row.sum())} |')
    ntr = img[img['split'] == 'train'].groupby('label').size()
    lines += ['', f'- 每类训练图: median {int(ntr.median())}, mean {ntr.mean():.1f}, max {int(ntr.max())}',
              f'- variant 对 (未合并, 见 variants.json): {len(variants)}', '', '成员最多的合并类:']
    for c in big:
        lines.append(f"- {c['name']}: " + ', '.join(f"{m['source']}:{m['key']}" for m in c['members']))
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


class ShardWriter:
    def __init__(self, split):
        self.split, self.idx, self.tf, self.size = split, -1, None, 0
        (OUT / 'shards').mkdir(parents=True, exist_ok=True)

    def write(self, key, data):
        if self.tf is None or self.size >= SHARD_BYTES:
            if self.tf:
                self.tf.close()
            self.idx += 1
            self.name = f'{self.split}-{self.idx:05d}.tar'
            self.tf = tarfile.open(OUT / 'shards' / self.name, 'w', format=tarfile.USTAR_FORMAT)
            self.size = 0
        ti = tarfile.TarInfo(key)
        ti.size = len(data)
        offset = self.tf.offset + tarfile.BLOCKSIZE  # addfile 内部复制 TarInfo, 不能读 ti.offset_data; 短文件名 = 单个 512B 头
        self.tf.addfile(ti, io.BytesIO(data))
        self.size += len(data) + 1024
        return self.name, offset

    def close(self):
        if self.tf:
            self.tf.close()


def pack():
    img = pd.read_parquet(OUT / 'plan.parquet')
    want = {(s, p): i for i, (s, p) in enumerate(zip(img['source'], img['orig_path']))}
    loc = [None] * len(img)
    writers = {s: ShardWriter(s) for s in ['train', 'val', 'test']}
    splits, labels = img['split'].tolist(), img['label'].tolist()

    def put(i, data, ext):
        key = f'{labels[i]:06d}_{i:08d}.{ext}'
        loc[i] = (key, *writers[splits[i]].write(key, data), len(data), ext)

    for split in ['train', 'val', 'test']:  # zc800k
        for tp in sorted((ZC / split).glob('*.tar')):
            with tarfile.open(tp) as tf:
                for m in tf:
                    stem, ext = m.name.rsplit('.', 1)
                    i = want.get(('zc800k', f'{split}/{tp.name}:{stem}'))
                    if i is not None and ext != 'json':
                        put(i, tf.extractfile(m).read(), ext)
            print(tp, sum(x is not None for x in loc), flush=True)
    for source, arc, cmd in [('v1', 'images_pruned_v1.tar.xz', ['xz', '-T8', '-dc']),
                             ('v2', 'images_v2.tar.gz', ['pigz', '-dc'])]:
        proc = subprocess.Popen(cmd + [str(CS / arc)], stdout=subprocess.PIPE, bufsize=1 << 24)
        with tarfile.open(fileobj=proc.stdout, mode='r|') as tf:
            for n, m in enumerate(tf):
                i = want.get((source, m.name)) if m.isfile() else None
                if i is not None:
                    put(i, tf.extractfile(m).read(), m.name.rsplit('.', 1)[1].lower())
                if n % 100000 == 0:
                    print(source, n, sum(x is not None for x in loc), flush=True)
        proc.wait()
    for w in writers.values():
        w.close()
    miss = [i for i, x in enumerate(loc) if x is None]
    assert not miss, f'{len(miss)} images not found, e.g. {img.iloc[miss[:3]].to_dict("records")}'
    img['key'], img['shard'], img['offset'], img['size'], img['ext'] = zip(*loc)
    img[['key', 'split', 'label', 'shard', 'offset', 'size', 'ext', 'zid', 'source', 'orig_path', 'orig_cid', 'crop']] \
        .to_parquet(OUT / 'index.parquet')
    print('packed', len(img))


def fix_offsets():
    '''从 tar 头重建 index.parquet 的 offset (修复旧版 pack 写成 0 的问题)'''
    idx = pd.read_parquet(OUT / 'index.parquet')
    off = {}
    for shard in sorted(idx['shard'].unique()):
        with tarfile.open(OUT / 'shards' / shard) as tf:
            for m in tf:
                off[(shard, m.name)] = (m.offset_data, m.size)
    got = [off[(s, k)] for s, k in zip(idx['shard'], idx['key'])]
    assert all(sz == z for (_, sz), z in zip(got, idx['size']))
    idx['offset'] = [o for o, _ in got]
    idx.to_parquet(OUT / 'index.parquet')
    print('fixed', len(idx))


if __name__ == '__main__':
    {'plan': plan, 'pack': pack, 'fix_offsets': fix_offsets}[sys.argv[1]]()
