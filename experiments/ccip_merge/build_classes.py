"""
合并 CCIP 训练集第 1 步: 汇总三个来源的角色类别和图片 (zerochan id), 生成同角色候选对
  zc800k : deepghs/zerochan-character-w640-ws-m50-800k  (webdataset, json 里的 character 字段)
  v1     : deepghs/character_similarity images_pruned_v1   (game/character/{id}.jpg)
  v2     : deepghs/character_similarity images_v2          (series/char[/subchar...]/zerochan_{id}[_personN].jpg, 不同路径=不同角色)
输出 (OUT=data/ccip_merge):
  classes.json     [{cid, source, key, name, context, n, split_n}]
  images.parquet   source, cid, path(tar 内路径或 tar:member), zid, split
  candidates.json  [{a, b, reasons, shared}]  待 codex 判断的候选对
"""
import collections
import json
import os
import re
import tarfile
import unicodedata
from pathlib import Path

import pandas as pd

R = Path(os.environ.get('REPRO_ROOT', '/data/neko/cxip_repro'))
ZC = R / 'data/zerochan_m50_800k'
CS = R / 'data/zips/character_similarity'
OUT = R / 'data/ccip_merge'


def tar_list(path):
    for line in open(path):
        parts = line.split(None, 5)
        if parts[0].startswith('-'):
            yield parts[5].rstrip('\n')


def norm(s):
    s = unicodedata.normalize('NFKC', s).lower()
    s = re.sub(r'[_\-\s.·・]+', ' ', s)
    return re.sub(r'[^\w ()]', '', s).strip()


def split_qual(s):
    m = re.match(r'^(.*?)\s*\((.*)\)\s*$', s)
    return (m.group(1), m.group(2)) if m else (s, '')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    classes, rows = [], []

    def add_class(source, key, name, context):
        classes.append(dict(cid=len(classes), source=source, key=key, name=name, context=context))
        return len(classes) - 1

    # zc800k
    zc_cls = {}
    for split in ['train', 'val', 'test']:
        for tp in sorted((ZC / split).glob('*.tar')):
            with tarfile.open(tp) as tf:  # 可 seek, 只读 json
                for m in tf:
                    if not m.name.endswith('.json'):
                        continue
                    d = json.load(tf.extractfile(m))
                    ch = d['character']
                    if ch not in zc_cls:
                        base, qual = split_qual(ch)
                        zc_cls[ch] = add_class('zc800k', ch, ch, qual)
                    rows.append(('zc800k', zc_cls[ch], f'{split}/{tp.name}:{m.name[:-5]}', int(d['id']), split))
            print(tp, len(rows), flush=True)

    # v1 / v2
    for source, lst in [('v1', 'pruned_v1.list'), ('v2', 'v2.list')]:
        cls = {}
        for f in tar_list(CS / lst):
            d, base = f.rsplit('/', 1)
            if d not in cls:
                parts = d.split('/')
                if source == 'v1':
                    name, context = parts[-1].replace('_', ' '), parts[0]
                else:
                    series = re.sub(r'^\d+_\d+_', '', parts[0]).replace('_', ' ')
                    name = parts[-1].replace('_', ' ')
                    context = ' / '.join([series] + [p.replace('_', ' ') for p in parts[1:-1]])
                cls[d] = add_class(source, d, name, context)
            zid = int(re.search(r'(\d+)', base).group(1))
            rows.append((source, cls[d], f, zid, 'train'))
        print(source, len(cls), flush=True)

    img = pd.DataFrame(rows, columns=['source', 'cid', 'path', 'zid', 'split'])
    img.to_parquet(OUT / 'images.parquet')
    cnt = img.groupby('cid').size()
    sp = img.groupby(['cid', 'split']).size().unstack(fill_value=0)
    for c in classes:
        c['n'] = int(cnt[c['cid']])
        c['split_n'] = {k: int(v) for k, v in sp.loc[c['cid']].items() if v}
    with open(OUT / 'classes.json', 'w') as f:
        json.dump(classes, f, ensure_ascii=False)

    # 候选对
    cand = collections.defaultdict(lambda: dict(reasons=set(), shared=0))
    # 1) 名字: 规范化全名 / 去括号限定的基名 / 词序无关
    keys = collections.defaultdict(set)
    for c in classes:
        full = norm(c['name'])
        base, qual = (norm(x) for x in split_qual(c['name']))
        # 'qual': FGO 之类 "Archer (Robin Hood)" 的真名在括号里, 和其他来源的 "robin hood" 对上
        for kind, k in [('name', full), ('base', base), ('tokens', ' '.join(sorted(base.split()))), ('qual', qual)]:
            if k:
                keys[(kind, k)].add(c['cid'])
    for c in classes:  # 括号内真名 也和其他类的全名/基名比
        qual = norm(split_qual(c['name'])[1])
        if qual and ('base', qual) in keys:
            keys[('qual', qual)].update(keys[('base', qual)])
    for (kind, k), cids in keys.items():
        cids = sorted(cids)
        if len(cids) > 30:  # 太常见的名字 (如 "saber" 之类) 只在有共享图片时才考虑
            continue
        for i, a in enumerate(cids):
            for b in cids[i + 1:]:
                cand[(a, b)]['reasons'].add(kind)
    # 2) 共享 zerochan 图片
    by_zid = img.drop_duplicates(['cid', 'zid']).groupby('zid')['cid'].apply(list)
    for cids in by_zid[by_zid.map(len) > 1]:
        cids = sorted(set(cids))
        for i, a in enumerate(cids):
            for b in cids[i + 1:]:
                cand[(a, b)]['reasons'].add('shared_image')
                cand[(a, b)]['shared'] += 1
    out = [dict(a=a, b=b, reasons=sorted(v['reasons']), shared=v['shared']) for (a, b), v in sorted(cand.items())]
    with open(OUT / 'candidates.json', 'w') as f:
        json.dump(out, f)
    print(len(classes), 'classes,', len(img), 'images,', len(out), 'candidate pairs')
    print(collections.Counter(tuple(c['reasons']) for c in out).most_common(12))


if __name__ == '__main__':
    main()
