"""评测集注册表与读取 (只读索引, 不复制图片).

- 开放集 CCIP: load_ccip_openset('full' | 'fast') -> (DataFrame, read_image); 图片从打包 tar (images/full-*.tar,
  按偏移 pread) 读, 打包缺失时回退到 ccip_merged_nd 的原始 shard + 偏移.
- 旧评测集 (与 HF 模型可比): LEGACY_SETS, 规格同 experiments/common/pair_eval.load_items
  (folder:<dir> 或 json:<label_json>:<img_root>); load_legacy(name) -> (DataFrame[path, label], read_image).
- CSIP 跨角色: danbooru_char_labels(post_ids) 从 isek-ai/danbooru-tags-2024 读角色标签 (只认单角色帖);
  build_csip_crosschar() 为 test100 系列写 index parquet (画师 + 角色).
"""
from __future__ import annotations

import glob
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPRO = Path('/data/neko/cxip_repro')
MERGED = REPRO / 'data/ccip_merged_nd'
EVAL_ROOT = Path('/data/neko/cxip_v3/eval')
OPENSET = EVAL_ROOT / 'ccip_openset_v1'
CROSSCHAR = EVAL_ROOT / 'csip_crosschar_v1'
# index.parquet 形式的评测集 (path, label_name, label, sample_key); csip_unseen_v1 见 v3/scripts/build_csip_unseen.py
INDEX_SETS = {'csip_unseen_v1': Path('/data/neko/cxip_v3/data/csip_unseen_v1')}
ISEK_GLOB = '/data/.hf/hub/datasets--isek-ai--danbooru-tags-2024/snapshots/*/data/*.parquet'
_A = 'data/danbooru-w640-ws-full-artist'
LEGACY_SETS = {
    # CSIP (画师)
    'test100_0918': f'json:data/danbooru-artist-split-0918/test_split_100_avail.json:{_A}/train',
    'test100': f'json:{_A}/test_split_100.json:{_A}/train',
    'test100v2': f'json:data/danbooru-artist-split-0918/test_split_100.json:{_A}/train',
    'csip_eval_public': 'folder:data/csip_eval',
    # CCIP (角色, 闭集)
    'zc50x15': 'folder:data/zerochan_test_c50x15',
    'zc500x10': 'folder:data/zerochan_test_c500x10',
    'ccip_eval': 'folder:data/ccip/ccip_eval/eval',
    'test_ccip': 'folder:data/ccip/test_ccip',
    'ccip_test_v1': 'folder:data/ccip_test_v1',
}
CCIP_LEGACY = ['zc50x15', 'zc500x10', 'test_ccip', 'ccip_eval', 'ccip_test_v1']
CSIP_LEGACY = ['test100_0918', 'csip_eval_public', 'test100', 'test100v2']
IMG_EXT = ('.jpg', '.jpeg', '.png', '.webp')


def decode_image(data: bytes):
    """同 pair_eval.ImgDataset: 透明图垫白底, 转 RGB"""
    from PIL import Image
    img = Image.open(io.BytesIO(data))
    if img.mode in ('RGBA', 'LA', 'P'):
        img = img.convert('RGBA')
        bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert('RGB')


def _pread(path, offset, size):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        return os.pread(fd, int(size), int(offset))
    finally:
        os.close(fd)


# ------------------------------------------------------------------------------------------------ open-set CCIP
def load_ccip_openset(which='full', root=OPENSET, prefer_pack=True):
    """which: 'full' | 'fast' | 'all' (eval_index 全部被剔除的图). 返回 (df, read_image(i_or_row, decode=True))"""
    root = Path(root)
    df = pd.read_parquet(root / {'full': 'full.parquet', 'fast': 'fast.parquet', 'all': 'eval_index.parquet'}[which])
    df = df.reset_index(drop=True)

    def read_bytes(row):
        if prefer_pack and isinstance(row.get('pack_tar'), str) and row['pack_tar']:
            p = root / row['pack_tar']
            if p.exists():
                return _pread(p, row['pack_offset'], row['size'])
        return _pread(MERGED / row['shard_path'], row['offset'], row['size'])

    def read_image(i, decode=True):
        row = df.iloc[i] if isinstance(i, (int, np.integer)) else i
        b = read_bytes(row)
        return decode_image(b) if decode else b
    return df, read_image


def ood_protocol_path(root=OPENSET):
    return Path(root) / 'ood_protocol.json'


# ------------------------------------------------------------------------------------------------ legacy sets
def legacy_items(spec, base=REPRO):
    """同 pair_eval.load_items: 返回 [(path, label)], folder 按路径排序, json 按文件名排序"""
    kind, rest = spec.split(':', 1)
    if kind == 'folder':
        root = Path(base) / rest
        return [(str(p), p.parent.name) for p in sorted(root.glob('*/*')) if p.suffix.lower() in IMG_EXT]
    if kind == 'json':
        label_file, img_root = rest.split(':')
        with open(Path(base) / label_file) as f:
            labels = json.load(f)
        return [(os.path.join(str(Path(base) / img_root), k), v) for k, v in sorted(labels.items())]
    raise ValueError(spec)


def load_legacy(name):
    """返回 (df[path, label_name, label], read_image); label = 类名排序后的下标 (同 pair_eval 的 np.unique 编码)"""
    items = legacy_items(LEGACY_SETS[name])
    df = pd.DataFrame(items, columns=['path', 'label_name'])
    df['label'] = np.unique(df['label_name'].values, return_inverse=True)[1]
    df['sample_key'] = df['path'].map(lambda p: os.path.relpath(p, REPRO))

    def read_image(i, decode=True):
        row = df.iloc[i] if isinstance(i, (int, np.integer)) else i
        with open(row['path'], 'rb') as f:
            b = f.read()
        return decode_image(b) if decode else b
    return df, read_image


def load_index_set(name):
    """INDEX_SETS: 返回 (df[path, label_name, label, sample_key, ...], read_image)"""
    df = pd.read_parquet(INDEX_SETS[name] / 'index.parquet').reset_index(drop=True)

    def read_image(i, decode=True):
        row = df.iloc[i] if isinstance(i, (int, np.integer)) else i
        with open(row['path'], 'rb') as f:
            b = f.read()
        return decode_image(b) if decode else b
    return df, read_image


# ------------------------------------------------------------------------------------------------ CSIP cross-character
def danbooru_char_labels(post_ids, solo_only=True) -> pd.DataFrame:
    """post id -> 角色标签 (isek-ai/danbooru-tags-2024). solo_only: 只有恰好 1 个角色标签的帖给出角色, 其余为 None.
    返回 DataFrame[id, character, artist, n_characters]"""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    want = np.unique(np.asarray(post_ids, dtype=np.int64))
    parts = []
    for f in sorted(glob.glob(ISEK_GLOB)):
        t = pq.read_table(f, columns=['id', 'character', 'artist'])
        t = t.filter(pc.is_in(t['id'], value_set=__import__('pyarrow').array(want)))
        if t.num_rows:
            parts.append(t.to_pandas())
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=['id', 'character', 'artist'])
    chars = df['character'].fillna('')
    df['n_characters'] = chars.map(lambda s: len(s.split(', ')) if s else 0)
    if solo_only:
        df['character'] = np.where(df['n_characters'] == 1, chars, None)
    return df


def build_csip_crosschar(names=('test100_0918', 'test100', 'test100v2'), out=CROSSCHAR):
    """为 CSIP 测试集写 <name>.parquet: sample_key, path, artist, post_id, character (单角色帖, 否则空), char_label (-1 未知)"""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name in names:
        df, _ = load_legacy(name)
        df['post_id'] = df['path'].map(lambda p: int(Path(p).stem) if Path(p).stem.isdigit() else -1)
        tags = danbooru_char_labels(df['post_id'].values)
        m = df.merge(tags[['id', 'character', 'n_characters', 'artist']].rename(columns={'id': 'post_id',
                                                                                       'artist': 'db_artist'}),
                     on='post_id', how='left')
        m['character'] = m['character'].where(m['character'].notna(), None)
        codes = {c: i for i, c in enumerate(sorted(m['character'].dropna().unique()))}
        m['char_label'] = m['character'].map(lambda c: codes.get(c, -1) if isinstance(c, str) else -1).astype(int)
        m = m.rename(columns={'label_name': 'artist'})
        m[['sample_key', 'path', 'artist', 'label', 'post_id', 'db_artist', 'n_characters', 'character',
           'char_label']].to_parquet(out / f'{name}.parquet', index=False)
        # 统计: 同角色不同画师的对数 (困难负样本)
        known = m[m['char_label'] >= 0]
        n_hard = 0
        for _, g in known.groupby('char_label'):
            if len(g) < 2:
                continue
            vc = g['label'].value_counts().values
            n_hard += (len(g) * (len(g) - 1) - (vc * (vc - 1)).sum()) // 2
        stats[name] = dict(n_images=len(m), n_artists=int(m['label'].nunique()), n_char_known=int(len(known)),
                           n_characters=int(known['char_label'].nunique()), n_hardneg_pairs=int(n_hard),
                           db_artist_match=float((m['db_artist'].fillna('').str.replace(' ', '_') ==
                                                  m['artist']).mean()))
    with open(out / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=1)
    return stats


def load_crosschar(name, root=CROSSCHAR) -> pd.DataFrame:
    return pd.read_parquet(Path(root) / f'{name}.parquet')
