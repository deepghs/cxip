import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from v3.data.build_index import build_split_index, scan_tar
from v3.data.indexed_tar import IndexedTarDataset, decode_image, list_collate

CCIP_ROOT = Path('/data/neko/cxip_repro/data/ccip_merged_nd')
CCIP_INDEX = Path('/data/neko/cxip_v3/data/ccip-merged-v1-index/index')
DANBOORU_ROOT = Path('/data/neko/cxip_v3/data/danbooru-artist-v1')


def _img_bytes(mode, fmt, color, size=(37, 23)):
    im = Image.new(mode, size, color)
    buf = io.BytesIO()
    im.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture(scope='module')
def tiny_dataset(tmp_path_factory):
    """A small neko-data dataset (2 splits, group_key=label) with mixed image formats/modes."""
    from neko_data.build.job import BuildJob, build_dataset
    from neko_data.contract.records import NormalizedSample

    root = tmp_path_factory.mktemp('nd')
    kinds = [('RGB', 'JPEG', (200, 10, 10), 'jpg'), ('RGBA', 'PNG', (0, 0, 255, 0), 'png'),
             ('RGB', 'WEBP', (10, 200, 10), 'webp'), ('LA', 'PNG', (0, 128), 'png'),
             ('RGBA', 'WEBP', (255, 0, 0, 255), 'webp'), ('P', 'PNG', 3, 'png')]

    def samples(split, n):
        for i in range(n):
            mode, fmt, color, ext = kinds[i % len(kinds)]
            data = _img_bytes(mode, fmt, color, size=(30 + i % 7, 20 + i % 5))
            yield NormalizedSample(sample_key=f'{split}_{i:05d}', image=data, image_extension=ext, split=split,
                                   source_id='t', caption=f'c{i // 5}',
                                   metadata={'label': i // 5, 'name': f'class{i // 5}', 'tags': ['a', str(i)],
                                             'kind': f'{mode}-{fmt}'},
                                   width=30 + i % 7, height=20 + i % 5)

    for split, n in [('train', 120), ('val', 30)]:
        build_dataset(BuildJob(source=samples(split, n), output_dir=root, dataset_id='t', version='v1',
                               split=split, max_shard_size=4096, group_key='label', overwrite=True,
                               skip_invalid_samples=False))
    return root


def test_index_offsets_match_tarfile(tiny_dataset):
    df = build_split_index(tiny_dataset, 'train', list_cols=('tags',), workers=2)
    assert len(df) == 120 and df.sample_key.is_unique
    assert df.shard.nunique() > 1, 'expected several shards with tiny max_shard_size'
    assert list(df.columns[:6]) == ['sample_key', 'shard', 'offset', 'size', 'ext', 'label']
    assert df.tags.iloc[0] == ['a', df.sample_key.iloc[0].split('_')[1].lstrip('0') or '0']
    # labels never split across shards (group_key)
    assert (df.groupby('label').shard.nunique() == 1).all()
    for _, r in df.iterrows():
        with tarfile.open(tiny_dataset / r.shard) as tf:
            ref = tf.extractfile(f'{r.sample_key}.{r.ext}').read()
        with open(tiny_dataset / r.shard, 'rb') as f:
            f.seek(r.offset)
            assert f.read(r['size']) == ref


def test_dataset_decode_and_alpha(tiny_dataset):
    df = build_split_index(tiny_dataset, 'train', workers=2)
    ds = IndexedTarDataset(tiny_dataset, df, columns=['name', 'kind'])
    assert len(ds) == 120
    for i in range(12):
        item = ds[i]
        assert item['image'].mode == 'RGB'
        assert item['label'] == df.label.iloc[i] and item['index'] == i
        assert item['name'] == f"class{item['label']}"
        px = item['image'].getpixel((0, 0))
        if item['kind'] == 'RGBA-PNG':          # fully transparent -> white
            assert px == (255, 255, 255)
        if item["kind"] == "LA-PNG":            # black at alpha 128 over white -> ~127
            assert all(abs(c - 127) <= 2 for c in px)
        if item['kind'] == 'RGBA-WEBP':         # opaque red stays red
            assert px[0] > 200 and px[1] < 50
    raw = IndexedTarDataset(tiny_dataset, df, return_bytes=True)
    assert isinstance(raw[0]['image'], bytes)


def test_dataset_parquet_and_dataloader_workers(tiny_dataset, tmp_path):
    import torch
    df = build_split_index(tiny_dataset, 'train', workers=2)
    p = tmp_path / 'train.parquet'
    df.to_parquet(p)
    ds = IndexedTarDataset(tiny_dataset, p, columns='all',
                           transform=lambda im: torch.from_numpy(np.array(im.resize((8, 8)))).permute(2, 0, 1))
    _ = ds[0]  # open fds in the parent before forking
    dl = torch.utils.data.DataLoader(ds, batch_size=16, num_workers=3, collate_fn=list_collate, shuffle=True,
                                     generator=torch.Generator().manual_seed(0))
    seen = []
    for batch in dl:
        assert batch['image'].shape[1:] == (3, 8, 8)
        assert (batch['label'] == torch.as_tensor(df.label.values[batch['index'].numpy()])).all()
        seen.extend(batch['index'].tolist())
    assert sorted(seen) == list(range(120))
    with pytest.raises(KeyError):
        IndexedTarDataset(tiny_dataset, p, columns=['nope'])


def _check_real(root, index_path, n=200, n_shards=4, seed=0):
    """pread bytes == tarfile extraction for random samples of a few random shards (one header scan
    per shard), plus decoding."""
    df = pd.read_parquet(index_path, columns=['sample_key', 'shard', 'offset', 'size', 'ext', 'label'])
    rng = np.random.default_rng(seed)
    ds = IndexedTarDataset(root, df)
    shards = rng.choice(df.shard.unique(), size=min(n_shards, df.shard.nunique()), replace=False)
    pick = []
    for shard in shards:
        rows = np.nonzero((df.shard == shard).to_numpy())[0]
        ids = rng.choice(rows, size=min(n // len(shards), len(rows)), replace=False)
        pick.extend(ids.tolist())
        with tarfile.open(root / shard) as tf:
            members = {m.name: m for m in tf.getmembers()}
            for i in ids:
                r = df.iloc[i]
                m = members[f'{r.sample_key}.{r.ext}']
                assert (m.offset_data, m.size) == (r.offset, r['size'])
                assert ds.read_bytes(i) == tf.extractfile(m).read()
    for i in rng.choice(len(df), size=20, replace=False):  # decode anywhere
        assert ds[int(i)]['image'].mode == 'RGB'


@pytest.mark.skipif(not (CCIP_INDEX / 'train.parquet').exists(), reason='ccip index not built')
@pytest.mark.parametrize('split', ['train', 'val', 'test'])
def test_real_ccip_index(split):
    _check_real(CCIP_ROOT, CCIP_INDEX / f'{split}.parquet')
    df = pd.read_parquet(CCIP_INDEX / f'{split}.parquet', columns=['label', 'name', 'series', 'shard'])
    assert df.groupby('label').name.nunique().max() == 1
    assert df.groupby('label').series.nunique(dropna=False).max() == 1
    m = json.load(open(CCIP_ROOT / 'manifest.json'))
    assert len(df) == m['metadata']['split_stats'][split]['num_samples']


@pytest.mark.skipif(not (DANBOORU_ROOT / 'index' / 'all.parquet').exists(), reason='danbooru index not built')
def test_real_danbooru_index():
    _check_real(DANBOORU_ROOT, DANBOORU_ROOT / 'index' / 'all.parquet', n=400)
    df = pd.read_parquet(DANBOORU_ROOT / 'index' / 'all.parquet', columns=['label', 'artist', 'shard'])
    assert df.groupby('label').artist.nunique().max() == 1
    assert (df.groupby('label').shard.nunique() == 1).all()


def test_scan_tar_skips_non_images(tmp_path):
    p = tmp_path / 'x.tar'
    with tarfile.open(p, 'w') as tf:
        for name, data in [('a.b.c.jpg', b'123'), ('a.b.c.json', b'{}'), ('k.webp', b'45678')]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    rows = scan_tar(p)
    assert [(r[0], r[1], r[3]) for r in rows] == [('a.b.c', 'jpg', 3), ('k', 'webp', 5)]
    raw = p.read_bytes()
    assert raw[rows[1][2]:rows[1][2] + 5] == b'45678'


def test_decode_palette_transparency():
    im = Image.new('P', (4, 4), 0)
    im.putpalette([0, 0, 0] * 256)
    buf = io.BytesIO()
    im.save(buf, format='PNG', transparency=0)
    out = decode_image(buf.getvalue())
    assert out.mode == 'RGB' and out.getpixel((0, 0)) == (255, 255, 255)
