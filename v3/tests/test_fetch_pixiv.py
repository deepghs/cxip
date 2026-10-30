import json

import pandas as pd

from v3.data.fetch_pixiv_s1 import image_table, parse_tar_index, verify_sha256


def _write(p, files, **kw):
    p.write_text(json.dumps({'filesize': 10, 'files': files, **kw}))


def test_parse_tar_index(tmp_path):
    p = tmp_path / '0007.json'
    _write(p, {'43000.webp': {'offset': 1536, 'size': 82478, 'sha256': 'ab'},
               'x.txt': {'offset': 0, 'size': 1, 'sha256': 'cd'}})
    df = parse_tar_index(p)
    assert df.to_dict('records') == [dict(image_id=43000, tar=str(tmp_path / '0007.tar'), offset=1536, size=82478, sha256='ab')]
    assert dict(df.dtypes.astype(str)) == dict(image_id='int64', tar='object', offset='int64', size='int64', sha256='object')


def test_parse_tar_index_empty(tmp_path):
    p = tmp_path / '0001.json'
    _write(p, {'x.txt': {'offset': 0, 'size': 1, 'sha256': 'cd'}})
    df = parse_tar_index(p)
    assert len(df) == 0 and dict(df.dtypes.astype(str))['image_id'] == 'int64'


def test_image_table_merges_and_caches(tmp_path):
    (tmp_path / 'images').mkdir()
    _write(tmp_path / 'images/0000.json', {'5.webp': {'offset': 512, 'size': 3, 'sha256': 'a'}})
    _write(tmp_path / 'images/0001.json', {'9.webp': {'offset': 1024, 'size': 4, 'sha256': 'b'}})
    df = image_table(tmp_path)
    assert df.image_id.tolist() == [5, 9]
    assert (tmp_path / 'image_table.parquet').exists()
    # cached: a new json is ignored until the cache is removed
    _write(tmp_path / 'images/0002.json', {'11.webp': {'offset': 0, 'size': 1, 'sha256': 'c'}})
    assert len(image_table(tmp_path)) == 2


def test_verify_sha256(tmp_path):
    import hashlib
    p = tmp_path / 'a.bin'
    p.write_bytes(b'hello')
    assert verify_sha256(p, hashlib.sha256(b'hello').hexdigest())
    assert not verify_sha256(p, '0' * 64)
