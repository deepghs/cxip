import io
import tarfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from v3.data.resize_parts import _tar_size, iter_tar_members, process_tar, resize_image, run_jobs


def _webp(w, h, mode='RGB'):
    b = io.BytesIO(); Image.new(mode, (w, h), (10, 20, 30, 128)[:len(mode)]).save(b, 'WEBP'); return b.getvalue()


def _tar(path, items, fmt=tarfile.PAX_FORMAT):
    with tarfile.open(path, 'w', format=fmt) as t:
        for name, data in items:
            ti = tarfile.TarInfo(name); ti.size = len(data); t.addfile(ti, io.BytesIO(data))


def test_resize_rules():
    b, w, h = resize_image(_webp(1000, 2000)); assert (w, h) == (640, 1280)
    b, w, h = resize_image(_webp(300, 200)); assert (w, h) == (300, 200)
    b, w, h = resize_image(_webp(800, 800, 'RGBA')); assert Image.open(io.BytesIO(b)).mode == 'RGBA'


def test_resize_modes():
    p = Image.new('P', (700, 900)); bio = io.BytesIO(); p.save(bio, 'PNG')
    b, w, h = resize_image(bio.getvalue())
    im = Image.open(io.BytesIO(b))
    assert im.format == 'WEBP' and im.mode == 'RGB' and (w, h) == (640, 823) == im.size
    la = Image.new('LA', (100, 50), (5, 100)); bio = io.BytesIO(); la.save(bio, 'PNG')
    b, w, h = resize_image(bio.getvalue())
    assert Image.open(io.BytesIO(b)).mode == 'RGBA' and (w, h) == (100, 50)
    cm = Image.new('CMYK', (64, 64)); bio = io.BytesIO(); cm.save(bio, 'JPEG')
    assert Image.open(io.BytesIO(resize_image(bio.getvalue())[0])).mode == 'RGB'


def test_iter_tar_members_offsets(tmp_path):
    a, c = _webp(10, 10), _webp(20, 20)
    long = 'x' * 150 + '.webp'
    _tar(tmp_path / 's.tar', [('1.webp', a), (long, c)])
    raw = (tmp_path / 's.tar').read_bytes()
    got = {n: raw[o:o + s] for n, o, s in iter_tar_members(tmp_path / 's.tar')}
    assert got['1.webp'] == a and got[long] == c


@pytest.mark.parametrize('fmt', [tarfile.GNU_FORMAT, tarfile.USTAR_FORMAT])
def test_iter_tar_members_gnu_and_ustar_prefix(tmp_path, fmt):
    a, c = _webp(10, 10), _webp(20, 20)
    long = ('d' * 60 + '/') * 2 + 'y' * 80 + '.webp' if fmt == tarfile.USTAR_FORMAT else 'z' * 300 + '.webp'
    _tar(tmp_path / 's.tar', [(long, c), ('2.webp', a)], fmt=fmt)
    raw = (tmp_path / 's.tar').read_bytes()
    got = {n: raw[o:o + s] for n, o, s in iter_tar_members(tmp_path / 's.tar')}
    assert set(got) == {long, '2.webp'} and got[long] == c and got['2.webp'] == a


def test_tar_size_base256():
    f = bytes([0x80]) + (9 * 2**30).to_bytes(11, 'big')
    assert _tar_size(f) == 9 * 2**30
    assert _tar_size(b'00000001750\0') == 1000


def test_process_tar_roundtrip(tmp_path):
    _tar(tmp_path / 's.tar', [('5.webp', _webp(1280, 960)), ('6.webp', _webp(64, 64))])
    df = process_tar(tmp_path / 's.tar', {'5.webp': '5'}, tmp_path / 'out' / 'p.tar')
    assert df.key.tolist() == ['5'] and df.error.isna().all()
    raw = (tmp_path / 'out' / 'p.tar').read_bytes()
    r = df.iloc[0]
    assert Image.open(io.BytesIO(raw[r.offset:r.offset + r['size']])).size == (853, 640)


def test_process_tar_errors_skip_and_valid_tar(tmp_path):
    _tar(tmp_path / 's.tar', [('5.webp', _webp(1280, 960)), ('7.webp', b'not an image'), ('6.webp', _webp(64, 64))])
    out = tmp_path / 'out' / 'p.tar'
    df = process_tar(tmp_path / 's.tar', {'5.webp': 'a', '7.webp': 'b', '6.webp': 'c', '9.webp': 'd'}, out)
    assert df.key.tolist() == ['a', 'b', 'c', 'd']
    assert df.error.isna().tolist() == [True, False, True, False]
    assert 'missing' in df.error.iloc[3]
    assert (df.width.iloc[2], df.height.iloc[2]) == (64, 64)
    with tarfile.open(out) as t:                      # the part is a normal tar readable by tarfile
        assert t.getnames() == ['a.webp', 'c.webp']
    assert out.with_suffix('.parquet').exists() and not list(out.parent.glob('*.tmp'))
    # finished parts are never redone or overwritten
    before = out.stat().st_mtime_ns
    df2 = process_tar(tmp_path / 's.tar', {'5.webp': 'zzz'}, out)
    assert df2.key.tolist() == df.key.tolist() and out.stat().st_mtime_ns == before


def test_run_jobs_merges_parts(tmp_path):
    src = tmp_path / 'src'
    src.mkdir()
    _tar(src / 'a.tar', [('1.webp', _webp(900, 700)), ('2.webp', _webp(50, 60))])
    _tar(src / 'b.tar', [('3.webp', _webp(30, 30))])
    jobs = pd.DataFrame({'src_tar': [str(src / 'a.tar'), str(src / 'b.tar'), str(src / 'a.tar')],
                         'member': ['1.webp', '3.webp', '2.webp'], 'key': ['1', '3', '2']})
    parts = run_jobs(jobs, tmp_path / 'resized', workers=2)
    assert sorted(parts.key) == ['1', '2', '3'] and parts.error.isna().all()
    assert parts.part.nunique() == 2
    r = parts.set_index('key').loc['1']
    raw = (tmp_path / 'resized' / r.part).read_bytes()
    assert Image.open(io.BytesIO(raw[r.offset:r.offset + r['size']])).size == (823, 640)
    again = run_jobs(jobs, tmp_path / 'resized', workers=2)      # resume: everything skipped, same table
    pd.testing.assert_frame_equal(parts.sort_values('key').reset_index(drop=True),
                                  again.sort_values('key').reset_index(drop=True))


# ------------------------------------------------------------------ danbooru-artist-v2 build / index helpers

def _posts():
    return pd.DataFrame({'id': [10, 11, 12, 20, 21, 30], 'artist': ['b', 'b', 'b', 'a', 'a', 'c'],
                         'ds': [0, 0, 1, 0, 0, 1], 'shard': [3, 3, 0, 4, 4, 1]})


def test_make_jobs_paths():
    from v3.data.build_danbooru_artist_v2 import make_jobs
    j = make_jobs(_posts())
    assert j.columns.tolist()[:3] == ['src_tar', 'member', 'key']
    r = j.set_index('key').loc['12']
    assert r.member == '12.webp' and r.src_tar.endswith('danbooru-newest-4mpixel-v1/wds/train/train-000000.tar')
    assert j.set_index('key').loc['20'].src_tar.endswith('danbooru-full-4mpixel-v1/wds/train/train-000004.tar')


def test_make_plan_labels_order_and_errors():
    from v3.data.build_danbooru_artist_v2 import make_plan
    posts = _posts()
    parts = pd.DataFrame({'key': ['10', '11', '12', '20', '21', '30'], 'part': ['p'] * 6, 'offset': range(6),
                          'size': [1] * 6, 'width': [5] * 6, 'height': [5] * 6,
                          'error': [None, 'boom', None, None, None, None]})
    plan, errors = make_plan(posts, parts, seed=0)
    assert errors.id.tolist() == [11]
    assert dict(zip(plan.artist, plan.label)) == {'a': 0, 'b': 1, 'c': 2}
    # artists contiguous, ids ascending within artist
    runs = plan.artist.ne(plan.artist.shift()).sum()
    assert runs == plan.artist.nunique()
    for _, g in plan.groupby('artist'):
        assert g.id.is_monotonic_increasing
    assert make_plan(posts, parts, seed=0)[0].id.tolist() == plan.id.tolist()
    # a post with no part row at all is an error, never silently dropped
    with pytest.raises(AssertionError):
        make_plan(posts, parts[parts.key != '30'], seed=0)


def test_train_split_excludes_eval_artists():
    from v3.data.build_index import danbooru_v2_train
    df = pd.DataFrame({'artist': ['a', 'b', 'c', 'd', 'e'], 'n_artist_local': [40, 40, 15, 40, 40],
                       'eval_0918_test': [False, False, False, False, True], 'eval_public': [False] * 5,
                       'eval_unseen_v1': [False] * 5})
    assert danbooru_v2_train(df, {'b', 'e'}, 30).artist.tolist() == ['a', 'd']
    assert danbooru_v2_train(df, {'b', 'e'}, 10).artist.tolist() == ['a', 'c', 'd']
    # an eval-flagged artist missing from the exclusion list must fail loudly, not leak into training
    with pytest.raises(AssertionError):
        danbooru_v2_train(df, {'b'}, 30)


def test_danbooru_v2_end_to_end(tmp_path, monkeypatch):
    """jobs -> resize -> build -> verify -> training indexes on a tiny synthetic Danbooru."""
    import json
    from v3.data import build_danbooru_artist_v2 as b
    from v3.data.build_index import build_danbooru_v2
    shards, work, out = tmp_path / 'shards', tmp_path / 'work', tmp_path / 'ds'
    work.mkdir()
    monkeypatch.setattr(b, 'SHARDS', shards)
    monkeypatch.setattr(b, 'WORK', work)
    monkeypatch.setattr(b, 'RESIZED', work / 'resized')
    monkeypatch.setattr(b, 'OUT', out)
    rng = np.random.default_rng(0)
    rows, tars = [], {}
    artists = {'p': (0, 40), 'q': (1, 40), 'r': (0, 12), 's': (0, 40)}       # s: evaluation artist (excluded)
    pid = 100
    for a, (ds, n) in artists.items():
        for k in range(n):
            pid += 1
            shard = int(rng.integers(0, 2))
            w, h = int(rng.integers(100, 1500)), int(rng.integers(100, 1500))
            data = b'garbage' if pid == 105 else _webp(w, h)
            tars.setdefault((ds, shard), []).append((f'{pid}.webp', data))
            rows.append({'id': pid, 'artist': a, 'family': pid, 'ds': ds, 'shard': shard,
                         'date': pd.Timestamp('2020-01-01', tz='UTC') + pd.Timedelta(days=k), 'date_source': 'upload',
                         't_year': 2020.0 + k / 365, 'date_day': 18262 + k, 'upload_day': 18262 + k,
                         'characters': ['c1'] if k % 2 else [], 'copyrights': ['original'], 'general': 'g1 g2',
                         'medium': 'color', 'rating': 'g', 'orig_width': w, 'orig_height': h, 'pixiv_id': None,
                         'n_artist_local': n, 'eval_0918_test': a == 's', 'eval_test100': False,
                         'eval_public': False, 'eval_unseen_v1': False, 'in_v1': True, 'in_kaloscope': False})
    tars[(0, 0)].append(('999999.webp', _webp(10, 10)))                     # a member nobody wants
    for (ds, shard), items in tars.items():
        p = shards / b.DS_DIRS[ds] / 'wds/train' / f'train-{shard:06d}.tar'
        p.parent.mkdir(parents=True, exist_ok=True)
        _tar(p, items)
    posts = pd.DataFrame(rows)
    posts['pixiv_id'] = posts.pixiv_id.astype('Int64')
    posts.sort_values(['artist', 'id']).to_parquet(work / 'posts.parquet', index=False)

    b.jobs()
    b.resize(workers=2, limit_tars=0)
    b.build(max_shard_size='60KiB', threads=4)
    m = json.load(open(out / 'manifest.json'))
    assert m['metadata']['num_samples'] == len(posts) - 1 and len(m['shards']) > 1
    assert pd.read_csv(work / 'resize_errors.csv').id.tolist() == [105]
    b.verify()
    with pytest.raises(SystemExit):                                          # finished dataset is never rebuilt
        b.build(max_shard_size='60KiB', threads=4)
    classes = json.load(open(out / 'classes.json'))
    assert [c['name'] for c in classes] == ['p', 'q', 'r', 's']

    splits = tmp_path / 'splits.json'
    splits.write_text(json.dumps({'exclude_danbooru_artists': ['s', 'not-present']}))
    build_danbooru_v2(out, splits, workers=2)
    idx = out / 'index'
    al = pd.read_parquet(idx / 'all.parquet')
    assert len(al) == len(posts) - 1
    r = al.iloc[0]
    assert r.date.startswith('20') and list(r.characters) in ([], ['c1']) and r.width <= 1500
    assert min(r.width, r.height) <= 640
    g30 = pd.read_parquet(idx / 'train_ge30.parquet')
    g10 = pd.read_parquet(idx / 'train_ge10.parquet')
    assert sorted(g30.artist.unique()) == ['p', 'q'] and sorted(g10.artist.unique()) == ['p', 'q', 'r']
    # offsets in the index point at the exact image bytes
    raw = (out / r.shard).read_bytes()
    assert Image.open(io.BytesIO(raw[r.offset:r.offset + r['size']])).size == (r.width, r.height)
    with pytest.raises(SystemExit):
        build_danbooru_v2(out, splits, workers=2)
