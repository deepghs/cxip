import io
import tarfile

import numpy as np
import pandas as pd
from PIL import Image

from v3.data import bench2_sets as B
from v3.data.bench2_sets import spread_pick
from v3.eval import evalsets


def test_spread_pick_covers_span():
    t = np.array([2010.0, 2010.1, 2010.2, 2015.0, 2020.0, 2020.1])
    idx = spread_pick(t, 3)
    assert sorted(t[idx]) == [2010.0, 2015.0, 2020.1]
    assert len(spread_pick(t, 10)) == 6


def test_load_index_set_pread(tmp_path, monkeypatch):
    b = io.BytesIO(); Image.new('RGB', (8, 8)).save(b, 'WEBP'); data = b.getvalue()
    with tarfile.open(tmp_path / 'a.tar', 'w') as t:
        ti = tarfile.TarInfo('1.webp'); ti.size = len(data); t.addfile(ti, io.BytesIO(data))
    d = tmp_path / 'set'; d.mkdir()
    pd.DataFrame(dict(path=[str(tmp_path / 'a.tar')], tar=[str(tmp_path / 'a.tar')], offset=[512], size=[len(data)],
                      label_name=['x'], label=[0], sample_key=['d1'], split=['dev'])).to_parquet(d / 'index.parquet')
    monkeypatch.setitem(evalsets.INDEX_SETS, 'tmp_set', d)
    df, read = evalsets.load_index_set('tmp_set')
    assert read(0).size == (8, 8)


def test_sealed_never_in_regular_sets():
    for name in ('bench2_b1', 'bench2_b2', 'bench2_b3p1', 'bench2_b3p2'):
        assert evalsets.INDEX_SETS[name] != evalsets.INDEX_SETS[name + '_sealed']


# ---------------------------------------------------------------------------------------------- synthetic helpers
def test_spread_pick_unsorted_input_and_edges():
    t = np.array([2020.0, 2010.0, 2015.0, 2012.0])
    idx = spread_pick(t, 2)
    assert sorted(t[idx]) == [2010.0, 2020.0]
    assert len(spread_pick(t, 0)) == 0 and len(spread_pick(np.array([]), 3)) == 0
    assert spread_pick(np.array([5.0]), 3).tolist() == [0]


def test_pick_per_artist_cap_keeps_span():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(dict(label_name=['a'] * 100 + ['b'] * 5,
                           t_year=np.r_[rng.uniform(2005, 2020, 100), rng.uniform(2010, 2011, 5)],
                           sample_key=[f'd{i}' for i in range(105)]))
    out = B.pick_per_artist(df, 'label_name', cap=10, seed=0)
    assert out.groupby('label_name').size().to_dict() == {'a': 10, 'b': 5}
    a = df[df.label_name == 'a']
    sel = out[out.label_name == 'a']
    assert sel.t_year.min() == a.t_year.min() and sel.t_year.max() == a.t_year.max()
    assert out.equals(B.pick_per_artist(df, 'label_name', cap=10, seed=0))


def test_first_page_per_work():
    df = pd.DataFrame(dict(artwork_id=[1, 1, 1, 2, 3, 3], page=[3, 1, 2, 1, 2, 5], sample_key=list('abcdef')))
    out = B.first_page_per_work(df)
    assert sorted(out.sample_key) == ['b', 'd', 'e']


def test_std_pixiv_rating_family_and_empty_tags(tmp_path):
    p = pd.DataFrame(dict(sample_key=['p10', 'p11'], shard=['wds/all/all-000000.tar'] * 2, offset=[512, 2048],
                          size=[10, 20], artwork_id=[100, 101], user_id=[7, 7], page=[1, 1], t_year=[2020.5, 2021.0],
                          date_day=[18500, 18700], date_source=['pixiv', 'pixiv'], medium=['color', 'mono'],
                          x_restrict=[0, 2], family=[100, 101]))
    s = B.std_pixiv(p, tmp_path, label_names=pd.Series(['pixiv:7', 'pixiv:7']))
    assert s.rating.tolist() == ['g', 'e']
    assert (s.family >= B.PIXIV_FAMILY_OFFSET).all()
    assert s.source.unique().tolist() == ['pixiv'] and s.id.tolist() == [10, 11]
    assert s.path.tolist() == [str(tmp_path / 'wds/all/all-000000.tar')] * 2 and (s.path == s.tar).all()
    assert all(len(x) == 0 for x in s.characters) and all(len(x) == 0 for x in s.copyrights)


def test_std_danbooru_keys():
    d = pd.DataFrame(dict(sample_key=['42'], shard=['wds/all/all-000001.tar'], offset=[512], size=[9], danbooru_id=[42],
                          artist=['a'], t_year=[2019.0], date_day=[17900], date_source=['twitter'], family=[40],
                          medium=['color'], rating=['s'], characters=[np.array(['c1'])],
                          copyrights=[np.array(['x'])]))
    s = B.std_danbooru(d, '/root')
    assert s.sample_key.tolist() == ['d42'] and s.id.tolist() == [42] and s.source.tolist() == ['danbooru']
    assert s.tar.tolist() == ['/root/wds/all/all-000001.tar'] and s.family.tolist() == [40]
    assert s.characters.iloc[0] == ['c1'] and s.label_name.tolist() == ['a']


def _std(names, keys):
    return pd.DataFrame(dict(label_name=names, sample_key=keys, t_year=np.arange(len(names), dtype=float)))


def test_split_and_label_sealed_separate():
    df = _std(['b', 'a', 'c', 'a', 'd'], ['k1', 'k2', 'k3', 'k4', 'k5'])
    reg, sealed = B.split_and_label(df, {'a': 'dev', 'b': 'test', 'c': 'sealed', 'd': 'sealed'})
    assert set(reg.split) == {'dev', 'test'} and set(sealed.split) == {'sealed'}
    assert dict(zip(reg.label_name, reg.label)) == {'a': 0, 'b': 1}
    assert dict(zip(sealed.label_name, sealed.label)) == {'c': 0, 'd': 1}
    # B3-P1 style: labels come from the matching B1 artist
    p1 = _std(['b', 'c'], ['p1', 'p2'])
    r1, s1 = B.split_and_label(p1, {'a': 'dev', 'b': 'test', 'c': 'sealed'},
                               label_of=({'a': 0, 'b': 1}, {'c': 0, 'd': 1}))
    assert r1.label.tolist() == [1] and r1.split.tolist() == ['test'] and s1.label.tolist() == [0]


def test_write_set_roundtrip_list_columns(tmp_path):
    df = pd.DataFrame(dict(path=['/x.tar'], tar=['/x.tar'], offset=[512], size=[3], sample_key=['p1'], label=[0],
                           label_name=['pixiv:1'], id=[1], source=['pixiv'], split=['dev'],
                           family=[B.PIXIV_FAMILY_OFFSET], t_year=[2020.0], date_day=[18000], date_source=['pixiv'], characters=[[]], copyrights=[[]],
                           medium=['color'], rating=['g'], user_id=[1]))
    B.write_set(df, tmp_path / 's', {'rules': []})
    back = pd.read_parquet(tmp_path / 's' / 'index.parquet')
    assert list(back.characters.iloc[0]) == [] and (tmp_path / 's' / 'manifest.json').exists()
    assert list(back.columns[:len(B.COLUMNS)]) == B.COLUMNS
