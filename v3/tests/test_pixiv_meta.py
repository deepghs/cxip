import numpy as np
import pandas as pd
from v3.data.pixiv_meta import map_accounts, pixiv_medium, shared_table, build_tables


def test_map_accounts_unique_only():
    shared = pd.DataFrame(dict(user_id=[1, 2, 2, 3, 4], artist=['x', 'x', 'y', 'z', 'x'], n_shared=[3, 3, 4, 2, 5]))
    m = map_accounts(shared)
    assert dict(zip(m.user_id, m.artist)) == {1: 'x', 4: 'x'}   # 2 is one-to-many, 3 below threshold


def test_pixiv_medium():
    assert pixiv_medium(['オリジナル', '漫画']) == 'comic'
    assert pixiv_medium(['モノクロ']) == 'mono'
    assert pixiv_medium(['女の子']) == 'color'


def test_pixiv_medium_variants():
    assert pixiv_medium(['4コマ', 'モノクロ']) == 'comic'
    assert pixiv_medium(['白黒']) == 'mono'
    assert pixiv_medium(np.array(['x', '漫画'])) == 'comic'
    assert pixiv_medium(None) == 'color'
    assert pixiv_medium([]) == 'color'


def _posts():
    # filtered danbooru posts: pixiv_id + artist
    return pd.DataFrame(dict(pixiv_id=[10, 11, 12, 13, 20, 20, 30], tag_string_artist=['a', 'a', 'a', 'b', 'b', 'b', '']))


def _artworks():
    return pd.DataFrame(dict(id=[10, 11, 12, 13, 20, 30, 40], user_id=[1, 1, 1, 2, 2, 3, 3]))


def test_shared_table_counts_distinct_artworks():
    s = shared_table(_posts(), _artworks())
    got = {(r.user_id, r.artist): r.n_shared for r in s.itertuples()}
    assert got == {(1, 'a'): 3, (2, 'b'): 2}          # dup post of artwork 20 counted once; empty artist dropped


def _inputs():
    img = pd.DataFrame(dict(image_id=[100, 101, 102, 103, 104, 105, 106, 107], tar=['t'] * 8,
                            offset=range(8), size=[5] * 8, sha256=['h'] * 8))
    images = pd.DataFrame(dict(id=[100, 101, 102, 103, 104, 105, 106, 107], artwork_id=[10, 10, 50, 51, 52, 53, 54, 55],
                               order=[0, 1, 0, 0, 0, 0, 0, 0], is_ugoira=[False, False, False, False, False, False, True, False]))
    ts = lambda y: pd.Timestamp(f'{y}-07-01', tz='UTC').timestamp()
    aw = pd.DataFrame(dict(
        id=[10, 50, 51, 52, 53, 54, 55, 60], user_id=[1, 1, 1, 1, 2, 2, 2, 1],
        type=['illust', 'illust', 'illust', 'manga', 'illust', 'illust', 'illust', 'illust'],
        illust_ai_type=[0, 0, 1, 0, 2, 0, 0, 0], x_restrict=[0, 0, 1, 0, 0, 0, 0, 0],
        created_at=[ts(2010), ts(2012), ts(2014), ts(2015), ts(2016), ts(2017), ts(2018), ts(2019)],
        tags=[['漫画'], ['モノクロ'], ['x'], ['x'], ['x'], ['x'], ['x'], ['x']]))
    return img, images, aw


def test_build_tables_rules():
    img, images, aw = _inputs()
    shared = pd.DataFrame(dict(user_id=[1], artist=['a'], n_shared=[3]))
    works, acc = build_tables(img, images, aw, snap_pixiv_ids={10, 99}, shared=shared)
    # usable: 100,101 (art 10, on danbooru), 102, 103 ; 104 manga, 105 ai==2, 106 ugoira dropped ; 107 ok (user 2)
    assert sorted(works.image_id) == [100, 101, 102, 103, 107]
    w = works.set_index('image_id')
    assert w.loc[100, 'on_danbooru'] and w.loc[101, 'on_danbooru'] and not w.loc[102, 'on_danbooru']
    assert w.loc[100, 'medium'] == 'comic' and w.loc[102, 'medium'] == 'mono' and w.loc[103, 'medium'] == 'color'
    assert w.loc[103, 'x_restrict'] == 1 and w.loc[101, 'page'] == 1
    assert (works.date_source == 'pixiv').all()
    assert abs(w.loc[102, 't_year'] - 2012.5) < 0.01
    assert w.loc[102, 'date_day'] == int(pd.Timestamp('2012-07-01', tz='UTC').timestamp() // 86400)
    a = acc.set_index('user_id')
    assert a.loc[1, 'artist'] == 'a' and a.loc[1, 'n_shared'] == 3
    assert a.loc[1, 'n_works'] == 2 and a.loc[1, 'n_images'] == 2          # art 50, 51 (art 10 on danbooru excluded)
    assert abs(a.loc[1, 't_first'] - 2012.5) < 0.01 and abs(a.loc[1, 't_last'] - 2014.5) < 0.01
    assert abs(a.loc[1, 'span'] - 2.0) < 0.01
    assert not a.loc[1, 'pixiv_only']
    assert a.loc[2, 'artist'] is None and a.loc[2, 'n_works'] == 1 and a.loc[2, 'pixiv_only']


def test_pixiv_only_uses_full_artwork_list():
    img, images, aw = _inputs()
    # artwork 60 (user 1, not in s1) is on danbooru -> user 1 is not pixiv_only even if no s1 work is
    works, acc = build_tables(img, images, aw, snap_pixiv_ids={60}, shared=pd.DataFrame(dict(user_id=[], artist=[], n_shared=[])))
    a = acc.set_index('user_id')
    assert not a.loc[1, 'pixiv_only'] and a.loc[2, 'pixiv_only']
    assert a.loc[1, 'n_works'] == 3      # 10, 50, 51: nothing of s1 is on danbooru


def test_missing_created_at_gives_nan_not_1970():
    img, images, aw = _inputs()
    aw.loc[aw.id == 50, 'created_at'] = np.nan
    works, acc = build_tables(img, images, aw, snap_pixiv_ids=set(), shared=pd.DataFrame(dict(user_id=[], artist=[], n_shared=[])))
    w = works.set_index('image_id')
    assert np.isnan(w.loc[102, 't_year']) and pd.isna(w.loc[102, 'date_day'])
    assert acc.set_index('user_id').loc[1, 't_first'] == w.loc[101, 't_year']      # NaN ignored in min
