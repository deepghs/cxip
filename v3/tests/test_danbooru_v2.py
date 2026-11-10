import pandas as pd
from v3.data import build_danbooru_artist_v2 as B

def _snap():
    return pd.DataFrame(dict(
        id=[1, 2, 3, 4, 5, 6], tag_count_artist=[1, 1, 2, 1, 1, 1], is_deleted=[False, False, False, True, False, False],
        file_ext=['jpg', 'png', 'jpg', 'jpg', 'gif', 'webp'], parent_id=[None, 1, None, None, None, None],
        tag_string=['a', 'b', 'c', 'd', 'e', 'ai-generated x'], tag_string_artist=['p', 'p', 'p q', 'p', 'p', 'p']))

def test_filter_posts_rules():
    assert B.filter_posts(_snap()).id.tolist() == [1, 2]

def test_dedupe_families_keeps_min_id():
    p = pd.DataFrame(dict(id=[2, 1, 7], artist=['p', 'p', 'p'], family=[1, 1, 7]))
    assert sorted(B.dedupe_families(p).id) == [1, 7]

def test_v1_name_map_follows_rename():
    v1 = pd.DataFrame(dict(id=[1, 2, 3, 9], artist=['old_name', 'old_name', 'old_name', 'gone']))
    snap = pd.DataFrame(dict(id=[1, 2, 3], tag_string_artist=['new_name', 'new_name', 'other']))
    assert B.v1_name_map(v1, snap) == {'old_name': 'new_name', 'gone': 'gone'}

def test_medium_of():
    assert B.medium_of({'comic', 'monochrome'}) == 'comic'
    assert B.medium_of({'greyscale'}) == 'mono'
    assert B.medium_of({'1girl'}) == 'color'

def test_filter_posts_family_column():
    f = B.filter_posts(_snap())
    assert f.family.tolist() == [1, 1]
