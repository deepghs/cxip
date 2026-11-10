import pandas as pd
from v3.data.build_pixiv_artist import _md, check_train_disjoint, make_jobs, make_plan, select_works

def test_select_works_rules():
    works = pd.DataFrame(dict(image_id=[1, 2, 3, 4, 5], artwork_id=[10, 10, 11, 12, 13], user_id=[7, 7, 7, 8, 9],
                              on_danbooru=[False, False, True, False, False]))
    accounts = pd.DataFrame(dict(user_id=[7, 8, 9], artist=['x', None, None], pixiv_only=[False, True, False],
                                 n_works=[2, 30, 40]))
    s = select_works(works, accounts, exclude_users={8})
    assert s.image_id.tolist() == [1, 2, 4]                  # on_danbooru dropped; user 9 neither mapped nor pixiv_only
    assert s.set_index('image_id').eval_excluded.to_dict() == {1: False, 2: False, 4: True}
    assert s.set_index('image_id').artist.to_dict() == {1: 'x', 2: 'x', 4: 'pixiv:8'}


def _sel():
    return pd.DataFrame(dict(image_id=[5, 3, 4, 9], artwork_id=[50, 30, 40, 90], page=[0, 0, 1, 0], user_id=[1, 2, 2, 3],
                             artist=['b', 'a', 'a', 'pixiv:3'], eval_excluded=[False, False, False, True],
                             t_year=[2020.5, 2019.0, float('nan'), 2021.0], date_day=pd.array([18000, 17900, None, 18500], 'Int64'),
                             date_source='pixiv', medium='color', x_restrict=0,
                             tar=['/r/images/0001.tar', '/r/images/0001.tar', '/r/images/0002.tar', '/r/images/0002.tar'],
                             offset=1, size=1))


def _parts(keys, err=()):
    return pd.DataFrame(dict(key=[str(k) for k in keys], part='p.tar', offset=[512 * i for i in range(len(keys))], size=10,
                             width=64, height=64, error=[('boom' if k in err else None) for k in keys]))


def test_make_jobs_members_and_keys():
    j = make_jobs(_sel())
    assert j.src_tar.tolist() == ['/r/images/0001.tar'] * 2 + ['/r/images/0002.tar'] * 2
    assert j.member.tolist() == ['3.webp', '5.webp', '4.webp', '9.webp']
    assert j.key.tolist() == ['3', '5', '4', '9']


def test_make_plan_labels_order_and_errors():
    plan, errors = make_plan(_sel(), _parts([3, 4, 5, 9], err={9}))
    # labels follow the sorted class names; ids ascending within a class; failed resize dropped into the error table
    assert plan.artist.map({'a': 0, 'b': 1, 'pixiv:3': 2}).tolist() == plan.label.tolist()
    assert sorted(plan.image_id) == [3, 4, 5]
    assert errors.image_id.tolist() == [9]
    assert plan.groupby('label').image_id.apply(lambda s: s.is_monotonic_increasing).all()
    assert plan.dtypes['offset'] == 'int64'


def test_md_handles_missing_date():
    plan, _ = make_plan(_sel(), _parts([3, 4, 5, 9]))
    md = {int(r['image_id']): _md(r) for r in plan.to_dict('records')}
    assert md[4]['date'] is None and md[4]['date_day'] is None and md[4]['t_year'] is None
    assert md[3]['date'].startswith('2019-') and md[3]['date_day'] == 17900
    assert md[9]['eval_excluded'] is True and md[9]['family'] == 90


def test_check_train_disjoint():
    import pytest
    check_train_disjoint(pd.DataFrame(dict(user_id=[1, 2])), {3})
    with pytest.raises(AssertionError):
        check_train_disjoint(pd.DataFrame(dict(user_id=[1, 2])), {2})
