import numpy as np

from v3.data.bench2_splits import assign_splits, span_bin


def test_assign_splits_sizes_forced_and_deterministic():
    keys = np.array([f'a{i}' for i in range(100)]); strata = np.arange(100) % 3
    forced = {'test': {'a0', 'a1', 'a2'}}
    s = assign_splits(keys, strata, dict(dev=40, test=30, sealed=30), seed=0, forced=forced)
    assert sorted(set(s.values())) == ['dev', 'sealed', 'test']
    assert [list(s.values()).count(k) for k in ('dev', 'test', 'sealed')] == [40, 30, 30]
    assert all(s[k] == 'test' for k in forced['test'])
    assert s == assign_splits(keys, strata, dict(dev=40, test=30, sealed=30), seed=0, forced=forced)
    for z in range(3):  # stratified: each stratum ~40/30/30
        sub = [s[k] for k in keys[strata == z]]
        assert abs(sub.count('dev') / len(sub) - 0.4) < 0.1


def test_span_bin():
    assert span_bin(np.array([0, 2.9, 3, 8, 20]), (3, 8)).tolist() == [0, 0, 1, 2, 2]


def test_assign_splits_uneven_strata_and_seed_changes_result():
    keys = np.array([f'k{i}' for i in range(61)]); strata = np.array([0] * 10 + [1] * 30 + [2] * 21)
    sizes = dict(a=30, b=20, c=11)
    s = assign_splits(keys, strata, sizes, seed=0)
    assert [list(s.values()).count(k) for k in 'abc'] == [30, 20, 11]
    assert s != assign_splits(keys, strata, sizes, seed=1)


def test_assign_splits_rejects_bad_sizes_and_overfull_forced():
    keys = np.array(['x', 'y', 'z']); strata = np.zeros(3, int)
    for bad in (dict(a=1, b=1), dict(a=2, b=2)):
        try:
            assign_splits(keys, strata, bad, seed=0)
        except ValueError:
            pass
        else:
            raise AssertionError('sizes != len(keys) must raise')
    try:
        assign_splits(keys, strata, dict(a=1, b=2), seed=0, forced={'a': {'x', 'y'}})
    except ValueError:
        pass
    else:
        raise AssertionError('forced larger than the split size must raise')


def test_stratified_draw_tops_up_from_neighbours():
    from v3.data.bench2_splits import stratified_draw
    strata = np.array([0] * 500 + [1] * 120 + [2] * 400)
    idx = stratified_draw(strata, 200, 600, seed=0)
    assert len(idx) == 600 and len(set(idx.tolist())) == 600
    got = [int((strata[idx] == z).sum()) for z in range(3)]
    assert got[1] == 120 and sum(got) == 600 and got[0] >= 200 and got[2] >= 200
    assert idx.tolist() == stratified_draw(strata, 200, 600, seed=0).tolist()
    try:
        stratified_draw(strata[:300], 200, 600, seed=0)
    except ValueError:
        pass
    else:
        raise AssertionError('pool smaller than total must raise')
