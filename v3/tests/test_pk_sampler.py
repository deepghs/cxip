import numpy as np
import pytest

from v3.data.pk_sampler import (PKBatchSampler, build_crossed_index, crossed_batches, encode_ids,
                                single_value_ids)


def _labels(n_classes=200, seed=0, min_n=1, max_n=60):
    rng = np.random.default_rng(seed)
    counts = rng.integers(min_n, max_n, size=n_classes)
    labels = np.repeat(np.arange(n_classes), counts)
    return rng.permutation(labels)


def test_global_batch_invariants():
    labels = _labels()
    s = PKBatchSampler(labels, P=32, K=4, seed=1)
    counts = np.bincount(labels)
    for b in range(20):
        gb = s.global_batch(b)
        assert len(gb) == 32
        cls = [int(labels[g[0]]) for g in gb]
        assert len(set(cls)) == 32, 'classes must be distinct'
        for c, g in zip(cls, gb):
            assert len(g) == 4
            assert all(labels[i] == c for i in g)
            if counts[c] >= 4:
                assert len(set(g)) == 4, 'samples distinct when class is big enough'
            else:
                assert len(set(g)) == counts[c], 'small class: all samples used, then repeats'


def test_rank_split_partitions_global_batch():
    labels = _labels()
    P, K, W = 32, 4, 4
    full = PKBatchSampler(labels, P=P, K=K, seed=3)
    ranks = [PKBatchSampler(labels, P=P, K=K, seed=3, rank=r, world_size=W) for r in range(W)]
    for epoch in range(2):
        full.set_epoch(epoch)
        for r in ranks:
            r.set_epoch(epoch)
        its = [iter(r) for r in ranks]
        for b, fb in enumerate(full):
            parts = [next(it) for it in its]
            assert all(len(p) == P * K // W for p in parts)
            assert sum(parts, []) == fb
            # classes whole within a rank
            cls_sets = [set(labels[p].tolist()) for p in parts]
            for i in range(W):
                for j in range(i + 1, W):
                    assert not cls_sets[i] & cls_sets[j]
            if b > 5:
                break


def test_deterministic_and_epoch_dependent():
    labels = _labels()
    a = PKBatchSampler(labels, P=16, K=4, seed=7, num_batches=5)
    b = PKBatchSampler(labels, P=16, K=4, seed=7, num_batches=5)
    assert list(a) == list(b)
    a.set_epoch(1)
    assert list(a) != list(b)
    c = PKBatchSampler(labels, P=16, K=4, seed=8, num_batches=5)
    assert list(c) != list(b)


def test_resume_start_batch():
    labels = _labels()
    a = PKBatchSampler(labels, P=16, K=4, seed=7, num_batches=10)
    a.set_epoch(3)
    full = list(a)
    a.set_epoch(3, start_batch=6)
    assert len(a) == 4
    assert list(a) == full[6:]


def test_alpha_controls_class_frequency():
    # two populations: 10 big classes (500 imgs), 190 small (5 imgs)
    labels = np.concatenate([np.repeat(np.arange(10), 500), np.repeat(np.arange(10, 200), 5)])

    def big_frac(alpha):
        s = PKBatchSampler(labels, P=20, K=2, alpha=alpha, seed=0, num_batches=300)
        cls = np.concatenate([labels[np.asarray(b)] for b in s])
        return np.isin(cls, np.arange(10)).mean()

    f0, f05, f1 = big_frac(0.0), big_frac(0.5), big_frac(1.0)
    assert abs(f0 - 10 / 200) < 0.02          # class-balanced
    assert f0 < f05 < f1
    assert f1 > 0.4                           # instance-balanced favours big classes (capped by P)


def test_group_sampling_creates_same_group_classes():
    labels = _labels(n_classes=400, min_n=4)
    n_classes = labels.max() + 1
    group_of_class = np.arange(n_classes) // 8  # 50 groups of 8 classes
    s_g = PKBatchSampler(labels, P=32, K=2, group_ids=group_of_class, group_frac=0.5, seed=0, num_batches=50)
    s_0 = PKBatchSampler(labels, P=32, K=2, group_ids=None, seed=0, num_batches=50)

    def same_group_pairs(s):
        tot = 0
        for b in range(s.num_batches):
            cls, _ = s.global_batch_classes(b)
            g = group_of_class[cls]
            _, cnt = np.unique(g, return_counts=True)
            tot += (cnt * (cnt - 1) // 2).sum()
        return tot / s.num_batches

    assert same_group_pairs(s_g) > 3 * same_group_pairs(s_0)
    # per-sample group ids give the same class grouping
    s_ps = PKBatchSampler(labels, P=32, K=2, group_ids=group_of_class[labels], group_frac=0.5, seed=0,
                          num_batches=50)
    assert [list(x) for x in [s_ps.global_batch_classes(0)[0]]] == [list(s_g.global_batch_classes(0)[0])]


def test_avoid_same_prefers_distinct_values():
    # each class: 40 samples, 4 distinct "characters" with 10 samples each
    labels = np.repeat(np.arange(50), 40)
    avoid = np.tile(np.repeat(np.arange(4), 10), 50) + 100 * labels
    s = PKBatchSampler(labels, P=16, K=4, avoid_same=avoid, seed=0, num_batches=20)
    for b in range(20):
        for g in s.global_batch(b):
            assert len(set(avoid[g].tolist())) == 4
    # without avoid_same collisions happen
    s2 = PKBatchSampler(labels, P=16, K=4, seed=0, num_batches=20)
    coll = sum(len(set(avoid[g].tolist())) < 4 for b in range(20) for g in s2.global_batch(b))
    assert coll > 0


def test_crossed_groups():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(100), 30)
    chars = rng.integers(0, 40, size=len(labels))
    chars[rng.random(len(labels)) < 0.3] = -1
    idx = build_crossed_index(chars, labels)
    for classes, samples in idx[:20]:
        assert len(set(classes.tolist())) == len(classes) >= 2
        v = chars[samples[0][0]]
        for c, smp in zip(classes, samples):
            assert (labels[smp] == c).all() and (chars[smp] == v).all()
    groups = crossed_batches(chars, labels, n_groups=50, group_size=4, seed=0)
    for g in groups:
        assert len(set(chars[g].tolist())) == 1
        assert len(set(labels[g].tolist())) == len(g) >= 2
    s = PKBatchSampler(labels, P=32, K=4, cross_ids=chars, cross_frac=0.5, cross_size=4, seed=0, num_batches=10)
    for b in range(10):
        cls, fixed = s.global_batch_classes(b)
        assert len(fixed) >= 8
        gb = s.global_batch(b)
        for pos, forced in fixed.items():
            assert gb[pos][0] == forced and labels[forced] == cls[pos]
        # forced samples come in groups sharing the content value
        vals = [chars[f] for f in fixed.values()]
        _, cnt = np.unique(vals, return_counts=True)
        assert (cnt >= 2).all()


def test_errors_and_helpers():
    with pytest.raises(ValueError):
        PKBatchSampler(np.arange(10), P=16, K=2)
    with pytest.raises(ValueError):
        PKBatchSampler(_labels(), P=30, K=2, world_size=4)
    assert encode_ids(['a', None, 'b', 'a', '']).tolist() == [0, -1, 1, 0, -1]
    assert single_value_ids([['x'], [], ['x', 'y'], ['y'], None]).tolist() == [0, -1, -1, 1, -1]


def test_min_count_excludes_small_classes():
    labels = np.concatenate([np.repeat(np.arange(50), 1), np.repeat(np.arange(50, 100), 6)])
    s = PKBatchSampler(labels, P=20, K=4, min_count=2, seed=0, num_batches=30)
    seen = set(np.concatenate([labels[np.asarray(b)] for b in s]).tolist())
    assert seen <= set(range(50, 100))


def test_spread_covers_quantile_strata():
    # each class: 40 samples with years 2000..2019 (2 each); K=4 -> one per 5-year stratum
    labels = np.repeat(np.arange(50), 40)
    year = np.tile(np.repeat(np.arange(2000, 2020), 2), 50)
    s = PKBatchSampler(labels, P=16, K=4, spread=year, seed=0, num_batches=10)
    for b in range(10):
        for g in s.global_batch(b):
            ys = sorted(year[g].tolist())
            assert [(y - 2000) // 5 for y in ys] == [0, 1, 2, 3]
    # avoid_same still preferred inside strata; unknown (-1) spread values are never picked first
    avoid = np.tile(np.arange(40) % 2, 50) + 10 * labels
    year2 = year.copy()
    year2[::7] = -1
    s2 = PKBatchSampler(labels, P=16, K=4, spread=year2, avoid_same=avoid, seed=1, num_batches=10)
    for b in range(10):
        for g in s2.global_batch(b):
            assert (year2[g] >= 0).all() and len(g) == 4
