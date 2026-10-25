"""ccip_openset_v1 产物的不变量 (产物不存在时跳过). 先跑 python -m v3.eval.build_ccip_openset"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

D = Path('/data/neko/cxip_v3/eval/ccip_openset_v1')
pytestmark = pytest.mark.skipif(not (D / 'eval_index.parquet').exists(), reason='open-set outputs not built')


@pytest.fixture(scope='module')
def art():
    return dict(held=json.load(open(D / 'heldout_classes.json')), excl=json.load(open(D / 'train_exclude.json')),
                info=json.load(open(D / 'build_info.json')), ood=json.load(open(D / 'ood_protocol.json')),
                idx=pd.read_parquet(D / 'eval_index.parquet'), full=pd.read_parquet(D / 'full.parquet'),
                fast=pd.read_parquet(D / 'fast.parquet'), meta=pd.read_parquet(D / 'class_meta.parquet'))


def test_heldout_consistent(art):
    labels = {c['label'] for c in art['held']['classes']}
    assert labels == set(art['excl']['heldout_labels'])
    assert set(art['idx']['label']) <= labels
    assert set(art['idx']['sample_key']) == set(art['excl']['exclude_sample_keys'])
    n_eval = sum(c['eval'] for c in art['held']['classes'])
    assert 900 <= n_eval <= 1100


def test_new_franchise_fully_removed(art):
    """整组剔除的 franchise: 训练里剩下的类不能再有这些 franchise"""
    held = set(art['excl']['heldout_labels'])
    fr = set(art['excl']['heldout_franchises'])
    m = art['meta']
    left = m[~m['label'].isin(held)]
    bad = [(r.label, r.name) for r in left.itertuples() if fr & set(r.franchises)]
    assert not bad, bad[:10]


def test_no_overlap_with_old_eval_sets(art):
    held = set(art['excl']['heldout_labels'])
    for name, v in art['info']['eval_sets'].items():
        assert not (held & set(v['labels'])), name


def test_subsets_and_sizes(art):
    full, fast = art['full'], art['fast']
    assert set(fast['sample_key']) <= set(full['sample_key'])
    assert full['eval_class'].all() and full['subset'].isin(['new_franchise', 'hard']).all()
    assert full.groupby('label').size().min() >= 5
    assert fast.groupby('label').size().max() <= 10
    assert fast['label'].nunique() == 300
    frac_new = (fast.groupby('label')['subset'].first() == 'new_franchise').mean()
    assert 0.2 <= frac_new <= 0.4
    for s in ('seen', 'unseen'):
        assert (full.groupby('label')['backbone_seen'].first() == s).sum() >= 100
    assert full['sample_key'].is_unique


def test_ood_protocol(art):
    keys = set(art['full']['sample_key'])
    lab = art['full'].groupby('label').size()
    cfg = {(p['N'], p['X']) for p in art['ood']['protocols']}
    assert cfg == {(20, 20), (40, 20), (60, 40), (500, 500)}
    for p in art['ood']['protocols']:
        assert len(p['repeats']) == 20
        for r in p['repeats']:
            assert len(r['known']) == p['N'] and len(r['unknown']) == p['X']
            assert not set(r['known']) & set(r['unknown'])
            assert set(r['known']) | set(r['unknown']) <= set(lab.index)
            for c in r['known']:
                a = r['anchors'][str(c)]
                assert len(a) == 5 and set(a) <= keys and lab[c] >= 6


def test_packed_bytes_match_source(art):
    from v3.eval.evalsets import MERGED, _pread
    full = art['full'].sample(20, random_state=0)
    for r in full.itertuples():
        a = _pread(D / r.pack_tar, r.pack_offset, r.size)
        b = _pread(MERGED / r.shard_path, r.offset, r.size)
        assert a == b and len(a) == r.size


def test_reader_decodes():
    from v3.eval.evalsets import load_ccip_openset
    df, read = load_ccip_openset('fast')
    img = read(0)
    assert img.mode == 'RGB' and img.size[0] > 0
