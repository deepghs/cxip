"""bias: 与 experiments/common/bias_eval.run_content / run_char (CPU) 逐位一致; I 的方向性"""
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from v3.eval import bias as B

B._DEVICE = "cpu"  # 与旧实现逐位相等只在 CPU 上成立; 实际评测默认用 GPU (仅浮点舍入差异)

ROOT = Path(__file__).resolve().parents[2]


def old_bias_eval():
    sys.path.insert(0, str(ROOT / 'experiments/common'))
    sys.path.insert(0, str(ROOT))
    try:
        import bias_eval
    except Exception as e:  # transformers / timm 缺失时跳过
        pytest.skip(f'cannot import old bias_eval: {e}')
    return bias_eval


def data(n=400, d=32, t=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    w = torch.randn(d, t, generator=g)
    y = torch.sigmoid(x @ w * 0.3 + torch.randn(n, t, generator=g))
    return x, y


def test_content_bitwise_equal_old():
    be = old_bias_eval()
    x, y = data()
    args = types.SimpleNamespace(seed=42, content_epochs=3, char_epochs=3, repeats=2)
    old = be.run_content(x, y, torch.device('cpu'), args)
    new = B.content_bias(x, y, seed=42, epochs=3, repeats=2)
    assert new['real'] == old['real'] and new['base'] == old['base'] and new['I'] == old['I']


def test_char_bitwise_equal_old():
    be = old_bias_eval()
    x, _ = data(seed=1)
    tgt = x @ torch.randn(32, 24) + torch.randn(len(x), 24)
    labels = [f'c{i % 25}' for i in range(len(x))]
    args = types.SimpleNamespace(seed=42, content_epochs=3, char_epochs=3, repeats=2)
    old = be.run_char(x, tgt, labels, torch.device('cpu'), args)
    new = B.char_bias(x, tgt, labels, seed=42, epochs=3, repeats=2)
    assert new['real'] == old['real'] and new['base'] == old['base'] and new['I'] == old['I']


def test_regression_direction():
    g = torch.Generator().manual_seed(3)
    x = torch.randn(1500, 32, generator=g)
    groups = [i % 30 for i in range(1500)]
    dep = x @ torch.randn(32, 16, generator=g)                     # 目标完全由输入决定
    ind = torch.randn(1500, 16, generator=g)                       # 目标与输入独立
    r_dep = B.regression_bias(x, dep, groups, epochs=30, repeats=1, lr=3e-3)
    r_ind = B.regression_bias(x, ind, groups, epochs=30, repeats=1, lr=3e-3)
    assert r_dep['I'] > 0.5
    assert abs(r_ind['I']) < 0.05


def test_style_ce_direction():
    g = torch.Generator().manual_seed(4)
    labels = np.repeat(np.arange(20), 40)
    centers = torch.randn(20, 16, generator=g)
    x = centers[labels] + 0.3 * torch.randn(len(labels), 16, generator=g)
    r = B.style_bias(x, None, labels, form='ce', epochs=40, repeats=1, lr=3e-3)
    assert r['I'] > 0.5 and r['form'] == 'ce' and r['n_classes'] == 20
    r0 = B.style_bias(torch.randn(len(labels), 16, generator=g), None, labels, form='ce', epochs=40, repeats=1,
                      lr=3e-3)
    assert r0['I'] < 0.1


def test_stratified_split_matches_old_rule():
    labels = ['a'] * 10 + ['b'] * 3 + ['c'] * 1
    tr, va = B.stratified_split(labels, 0.2, seed=0)
    assert sorted(tr + va) == list(range(14))
    va_l = [labels[i] for i in va]
    assert va_l.count('a') == 2 and va_l.count('b') == 1 and va_l.count('c') == 1
