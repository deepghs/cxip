"""pairwise: 直方图版 vs 精确版 vs sklearn / 旧 pair_eval.pr_metrics; F1 平台; 固定阈值; 跨角色协议"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from v3.eval import pairwise as pw

ROOT = Path(__file__).resolve().parents[2]


def synth(n=600, n_cls=40, dim=32, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_cls, n)
    centers = rng.standard_normal((n_cls, dim))
    feats = centers[labels] * 0.6 + noise * rng.standard_normal((n, dim))
    return feats.astype(np.float32), labels


def old_pr_metrics():
    """experiments/common/pair_eval.pr_metrics 的原文 (该模块顶层 import transformers, 这里只取函数源码)"""
    src = (ROOT / 'experiments/common/pair_eval.py').read_text()
    start = src.index('def pr_metrics')
    end = src.index('\ndef ', start + 10)
    ns = {'np': np}
    if not hasattr(np, 'trapz'):
        np.trapz = np.trapezoid
    exec(src[start:end], ns)
    return ns['pr_metrics']


def test_exact_matches_old_pair_eval():
    f, l = synth()
    s, y = pw.exact_pair_scores(f, l, 20.0, -10.0)
    old = old_pr_metrics()(s, y)
    new = pw.metrics_exact(s, y)
    for k in old:
        assert new[k] == pytest.approx(old[k], abs=1e-12), k


def test_exact_matches_sklearn():
    f, l = synth(seed=1)
    s, y = pw.exact_pair_scores(f, l)
    m = pw.metrics_exact(s, y)
    assert m['ap'] == pytest.approx(average_precision_score(y, s), abs=1e-9)
    assert m['roc_auc'] == pytest.approx(roc_auc_score(y, s), abs=1e-9)
    P, R, T = precision_recall_curve(y, s)
    f1 = 2 * P * R / np.maximum(P + R, 1e-12)
    assert m['best_f1'] == pytest.approx(f1[:-1].max(), abs=1e-9)
    assert m['best_threshold'] == pytest.approx(T[np.argmax(f1[:-1])], abs=1e-6)


@pytest.mark.parametrize('seed,noise', [(0, 1.0), (1, 0.5), (2, 2.0)])
def test_hist_matches_exact(seed, noise):
    f, l = synth(n=900, noise=noise, seed=seed)
    e = pw.pairwise_eval(f, l, 30.0, -15.0, exact=True)
    h = pw.pairwise_eval(f, l, 30.0, -15.0, exact=False, chunk=128)    # 小 chunk: 测分块边界
    assert h['n_pairs'] == e['n_pairs'] == 900 * 899 // 2
    assert h['n_pos_pairs'] == e['n_pos_pairs']
    for k in ['pr_auc', 'ap', 'roc_auc', 'best_f1']:
        assert h[k] == pytest.approx(e[k], abs=2e-4), k
    # 阈值误差 = 一个箱宽 (cos 2/2^20) * scale
    assert h['best_threshold'] == pytest.approx(e['best_threshold'], abs=30 * 4 / pw.N_BINS + 1e-6)
    assert h['f1_plateau_width'] == pytest.approx(e['f1_plateau_width'], abs=1e-3)


def test_histogram_counts_consistent():
    f, l = synth(n=500, seed=3)
    hist = pw.pair_histogram(f, l, chunk=97)
    assert hist.neg.min() >= 0
    assert hist.pos.sum() + hist.neg.sum() == 500 * 499 // 2
    _, c = np.unique(l, return_counts=True)
    assert hist.pos.sum() == (c * (c - 1) // 2).sum()


def test_f1_plateau():
    thr = np.array([5., 4., 3., 2., 1., 0.])
    F1 = np.array([0.1, 0.5, 0.99, 1.0, 0.985, 0.3])
    p = pw.f1_plateau(F1, thr, rel=0.98)
    assert p['f1_plateau_hi'] == 3.0 and p['f1_plateau_lo'] == 1.0 and p['f1_plateau_width'] == 2.0
    # 不连续的高 F1 区不算
    F1b = np.array([0.99, 0.5, 0.99, 1.0, 0.5, 0.99])
    assert pw.f1_plateau(F1b, thr)['f1_plateau_width'] == 1.0


def test_metrics_at_threshold():
    f, l = synth(n=400, seed=4)
    scale, bias = 10.0, -5.0
    s, y = pw.exact_pair_scores(f, l, scale, bias)
    h = pw.pair_histogram(f, l, scale, bias)
    t = float(np.quantile(s, 0.99))
    m = pw.metrics_at_threshold(h, t)
    pred = s >= t
    P = (pred & y).sum() / pred.sum()
    R = (pred & y).sum() / y.sum()
    assert m['precision'] == pytest.approx(P, abs=2e-3)
    assert m['recall'] == pytest.approx(R, abs=2e-3)


def test_threshold_drift():
    f1, l1 = synth(n=300, seed=5)
    f2, l2 = synth(n=300, seed=6, noise=1.5)
    r = {'a': pw.pairwise_eval(f1, l1, 10, -5), 'b': pw.pairwise_eval(f2, l2, 10, -5)}
    d = pw.threshold_drift(r)
    assert d['spread'] == pytest.approx(abs(r['a']['best_threshold'] - r['b']['best_threshold']))
    assert 0 <= d['cross_f1_loss']['a->b'] <= 1 and 0 <= d['max_cross_f1_loss'] <= 1


def test_cross_character_bruteforce():
    rng = np.random.default_rng(7)
    n = 120
    style = rng.integers(0, 6, n)
    char = rng.integers(-1, 8, n)
    f = rng.standard_normal((n, 16)).astype(np.float32)
    r = pw.csip_cross_character(f, style, char)
    i, j = np.triu_indices(n, 1)
    pos = style[i] == style[j]
    hard = (char[i] == char[j]) & (char[i] >= 0) & (style[i] != style[j])
    cross = pos & (char[i] >= 0) & (char[j] >= 0) & (char[i] != char[j])
    assert r['n_pos_pairs'] == pos.sum() and r['n_hardneg_pairs'] == hard.sum() and r['n_pos_crosschar'] == cross.sum()
    fn = f / np.linalg.norm(f, axis=1, keepdims=True)
    s = (fn[i] * fn[j]).sum(1)
    yy = np.r_[np.ones(pos.sum()), np.zeros(hard.sum())]
    assert r['roc_auc_hardneg'] == pytest.approx(roc_auc_score(yy, np.r_[s[pos], s[hard]]), abs=1e-6)


def test_balanced_pr_auc():
    rng = np.random.default_rng(8)
    ps, ns = rng.normal(1, 1, 300), rng.normal(0, 1, 300)
    m = pw.metrics_exact(np.r_[ps, ns], np.r_[np.ones(300), np.zeros(300)])
    assert pw.balanced_pr_auc(ps, ns) == pytest.approx(m['pr_auc'], abs=1e-12)
    # 负样本复制 10 份 (同分布) -> 加权后基本不变, 不加权的 PR-AUC 明显下降
    ns10 = np.tile(ns, 10)
    assert pw.balanced_pr_auc(ps, ns10) == pytest.approx(m['pr_auc'], abs=1e-9)
