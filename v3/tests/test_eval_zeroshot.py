"""zeroshot: 与旧 eval_ccip_zero.compute_metrics_with_random_anchors / pair_eval.zero_shot 的等价性, OOD 协议"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)

from v3.eval import zeroshot as zs

ROOT = Path(__file__).resolve().parents[2]


def synth(n_cls=30, per=(4, 15), dim=24, noise=1.2, seed=0):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(per[0], per[1], n_cls)
    labels = np.repeat(np.arange(n_cls), sizes)
    centers = rng.standard_normal((n_cls, dim))
    f = centers[labels] + noise * rng.standard_normal((len(labels), dim))
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    return f.astype(np.float64), labels


def old_compute(embs_topN, labels_topN, seen, A, logit_scale, logit_bias, ood_thr):
    """scripts/eval/eval_ccip_zero.py compute_metrics_with_random_anchors 的原逻辑 (去掉打印 / 画图)"""
    anchor_indices_list, anchor_class_mapping = [], []
    class_indices = [np.where(labels_topN == cls)[0] for cls in range(seen)]
    for cls, matched_indices in enumerate(class_indices):
        num_anchors = min(A, len(matched_indices))
        random_indices = np.random.choice(matched_indices, size=num_anchors, replace=False)
        anchor_indices_list.extend(random_indices)
        anchor_class_mapping.extend([cls] * num_anchors)
    anchor_indices_array = np.array(anchor_indices_list)
    anchor_class_mapping = np.array(anchor_class_mapping)
    all_anchors = embs_topN[anchor_indices_array]
    similarity_scores = embs_topN @ all_anchors.T
    one_hot_matrix = np.zeros((len(all_anchors), seen))
    one_hot_matrix[np.arange(len(anchor_class_mapping)), anchor_class_mapping] = 1
    pred_scores = (similarity_scores @ one_hot_matrix) / np.maximum(one_hot_matrix.sum(axis=0), 1)[np.newaxis, :]
    pred_scores = pred_scores * logit_scale + logit_bias
    scores = np.max(pred_scores, axis=1)
    y_true_ood = (labels_topN >= seen).astype(int)
    P, R, T = precision_recall_curve(y_true_ood, -scores)
    with np.errstate(invalid='ignore'):
        best_f1 = np.nanmax((2 * P * R / (P + R))[:-1])
    y_pred = np.argmax(pred_scores, axis=1)
    ood = scores < ood_thr
    y_pred[ood] = seen
    y_true = np.clip(labels_topN, None, seen)
    return dict(top1=accuracy_score(y_true, y_pred),
                precision_macro=precision_score(y_true, y_pred, average='macro', zero_division=0),
                recall_macro=recall_score(y_true, y_pred, average='macro', zero_division=0),
                f1_macro=f1_score(y_true, y_pred, average='macro', zero_division=0), ood_f1_best=best_f1,
                anchors=anchor_indices_array, owner=anchor_class_mapping)


@pytest.mark.parametrize('N,seen,A', [(20, 10, 1), (30, 20, 3), (30, 15, 5)])
def test_ood_matches_legacy(N, seen, A):
    f, l = synth(n_cls=40)
    scale, bias, thr = 23.5, -12.25, -1.3
    m = l < N
    e, y = f[m], l[m]
    np.random.seed(42)
    olds = [old_compute(e, y, seen, A, scale, bias, thr) for _ in range(5)]
    # legacy port 与原逻辑一致
    leg = zs.legacy_ccip_zero(f, l, N, A, seen, scale, bias, thr, repeats=5)
    for k in ('precision_macro', 'recall_macro', 'f1_macro', 'ood_f1_best'):
        assert leg[k] == pytest.approx(np.mean([o[k] for o in olds]), abs=1e-12)
    assert leg['top1_accuracy'] == pytest.approx(np.mean([o['top1'] for o in olds]), abs=1e-12)
    # 新实现 (同样锚点, 查询含锚点, 固定 tau) 与原逻辑一致
    ft = torch.as_tensor(e).float()
    for o in olds:
        anchors = [o['anchors'][o['owner'] == c] for c in range(seen)]
        r = zs.ood_eval_once(ft, y, list(range(seen)), list(range(seen, N)), anchors, scale, bias, tau=thr,
                             include_anchors=True)
        assert r['open_tau_top1'] == pytest.approx(o['top1'], abs=1e-9)
        assert r['open_tau_precision_macro'] == pytest.approx(o['precision_macro'], abs=1e-9)
        assert r['open_tau_recall_macro'] == pytest.approx(o['recall_macro'], abs=1e-9)
        assert r['open_tau_f1_macro'] == pytest.approx(o['f1_macro'], abs=1e-9)
        assert r['ood_f1_best'] == pytest.approx(o['ood_f1_best'], abs=1e-6)


def test_macro_prf_matches_sklearn():
    rng = np.random.default_rng(1)
    yt = rng.integers(0, 12, 500)
    yp = np.where(rng.random(500) < 0.6, yt, rng.integers(0, 14, 500))
    m = zs.macro_prf(yt, yp)
    assert m['precision_macro'] == pytest.approx(precision_score(yt, yp, average='macro', zero_division=0))
    assert m['recall_macro'] == pytest.approx(recall_score(yt, yp, average='macro', zero_division=0))
    assert m['f1_macro'] == pytest.approx(f1_score(yt, yp, average='macro', zero_division=0))
    assert m['accuracy'] == pytest.approx(accuracy_score(yt, yp))


def test_refmean_matches_pair_eval():
    src = (ROOT / 'experiments/common/pair_eval.py').read_text()
    start = src.index('def zero_shot')
    end = src.index('\ndef ', start + 10)
    ns = {'np': np, 'torch': torch}
    exec(src[start:end], ns)
    f, l = synth(seed=2)
    for n_ref in (1, 3):
        old = ns['zero_shot'](torch.as_tensor(f).float(), l, n_ref=n_ref, device='cpu')
        new = zs.zero_shot_refmean(f, l, n_ref)
        assert new == old


def test_ood_eval_once_basic():
    f, l = synth(seed=3, noise=0.8)
    ft = torch.as_tensor(f).float()
    known, unknown = list(range(10)), list(range(10, 20))
    rng = np.random.default_rng(0)
    anchors = [rng.choice(np.nonzero(l == c)[0], 3, replace=False) for c in known]
    r = zs.ood_eval_once(ft, l, known, unknown, anchors, 10.0, -5.0, tau=0.0)
    n_query = np.isin(l, known + unknown).sum() - 30
    assert r['n_query'] == n_query
    # AUROC 手算: 分数 = -max 类分数
    q = np.isin(l, known + unknown)
    q[np.concatenate(anchors)] = False
    qi = np.nonzero(q)[0]
    S = np.stack([(f[qi] @ f[a].T).mean(1) for a in anchors], 1) * 10 - 5
    assert r['ood_auroc'] == pytest.approx(roc_auc_score(np.isin(l[qi], unknown), -S.max(1)), abs=1e-6)
    assert r['closed_top1'] == pytest.approx((S.argmax(1) == l[qi])[~np.isin(l[qi], unknown)].mean(), abs=1e-9)


def test_protocol_json_roundtrip(tmp_path):
    f, l = synth(n_cls=50, per=(7, 12), seed=4)
    keys = np.array([f'k{i:05d}' for i in range(len(l))])
    proto = zs.random_protocol(l, 10, 8, A_max=5, repeats=3, seed=0)
    assert proto[0]['known'] == zs.random_protocol(l, 10, 8, A_max=5, repeats=3, seed=0)[0]['known']
    js = dict(anchors=[1, 3, 5], protocols=[dict(N=10, X=8, repeats=[
        dict(known=r['known'], unknown=r['unknown'],
             anchors={str(c): keys[a].tolist() for c, a in zip(r['known'], r['anchors'])}) for r in proto])])
    p = tmp_path / 'p.json'
    p.write_text(json.dumps(js, default=int))
    a = zs.ood_protocol_eval(f, keys, l, p, 10.0, -5.0)
    b = zs.ood_eval(f, l, proto, (1, 3, 5), 10.0, -5.0)
    for A in ('A1', 'A3', 'A5'):
        assert a['N10_X8'][A]['ood_auroc'] == pytest.approx(b[A]['ood_auroc'])
        assert a['N10_X8'][A]['open_best_f1_macro'] == pytest.approx(b[A]['open_best_f1_macro'])


def test_nway_eval_perfect():
    f, l = synth(noise=0.01, seed=5)
    r = zs.nway_eval(f, l, A=1, repeats=2)
    assert r['closed_top1'] == 1.0 and r['closed_f1_macro'] == 1.0
