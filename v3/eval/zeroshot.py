"""zero-shot N 路 A 锚点分类 与 OOD (N 个已知类 + X 个未知类) 评测. 模型无关: 输入特征 + 标签.

类分数 = 该类 A 个锚点 cos 的均值 * scale + bias (与 scripts/eval/eval_ccip_zero.py 一致; 不是锚点均值向量的 cos).
OOD: max 类分数 < tau 判为未知. 指标:
- closed: 已知类查询图上的 top1 / top5 (不拒识)、宏平均 acc(= 每类召回均值) / P / R / F1;
- ood: 已知 vs 未知的 AUROC (分数 = -max 类分数)、最佳 tau 下的二分类 F1 (best-tau F1, 与旧脚本 find_best_f1_threshold 相同);
- open: N+1 类 (未知为第 N 类) 的 top1 与宏平均 P/R/F1 (sklearn average='macro', zero_division=0 的语义),
  分别在给定 tau (若提供) 和最佳 tau 下.
与旧脚本的差别: 查询默认不含锚点 (旧脚本把锚点自己也当查询, include_anchors=True 可复现), 旧脚本的 macro 指标用固定
ood_thr (默认 -1.3), 这里用 tau 参数. legacy_ccip_zero() 是旧脚本 main() 的逐行等价移植 (测试里对照).

zero_shot_refmean() 是 experiments/common/pair_eval.zero_shot 的移植 (参考图取均值向量), 用于复现旧 json 里的
zero_shot_ref1 / zero_shot_ref5.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_auc_score


def _normalize(feats) -> torch.Tensor:
    f = torch.as_tensor(np.asarray(feats) if not torch.is_tensor(feats) else feats).float()
    return torch.nn.functional.normalize(f, dim=-1)


# ------------------------------------------------------------------------------------------------ pair_eval port
def zero_shot_refmean(feats, labels, n_ref=1, seed=0) -> dict:
    """每类随机取 n_ref 张做参考 (取均值), 其余图片做最近类分类 (与 pair_eval.zero_shot 相同的随机数用法)"""
    rng = np.random.default_rng(seed)
    cls = np.asarray(labels)
    f = _normalize(feats)
    refs, query = [], np.ones(len(cls), bool)
    classes = np.unique(cls)
    for c in classes:
        idx = rng.choice(np.nonzero(cls == c)[0], n_ref, replace=False)
        query[idx] = False
        refs.append(f[torch.as_tensor(idx)].mean(0))
    refs = torch.nn.functional.normalize(torch.stack(refs), dim=-1)
    sim = f[torch.as_tensor(np.nonzero(query)[0])] @ refs.T
    target = torch.as_tensor(np.searchsorted(classes, cls[query]))
    top5 = sim.topk(min(5, len(classes)), dim=1).indices
    return dict(top1=float((top5[:, 0] == target).float().mean()),
                top5=float((top5 == target[:, None]).any(1).float().mean()))


# ------------------------------------------------------------------------------------------------ core helpers
def macro_prf(y_true, y_pred) -> dict:
    """sklearn precision/recall/f1_score(average='macro', zero_division=0) 的向量化版本:
    类集合 = y_true ∪ y_pred; 另给 accuracy"""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labs, inv = np.unique(np.r_[y_true, y_pred], return_inverse=True)
    t, p = inv[:len(y_true)], inv[len(y_true):]
    k = len(labs)
    tp = np.bincount(t[t == p], minlength=k).astype(float)
    n_true = np.bincount(t, minlength=k).astype(float)
    n_pred = np.bincount(p, minlength=k).astype(float)
    P = np.divide(tp, n_pred, out=np.zeros(k), where=n_pred > 0)
    R = np.divide(tp, n_true, out=np.zeros(k), where=n_true > 0)
    F = np.divide(2 * P * R, P + R, out=np.zeros(k), where=(P + R) > 0)
    return dict(accuracy=float((y_true == y_pred).mean()), precision_macro=float(P.mean()),
                recall_macro=float(R.mean()), f1_macro=float(F.mean()))


def class_scores(f: torch.Tensor, query_idx, anchor_lists, scale=1.0, bias=0.0) -> np.ndarray:
    """(Q, C): 第 c 列 = 查询与第 c 类各锚点 cos 的均值 * scale + bias"""
    flat = np.concatenate([np.asarray(a) for a in anchor_lists])
    owner = np.concatenate([np.full(len(a), c) for c, a in enumerate(anchor_lists)])
    sim = f[torch.as_tensor(np.asarray(query_idx))] @ f[torch.as_tensor(flat)].T       # (Q, total anchors)
    M = torch.zeros(len(flat), len(anchor_lists))
    M[torch.arange(len(flat)), torch.as_tensor(owner)] = 1.0
    M /= M.sum(0, keepdim=True)
    return ((sim @ M) * scale + bias).double().numpy()


def _topk_acc(scores, target, k):
    k = min(k, scores.shape[1])
    top = np.argpartition(-scores, k - 1, axis=1)[:, :k] if k < scores.shape[1] else np.tile(
        np.arange(scores.shape[1]), (len(scores), 1))
    return float((top == target[:, None]).any(1).mean())


def best_tau_f1(is_unknown, max_score):
    """已知/未知二分类: 分数越低越可能是未知. 返回 (best F1, tau, AUROC); 预测未知 = max_score <= tau"""
    if is_unknown.all() or (~is_unknown).all():
        return float('nan'), float('nan'), float('nan')
    P, R, thr = precision_recall_curve(is_unknown.astype(int), -max_score)
    f1 = 2 * P * R / np.maximum(P + R, 1e-12)
    f1 = f1[:-1]
    b = int(np.argmax(f1))
    auroc = roc_auc_score(is_unknown.astype(int), -max_score)
    return float(f1[b]), float(-thr[b]), float(auroc)


def ood_eval_once(f: torch.Tensor, labels, known, unknown, anchors, scale=1.0, bias=0.0, tau=None,
                  include_anchors=False) -> dict:
    """一次 OOD 评测. known / unknown: 类标签列表; anchors: 与 known 对齐的锚点下标数组列表.
    查询 = 已知类非锚点图 + 未知类全部图 (include_anchors=True 时已知类全部图)"""
    labels = np.asarray(labels)
    known = np.asarray(known)
    kpos = {c: i for i, c in enumerate(known.tolist())}
    is_k = np.isin(labels, known)
    is_u = np.isin(labels, np.asarray(unknown))
    q = is_k | is_u
    if not include_anchors:
        q[np.concatenate([np.asarray(a) for a in anchors])] = False
    qidx = np.nonzero(q)[0]
    N = len(known)
    y = np.array([kpos.get(c, N) for c in labels[qidx].tolist()])
    S = class_scores(f, qidx, anchors, scale, bias)
    mx, am = S.max(1), S.argmax(1)
    unk = y == N
    out = dict(n_known=N, n_unknown=int(len(unknown)), n_query=int(len(qidx)), n_query_unknown=int(unk.sum()))
    # closed set (已知类查询, 不拒识)
    kq = ~unk
    out['closed_top1'] = float((am[kq] == y[kq]).mean())
    out['closed_top5'] = _topk_acc(S[kq], y[kq], 5)
    m = macro_prf(y[kq], am[kq])
    out['closed_macro_acc'] = float(np.mean([(am[kq][y[kq] == c] == c).mean() for c in np.unique(y[kq])]))
    out.update({f'closed_{k}': v for k, v in m.items() if k != 'accuracy'})
    # OOD 二分类
    f1b, tau_b, auroc = best_tau_f1(unk, mx)
    out.update(ood_auroc=auroc, ood_f1_best=f1b, tau_best=tau_b)

    def open_metrics(t, strict):
        pred = am.copy()
        pred[(mx < t) if strict else (mx <= t)] = N
        m = macro_prf(y, pred)
        rej = pred == N
        tp = float((rej & unk).sum())
        P = tp / max(rej.sum(), 1)
        R = tp / max(unk.sum(), 1)
        return dict(top1=m['accuracy'], precision_macro=m['precision_macro'], recall_macro=m['recall_macro'],
                    f1_macro=m['f1_macro'], ood_f1=2 * P * R / max(P + R, 1e-12))
    if not np.isnan(tau_b):
        out.update({f'open_best_{k}': v for k, v in open_metrics(tau_b, strict=False).items()})
    if tau is not None:
        out['tau'] = float(tau)
        out.update({f'open_tau_{k}': v for k, v in open_metrics(tau, strict=True).items()})
    return out


def _aggregate(runs: list) -> dict:
    keys = [k for k, v in runs[0].items() if isinstance(v, (int, float))]
    out = {}
    for k in keys:
        v = np.array([r[k] for r in runs], dtype=float)
        out[k] = float(np.nanmean(v))
        if k not in ('n_known', 'n_unknown', 'tau'):
            out[k + '_std'] = float(np.nanstd(v))
    out['repeats'] = len(runs)
    return out


# ------------------------------------------------------------------------------------------------ protocols
def random_protocol(labels, N, X, A_max=5, repeats=20, seed=0, min_known_images=None) -> list:
    """均匀随机的 (N 已知, X 未知) 协议: 返回 [{known, unknown, anchors(下标, 每类 A_max 个)}]"""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    need = A_max + 1 if min_known_images is None else min_known_images
    ok_known = classes[counts >= need]
    out = []
    for _ in range(repeats):
        known = rng.choice(ok_known, N, replace=False)
        rest = np.setdiff1d(classes, known)
        unknown = rng.choice(rest, X, replace=False)
        anchors = [rng.choice(np.nonzero(labels == c)[0], A_max, replace=False) for c in known]
        out.append(dict(known=known.tolist(), unknown=unknown.tolist(), anchors=anchors))
    return out


def nway_eval(feats, labels, A=1, n_way=None, repeats=20, seed=0, scale=1.0, bias=0.0) -> dict:
    """闭集 N 路 A 锚点 (无未知类). n_way=None 用全部 (张数 > A 的) 类"""
    f = _normalize(feats)
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    classes = classes[counts > A]
    n_way = len(classes) if n_way is None else n_way
    rng = np.random.default_rng(seed)
    runs = []
    for _ in range(repeats):
        known = rng.choice(classes, n_way, replace=False) if n_way < len(classes) else classes
        anchors = [rng.choice(np.nonzero(labels == c)[0], A, replace=False) for c in known]
        r = ood_eval_once(f, labels, known, [], anchors, scale, bias)
        runs.append({k: v for k, v in r.items() if k.startswith('closed_') or k == 'n_query'})
    out = _aggregate(runs)
    out.update(A=A, n_way=int(n_way))
    return out


def ood_eval(feats, labels, protocol: list, A_list=(1, 3, 5), scale=1.0, bias=0.0, tau=None,
             include_anchors=False) -> dict:
    """protocol: random_protocol() 的输出 (anchors 为下标). 返回 {f'A{a}': 聚合指标}"""
    f = _normalize(feats)
    res = {}
    for A in A_list:
        runs = [ood_eval_once(f, labels, r['known'], r['unknown'], [np.asarray(a)[:A] for a in r['anchors']],
                              scale, bias, tau, include_anchors) for r in protocol]
        res[f'A{A}'] = _aggregate(runs)
    return res


def load_ood_protocol(path_or_dict) -> dict:
    if isinstance(path_or_dict, (str, Path)):
        with open(path_or_dict) as fp:
            return json.load(fp)
    return path_or_dict


def ood_protocol_eval(feats, sample_keys, labels, protocol, scale=1.0, bias=0.0, tau=None, A_list=None,
                      configs=None) -> dict:
    """消费 ccip_openset_v1/ood_protocol.json: feats / sample_keys / labels 为评测集 (FULL) 的特征与键.
    返回 {'N20_X20': {'A1': {...}, ...}, ...}"""
    proto = load_ood_protocol(protocol)
    keys = np.asarray(sample_keys)
    pos = {k: i for i, k in enumerate(keys.tolist())}
    labels = np.asarray(labels)
    f = _normalize(feats)
    A_list = A_list or proto.get('anchors', [1, 3, 5])
    out = {}
    for p in proto['protocols']:
        name = f"N{p['N']}_X{p['X']}"
        if configs and name not in configs and (p['N'], p['X']) not in configs:
            continue
        res = {}
        for A in A_list:
            runs = []
            for r in p['repeats']:
                missing = [k for c in r['known'] for k in r['anchors'][str(c)][:A] if k not in pos]
                if missing:
                    raise KeyError(f'{len(missing)} anchor keys not in eval features, e.g. {missing[:3]}')
                anchors = [np.array([pos[k] for k in r['anchors'][str(c)][:A]]) for c in r['known']]
                runs.append(ood_eval_once(f, labels, r['known'], r['unknown'], anchors, scale, bias, tau))
            res[f'A{A}'] = _aggregate(runs)
        out[name] = res
    return out


# ------------------------------------------------------------------------------------------------ legacy port
def legacy_ccip_zero(embs, labels, N, A, seen, logit_scale, logit_bias, ood_thr=-1.3, repeats=20, seed=42) -> dict:
    """scripts/eval/eval_ccip_zero.py main() 的等价移植 (不画图): 前 N 个标签, 前 seen 个为已知类,
    np.random.seed(seed) 后 repeats 次 compute_metrics_with_random_anchors; 查询包含锚点; 固定 ood_thr.
    返回各指标均值 (top1, top5, precision_macro, recall_macro, f1_macro) 及最佳 OOD 阈值 F1 的均值"""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    embs, labels = np.asarray(embs, dtype=np.float64), np.asarray(labels)
    e, l = embs[labels < N], labels[labels < N]
    np.random.seed(seed)
    acc = {k: [] for k in ('top1_accuracy', 'top5_accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
                           'ood_f1_best')}
    for _ in range(repeats):
        idx, owner = [], []
        for c in range(seen):
            m = np.where(l == c)[0]
            na = min(A, len(m))
            idx.extend(np.random.choice(m, size=na, replace=False))
            owner.extend([c] * na)
        idx, owner = np.array(idx), np.array(owner)
        sim = e @ e[idx].T
        oh = np.zeros((len(idx), seen))
        oh[np.arange(len(owner)), owner] = 1
        ps = (sim @ oh) / np.maximum(oh.sum(0), 1)[None] * logit_scale + logit_bias
        sc = ps.max(1)
        y_ood = (l >= seen).astype(int)
        if len(np.unique(y_ood)) > 1:
            P, R, _ = precision_recall_curve(y_ood, -sc)
            with np.errstate(invalid='ignore'):   # 旧脚本 P+R=0 时得 nan, 这里用 nanmax
                acc['ood_f1_best'].append(float(np.nanmax((2 * P * R / (P + R))[:-1])))
        pred = ps.argmax(1)
        ood = sc < ood_thr
        pred[ood] = seen
        ps = np.hstack([ps, ood[:, None] * logit_scale + logit_bias])
        yt = np.clip(l, None, seen)
        acc['top1_accuracy'].append(accuracy_score(yt, pred))
        k = min(5, ps.shape[1])
        acc['top5_accuracy'].append(float(np.any(np.argsort(ps, axis=1)[:, -k:] == yt[:, None], axis=1).mean()))
        acc['precision_macro'].append(precision_score(yt, pred, average='macro', zero_division=0))
        acc['recall_macro'].append(recall_score(yt, pred, average='macro', zero_division=0))
        acc['f1_macro'].append(f1_score(yt, pred, average='macro', zero_division=0))
    return {k: float(np.mean(v)) if v else float('nan') for k, v in acc.items()}
