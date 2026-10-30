"""模型无关的评测入口. 模型只需提供 encode(list[PIL.Image]) -> (N, D) tensor/ndarray, 以及可选的 scale / bias
(score = scale * cos + bias, scale 为 exp(logit_scale) 之后的值; 不给则 scale=1, bias=0 即纯 cos).

    from v3.eval.runner import run_suite
    res = run_suite(encode, model_name='ccip_v3_step10k', scale=s, bias=b, sets=['openset_fast', 'zc500x10'],
                    out='/data/neko/cxip_v3/eval/results/ccip_v3_step10k.json', feats_dir='.../feats/ccip_v3_step10k')

评测集名: openset_full / openset_fast (CCIP 开放集, 另按 subset 和 backbone_seen 分组), LEGACY_SETS 里的旧集
(zc50x15, zc500x10, test_ccip, ccip_eval, ccip_test_v1; test100_0918, test100, test100v2, csip_eval_public).
任务: pairwise (PR-AUC 等, F1 平台, 固定阈值), zeroshot (旧的参考均值 top1/5 + A 锚点 N 路), retrieval (mAP@R, R@1),
clustering (UMAP+HDBSCAN, 默认只在 FAST 和小集合上跑), ood (openset_full 的 OOD 协议), crosschar (CSIP 跨角色).
特征可缓存为 npz (feats fp16 + sample_key), 再次评测直接读. 也可绕过图片直接调 evaluate_features().

bias 探针: run_bias(encode, kind='csip') 计算 I_c / I_char; run_bias(encode, kind='ccip') 计算 I_style.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from . import clustering, evalsets, pairwise, retrieval, zeroshot

CCIP_SETS = ['openset_fast', 'openset_full'] + evalsets.CCIP_LEGACY
CSIP_SETS = evalsets.CSIP_LEGACY
DEFAULT_TASKS = ('pairwise', 'zeroshot', 'retrieval', 'clustering', 'ood', 'crosschar')
CLUSTER_MAX_IMAGES = 12000      # 更大的集合默认跳过聚类 (UMAP 单线程 + HDBSCAN, 数分钟)


def load_set(name):
    """返回 (df, read_image), df 至少有 sample_key, label 两列"""
    if name in ('openset_full', 'openset_fast'):
        return evalsets.load_ccip_openset(name.split('_')[1])
    if name in evalsets.INDEX_SETS:
        return evalsets.load_index_set(name)
    return evalsets.load_legacy(name)


def extract_features(encode, read_image, n, batch_size=64, workers=16, desc='') -> np.ndarray:
    """多线程解码图片, 按批调用 encode, 返回 float32 (N, D)"""
    out = []
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        for b0 in range(0, n, batch_size):
            imgs = list(ex.map(read_image, range(b0, min(n, b0 + batch_size))))
            f = encode(imgs)
            out.append(f.detach().float().cpu().numpy() if torch.is_tensor(f) else np.asarray(f, np.float32))
    feats = np.concatenate(out) if out else np.zeros((0, 0), np.float32)
    print(f'[extract] {desc} {n} images {time.time() - t0:.0f}s', flush=True)
    return feats


def cached_features(encode, name, df, read_image, feats_dir=None, batch_size=64, workers=16):
    if feats_dir:
        p = Path(feats_dir) / f'{name}.npz'
        if p.exists():
            d = np.load(p, allow_pickle=False)
            if len(d['sample_key']) == len(df) and (d['sample_key'] == df['sample_key'].values.astype(str)).all():
                return d['feats'].astype(np.float32)
    feats = extract_features(encode, read_image, len(df), batch_size, workers, name)
    if feats_dir:
        Path(feats_dir).mkdir(parents=True, exist_ok=True)
        np.savez(Path(feats_dir) / f'{name}.npz', feats=feats.astype(np.float16),
                 sample_key=df['sample_key'].values.astype(str))
    return feats


def evaluate_features(feats, labels, scale=1.0, bias=0.0, tasks=DEFAULT_TASKS, fixed_thresholds=(), seed=0,
                      cluster=None) -> dict:
    """一组特征 + 标签上的通用指标 (pairwise / zeroshot / retrieval / clustering)"""
    labels = np.asarray(labels)
    out = {}
    if 'pairwise' in tasks:
        out.update(pairwise.pairwise_eval(feats, labels, scale, bias, fixed_thresholds=fixed_thresholds))
    else:
        out.update(n_images=int(len(labels)), n_classes=int(len(np.unique(labels))))
    if 'zeroshot' in tasks:
        _, cnt = np.unique(labels, return_counts=True)
        for n_ref in (1, 5):
            if cnt.min() > n_ref:
                out[f'zero_shot_ref{n_ref}'] = zeroshot.zero_shot_refmean(feats, labels, n_ref)
        for A in (1, 5):
            if (cnt > A).sum() >= 2:
                out[f'nway_A{A}'] = zeroshot.nway_eval(feats, labels, A=A, repeats=5, seed=seed, scale=scale,
                                                       bias=bias)
    if 'retrieval' in tasks:
        out['retrieval'] = retrieval.retrieval_eval(feats, labels)
    do_cluster = cluster if cluster is not None else len(labels) <= CLUSTER_MAX_IMAGES
    if 'clustering' in tasks and do_cluster:
        r = clustering.clustering_eval(feats, labels, seed=seed)
        out['clustering'] = {k: v for k, v in r.items()}
    return out


def _subgroups(df):
    """开放集的分组: subset (new_franchise / hard) 与 backbone_seen (seen / unseen)"""
    g = {}
    if 'subset' in df:
        for s in ('new_franchise', 'hard'):
            g[s] = (df['subset'] == s).values
    if 'backbone_seen' in df:
        for s in ('seen', 'unseen'):
            g[f'backbone_{s}'] = (df['backbone_seen'] == s).values
    return g


def evaluate_set(name, feats, df, scale=1.0, bias=0.0, tasks=DEFAULT_TASKS, tau=None, seed=0) -> dict:
    labels = df['label'].values
    res = evaluate_features(feats, labels, scale, bias, tasks, seed=seed)
    if name.startswith('openset'):
        groups = {}
        for g, m in _subgroups(df).items():
            if m.sum() < 10:
                continue
            groups[g] = evaluate_features(feats[m], labels[m], scale, bias,
                                          [t for t in tasks if t in ('pairwise', 'zeroshot', 'retrieval')],
                                          seed=seed)
            groups[g].pop('curve', None)
        res['groups'] = groups
    if 'ood' in tasks and name == 'openset_full':
        res['ood'] = zeroshot.ood_protocol_eval(feats, df['sample_key'].values, labels,
                                                evalsets.ood_protocol_path(), scale, bias, tau)
    if 'crosschar' in tasks and name in evalsets.CSIP_LEGACY:
        try:
            cc = evalsets.load_crosschar(name)
        except FileNotFoundError:
            cc = None
        if cc is not None:
            pos = dict(zip(df['sample_key'].values, range(len(df))))
            idx = np.array([pos[k] for k in cc['sample_key'].values])
            res['crosschar'] = pairwise.csip_cross_character(feats[idx], cc['label'].values,
                                                             cc['char_label'].values, scale, bias,
                                                             threshold=res.get('best_threshold'))
    return res


def run_suite(encode, model_name, scale=None, bias=None, sets=('openset_fast',), tasks=DEFAULT_TASKS, out=None,
              feats_dir=None, batch_size=64, workers=16, tau=None, seed=0, extra=None) -> dict:
    scale = 1.0 if scale is None else float(scale)
    bias = 0.0 if bias is None else float(bias)
    result = dict(model=model_name, logit_scale=scale, logit_bias=bias, score='scale * cos + bias', datasets={},
                  time=time.strftime('%Y-%m-%d %H:%M:%S'))
    if extra:
        result.update(extra)
    for name in sets:
        t0 = time.time()
        df, read_image = load_set(name)
        feats = cached_features(encode, name, df, read_image, feats_dir, batch_size, workers)
        result['datasets'][name] = evaluate_set(name, feats, df, scale, bias, tasks, tau, seed)
        result['datasets'][name]['seconds'] = time.time() - t0
        print(f'[eval] {name}: pr_auc {result["datasets"][name].get("pr_auc", float("nan")):.4f} '
              f'({time.time() - t0:.0f}s)', flush=True)
        if out:
            _dump(result, out)
    pw = {k: v for k, v in result['datasets'].items() if 'best_threshold' in v and 'curve' in v}
    if len(pw) >= 2:
        result['threshold_drift'] = pairwise.threshold_drift(pw)
    if out:
        _dump(result, out)
    return result


def _dump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1, default=float)


# ------------------------------------------------------------------------------------------------ bias probes
BIAS_TARGETS = evalsets.REPRO / 'results/bias_targets.pt'
STYLE_TARGETS = evalsets.REPRO / 'results/feats/hf_csip_v2.1/test100.npz'


def _read_path(items):
    def read(i):
        with open(items[i], 'rb') as f:
            return evalsets.decode_image(f.read())
    return read


def run_bias(encode, kind, model_name='', out=None, batch_size=64, workers=16, feats_dir=None, repeats=3,
             seed=42) -> dict:
    """kind='csip': I_c (bias_targets.pt 的 10k 张 + tagger 概率) 与 I_char (zc500x10 + HF AnimeCCIP 目标);
    kind='ccip': I_style (test100 的 9,509 张 + HF AnimeCSIP-v2.1 目标, 按画师分层), 回归版和分类版"""
    from . import bias as B
    res = dict(model=model_name, kind=kind)
    if kind == 'csip':
        t = torch.load(BIAS_TARGETS, weights_only=True)   # 只有张量 / 列表 / 字符串
        ci = [p for p, _ in t['content_items']]
        hi = [p for p, _ in t['char_items']]
        sc = _feats_or_extract(encode, 'bias_content', ci, feats_dir, batch_size, workers)
        sh = _feats_or_extract(encode, 'bias_char', hi, feats_dir, batch_size, workers)
        res['content'] = B.content_bias(sc, t['content'], seed=seed, repeats=repeats)
        res['char'] = B.char_bias(sh, t['char'], [c for _, c in t['char_items']], seed=seed, repeats=repeats)
    elif kind == 'ccip':
        d = np.load(STYLE_TARGETS)
        items = [str(evalsets.REPRO / f) for f in d['files']]
        x = _feats_or_extract(encode, 'bias_style', items, feats_dir, batch_size, workers)
        res['style'] = B.style_bias(x, d['feats'], list(d['cls']), seed=seed, repeats=repeats)
        res['style_ce'] = B.style_bias(x, None, list(d['cls']), form='ce', seed=seed, repeats=repeats)
    else:
        raise ValueError(kind)
    if out:
        _dump(res, out)
    return res


def _feats_or_extract(encode, name, paths, feats_dir, batch_size, workers):
    if feats_dir and (Path(feats_dir) / f'{name}.npz').exists():
        d = np.load(Path(feats_dir) / f'{name}.npz')
        if len(d['feats']) == len(paths):
            return d['feats'].astype(np.float32)
    f = extract_features(encode, _read_path(paths), len(paths), batch_size, workers, name)
    if feats_dir:
        Path(feats_dir).mkdir(parents=True, exist_ok=True)
        np.savez(Path(feats_dir) / f'{name}.npz', feats=f.astype(np.float16))
    return f
