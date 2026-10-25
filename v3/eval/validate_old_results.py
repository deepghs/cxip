"""用已存的特征复现旧评测数字, 并在真实特征上跑一遍新增指标 (CPU). 结果写 /data/neko/cxip_v3/eval/validation/.

cd /data/neko/cxip && python -m v3.eval.validate_old_results [--skip-bias] [--skip-timing]

1. CSIP: results/feats/hf_csip_{v1,v2,v2.1}/{test100,test100v2}.npz vs results/hf_csip_*_{test100,test100v2}.json
   (PR-AUC / AP / ROC / best F1 / 阈值: 直方图版和精确版; zero_shot_ref1/ref5);
2. CCIP: results/bias_targets.pt 的 'char' (HF AnimeCCIP 在 zc500x10 上的特征) vs results/hf_ccip_zerochan.json;
3. OOD: legacy_ccip_zero (旧脚本移植) 与 ood_eval_once(include_anchors=True, 同样的锚点) 在 zc500x10 上逐项一致;
4. 新指标: test100 上 CSIP 跨角色协议; I_style (HF AnimeCCIP → CSIP v2.1, 4,887 张与 train_ccip_feats 重叠的图) 与
   I_char (CSIP → HF AnimeCCIP, 同一批图, 按画师分层);
5. 计时: 25,000 张 (1000 类, D=1024) 全部 3.1 亿对的直方图版 pairwise, 以及与精确排序版的差异.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import bias as B
from . import pairwise as pw
from . import zeroshot as zs
from .evalsets import REPRO, load_crosschar

OUT = Path('/data/neko/cxip_v3/eval/validation')
KEYS = ['pr_auc', 'ap', 'roc_auc', 'best_f1', 'best_threshold']


def cmp(ref, new, keys=KEYS):
    return {k: dict(ref=ref[k], new=new[k], diff=new[k] - ref[k]) for k in keys if k in ref and k in new}


def check_csip():
    rows = []
    for m in ['hf_csip_v1', 'hf_csip_v2', 'hf_csip_v2.1']:
        for ds in ['test100', 'test100v2']:
            d = np.load(REPRO / f'results/feats/{m}/{ds}.npz')
            r = json.load(open(REPRO / f'results/{m}_{ds}.json'))
            ref = r['datasets'][ds]
            t0 = time.time()
            h = pw.pairwise_eval(d['feats'], d['cls'], r['logit_scale'], r['logit_bias'], exact=False, curve_points=0)
            th = time.time() - t0
            e = pw.pairwise_eval(d['feats'], d['cls'], r['logit_scale'], r['logit_bias'], exact=True, curve_points=0)
            z = {n: zs.zero_shot_refmean(d['feats'], d['cls'], n) for n in (1, 5)}
            rows.append(dict(model=m, dataset=ds, n_images=int(len(d['cls'])), hist=cmp(ref, h), exact=cmp(ref, e),
                             hist_seconds=th,
                             zero_shot={f'ref{n}': {k: dict(ref=ref[f'zero_shot_ref{n}'][k], new=z[n][k],
                                                             diff=z[n][k] - ref[f'zero_shot_ref{n}'][k])
                                                     for k in ('top1', 'top5')} for n in (1, 5)},
                             plateau=dict(width=h['f1_plateau_width'], width_cos=h['f1_plateau_width_cos'])))
            print(m, ds, {k: f"{v['diff']:.1e}" for k, v in rows[-1]['hist'].items()}, flush=True)
    return rows


def zc500_feats():
    t = torch.load(REPRO / 'results/bias_targets.pt', weights_only=True)
    names = [c for _, c in t['char_items']]
    labels = np.unique(names, return_inverse=True)[1]
    return t['char'].float().numpy(), labels


def check_ccip():
    r = json.load(open(REPRO / 'results/hf_ccip_zerochan.json'))
    ref = r['datasets']['zc500x10']
    f, l = zc500_feats()
    h = pw.pairwise_eval(f, l, r['logit_scale'], r['logit_bias'], exact=False, curve_points=0)
    e = pw.pairwise_eval(f, l, r['logit_scale'], r['logit_bias'], exact=True, curve_points=0)
    z = {n: zs.zero_shot_refmean(f, l, n) for n in (1, 5)}
    return dict(dataset='zc500x10', model='dghstmp/AnimeCCIP', hist=cmp(ref, h), exact=cmp(ref, e),
                zero_shot={f'ref{n}': {k: dict(ref=ref[f'zero_shot_ref{n}'][k], new=z[n][k],
                                                diff=z[n][k] - ref[f'zero_shot_ref{n}'][k]) for k in ('top1', 'top5')}
                           for n in (1, 5)})


def legacy_anchor_protocol(labels, N, A, seen, repeats=20, seed=42):
    """复现旧脚本的锚点抽取 (np.random.seed(42), 每次重复对 seen 个类依次 np.random.choice)"""
    l = labels[labels < N]
    np.random.seed(seed)
    out = []
    for _ in range(repeats):
        out.append([np.random.choice(np.where(l == c)[0], size=min(A, int((l == c).sum())), replace=False)
                    for c in range(seen)])
    return out


def check_ood_legacy():
    """zc500x10 (HF AnimeCCIP): 旧 eval_ccip_zero 的 (N, seen) 配置; 新实现用同一组锚点 + include_anchors + 固定 tau"""
    f, l = zc500_feats()
    f = f / np.linalg.norm(f, axis=1, keepdims=True)
    scale, bias, thr = 23.521757125854492, -12.256535530090332, -1.3      # 旧脚本里写死的值
    res = []
    for N, seen in [(40, 20), (60, 40), (100, 60)]:
        for A in (1, 3, 5):
            old = zs.legacy_ccip_zero(f, l, N, A, seen, scale, bias, thr)
            m = l < N
            fe, le = torch.as_tensor(f[m]).float(), l[m]
            runs = [zs.ood_eval_once(fe, le, list(range(seen)), list(range(seen, N)), anc, scale, bias, tau=thr,
                                     include_anchors=True) for anc in legacy_anchor_protocol(l, N, A, seen)]
            new = {k: float(np.mean([r[k] for r in runs])) for k in runs[0] if isinstance(runs[0][k], float)}
            res.append(dict(N=N, seen=seen, A=A,
                            top1=dict(old=old['top1_accuracy'], new=new['open_tau_top1']),
                            precision_macro=dict(old=old['precision_macro'], new=new['open_tau_precision_macro']),
                            recall_macro=dict(old=old['recall_macro'], new=new['open_tau_recall_macro']),
                            f1_macro=dict(old=old['f1_macro'], new=new['open_tau_f1_macro']),
                            ood_f1_best=dict(old=old['ood_f1_best'], new=new['ood_f1_best']),
                            ood_auroc=new['ood_auroc'], closed_top1=new['closed_top1']))
            print('ood', N, seen, A, {k: f"{v['new'] - v['old']:.1e}" for k, v in res[-1].items()
                                      if isinstance(v, dict)}, flush=True)
    return res


def overlap_feats():
    """test100 ∩ train_ccip_feats: HF AnimeCCIP 特征 + CSIP v1/v2/v2.1 特征 + 画师标签"""
    base = REPRO / 'data/danbooru-artist-split-0918/train_ccip_feats'
    names = {}
    for r in range(4):
        for i, fn in enumerate(json.load(open(base / f'files_r{r}.json'))):
            names[Path(fn).name] = (r, i)
    t100 = {m: np.load(REPRO / f'results/feats/{m}/test100.npz') for m in ['hf_csip_v1', 'hf_csip_v2', 'hf_csip_v2.1']}
    files = [Path(f).name for f in t100['hf_csip_v2.1']['files']]
    sel = [i for i, f in enumerate(files) if f in names]
    mm = {r: np.load(base / f'feats_r{r}.npy', mmap_mode='r') for r in range(4)}
    ccip = np.stack([np.asarray(mm[names[files[i]][0]][names[files[i]][1]], dtype=np.float32) for i in sel])
    csip = {m: d['feats'][sel] for m, d in t100.items()}
    return ccip, csip, t100['hf_csip_v2.1']['cls'][sel], [t100['hf_csip_v2.1']['files'][i] for i in sel]


def check_new_metrics(skip_bias=False):
    out = {}
    cc = load_crosschar('test100')
    for m in ['hf_csip_v1', 'hf_csip_v2', 'hf_csip_v2.1']:
        d = np.load(REPRO / f'results/feats/{m}/test100.npz')
        r = json.load(open(REPRO / f'results/{m}_test100.json'))
        key = {str(Path(f)): i for i, f in enumerate(d['files'])}
        idx = np.array([key[k] for k in cc['sample_key']])
        out[f'crosschar_{m}'] = pw.csip_cross_character(d['feats'][idx], cc['label'].values, cc['char_label'].values,
                                                        r['logit_scale'], r['logit_bias'],
                                                        threshold=r['datasets']['test100']['best_threshold'])
        print('crosschar', m, {k: round(v, 4) for k, v in out[f'crosschar_{m}'].items()
                               if 'balanced' in k or 'fpr' in k or 'roc' in k}, flush=True)
    if skip_bias:
        return out
    ccip, csip, artists, _ = overlap_feats()
    out['bias_n_images'] = int(len(artists))
    nt = torch.get_num_threads()
    torch.set_num_threads(16)
    out['I_style_hf_ccip_to_csip_v2.1'] = B.style_bias(ccip, csip['hf_csip_v2.1'], list(artists))
    out['I_style_ce_hf_ccip'] = B.style_bias(ccip, None, list(artists), form='ce')
    out['I_style_csip_v2_to_csip_v2.1 (sanity, 应很高)'] = B.style_bias(csip['hf_csip_v2'], csip['hf_csip_v2.1'],
                                                                      list(artists))
    for m in ['hf_csip_v1', 'hf_csip_v2', 'hf_csip_v2.1']:
        out[f'I_char_{m}_to_hf_ccip (按画师分层)'] = B.char_bias(csip[m], ccip, list(artists))
    torch.set_num_threads(nt)
    for k, v in out.items():
        if isinstance(v, dict) and 'I' in v:
            print(k, f"I={v['I']:.4f} real={v['real']:.4f} base={v['base']:.4f} ({v['seconds']:.0f}s)", flush=True)
    return out


def timing(n=25000, n_cls=1000, dim=1024, exact=True):
    """合成特征: 类中心 + 噪声 (类大小不均), 测直方图版耗时; exact=True 时再跑精确排序版对比"""
    rng = np.random.default_rng(0)
    sizes = rng.dirichlet(np.ones(n_cls) * 2) * n
    labels = np.repeat(np.arange(n_cls), np.maximum(1, np.round(sizes)).astype(int))[:n]
    labels = np.r_[labels, rng.integers(0, n_cls, n - len(labels))]
    centers = rng.standard_normal((n_cls, dim)).astype(np.float32)
    feats = centers[labels] * 0.35 + rng.standard_normal((n, dim)).astype(np.float32)
    out = dict(n_images=n, n_classes=n_cls, dim=dim, threads=torch.get_num_threads())
    t0 = time.time()
    h = pw.pairwise_eval(feats, labels, 20.0, -10.0, exact=False)
    out['hist_seconds'] = time.time() - t0
    out['n_pairs'] = h['n_pairs']
    torch.set_num_threads(32)
    t0 = time.time()
    pw.pairwise_eval(feats, labels, 20.0, -10.0, exact=False)
    out['hist_seconds_32threads'] = time.time() - t0
    torch.set_num_threads(out['threads'])
    if exact:
        t0 = time.time()
        e = pw.pairwise_eval(feats, labels, 20.0, -10.0, exact=True, curve_points=0)
        out['exact_seconds'] = time.time() - t0
        out['hist_vs_exact'] = {k: h[k] - e[k] for k in KEYS + ['f1_plateau_width']}
    print('timing', out, flush=True)
    return out


def suite_timing():
    """FULL 规模 (22,501 张, 真实标签 / 分组 / OOD 协议) 的合成特征上跑 runner.evaluate_set 的各部分并计时"""
    from . import clustering as cl
    from . import retrieval as rt
    from .evalsets import load_ccip_openset, ood_protocol_path
    from .runner import evaluate_set
    df, _ = load_ccip_openset('full')
    rng = np.random.default_rng(0)
    lab = df['label'].values
    _, inv = np.unique(lab, return_inverse=True)
    centers = rng.standard_normal((inv.max() + 1, 1024)).astype(np.float32)
    feats = centers[inv] * 0.3 + rng.standard_normal((len(lab), 1024)).astype(np.float32)
    out = dict(n_images=len(lab), threads=torch.get_num_threads())
    t0 = time.time()
    pw.pairwise_eval(feats, lab, 20.0, -10.0)
    out['pairwise_s'] = time.time() - t0
    t0 = time.time()
    rt.retrieval_eval(feats, lab)
    out['retrieval_s'] = time.time() - t0
    t0 = time.time()
    r = zs.ood_protocol_eval(feats, df['sample_key'].values, lab, ood_protocol_path(), 20.0, -10.0)
    out['ood_protocol_s'] = time.time() - t0
    out['ood_example'] = {k: r['N500_X500']['A1'][k] for k in ('ood_auroc', 'ood_f1_best', 'closed_top1')}
    t0 = time.time()
    evaluate_set('openset_full', feats, df, 20.0, -10.0,
                 tasks=('pairwise', 'zeroshot', 'retrieval', 'ood'))
    out['evaluate_set_full_no_clustering_s'] = time.time() - t0
    t0 = time.time()
    c = cl.clustering_eval(feats, lab, seed=0)
    out['clustering_umap_full_s'] = time.time() - t0
    out['clustering_umap_reduce_s'] = c['seconds_reduce']
    print('suite timing', out, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-bias', action='store_true')
    ap.add_argument('--skip-timing', action='store_true')
    ap.add_argument('--no-exact-timing', action='store_true')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res = dict(csip=check_csip(), ccip=check_ccip(), ood_legacy=check_ood_legacy(),
               new_metrics=check_new_metrics(a.skip_bias))
    if not a.skip_timing:
        res['timing'] = timing(exact=not a.no_exact_timing)
        res['suite_timing'] = suite_timing()
    with open(OUT / 'validation.json', 'w') as f:
        json.dump(res, f, indent=1, default=float)
    print('wrote', OUT / 'validation.json')


if __name__ == '__main__':
    main()
