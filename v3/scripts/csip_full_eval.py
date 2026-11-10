"""CSIP 全面评测 (在缓存特征上, CPU 可跑): test100_0918 (Danbooru 100 画师) + csip_eval_public (122 画师)。

任务 (每个集合):
  pairwise    同画师验证 PR-AUC / best F1                        (去重、署名核验)
  nway        已知画师 1 / 5 张参考图的闭集识别 top1 / macro-F1   (画师识别)
  retrieval   MAP@R / R@1                                         (以图找同画师作品)
  clustering  UMAP + HDBSCAN 的 NMI / ARI / 纯度                  (无标注图集的画师发现, CCIP 伪画师就靠它)
  open-set    一半画师已知、一半未知, 拒识 AUROC / F1 (1 / 5 张参考)  (陌生画师不能被硬套成已知画师)
  crosschar   同角色不同画师 = 困难负样本 (仅 0918, 有角色标注)     (画风 vs 内容)
另外: 两个集合之间的阈值迁移 (一个阈值全局部署的 F1 损失)、Matryoshka 前缀维度 (--matryoshka)、
0918 上按年份差 / 媒介 / 同角色分桶的召回与误报 (csip_error_analysis)。
--erase: 先对特征做 x @ M + b (事后线性擦除, posthoc_kc*_kh*.pt) 再评测。
[--sets csip_unseen_v1 ...] (默认两个旧集合; 不含 0918 时跳过分桶)
用法: python -m v3.scripts.csip_full_eval --name x2_erased --feats-dir <exp>/eval_native/feats --scale S --bias B
      [--erase <exp>/eval/posthoc_kc3_kh4.pt] [--matryoshka] --out <json>
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from v3.eval import pairwise, zeroshot
from v3.eval.clustering import clustering_eval
from v3.eval.runner import evaluate_set, load_set

SETS = ('test100_0918', 'csip_eval_public')


def load_aligned(feats_dir, name, df, erase=None):
    d = np.load(Path(feats_dir) / f'{name}.npz')
    pos = {str(k): i for i, k in enumerate(d['sample_key'])}
    f = d['feats'][[pos[str(k)] for k in df['sample_key']]].astype(np.float32)
    if erase is not None:
        f = (torch.from_numpy(f) @ erase['M'] + erase['b']).numpy()
    return f


def summarize(name, r):
    s = dict(pr_auc=r['pr_auc'], best_f1=r['best_f1'], zs1=r['zero_shot_ref1']['top1'],
             nway_A1=r['nway_A1']['closed_top1'], nway_A5=r['nway_A5']['closed_top1'],
             nway_A5_f1=r['nway_A5']['closed_f1_macro'],
             map_at_r=r['retrieval']['map_at_r'], r_at_1=r['retrieval']['recall_at_1'])
    c = r.get('clustering')
    if c:
        s.update(clus_nmi=c['nmi'], clus_ari=c['ari'], clus_purity=c['purity'], clus_noise=c['noise_frac'],
                 clus_n=c['n_clusters'])
    for A in (1, 5):
        o = r['openset'][f'A{A}']
        s[f'ood_auroc_A{A}'] = o['ood_auroc']
        s[f'ood_f1_A{A}'] = o['ood_f1_best']
    cc = r.get('crosschar')
    if cc:
        s.update(cc_pr_balanced=cc['pr_auc_balanced'], cc_hardneg_fpr=cc['hardneg_fpr'],
                 cc_randneg_fpr=cc['randneg_fpr'])
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--feats-dir', required=True)
    ap.add_argument('--scale', type=float, required=True)
    ap.add_argument('--bias', type=float, required=True)
    ap.add_argument('--erase')
    ap.add_argument('--matryoshka', action='store_true')
    ap.add_argument('--out', required=True)
    ap.add_argument('--sets', nargs='+', default=list(SETS))
    args = ap.parse_args()
    erase = torch.load(args.erase, weights_only=True) if args.erase else None
    out = dict(name=args.name, feats_dir=args.feats_dir, erase=args.erase, sets={}, summary={})
    full = {}
    for name in args.sets:
        t0 = time.time()
        df, _ = load_set(name)
        f = load_aligned(args.feats_dir, name, df, erase)
        y = df['label'].values
        r = evaluate_set(name, f, df, args.scale, args.bias,
                         tasks=('pairwise', 'zeroshot', 'retrieval', 'crosschar'))
        # 画师每人几十到几百张、常有几种子画风: min_cluster_size 扫得比 CCIP (2-7) 宽, 按 NMI 取最优 (各模型同一规则)
        r['clustering'] = {k: v for k, v in clustering_eval(f, y, seed=0, mcs_range=(2, 3, 5, 8, 12, 20, 30, 50)).items()
                           if k != 'per_mcs'}
        n_cls = len(np.unique(y))
        proto = zeroshot.random_protocol(y, n_cls // 2, n_cls - n_cls // 2, A_max=5, repeats=20, seed=0)
        r['openset'] = zeroshot.ood_eval(f, y, proto, A_list=(1, 5), scale=args.scale, bias=args.bias)
        if args.matryoshka:
            r['matryoshka'] = {}
            for k in (512, 256, 128):
                if k < f.shape[1]:
                    m = evaluate_set(name, f[:, :k], df, args.scale, args.bias, tasks=('pairwise',))
                    r['matryoshka'][k] = dict(pr_auc=m['pr_auc'], best_f1=m['best_f1'])
        full[name] = r
        out['summary'][name] = summarize(name, r)
        out['sets'][name] = {k: v for k, v in r.items() if k != 'curve'}
        print(name, json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                for k, v in out['summary'][name].items()}), f'{time.time() - t0:.0f}s', flush=True)
    if len(full) > 1:
        out['threshold_drift'] = pairwise.threshold_drift(full)
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    if 'test100_0918' not in args.sets:
        return
    # 0918 分桶 (年份差 / 媒介 / 同角色 / 同作品)
    od = Path(args.out).parent
    df, _ = load_set('test100_0918')
    tmp = od / f'_{args.name}_0918.npz'
    np.savez(tmp, feats=load_aligned(args.feats_dir, 'test100_0918', df, erase), sample_key=np.asarray(df['sample_key'], dtype=str))
    ej = od / f'{args.name}_err.json'
    p = subprocess.run([sys.executable, '-m', 'v3.scripts.csip_error_analysis', str(tmp), '--out', str(ej)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    tmp.unlink()
    if p.returncode:
        print('error analysis failed:', p.stderr[-2000:], flush=True)
        return
    e = json.loads(ej.read_text())['buckets']
    out['buckets_0918'] = {k: dict(recall=v.get('pos', {}).get('rate_above'), fp=v.get('neg', {}).get('rate_above'))
                           for k, v in e.items()}
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    print('drift', json.dumps(out.get('threshold_drift'), default=float)[:300])


if __name__ == '__main__':
    main()
