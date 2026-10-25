"""事后线性擦除 (截断 LEACE): 不再训练, 只在 CSIP 向量后面接一个固定线性映射, 看偏置 / 画风精度怎么变。

x' = x − (x − μ) W U_k U_kᵀ W⁻¹,  W = Σ^{-1/2},  U_k = W·Σ_xz 的前 k 个左奇异向量
(k = rank(Σ_xz) 即 LEACE: 擦掉后任何线性探针都读不出 z; 截断版只擦协方差最强的 k 个方向)。

拟合数据 = 偏置探针自己的训练划分 (内容: seed 42 的前 80%; 角色: 按角色分层的 80%),
测量 = 探针原协议 (I_c / I_char 在留出部分上) + test100_0918 / csip_eval_public 的 PR-AUC (缓存的 crop384 特征)。
用法: python -m v3.scripts.posthoc_erase <exp-dir> [--ks 0 8 16 32 64 128] [--test-feats eval_native/feats]
(--test-feats: 测画风精度用的特征目录 (相对 exp-dir), 默认与偏置特征同目录 = crop384; 擦除器始终在偏置特征上拟合)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from v3.debias import STYLE_TAGS
from v3.eval import bias as B
from v3.eval.runner import BIAS_TARGETS, load_set


def fit_eraser(x, z, k, eps=1e-4):
    """返回 (μ, A): x' = x − (x − μ) A。x (n, d), z (n, m) 已在 float64。"""
    mu = x.mean(0)
    xc, zc = x - mu, z - z.mean(0)
    cov = xc.T @ xc / len(x)
    ev, V = torch.linalg.eigh(cov + eps * cov.diagonal().mean() * torch.eye(len(cov), dtype=x.dtype))
    W = V @ torch.diag(ev.rsqrt()) @ V.T
    W_inv = V @ torch.diag(ev.sqrt()) @ V.T
    U, S, _ = torch.linalg.svd(W @ (xc.T @ zc / len(x)), full_matrices=False)
    k = min(k, int((S > S[0] * 1e-6).sum()))
    Uk = U[:, :k]
    return mu, W @ Uk @ Uk.T @ W_inv, S


def apply(x, er):
    for mu, A in er:
        x = x - (x - mu) @ A
    return x


@torch.no_grad()
def fast_ap(feats, labels, bins=1 << 20, chunk=4096):
    """全部两两配对的 AP (cos 直方图法, GPU), 与 pairwise_eval 的 hist 法一致到 ~1e-4。"""
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    z = torch.nn.functional.normalize(torch.as_tensor(feats, dtype=torch.float32).to(dev), dim=-1)
    y = torch.as_tensor(np.unique(np.asarray(labels), return_inverse=True)[1]).to(dev)
    pos = torch.zeros(bins, dtype=torch.float64, device=dev)
    neg = torch.zeros_like(pos)
    n = len(z)
    for s in range(0, n, chunk):
        sim = z[s:s + chunk] @ z.T
        r = torch.arange(s, min(n, s + chunk), device=dev)
        upper = torch.arange(n, device=dev)[None] > r[:, None]
        same = (y[s:s + chunk, None] == y[None]) & upper
        b = ((sim.clamp(-1, 1) + 1) / 2 * (bins - 1)).long()
        pos += torch.bincount(b[same], minlength=bins).double()
        neg += torch.bincount(b[upper & ~same], minlength=bins).double()
    tp, fp = pos.flip(0).cumsum(0), neg.flip(0).cumsum(0)  # 从高分往低分扫
    prec = tp / (tp + fp).clamp_min(1)
    return float((prec * pos.flip(0)).sum() / pos.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('exp')
    ap.add_argument('--ks', type=int, nargs='+', default=[0, 4, 8, 16, 32, 64, 128])
    ap.add_argument('--concepts', nargs='+', default=['content', 'char', 'both'])
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--test-feats', default='eval/feats')
    ap.add_argument('--pairs', nargs='+', help='kc:kh ... (先擦内容 kc 维再擦角色 kh 维; 给了就只跑这些, 忽略 --ks/--concepts)')
    args = ap.parse_args()
    fd = Path(args.exp) / 'eval' / 'feats'
    t = torch.load(BIAS_TARGETS, weights_only=True)
    keep = [i for i, n in enumerate(t['tag_names']) if n not in STYLE_TAGS]
    xc = torch.tensor(np.load(fd / 'bias_content.npz')['feats'], dtype=torch.float64)
    xh = torch.tensor(np.load(fd / 'bias_char.npz')['feats'], dtype=torch.float64)
    # 与探针相同的划分, 擦除器只用训练部分拟合
    n = len(xc)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    tr_c = perm[:n - int(n * 0.1) - int(n * 0.1)]
    tr_h, _ = B.stratified_split([c for _, c in t['char_items']], 0.2, 42)
    zc = t['content'].double()[:, keep]
    zh = t['char'].double()
    tests = {}
    for name in ('test100_0918', 'csip_eval_public'):
        d = np.load(Path(args.exp) / args.test_feats / f'{name}.npz')
        df, _ = load_set(name)
        lab = dict(zip(df['sample_key'].astype(str), df['label']))
        tests[name] = (torch.tensor(d['feats'], dtype=torch.float64), np.array([lab[k] for k in d['sample_key']]))
    if args.pairs:
        jobs = [(int(kc), int(kh)) for kc, kh in (p.split(':') for p in args.pairs)]
    else:
        jobs = [(k * (c in ('content', 'both')), k * (c in ('char', 'both')), c) for c in args.concepts for k in args.ks
                if k > 0 or c == args.concepts[0]]
    out = []
    for job in jobs:
        kc, kh = job[:2]
        er = []
        if kc > 0:
            mu, A, _ = fit_eraser(xc[tr_c], zc[tr_c], kc)
            er.append((mu, A))
        if kh > 0:
            xh_ = apply(xh, er)
            mu, A, _ = fit_eraser(xh_[tr_h], zh[tr_h], kh)
            er.append((mu, A))
        r = dict(concept=job[2] if len(job) > 2 and max(kc, kh) else ('none' if not max(kc, kh) else 'pair'),
                 k=max(kc, kh), kc=kc, kh=kh)
        for name, (x, y) in tests.items():
            r[name] = fast_ap(apply(x, er).float().numpy(), y)
        r['I_content'] = B.content_bias(apply(xc, er).float(), t['content'], repeats=args.repeats)['I']
        r['I_char'] = B.char_bias(apply(xh, er).float(), t['char'], [c for _, c in t['char_items']],
                                  repeats=args.repeats)['I']
        print(json.dumps(r), flush=True)
        out.append(r)
    tag = '' if args.test_feats == 'eval/feats' else '_' + Path(args.test_feats).parent.name
    tag += '_pairs' if args.pairs else ''
    (Path(args.exp) / 'eval' / f'posthoc_erase{tag}.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
