"""发布 README 的详细分析数据 (CPU, 基于评测缓存特征), 结果缓存到 /data/neko/cxip_v3/release/_analysis/<task>.npz。

每个 (模型, 测试集):
  hist_pos / hist_neg   全部 i<j 图片对的 cos 直方图 (2^20 箱; 分数 = scale·cos + bias)
  ood_mx / ood_unk      开集协议 (CCIP: N500_X500, CSIP: 一半已知一半未知, 各 20 次) 的查询最大类分数与是否未知
                        (A=1 与 A=5 参考图), ood_tau: 每次重复的最优阈值
  fewshot               闭集识别 top-1 vs 每类参考图数 A (固定类集合: 张数 ≥ 11 的类, 10 次重复)
  matry_*               v3 Matryoshka 前缀的 PR-AUC
  umap_*                随机 24 类的 2D UMAP (聚类可视化, 新旧模型同一批图)
用法: python -m v3.scripts.release_analysis {ccip,csip}
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from v3.eval import pairwise, zeroshot
from v3.eval.runner import load_set

EX = Path('/data/neko/cxip_v3/exps')
EV = Path('/data/neko/cxip_v3/eval')
OUT = Path('/data/neko/cxip_v3/release/_analysis')
ERASE = EX / 'csip_p2/dft_r640_tag001_x2/eval/posthoc_kc3_kh4.pt'


def J(p):
    return json.loads(Path(p).read_text())


def models(task):
    """[(key, {set: (feats_dir, scale, bias)}, erase)]; 最后一个是 v3。"""
    if task == 'ccip':
        hf = J(EV / 'hf_ccip/suite.json')['datasets']['openset_full']
        v3 = J(EX / 'final/ccip_teacher_v1/eval/suite.json')['datasets']['openset_full']
        return [('AnimeCCIP', {'openset_full': (EV / 'hf_ccip/feats', hf['scale'], hf['bias'])}, None),
                ('AnimeCCIP-v3', {'openset_full': (EX / 'final/ccip_teacher_v1/eval/feats', v3['scale'], v3['bias'])}, None)]
    out = []
    for name, f in [('AnimeCSIP', 'hf_AnimeCSIP'), ('AnimeCSIP-v2', 'hf_AnimeCSIP-v2'), ('AnimeCSIP-v2.1', 'hf_AnimeCSIP-v2.1'),
                    ('AnimeCSIP-v3', 'v3_x2_erased')]:
        r = J(EV / f'csip_full/{f}.json')
        out.append((name, {s: (Path(r['feats_dir']), r['sets'][s]['scale'], r['sets'][s]['bias']) for s in r['sets']},
                    ERASE if r.get('erase') else None))
    return out


def load_feats(feats_dir, name, df, erase):
    d = np.load(Path(feats_dir) / f'{name}.npz')
    pos = {str(k): i for i, k in enumerate(d['sample_key'])}
    f = torch.from_numpy(d['feats'][[pos[str(k)] for k in df['sample_key']]].astype(np.float32))
    if erase is not None:
        e = torch.load(erase, weights_only=True)
        f = f @ e['M'] + e['b']
    return torch.nn.functional.normalize(f, dim=-1)


def ood_runs(f, y, protocol, scale, bias, A):
    mx, unk, tau = [], [], []
    for r in protocol:
        known = np.asarray(r['known'])
        anchors = [np.asarray(a)[:A] for a in r['anchors']]
        q = np.isin(y, known) | np.isin(y, np.asarray(r['unknown']))
        q[np.concatenate(anchors)] = False
        qi = np.nonzero(q)[0]
        S = zeroshot.class_scores(f, qi, anchors, scale, bias)
        m, u = S.max(1), ~np.isin(y[qi], known)
        mx.append(m)
        unk.append(u)
        tau.append(zeroshot.best_tau_f1(u, m)[1])
    return np.concatenate(mx).astype(np.float32), np.concatenate(unk), np.asarray(tau)


def ccip_protocol(df, y):
    """ccip_openset_v1/ood_protocol.json 的 N500_X500 (与正式评测一致), 锚点 key → 下标。"""
    proto = J(EV / 'ccip_openset_v1/ood_protocol.json')
    p = next(p for p in proto['protocols'] if p['N'] == 500 and p['X'] == 500)
    pos = {str(k): i for i, k in enumerate(df['sample_key'])}
    return [dict(known=r['known'], unknown=r['unknown'],
                 anchors=[np.array([pos[k] for k in r['anchors'][str(c)]]) for c in r['known']]) for r in p['repeats']]


def fewshot(f, y, scale, bias, A_list=(1, 2, 3, 5, 10), repeats=10, seed=0):
    classes, counts = np.unique(y, return_counts=True)
    classes = classes[counts > max(A_list)]
    rng = np.random.default_rng(seed)
    acc = {A: [] for A in A_list}
    for _ in range(repeats):
        perm = [rng.permutation(np.nonzero(y == c)[0]) for c in classes]
        # 查询 = 每类第 max(A) 张以后的图, 各 A 共用同一批查询
        qi = np.concatenate([p[max(A_list):] for p in perm])
        qy = np.searchsorted(classes, y[qi])
        for A in A_list:
            S = zeroshot.class_scores(f, qi, [p[:A] for p in perm], scale, bias)
            acc[A].append(float((S.argmax(1) == qy).mean()))
    return np.array([[A, np.mean(acc[A]), np.std(acc[A])] for A in A_list]), len(classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('task', choices=['ccip', 'csip'])
    args = ap.parse_args()
    torch.set_num_threads(64)
    OUT.mkdir(parents=True, exist_ok=True)
    ms = models(args.task)
    sets = list(ms[-1][1])
    res, meta = {}, dict(models=[m[0] for m in ms], sets=sets, scale={}, bias={})
    for s in sets:
        df, _ = load_set(s)
        y = df['label'].values
        n_cls = len(np.unique(y))
        protocol = ccip_protocol(df, y) if args.task == 'ccip' else \
            zeroshot.random_protocol(y, n_cls // 2, n_cls - n_cls // 2, A_max=5, repeats=20, seed=0)
        # 聚类可视化: 随机 24 类, 每类最多 80 张
        rng = np.random.default_rng(1)
        cls, cnt = np.unique(y, return_counts=True)
        pick = rng.choice(cls[cnt >= 20], 24, replace=False)
        sub = np.concatenate([rng.permutation(np.nonzero(y == c)[0])[:80] for c in pick])
        res[f'{s}/umap_y'] = np.searchsorted(np.sort(pick), y[sub])
        for name, by_set, erase in ms:
            fd, scale, bias = by_set[s]
            k = f'{s}/{name}'
            meta['scale'][k], meta['bias'][k] = scale, bias
            f = load_feats(fd, s, df, erase)
            h = pairwise.pair_histogram(f, y, scale, bias)
            res[f'{k}/hist_pos'], res[f'{k}/hist_neg'] = h.pos, h.neg
            for A in (1, 5):
                mx, unk, tau = ood_runs(f, y, protocol, scale, bias, A)
                res[f'{k}/ood{A}_mx'], res[f'{k}/ood{A}_unk'], res[f'{k}/ood{A}_tau'] = mx, unk, tau
            res[f'{k}/fewshot'], meta['fewshot_classes'] = fewshot(f, y, scale, bias)
            if name == ms[-1][0]:
                for dim in (512, 256, 128):
                    fp = torch.nn.functional.normalize(f[:, :dim], dim=-1)
                    res[f'{k}/matry_{dim}'] = pairwise.metrics_from_hist(
                        pairwise.pair_histogram(fp, y, scale, bias), curve_points=0)['pr_auc']
            import umap
            res[f'{k}/umap'] = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=0).fit_transform(
                f[sub].numpy()).astype(np.float32)
            print(k, 'pr_auc', round(pairwise.metrics_from_hist(h, curve_points=0)['pr_auc'], 4),
                  'fewshot', np.round(res[f'{k}/fewshot'][:, 1], 3).tolist(), flush=True)
    np.savez_compressed(OUT / f'{args.task}.npz', **res)
    (OUT / f'{args.task}.json').write_text(json.dumps(meta, indent=1, default=float))


if __name__ == '__main__':
    main()
