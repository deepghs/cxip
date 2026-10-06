"""
复现 HF 模型卡片上的评测: 测试集内全部图片两两配对, score = exp(logit_scale) * cos + logit_bias,
同类为正样本, 报告 PR 曲线 AUC (卡片上的 "PR Curve (AUC = x)")、AP、ROC AUC、best F1 及阈值。

模型用 transformers AutoModel 加载 (HF repo 或 transfs/trans_*.py 导出的目录), 与卡片的使用方式一致。

数据集 (--data):
  folder:<dir>                      dir/<class>/<img>
  json:<label_json>:<img_root>      {"xxx.webp": "class", ...}

python repro/eval/pair_eval.py --model dghstmp/AnimeCSIP-v2.1 --data json:.../test_split_100.json:.../train --out results/x.json
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel


def load_items(spec):
    kind, rest = spec.split(':', 1)
    if kind == 'folder':
        root = Path(rest)
        items = [(str(p), p.parent.name) for p in sorted(root.glob('*/*'))
                 if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')]
    elif kind == 'json':
        label_file, img_root = rest.split(':')
        with open(label_file) as f:
            labels = json.load(f)
        items = [(os.path.join(img_root, k), v) for k, v in sorted(labels.items())]
    else:
        raise ValueError(spec)
    return items


class ImgDataset(Dataset):
    def __init__(self, items, processor):
        self.items, self.processor = items, processor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img = Image.open(self.items[i][0])
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGBA')
            bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img)
        img = img.convert('RGB')
        return torch.tensor(np.asarray(self.processor(img)['pixel_values'][0]))


@torch.no_grad()
def extract(model, processor, items, device, bs=128, workers=16):
    dl = DataLoader(ImgDataset(items, processor), batch_size=bs, num_workers=workers)
    feats = []
    for x in dl:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(x.to(device))
        feats.append(out['features'].float())
    return torch.cat(feats)


def pr_metrics(scores, labels):
    order = np.argsort(-scores, kind='stable')
    s, y = scores[order], labels[order]
    # 同分只保留最后一个位置 (与 sklearn precision_recall_curve 一致)
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    tp = np.cumsum(y)[last].astype(np.float64)
    fp = (last + 1) - tp
    P, R = tp / (tp + fp), tp / y.sum()
    f1 = 2 * P * R / np.maximum(P + R, 1e-12)
    best = int(np.argmax(f1))
    ap = float(np.sum(np.diff(np.r_[0, R]) * P))
    pr_auc = float(np.trapz(np.r_[1.0, P], np.r_[0.0, R]))
    fpr, tpr = fp / (len(y) - y.sum()), R
    roc = float(np.trapz(np.r_[0, tpr], np.r_[0, fpr]))
    return dict(pr_auc=pr_auc, ap=ap, roc_auc=roc, best_f1=float(f1[best]), best_threshold=float(s[last][best]),
                precision_at_best=float(P[best]), recall_at_best=float(R[best]))


def pair_scores(feats, cls, logit_scale, logit_bias, device):
    f = torch.nn.functional.normalize(feats.to(device), dim=-1)
    sim = (f @ f.T) * logit_scale + logit_bias
    iu = torch.triu_indices(len(f), len(f), offset=1, device=device)
    c = torch.as_tensor(cls, device=device)
    return sim[iu[0], iu[1]].cpu().numpy(), (c[iu[0]] == c[iu[1]]).cpu().numpy()


def zero_shot(feats, cls, n_ref=1, seed=0, device='cuda'):
    '''每类随机取 n_ref 张做参考 (取均值), 其余图片做最近类分类, 返回 top1/top5'''
    rng = np.random.default_rng(seed)
    cls = np.asarray(cls)
    f = torch.nn.functional.normalize(feats.to(device), dim=-1)
    refs, query = [], np.ones(len(cls), bool)
    classes = np.unique(cls)
    for c in classes:
        idx = rng.choice(np.nonzero(cls == c)[0], n_ref, replace=False)
        query[idx] = False
        refs.append(f[torch.as_tensor(idx, device=device)].mean(0))
    refs = torch.nn.functional.normalize(torch.stack(refs), dim=-1)
    sim = f[torch.as_tensor(np.nonzero(query)[0], device=device)] @ refs.T
    target = torch.as_tensor(np.searchsorted(classes, cls[query]), device=device)
    top5 = sim.topk(min(5, len(classes)), dim=1).indices
    return dict(top1=float((top5[:, 0] == target).float().mean()), top5=float((top5 == target[:, None]).any(1).float().mean()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True, nargs='+', help='name=spec, 见文件头')
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--save-feats', default=None, help='目录, 保存特征 npz')
    args = parser.parse_args()

    model = AutoModel.from_pretrained(args.model, trust_remote_code=True).to(args.device).eval()
    processor = AutoImageProcessor.from_pretrained(args.model, trust_remote_code=True)
    logit_scale = float(model.sim.logit_scale.exp())
    logit_bias = float(model.sim.logit_bias)

    results = dict(model=args.model, logit_scale=logit_scale, logit_bias=logit_bias, datasets={})
    for spec in args.data:
        name, spec = spec.split('=', 1)
        items = load_items(spec)
        names = sorted({c for _, c in items})
        cls = np.searchsorted(names, [c for _, c in items])
        feats = extract(model, processor, items, args.device, args.bs)
        if args.save_feats:
            Path(args.save_feats).mkdir(parents=True, exist_ok=True)
            np.savez(f'{args.save_feats}/{name}.npz', feats=feats.cpu().numpy(), cls=cls,
                     files=np.array([p for p, _ in items]), names=np.array(names))
        scores, labels = pair_scores(feats, cls, logit_scale, logit_bias, args.device)
        r = dict(n_images=len(items), n_classes=len(names), n_pos_pairs=int(labels.sum()), n_pairs=len(labels))
        r.update(pr_metrics(scores, labels))
        for k in (1, 5):
            if np.bincount(cls).min() > k:
                r[f'zero_shot_ref{k}'] = zero_shot(feats, cls, k, device=args.device)
        results['datasets'][name] = r
        print(name, json.dumps(r))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
