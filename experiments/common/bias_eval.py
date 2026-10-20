"""CSIP 的 content bias / character bias (Lark 文档与 HF 卡片的定义: I = 1 - L(f(x), y) / L(ε, y), ε ~ N(μ, σ)).
探针、loss、高斯基线直接复用 evaluate/content_bias.py 和 evaluate/char_bias.py, 特征全在内存里 (不写逐图 .pt).

与原评测的差异 (原数据 /nfs/danbooru-wdtagger-v4-w640-ws-30k 及其 tags_map.csv 已丢失):
- content: 图片为原始划分 test_split (held-out 画师) 中固定抽样的 N 张; 标签为 wd-vit-tagger-v3 在这些图上
  出现频率最高的 320 个 general 标签 (原来是 ws-30k 数据集自带的 320 个标签), 目标为 tagger 概率
- character: zerochan 500 角色 x 10 张 (zc500x10), 目标为 AnimeCCIP 768 维特征, 按角色分层划分 val 0.2

python experiments/common/bias_eval.py --models dghstmp/AnimeCSIP dghstmp/AnimeCSIP-v2 --out results/bias_hf.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # CxIP 根目录
from pair_eval import extract, load_items  # noqa: E402
from evaluate import content_bias as cb  # noqa: E402
from evaluate import char_bias as chb  # noqa: E402

R = Path('/data/neko/cxip_repro')
CONTENT_SPEC = f'json:{R}/data/danbooru-artist-split-0918/test_split_avail.json:{R}/data/danbooru-w640-ws-full-artist/train'
CHAR_SPEC = f'folder:{R}/data/zerochan_test_c500x10'


class TaggerDataset(Dataset):
    """wd v3 tagger 的预处理: 白底 pad 成方图 -> timm transform -> RGB 转 BGR (同 scripts/tagger/img2prob.py)"""

    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img = Image.open(self.items[i][0])
        img = img.convert('RGBA') if img.mode in ('RGBA', 'LA', 'P') else img.convert('RGB')
        if img.mode == 'RGBA':
            canvas = Image.new('RGBA', img.size, (255, 255, 255))
            canvas.alpha_composite(img)
            img = canvas.convert('RGB')
        edge = max(img.size)
        canvas = Image.new('RGB', (edge, edge), (255, 255, 255))
        canvas.paste(img, ((edge - img.size[0]) // 2, (edge - img.size[1]) // 2))
        return self.transform(canvas)[[2, 1, 0]]


@torch.no_grad()
def content_targets(items, device, n_tags=320, bs=64):
    repo = 'SmilingWolf/wd-vit-tagger-v3'
    model = timm.create_model(f'hf-hub:{repo}', pretrained=True).to(device).eval()
    transform = timm.data.create_transform(**timm.data.resolve_data_config(model.pretrained_cfg, model=model))
    probs = []
    for x in DataLoader(TaggerDataset(items, transform), batch_size=bs, num_workers=16):
        with torch.autocast('cuda', dtype=torch.float16):
            probs.append(torch.sigmoid(model(x.to(device))).float().cpu())
    probs = torch.cat(probs)
    from huggingface_hub import hf_hub_download
    tags = pd.read_csv(hf_hub_download(repo, 'selected_tags.csv'))
    general = np.where(tags['category'].values == 0)[0]
    freq = (probs[:, general] > 0.35).float().mean(0)
    keep = general[freq.argsort(descending=True)[:n_tags].numpy()]
    return probs[:, keep], tags['name'].values[keep].tolist()


def load_or_make_targets(cache, n_content, device, seed=0):
    cache = Path(cache)
    if cache.exists():
        d = torch.load(cache)
        return d
    content_items = load_items(CONTENT_SPEC)
    content_items = random.Random(seed).sample(content_items, n_content)
    char_items = load_items(CHAR_SPEC)
    probs, tag_names = content_targets(content_items, device)
    ccip = AutoModel.from_pretrained('dghstmp/AnimeCCIP', trust_remote_code=True).to(device).eval()
    ccip_proc = AutoImageProcessor.from_pretrained('dghstmp/AnimeCCIP', trust_remote_code=True)
    char_feats = extract(ccip, ccip_proc, char_items, device).float().cpu()
    d = dict(content_items=content_items, content=probs, tag_names=tag_names, char_items=char_items, char=char_feats)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(d, cache)
    return d


def run_content(style, target, device, args):
    ds = TensorDataset(style, target)
    train_ds, val_ds, test_ds = cb.split_dataset(ds, 0.1, 0.1, args.seed)
    loaders = [cb.build_dataloader(d, 128, s, 0, device) for d, s in ((train_ds, True), (val_ds, False), (test_ds, False))]
    cb.set_random_seed(args.seed)
    real = cb.train_probe(cb.ProbeMLP(style.shape[1], target.shape[1]).to(device), *loaders, epochs=args.content_epochs,
                          device=device, lr=1e-3)
    mean, std = cb.style_stats(train_ds)
    base = []
    for r in range(args.repeats):
        b_loaders = [cb.build_dataloader(cb.GaussianStyleDataset(d, mean, std), 128, s, 0, device)
                     for d, s in ((train_ds, True), (val_ds, False), (test_ds, False))]
        base.append(cb.train_probe(cb.ProbeMLP(style.shape[1], target.shape[1]).to(device), *b_loaders,
                                   epochs=args.content_epochs, device=device, lr=1e-3).test_loss)
    return dict(I=1 - real.test_loss / float(np.mean(base)), real=real.test_loss, base=float(np.mean(base)))


def run_char(style, target, labels, device, args):
    rng = random.Random(args.seed)
    groups = {}
    for i, c in enumerate(labels):
        groups.setdefault(c, []).append(i)
    train_idx, val_idx = [], []
    for idx in groups.values():
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * 0.2))
        val_idx += idx[:n_val]
        train_idx += idx[n_val:]
    train_ds = TensorDataset(style[train_idx], target[train_idx])
    val_ds = TensorDataset(style[val_idx], target[val_idx])
    make = lambda: chb.RegressProbe(style.shape[1], target.shape[1], 512).to(device)  # noqa: E731
    loaders = [chb.build_dataloader(train_ds, 512, True, 0, device)] + [chb.build_dataloader(val_ds, 512, False, 0, device)] * 2
    chb.set_random_seed(args.seed)
    real = chb.train_probe(make(), *loaders, epochs=args.char_epochs, device=device, lr=2e-4, method='rmse')
    mean, std = chb.style_stats(train_ds)
    base = []
    for r in range(args.repeats):
        g_val = chb.GaussianStyleDataset(val_ds, mean, std)
        b_loaders = [chb.build_dataloader(chb.GaussianStyleDataset(train_ds, mean, std), 512, True, 0, device)] + \
                    [chb.build_dataloader(g_val, 512, False, 0, device)] * 2
        base.append(chb.train_probe(make(), *b_loaders, epochs=args.char_epochs, device=device, lr=2e-4,
                                    method='rmse').test_loss)
    return dict(I=1 - real.test_loss / float(np.mean(base)), real=real.test_loss, base=float(np.mean(base)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--cache', default=str(R / 'results/bias_targets.pt'))
    parser.add_argument('--n-content', type=int, default=10000)
    parser.add_argument('--content-epochs', type=int, default=100)
    parser.add_argument('--char-epochs', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    device = torch.device('cuda')

    t = load_or_make_targets(args.cache, args.n_content, device)
    results = {}
    for name in args.models:
        model = AutoModel.from_pretrained(name, trust_remote_code=True).to(device).eval()
        proc = AutoImageProcessor.from_pretrained(name, trust_remote_code=True)
        sc = extract(model, proc, t['content_items'], device).float().cpu()
        sh = extract(model, proc, t['char_items'], device).float().cpu()
        del model
        results[name] = dict(content=run_content(sc, t['content'], device, args),
                             char=run_char(sh, t['char'], [c for _, c in t['char_items']], device, args))
        print(name, json.dumps(results[name]), flush=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=1)
    print('| model | content I_c | char I_char |\n|-|-|-|')
    for name, r in results.items():
        print(f"| {name} | {r['content']['I']:.4f} | {r['char']['I']:.4f} |")


if __name__ == '__main__':
    main()
