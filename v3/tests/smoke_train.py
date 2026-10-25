"""训练器冒烟测试: 用合成数据替换数据层, 跑通 DDP + LP→FT 切换 + patch drop 关闭 + 评测 + 存档。

CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node 2 -m v3.tests.smoke_train
"""
import sys

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

import v3.train_ccip as T

N_CLASSES, RES = 40, 224


class Synthetic(Dataset):
    def __init__(self, labels):
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(int(self.labels[i]))
        base = torch.randn(3, RES, RES, generator=g)
        return dict(image=base + 0.3 * torch.randn(3, RES, RES), label=int(self.labels[i]))


def fake_df(dc):
    labels = np.repeat(np.arange(N_CLASSES), 8)
    return pd.DataFrame(dict(label=labels)), N_CLASSES, None


def fake_loader(dc, df, group, seed, steps):
    rank = dist.get_rank() if dist.is_initialized() else 0
    rng = np.random.RandomState(seed + rank)

    def batches():
        for _ in range(steps):
            cls = rng.choice(N_CLASSES, dc['P'] // max(1, T.dist.get_world_size() if dist.is_initialized() else 1), replace=False)
            yield np.repeat(cls * 8, dc['K']) + np.tile(np.arange(dc['K']), len(cls))
    return DataLoader(Synthetic(df['label'].values), batch_sampler=list(batches()), collate_fn=T.collate, num_workers=2)


def fake_eval(ec, res):
    labels = np.repeat(np.arange(N_CLASSES), 3)
    return [('synthetic', Synthetic(labels), labels)]


T.load_train_df, T.make_train_loader, T.load_eval_sets = fake_df, fake_loader, fake_eval
sys.argv = ['x', '--config', 'v3/configs/ccip_p1_base.py',
            'exp_dir=/data/neko/cxip_v3/exps/_smoke', 'data.res=224', 'data.P=16', 'data.K=4',
            'optim.total_samples=1280', 'optim.lp_samples=256', 'optim.warmup_steps=2',
            'eval.every_samples=640', 'log_every=2', 'model.patch_drop_off_frac=0.3',
            'save_every_samples=1280']
T.main()
