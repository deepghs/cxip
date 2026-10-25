"""偏置探针 (Lark 文档 / HF 卡片的定义): I = 1 - L(f(x), y) / L(ε, y), ε ~ N(μ, σ) (μ, σ 为训练集输入的逐维统计).
I 越接近 0 说明输入向量里越没有关于 y 的可学信息 (< 0.05 无可辨识信息, < 0.1 微量, >= 0.1 会干扰).

- content_bias (I_c, CSIP): 输入 = 画风向量, y = tagger 的 320 个 general 标签概率; ProbeMLP(D→256→T),
  BCEWithLogits, AdamW lr 1e-3, bs 128, 100 epoch, random_split 0.8/0.1/0.1 (torch.Generator 种子), 取 val 最优.
- char_bias (I_char, CSIP): 输入 = 画风向量, y = 参考 CCIP 向量 (HF AnimeCCIP 768 维); MLP(D→512→768),
  L = sqrt(MSE), AdamW lr 2e-4, bs 512, 20 epoch, 按角色分层取 20% 作 val (= test).
- style_bias (I_style, CCIP, 新增): 与 I_char 对称 —— 输入 = CCIP 向量, y = 参考 CSIP 向量 (默认 HF AnimeCSIP-v2.1),
  L = sqrt(MSE), 同 I_char 的超参, 按画师分层取 20% 作 val (= test). 衡量 "CCIP 向量里能读出多少画风".
  form='ce' 为分类版: 从 CCIP 向量预测画师 id (交叉熵), 基线同为高斯输入 (相当于只能学到类先验).
基线重复 repeats 次 (默认 3, 同 experiments/common/bias_eval.py) 取 loss 均值.

实现: 不用 DataLoader, 直接张量切批; exact=True 时逐步复现 DataLoader(num_workers=0) 的全局 RNG 消耗
(每次 iter 抽 base_seed, shuffle 时 RandomSampler 再抽一个种子做 randperm, 高斯数据集按取样顺序逐条 randn),
因此与 bias_eval.run_content / run_char (CPU) 的结果逐位一致 (见 v3/tests/test_eval_bias.py), 但快数倍.
"""
from __future__ import annotations

import copy
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ProbeMLP(nn.Module):
    """与 evaluate/content_bias.ProbeMLP 相同 (含初始化及其 RNG 消耗顺序)"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.layer2(self.activation(self.layer1(x)))


def rmse_loss(pred, target):
    return torch.sqrt(nn.functional.mse_loss(pred, target))


class _Data:
    """一个划分: 真实输入 x 或高斯输入 (mean, std), 目标 y"""

    def __init__(self, x, y, gauss=None):
        self.x, self.y, self.gauss = x, y, gauss

    def __len__(self):
        return len(self.y)

    def batch(self, idx):
        if self.gauss is None:
            return self.x[idx], self.y[idx]
        mean, std = self.gauss
        # 与 GaussianStyleDataset 逐条 randn_like(mean) 的 RNG 序列相同 (D % 16 == 0 时一次生成等价, 否则逐条)
        if mean.numel() % 16 == 0:
            eps = torch.randn(len(idx), mean.numel(), dtype=mean.dtype)
        else:
            eps = torch.stack([torch.randn_like(mean) for _ in range(len(idx))])
        return eps * std + mean, self.y[idx]


def _batches(n, bs, shuffle):
    """复现 DataLoader(num_workers=0) 的一次迭代的 RNG 消耗, 返回批下标列表"""
    torch.empty((), dtype=torch.int64).random_()                  # _BaseDataLoaderIter._base_seed
    if shuffle:
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        g = torch.Generator()
        g.manual_seed(seed)
        order = torch.randperm(n, generator=g)
    else:
        order = torch.arange(n)
    return [order[i:i + bs] for i in range(0, n, bs)]


_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 随机数仍在 CPU 上生成 (RNG 序列不变), 只把计算放到 GPU


def _run_epoch(model, data, bs, loss_fn, opt=None):
    train = opt is not None
    model.train(train)
    total, count = 0.0, 0
    with torch.set_grad_enabled(train):
        for idx in _batches(len(data), bs, shuffle=train):
            x, y = data.batch(idx)
            x, y = x.float().to(_DEVICE, non_blocking=True), y.float().to(_DEVICE, non_blocking=True)
            if train:
                opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            if train:
                loss.backward()
                opt.step()
            total += loss.item() * len(idx)
            count += len(idx)
    return total / max(count, 1)


def train_probe(make_model, train, val, test, epochs, lr, bs, loss_fn) -> dict:
    """同 evaluate/content_bias.train_probe: 每 epoch 训练 + 验证, 取 val 最优的权重在 test 上算 loss"""
    model = make_model().to(_DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best_state = copy.deepcopy(model.state_dict())
    best_val, train_loss = float('inf'), float('inf')
    for _ in range(epochs):
        train_loss = _run_epoch(model, train, bs, loss_fn, opt)
        v = _run_epoch(model, val, bs, loss_fn)
        if v < best_val:
            best_val = v
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return dict(train_loss=train_loss, val_loss=best_val, test_loss=_run_epoch(model, test, bs, loss_fn))


def _probe_with_baseline(make_model, splits, epochs, lr, bs, loss_fn, repeats, seed, val_is_test):
    """splits = (x_tr, y_tr, x_va, y_va, x_te, y_te); 先 set_random_seed(seed) 训真实探针, 再训 repeats 个高斯基线"""
    x_tr, y_tr, x_va, y_va, x_te, y_te = splits
    set_random_seed(seed)
    tr, va = _Data(x_tr, y_tr), _Data(x_va, y_va)
    te = va if val_is_test else _Data(x_te, y_te)
    real = train_probe(make_model, tr, va, te, epochs, lr, bs, loss_fn)
    g = (x_tr.float().mean(0), x_tr.float().std(0))
    base = []
    for _ in range(repeats):
        gva = _Data(None, y_va, g)
        gte = gva if val_is_test else _Data(None, y_te, g)
        base.append(train_probe(make_model, _Data(None, y_tr, g), gva, gte, epochs, lr, bs, loss_fn)['test_loss'])
    b = float(np.mean(base))
    return dict(I=1 - real['test_loss'] / b, real=real['test_loss'], base=b, base_std=float(np.std(base)),
                base_runs=base, real_val=real['val_loss'], n_train=int(len(y_tr)), n_val=int(len(y_va)),
                n_test=int(len(y_te)))


def _as_tensor(a):
    return torch.as_tensor(np.asarray(a)) if not torch.is_tensor(a) else a


# ----------------------------------------------------------------------------------------------- content I_c
def content_bias(style, target, seed=42, epochs=100, repeats=3, lr=1e-3, bs=128, hidden=256,
                 val_ratio=0.1, test_ratio=0.1) -> dict:
    """I_c: style (N, D) 画风向量, target (N, T) tagger 概率. 划分同 evaluate/content_bias.split_dataset"""
    t0 = time.time()
    style, target = _as_tensor(style).float(), _as_tensor(target).float()
    n = len(style)
    n_val, n_test = int(n * val_ratio), int(n * test_ratio)
    n_tr = n - n_val - n_test
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    i_tr, i_va, i_te = perm[:n_tr], perm[n_tr:n_tr + n_val], perm[n_tr + n_val:]
    splits = (style[i_tr], target[i_tr], style[i_va], target[i_va], style[i_te], target[i_te])
    out = _probe_with_baseline(lambda: ProbeMLP(style.shape[1], target.shape[1], hidden), splits, epochs, lr, bs,
                               nn.BCEWithLogitsLoss(), repeats, seed, val_is_test=False)
    out.update(kind='content', seconds=time.time() - t0)
    return out


# ----------------------------------------------------------------------------------------------- regression form
def stratified_split(labels, val_ratio=0.2, seed=42):
    """同 bias_eval.run_char: 按标签首次出现顺序分组, random.Random(seed) 组内打乱, 每组 max(1, int(0.2 n)) 个作 val"""
    rng = random.Random(seed)
    groups = {}
    for i, c in enumerate(list(labels)):
        groups.setdefault(c, []).append(i)
    tr, va = [], []
    for idx in groups.values():
        rng.shuffle(idx)
        k = max(1, int(len(idx) * val_ratio))
        va += idx[:k]
        tr += idx[k:]
    return tr, va


def regression_bias(inputs, targets, groups, seed=42, epochs=20, repeats=3, lr=2e-4, bs=512, hidden=512,
                    val_ratio=0.2) -> dict:
    """I = 1 - RMSE(probe(inputs), targets) / RMSE(probe(ε), targets); 按 groups 分层划分, val 兼作 test"""
    t0 = time.time()
    x, y = _as_tensor(inputs).float(), _as_tensor(targets).float()
    tr, va = stratified_split(groups, val_ratio, seed)
    tr, va = torch.as_tensor(tr), torch.as_tensor(va)
    splits = (x[tr], y[tr], x[va], y[va], x[va], y[va])
    out = _probe_with_baseline(lambda: ProbeMLP(x.shape[1], y.shape[1], hidden), splits, epochs, lr, bs, rmse_loss,
                               repeats, seed, val_is_test=True)
    out['seconds'] = time.time() - t0
    return out


def char_bias(style, ccip_target, char_labels, **kw) -> dict:
    """I_char (CSIP): 画风向量 → 参考 CCIP 向量, 按角色分层"""
    out = regression_bias(style, ccip_target, char_labels, **kw)
    out['kind'] = 'char'
    return out


def style_bias(ccip, csip_target=None, artist_labels=None, form='regress', **kw) -> dict:
    """I_style (CCIP): form='regress': CCIP 向量 → 参考 CSIP 向量 (sqrt(MSE)), 按画师分层;
    form='ce': CCIP 向量 → 画师 id 的分类探针 (交叉熵)"""
    if form == 'regress':
        out = regression_bias(ccip, csip_target, artist_labels, **kw)
    elif form == 'ce':
        out = classification_bias(ccip, artist_labels, **kw)
    else:
        raise ValueError(form)
    out.update(kind='style', form=form)
    return out


def classification_bias(inputs, labels, seed=42, epochs=20, repeats=3, lr=2e-4, bs=512, hidden=512,
                        val_ratio=0.2) -> dict:
    """分类版: I = 1 - CE(probe(x), y) / CE(probe(ε), y); 高斯基线只能学到类先验, 所以 CE_base ≈ 先验熵"""
    t0 = time.time()
    x = _as_tensor(inputs).float()
    _, y = np.unique(np.asarray(labels), return_inverse=True)
    y = torch.as_tensor(y)
    tr, va = stratified_split(list(np.asarray(labels)), val_ratio, seed)
    tr, va = torch.as_tensor(tr), torch.as_tensor(va)
    n_cls = int(y.max()) + 1
    splits = (x[tr], y[tr], x[va], y[va], x[va], y[va])
    ce = lambda logits, t: nn.functional.cross_entropy(logits, t.long())  # noqa: E731
    out = _probe_with_baseline(lambda: ProbeMLP(x.shape[1], n_cls, hidden), splits, epochs, lr, bs, ce,
                               repeats, seed, val_is_test=True)
    out.update(seconds=time.time() - t0, n_classes=n_cls)
    return out
