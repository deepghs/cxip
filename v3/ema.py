"""权重 EMA: 只跟踪会被训练的参数 (fp32 影子), 评测/存档时把影子换进模型, 结束后换回。"""
from contextlib import contextmanager

import torch
from torch import nn


class ModelEMA:
    """shadow ← d·shadow + (1-d)·param, d_t = min(decay, (1+t)/(10+t)) (早期快速跟上)。

    params: 需要跟踪的参数列表 (LP 阶段冻结的骨干参数也可以放进来, 它们不变, EMA 就是恒等)。
    buffers (BN running stats 等) 不做 EMA, 评测时直接用当前值。
    """

    def __init__(self, params, decay: float):
        self.params = [p for p in params]
        self.decay = decay
        self.shadow = [p.detach().float().clone() for p in self.params]
        self.updates = 0

    @torch.no_grad()
    def update(self):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        torch._foreach_lerp_(self.shadow, [p.detach().float() for p in self.params], 1 - d)

    @contextmanager
    def swapped(self):
        """with ema.swapped(): 模型里是 EMA 权重; 原权重暂存在 CPU。"""
        backup = [p.detach().to('cpu', copy=True) for p in self.params]
        with torch.no_grad():
            for p, s in zip(self.params, self.shadow):
                p.copy_(s)
        try:
            yield
        finally:
            with torch.no_grad():
                for p, b in zip(self.params, backup):
                    p.copy_(b.to(p.device, non_blocking=True))
