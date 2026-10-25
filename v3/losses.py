"""CxIP v3 损失: Sub-center ArcFace (动态 margin) + 跨卡 SigLIP + Matryoshka 包装。"""
import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


def dynamic_margins(class_counts, m_min: float = 0.05, m_max: float = 0.5, power: float = -0.25) -> torch.Tensor:
    """m_c = a·n_c^power + b, 线性映射到 [m_min, m_max]: 图最少的类 margin 最大 (GLR 2020 3rd)。"""
    v = np.asarray(class_counts, dtype=np.float64).clip(min=1) ** power
    lo, hi = v.min(), v.max()
    m = m_min + (v - lo) / max(hi - lo, 1e-12) * (m_max - m_min)
    return torch.tensor(m, dtype=torch.float32)


class SubCenterArcFace(nn.Module):
    def __init__(self, dim: int, num_classes: int, k: int = 3, scale: float = 30.0,
                 margins: Optional[torch.Tensor] = None, margin: float = 0.3):
        super().__init__()
        self.k = k
        self.scale = scale
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.randn(num_classes * k, dim) * dim ** -0.5)
        self.register_buffer('margins', margins if margins is not None else torch.full((num_classes,), margin))
        self.register_buffer('seen', torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def imprint(self, emb_all, y_all):
        """weight imprinting: 类首次出现时用 (全局 batch 内) 归一化均值初始化其子中心, 子中心 j 再混入该类第 j 个样本。

        短预算下每类只被访问几次, 随机初始化的类中心几乎学不动; 输入必须是各卡一致的 all-gather 结果,
        且不用随机数, 保证 DDP 各副本相同。
        """
        new = ~self.seen[y_all]
        if not new.any():
            return
        ys, zs = y_all[new], F.normalize(emb_all[new].float(), dim=-1)
        uniq, inv = ys.unique(return_inverse=True)
        mean = F.normalize(torch.zeros(len(uniq), zs.shape[1], device=zs.device).index_add_(0, inv, zs), dim=-1)
        order = torch.argsort(inv, stable=True)
        counts = torch.bincount(inv, minlength=len(uniq))
        starts = torch.cumsum(counts, 0) - counts
        j = torch.arange(self.k, device=zs.device)
        exemplar = zs[order][starts[:, None] + j[None] % counts[:, None]]  # (U, k, D)
        w = F.normalize(mean[:, None] + 0.5 * exemplar, dim=-1)
        rows = (uniq[:, None] * self.k + j[None]).reshape(-1)
        scale = self.weight.norm(dim=-1).mean()
        self.weight[rows] = (w.reshape(-1, zs.shape[1]) * scale).to(self.weight.dtype)
        self.seen[uniq] = True

    def cosine(self, x):
        cos = F.normalize(x.float(), dim=-1) @ F.normalize(self.weight.float(), dim=-1).t()
        return cos.view(len(x), self.num_classes, self.k).amax(-1)

    def forward(self, x, label, present: Optional[torch.Tensor] = None):
        """present: (num_classes,) bool, 只让这些类参与 softmax (None = 全部类)。"""
        cos = self.cosine(x).clamp(-1 + 1e-7, 1 - 1e-7)
        m = self.margins[label]
        target = cos.gather(1, label[:, None]).squeeze(1)
        theta = target.acos()
        # θ+m 超过 π 时用 CosFace 式线性惩罚, 保持单调
        with_margin = torch.where(theta + m < math.pi, (theta + m).cos(), target - torch.sin(math.pi - m) * m)
        logits = cos.scatter(1, label[:, None], with_margin[:, None]) * self.scale
        if present is not None:
            logits = logits.masked_fill(~present[None], float('-inf'))
        return F.cross_entropy(logits, label)


def gather_nograd(x: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return x
    out = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(out, x.contiguous())
    return torch.cat(out)


class GatheredSigLIP(nn.Module):
    """本卡样本 × 全局样本的 sigmoid 对比损失 (同类为正, 去掉自身配对)。

    列特征 all-gather 不回传梯度: 相似度矩阵对称, (i,j) 的梯度在 i、j 各自的卡上都会算一次, 只差常数倍。
    gather=False 退化为旧版的单卡 SigLIP (对照组用)。
    """

    def __init__(self, gather: bool = True):
        super().__init__()
        self.gather = gather

    def forward(self, emb, label, sim_module):
        z = F.normalize(emb.float(), dim=-1)
        if not self.gather:
            z_all, y_all, rank = z, label, 0
        else:
            z_all = gather_nograd(z.detach())
            y_all = gather_nograd(label)
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        offset = rank * len(z)
        logits = sim_module.logit_scale.exp() * z @ z_all.t() + sim_module.logit_bias
        target = (label[:, None] == y_all[None]).float()
        self_mask = torch.ones_like(target)
        idx = torch.arange(len(z), device=z.device)
        self_mask[idx, idx + offset] = 0
        loss = F.binary_cross_entropy_with_logits(logits, target, weight=self_mask, reduction='sum')
        return loss / self_mask.sum()


class MatryoshkaLoss(nn.Module):
    """对 embedding 的每个前缀维度分别计算 loss 再平均; arcface 每个前缀一个独立分类器。"""

    def __init__(self, dims: Sequence[int], num_classes: int, arc_weight: float = 1.0, siglip_weight: float = 1.0,
                 margins: Optional[torch.Tensor] = None, k: int = 3, scale: float = 30.0,
                 siglip_gather: bool = True, imprint: bool = True, arc_negatives: str = 'all'):
        super().__init__()
        assert arc_negatives in ('all', 'batch')
        self.dims = tuple(dims)
        self.imprint = imprint
        # 'batch': arcface 负类只取当前全局 batch 内的类。类很多、每类很少被重访时, batch 外的类中心是用旧参数下的
        # embedding imprint 的 (相对当前分布整体偏移), 所有样本共享这些难负例, 推离它们等于整体平移 -> 坍缩
        # (2026-10-20 csip_p2/_base_collapsed_statBN; 哪怕只用 4 步前的中心也坍缩)。
        self.arc_negatives = arc_negatives
        self.arc_weight = arc_weight
        self.siglip_weight = siglip_weight
        self.arcs = nn.ModuleList([SubCenterArcFace(d, num_classes, k, scale, margins) for d in self.dims]) \
            if arc_weight > 0 else None
        self.siglip = GatheredSigLIP(siglip_gather)

    def forward(self, emb, label, sim_module):
        # loss 全程 fp32: autocast 会把 cos 的 matmul 降到 bf16, 精度不够分辨 margin
        with torch.autocast(device_type=emb.device.type, enabled=False):
            return self._forward(emb.float(), label, sim_module)

    def _forward(self, emb, label, sim_module):
        logs, total = {}, 0.0
        if self.arcs is not None:
            y_all = gather_nograd(label) if (self.imprint and self.training) or self.arc_negatives == 'batch' else None
            if self.imprint and self.training:
                emb_all = gather_nograd(emb.detach())
                for d, head in zip(self.dims, self.arcs):
                    head.imprint(emb_all[:, :d], y_all)
            present = None
            if self.arc_negatives == 'batch':
                present = torch.zeros(self.arcs[0].num_classes, dtype=torch.bool, device=label.device).index_fill_(0, y_all, True)
            arc = 0.0
            for d, head in zip(self.dims, self.arcs):
                l = head(emb[:, :d], label, present)
                logs[f'arc{d}'] = l.detach()
                arc = arc + l
            total = total + self.arc_weight * arc / len(self.dims)
        if self.siglip_weight > 0:  # sim 头只校准完整维度
            l = self.siglip(emb, label, sim_module)
            logs['siglip'] = l.detach()
            total = total + self.siglip_weight * l
        return total, logs
