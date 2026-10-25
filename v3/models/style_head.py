"""CSIP 头: 多层 patch token 统计 → 每层画风 token (cond_tokens) → Matryoshka embedding。"""
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from v3.models.heads import mlp


class StyleStatsHead(nn.Module):
    """taps[l] (B, 1+N, D) 为第 l 个 block 的输出 (未过最终 LayerNorm)。

    每层先做无仿射 LayerNorm: 残差流里少数通道数值极大 (冻结特征中间层直接算 cos 几乎无区分力),
    不归一化的话统计量会被这几个通道主导。然后取 patch token 的 mean / std (Gram 式二阶统计的对角近似,
    与内容位置无关), 每层 MLP 得到一个画风 token; embedding = MLP(concat(style tokens))。
    stats='mean_std' | 'mean' (消融); use_cls: 额外拼上各层 CLS。
    """

    uses_taps = True

    def __init__(self, layers: Sequence[int], in_dim: int = 2688, out_dim: int = 768, token_dim: int = 512,
                 hidden: int = 2048, stats: str = 'mean_std', use_cls: bool = False, center: str = 'bn'):
        super().__init__()
        self.layers = tuple(layers)
        self.stats = stats
        self.use_cls = use_cls
        n_in = in_dim * ((2 if stats == 'mean_std' else 1) + int(use_cls))
        # 统计量含一个所有图片共有的大分量 (|样本偏差| 仅为 |均值| 的 16-28%): 不中心化时初始 embedding 两两 cos≈0.93,
        # imprinting + ArcFace 会把它放大成完全坍缩 (2026-10-20 csip_p2/_base_collapsed_noBN)。BN 减去跨样本均值。
        self.stat_norm = nn.ModuleList([nn.BatchNorm1d(n_in) if center == 'bn' else nn.Identity() for _ in self.layers])
        self.per_layer = nn.ModuleList([mlp(n_in, token_dim * 2, token_dim) for _ in self.layers])
        self.token_norm = nn.LayerNorm(token_dim, eps=1e-6)
        self.proj = mlp(token_dim * len(self.layers), hidden, out_dim)

    def compute_stats(self, taps: dict, mask: torch.Tensor = None):
        """每层 [mean; std (; cls)] of 无仿射 LayerNorm 后的 patch token, 无可学习参数。"""
        out = []
        for l in self.layers:
            h = F.layer_norm(taps[l].float(), taps[l].shape[-1:], eps=1e-6)
            cls, p = h[:, 0], h[:, 1:]
            w = None if mask is None else mask.float()[..., None]
            n = p.shape[1] if w is None else w.sum(1).clamp(min=1)
            mean = p.mean(1) if w is None else (p * w).sum(1) / n
            feats = [mean]
            if self.stats == 'mean_std':
                d2 = (p - mean[:, None]).pow(2)
                feats.append((d2.mean(1) if w is None else (d2 * w).sum(1) / n).add(1e-6).sqrt())
            if self.use_cls:
                feats.append(cls)
            out.append(torch.cat(feats, -1))
        return out

    def from_stats(self, stats):
        tokens = [f(norm(x)) for x, norm, f in zip(stats, self.stat_norm, self.per_layer)]
        tokens = self.token_norm(torch.stack(tokens, 1))  # (B, L, token_dim)
        return dict(embedding=self.proj(tokens.flatten(1)), cond_tokens=tokens)

    def forward(self, taps: dict, mask: torch.Tensor = None):
        """mask: (B, N) 有效 patch (排除白边 padding), None 表示全部有效。"""
        return self.from_stats(self.compute_stats(taps, mask))
