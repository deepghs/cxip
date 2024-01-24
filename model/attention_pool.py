from typing import Callable, Union

import torch
from torch import nn


class AttentionPool2d(nn.Module):
    "Attention for Learned Aggregation"

    def __init__(self,
                 ni: int,
                 bias: bool = True,
                 norm: Callable[[int], nn.Module] = nn.LayerNorm
                 ):
        super().__init__()
        self.norm = norm(ni)
        self.q = nn.Linear(ni, ni, bias=bias)
        self.vk = nn.Linear(ni, ni * 2, bias=bias)
        self.proj = nn.Linear(ni, ni)

    def forward(self, x, cls_q):
        x = self.norm(x.flatten(2).transpose(1, 2))
        B, N, C = x.shape

        q = self.q(cls_q.expand(B, -1, -1))
        k, v = self.vk(x).reshape(B, N, 2, C).permute(2, 0, 1, 3).chunk(2, 0)

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, C)
        return self.proj(x)


class AvgAttnPooling2d(nn.Module):
    def __init__(self,
                 ni: int,
                 attn_bias: bool = True,
                 ffn_expand: Union[int, float] = 3,
                 norm: Callable[[int], nn.Module] = nn.LayerNorm,
                 act_cls: Callable[[None], nn.Module] = nn.GELU,
                 ):
        super().__init__()
        self.cls_q = nn.Parameter(torch.zeros([1, ni]))
        self.attn = AttentionPool2d(ni, attn_bias, norm)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = norm(ni)
        self.ffn = nn.Sequential(
            nn.Linear(ni, int(ni * ffn_expand)),
            act_cls(),
            norm(int(ni * ffn_expand)),
            nn.Linear(int(ni * ffn_expand), ni)
        )
        nn.init.trunc_normal_(self.cls_q, std=0.02)
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.norm(self.pool(x).flatten(1) + self.attn(x, self.cls_q))
        return x + self.ffn(x)

    @torch.no_grad()
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
