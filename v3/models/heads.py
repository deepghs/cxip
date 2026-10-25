"""CxIP v3 头部: cond_tokens + Matryoshka embedding + 校准相似度。"""
import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def mlp(in_dim, hidden, out_dim):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))


class QueryPoolHead(nn.Module):
    """CCIP 头。

    pool='query': K 个可学习 query 对 (CLS + patch) tokens 做交叉注意力 → cond_tokens (B,K,dim);
                  embedding = MLP([CLS; mean(cond_tokens)])
    pool='cls':   embedding = MLP(CLS), cond_tokens = patch tokens 的线性投影 (仅作对照)
    pool='cls_mean': embedding = MLP([CLS; mean(patches)])
    """

    def __init__(self, in_dim: int = 2688, out_dim: int = 768, pool: str = 'query', num_queries: int = 16,
                 dim: int = 1024, heads: int = 16, hidden: int = 2048):
        super().__init__()
        self.pool = pool
        self.in_norm = nn.LayerNorm(in_dim, eps=1e-6)
        if pool == 'query':
            self.queries = nn.Parameter(torch.randn(num_queries, dim) * dim ** -0.5)
            self.attn = nn.MultiheadAttention(dim, heads, kdim=in_dim, vdim=in_dim, batch_first=True)
            self.ffn_norm = nn.LayerNorm(dim, eps=1e-6)
            self.ffn = mlp(dim, dim * 4, dim)
            self.proj = mlp(in_dim + dim, hidden, out_dim)
        elif pool == 'cls':
            self.proj = mlp(in_dim, hidden, out_dim)
        elif pool == 'cls_mean':
            self.proj = mlp(in_dim * 2, hidden, out_dim)
        else:
            raise ValueError(pool)

    def forward(self, cls: torch.Tensor, patches: torch.Tensor):
        cls = self.in_norm(cls)
        patches = self.in_norm(patches)
        if self.pool == 'query':
            kv = torch.cat([cls[:, None], patches], 1)
            q = self.queries[None].expand(len(cls), -1, -1)
            tokens, _ = self.attn(q, kv, kv, need_weights=False)
            tokens = tokens + self.ffn(self.ffn_norm(tokens))
            emb = self.proj(torch.cat([cls, tokens.mean(1)], -1))
        elif self.pool == 'cls':
            tokens = None
            emb = self.proj(cls)
        else:
            tokens = None
            emb = self.proj(torch.cat([cls, patches.mean(1)], -1))
        return dict(embedding=emb, cond_tokens=tokens)


class CalibratedSim(nn.Module):
    """score = exp(scale)·cos + bias, 与旧版 BatchCosineSimilarity 相同的参数化 (导出/阈值兼容)。"""

    def __init__(self, init_scale: float = math.log(10), init_bias: float = -10.0):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(init_scale))
        self.logit_bias = nn.Parameter(torch.tensor(init_bias))

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
        return self.logit_scale.exp() * a @ b.t() + self.logit_bias


class CxIPModel(nn.Module):
    """backbone + head + sim。forward 返回未归一化 embedding 及 cond_tokens。"""

    def __init__(self, backbone: nn.Module, head: nn.Module, matryoshka: Sequence[int] = (768, 512, 256, 128)):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.sim = CalibratedSim()
        self.matryoshka = tuple(matryoshka)

    def forward(self, x):
        out = self.backbone(x)
        if getattr(self.head, 'uses_taps', False):
            return self.head(out['taps'])
        return self.head(out['cls'], out['patches'])
