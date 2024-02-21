from torch import nn
from timm.layers import use_fused_attn
from typing import Final
import torch.nn.functional as F
from einops import rearrange

class SDP_Attention(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.
    """
    fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            head_dim=32,
            num_heads=None,
            qkv_bias=False,
            attn_drop=0.,
            proj_drop=0.,
            proj_bias=False,
            **kwargs
    ):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim

        self.to_q = nn.Linear(dim, self.attention_dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, self.attention_dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, self.attention_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        q = rearrange(self.to_q(q), 'b n (nh h) -> b nh n h')
        k = rearrange(self.to_k(k), 'b n (nh h) -> b nh n h')
        v = rearrange(self.to_v(v), 'b n (nh h) -> b nh n h')

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x