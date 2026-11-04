# SPDX-License-Identifier: AGPL-3.0-only
from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from timm.layers import SwiGLUPacked
from timm.models.vision_transformer import VisionTransformer
from torch import nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_cxip import CxIPConfig


def _mlp(in_dim, hidden, out_dim):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))


class QueryPoolHead(nn.Module):
    """Learnable queries cross-attend to [CLS; patch tokens] -> cond tokens; embedding = MLP([CLS; mean(cond tokens)])."""

    def __init__(self, in_dim, out_dim, num_queries, dim, heads, hidden):
        super().__init__()
        self.in_norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.queries = nn.Parameter(torch.zeros(num_queries, dim))
        self.attn = nn.MultiheadAttention(dim, heads, kdim=in_dim, vdim=in_dim, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = _mlp(dim, dim * 4, dim)
        self.proj = _mlp(in_dim + dim, hidden, out_dim)

    def forward(self, tokens):
        tokens = self.in_norm(tokens)
        cls = tokens[:, 0]
        q = self.queries[None].expand(len(tokens), -1, -1)
        cond, _ = self.attn(q, tokens, tokens, need_weights=False)
        cond = cond + self.ffn(self.ffn_norm(cond))
        return self.proj(torch.cat([cls, cond.mean(1)], -1)), cond


class CalibratedSim(nn.Module):
    """logit = exp(logit_scale) * cos(a, b) + logit_bias; sigmoid(logit) is the match probability."""

    def __init__(self, init_scale, init_bias):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.logit_bias = nn.Parameter(torch.tensor(float(init_bias)))

    def forward(self, a, b):
        a, b = F.normalize(a.float(), dim=-1), F.normalize(b.float(), dim=-1)
        return self.logit_scale.float().exp() * a @ b.t() + self.logit_bias.float()


@dataclass
class CxIPOutput(ModelOutput):
    features: torch.FloatTensor = None      # (B, out_dim) raw embeddings (L2-normalize before use)
    logits: torch.FloatTensor = None        # (B, B) calibrated pairwise similarity logits
    cond_tokens: torch.FloatTensor = None   # (B, num_queries, head_dim)


class CxIPModel(PreTrainedModel):
    config_class = CxIPConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["Block"]

    def __init__(self, config: CxIPConfig):
        super().__init__(config)
        self.backbone = VisionTransformer(
            img_size=config.image_size, patch_size=config.patch_size, embed_dim=config.embed_dim,
            depth=config.depth, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio, qkv_bias=True,
            init_values=1.0, class_token=True, reg_tokens=0, dynamic_img_size=True, num_classes=0,
            global_pool="token", norm_layer=partial(nn.LayerNorm, eps=1e-6), mlp_layer=SwiGLUPacked,
            act_layer=nn.SiLU, weight_init="skip")
        self.head = QueryPoolHead(config.embed_dim, config.out_dim, config.num_queries, config.head_dim,
                                  config.head_num_heads, config.head_hidden)
        self.sim = CalibratedSim(config.logit_scale_init, config.logit_bias_init)
        self.post_init()

    def _init_weights(self, module):
        pass  # weights always come from the checkpoint

    def _encode(self, x):
        tokens = self.backbone.forward_features(x.to(self.dtype))
        return self.head(tokens)

    def forward(self, pixel_values: Union[torch.Tensor, List[torch.Tensor]], return_logits: bool = True,
                **kwargs) -> CxIPOutput:
        """pixel_values: (B, 3, H, W) with H, W multiples of 16, or a list of (3, H, W) tensors of mixed sizes
        (CSIP keeps native resolution; images of the same size are batched together)."""
        if isinstance(pixel_values, (list, tuple)):
            groups = {}
            for i, x in enumerate(pixel_values):
                groups.setdefault(tuple(x.shape), []).append(i)
            feats, conds = [None] * len(pixel_values), [None] * len(pixel_values)
            for idx in groups.values():
                f, c = self._encode(torch.stack([pixel_values[i] for i in idx]).to(self.device))
                for j, i in enumerate(idx):
                    feats[i], conds[i] = f[j], c[j]
            feats, conds = torch.stack(feats), torch.stack(conds)
        else:
            feats, conds = self._encode(pixel_values.to(self.device))
        logits = self.sim(feats, feats) if return_logits else None
        return CxIPOutput(features=feats, logits=logits, cond_tokens=conds)

    @torch.no_grad()
    def get_image_features(self, pixel_values, dim: Optional[int] = None, normalize: bool = True):
        """Embeddings for retrieval / clustering. dim: Matryoshka prefix (768 / 512 / 256 / 128)."""
        f = self.forward(pixel_values, return_logits=False).features.float()
        if dim is not None:
            f = f[:, :dim]
        return F.normalize(f, dim=-1) if normalize else f

    def compute_logits(self, a: torch.Tensor, b: torch.Tensor):
        """Calibrated similarity logits between two sets of features, (len(a), len(b)).
        logit > config.threshold means the same character (CCIP) / the same artist (CSIP)."""
        return self.sim(a, b)

    def compute_probs(self, a: torch.Tensor, b: torch.Tensor):
        return torch.sigmoid(self.compute_logits(a, b))
