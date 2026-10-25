"""AnimeDINOv3 (AnimeDINO-2B-P16-512) 骨干封装: 加载、微调方式 (冻结/最后 N 层/分块/全量/LoRA)、多层特征、提速开关。

输入约定与发布版一致: 白底 pad 成方形, bicubic resize, ImageNet mean/std, 边长为 16 的倍数。
"""
from __future__ import annotations

import json
import math
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from timm.layers import PatchDropout, SwiGLUPacked
from timm.models.vision_transformer import VisionTransformer
from torch import nn
from torch.utils.checkpoint import checkpoint

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LORA_TARGETS = ('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2')


def build_vit(config: dict) -> VisionTransformer:
    return VisionTransformer(
        img_size=config['image_size'], patch_size=config['patch_size'], embed_dim=config['embed_dim'],
        depth=config['depth'], num_heads=config['num_heads'], mlp_ratio=config['mlp_ratio'], qkv_bias=True,
        init_values=1.0, class_token=True, reg_tokens=0, dynamic_img_size=True, num_classes=0,
        global_pool='token', norm_layer=partial(nn.LayerNorm, eps=1e-6), mlp_layer=SwiGLUPacked,
        act_layer=nn.SiLU)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: Optional[float] = None, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = (alpha or rank) / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b) * self.scale

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        self.base.weight += (self.lora_b @ self.lora_a).to(self.base.weight.dtype) * self.scale
        return self.base


class AnimeDINOBackbone(nn.Module):
    """forward(x) -> dict(cls=(B,D), patches=(B,N,D), taps={layer: (B,1+N,D)})。

    taps 是第 layer 个 block (1-based) 的输出, 未过最终 LayerNorm; cls/patches 过最终 LayerNorm。
    patch_drop>0 时训练态随机保留 (1-p) 的 patch token (位置编码已加, FLIP 式提速), patches 只含保留的 token。
    """

    def __init__(self, path: str | Path, tap_layers: Sequence[int] = (), patch_drop: float = 0.0):
        super().__init__()
        path = Path(path)
        self.config = json.loads((path / 'config.json').read_text())
        self.vit = build_vit(self.config)
        self.vit.load_state_dict(load_file(str(path / 'model.safetensors')), strict=True)
        self.embed_dim = self.config['embed_dim']
        self.depth = self.config['depth']
        self.patch_size = self.config['patch_size']
        self.tap_layers = tuple(sorted(tap_layers))
        self.set_patch_drop(patch_drop)
        self.grad_ckpt = False
        self.ckpt_ids = set()
        self.first_trainable = 0  # 之前的 block 没有可训练参数, 放在 no_grad 下跑

    def set_patch_drop(self, prob: float):
        self.patch_drop_prob = prob
        self.vit.patch_drop = PatchDropout(prob, num_prefix_tokens=1) if prob > 0 else nn.Identity()

    # ---- 微调方式 ----
    def setup_finetune(self, mode: str, n_last: int = 6, blocks: Iterable[int] = (), lora_rank: int = 32,
                       lora_alpha: Optional[float] = None, lora_targets: Sequence[str] = LORA_TARGETS):
        """mode: frozen | last_n | blocks | full | lora。"""
        self.requires_grad_(False)
        vit = self.vit
        if mode == 'frozen':
            self.first_trainable = self.depth
        elif mode in ('last_n', 'blocks'):
            ids = sorted(range(self.depth - n_last, self.depth) if mode == 'last_n' else blocks)
            for i in ids:
                vit.blocks[i].requires_grad_(True)
            vit.norm.requires_grad_(True)
            self.first_trainable = ids[0]
        elif mode == 'full':
            self.requires_grad_(True)
            self.first_trainable = 0
        elif mode == 'lora':
            for blk in vit.blocks:
                for name in lora_targets:
                    parent_name, attr = name.rsplit('.', 1)
                    parent = blk.get_submodule(parent_name)
                    setattr(parent, attr, LoRALinear(getattr(parent, attr), lora_rank, lora_alpha))
            vit.norm.requires_grad_(True)
            self.first_trainable = 0
        else:
            raise ValueError(mode)
        return self

    def layer_id(self, name: str) -> int:
        """LLRD 用的层号: patch_embed/pos_embed/cls_token=0, blocks.i=i+1, norm=depth+1。"""
        if name.startswith('vit.blocks.'):
            return int(name.split('.')[2]) + 1
        if name.startswith(('vit.patch_embed', 'vit.pos_embed', 'vit.cls_token')):
            return 0
        return self.depth + 1

    def param_groups(self, lr: float, layer_decay: float = 1.0, weight_decay: float = 0.05):
        groups = {}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            lid = self.layer_id(name)
            no_wd = p.ndim == 1 or name.endswith(('pos_embed', 'cls_token')) or 'lora_' in name
            key = (lid, no_wd)
            if key not in groups:
                scale = layer_decay ** (self.depth + 1 - lid)
                groups[key] = dict(params=[], lr=lr * scale, lr_scale=scale, weight_decay=0.0 if no_wd else weight_decay,
                                   name=f'backbone.l{lid}{".no_wd" if no_wd else ""}')
            groups[key]['params'].append(p)
        return list(groups.values())

    # ---- 提速 ----
    def set_grad_checkpointing(self, enable: bool | int = True):
        """True/False: 全部/不做; 整数 n: 只对 n 个可训练 block 做 checkpoint (均匀分布), 用少量重算换显存。"""
        if enable is True or enable is False:
            self.ckpt_ids = set(range(self.depth)) if enable else set()
        else:
            trainable = list(range(self.first_trainable, self.depth))
            n = min(int(enable), len(trainable))
            self.ckpt_ids = {trainable[int(i * len(trainable) / n)] for i in range(n)} if n > 0 else set()
        self.grad_ckpt = bool(self.ckpt_ids)

    def compile_blocks(self, **kwargs):
        for blk in self.vit.blocks:
            blk.compile(**kwargs)

    # ---- 前向 ----
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        vit = self.vit
        taps = {}
        frozen_prefix = self.first_trainable > 0 and torch.is_grad_enabled()
        with torch.set_grad_enabled(torch.is_grad_enabled() and not frozen_prefix):
            x = vit.patch_embed(x)
            x = vit._pos_embed(x)
            x = vit.patch_drop(x)
            x = vit.norm_pre(x)
        for i, blk in enumerate(vit.blocks):
            if i < self.first_trainable and frozen_prefix:
                with torch.no_grad():
                    x = blk(x)
            elif self.grad_ckpt and i in self.ckpt_ids and self.training and torch.is_grad_enabled():
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
            if i + 1 in self.tap_layers:
                taps[i + 1] = x
        x = vit.norm(x)
        return dict(cls=x[:, 0], patches=x[:, 1:], taps=taps)
