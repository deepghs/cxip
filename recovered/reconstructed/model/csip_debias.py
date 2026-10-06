# [RECONSTRUCTED] CSIP v2 / v2.1 训练模型，原实现(服务器 model/caformer.py 新版)已丢失
# 结构依据 Lark 文档 "AnimeDINO + cxip训练" 的白板:
#   backbone 4个stage -> (mean, std) -> raw style emb -> MLP -> style emb -> L_siglip
#   v2  : raw style emb -> GRL -> MLP -> probs  <-L_bce->  Anime Tagger(冻结) probs        (Content Eraser)
#   v2.1: raw style emb -> GRL -> MLP -> probs  <-L_kl ->  CCIP(冻结) -> MLP(训练) -> probs (Character Eraser)
# 以下两点依据 HF 原版 dghstmp/AnimeCSIP-v2(.1) 的 modeling_csip.py (原文):
#   - v2 起在 head 前加了 LayerNorm(3328, eps=1e-6)，v1 没有
#   - BatchCosineSimilarity 初始化为 logit_scale=log(10), logit_bias=-10
# eraser 接在 LayerNorm 之后还是之前无法确认，这里接在之后 (各 stage 的 mean/std 量级差异大，归一化后更好训)
# 推理时只用 caformer + norm + head + sim；导出见 transfs/trans_csip.py (caformer.* -> backbone.*)
import math

import torch
from torch import nn

from .caformer import CAFormerStyleBackbone
from .grl import GRL


def eraser_mlp(in_dims, hidden_dims, out_dims):
    return nn.Sequential(
        nn.Linear(in_dims, hidden_dims),
        nn.SiLU(),
        nn.Linear(hidden_dims, out_dims),
    )


class CAFormerStyleDebiasBackbone(CAFormerStyleBackbone):
    def __init__(self, model_name='caformer_b36', input_resolution: int = 384, heads: int = 8, out_dims: int = 768,
                 tagger: nn.Module = None, ccip: nn.Module = None, ccip_dims: int = 768, num_char_cls: int = 100,
                 eraser_dims: int = 512, grl_alpha: float = 1.0):
        super().__init__(model_name, input_resolution, heads, out_dims)
        all_chs = (128+256+512+768)*2
        self.norm = nn.LayerNorm(all_chs, eps=1e-6)
        nn.init.constant_(self.sim.logit_scale, math.log(10))
        nn.init.constant_(self.sim.logit_bias, -10)
        self.grl = GRL(grl_alpha)

        self.tagger = tagger
        if tagger is not None:
            self.content_eraser = eraser_mlp(all_chs, eraser_dims, tagger.num_tags)

        self.ccip = ccip
        if ccip is not None:
            self.char_eraser = eraser_mlp(all_chs, eraser_dims, num_char_cls)
            self.char_teacher_head = eraser_mlp(ccip_dims, eraser_dims, num_char_cls)

    def forward(self, image, content_image=None):
        self.caformer.forward_features(image)
        style_raw = self.norm(torch.cat(self.style_list, dim=1))
        self.style_list.clear()
        feat = self.head(style_raw)
        sim = self.sim(feat)
        out = {'pred': sim, 'feat': [feat + 0. * sim.mean()]}  # 0.*sim.mean() for DDP

        style_rev = self.grl(style_raw)
        if self.tagger is not None and content_image is not None:
            out['content_logits'] = self.content_eraser(style_rev)
            out['content_probs'] = self.tagger(content_image)
        if self.ccip is not None:
            out['char_logits'] = self.char_eraser(style_rev)
            out['char_teacher_logits'] = self.char_teacher_head(self.ccip(image))
        return out
