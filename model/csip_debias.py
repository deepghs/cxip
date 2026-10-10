# [RECONSTRUCTED] CSIP v2 / v2.1 训练模型，原实现(服务器 model/caformer.py 新版)已丢失
# 结构依据 Lark 文档 "AnimeDINO + cxip训练" 的白板:
#   backbone 4个stage -> (mean, std) -> raw style emb -> MLP -> style emb -> L_siglip
#   v2  : raw style emb -> GRL -> MLP -> probs  <-L_bce->  Anime Tagger(冻结) probs        (Content Eraser)
#   v2.1: raw style emb -> GRL -> MLP -> probs  <-L_kl ->  CCIP(冻结) -> MLP(训练) -> probs (Character Eraser)
# 依据 HF 原版 dghstmp/AnimeCSIP-v2(.1) 的 modeling_csip.py (原文): v2 起在 head 前加了 LayerNorm(3328, eps=1e-6)，v1 没有
# (BatchCosineSimilarity 的 SigLIP 初始化 log(10)/-10 已改在 model/caformer.py 中)
# eraser 接在 LayerNorm 之后还是之前无法确认，这里接在之后 (各 stage 的 mean/std 量级差异大，归一化后更好训)
# 推理时只用 caformer + norm + head + sim；导出见 transfs/trans_csip.py (caformer.* -> backbone.*)
import torch
import torch.nn.functional as F
from torch import nn

from .caformer import CAFormerStyleBackbone
from .grl import GRL


def eraser_mlp(in_dims, hidden_dims, out_dims):
    return nn.Sequential(
        nn.Linear(in_dims, hidden_dims),
        nn.SiLU(),
        nn.Linear(hidden_dims, out_dims),
    )


class ConstLogits(nn.Module):
    '''与输入无关的常数 logits (作为 module 才能被 CfgWDModelParser 按层名选中)'''

    def __init__(self, dims):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(dims))

    def forward(self, batch_size):
        return self.logits.expand(batch_size, -1)


class CAFormerStyleDebiasBackbone(CAFormerStyleBackbone):
    def __init__(self, model_name='caformer_b36', input_resolution: int = 384, heads: int = 8, out_dims: int = 768,
                 tagger: nn.Module = None, ccip: nn.Module = None, ccip_dims: int = 768, num_char_cls: int = 100,
                 eraser_dims: int = 512, grl_alpha: float = 1.0, eraser_pre_norm: bool = False,
                 detach_teacher: bool = False, teacher_scale: float = None, with_prior: bool = False,
                 compile_model: bool = False):
        super().__init__(model_name, input_resolution, heads, out_dims)
        all_chs = (128+256+512+768)*2
        self.norm = nn.LayerNorm(all_chs, eps=1e-6)
        self.grl = GRL(grl_alpha)
        self.eraser_pre_norm = eraser_pre_norm  # True: eraser 接在 LayerNorm 之前 (原版位置未知, 用于超参搜索)
        # teacher MLP 可训练且不 detach 时, teacher 和 char eraser 会一起塌成同一个近均匀分布, KL->0 但不含角色信息
        # detach_teacher: teacher 不接收 KL 梯度 (固定的随机投影); teacher_scale: 把 teacher logits 标准化后乘以该值, 避免随机投影的 softmax 过平
        self.detach_teacher = detach_teacher
        self.teacher_scale = teacher_scale
        # with_prior: 额外输出只学常数的 logits, 作为 eraser loss 的 "无信息" 基线 (诊断用, 不经过 backbone)
        self.with_prior = with_prior

        self.tagger = tagger
        if tagger is not None:
            self.content_eraser = eraser_mlp(all_chs, eraser_dims, tagger.num_tags)
            if with_prior:
                self.content_prior = ConstLogits(tagger.num_tags)

        self.ccip = ccip
        if ccip is not None:
            self.char_eraser = eraser_mlp(all_chs, eraser_dims, num_char_cls)
            self.char_teacher_head = eraser_mlp(ccip_dims, eraser_dims, num_char_cls)
            if with_prior:
                self.char_prior = ConstLogits(num_char_cls)

        if compile_model:
            self.compile_parts()

    def compile_parts(self):
        '''
        caformer 训练时 70% 以上时间耗在逐元素/LayerNorm/类型转换这类访存型 kernel 上, torch.compile 融合后 bf16 下 backbone 提速约 2.6x。
        按 stage 分别 compile: stage 上挂着取 style 特征的 forward hook, 整体 compile 会被 hook 打断。
        Module.compile() 是原地的, state_dict 的 key 不变 (不会多出 _orig_mod.)
        '''
        self.caformer.stem.compile()
        for stage in self.caformer.stages:
            stage.compile()
        if self.tagger is not None:
            self.tagger.model.compile()
        if self.ccip is not None:
            self.ccip.ccip.compile()

    def forward(self, image, content_image=None):
        self.caformer.forward_features(image)
        style_cat = torch.cat(self.style_list, dim=1)
        style_raw = self.norm(style_cat)
        self.style_list.clear()
        feat = self.head(style_raw)
        sim = self.sim(feat)
        out = {'pred': sim, 'feat': [feat + 0. * sim.mean()]}  # 0.*sim.mean() for DDP

        style_rev = self.grl(style_cat if self.eraser_pre_norm else style_raw)
        if self.tagger is not None and content_image is not None:
            out['content_logits'] = self.content_eraser(style_rev)
            out['content_probs'] = self.tagger(content_image)
            if self.with_prior:
                out['content_prior_logits'] = self.content_prior(len(image))
        if self.ccip is not None:
            out['char_logits'] = self.char_eraser(style_rev)
            teacher = self.char_teacher_head(self.ccip(image))
            if self.teacher_scale is not None:
                teacher = F.layer_norm(teacher, teacher.shape[-1:]) * self.teacher_scale
            out['char_teacher_logits'] = teacher.detach() if self.detach_teacher else teacher
            if self.with_prior:
                out['char_prior_logits'] = self.char_prior(len(image))
        return out
