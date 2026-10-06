from typing import Callable, Union
import math

from timm.models.resnet import BasicBlock, Bottleneck, ResNet
import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from timm import create_model

from .configuration_csip import CSIPConfig


class AttentionPool2d(nn.Module):
    """
    If the CNN's output is (1, 2048, 7, 7), then the parameters should be (7, 2048, 32, 1024),
    so if the CNN's output is (1, 512, 12, 12), then it should be (12, 512, 32?, 1024).
    """

    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)

class BatchCosineSimilarity(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([])*math.log(10))
        self.logit_bias = nn.Parameter(torch.ones([])*-10)

    def forward(self, image_features):  # x: BxN
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * torch.mm(image_features, image_features.transpose(0, 1)) + self.logit_bias

        return logits_per_image


class CSIPModel(PreTrainedModel):
    config_class = CSIPConfig

    def __init__(self, config):
        super().__init__(config)
        self.input_resolution = config.input_resolution
        backbone = create_model(config.backbone, pretrained=False)
        backbone.set_grad_checkpointing(True)
        del backbone.head
        self.backbone = backbone

        all_chs = (128+256+512+768)*2
        self.head = nn.Sequential(
            nn.Linear(all_chs, config.out_dims),
            nn.SiLU(),
            nn.Linear(config.out_dims, config.out_dims),
        )
        self.sim = BatchCosineSimilarity()

        self.style_list = []
        self.backbone.stages[0].register_forward_hook(self.feat_hook)
        self.backbone.stages[1].register_forward_hook(self.feat_hook)
        self.backbone.stages[2].register_forward_hook(self.feat_hook)
        self.backbone.stages[3].register_forward_hook(self.feat_hook)

        # Fix bugs: https://github.com/huggingface/transformers/issues/38996
        for module in self.children():
            # We found a sub-model: recursively dispatch its own init function now!
            if hasattr(module, "_init_weights"):
                module._initialize_weights = module._init_weights

    def feat_hook(self, host, fea_int, fea_out):
        mean = fea_out.mean(dim=(2,3))
        std = fea_out.std(dim=(2,3))
        self.style_list.append(torch.cat([mean, std], dim=1))

    def forward(self, x):
        x = self.backbone.forward_features(x)
        style_all = torch.cat(self.style_list, dim=1)
        self.style_list.clear()
        feat = self.head(style_all)
        out = self.sim(feat)
        return {"logits": out, "features": feat}