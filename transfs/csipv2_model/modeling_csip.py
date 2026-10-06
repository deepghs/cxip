from typing import Callable, Union
import math

from timm.models.resnet import BasicBlock, Bottleneck, ResNet
import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from timm import create_model

from .configuration_csip import CSIPConfig

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
        self.norm = nn.LayerNorm(all_chs, eps=1e-6)
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
        feat = self.head(self.norm(style_all))
        out = self.sim(feat)
        return {"logits": out, "features": feat}