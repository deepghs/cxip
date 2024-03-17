from torch import nn
import torch
from timm import create_model
from timm.models import register_model
from timm.models.metaformer import SepConv, Attention, MetaFormer, LayerNorm2dNoBias, LayerNormNoBias, _create_metaformer
from torch import nn
from .attention import SDP_Attention

from .attention_pool import AvgAttnPooling2d
from torchvision.models import vgg19, VGG19_Weights


class VGGStyleBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        self.vgg_blocks = [
            nn.Sequential(vgg[:1]),
            nn.Sequential(vgg[1:3]),
            nn.Sequential(vgg[3:6]),
            nn.Sequential(vgg[6:8]),
            nn.Sequential(vgg[8:11]),
        ]

    def forward(self, x):
        feat_list = []
        for block in self.vgg_blocks:
            x = block(x)
            feat_list.append(x)

        return x, feat_list
