from torch import nn
import torch
import numpy as np
from timm import create_model
from timm.models import register_model
from timm.models.metaformer import SepConv, Attention, MetaFormer, LayerNorm2dNoBias, LayerNormNoBias, _create_metaformer
from torch import nn
from rainbowneko.models.layers import BatchCosineSimilarity
from .attention import SDP_Attention

from .attention_pool import AvgAttnPooling2d

class BatchCosineSimilarityONNX(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image_features):  # x: BxN
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * torch.mm(image_features, image_features.transpose(0, 1))

        return logits_per_image

@register_model
def caformer_t15(pretrained=False, **kwargs) -> MetaFormer:
    model_kwargs = dict(
        depths=[3, 3, 6, 3],
        dims=[64, 128, 192, 256],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        norm_layers=[LayerNorm2dNoBias] * 2 + [LayerNormNoBias] * 2,
        **kwargs)
    return _create_metaformer('caformer_t15', pretrained=pretrained, **model_kwargs)

class CAFormerBackbone(nn.Module):
    def __init__(self, model_name='caformer_m36', input_resolution=384, heads=8, out_dims: int = 768):
        super().__init__()
        caformer = create_model(model_name, pretrained=True)
        caformer.set_grad_checkpointing(True)
        del caformer.head
        self.caformer = caformer

        self.attnpool = AvgAttnPooling2d(caformer.num_features)
        self.sim = BatchCosineSimilarityONNX()

    def forward(self, x):
        x = self.caformer.forward_features(x)
        feat = self.attnpool(x)
        out = self.sim(feat)  # [B,B]
        return out, feat + 0. * out.mean()  # 0.*out.mean() for DDP


class CAFormerStyleBackbone(nn.Module):
    def __init__(self, model_name='caformer_m36'):
        super().__init__()
        caformer = create_model(model_name, pretrained=True)
        #caformer.set_grad_checkpointing(True)
        del caformer.head
        self.caformer = caformer

        self.feat_list = []
        self.caformer.stages[0].register_forward_hook(self.feat_hook)
        self.caformer.stages[1].register_forward_hook(self.feat_hook)
        self.caformer.stages[2].register_forward_hook(self.feat_hook)
        self.caformer.stages[3].register_forward_hook(self.feat_hook)

    def feat_hook(self, host, fea_int, fea_out):
        self.feat_list.append(fea_out)

    def forward(self, x):
        x = self.caformer.forward_features(x)

        feat_all = self.feat_list.copy()
        self.feat_list.clear()
        return x, feat_all

class CAFormerCatBackbone(nn.Module):
    def __init__(self, model_name='caformer_m36', input_resolution=384, heads=8, out_dims: int = 768):
        super().__init__()
        try:
            caformer = create_model(model_name, pretrained=True)
        except:
            caformer = create_model(model_name, pretrained=False)
        caformer.set_grad_checkpointing(True)
        del caformer.head
        self.caformer = caformer

        self.attnpool = AvgAttnPooling2d(caformer.num_features)
        self.out_layers = nn.Sequential(
            nn.Linear(caformer.num_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x, x_ref=None):
        if x_ref is None:
            anchor, pos, neg = x.chunk(3)

            a_pos = torch.cat([anchor, pos], dim=2) # [B,C,2H,W]
            a_neg = torch.cat([anchor, neg], dim=2) # [B,C,2H,W]
            x = torch.cat([a_pos, a_neg])
        else:
            x = torch.cat([x, x_ref], dim=2)

        x = self.caformer.forward_features(x)
        feat = self.attnpool(x)
        out = self.out_layers(feat).flatten()
        return out, feat + 0. * out.mean()  # 0.*out.mean() for DDP

class CAFormerCrossBackbone(nn.Module):
    def __init__(self, model_name='caformer_m36', input_resolution=384, heads=8, out_dims: int = 768):
        super().__init__()
        try:
            caformer = create_model(model_name, pretrained=True)
        except:
            caformer = create_model(model_name, pretrained=False)
        caformer.set_grad_checkpointing(True)
        del caformer.head
        self.caformer = caformer

        self.cross_attn = SDP_Attention(caformer.num_features)

        self.attnpool = AvgAttnPooling2d(caformer.num_features)
        self.out_layers = nn.Sequential(
            nn.Linear(caformer.num_features, 1),
            nn.Sigmoid()
        )

    def attn_v1(self, anchor, pos, neg):
        a_pos = self.cross_attn(anchor, pos, pos)
        a_neg = self.cross_attn(anchor, neg, neg)
        return a_pos, a_neg

    def attn_v2(self, anchor, pos, neg):
        a_pos = torch.cat([anchor, pos], dim=2) # [B,C,2H,W]
        a_neg = torch.cat([anchor, neg], dim=2) # [B,C,2H,W]
        a_pos = self.cross_attn(a_pos, a_pos, a_pos)
        a_neg = self.cross_attn(a_neg, a_neg, a_neg)
        return a_pos, a_neg
    
    def attn_v3(self, anchor, pos, neg):
        ap = torch.cat([anchor, pos])
        pa = torch.cat([pos, anchor])
        an = torch.cat([anchor, neg])
        na = torch.cat([neg, anchor])

        a_pos = self.cross_attn(ap, pa, pa).view(2, -1, *ap.shape[1:]).mean(dim=0)
        a_neg = self.cross_attn(an, na, na).view(2, -1, *an.shape[1:]).mean(dim=0)
        return a_pos, a_neg

    def forward(self, x, x_ref=None):
        x = self.caformer.forward_features(x)

        if x_ref is None:
            anchor, pos, neg = x.chunk(3)
            a_pos, a_neg = self.attn_v1(anchor, pos, neg)
            x = torch.cat([a_pos, a_neg])
        else:
            x = self.cross_attn(x, x_ref, x_ref)

        feat = self.attnpool(x)
        out = self.out_layers(feat).flatten()
        return out, feat + 0. * out.mean()  # 0.*out.mean() for DDP