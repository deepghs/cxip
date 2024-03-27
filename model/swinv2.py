import torch
from rainbowneko.models.layers import BatchCosineSimilarity
from timm import create_model
from torch import nn

from .attention_pool import AvgAttnPooling2d
from einops import rearrange


class SwinV2Backbone(nn.Module):
    def __init__(self, model_name='hf-hub:SmilingWolf/wd-swinv2-tagger-v3', img_size=448):
        super().__init__()
        swin = create_model(model_name, pretrained=True, img_size=img_size)
        swin.set_grad_checkpointing(True)
        del swin.head
        self.swin = swin

        self.attnpool = AvgAttnPooling2d(swin.num_features)
        self.sim = BatchCosineSimilarity()

    def forward(self, x):
        x = self.swin.forward_features(x)
        x = rearrange(x, 'b h w c -> b c (h w)')
        feat = self.attnpool(x)
        out = self.sim(feat)  # [B,B]
        return out, feat + 0. * out.mean()  # 0.*out.mean() for DDP


if __name__ == '__main__':
    model = SwinV2Backbone()
    x = torch.randn(1, 3, 224, 224)
    out, feat = model(x)
    print(out.shape, feat.shape)
