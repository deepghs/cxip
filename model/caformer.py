from torch import nn
from timm import create_model
from torch import nn
from rainbowneko.models.layers import BatchCosineSimilarity

from .attention_pool import AvgAttnPooling2d


class CAFormerBackbone(nn.Module):
    def __init__(self, model_name='caformer_m36', input_resolution=384, heads=8, out_dims: int = 768):
        super().__init__()
        caformer = create_model(model_name, pretrained=True)
        caformer.set_grad_checkpointing(True)
        del caformer.head
        self.caformer = caformer

        self.attnpool = AvgAttnPooling2d(caformer.num_features)
        self.sim = BatchCosineSimilarity()

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