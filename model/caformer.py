from torch import nn
from timm import create_model
from .attention_pool import AvgAttnPooling2d
from rainbowneko.models.layers import BatchCosineSimilarity

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
        out = self.sim(feat) # [B,B]
        return out, feat+0.*out.mean() # 0.*out.mean() for DDP