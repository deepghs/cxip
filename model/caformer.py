from torch import nn
from timm import create_model
from .attention_pool import AttentionPool2d

class CAFormerBackbone(nn.Module):
    def __init__(self, model_name='caformer_m36', input_resolution=384, heads=8, out_dims: int = 768):
        super().__init__()
        caformer = create_model(model_name, pretrained=True)
        caformer.set_grad_checkpointing(True)
        del caformer.head
        self.caformer = caformer

        self.attnpool = AttentionPool2d(input_resolution // 32, caformer.num_features, heads, out_dims)

    def forward(self, x):
        x = self.caformer.forward_features(x)
        x = self.attnpool(x)
        return x