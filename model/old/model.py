import numpy as np
import torch.nn
from torch import nn

from .backbone import get_backbone


class CCIPBatchMetrics(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # self.sim = nn.CosineSimilarity(dim=-1)

    def forward(self, image_features):  # x: BxN
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ image_features.t()
        if self.training:
            logits_per_image = logits_per_image - torch.diag_embed(torch.diag(logits_per_image))

        return logits_per_image


class CCIPFeature(nn.Module):
    def __init__(self, name: str = "clip/ViT-B/32"):
        super().__init__()
        self.backbone, self.preprocess = get_backbone(name)

    def forward(self, x):
        x = self.backbone(x)
        # x = x / x.norm(dim=-1, keepdim=True)
        return x


class CCIP(nn.Module):
    def __init__(self, name: str = "caformer_b36"):
        super().__init__()
        self.feature = CCIPFeature(name)
        self.metrics = CCIPBatchMetrics()

    @property
    def preprocess(self):
        return self.feature.preprocess

    def forward(self, x):
        # x: BxCxHxW
        feat = self.feature(x)  # BxF
        x = self.metrics(feat)  # BxB
        return x, feat

    @classmethod
    def from_ckpt(cls, ckpt_path: str):
        ckpt = torch.load(ckpt_path)
        model = cls()
        model.load_state_dict(ckpt)
        return model


class LogitToDiff(nn.Module):
    def __init__(self, scale):
        nn.Module.__init__(self)
        self.register_buffer('scale', torch.tensor(scale))
        self.scale: torch.Tensor

    def forward(self, x):
        return (self.scale - x) / (self.scale * 2)


if __name__ == '__main__':
    # image_files = [
    #     get_testfile('6124220.jpg'),
    #     get_testfile('6125785.jpg'),
    #     get_testfile('6125901.jpg'),
    # ]
    #
    # model = CCIP()
    # data = torch.stack([
    #     model.preprocess(Image.open(img))
    #     for img in image_files
    # ])
    # print(data.dtype, data.shape)
    #
    # print(F.softmax(model.forward(data), dim=-1))

    data = torch.randn(4, 3, 384, 384).cuda()
    model = CCIP('caformer').cuda()
    print(model.feature.backbone.attnpool.num_heads)
    print(model(data))
