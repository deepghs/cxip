# [RECONSTRUCTED] 服务器版本为12行；本地提交版本中 self.reduction 未定义、target 为 long，无法运行，此处修复
from torch import nn
from torch.nn import functional as F


class SiglipLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, label):
        same_mask = (label.unsqueeze(0) == label.unsqueeze(1)).to(pred.dtype)  # [B,B]
        return F.binary_cross_entropy_with_logits(pred, same_mask, reduction=self.reduction)
