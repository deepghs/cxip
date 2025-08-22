from torch import nn
from torch.nn import functional as F


class SiglipLoss(nn.Module):
    def forward(self, pred, label):
        prob = F.sigmoid(pred)  # [B,B]
        same_mask = (label.unsqueeze(0) == label.unsqueeze(1)).long()  # [B,B]
        return F.binary_cross_entropy(prob, same_mask, reduction=self.reduction)
