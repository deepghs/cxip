# 2024 年三元组路线 (TripletBucket) 用的损失。TripletBucket 的每个 batch 依次是 anchor / positive / negative 三段。
import torch
from rainbowneko.models.layers import StyleSimilarity
from torch import nn
from torch.nn import functional as F


class StyleBCELoss(nn.Module):
    """csip_cat / csip_cross: 模型输出前一半为 (anchor, pos) 对、后一半为 (anchor, neg) 对的同画风概率。"""

    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, pred):  # pred: [B] sigmoid 概率
        n_pos = len(pred) // 2
        target = torch.cat([torch.ones(n_pos), torch.zeros(len(pred) - n_pos)]).to(pred)
        return self.bce_loss(pred, target)


class TripletStyleLoss(nn.Module):
    """csip_style_loss: 多层特征 Gram 矩阵距离的三元组损失 (原 rainbowneko.train.loss.StyleLoss)。"""

    def __init__(self, margin=5.):
        super().__init__()
        self.margin = margin
        self.style_sim = StyleSimilarity()

    def forward(self, feat):  # feat: list([B,C,H,W])
        loss = 0.
        for item in feat:
            q, pos, neg = item.chunk(3)
            loss = loss + self.style_sim(q, pos) + F.relu(self.margin - self.style_sim(q, neg))
        return loss / len(feat)
