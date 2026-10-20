# 2024 年 CSIP v1 各条路线用的对比损失。原先在 RainbowNekoEngine 里 (rainbowneko.train.loss)，
# 是旧接口 forward(pred, target)；这里改成按 key_map 取参数的普通 nn.Module，配合 LossContainer 使用。
import torch
from einops import repeat
from torch import nn
from torch.nn import functional as F


class MLCEImageLoss(nn.Module):
    """多正样本的 softmax CE: 同类样本的概率合并到对角线上。"""

    def __init__(self, eps=1e-4, reduction='mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred, label):  # pred: [B,B] 相似度 logits
        prob = F.softmax(pred, dim=1)

        same_mask = (label.unsqueeze(0) == label.unsqueeze(1)).long()  # [B,B]
        same_mask_diag0 = same_mask - torch.diag_embed(torch.diag(same_mask))  # diag=0

        prob_x = prob.clone()
        prob_x[same_mask_diag0.bool()] = self.eps
        prob_x.diagonal().copy_((prob * same_mask_diag0.detach()).sum(dim=1))
        prob_x = prob_x + torch.diag_embed(torch.full((len(label),), self.eps, device=pred.device))
        y = torch.arange(0, len(label), device=pred.device)

        return F.nll_loss(prob_x.log(), y, reduction=self.reduction)


# Copy from https://github.com/RElbers/info-nce-pytorch/blob/main/info_nce/__init__.py
class InfoNCELoss(nn.Module):
    """
    PosNegBucket 的 batch 中，与第一个样本同类的为正样本 (一半作 query、一半作 positive key)，其余为负样本。
    """

    def __init__(self, temperature=0.1, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, feat, label):  # feat: [B,C]
        pos = label[0] == label
        query, positive_key = feat[pos, :].chunk(2)
        if query.shape[0] != positive_key.shape[0]:
            query = query[:-1]
        negative_keys = feat[~pos, :]

        return self.info_nce(query, positive_key, negative_keys, T=self.temperature)

    def sim_logits(self, query, pos, neg):
        '''
        :param query: [Nq, C]
        :param pos: [Np, C]
        :param neg: [Nn, C]
        '''
        query = F.normalize(query, dim=-1)
        pos = F.normalize(pos, dim=-1)
        neg = F.normalize(neg, dim=-1)

        sim_pos = torch.mm(query, pos.transpose(0, 1))  # [Nq, Np]
        sim_neg = torch.mm(query, neg.transpose(0, 1))  # [Nq, Nn]
        logits = torch.cat([sim_pos, sim_neg], dim=1)
        labels = torch.cat([torch.ones(sim_pos.shape[-1]), torch.zeros(sim_neg.shape[-1])]).to(query.device, dtype=query.dtype)
        labels = repeat(labels, 'n -> Nq n', Nq=len(query))
        return logits, labels

    def info_nce(self, query, pos, neg, T=0.1):
        logits, labels = self.sim_logits(query, pos, neg)
        return F.cross_entropy(logits / T, labels, reduction=self.reduction)


class NoisyInfoNCELoss(InfoNCELoss):
    # RCE+NCE https://arxiv.org/pdf/2006.13554.pdf
    def info_nce(self, query, pos, neg, T=0.1):
        logits, labels = self.sim_logits(query, pos, neg)
        loss = self.reverse_cross_entropy(logits / T, labels) + self.normalized_cross_entropy(logits / T, labels)
        return loss.mean() if self.reduction == 'mean' else loss

    def reverse_cross_entropy(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        labels = torch.clamp(labels, min=1e-4, max=1.0)
        return -torch.sum(pred * torch.log(labels), dim=1)

    def normalized_cross_entropy(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        return -torch.sum(labels * pred, dim=1) / (-pred.sum(dim=1))
