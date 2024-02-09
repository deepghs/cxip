import torch
from rainbowneko.train.loss import LossContainer
from torch import nn


class StyleBCELoss(LossContainer):
    def __init__(self, weight=1.0, eps=1e-4, reduction='mean'):
        super().__init__(None, weight=weight)
        self.bce_loss = nn.BCELoss()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred, target):
        input_tensor = pred['pred']
        # target_tensor = target['label']
        target_tensor = torch.cat([torch.ones(len(input_tensor) // 2), torch.zeros(len(input_tensor) // 2)]).to(input_tensor.device)

        return self.bce_loss(input_tensor, target_tensor)
