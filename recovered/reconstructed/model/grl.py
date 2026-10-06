# [RECONSTRUCTED] 原文件 /data/cxip/model/grl.py 已丢失，依据 Lark 文档白板中的 "-grad" 模块重建
import torch
from torch import nn


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GRL(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradReverse.apply(x, self.alpha)

    def extra_repr(self):
        return f'alpha={self.alpha}'
