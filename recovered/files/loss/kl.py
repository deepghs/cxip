import torch
import torch.nn as nn
import torch.nn.functional as F

class LogitsKLLoss(nn.Module):
    """
    直接接受两个 Logits 的 KL Loss
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.T = temperature

    def forward(self, student_logits, teacher_logits):
        s_log_probs = F.log_softmax(student_logits / self.T, dim=1)
        t_probs = F.softmax(teacher_logits / self.T, dim=1)
        loss = F.kl_div(s_log_probs, t_probs, reduction='batchmean')
        return loss * (self.T ** 2)