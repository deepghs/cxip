from rainbowneko.evaluate import MetricContainer
from rainbowneko.models.layers import StyleSimilarity
import torch

class CXIPMetricContainer(MetricContainer):
    def reset(self):
        self.pred_list = []
        self.target_list = []

    def update(self, pred, inputs):
        args, kwargs = self.key_mapper(pred=pred, inputs=inputs)
        pred = kwargs['pred']
        target = kwargs['label']

        pred = pred - torch.diag_embed(torch.diag(pred))
        same_mask = (target.unsqueeze(0) == target.unsqueeze(1)).long()
        pos = (pred*same_mask).max(dim=1).values
        neg = (pred*(1.-same_mask)).max(dim=1).values
        self.pred_list.append(pos.cpu())
        self.pred_list.append(neg.cpu())
        self.target_list.append(torch.ones(len(target), device='cpu', dtype=torch.long))
        self.target_list.append(torch.zeros(len(target), device='cpu', dtype=torch.long))

    def finish(self, gather, is_local_main_process):
        pred = torch.cat(self.pred_list)  # [N,B]
        target = torch.cat(self.target_list)  # [N]

        pred = gather(pred)
        target = gather(target)

        v_metric = self.metric(pred, target)

        return v_metric.item()

class CSIPTripletAPContainer(MetricContainer):
    """TripletBucket (anchor / positive / negative 三段) 上，用多层特征的 Gram 距离区分正负对。"""

    def __init__(self, metric, device='cpu', key_map=('pred.feat -> feat',)):
        super().__init__(metric, device=device, key_map=key_map)
        self.style_sim = StyleSimilarity(batch_mean=True)

    def reset(self):
        self.pred_list = []
        self.target_list = []

    def update(self, pred, inputs):
        args, kwargs = self.key_mapper(pred=pred, inputs=inputs)
        feat = kwargs['feat']  # list([B,C,H,W])

        triplets = [item.chunk(3) for item in feat]
        sim_pos = sum(self.style_sim(q, pos) for q, pos, neg in triplets) / len(triplets)
        sim_neg = sum(self.style_sim(q, neg) for q, pos, neg in triplets) / len(triplets)

        self.pred_list.append(1. - sim_pos.cpu())
        self.pred_list.append(1. - sim_neg.cpu())
        self.target_list.append(torch.ones(len(sim_pos), device='cpu', dtype=torch.long))
        self.target_list.append(torch.zeros(len(sim_neg), device='cpu', dtype=torch.long))

    def finish(self, gather, is_local_main_process):
        pred = gather(torch.cat(self.pred_list))  # [N]
        target = gather(torch.cat(self.target_list))  # [N]
        return self.metric(pred, target).item()

class CSIP_PN_APContainer(MetricContainer):
    """模型输出前一半为 (anchor, pos)、后一半为 (anchor, neg) 的同画风概率 (CAFormerCat/CrossBackbone)。"""

    def __init__(self, metric, device='cpu', key_map=('pred.pred -> pred',)):
        super().__init__(metric, device=device, key_map=key_map)

    def reset(self):
        self.pred_list = []
        self.target_list = []

    def update(self, pred, inputs):
        args, kwargs = self.key_mapper(pred=pred, inputs=inputs)
        pred = kwargs['pred']  # [B]
        n_pos = len(pred) // 2

        self.pred_list.append(pred.cpu())
        self.target_list.append(torch.ones(n_pos, device='cpu', dtype=torch.long))
        self.target_list.append(torch.zeros(len(pred) - n_pos, device='cpu', dtype=torch.long))

    def finish(self, gather, is_local_main_process):
        pred = gather(torch.cat(self.pred_list))  # [N]
        target = gather(torch.cat(self.target_list))  # [N]
        return self.metric(pred, target).item()
