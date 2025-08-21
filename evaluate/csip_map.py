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
    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.style_sim = StyleSimilarity(batch_mean=True)

    def reset(self):
        self.pred_list = []
        self.target_list = []

    def update(self, pred, target):
        for pred_cudai, target_cudai in zip(pred['feat'], target['label']):
            query = [item[target_cudai == 0, ...] for item in pred_cudai]
            positive_key = [item[target_cudai == 1, ...] for item in pred_cudai]
            negative_keys = [item[target_cudai == 2, ...] for item in pred_cudai]

            sim_pos = sum(self.style_sim(q, pos) for q, pos in zip(query, positive_key)) / len(query)
            sim_neg = sum(self.style_sim(q, neg) for q, neg in zip(query, negative_keys)) / len(query)

            self.pred_list.append(1.-sim_pos.cpu())
            self.pred_list.append(1.-sim_neg.cpu())
            self.target_list.append(torch.ones(len(query[0]), device='cpu', dtype=torch.long))
            self.target_list.append(torch.zeros(len(query[0]), device='cpu', dtype=torch.long))

    def evaluate(self):
        pred = torch.cat(self.pred_list) #[N]
        target = torch.cat(self.target_list) #[N]
        return self.evaluator(pred, target)

class CSIP_PN_APContainer(MetricContainer):
    def reset(self):
        self.pred_list = []
        self.target_list = []

    def update(self, pred, target):
        for pred_cudai, target_cudai in zip(pred['pred'], target['label']):
            self.pred_list.append(pred_cudai.cpu())
            self.target_list.append(torch.ones(len(pred_cudai)//2, device='cpu', dtype=torch.long))
            self.target_list.append(torch.zeros(len(pred_cudai)//2, device='cpu', dtype=torch.long))

    def evaluate(self):
        pred = torch.cat(self.pred_list) #[N,B]
        target = torch.cat(self.target_list) #[N]
        return self.evaluator(pred, target)