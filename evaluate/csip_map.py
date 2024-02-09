from rainbowneko.evaluate import EvaluatorContainer
from rainbowneko.models.layers import StyleSimilarity
import torch

class CSIPmAPContainer(EvaluatorContainer):
    def reset(self):
        self.pred_list = []
        self.target_list = []

    def update(self, pred, target):
        for pred_cudai, target_cudai in zip(pred['pred'], target['label']):
            same_mask = (target_cudai.unsqueeze(0) == target_cudai.unsqueeze(1)).long()
            pos = (pred_cudai*same_mask).max(dim=1).values
            neg = (pred_cudai*(1.-same_mask)).max(dim=1).values
            self.pred_list.append(pos.cpu())
            self.pred_list.append(neg.cpu())
            self.target_list.append(torch.ones(len(target_cudai), device='cpu', dtype=torch.long))
            self.target_list.append(torch.zeros(len(target_cudai), device='cpu', dtype=torch.long))

    def evaluate(self):
        pred = torch.cat(self.pred_list) #[N,B]
        target = torch.cat(self.target_list) #[N]
        return self.evaluator(pred, target)

class CSIPTripletAPContainer(EvaluatorContainer):
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

class CSIP_PN_APContainer(EvaluatorContainer):
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