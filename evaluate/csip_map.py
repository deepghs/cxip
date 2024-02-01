from rainbowneko.evaluate import EvaluatorContainer
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
