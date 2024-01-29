from rainbowneko.evaluate import EvaluatorContainer
import torch

class CSIPmAPContainer(EvaluatorContainer):
    def reset(self):
        self.pred_list = []
        self.target_list = []

    def update(self, pred, target):
        self.pred_list.append(pred['pred'].cpu())
        self.target_list.append(target['label'].cpu())

    def evaluate(self):
        pred = torch.cat(self.pred_list) #[N,B]
        target = torch.cat(self.target_list) #[N]
        print(pred.shape, target.shape)
