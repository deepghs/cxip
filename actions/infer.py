import logging
import math
from typing import Dict, Any

import torch
from rainbowneko.infer import BasicAction, MemoryMixin, feedback_input
from tqdm import tqdm


class CXIPForwardAction(BasicAction, MemoryMixin):
    def __init__(self, bs: int = 32):
        MemoryMixin.__init__(self)
        self.bs = bs

    @feedback_input
    def forward(self, input: Dict[str, Any], memory, **states):
        with torch.inference_mode():
            model = memory.model
            model.eval()
            x = input['x']

            batch_count = int(math.ceil(x.shape[0] / self.bs))
            feats = []
            for i in tqdm(range(batch_count)):
                x_ = x[i * self.bs: (i + 1) * self.bs]
                feats.append(model(x_)['feat'][0])

            feats = torch.concat(feats)
            logging.info(f'Features shape: {feats.shape}.')
            output = model.model.calc_sim(feats)
            logging.info(f'Output shape: {output.shape}.')

        return {
            'output': {
                'pred': output,
                'feat': feats,
            }
        }


class SaveFeatureAction(BasicAction, MemoryMixin):
    def __init__(self, file):
        MemoryMixin.__init__(self)
        self.file = file

    @feedback_input
    def forward(self, output: Dict[str, Any], memory, **states):
        feats = output['feat']
        feats = feats.detach().cpu()
        logging.info(f'Shape of features {feats.shape}.')
        logging.info(f'Save to {self.file}')
        torch.save(feats, self.file)


class SimilarCompareAction(BasicAction, MemoryMixin):
    def __init__(self, file_1, file_2):
        MemoryMixin.__init__(self)
        self.file_1 = file_1
        self.file_2 = file_2

    @feedback_input
    def forward(self, device, memory, **states):
        with torch.inference_mode():
            model = memory.model
            model.eval()

            f1 = torch.load(self.file_1).to(device)
            f2 = torch.load(self.file_2).to(device)
            n_f1, n_f2 = f1.shape[0], f2.shape[0]
            feats = torch.concat([f1, f2])

            logits = model.model.calc_sim(feats)

            scale = logits[0, 0]
            logging.info(f'Model scale: {scale:.4f}')

            scores = torch.clip((scale - logits) / (scale * 2), 0.0, 1.0).detach().cpu()
            cropped = scores[:n_f1, n_f1:]
            logging.info(f'Mean: {cropped.mean():.3f}, Std: {cropped.std():.3f}')
            # return cropped.mean()
