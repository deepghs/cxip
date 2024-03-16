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
