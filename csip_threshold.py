import math
import math
import os.path
from functools import partial

import numpy as np
import torch
from ditk import logging
from huggingface_hub import hf_hub_download
from rainbowneko.infer import WorkflowRunner, LoadImageAction, BuildModelAction, \
    PrepareAction, LoadModelAction, BasicAction, feedback_input, VisPredAction
from rainbowneko.models.wrapper import FeatWrapper
from torchvision import transforms

from actions import ContrastiveAnalysisAction, ClusterTestAction, CXIPForwardAction
from model import CAFormerBackbone

logging.try_init_root(logging.INFO)

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])


class ReshapeAction(BasicAction):
    @feedback_input
    def forward(self, output, **states):
        bs = int(math.sqrt(len(output['pred'])))
        output['pred'] = output['pred'].view(bs, bs)
        return {'output': output}


eval_dataset_dir = '/data/csip_eval_v0_tiny'
image_files = natsorted(glob.glob(os.path.join(eval_dataset_dir, '*', '*.jpg')))
# image_files = [
#     'test_cls/1/1cf603fd8dc873023793ac9d93912a45.jpg',
#     'test_cls/1/3b5312cc5dfc9f1b2becb8dce388d38f.jpg',
#     'test_cls/1/4cb73db283d6853112e4cd150ccd95ef.jpg',
#
#     'test_cls/11/1b09db63a6788dc2141e21c7638b8f77.jpg',
#     'test_cls/11/1c7ebaf143ea7a29937b428cd4e4bc25.jpg',
#     'test_cls/11/3b0001960cb1ae97b4159a97244b485d.jpg',
#
#     'test_cls/20/08a8719df9e6e6f920f38e23e7f7f823.jpg',
#     'test_cls/20/14ba62a94e8c25c366e1136cffb3d9fd.jpg',
#     'test_cls/20/42a8e8faf8e2773e58492162c591ec30.jpg',
#
#     'test_cls/4/11fcec9e3c5c725de47c52fc074f6f20.jpg',
#     'test_cls/4/74d3b3fe027c0454d26f1e6ff94c40bb.jpg',
#     'test_cls/4/9112fec40d41aad47af413e5fe0a5a05.jpg',
#
#     'test_cls/7/2fb475e28eb7f010ceeccbf9dde4d44a.jpg',
#     'test_cls/7/07940f6c7b3543f82c49fe5af94d975d.jpg',
#     'test_cls/7/8307e50a78d91ad61bc7689faf7c5a64.jpg',
# ]
image_cls = np.array([os.path.basename(os.path.dirname(f)) for f in image_files])

actions = [
    PrepareAction(device='cuda', dtype=torch.float32),
    BuildModelAction(partial(FeatWrapper, model=CAFormerBackbone('caformer_m36'))),
    LoadModelAction({'model': hf_hub_download(
        repo_id='deepghs/csip_model',
        filename='csip-caformer-m36-12000.ckpt',
    )}),

    LoadImageAction(
        image_paths=image_files,
        image_transforms=EVAL_TRANSFORM,
    ),
    # CatImageAction(),
    CXIPForwardAction(bs=32),
    # ReshapeAction(),
    VisPredAction(),
    ContrastiveAnalysisAction(image_cls),
    ClusterTestAction()
]

if __name__ == '__main__':
    runner = WorkflowRunner()
    runner.run(actions)
