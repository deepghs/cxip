import glob
import os.path
from functools import partial

import torch
from ditk import logging
from huggingface_hub import hf_hub_download
from natsort import natsorted
from rainbowneko.infer import WorkflowRunner, LoadImageAction, BuildModelAction, \
    PrepareAction, LoadModelAction
from rainbowneko.models.wrapper import FeatWrapper
from torchvision import transforms

from actions import CXIPForwardAction, SaveFeatureAction, SimilarCompareAction
from model import CAFormerBackbone

logging.try_init_root(logging.INFO)

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

base_actions = [
    PrepareAction(device='cuda', dtype=torch.float32),
    BuildModelAction(partial(FeatWrapper, model=CAFormerBackbone('caformer_m36'))),
    LoadModelAction({'model': hf_hub_download(
        repo_id='deepghs/csip_model',
        filename='csip-caformer-m36-12000.ckpt',
    )}),
]


class CHMRunner:
    def __init__(self, base_actions):
        self.runner = WorkflowRunner()
        self.states = self.runner.run(base_actions)

    def _run(self, actions):
        self.states = self.runner.run(actions)

    def extract_images_save_feat(self, directory, pt_file):
        image_files = natsorted(glob.glob(os.path.join(directory, '**', '*.jpg'), recursive=True))
        self.states = self.runner.run([
            LoadImageAction(
                image_paths=image_files,
                image_transforms=EVAL_TRANSFORM,
            ),
            CXIPForwardAction(bs=32),
            SaveFeatureAction(pt_file)
        ], self.states)

    def compare_feats(self, pt_file_1, pt_file_2):
        self.states = self.runner.run([
            SimilarCompareAction(pt_file_1, pt_file_2),
        ], self.states)


if __name__ == '__main__':
    runner = CHMRunner(base_actions)
    # runner.extract_images_save_feat('test_clsx/1', 'test_feat_1.pt')
    # runner.extract_images_save_feat('test_clsx/2', 'test_feat_2.pt')
    runner.extract_images_save_feat('test_cls/1', 'test_feat_x1.pt')
    runner.extract_images_save_feat('test_cls/4', 'test_feat_x2.pt')
    runner.extract_images_save_feat('test_cls/7', 'test_feat_x3.pt')
    runner.extract_images_save_feat('test_cls/11', 'test_feat_x4.pt')
    runner.extract_images_save_feat('test_cls/20', 'test_feat_x5.pt')
    runner.compare_feats('test_feat_1.pt', 'test_feat_2.pt')
    runner.compare_feats('test_feat_1.pt', 'test_feat_1.pt')
    runner.compare_feats('test_feat_1.pt', 'test_feat_x1.pt')
    runner.compare_feats('test_feat_1.pt', 'test_feat_x5.pt')
