import glob
import os.path
from functools import partial

import numpy as np
import pandas as pd
import torch
from ditk import logging
from hbutils.reflection import progressive_for
from huggingface_hub import hf_hub_download
from natsort import natsorted
from rainbowneko.infer import WorkflowRunner, LoadImageAction, BuildModelAction, \
    PrepareAction, LoadModelAction
from rainbowneko.models.wrapper import FeatWrapper
from torchvision import transforms
from tqdm import tqdm

from actions import CXIPForwardAction, SaveFeatureAction, SimilarCompareAction
from model import CAFormerBackbone

logging.try_init_root(logging.INFO)

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(384),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

base_actions = [
    PrepareAction(device='cuda', dtype=torch.float32),
    BuildModelAction(partial(FeatWrapper, model=CAFormerBackbone('caformer_m36'))),
    LoadModelAction({'model': hf_hub_download(
        repo_id='deepghs/cwip_models',
        filename='cwip-caformer-m36-8100.ckpt',
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
        return self.states['mean']


if __name__ == '__main__':
    runner = CHMRunner(base_actions)

    eval_dataset_dir = '/data/cwip'
    pt_dir = 'test_pts_cwip'
    os.makedirs(pt_dir, exist_ok=True)
    for d in tqdm(os.listdir(eval_dataset_dir)):
        runner.extract_images_save_feat(
            os.path.join(eval_dataset_dir, d),
            os.path.join(pt_dir, f'{d}.pt'),
        )

    pt_dir = 'test_pts_cwip'
    pt_files = np.array(natsorted(glob.glob(os.path.join(pt_dir, '*.pt'))))
    diff_data = []
    for if1, if2 in tqdm(list(progressive_for(range(pt_files.shape[0]), n=2))):
        pt_file_1 = pt_files[if1]
        pt_file_2 = pt_files[if2]

        mean_diff = runner.compare_feats(pt_file_1, pt_file_2)
        diff_data.append({
            'artist_id_1': os.path.basename(pt_file_1).split('.')[0],
            'pt_file_1': pt_file_1,
            'artist_id_2': os.path.basename(pt_file_2).split('.')[0],
            'pt_file_2': pt_file_2,
            'diff': mean_diff.detach().numpy().item(),
        })

    df_diff = pd.DataFrame(diff_data)
    print(df_diff)
    print(df_diff[df_diff['diff'] < 0.30])
    df_diff.to_csv('test_data_diff_cwip.csv', index=False)

    same_data = []
    for if1 in tqdm(range(pt_files.shape[0])):
        pt_file_1 = pt_files[if1]
        mean_diff = runner.compare_feats(pt_file_1, pt_file_1)
        same_data.append({
            'artist_id_1': os.path.basename(pt_file_1).split('.')[0],
            'pt_file_1': pt_file_1,
            'count': len(os.listdir(os.path.join(eval_dataset_dir, os.path.basename(pt_file_1).split('.')[0]))),
            'diff': mean_diff.detach().numpy().item(),
        })

    df_same = pd.DataFrame(same_data)
    print(df_same)
    # print(df_same[df_same['diff'] >= 0.25])
    df_same.to_csv('test_data_same_cwip.csv', index=False)

    # runner.extract_images_save_feat('test_cls/4', 'test_feat_x2.pt')
    # runner.extract_images_save_feat('test_cls/7', 'test_feat_x3.pt')
    # runner.extract_images_save_feat('test_cls/11', 'test_feat_x4.pt')
    # runner.extract_images_save_feat('test_cls/20', 'test_feat_x5.pt')
    # runner.compare_feats('test_feat_1.pt', 'test_feat_2.pt')
    # runner.compare_feats('test_feat_1.pt', 'test_feat_1.pt')
    # runner.compare_feats('test_feat_1.pt', 'test_feat_x1.pt')
    # runner.compare_feats('test_feat_1.pt', 'test_feat_x5.pt')
