import glob
import math
import os.path
import shutil
from functools import partial

import pandas as pd
import torch
from ditk import logging
from huggingface_hub import hf_hub_download
from natsort import natsorted
from rainbowneko.infer import WorkflowRunner, LoadImageAction, BuildModelAction, \
    PrepareAction, LoadModelAction, BasicAction, feedback_input, VisPredAction
from rainbowneko.models.wrapper import FeatWrapper
from torchvision import transforms
from tqdm import tqdm

from actions import CXIPForwardAction
from actions.postprocess import ClusterAction
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


eval_dataset_dir = '/data/cwip/pokemon_cosplay'

base_actions = [
    PrepareAction(device='cuda', dtype=torch.float32),
    BuildModelAction(partial(FeatWrapper, model=CAFormerBackbone('caformer_m36'))),
    LoadModelAction({'model': hf_hub_download(
        repo_id='deepghs/cwip_models',
        filename='cwip-caformer-m36-8100.ckpt',
    )}),
]


class XWorkflow:
    def __init__(self, ):
        self.runner = WorkflowRunner()
        self.states = self.runner.run(base_actions)

    def cluster(self, images_dir, cluster_dir, min_samples: int = 5, max_eps: float = 0.25):
        image_files = natsorted(glob.glob(os.path.join(images_dir, '**', '*.jpg'), recursive=True))
        self.states = self.runner.run([
            LoadImageAction(
                image_paths=image_files,
                image_transforms=EVAL_TRANSFORM,
            ),
            CXIPForwardAction(bs=32),
            VisPredAction(),
            ClusterAction(min_samples=min_samples, max_eps=max_eps),
        ], self.states)

        for label_id, filename in zip(self.states['labels'], image_files):
            dst_file = os.path.join(cluster_dir, str(label_id), os.path.basename(filename))
            os.makedirs(os.path.dirname(dst_file), exist_ok=True)
            shutil.copy(filename, dst_file)


src_dir = '/data/cwip'
# dst_dir = '/data/cwip_cluster'

processed_dir = '/data/cwip_cluster'
dst_dir = '/data/cwip_append'

if __name__ == '__main__':
    runner = XWorkflow()

    df = pd.read_csv(hf_hub_download(
        repo_id='narugo/f',
        repo_type='dataset',
        filename='data_same_cwip.csv'
    ))
    df = df[df['name'].isin(os.listdir(src_dir))]

    selected = df[(df['count'] * df['diff']) >= 9.5].sort_values(['diff'], ascending=False)
    # selected = df[df['name'] == 'jeanne_d_arc_ruler_fate']
    # print(selected)
    # print(len(selected))

    # quit()
    #
    for name in tqdm(os.listdir(processed_dir)):
        ids = map(int, os.listdir(os.path.join(processed_dir, name)))
        ids = sorted(list(filter(lambda x: x > -1 and os.listdir(os.path.join(processed_dir, name, str(x))), ids)))

        for id_ in ids:
            shutil.copytree(
                os.path.join(processed_dir, name, str(id_)),
                os.path.join(dst_dir, f'{name}_p{id_}'),
            )
        shutil.rmtree(os.path.join(src_dir, name))
        print(name, ids)

    # for name in tqdm(sorted(selected['name'])):
    #     runner.cluster(
    #         images_dir=os.path.join(src_dir, name),
    #         cluster_dir=os.path.join(dst_dir, name),
    #         min_samples=5,
    #         max_eps=0.3,
    #     )
