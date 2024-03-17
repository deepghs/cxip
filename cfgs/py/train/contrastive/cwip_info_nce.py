import random
from functools import partial
from typing import Tuple, Dict

import torch
import torchvision
from rainbowneko.ckpt_manager import CkptManagerPKL
from rainbowneko.evaluate import EvaluatorGroup
from rainbowneko.models.wrapper import FeatWrapper
from rainbowneko.train.data import ImageLabelDataset
from rainbowneko.train.data.bucket import PosNegBucket
from rainbowneko.train.data.source import ImageFolderClassSource
from rainbowneko.train.loggers import CLILogger, TBLogger
from rainbowneko.train.loss import InfoNCELoss
from torch import nn
from torchmetrics.classification import AveragePrecision
from torchvision import transforms

from evaluate import CSIPmAPContainer
from model import CAFormerBackbone

num_classes = 10


def load_resnet():
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class WeakRandAugment2(transforms.RandAugment):
    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[torch.Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.2, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.2, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 0.08 * image_size[1], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 0.08 * image_size[0], num_bins), True),
            "Rotate": (torch.linspace(0.0, 60.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.3, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }


class HeadCutAugment:
    def __init__(self, p: float = 0.6):
        self.p = p

    def __call__(self, image):
        r = random.random()
        if r < self.p / 2:
            from waifuc.action import HeadCutOutAction
            from waifuc.model import ImageItem
            try:
                item = next(HeadCutOutAction().iter(ImageItem(image)))
            except StopIteration:
                return image
            else:
                return item.image

        elif r < self.p:
            from waifuc.action import HeadCoverAction
            from waifuc.model import ImageItem
            random_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
            action = HeadCoverAction(scale=0.65 + random.random() * (1.4 - 0.65), color=random_color)
            return action.process(ImageItem(image)).image

        else:
            return image


TRAIN_TRANSFORM = transforms.Compose([
    HeadCutAugment(p=0.6),
    transforms.Resize(416),
    WeakRandAugment2(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(384),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(416),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

config = dict(
    _base_=[
        'cfgs/py/train/train_base.py',
        'cfgs/py/train/tuning_base.py',
    ],

    model_part=[
        dict(
            lr=1e-4,
            layers=[''],  # train all layers
        )
    ],

    ckpt_manager=partial(CkptManagerPKL, saved_model=(
        {'model': 'model', 'trainable': False},
    )),

    exp_dir='exps/cwip_info_nce-416-50_v0.1',
    logger=[
        partial(CLILogger, out_path='train.log', log_step=20),
        partial(TBLogger, out_path='tb_log', log_step=10),
    ],

    train=dict(
        train_epochs=200,
        workers=2,
        max_grad_norm=None,
        save_step=100,

        loss=partial(InfoNCELoss),

        optimizer=partial(torch.optim.AdamW, weight_decay=5e-4),

        scale_lr=False,
        scheduler=dict(
            name='cosine',
            num_warmup_steps=100,
        ),
        metrics=None,
    ),

    model=dict(
        name='csip-caformer-m36',
        wrapper=partial(FeatWrapper, model=CAFormerBackbone('caformer_m36', input_resolution=384))
    ),

    evaluator=partial(EvaluatorGroup, interval=100,
                      evaluator_dict=dict(
                          AP=CSIPmAPContainer(AveragePrecision(task="binary")),
                      )
                      ),

    data_train=dict(
        dataset1=partial(ImageLabelDataset, batch_size=128, loss_weight=1.0,
                         source=dict(
                             data_source1=ImageFolderClassSource(
                                 img_root=r'/data/cwip',
                                 image_transforms=TRAIN_TRANSFORM,
                             ),
                         ),
                         bucket=PosNegBucket(target_size=384, pos_rate=0.5),
                         )
    ),

    data_eval=dict(
        dataset1=partial(ImageLabelDataset, batch_size=128, loss_weight=1.0,
                         source=dict(
                             data_source1=ImageFolderClassSource(
                                 img_root=r'/data/cwip_eval',
                                 image_transforms=EVAL_TRANSFORM,
                             ),
                         ),
                         bucket=PosNegBucket(target_size=384, pos_rate=0.5),
                         )
    ),
)
