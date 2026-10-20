# 2024 CSIP 三元组路线: VGG19 多层特征的 Gram 矩阵距离 (style loss)
from functools import partial
from typing import Tuple, Dict

import torch
import torchvision.transforms as T
from rainbowneko.ckpt_manager import ckpt_saver
from rainbowneko.data import BaseDataset, TripletBucket
from rainbowneko.data.handler import HandlerChain, ImageHandler, LoadImageHandler
from rainbowneko.data.source import ImageFolderClassSource
from rainbowneko.evaluate import MetricGroup, Evaluator
from rainbowneko.loggers import CLILogger, TBLogger
from rainbowneko.models.wrapper import FeatWrapper
from rainbowneko.parser import CfgModelParser, neko_cfg
from rainbowneko.train.loss import LossContainer, LossGroup
from rainbowneko.utils import CosineLR
from torchmetrics.classification import AveragePrecision

from cfgs.py.train import train_base, tuning_base
from evaluate import CSIPTripletAPContainer
from loss import TripletStyleLoss
from model import VGGStyleBackbone

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

class WeakRandAugment2(T.RandAugment):
    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[torch.Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity":(torch.tensor(0.0), False),
            "ShearX":(torch.linspace(0.0, 0.2, num_bins), True),
            "ShearY":(torch.linspace(0.0, 0.2, num_bins), True),
            "TranslateX":(torch.linspace(0.0, 0.08*image_size[1], num_bins), True),
            "TranslateY":(torch.linspace(0.0, 0.08*image_size[0], num_bins), True),
            "Rotate":(torch.linspace(0.0, 30.0, num_bins), True),
            #"Brightness":(torch.linspace(0.0, 0.5, num_bins), True),
            #"Contrast":(torch.linspace(0.0, 0.05, num_bins), True),
            #"Sharpness":(torch.linspace(0.0, 0.5, num_bins), True),
            #"Posterize":(8-(torch.arange(num_bins)/((num_bins-1)/4)).round().int(), False),
            #"AutoContrast":(torch.tensor(0.0), False),
            #"Equalize":(torch.tensor(0.0), False),
        }


@neko_cfg
def make_cfg():
    return dict(
        _base_=[train_base, tuning_base],
        exp_dir='exps/csip_vgg19_styleloss-2',

        model_part=CfgModelParser([
            dict(
                lr=1e-4,
                layers=[
                    '',
                ],  # train all layers
            ),
        ]),

        ckpt_saver=dict(
            model=ckpt_saver(target_module='model'),
        ),

        logger=[
            partial(CLILogger, out_path='train.log', log_step=20),
            partial(TBLogger, out_path='tb', log_step=20),
        ],

        train=dict(
            train_epochs=50,
            workers=4,
            max_grad_norm=None,
            save_step=2000,

            loss=LossGroup(style=LossContainer(TripletStyleLoss(), key_map=('pred.feat -> feat',))),

            optimizer=partial(torch.optim.AdamW, weight_decay=5e-4),

            scale_lr=False,
            lr_scheduler=CosineLR(
                _partial_=True,
                warmup_steps=100,
            ),
            metrics=None,
        ),

        model=dict(
            name='csip-vgg-19-conv5',
            wrapper=partial(FeatWrapper, model=VGGStyleBackbone()),
        ),

        # TripletBucket: batch_size 要是 3 的倍数
        data_train=cfg_data(),
        evaluator=cfg_evaluator(),
    )

@neko_cfg
def cfg_data():
    return dict(
        dataset1=partial(BaseDataset, batch_size=120, loss_weight=1.0,
            source=dict(
                data_source1=ImageFolderClassSource(img_root='/data/csip_v1'),
            ),
            handler=HandlerChain(
                load=LoadImageHandler(),
                image=ImageHandler(transform=T.Compose([
                        T.Resize(512),
                        WeakRandAugment2(),
                        T.RandomHorizontalFlip(),
                        T.RandomCrop(224),
                        T.ToTensor(),
                        T.Normalize(CLIP_MEAN, CLIP_STD),
                    ]),
                )
            ),
            bucket=TripletBucket(target_size=224),
        )
    )

@neko_cfg
def cfg_evaluator():
    return partial(Evaluator,
        interval=100,
        metric=MetricGroup(
            AP=CSIPTripletAPContainer(AveragePrecision(task="binary")),
        ),
        dataset=partial(BaseDataset, batch_size=60, loss_weight=1.0,
            source=dict(
                data_source1=ImageFolderClassSource(img_root='/data/csip_eval_v0'),
            ),
            handler=HandlerChain(
                load=LoadImageHandler(),
                image=ImageHandler(transform=T.Compose([
                        T.Resize(512),
                        T.CenterCrop(224),
                        T.ToTensor(),
                        T.Normalize(CLIP_MEAN, CLIP_STD),
                    ]),
                )
            ),
            bucket=TripletBucket(target_size=224),
        )
    )
