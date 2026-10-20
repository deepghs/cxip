# 2024 cwip 路线 (用途推测是服装/cosplay): CAFormer-M36 + InfoNCE
from functools import partial
from typing import Tuple, Dict

import torch
import torchvision.transforms as T
from rainbowneko.ckpt_manager import ckpt_saver
from rainbowneko.data import BaseDataset, PosNegBucket
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
from evaluate import CXIPMetricContainer
from loss import InfoNCELoss
from model import CAFormerBackbone

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
KEY_MAP = ('pred.pred -> pred', 'inputs.label -> label')
FEAT_KEY_MAP = ('pred.feat.0 -> feat', 'inputs.label -> label')

class WeakRandAugment2(T.RandAugment):
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


@neko_cfg
def make_cfg():
    return dict(
        _base_=[train_base, tuning_base],
        exp_dir='exps/cwip_info_nce-416-50_v0.1_raw',

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
            partial(TBLogger, out_path='tb_log', log_step=10),
        ],

        train=dict(
            train_epochs=200,
            workers=2,
            max_grad_norm=None,
            save_step=100,

            loss=LossGroup(nce=LossContainer(InfoNCELoss(), key_map=FEAT_KEY_MAP)),

            optimizer=partial(torch.optim.AdamW, weight_decay=5e-4),

            scale_lr=False,
            lr_scheduler=CosineLR(
                _partial_=True,
                warmup_steps=100,
            ),
            metrics=None,
        ),

        model=dict(
            name='cwip-caformer-m36',
            wrapper=partial(FeatWrapper, model=CAFormerBackbone('caformer_m36', input_resolution=384)),
        ),

        data_train=cfg_data(),
        evaluator=cfg_evaluator(),
    )

@neko_cfg
def cfg_data():
    return dict(
        dataset1=partial(BaseDataset, batch_size=128, loss_weight=1.0,
            source=dict(
                data_source1=ImageFolderClassSource(img_root='/data/cwip_v0.1'),
            ),
            handler=HandlerChain(
                load=LoadImageHandler(),
                image=ImageHandler(transform=T.Compose([
                        T.Resize(416),
                        WeakRandAugment2(),
                        T.RandomHorizontalFlip(),
                        T.RandomResizedCrop(384),
                        T.ToTensor(),
                        T.Normalize(CLIP_MEAN, CLIP_STD),
                    ]),
                )
            ),
            bucket=PosNegBucket(target_size=384, pos_rate=0.5),
        )
    )

@neko_cfg
def cfg_evaluator():
    return partial(Evaluator,
        interval=100,
        metric=MetricGroup(
            AP=CXIPMetricContainer(AveragePrecision(task="binary"), key_map=KEY_MAP),
        ),
        dataset=partial(BaseDataset, batch_size=128, loss_weight=1.0,
            source=dict(
                data_source1=ImageFolderClassSource(img_root='/data/cwip_v0.1_eval'),
            ),
            handler=HandlerChain(
                load=LoadImageHandler(),
                image=ImageHandler(transform=T.Compose([
                        T.Resize(416),
                        T.CenterCrop(384),
                        T.ToTensor(),
                        T.Normalize(CLIP_MEAN, CLIP_STD),
                    ]),
                )
            ),
            bucket=PosNegBucket(target_size=384, pos_rate=0.5),
        )
    )
