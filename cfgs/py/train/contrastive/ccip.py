from functools import partial
from typing import Tuple, Dict

import torch
import torchvision.transforms as T
from rainbowneko.ckpt_manager import ckpt_saver, NekoResumer, NekoModelLoader
from rainbowneko.data import BaseDataset
from rainbowneko.data import PosNegBucket
from rainbowneko.data.handler import HandlerChain, ImageHandler, LoadImageHandler
from rainbowneko.data.source import ImageFolderClassSource
from rainbowneko.evaluate import MetricGroup, Evaluator
from rainbowneko.models.wrapper import FeatWrapper
from rainbowneko.parser import CfgWDModelParser, neko_cfg
from rainbowneko.train.loss import MLCEImageLoss
from rainbowneko.utils import CosineLR
from torchmetrics.classification import AveragePrecision, AUROC

from cfgs.py.train import train_base, tuning_base
from evaluate import CXIPMetricContainer
from model import CAFormerBackbone


class WeakRandAugment2(T.RandAugment):
    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[torch.Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity":(torch.tensor(0.0), False),
            "ShearX":(torch.linspace(0.0, 0.08, num_bins), True),
            "ShearY":(torch.linspace(0.0, 0.08, num_bins), True),
            "TranslateX":(torch.linspace(0.0, 0.08*image_size[1], num_bins), True),
            "TranslateY":(torch.linspace(0.0, 0.08*image_size[0], num_bins), True),
            "Rotate":(torch.linspace(0.0, 8.0, num_bins), True),
            "Brightness":(torch.linspace(0.0, 0.05, num_bins), True),
            "Contrast":(torch.linspace(0.0, 0.05, num_bins), True),
            "Sharpness":(torch.linspace(0.0, 0.2, num_bins), True),
            "Posterize":(8-(torch.arange(num_bins)/((num_bins-1)/4)).round().int(), False),
            "AutoContrast":(torch.tensor(0.0), False),
            "Equalize":(torch.tensor(0.0), False),
        }

ANIME_MEAN = (0.67713961, 0.618461,   0.61174436)
ANIME_STD = (0.31741801, 0.31876716, 0.31473021)


@neko_cfg
def make_cfg():
    return dict(
        _base_=[train_base, tuning_base],

        model_part=CfgWDModelParser([
            dict(
                lr=1e-4,
                layers=[
                    'model.caformer.attnpool',
                    'model.caformer.sim',
                ],  # train all layers
            )
        ], weight_decay=1e-2),

        # func(_partial_=True, ...) same as partial(func, ...)
        ckpt_saver=dict(
            model=ckpt_saver(target_module='model'),
        ),

        train=dict(
            train_epochs=10,
            workers=2,
            max_grad_norm=None,
            save_step=2000,

            loss=MLCEImageLoss(),

            resume=NekoResumer(
                start_step=0,
                loader=dict(
                    model=NekoModelLoader(
                        target_module='model.caformer',
                        path='exps/anime_dinov2_bs256-v2.1-lr4e-4/ckpts/model-4.safetensors'
                    ),
                )
            ),

            optimizer=partial(torch.optim.AdamW, weight_decay=1e-2),

            scale_lr=False,
            lr_scheduler=CosineLR(
                _partial_=True,
                warmup_steps=5000,
            ),
            metrics=MetricGroup(
                AP=CXIPMetricContainer(AveragePrecision(task="binary")),
                roc=CXIPMetricContainer(AUROC(task="binary")),
            ),
        ),

        model=dict(
            name='ccip-dino',
            wrapper=partial(FeatWrapper, model=CAFormerBackbone('caformer_b36', input_resolution=384))
        ),

        data_train=cfg_data(), # config can be split into another function with @neko_cfg

        evaluator=cfg_evaluator(),
    )

@neko_cfg
def cfg_data():
    return dict(
        dataset1=partial(BaseDataset, batch_size=128, loss_weight=1.0,
            source=dict(
                data_source1=ImageFolderClassSource(
                    img_root=""
                ),
            ),
            handler=HandlerChain(
                load=LoadImageHandler(),
                image=ImageHandler(transform=T.Compose([
                        T.Resize((416, 416)),
                        T.RandomHorizontalFlip(),
                        WeakRandAugment2(),
                        T.RandomCrop(384),
                        T.ToTensor(),
                        T.Normalize(mean=ANIME_MEAN, std=ANIME_STD),
                    ]),
                )
            ),
            bucket=PosNegBucket(target_size=384, pos_rate=0.5),
        )
    )

@neko_cfg
def cfg_evaluator():
    return partial(Evaluator,
        interval=50,
        metric=MetricGroup(
            AP=CXIPMetricContainer(AveragePrecision(task="binary")),
            roc=CXIPMetricContainer(AUROC(task="binary")),
        ),
        dataset=partial(BaseDataset, batch_size=128, loss_weight=1.0,
            source=dict(
                data_source1=ImageFolderClassSource(
                    img_root=""
                ),
            ),
            handler=HandlerChain(
                load=LoadImageHandler(),
                image=ImageHandler(transform=T.Compose([
                        T.Resize((384, 384)),
                        T.ToTensor(),
                        T.Normalize(mean=ANIME_MEAN, std=ANIME_STD),
                    ]),
                )
            ),
            bucket=PosNegBucket(target_size=384, pos_rate=0.5),
        )
    )