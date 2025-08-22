from functools import partial
from typing import Tuple, Dict
import webdataset as wds

import torch
import torchvision.transforms as T
from rainbowneko.ckpt_manager import ckpt_saver, NekoResumer, NekoModelLoader
from rainbowneko.data import BaseDataset, ImageLabelSource, WebDataset
from rainbowneko.data import PosNegBucket, BaseBucket
from rainbowneko.data.handler import HandlerChain, ImageHandler, LoadImageHandler
from rainbowneko.data.source import ImageFolderClassSource
from rainbowneko.evaluate import MetricGroup, Evaluator
from rainbowneko.models.wrapper import FeatWrapper
from rainbowneko.parser import CfgWDModelParser, neko_cfg
from rainbowneko.train.loss import LossContainer
from rainbowneko.utils import CosineLR
from torchmetrics.classification import AveragePrecision, AUROC, CalibrationError

from cfgs.py.train import train_base, tuning_base
from evaluate import CXIPMetricContainer, cohens_d
from model import CAFormerBackbone, CAFormerStyleBackbone
from data import StyleImageLabelSource, UniformPatch
from loss import SiglipLoss

class WeakRandAugment2(T.RandAugment):
    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[torch.Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity":(torch.tensor(0.0), False),
            # "ShearX":(torch.linspace(0.0, 0.08, num_bins), True),
            # "ShearY":(torch.linspace(0.0, 0.08, num_bins), True),
            "TranslateX":(torch.linspace(0.0, 0.08*image_size[1], num_bins), True),
            "TranslateY":(torch.linspace(0.0, 0.08*image_size[0], num_bins), True),
            # "Rotate":(torch.linspace(0.0, 30.0, num_bins), True),
            # "Color": (torch.linspace(0.0, 0.03, num_bins), True),
            # "Brightness":(torch.linspace(0.0, 0.05, num_bins), True),
            # "Contrast":(torch.linspace(0.0, 0.05, num_bins), True),
            # "Sharpness":(torch.linspace(0.0, 0.2, num_bins), True),
            # "Posterize":(8-(torch.arange(num_bins)/((num_bins-1)/4)).round().int(), False),
            # "AutoContrast":(torch.tensor(0.0), False),
            # "Equalize":(torch.tensor(0.0), False),
        }

ANIME_MEAN = (0.67713961, 0.618461,   0.61174436)
ANIME_STD = (0.31741801, 0.31876716, 0.31473021)


@neko_cfg
def make_cfg():
    return dict(
        _base_=[train_base, tuning_base],
        exp_dir=f'exps/csip_dino_style',

        model_part=CfgWDModelParser([
            dict(
                lr=1e-5,
                layers=[
                    '',
                    # 'model.attnpool',
                    # 'model.sim',
                ],  # train all layers
            )
        ], weight_decay=1e-2),

        # func(_partial_=True, ...) same as partial(func, ...)
        ckpt_saver=dict(
            model=ckpt_saver(target_module='model'),
        ),

        train=dict(
            train_epochs=2,
            workers=2,
            max_grad_norm=None,
            save_step=1000,

            loss=LossContainer(SiglipLoss()),

            resume=NekoResumer(
                start_step=0,
                loader=dict(
                    model=NekoModelLoader(
                        target_module='model.caformer',
                        state_prefix='model_student.backbone.',
                        path='/data/anime_dino/exps/anime_dinov2_bs128-lr1e-3/ckpts/model-191000.safetensors'
                    ),
                )
            ),

            optimizer=partial(torch.optim.AdamW, weight_decay=1e-2),

            scale_lr=False,
            lr_scheduler=CosineLR(
                _partial_=True,
                warmup_steps=100,
            ),
            metrics=MetricGroup(
                AP=CXIPMetricContainer(AveragePrecision(task="binary"), key_map=('pred.pred -> pred', 'inputs.label -> label')),
                roc=CXIPMetricContainer(AUROC(task="binary"), key_map=('pred.pred -> pred', 'inputs.label -> label')),
                cohens=CXIPMetricContainer(cohens_d, key_map=('pred.pred -> pred', 'inputs.label -> label')),
            ),
        ),

        model=dict(
            name='csip-dino',
            wrapper=partial(FeatWrapper, model=CAFormerStyleBackbone('caformer_b36', input_resolution=384))
        ),

        data_train=cfg_data(), # config can be split into another function with @neko_cfg

        evaluator=cfg_evaluator(),
    )

@neko_cfg
def cfg_data():
    return dict(
        dataset1=partial(BaseDataset, batch_size=128, loss_weight=1.0,
            source=dict(
                data_source1=StyleImageLabelSource(
                    img_root="/nfs/danbooru-w640-ws-full-artist/train",
                    label_file="/nfs/danbooru-w640-ws-full-artist/train_split.json",
                ),
            ),
            handler=HandlerChain(
                load=LoadImageHandler(),
                image=ImageHandler(transform=T.Compose([
                        T.Resize((384, 384)),
                        T.RandomHorizontalFlip(),
                        WeakRandAugment2(),
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
        interval=5000,
        metric=MetricGroup(
            AP=CXIPMetricContainer(AveragePrecision(task="binary"), key_map=('pred.pred -> pred', 'inputs.label -> label')),
            roc=CXIPMetricContainer(AUROC(task="binary"), key_map=('pred.pred -> pred', 'inputs.label -> label')),
            cohens=CXIPMetricContainer(cohens_d, key_map=('pred.pred -> pred', 'inputs.label -> label')),
        ),
        dataset=partial(BaseDataset, batch_size=64, loss_weight=1.0,
            source=dict(
                data_source1=StyleImageLabelSource(
                    img_root="/nfs/danbooru-w640-ws-full-artist/train",
                    label_file="/nfs/danbooru-w640-ws-full-artist/test_split.json",
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