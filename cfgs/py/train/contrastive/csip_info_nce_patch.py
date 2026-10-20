# 2024 CSIP v1: 随机小块拼接 (CropAndStitch) 打乱内容，NoisyInfoNCE
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
import random
from PIL import Image

from cfgs.py.train import train_base, tuning_base
from evaluate import CXIPMetricContainer
from loss import NoisyInfoNCELoss
from model import CAFormerBackbone, SwinV2Backbone

KEY_MAP = ('pred.pred -> pred', 'inputs.label -> label')
FEAT_KEY_MAP = ('pred.feat.0 -> feat', 'inputs.label -> label')

img_size = 448

class WeakRandAugment2(T.RandAugment):
    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[torch.Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity":(torch.tensor(0.0), False),
            "ShearX":(torch.linspace(0.0, 0.2, num_bins), True),
            "ShearY":(torch.linspace(0.0, 0.2, num_bins), True),
            #"TranslateX":(torch.linspace(0.0, 0.08*image_size[1], num_bins), True),
            #"TranslateY":(torch.linspace(0.0, 0.08*image_size[0], num_bins), True),
            "Rotate":(torch.linspace(0.0, 30.0, num_bins), True),
        }

class CropAndStitch:
    def __init__(self, img_size=384, n=12):
        assert img_size % n == 0
        self.n = n
        self.m = img_size // n

    def __call__(self, image):
        width, height = image.size

        # 创建一个新的空白图像,用于拼接裁剪后的图像块
        stitched_image = Image.new('RGB', (self.n * self.m, self.n * self.m))

        # 随机裁剪图像块并拼接
        for i in range(self.n):
            for j in range(self.n):
                # 随机生成裁剪位置的左上角坐标
                left = random.randint(0, width - self.m)
                top = random.randint(0, height - self.m)

                # 裁剪图像块
                patch = image.crop((left, top, left + self.m, top + self.m))

                # 将裁剪后的图像块拼接到新图像上
                stitched_image.paste(patch, (i * self.m, j * self.m))

        return stitched_image


@neko_cfg
def make_cfg():
    return dict(
        _base_=[train_base, tuning_base],
        exp_dir='exps/csip_v1_noisy_info_nce_swinv2-p448-patch',

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
            train_epochs=100,
            workers=8,
            max_grad_norm=None,
            save_step=1000,

            loss=LossGroup(nce=LossContainer(NoisyInfoNCELoss(temperature=1/16.), key_map=FEAT_KEY_MAP)),

            optimizer=partial(torch.optim.AdamW, weight_decay=5e-4),

            scale_lr=False,
            lr_scheduler=CosineLR(
                _partial_=True,
                warmup_steps=100,
            ),
            metrics=None,
        ),

        model=dict(
            name='csip-swinv2-base',
            # 原配置这里是 input_resolution=384，与 img_size=448 的输入对不上 (AttentionPool2d 的位置编码按分辨率固定)，改为 img_size
            wrapper=partial(FeatWrapper, model=CAFormerBackbone('caformer_m36', input_resolution=img_size)),
            #wrapper=partial(FeatWrapper, model=SwinV2Backbone('hf-hub:SmilingWolf/wd-swinv2-tagger-v3', img_size=img_size)),
        ),

        data_train=cfg_data(),
        evaluator=cfg_evaluator(),
    )

@neko_cfg
def cfg_data():
    return dict(
        dataset1=partial(BaseDataset, batch_size=64, loss_weight=1.0,
            source=dict(
                data_source1=ImageFolderClassSource(img_root='/root/autodl-tmp/datas/csip_v1/train'),
            ),
            handler=HandlerChain(
                load=LoadImageHandler(),
                image=ImageHandler(transform=T.Compose([
                        T.Resize(640),
                        WeakRandAugment2(),
                        T.RandomHorizontalFlip(),
                        #T.RandomCrop(384),
                        CropAndStitch(img_size, 14),
                        T.ToTensor(),
                        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]),
                )
            ),
            bucket=PosNegBucket(target_size=img_size, pos_rate=0.5),
        )
    )

@neko_cfg
def cfg_evaluator():
    return partial(Evaluator,
        interval=100,
        metric=MetricGroup(
            AP=CXIPMetricContainer(AveragePrecision(task="binary"), key_map=KEY_MAP),
        ),
        dataset=partial(BaseDataset, batch_size=64, loss_weight=1.0,
            source=dict(
                data_source1=ImageFolderClassSource(img_root='/root/autodl-tmp/datas/csip_v1/eval'),
            ),
            handler=HandlerChain(
                load=LoadImageHandler(),
                image=ImageHandler(transform=T.Compose([
                        T.Resize(640),
                        T.CenterCrop(img_size),
                        T.ToTensor(),
                        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]),
                )
            ),
            bucket=PosNegBucket(target_size=img_size, pos_rate=0.5),
        )
    )
