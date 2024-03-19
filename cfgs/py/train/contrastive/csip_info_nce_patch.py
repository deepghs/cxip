from functools import partial

from typing import Tuple, Dict
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.transforms import RandAugment
from torch.nn import CrossEntropyLoss
from torchmetrics.classification import MulticlassAccuracy, AveragePrecision

from rainbowneko.evaluate import EvaluatorGroup, ClsEvaluatorContainer
from rainbowneko.models.wrapper import SingleWrapper, FeatWrapper
from rainbowneko.train.data.bucket import PosNegBucket
from rainbowneko.train.data.source import IndexSource, ImageFolderClassSource
from rainbowneko.train.loss import MLCEImageLoss, NoisyInfoNCELoss
from rainbowneko.train.data import ImageLabelDataset
from rainbowneko.ckpt_manager import CkptManagerPKL
from rainbowneko.train.loggers import CLILogger, TBLogger
from model import CAFormerBackbone
from evaluate import CSIPmAPContainer
from PIL import Image
import random

num_classes = 10

class WeakRandAugment2(transforms.RandAugment):
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


TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(640),
    WeakRandAugment2(),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomCrop(384),
    CropAndStitch(384, 12),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(640),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

config = dict(
    _base_=[
        'cfgs/py/train/train_base.py',
        'cfgs/py/train/tuning_base.py',
    ],

    exp_dir='exps/csip_v1_noisy_info_nce_s18-p384-patch',
    logger=[
        partial(CLILogger, out_path='train.log', log_step=20),
        partial(TBLogger, out_path='tb_log', log_step=10),
    ],

    model_part=[
        dict(
            lr=1e-4,
            layers=[''],  # train all layers
        )
    ],

    ckpt_manager=partial(CkptManagerPKL, saved_model=(
        {'model':'model', 'trainable':False},
    )),

    train=dict(
        train_epochs=30,
        workers=4,
        max_grad_norm=None,
        save_step=1000,

        loss=partial(NoisyInfoNCELoss, temperature=1/16.),

        optimizer=partial(torch.optim.AdamW, weight_decay=5e-4),

        scale_lr=False,
        scheduler=dict(
            name='cosine',
            num_warmup_steps=100,
        ),
        metrics=None,
    ),

    model=dict(
        name='csip-caformer-s18',
        wrapper=partial(FeatWrapper, model=CAFormerBackbone('caformer_s18', input_resolution=384))
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
                    img_root=r'/root/autodl-tmp/datas/csip_v1/train',
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
                    img_root=r'/root/autodl-tmp/datas/csip_v1/eval',
                    image_transforms=EVAL_TRANSFORM,
                ),
            ),
            bucket=PosNegBucket(target_size=384, pos_rate=0.5),
        )
    ),
)
