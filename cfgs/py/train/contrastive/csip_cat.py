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
from rainbowneko.train.data.bucket import TripletBucket
from rainbowneko.train.data.source import IndexSource, ImageFolderClassSource
from rainbowneko.train.data import ImageLabelDataset
from rainbowneko.ckpt_manager import CkptManagerPKL
from rainbowneko.train.loggers import CLILogger, TBLogger

from model.loss import StyleBCELoss
from model import CAFormerCatBackbone
from evaluate import CSIP_PN_APContainer

class WeakRandAugment2(transforms.RandAugment):
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

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(512),
    WeakRandAugment2(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(384),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

config = dict(
    _base_=[
        'cfgs/py/train/train_base.py',
        'cfgs/py/train/tuning_base.py',
    ],

    exp_dir='exps/csip_cat',
    model_part=[
        dict(
            lr=2e-4,
            layers=[''],  # train all layers
        )
    ],

    ckpt_manager=partial(CkptManagerPKL, saved_model=(
        {'model':'model', 'trainable':False},
    )),

    train=dict(
        train_epochs=50,
        workers=2,
        max_grad_norm=None,
        save_step=2000,

        loss=partial(StyleBCELoss),

        optimizer=partial(torch.optim.AdamW, weight_decay=5e-4),

        scale_lr=False,
        scheduler=dict(
            name='cosine',
            num_warmup_steps=100,
        ),
        metrics=None,
    ),

    logger=[
        partial(CLILogger, out_path='train.log', log_step=20),
        partial(TBLogger, out_path='tb', log_step=20),
    ],

    model=dict(
        name='csip-caformer-m36',
        wrapper=partial(FeatWrapper, model=CAFormerCatBackbone('caformer_m36'))
    ),

    evaluator=partial(EvaluatorGroup, interval=200,
        evaluator_dict=dict(
            AP=CSIP_PN_APContainer(AveragePrecision(task="binary")),
        )
    ),

    # batch_size要是3的倍数
    data_train=dict(
        dataset1=partial(ImageLabelDataset, batch_size=60, loss_weight=1.0,
            source=dict(
                data_source1=ImageFolderClassSource(
                    img_root=r'/root/autodl-tmp/datas/csip/train',
                    image_transforms=TRAIN_TRANSFORM,
                ),
            ),
            bucket=TripletBucket(target_size=384),
        )
    ),

    data_eval=dict(
        dataset1=partial(ImageLabelDataset, batch_size=60, loss_weight=1.0,
            source=dict(
                data_source1=ImageFolderClassSource(
                    img_root=r'/root/autodl-tmp/datas/csip/eval',
                    image_transforms=EVAL_TRANSFORM,
                ),
            ),
            bucket=TripletBucket(target_size=384),
        )
    ),
)
