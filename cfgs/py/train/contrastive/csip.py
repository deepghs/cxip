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
from rainbowneko.models.wrapper import SingleWrapper
from rainbowneko.train.data.bucket import PosNegBucket
from rainbowneko.train.data.source import IndexSource, ImageFolderClassSource
from rainbowneko.train.loss import MLCEImageLoss
from rainbowneko.train.data import ImageLabelDataset
from rainbowneko.ckpt_manager import CkptManagerPKL

from model import CAFormerBackbone
from evaluate import CSIPmAPContainer

num_classes = 10

def load_resnet():
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

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
            "Brightness":(torch.linspace(0.0, 0.5, num_bins), True),
            "Contrast":(torch.linspace(0.0, 0.05, num_bins), True),
            "Sharpness":(torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize":(8-(torch.arange(num_bins)/((num_bins-1)/4)).round().int(), False),
            "AutoContrast":(torch.tensor(0.0), False),
            "Equalize":(torch.tensor(0.0), False),
        }

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(512),
    WeakRandAugment2(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(384),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(384),
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
            lr=1e-3,
            layers=[''],  # train all layers
        )
    ],

    ckpt_manager=partial(CkptManagerPKL, saved_model=(
        {'model':'model', 'trainable':False},
    )),

    train=dict(
        train_epochs=100,
        workers=2,
        max_grad_norm=None,
        save_step=2000,

        loss=partial(MLCEImageLoss),

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
        wrapper=partial(SingleWrapper, model=CAFormerBackbone('caformer_m36', input_resolution=384))
    ),

    evaluator=partial(EvaluatorGroup, interval=500,
        evaluator_dict=dict(
            AP=CSIPmAPContainer(AveragePrecision(task="binary")),
        )
    ),

    data_train=dict(
        dataset1=partial(ImageLabelDataset, batch_size=16, loss_weight=1.0,
            source=dict(
                data_source1=ImageFolderClassSource(
                    img_root=r'/data/csip/train',
                    image_transforms=TRAIN_TRANSFORM,
                ),
            ),
            bucket=PosNegBucket(target_size=384, pos_rate=0.5),
        )
    ),

    data_eval=dict(
        dataset1=partial(ImageLabelDataset, batch_size=16, loss_weight=1.0,
            source=dict(
                data_source1=ImageFolderClassSource(
                    img_root=r'/data/csip/eval',
                    image_transforms=EVAL_TRANSFORM,
                ),
            ),
            bucket=PosNegBucket(target_size=384, pos_rate=0.5),
        )
    ),
)
