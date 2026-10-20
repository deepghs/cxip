# [REPRO] 由 cfgs/py/train/contrastive/ccip.py 生成, 只改了数据/权重路径和多卡设置 (见 experiments/README.md)
# 4 卡 x bs128 x grad_accum 2 = 原 launcher (cfgs/launcher/multi.yaml, 8 卡 x bs128) 的全局 batch 1024; SigLIP loss 不跨卡 gather, 每卡 128 不变
# 新版 CCIP (HF: dghstmp/AnimeCCIP, 2025-10-08) 的训练配置 [RECONSTRUCTED]
# 相对上一次提交的改动:
#   - loss: MLCEImageLoss -> SiglipLoss。依据: HF 权重中 sim.logit_bias = -13.948 (初始 -10)。
#     MLCE 按行 softmax，logit_bias 的梯度恒为 0，bias 不会变，所以新版 CCIP 只能是 sigmoid/SigLIP 训的。
#   - BatchCosineSimilarity 初始化改为 log(10)/-10 (见 model/caformer.py)
#   - 从 anime_dino 的 ema_teacher 初始化 backbone (与上一次提交相同)
# 导出 HF 格式: cd export && python trans_ccip.py --ckpt ../exps/ccip_dino/ckpts/model-xxx.safetensors --out ccip_model
from functools import partial
from typing import Tuple, Dict

import torch
import torchvision.transforms as T
from rainbowneko.ckpt_manager import ckpt_saver, NekoResumer, NekoModelLoader
from rainbowneko.data import BaseDataset
from rainbowneko.data import PosNegBucket, BaseBucket
from rainbowneko.data.handler import HandlerChain, ImageHandler, LoadImageHandler
from rainbowneko.data.source import ImageFolderClassSource
from rainbowneko.evaluate import MetricGroup, Evaluator
from rainbowneko.models.wrapper import FeatWrapper
from rainbowneko.parser import CfgWDModelParser, neko_cfg
from rainbowneko.train.loss import LossContainer, LossGroup
from rainbowneko.utils import CosineLR
from torchmetrics.classification import AveragePrecision, AUROC

from cfgs.py.train import train_base, tuning_base
from evaluate import CXIPMetricContainer, cohens_d
from loss import SiglipLoss
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
        exp_dir=f"/data/neko/cxip_repro/exps/ccip_dino",

        model_part=CfgWDModelParser([
            dict(
                lr=1e-4,
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
            gradient_accumulation_steps=2,
            workers=2,
            max_grad_norm=None,
            save_step=2000,

            # 当前 RainbowNekoEngine (>=2025-12-08, 7908b54) 要求 criterion 返回 dict，需包一层 LossGroup;
            # 旧版引擎直接写 loss=LossContainer(SiglipLoss())
            loss=LossGroup(siglip=LossContainer(SiglipLoss())),

            resume=NekoResumer(
                start_step=0,
                loader=dict(
                    model=NekoModelLoader(
                        target_module='model.caformer',
                        state_prefix='ema_teacher.backbone.',
                        path='/data/.hf/hub/models--dghstmp--anime_dino/snapshots/579ac91b043124c061bf9cf1108b50a92e6ce083/model-191000.safetensors'
                    ),
                )
            ),

            optimizer=partial(torch.optim.AdamW, weight_decay=1e-2),

            scale_lr=False,
            lr_scheduler=CosineLR(
                _partial_=True,
                warmup_steps=500,
            ),
            metrics=MetricGroup(
                AP=CXIPMetricContainer(AveragePrecision(task="binary"), key_map=('pred.pred -> pred', 'inputs.label -> label')),
                roc=CXIPMetricContainer(AUROC(task="binary"), key_map=('pred.pred -> pred', 'inputs.label -> label')),
                cohens=CXIPMetricContainer(cohens_d, key_map=('pred.pred -> pred', 'inputs.label -> label')),
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
                    img_root="/data/neko/cxip_repro/data/ccip_v1_flat"
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
        interval=200,
        metric=MetricGroup(
            AP=CXIPMetricContainer(AveragePrecision(task="binary"), key_map=('pred.pred -> pred', 'inputs.label -> label')),
            roc=CXIPMetricContainer(AUROC(task="binary"), key_map=('pred.pred -> pred', 'inputs.label -> label')),
            cohens=CXIPMetricContainer(cohens_d, key_map=('pred.pred -> pred', 'inputs.label -> label')),
        ),
        dataset=partial(BaseDataset, batch_size=128, loss_weight=1.0,
            source=dict(
                data_source1=ImageFolderClassSource(
                    img_root="/data/neko/cxip_repro/data/ccip/ccip_eval/eval"
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
            bucket=BaseBucket(),
        )
    )