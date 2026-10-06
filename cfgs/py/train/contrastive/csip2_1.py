# ======================================================================================
# [PARTIALLY RECOVERED] /data/cxip/cfgs/py/train/contrastive/csip2_1.py (原文件254行)
# - 第 "VERBATIM" 标记之后的内容: 从 Cursor 会话 35c897f4 的 read_file(offset=180) 结果中原样恢复
# - 之前的内容: 依据 csip.py、Lark 文档白板、恢复出的 import 行(第24行)重建
#   第24行原文: from data import StyleImageLabelSource, UniformPatch, CharBucket, StyleCharImageLabelSource
# 需人工确认的项 (标 TODO): loss 权重、GRL alpha、dataset1 batch size
# 冻结 CCIP 直接用 HF dghstmp/AnimeCCIP (2025-10-08，新版 anime_dino CCIP)
# 流程: ccip_save.py 提取训练集 CCIP 特征 -> tools/ccip_cls.py 聚类 -> tools/clusters_to_char_labels.py
#       -> train_char_labels.json -> 本配置训练 -> transfs/trans_csip.py 导出 -> csip_save.py + eval_csip_zero.py / evaluate/*_bias.py
# ======================================================================================
from functools import partial
from typing import Tuple, Dict

import torch
import torchvision.transforms as T
from rainbowneko.ckpt_manager import ckpt_saver, NekoResumer, NekoModelLoader
from rainbowneko.data import BaseDataset
from rainbowneko.data import PosNegBucket, CategoryBucket
from rainbowneko.data.handler import HandlerChain, ImageHandler, LoadImageHandler
from rainbowneko.evaluate import MetricGroup, Evaluator
from rainbowneko.models.wrapper import SingleWrapper
from rainbowneko.parser import CfgWDModelParser, neko_cfg
from rainbowneko.train.loss import LossContainer, LossGroup
from rainbowneko.utils import CosineLR
from torch import nn
from torchmetrics.classification import AveragePrecision, AUROC

from cfgs.py.train import train_base, tuning_base
from evaluate import CXIPMetricContainer, cohens_d
from model import CAFormerStyleDebiasBackbone, FrozenTagger, FrozenCCIP, load_preprocessor
from data import StyleImageLabelSource, UniformPatch, CharBucket, StyleCharImageLabelSource
from loss import SiglipLoss, LogitsKLLoss

class WeakRandAugment2(T.RandAugment):
    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[torch.Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity":(torch.tensor(0.0), False),
            "TranslateX":(torch.linspace(0.0, 0.08*image_size[1], num_bins), True),
            "TranslateY":(torch.linspace(0.0, 0.08*image_size[0], num_bins), True),
        }

ANIME_MEAN = (0.67713961, 0.618461,   0.61174436)
ANIME_STD = (0.31741801, 0.31876716, 0.31473021)

TAGGER = 'animetimm/caformer_b36.dbv4-full'
# content eraser 擦除内容标签 + 文字/元数据标签 (签名、水印、语种、年份等)，不擦画风/构图/光效/背景等；由 tools/content_tags.py 生成
CONTENT_TAGS = 'data/content_tags/caformer_b36.dbv4-full.csv'
CCIP = 'dghstmp/AnimeCCIP'


@neko_cfg
def loss_csip(w_content=1.0, w_char=1.0):  # TODO: 原始权重未知
    return LossGroup(
        siglip=LossContainer(SiglipLoss(), key_map=('pred.pred -> pred', 'inputs.label -> label')),
        content=LossContainer(nn.BCEWithLogitsLoss(), weight=w_content,
                              key_map=('pred.content_logits -> 0', 'pred.content_probs -> 1')),
        char=LossContainer(LogitsKLLoss(), weight=w_char,
                           key_map=('pred.char_logits -> 0', 'pred.char_teacher_logits -> 1')),
    )


@neko_cfg
def make_cfg():
    return dict(
        _base_=[train_base, tuning_base],
        exp_dir=f'exps/csip_dino_v2.1',

        model_part=CfgWDModelParser([
            dict(
                lr=1e-5,
                # 不能用 '' (会把冻结的 tagger/ccip 也设为可训练)
                layers=['model.caformer', 'model.norm', 'model.head', 'model.sim',
                        'model.content_eraser', 'model.char_eraser', 'model.char_teacher_head'],
            )
        ], weight_decay=1e-2),

        ckpt_saver=dict(
            # 只存可训练参数，不存冻结的 tagger / ccip (各 ~400MB)
            model=ckpt_saver(target_module='model', layers='trainable'),
        ),

        train=dict(
            train_epochs=2,
            workers=2,
            max_grad_norm=None,
            save_step=1000,

            loss=dict(_replace_=True,
                dataset1=loss_csip(),
                dataset_dechar=loss_csip(),
            ),

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
            name='csip-dino-v2.1',
            wrapper=partial(SingleWrapper,
                model=CAFormerStyleDebiasBackbone('caformer_b36', input_resolution=384,
                    tagger=FrozenTagger(TAGGER, tag_file=CONTENT_TAGS),
                    ccip=FrozenCCIP(CCIP),
                    num_char_cls=100,
                    grl_alpha=1.0,  # TODO
                ),
                key_map_in=('image -> image', 'content_image -> content_image'),
                key_map_out=('pred -> pred', 'feat -> feat', 'content_logits -> content_logits', 'content_probs -> content_probs',
                             'char_logits -> char_logits', 'char_teacher_logits -> char_teacher_logits'),
            )
        ),

        data_train=cfg_data(), # config can be split into another function with @neko_cfg

        evaluator=cfg_evaluator(),
    )

@neko_cfg
def cfg_data():
    return dict(
        dataset1=partial(BaseDataset, batch_size=128, loss_weight=1.0,  # TODO: batch size
            source=dict(
                data_source1=StyleImageLabelSource(
                    img_root="/nfs/danbooru-w640-ws-full-artist/train",
                    label_file="/nfs/danbooru-w640-ws-full-artist/train_split.json",
                ),
            ),
            handler=HandlerChain(
                load=LoadImageHandler(),
                tagger=ImageHandler(
                    transform=load_preprocessor(TAGGER),
# ----------------------------------- VERBATIM (原文件第180行起) -----------------------------------
                    key_map_out=('image -> content_image',),
                ),
                image=ImageHandler(transform=T.Compose([
                        T.Resize((384, 384)),
                        T.RandomHorizontalFlip(),
                        WeakRandAugment2(),
                        T.ToTensor(),
                        T.Normalize(mean=ANIME_MEAN, std=ANIME_STD),
                    ]),
                ),
                
            ),
            bucket=CategoryBucket(target_size=384, min_cat_size=4),
        ),
        dataset_dechar=partial(BaseDataset, batch_size=128, loss_weight=1.0,
            source=dict(
                data_source1=StyleCharImageLabelSource(
                    img_root="/nfs/danbooru-w640-ws-full-artist/train",
                    char_label_file="/nfs/danbooru-w640-ws-full-artist/train_char_labels.json",
                    style_label_file="/nfs/danbooru-w640-ws-full-artist/train_split.json",
                ),
            ),
            handler=HandlerChain(
                load=LoadImageHandler(),
                tagger=ImageHandler(
                    transform=load_preprocessor('animetimm/caformer_b36.dbv4-full'),
                    key_map_out=('image -> content_image',),
                ),
                image=ImageHandler(transform=T.Compose([
                        T.Resize((384, 384)),
                        T.RandomHorizontalFlip(),
                        WeakRandAugment2(),
                        T.ToTensor(),
                        T.Normalize(mean=ANIME_MEAN, std=ANIME_STD),
                    ]),
                ),
                key_map_out=('content_image -> content_image', 'image -> image', 'style -> label'),
            ),
            bucket=CharBucket(target_size=384, min_cat_size=8),
        )
    )

@neko_cfg
def cfg_evaluator():
    return partial(Evaluator,
        interval=2000,
        data_workers=8,
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
                tagger=ImageHandler(
                    transform=load_preprocessor('animetimm/caformer_b36.dbv4-full'),
                    key_map_out=('image -> content_image',),
                ),
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