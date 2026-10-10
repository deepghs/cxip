# [FIX] 修正 debias 训练 (见 repro/README.md "debias 实现对比"): 原重建配置中 eraser 与 backbone 共用 lr 1e-5,
#   2043 步后 eraser 仍停在随机初始化 (content BCE ≈ ln2), GRL 只是在推 backbone 去骗一个随机 MLP, 不去偏还毁特征。
#   - eraser / teacher head 单独参数组 lr 1e-3, 无 wd; 用 diag_eraser_0918 (冻结 backbone 预训 300 步) 的 eraser 权重初始化
#   - detach_teacher + teacher_scale=3: 防止 teacher MLP 与 char eraser 一起塌成均匀分布
#   - 提速: bf16 + 按 stage torch.compile + 每个数据集 12 个 worker (1 卡 fp32/2 worker 时被访存型 kernel 和数据加载卡住)
# 本组: grl_alpha=0.03, loss 权重 content/char=1.0
# [REPRO] 由 cfgs/py/train/contrastive/csip2_1.py 生成, 只改了数据/权重路径和多卡设置 (见 repro/README.md)
# 4 卡 x bs128 x grad_accum 2 = 原 launcher (cfgs/launcher/multi.yaml, 8 卡 x bs128) 的全局 batch 1024; SigLIP loss 不跨卡 gather, 每卡 128 不变
# [FROM V2] 以 HF dghstmp/AnimeCSIP-v2 初始化 (backbone/norm/head/sim), 只做 v2.1 的 debias 微调 1 epoch。
#   依据: HF v2.1 sim.logit_bias=-23.8 比 v2 (-18.4) 更远离初始 -10, lr 1e-5 下从 anime_dino 起训不可能到达;
#   原 csip2_1.py 的 lr/epoch/resume 部分本来就是重建的
# 切分: data/danbooru-artist-split-0918 (repro/data/match_split_0918.py, 按 2025-09-18 对话记录的指纹精确恢复的原始划分, 只保留服务器上有图的样本)
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
        mixed_precision='bf16',
        allow_tf32=True,
        exp_dir=f"/data/neko/cxip_repro/exps/fix_csip_0918_g003",

        model_part=CfgWDModelParser([
            dict(
                lr=1e-5,
                # 不能用 '' (会把冻结的 tagger/ccip 也设为可训练)
                layers=['model.caformer', 'model.norm', 'model.head', 'model.sim'],
            ),
            dict(
                lr=1e-3, weight_decay=0.,
                # teacher head 在 detach_teacher=True 时不接收梯度, 列出来只是为了存进 ckpt
                layers=['model.content_eraser', 'model.char_eraser', 'model.char_teacher_head'],
            ),
        ], weight_decay=1e-2),

        ckpt_saver=dict(
            # 只存可训练参数，不存冻结的 tagger / ccip (各 ~400MB)
            model=ckpt_saver(target_module='model', layers='trainable'),
        ),

        train=dict(
            train_epochs=1,
            gradient_accumulation_steps=8,  # 1 卡 x bs128 x 8 = 全局 1024, 与 4 卡版等价 (real_step 按优化器步计)
            workers=12,
            max_grad_norm=None,
            save_step=300,

            loss=dict(_replace_=True,
                dataset1=loss_csip(1.0, 1.0),
                dataset_dechar=loss_csip(1.0, 1.0),
            ),

            resume=NekoResumer(
                start_step=0,
                loader=dict(
                    model=NekoModelLoader(
                        target_module='model',
                        path='/data/neko/cxip_repro/exps/init/csip_v2_eraser_init.safetensors'
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
                    grl_alpha=0.03, eraser_pre_norm=False,
                    detach_teacher=True, teacher_scale=3.0, compile_model=True,
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
                    img_root="/data/neko/cxip_repro/data/danbooru-w640-ws-full-artist/train",
                    label_file="/data/neko/cxip_repro/data/danbooru-artist-split-0918/train_split_avail.json",
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
                    img_root="/data/neko/cxip_repro/data/danbooru-w640-ws-full-artist/train",
                    char_label_file="/data/neko/cxip_repro/data/danbooru-artist-split-0918/train_char_labels.json",
                    style_label_file="/data/neko/cxip_repro/data/danbooru-artist-split-0918/train_split_avail.json",
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
        interval=100000,
        data_workers=8,
        metric=MetricGroup(
            AP=CXIPMetricContainer(AveragePrecision(task="binary"), key_map=('pred.pred -> pred', 'inputs.label -> label')),
            roc=CXIPMetricContainer(AUROC(task="binary"), key_map=('pred.pred -> pred', 'inputs.label -> label')),
            cohens=CXIPMetricContainer(cohens_d, key_map=('pred.pred -> pred', 'inputs.label -> label')),
        ),
        dataset=partial(BaseDataset, batch_size=64, loss_weight=1.0, 
            source=dict(
                data_source1=StyleImageLabelSource(
                    img_root="/data/neko/cxip_repro/data/danbooru-w640-ws-full-artist/train",
                    label_file="/data/neko/cxip_repro/data/danbooru-artist-split-0918/test_split_avail.json",
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