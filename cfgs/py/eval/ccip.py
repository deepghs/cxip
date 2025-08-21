import time

import torchvision
from torch import nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from torchvision import transforms as T
from functools import partial
import webdataset as wds

from rainbowneko.ckpt_manager import NekoResumer, auto_ckpt_loader, NekoModelLoader, PKLFormat
from rainbowneko.data import BaseDataset, ImageHandler, BaseBucket, HandlerChain, LoadImageHandler, ImageFolderClassSource, WebDataset, PosNegBucket
from rainbowneko.evaluate import Evaluator, MetricGroup, MetricContainer
from rainbowneko.models.wrapper import FeatWrapper
from rainbowneko.parser import neko_cfg
from rainbowneko.loggers import CLILogger

from torchmetrics.classification import AveragePrecision, AUROC

from evaluate import CXIPMetricContainer, cohens_d
from model import CAFormerBackbone
# from model.old.model import CCIP
from data.source import ZerochanWebDatasetSource

time_format="%Y-%m-%d-%H-%M-%S"
num_classes = 10

ANIME_MEAN = (0.67713961, 0.618461,   0.61174436)
ANIME_STD = (0.31741801, 0.31876716, 0.31473021)

def image_pipeline(url, buffer_size=300):
    return wds.DataPipeline(
        wds.SimpleShardList(url),
        wds.split_by_node,
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.shuffle(buffer_size),
    )

@neko_cfg
def make_cfg():
    return Evaluator(
        _partial_=True,
        exp_dir=f'exps_eval/{time.strftime(time_format)}',
        mixed_precision='fp16',
        seed=42,

        interval=50,
        metric=MetricGroup(
            AP=CXIPMetricContainer(AveragePrecision(task="binary"), key_map=('pred.pred -> pred', 'inputs.label -> label')),
            roc=CXIPMetricContainer(AUROC(task="binary"), key_map=('pred.pred -> pred', 'inputs.label -> label')),
            cohens=CXIPMetricContainer(cohens_d, key_map=('pred.pred -> pred', 'inputs.label -> label')),
        ),
        model=dict(
            name='ccip',
            enable_xformers=True,
            gradient_checkpointing=False,
            force_cast_precision=False,

            wrapper=partial(FeatWrapper, model=CAFormerBackbone('caformer_b36', input_resolution=384))
            # wrapper=partial(FeatWrapper, model=CCIP.from_ckpt('ckpt/ccip-caformer_b36-24-neko.ckpt'))
        ),
        # resume=None,
        resume=NekoResumer(
            start_step=0,
            loader=dict(
                model=NekoModelLoader(
                    # format=PKLFormat(),
                    # target_module='model.caformer',
                    # state_prefix='ema_teacher.backbone.',
                    target_module='model',
                    path='/data/cxip/exps/2025-08-09-00-28-22/ckpts/model-4000.safetensors'
                ),
            )
        ),
        logger=[
            CLILogger(_partial_=True, out_path='eval.log', log_step=20),
        ],
        dataset=partial(WebDataset, batch_size=128, loss_weight=1.0, workers=1,
            source=dict(
                data_source1=ZerochanWebDatasetSource(
                    pipeline=image_pipeline("/nfs/zerochan-character-w640-ws-m50-full/test/{00001..00006}.tar")
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
            bucket=PosNegBucket(target_size=384, pos_rate=0.5, num_bucket=5031),
        )
        # dataset=partial(BaseDataset, batch_size=128, loss_weight=1.0,
        #     source=dict(
        #         data_source1=ImageFolderClassSource(
        #             img_root="/nfs/ccip/ccip_eval/train"
        #         ),
        #     ),
        #     handler=HandlerChain(
        #         load=LoadImageHandler(),
        #         image=ImageHandler(transform=T.Compose([
        #                 T.Resize((384, 384)),
        #                 T.ToTensor(),
        #                 T.Normalize(mean=ANIME_MEAN, std=ANIME_STD),
        #             ]),
        #         )
        #     ),
        #     bucket=BaseBucket(),
        # )
    )