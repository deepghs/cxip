# [RECONSTRUCTED] 服务器上的 cfgs/py/eval/csip.py 已丢失 (scripts/eval/eval.sh 中有引用)，按 eval/ccip.py 和 csip2_1.py 的 evaluator 重建
#   CUDA_VISIBLE_DEVICES=1 neko_eval_1gpu --cfg cfgs/py/eval/csip.py
import time
from functools import partial

from torchvision import transforms as T

from rainbowneko.ckpt_manager import NekoResumer, NekoModelLoader
from rainbowneko.data import BaseDataset, ImageHandler, HandlerChain, LoadImageHandler, PosNegBucket
from rainbowneko.evaluate import Evaluator, MetricGroup
from rainbowneko.models.wrapper import SingleWrapper
from rainbowneko.parser import neko_cfg
from rainbowneko.loggers import CLILogger

from torchmetrics.classification import AveragePrecision, AUROC

from evaluate import CXIPMetricContainer, cohens_d
from model import CAFormerStyleDebiasBackbone
from data import StyleImageLabelSource

time_format="%Y-%m-%d-%H-%M-%S"

ANIME_MEAN = (0.67713961, 0.618461,   0.61174436)
ANIME_STD = (0.31741801, 0.31876716, 0.31473021)

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
            name='csip',
            enable_xformers=True,
            gradient_checkpointing=False,
            force_cast_precision=False,

            # 不传 tagger/ccip: 评估只用 caformer + norm + head + sim
            wrapper=partial(SingleWrapper, model=CAFormerStyleDebiasBackbone('caformer_b36', input_resolution=384),
                            key_map_out=('pred -> pred', 'feat -> feat')),
            # v1: wrapper=partial(FeatWrapper, model=CAFormerStyleBackbone('caformer_b36', input_resolution=384))
        ),
        resume=NekoResumer(
            start_step=0,
            loader=dict(
                model=NekoModelLoader(
                    target_module='model',
                    layers=['caformer', 'norm', 'head', 'sim'],  # 跳过 ckpt 中的 eraser，否则 KeyError
                    path='exps/csip_dino_v2.1/ckpts/model-70000.safetensors'  # TODO: 换成实际的 ckpt
                ),
            )
        ),
        logger=[
            CLILogger(_partial_=True, out_path='eval.log', log_step=20),
        ],
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
