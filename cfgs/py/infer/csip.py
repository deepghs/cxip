import torch
import torchvision
import torchvision.transforms as T
from torch import nn

from rainbowneko.data import UnLabelSource, HandlerChain, LoadImageHandler, ImageHandler, BaseDataset, BaseBucket
from rainbowneko.infer import HandlerAction, DataLoaderAction
from rainbowneko.infer.workflow import (Actions, BuildModelAction, PrepareAction, FeedAction, ForwardAction,
                                        LambdaAction, VisClassAction, LoadModelAction)
from rainbowneko.models.wrapper import SingleWrapper
from rainbowneko.utils import KeyMapper
from rainbowneko.parser import neko_cfg
from rainbowneko.ckpt_manager import auto_ckpt_loader
from model import CAFormerBackbone, CAFormerStyleBackbone
from data import UniformPatch

ANIME_MEAN = (0.67713961, 0.618461,   0.61174436)
ANIME_STD = (0.31741801, 0.31876716, 0.31473021)

def show_res(output, id, **kwargs):
    print(id)
    print(output)

@neko_cfg
def infer_all(path):
    return DataLoaderAction(
        dataset=BaseDataset(_partial_=True, batch_size=64, loss_weight=1.0,
            source=dict(
                data_source1=UnLabelSource(
                    img_root=path
                ),
            ),
            handler=HandlerChain(
                load=LoadImageHandler(),
                image=ImageHandler(transform=T.Compose([
                        # T.Resize((384, 384)),
                        UniformPatch(target_size=(384, 384), patch_size=(32, 32)),
                        T.ToTensor(),
                        T.Normalize(mean=ANIME_MEAN, std=ANIME_STD),
                    ]),
                )
            ),
            bucket=BaseBucket(),
        ),
        actions=Actions([
            ForwardAction(key_map_in=('image -> input.image', 'model -> model', 'device -> device', 'dtype -> dtype')),
            LambdaAction(show_res)
        ])
    )

@neko_cfg
def make_cfg():
    return dict(
        img_path=r"imgs/",
        workflow=Actions(actions=[
            PrepareAction(device='cuda', dtype=torch.float16),
            BuildModelAction(SingleWrapper(_partial_=True, model=CAFormerBackbone('caformer_b36', input_resolution=384))),
            LoadModelAction(dict(
                model=auto_ckpt_loader(
                    target_module='model',
                    # path='exps/csip_dino_style/ckpts/model-5000.safetensors'
                    path = 'exps/csip_dino/ckpts/model-12000.safetensors'
                ),
            )),
            # infer_one(path="${img_path}")
            infer_all(path="${img_path}")
        ])
    )