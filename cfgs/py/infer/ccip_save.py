# [RECONSTRUCTED] 服务器上的 cfgs/py/infer/ccip_save.py 已丢失，按 eval.sh 中的用法重建:
#   neko_run --cfg cfgs/py/infer/ccip_save.py test_img_root="/nfs/ccip/test_ccip/" out_dir_name="feat_ccip"
#   neko_run --cfg cfgs/py/infer/ccip_save.py test_img_root="/nfs/zerochan-character-w640-ws-m50-full/small_val_folder" out_dir_name="feat_ccip_zero"
# test_img_root 下是平铺的图片 (文件名 {角色}_xxx.png)，每张图的特征存为 {out_dir_name}/{文件名}.pt
# 之后: python eval_ccip_zero.py --emb_dir feat_ccip --N 40 ; python tools/ccip_cls.py --feat-dir ...
import torch
import torchvision.transforms as T

from rainbowneko.data import UnLabelSource, HandlerChain, LoadImageHandler, ImageHandler, BaseDataset, BaseBucket
from rainbowneko.infer import DataLoaderAction
from rainbowneko.infer.workflow import Actions, BuildModelAction, PrepareAction, ForwardAction, LoadModelAction
from rainbowneko.models.wrapper import FeatWrapper
from rainbowneko.parser import neko_cfg
from rainbowneko.ckpt_manager import auto_ckpt_loader
from model import CAFormerBackbone
from actions import SaveFeatAction

ANIME_MEAN = (0.67713961, 0.618461,   0.61174436)
ANIME_STD = (0.31741801, 0.31876716, 0.31473021)

@neko_cfg
def infer_all(path, save_dir):
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
                        T.Resize((384, 384)),
                        T.ToTensor(),
                        T.Normalize(mean=ANIME_MEAN, std=ANIME_STD),
                    ]),
                )
            ),
            bucket=BaseBucket(),
        ),
        actions=Actions([
            ForwardAction(key_map_in=('image -> input.image', 'model -> model', 'device -> device', 'dtype -> dtype')),
            SaveFeatAction(save_dir=save_dir),
        ])
    )

@neko_cfg
def make_cfg():
    return dict(
        test_img_root=r"/nfs/ccip/test_ccip/",
        out_dir_name="feat_ccip",
        ckpt_path='exps/ccip_dino/ckpts/model-48000.safetensors',  # TODO: 换成实际的 ckpt
        workflow=Actions(actions=[
            PrepareAction(device='cuda', dtype=torch.float16),
            BuildModelAction(FeatWrapper(_partial_=True, model=CAFormerBackbone('caformer_b36', input_resolution=384))),
            LoadModelAction(dict(
                model=auto_ckpt_loader(
                    target_module='model',
                    path="${ckpt_path}",
                ),
            )),
            infer_all(path="${test_img_root}", save_dir="${out_dir_name}")
        ])
    )
