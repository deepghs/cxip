# [RECONSTRUCTED] 服务器上的 cfgs/py/infer/csip_save.py 已丢失，按 ccip_save.py 的同样方式重建
# 提取 CSIP v2/v2.1 风格特征，每张图存为 {out_dir_name}/{id}.pt (id 即 label_file 的 key，如 12345.webp -> 12345.webp.pt)
#   neko_run --cfg cfgs/py/infer/csip_save.py ckpt_path="exps/csip_dino_v2.1/ckpts/model-xxx.safetensors" out_dir_name="feats/feat_csip_v2.1"
# 之后: eval_csip_zero.py (emb_dir 写死为 feat/，需改成 out_dir_name)
# 平铺的图片文件夹 (无 label_file) 把 StyleImageLabelSource 换成 UnLabelSource(img_root=path) 即可，见 ccip_save.py
# v1 (无 LayerNorm) 换成 FeatWrapper + CAFormerStyleBackbone
import torch
import torchvision.transforms as T

from rainbowneko.data import HandlerChain, LoadImageHandler, ImageHandler, BaseDataset, BaseBucket
from rainbowneko.infer import DataLoaderAction
from rainbowneko.infer.workflow import Actions, BuildModelAction, PrepareAction, ForwardAction, LoadModelAction
from rainbowneko.models.wrapper import SingleWrapper
from rainbowneko.parser import neko_cfg
from rainbowneko.ckpt_manager import auto_ckpt_loader
from model import CAFormerStyleDebiasBackbone
from data import StyleImageLabelSource
from actions import SaveFeatAction

ANIME_MEAN = (0.67713961, 0.618461,   0.61174436)
ANIME_STD = (0.31741801, 0.31876716, 0.31473021)

@neko_cfg
def infer_all(path, label_file, save_dir):
    return DataLoaderAction(
        dataset=BaseDataset(_partial_=True, batch_size=64, loss_weight=1.0,
            source=dict(
                data_source1=StyleImageLabelSource(
                    img_root=path,
                    label_file=label_file,
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
        test_img_root=r"/nfs/danbooru-w640-ws-full-artist/train",
        label_file=r"/nfs/danbooru-w640-ws-full-artist/test_split.json",
        out_dir_name="feat",
        ckpt_path='exps/csip_dino_v2.1/ckpts/model-70000.safetensors',  # TODO: 换成实际的 ckpt
        workflow=Actions(actions=[
            PrepareAction(device='cuda', dtype=torch.float16),
            # 不传 tagger/ccip: 推理只需要 caformer + norm + head + sim
            BuildModelAction(SingleWrapper(_partial_=True, model=CAFormerStyleDebiasBackbone('caformer_b36', input_resolution=384),
                                           key_map_out=('pred -> pred', 'feat -> feat'))),
            LoadModelAction(dict(
                model=auto_ckpt_loader(
                    target_module='model',
                    # 必须指定 layers: NekoModelLoader 对 ckpt 中每个 key 取 states[k]，模型里没有的 eraser 等 key 会 KeyError
                    layers=['caformer', 'norm', 'head', 'sim'],
                    path="${ckpt_path}",
                ),
            )),
            infer_all(path="${test_img_root}", label_file="${label_file}", save_dir="${out_dir_name}")
        ])
    )
