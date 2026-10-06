# [RECONSTRUCTED] 服务器上的 CCIP 导出脚本已丢失 (原文件名未知)，按 transfs/trans_csip.py 的同样方式重建
# 把 rainbowneko 训练的 CCIP 权重导出为 HF 格式，目标代码为原版 transfs/ccip/modeling_ccip.py (从 dghstmp/AnimeCCIP 找回)
# 用法(在 transfs/ 目录下): python trans_ccip.py --ckpt ../exps/ccip_dino_siglip/ckpts/model-70000.safetensors --out ccip_model
import argparse
import json
import os

import torch
from safetensors.torch import load_file

from ccip.configuration_ccip import CCIPConfig
from ccip.modeling_ccip import CCIPModel

ANIME_MEAN = (0.67713961, 0.618461, 0.61174436)
ANIME_STD = (0.31741801, 0.31876716, 0.31473021)

KEY_MAP = {'caformer.': 'backbone.', 'attnpool.': 'attnpool.', 'sim.': 'sim.'}  # 丢弃 训练期多余的 key

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('--out', type=str, default='ccip_model')
args = parser.parse_args()

state = load_file(args.ckpt) if args.ckpt.endswith('.safetensors') else torch.load(args.ckpt, map_location='cpu')
state = state.get('base', state)
state = {(k[len('model.'):] if k.startswith('model.') else k): v for k, v in state.items()}
state = {dst + k[len(src):]: v for k, v in state.items() for src, dst in KEY_MAP.items() if k.startswith(src)}

CCIPConfig.register_for_auto_class()
CCIPModel.register_for_auto_class('AutoModel')
model = CCIPModel(CCIPConfig())
missing, unexpected = model.load_state_dict(state, strict=False)
assert not missing and not unexpected, f'missing: {missing}\nunexpected: {unexpected}'
model.save_pretrained(args.out)

with open(os.path.join(args.out, 'preprocessor_config.json'), 'w') as f:  # 与 HF 上原版文件内容一致
    json.dump({
        'do_center_crop': False,
        'do_normalize': True,
        'do_resize': True,
        'feature_extractor_type': 'CLIPFeatureExtractor',
        'image_mean': list(ANIME_MEAN),
        'image_std': list(ANIME_STD),
        'resample': 2,  # bilinear, 与训练时 T.Resize((384, 384)) 一致
        'size': {'width': 384, 'height': 384},
    }, f, indent=2)
