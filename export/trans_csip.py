# [RECONSTRUCTED] 原文件 /data/cxip/transfs/trans_csip.py (42行) 已丢失
# 把 rainbowneko 训练的 CSIP 权重导出为 HF 格式，目标代码为原版 export/csip/modeling_csip.py (从 dghstmp/AnimeCSIP-v2.1 找回)
# 用法(在 export/ 目录下): python trans_csip.py --ckpt ../exps/csip_dino_v2.1/ckpts/model-70000.safetensors --out csipv2.1_model
import argparse
import json
import os

import torch
from safetensors.torch import load_file

from csip.configuration_csip import CSIPConfig
from csip.modeling_csip import CSIPModel

ANIME_MEAN = (0.67713961, 0.618461, 0.61174436)
ANIME_STD = (0.31741801, 0.31876716, 0.31473021)

KEY_MAP = {'caformer.': 'backbone.', 'norm.': 'norm.', 'head.': 'head.', 'sim.': 'sim.'}  # 丢弃 eraser/tagger/ccip 等训练期模块

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('--out', type=str, default='csipv2.1_model')
args = parser.parse_args()

state = load_file(args.ckpt) if args.ckpt.endswith('.safetensors') else torch.load(args.ckpt, map_location='cpu')
state = state.get('base', state)
# rainbowneko 的 safetensors 把嵌套 dict 拍平成 'base:xxx'
state = {(k[len('base:'):] if k.startswith('base:') else k): v for k, v in state.items()}
state = {(k[len('model.'):] if k.startswith('model.') else k): v for k, v in state.items()}
state = {dst + k[len(src):]: v for k, v in state.items() for src, dst in KEY_MAP.items() if k.startswith(src)}

CSIPConfig.register_for_auto_class()
CSIPModel.register_for_auto_class('AutoModel')
model = CSIPModel(CSIPConfig())
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
