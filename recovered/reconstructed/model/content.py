# [RECONSTRUCTED] 原文件 /data/cxip/model/content.py (35行) 已丢失
# csip2_1.py 中可见: load_preprocessor('animetimm/caformer_b36.dbv4-full') 作为 tagger 的 ImageHandler transform
import json

import timm
import torch
from huggingface_hub import hf_hub_download
from timm.data import create_transform, resolve_data_config
from torch import nn


def load_preprocessor(repo_id='animetimm/caformer_b36.dbv4-full'):
    try:
        from imgutils.preprocess import create_torchvision_transforms
        with open(hf_hub_download(repo_id=repo_id, repo_type='model', filename='preprocess.json'), 'r') as f:
            return create_torchvision_transforms(json.load(f)['test'])
    except Exception:
        model = timm.create_model(f'hf-hub:{repo_id}', pretrained=False)
        return create_transform(**resolve_data_config(model.pretrained_cfg, model=model))


class FrozenModule(nn.Module):
    '''冻结的教师模型：始终 eval，不参与反传'''

    def train(self, mode: bool = True):
        return super().train(False)


class FrozenTagger(FrozenModule):
    def __init__(self, repo_id='animetimm/caformer_b36.dbv4-full', tag_idxs=None):
        super().__init__()
        self.model = timm.create_model(f'hf-hub:{repo_id}', pretrained=True)
        self.model.requires_grad_(False)
        self.tag_idxs = None if tag_idxs is None else list(tag_idxs)
        self.num_tags = self.model.num_classes if tag_idxs is None else len(self.tag_idxs)

    @torch.no_grad()
    def forward(self, x):
        probs = self.model(x).float().sigmoid()
        if self.tag_idxs is not None:
            probs = probs[:, self.tag_idxs]
        return probs


class FrozenCCIP(FrozenModule):
    '''
    ccip 可以是 HF repo id (如 'dghstmp/AnimeCCIP'，预处理与 CSIP 相同: 384x384 + ANIME_MEAN/STD，可直接喂 image)，
    也可以是本项目的 CAFormerBackbone 实例 (需另外加载权重)
    '''

    def __init__(self, ccip='dghstmp/AnimeCCIP'):
        super().__init__()
        if isinstance(ccip, str):
            from transformers import AutoModel
            ccip = AutoModel.from_pretrained(ccip, trust_remote_code=True)
        self.ccip = ccip
        self.ccip.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):
        out = self.ccip(x)
        feat = out['features'] if isinstance(out, dict) else out[1]
        return feat.float()
