# [RECONSTRUCTED] 原文件 /data/cxip/model/content.py (35行) 已丢失
# csip2_1.py 中可见: load_preprocessor('animetimm/caformer_b36.dbv4-full') 作为 tagger 的 ImageHandler transform
import json

import pandas as pd
import timm
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from torch import nn
from torchvision import transforms as T


class PadToSize:
    '''同 imgutils pad_to_size: 保持长宽比缩放到能放进 size，再居中贴到纯色画布上'''

    def __init__(self, size, background_color='white', interpolation=Image.BILINEAR):
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        self.background_color = background_color
        self.interpolation = interpolation

    def __call__(self, img):
        tw, th = self.size
        w, h = img.size
        ratio = min(tw / w, th / h)
        nw, nh = round(w * ratio), round(h * ratio)
        canvas = Image.new(img.mode, (tw, th), self.background_color)
        canvas.paste(img.resize((nw, nh), self.interpolation), ((tw - nw) // 2, (th - nh) // 2))
        return canvas


_INTERP = {'nearest': T.InterpolationMode.NEAREST, 'bilinear': T.InterpolationMode.BILINEAR,
           'bicubic': T.InterpolationMode.BICUBIC}


def _build_transforms(cfgs):
    '''不装 imgutils 时按 preprocess.json 搭 torchvision 的 transform (只支持 animetimm tagger 用到的几种)'''
    trans = []
    for c in cfgs:
        if c['type'] == 'pad_to_size':
            trans.append(PadToSize(c['size'], c.get('background_color', 'white'),
                                   getattr(Image, c.get('interpolation', 'bilinear').upper())))
        elif c['type'] == 'resize':
            trans.append(T.Resize(c['size'], interpolation=_INTERP[c.get('interpolation', 'bilinear')],
                                  max_size=c.get('max_size'), antialias=c.get('antialias', True)))
        elif c['type'] == 'center_crop':
            trans.append(T.CenterCrop(c['size']))
        elif c['type'] in ('to_tensor', 'maybe_to_tensor'):
            trans.append(T.ToTensor())
        elif c['type'] == 'normalize':
            trans.append(T.Normalize(c['mean'], c['std']))
        else:
            raise ValueError(f'unsupported preprocess: {c}')
    return T.Compose(trans)


def load_preprocessor(repo_id='animetimm/caformer_b36.dbv4-full'):
    with open(hf_hub_download(repo_id=repo_id, repo_type='model', filename='preprocess.json'), 'r') as f:
        cfgs = json.load(f)['test']
    try:
        from imgutils.preprocess import create_torchvision_transforms
        return create_torchvision_transforms(cfgs)
    except ImportError:
        return _build_transforms(cfgs)


def load_tag_idxs(tag_file, num_tags=None):
    '''读取 tools/content_tags.py 生成的 csv，返回 content==1 的标签下标'''
    df = pd.read_csv(tag_file, keep_default_na=False)
    if num_tags is not None and len(df) != num_tags:
        raise ValueError(f'{tag_file} has {len(df)} tags, but tagger outputs {num_tags}')
    return df.loc[df['content'] == 1, 'index'].tolist()


class FrozenModule(nn.Module):
    '''冻结的教师模型：始终 eval，不参与反传'''

    def train(self, mode: bool = True):
        return super().train(False)


class FrozenTagger(FrozenModule):
    '''
    tag_file: tools/content_tags.py 生成的 csv，只用其中 content==1 的 (描述画面内容的) 标签;
    tag_idxs: 直接指定标签下标; 都不传则用全部输出
    '''

    def __init__(self, repo_id='animetimm/caformer_b36.dbv4-full', tag_idxs=None, tag_file=None):
        super().__init__()
        self.model = timm.create_model(f'hf-hub:{repo_id}', pretrained=True)
        self.model.requires_grad_(False)
        if tag_file is not None:
            tag_idxs = load_tag_idxs(tag_file, self.model.num_classes)
        self.tag_idxs = None if tag_idxs is None else list(tag_idxs)
        self.num_tags = self.model.num_classes if tag_idxs is None else len(self.tag_idxs)
        self.train(False)  # timm 建出来是 train 模式 (dropout/drop_path 生效)，不经过 trainer 直接用时也要是 eval

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
        self.train(False)

    @torch.no_grad()
    def forward(self, x):
        out = self.ccip(x)
        feat = out['features'] if isinstance(out, dict) else out[1]
        return feat.float()
