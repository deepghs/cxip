import os

import torch
from rainbowneko.infer import BasicAction


class SaveFeatAction(BasicAction):
    '''
    把每张图的特征单独存成 {save_dir}/{id}.pt (float32, [C])，id 保留后缀，如 12345.webp -> 12345.webp.pt
    这样各读取方都能对上:
      - eval_csip*.py: emb_dir / (label_file 的 key + ".pt")，key 形如 "12345.webp"
      - eval_ccip_zero.py / evaluate/char_bias.py: 类别取 stem.split('_')[0]
      - evaluate/content_bias.py: 用 webdataset 时 id 是 __key__ (无后缀)，stem 与 probs.pkl 的 key 一致
      - tools/ccip_cls.py -> tools/clusters_to_char_labels.py: 去掉 ".pt" 即 train_split.json 的 key
    '''

    def __init__(self, save_dir, feat_key='feat', key_map_in=None, key_map_out=None):
        super().__init__(key_map_in=key_map_in, key_map_out=key_map_out)
        self.save_dir = save_dir
        self.feat_key = feat_key
        os.makedirs(save_dir, exist_ok=True)

    def forward(self, output, id, **states):
        feat = output[self.feat_key]
        if isinstance(feat, (list, tuple)):  # FeatWrapper 输出 feat=[feat]
            feat = feat[0]
        feat = feat.float().cpu()
        for img_id, f in zip(id, feat):
            name = str(img_id).replace('/', '_')  # ImageFolder 类 source 的 id 带子目录
            torch.save(f.clone(), os.path.join(self.save_dir, f'{name}.pt'))  # clone: 只存这一行，不存整个 batch 的 storage
