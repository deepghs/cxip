"""
CSIP v2.1 角色伪标签第一步: 用冻结的 CCIP (与 csip2_1.py 的 FrozenCCIP 相同, HF dghstmp/AnimeCCIP) 提取训练集特征。
原流程 (cfgs/py/infer/ccip_save.py) 每张图存一个 .pt, 400 多万个小文件会拖垮共享盘 (见 experiments/README.md),
这里每个 rank 写一个 fp16 memmap + 文件名列表, 预处理与 ccip_save.py 相同 (Resize 384 + ANIME_MEAN/STD)。

每张卡起一个进程:
  for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$((i+4)) python experiments/data_prep/extract_ccip_feats.py --rank $i --world 4 & done
输出: {out}/feats_r{rank}.npy (N,768 fp16), {out}/files_r{rank}.json; scripts/char_labels/ccip_cls.py --feat-npy {out} 读取
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

ANIME_MEAN = (0.67713961, 0.618461, 0.61174436)
ANIME_STD = (0.31741801, 0.31876716, 0.31473021)


class ImgDataset(Dataset):
    def __init__(self, root, files):
        self.root, self.files = root, files
        self.transform = T.Compose([T.Resize((384, 384)), T.ToTensor(), T.Normalize(mean=ANIME_MEAN, std=ANIME_STD)])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.root, self.files[i]))
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGBA')
            bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img)
        return self.transform(img.convert('RGB'))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    data = Path(os.environ.get('REPRO_ROOT', '/data/neko/cxip_repro')) / 'data/danbooru-w640-ws-full-artist'
    parser.add_argument('--label-file', default=str(data / 'train_split.json'))
    parser.add_argument('--img-root', default=str(data / 'train'))
    parser.add_argument('--out', default=str(data / 'train_ccip_feats'))
    parser.add_argument('--model', default='dghstmp/AnimeCCIP')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world', type=int, default=1)
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()

    with open(args.label_file) as f:
        files = sorted(json.load(f))[args.rank::args.world]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    feat_path, done_path = out / f'feats_r{args.rank}.npy', out / f'files_r{args.rank}.json'
    if done_path.exists():
        print(f'rank {args.rank} already done')
        return

    model = AutoModel.from_pretrained(args.model, trust_remote_code=True).cuda().eval()
    feats = np.lib.format.open_memmap(feat_path, mode='w+', dtype=np.float16, shape=(len(files), 768))
    dl = DataLoader(ImgDataset(args.img_root, files), batch_size=args.bs, num_workers=args.workers)
    i = 0
    for step, x in enumerate(dl):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            f = model(x.cuda(non_blocking=True))['features']
        feats[i:i + len(f)] = f.float().cpu().numpy().astype(np.float16)
        i += len(f)
        if step % 200 == 0:
            print(f'rank {args.rank}: {i}/{len(files)}', flush=True)
    feats.flush()
    assert i == len(files)
    with open(done_path, 'w') as f:  # 最后写, 作为完成标记
        json.dump(files, f)
    print(f'rank {args.rank} done: {len(files)}')


if __name__ == '__main__':
    main()
