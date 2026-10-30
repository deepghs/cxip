"""第三方 Kaloscope 3.0 preview (heathcliff01/Kaloscope3.0-preview, DINOv3 ViT-B/16) 在 CSIP 评测集上的特征。

预处理 / 前向与模型卡片一致 (代码 Chenkin-x/kaloscope-dinov3 @95ade97): rgb_loader (EXIF 转正, 透明合成白底)
→ Resize(512) → CenterCrop(512) → uint8, Encoder 内部做 ImageNet 归一化; 输出 1792 维 = 主干 1536 + 投影 256。
保存两份特征 (runner 的 npz 格式, 未归一化 fp16): proj256/feats (卡片公开指标用的检索向量) 与 backbone1536/feats。
打分用检查点里训练出的 log_temperature / bias (训练损失 softplus(-y (exp(t) cos + b)), 与 CxIP 的标定形式相同)。
检查点用 weights_only=True 加载 (只放行 RNG 状态里的 numpy 类型), 存出只含 model / model_config 的 model_only.pt。
用法: CUDA_VISIBLE_DEVICES=5 python -m v3.scripts.eval_kaloscope [--sets test100_0918 csip_eval_public]
然后对每份特征跑 v3.scripts.csip_full_eval (见 --print-cmds)。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from v3.eval.runner import load_set

EXT = Path('/data/neko/ext_models/kaloscope3_preview')
OUT = Path('/data/neko/cxip_v3/eval/kaloscope3')
sys.path.insert(0, str(EXT / 'code/repo'))
from dinov3.finetune.data import rgb_loader  # noqa: E402
from dinov3.finetune.temporal.model import Encoder  # noqa: E402
from torchvision import transforms  # noqa: E402


class Images(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.spatial = transforms.Compose([transforms.Resize(512), transforms.CenterCrop(512)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return torch.from_numpy(np.array(self.spatial(rgb_loader(self.paths[i])), copy=True)).permute(2, 0, 1)


def load_model():
    ck = torch.load(EXT / 'model_only.pt', map_location='cpu', weights_only=True)
    cfg = dict(ck['model_config'])
    cfg['weights'] = None
    m = Encoder(cfg, initialize=False)
    m.load_state_dict(ck['model'], strict=True)
    return m.eval(), float(ck['model']['log_temperature'].exp()), float(ck['model']['bias'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', nargs='+', default=['test100_0918', 'csip_eval_public'])
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--print-cmds', action='store_true')
    args = ap.parse_args()
    model, scale, bias = load_model()
    meta = dict(model='heathcliff01/Kaloscope3.0-preview@5e4bfa2', code='Chenkin-x/kaloscope-dinov3@95ade97',
                scale=scale, bias=bias, preprocess='rgb_loader, Resize(512), CenterCrop(512), uint8')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'meta.json').write_text(json.dumps(meta, indent=1))
    if args.print_cmds:
        for v, (s, b) in (('proj256', (scale, bias)), ('backbone1536', (1.0, 0.0))):
            print(f'python -m v3.scripts.csip_full_eval --name kaloscope3_{v} --feats-dir {OUT}/{v}/feats '
                  f'--scale {s} --bias {b} --out /data/neko/cxip_v3/eval/csip_full/kaloscope3_{v}.json')
        return
    model = model.cuda()
    for name in args.sets:
        df, _ = load_set(name)
        dl = DataLoader(Images(df['path'].tolist()), batch_size=args.batch_size, num_workers=16, pin_memory=True)
        outs = []
        with torch.inference_mode():
            for i, x in enumerate(dl):
                outs.append(model(x.cuda(non_blocking=True)).float().cpu())
                if i % 20 == 0:
                    print(f'{name} {i * args.batch_size}/{len(df)}', flush=True)
        o = torch.cat(outs)
        assert o.shape == (len(df), 1792) and torch.isfinite(o).all()
        print(name, 'proj norm (should be ~unnormalized)', o[:, 1536:].norm(dim=-1).mean().item())
        for v, f in (('backbone1536', o[:, :1536]), ('proj256', o[:, 1536:])):
            d = OUT / v / 'feats'
            d.mkdir(parents=True, exist_ok=True)
            np.savez(d / f'{name}.npz', feats=f.numpy().astype(np.float16), sample_key=df['sample_key'].values.astype(str))
        print('saved', name, flush=True)


if __name__ == '__main__':
    main()
