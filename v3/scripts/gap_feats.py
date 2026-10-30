"""CSIP v3 vs Kaloscope 差距分析用的补充特征 (结果写到独立目录, 不动实验目录里的 eval_native/suite.json)。

  exp       内部实验 checkpoint (未擦除) 在任意评测集上的 native_full 特征 (短边 <= 640, 与 eval_native.sh 相同)
  kalbias   Kaloscope 投影向量的偏置探针 I_content / I_char (runner.run_bias, 特征缓存到 kaloscope3/proj256/feats)
用法: CUDA_VISIBLE_DEVICES=5 python -m v3.scripts.gap_feats exp <exp-dir> <out-dir> --sets csip_unseen_v1
      CUDA_VISIBLE_DEVICES=5 python -m v3.scripts.gap_feats kalbias
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from v3.eval import runner
from v3.eval_exp import load_exp_model, make_encode
from v3.transforms import NativeFullEval


def exp_feats(exp, out, sets, bs):
    model, cfg = load_exp_model(Path(exp))
    model = model.cuda().eval().to(torch.bfloat16)
    if getattr(model.head, 'uses_taps', False):
        model.head.float()
    encode = make_encode(lambda x: model(x.to(torch.bfloat16))['embedding'], NativeFullEval(short_side=640))
    scale, bias = model.sim.logit_scale.exp().item(), model.sim.logit_bias.item()
    out = Path(out)
    (out / 'feats').mkdir(parents=True, exist_ok=True)
    (out / 'meta.json').write_text(json.dumps(dict(exp=str(exp), scale=scale, bias=bias), indent=1))
    for name in sets:
        df, read = runner.load_set(name)
        f = _extract(encode, df, read, bs)
        np.savez(out / 'feats' / f'{name}.npz', feats=f.astype(np.float16), sample_key=df['sample_key'].values.astype(str))
        print('saved', name, f.shape, 'scale', scale, 'bias', bias, flush=True)


def _extract(encode, df, read, bs):
    feats = []
    for b0 in range(0, len(df), bs):
        feats.append(encode([read(i) for i in range(b0, min(len(df), b0 + bs))]).cpu().numpy())
    f = np.concatenate(feats)
    assert len(f) == len(df) and np.isfinite(f).all()
    return f


def kal_bias(bs):
    from torchvision import transforms
    from v3.scripts.eval_kaloscope import OUT, load_model
    from dinov3.finetune.data import prepare_rgb
    model, _, _ = load_model()
    model = model.cuda()
    spatial = transforms.Compose([transforms.Resize(512), transforms.CenterCrop(512)])

    @torch.inference_mode()
    def encode(imgs):
        x = torch.stack([torch.from_numpy(np.array(spatial(prepare_rgb(im)), copy=True)).permute(2, 0, 1) for im in imgs])
        return model(x.cuda())[:, 1536:].float()

    res = runner.run_bias(encode, 'csip', 'kaloscope3_proj256', feats_dir=OUT / 'proj256' / 'feats', batch_size=bs)
    (OUT / 'bias_proj256.json').write_text(json.dumps(res, indent=1, default=float))
    print('I_content', res['content'].get('I'), 'I_char', res['char'].get('I'), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('exp')
    e.add_argument('exp')
    e.add_argument('out')
    e.add_argument('--sets', nargs='+', required=True)
    e.add_argument('--bs', type=int, default=16)
    k = sub.add_parser('kalbias')
    k.add_argument('--bs', type=int, default=64)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    if args.cmd == 'exp':
        exp_feats(args.exp, args.out, args.sets, args.bs)
    else:
        kal_bias(args.bs)


if __name__ == '__main__':
    main()
