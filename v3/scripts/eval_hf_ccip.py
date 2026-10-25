"""旧版 HF AnimeCCIP / AnimeCSIP 在 v3 评测套件上的结果 (默认: CCIP 开放集 FULL + I_style), 与 v3 teacher 直接可比。
CSIP: --model dghstmp/AnimeCSIP-v2.1 --sets test100_0918 csip_eval_public --no-bias (CSIP 偏置已有 results/bias_hf.json)。

预处理 / 前向与 HF 模型卡片一致 (AutoImageProcessor + AutoModel, 取 out['features']), 打分用模型自带的
logit_scale / logit_bias。GPU 不可用时自动用 CPU (CAFormer ~100M, 3.2 万张约半小时)。
用法: python -m v3.scripts.eval_hf_ccip [--model dghstmp/AnimeCCIP] [--out-dir /data/neko/cxip_v3/eval/hf_ccip]
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

from v3.eval.runner import run_bias, run_suite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='dghstmp/AnimeCCIP')
    ap.add_argument('--out-dir', default='/data/neko/cxip_v3/eval/hf_ccip')
    ap.add_argument('--sets', nargs='+', default=['openset_full'])
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--no-bias', action='store_true')
    args = ap.parse_args()
    out = Path(args.out_dir)
    (out / 'feats').mkdir(parents=True, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True).to(dev).eval()
    proc = AutoImageProcessor.from_pretrained(args.model, trust_remote_code=True)
    scale, bias = float(model.sim.logit_scale.exp()), float(model.sim.logit_bias)

    @torch.no_grad()
    def encode(imgs):
        x = torch.stack([torch.tensor(np.asarray(proc(im)['pixel_values'][0])) for im in imgs]).to(dev)
        with torch.autocast(dev, dtype=torch.bfloat16):
            return model(x)['features'].float()

    run_suite(encode, args.model, scale, bias, sets=tuple(args.sets), out=out / 'suite.json', feats_dir=out / 'feats',
              batch_size=args.batch_size)
    if args.no_bias:
        return
    r = run_bias(encode, 'ccip', args.model, out=out / 'bias.json', feats_dir=out / 'feats',
                 batch_size=args.batch_size)
    print({k: round(v['I'], 4) for k, v in r.items() if isinstance(v, dict)})


if __name__ == '__main__':
    main()
