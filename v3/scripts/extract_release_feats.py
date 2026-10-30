"""用发布的 HF CxIP 模型 (本地目录或 hub repo, AutoModel + AutoImageProcessor) 给任意评测集提特征, 存 runner 的 npz 格式。

特征 = get_image_features(normalize=False) (head 输出, CSIP v3 的事后擦除已折进 head), scale / bias 取模型 sim。
用法: CUDA_VISIBLE_DEVICES=5 python -m v3.scripts.extract_release_feats --model /data/neko/cxip_v3/release/AnimeCSIP-v3 \\
          --sets csip_unseen_v1 --out-dir /data/neko/cxip_v3/eval/release_csip_v3
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

from v3.eval.runner import load_set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--sets', nargs='+', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--batch-size', type=int, default=16)
    args = ap.parse_args()
    out = Path(args.out_dir)
    (out / 'feats').mkdir(parents=True, exist_ok=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
    proc = AutoImageProcessor.from_pretrained(args.model, trust_remote_code=True)
    scale, bias = float(model.sim.logit_scale.float().exp()), float(model.sim.logit_bias.float())
    (out / 'meta.json').write_text(json.dumps(dict(model=args.model, scale=scale, bias=bias), indent=1))
    print('scale', scale, 'bias', bias, flush=True)
    for name in args.sets:
        df, read = load_set(name)
        feats = []
        with ThreadPoolExecutor(16) as ex:
            for b0 in range(0, len(df), args.batch_size):
                imgs = list(ex.map(read, range(b0, min(len(df), b0 + args.batch_size))))
                feats.append(model.get_image_features(proc(imgs)['pixel_values'], normalize=False).float().cpu().numpy())
                if b0 % (args.batch_size * 50) == 0:
                    print(f'{name} {b0}/{len(df)}', flush=True)
        f = np.concatenate(feats)
        assert f.shape[0] == len(df) and np.isfinite(f).all()
        np.savez(out / 'feats' / f'{name}.npz', feats=f.astype(np.float16), sample_key=df['sample_key'].values.astype(str))
        print('saved', name, f.shape, flush=True)


if __name__ == '__main__':
    main()
