"""用发布的 HF CxIP 模型 (本地目录或 hub repo, AutoModel + AutoImageProcessor) 给任意评测集提特征, 存 runner 的 npz 格式。

特征 = get_image_features(normalize=False) (head 输出, CSIP v3 的事后擦除已折进 head), scale / bias 取模型 sim。
旧版 HF CxIP (CAFormer, 如 dghstmp/AnimeCSIP-v2.1) 没有 get_image_features: 同 eval_hf_ccip, fp32 权重 + bf16 autocast,
取 model(x)['features']。
用法: CUDA_VISIBLE_DEVICES=5 python -m v3.scripts.extract_release_feats --model /data/neko/cxip_v3/release/AnimeCSIP-v3 \\
          --sets csip_unseen_v1 --out-dir /data/neko/cxip_v3/eval/release_csip_v3
已存在的 npz 默认拒绝覆盖 (--overwrite 才覆盖), 符号链接目标一律拒绝; meta.json 内容不同也需要 --overwrite
(v3.eval.featio). 目标在加载模型之前全部检查.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

from v3.eval import featio
from v3.eval.runner import load_set


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--sets', nargs='+', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--overwrite', action='store_true', help='覆盖已存在的 npz / 不同的 meta.json (符号链接仍拒绝)')
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    targets = {name: out / 'feats' / f'{name}.npz' for name in args.sets}
    featio.preflight(targets.values(), out / 'meta.json', args.overwrite)          # 加载模型之前失败
    (out / 'feats').mkdir(parents=True, exist_ok=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True).eval()
    proc = AutoImageProcessor.from_pretrained(args.model, trust_remote_code=True)
    if hasattr(model, 'get_image_features'):
        model = model.to(torch.bfloat16).cuda()

        def encode(imgs):
            return model.get_image_features(proc(imgs)['pixel_values'], normalize=False)
    else:                                                    # 旧版 HF CxIP, 同 eval_hf_ccip
        model = model.cuda()

        @torch.no_grad()
        def encode(imgs):
            x = torch.stack([torch.tensor(np.asarray(proc(im)['pixel_values'][0])) for im in imgs]).cuda()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                return model(x)['features']
    scale, bias = float(model.sim.logit_scale.float().exp()), float(model.sim.logit_bias.float())
    featio.write_json(out / 'meta.json', dict(model=args.model, scale=scale, bias=bias), args.overwrite)
    print('scale', scale, 'bias', bias, flush=True)
    for name in args.sets:
        df, read = load_set(name)
        feats = []
        with ThreadPoolExecutor(16) as ex:
            for b0 in range(0, len(df), args.batch_size):
                imgs = list(ex.map(read, range(b0, min(len(df), b0 + args.batch_size))))
                feats.append(encode(imgs).float().cpu().numpy())
                if b0 % (args.batch_size * 50) == 0:
                    print(f'{name} {b0}/{len(df)}', flush=True)
        f = np.concatenate(feats)
        assert f.shape[0] == len(df) and np.isfinite(f).all()
        featio.save_feats(targets[name], f, df['sample_key'].values, args.overwrite)
        print('saved', name, f.shape, flush=True)


if __name__ == '__main__':
    main()
