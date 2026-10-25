"""验证导出的 HF 仓库: AutoModel/AutoImageProcessor (trust_remote_code) 的特征 vs 评测缓存特征 (cos), 以及打分参数。
python -m v3.scripts.verify_hf_export {ccip,csip} <repo_dir> [--n 24]
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

from v3.eval.runner import load_set

CASES = {
    'ccip': [('openset_full', '/data/neko/cxip_v3/exps/final/ccip_teacher_v1/eval/feats', None)],
    'csip': [(s, '/data/neko/cxip_v3/exps/csip_p2/dft_r640_tag001_x2/eval_native/feats',
              '/data/neko/cxip_v3/exps/csip_p2/dft_r640_tag001_x2/eval/posthoc_kc3_kh4.pt')
             for s in ('test100_0918', 'csip_eval_public')],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('task', choices=list(CASES))
    ap.add_argument('repo')
    ap.add_argument('--n', type=int, default=24)
    args = ap.parse_args()
    torch.set_num_threads(64)
    model = AutoModel.from_pretrained(args.repo, trust_remote_code=True, dtype=torch.bfloat16).eval()
    proc = AutoImageProcessor.from_pretrained(args.repo, trust_remote_code=True)
    print(type(model).__name__, model.dtype, type(proc).__name__, 'threshold', model.config.threshold)
    for name, feats_dir, erase in CASES[args.task]:
        df, read = load_set(name)
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(len(df), args.n, replace=False))
        imgs = [read(int(i)) for i in idx]
        t0 = time.time()
        with torch.no_grad():
            px = proc(imgs)['pixel_values']
            out = model(px)
        d = np.load(f'{feats_dir}/{name}.npz')
        pos = {str(k): i for i, k in enumerate(d['sample_key'])}
        ref = torch.from_numpy(d['feats'][[pos[str(df['sample_key'].iloc[i])] for i in idx]].astype(np.float32))
        if erase:
            e = torch.load(erase, weights_only=True)
            ref = ref @ e['M'] + e['b']
        cos = F.cosine_similarity(out.features.float(), ref, dim=-1)
        # 两两打分与参考特征打分的一致性
        s_ref = model.compute_logits(ref, ref)
        print(f'{name}: n={args.n} shapes={sorted({tuple(x.shape[-2:]) for x in px})[:4]} '
              f'cos min={cos.min():.4f} mean={cos.mean():.4f}  logit max|Δ|={(out.logits - s_ref).abs().max():.3f} '
              f'{time.time() - t0:.0f}s')
        f512 = model.get_image_features(px[:2] if torch.is_tensor(px) else px[:2], dim=512)
        print('  matryoshka 512 shape', tuple(f512.shape), 'norm', f512.norm(dim=-1).tolist())


if __name__ == '__main__':
    main()
