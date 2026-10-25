"""2B 骨干吞吐/显存实测 (单卡, 合成输入, bf16 autocast, 含 AdamW step)。

python -m v3.bench.bench_backbone --res 384 --mode last_n --bs 64 [--ckpt] [--compile] [--patch-drop 0.5]
每个配置输出一行 json 到 --out。
"""
import argparse
import json
import time

import torch

from v3.models.backbone import AnimeDINOBackbone

MODEL_PATH = '/data/neko/cxip_v3/models/AnimeDINOv3-timm'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', type=int, default=384)
    ap.add_argument('--mode', default='last_n', choices=['infer', 'frozen', 'last_n', 'full', 'lora'])
    ap.add_argument('--n-last', type=int, default=6)
    ap.add_argument('--lora-rank', type=int, default=32)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--ckpt', action='store_true')
    ap.add_argument('--compile', action='store_true')
    ap.add_argument('--patch-drop', type=float, default=0.0)
    ap.add_argument('--steps', type=int, default=12)
    ap.add_argument('--warmup', type=int, default=4)
    ap.add_argument('--out', default='/data/neko/cxip_v3/logs/bench_backbone.jsonl')
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device('cuda')
    model = AnimeDINOBackbone(MODEL_PATH, patch_drop=args.patch_drop).to(dev)
    train = args.mode != 'infer'
    model.setup_finetune('frozen' if args.mode == 'infer' else args.mode, n_last=args.n_last, lora_rank=args.lora_rank)
    model.to(dev)
    model.train(train)
    if args.ckpt:
        model.set_grad_checkpointing(True)
    if args.compile:
        model.compile_blocks()
    params = [p for p in model.parameters() if p.requires_grad]
    head = torch.nn.Linear(model.embed_dim, 768).to(dev)
    opt = torch.optim.AdamW(params + list(head.parameters()), lr=1e-5, fused=True) if train else None
    n_train = sum(p.numel() for p in params)

    x = torch.randn(args.bs, 3, args.res, args.res, device=dev)
    torch.cuda.reset_peak_memory_stats()
    times = []
    for step in range(args.warmup + args.steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if train:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(x)
                feat = head(torch.cat([out['cls'], out['patches'].mean(1)], 0))
                loss = feat.float().pow(2).mean()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        else:
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(x)
        torch.cuda.synchronize()
        if step >= args.warmup:
            times.append(time.perf_counter() - t0)
    dt = sorted(times)[len(times) // 2]
    rec = dict(res=args.res, mode=args.mode, n_last=args.n_last if args.mode == 'last_n' else None,
               lora_rank=args.lora_rank if args.mode == 'lora' else None, bs=args.bs, ckpt=args.ckpt,
               compile=args.compile, patch_drop=args.patch_drop, trainable_m=round(n_train / 1e6, 1),
               step_s=round(dt, 4), img_s=round(args.bs / dt, 1),
               peak_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 1))
    print(json.dumps(rec))
    with open(args.out, 'a') as f:
        f.write(json.dumps(rec) + '\n')


if __name__ == '__main__':
    main()
