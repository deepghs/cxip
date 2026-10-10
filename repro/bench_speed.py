"""测速: 1) 模型 fwd+bwd 在 fp32 / tf32 / bf16 autocast 下的吞吐; 2) dataloader 在不同 workers 下的吞吐。
python repro/bench_speed.py <cfg.py> [model|data]"""
import sys
import time

import hydra
import torch
from rainbowneko.parser import load_config

_, cfg = load_config(sys.argv[1])
what = sys.argv[2] if len(sys.argv) > 2 else 'model'

if what == 'model':
    model = hydra.utils.instantiate(cfg.model.wrapper)().model.cuda()
    model.requires_grad_(False)
    for n in ('caformer', 'norm', 'head', 'sim', 'content_eraser', 'char_eraser'):
        getattr(model, n).requires_grad_(True)
    bs = 128
    img = torch.randn(bs, 3, 384, 384, device='cuda')
    timg = torch.randn(bs, 3, 448, 448, device='cuda')
    for name, tf32, dtype in [('fp32', False, None), ('tf32', True, None), ('bf16', True, torch.bfloat16)]:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        for part in ('full', 'backbone_only', 'teachers_only'):
            def step():
                with torch.autocast('cuda', dtype=dtype or torch.float32, enabled=dtype is not None):
                    if part == 'teachers_only':
                        model.tagger(timg); model.ccip(img); return
                    if part == 'backbone_only':
                        model.caformer.forward_features(img); s = torch.cat(model.style_list, 1); model.style_list.clear()
                        loss = model.head(model.norm(s)).float().pow(2).mean()
                    else:
                        out = model(img, timg)
                        loss = out['feat'][0].float().pow(2).mean() + out['content_logits'].float().mean() + out['char_logits'].float().mean()
                loss.backward()
            for _ in range(2): step()
            torch.cuda.synchronize(); t = time.time(); n = 5
            for _ in range(n): step()
            torch.cuda.synchronize(); dt = (time.time() - t) / n
            print(f'{name:5s} {part:14s} {dt:.3f}s/iter  {bs / dt:7.1f} img/s', flush=True)
elif what == 'data':
    from torch.utils.data import DataLoader
    ds_cfg = cfg.data_train['dataset1']
    ds = hydra.utils.instantiate(ds_cfg)
    ds = ds() if callable(ds) and not hasattr(ds, 'build_bucket') else ds
    ds.build_bucket(bs=128, world_size=1)
    for w in (2, 8, 16, 32):
        dl = DataLoader(ds, batch_size=None, num_workers=w, persistent_workers=False, prefetch_factor=4)
        it = iter(dl); next(it)
        t = time.time(); n = 0
        for i in range(max(8, w)):
            b = next(it); n += 128
        dt = time.time() - t
        print(f'workers={w:2d}  {n / dt:7.1f} img/s', flush=True)
        del it, dl

if what == 'opt':
    # bf16 下再试 channels_last / torch.compile (只测 backbone 训练, 这是大头)
    model = hydra.utils.instantiate(cfg.model.wrapper)().model.cuda()
    model.requires_grad_(False)
    for n in ('caformer', 'norm', 'head'):
        getattr(model, n).requires_grad_(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    bs = 128
    img = torch.randn(bs, 3, 384, 384, device='cuda')
    for name in ('bf16', 'bf16+cl', 'bf16+compile'):
        x = img
        fwd = model.caformer.forward_features
        if name == 'bf16+cl':
            model.caformer.to(memory_format=torch.channels_last); x = img.contiguous(memory_format=torch.channels_last)
        if name == 'bf16+compile':
            model.caformer.to(memory_format=torch.contiguous_format)
            fwd = torch.compile(model.caformer.forward_features)
        def step():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                fwd(x); s = torch.cat(model.style_list, 1); model.style_list.clear()
                loss = model.head(model.norm(s)).float().pow(2).mean()
            loss.backward()
        t0 = time.time()
        for _ in range(3): step()
        torch.cuda.synchronize(); print(name, f'warmup {time.time() - t0:.0f}s', flush=True)
        t = time.time(); n = 5
        for _ in range(n): step()
        torch.cuda.synchronize(); dt = (time.time() - t) / n
        print(f'{name:13s} backbone_only {dt:.3f}s/iter  {bs / dt:7.1f} img/s', flush=True)
