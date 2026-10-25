"""C′ 事后分析: 训练结束时在进程内 (fp32) 统计骨干逐层权重变化, 并导出逐层特征供离线 CKA。

权重变化必须对照原始 fp32 权重现场算: 短预算微调的 ΔW 往往小于 bf16 舍入误差, 事后用 bf16 存档比会失真。
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file

from v3.models.backbone import LoRALinear


@torch.no_grad()
def weight_change(backbone, ref_dir) -> dict:
    """每个 block 的 ||W - W0||_F / ||W0||_F, 按 attn / mlp / norm 分组; LoRA 用合并后的等效权重。"""
    ref = load_file(str(Path(ref_dir) / 'model.safetensors'))
    num, den = defaultdict(float), defaultdict(float)
    lora_bases = set()
    for name, mod in backbone.vit.named_modules():
        if isinstance(mod, LoRALinear):
            w = mod.base.weight.float() + (mod.lora_b.float() @ mod.lora_a.float()) * mod.scale
            items = [(f'{name}.weight', w)] + ([(f'{name}.bias', mod.base.bias.float())] if mod.base.bias is not None else [])
            lora_bases.add(f'{name}.base')
        elif name in lora_bases:
            continue
        elif not any(True for _ in mod.children()):
            items = [(f'{name}.{k}' if name else k, p.float()) for k, p in mod.named_parameters(recurse=False)]
        else:
            continue
        for key, w in items:
            key = key.replace('.base.', '.')
            if key not in ref:
                continue
            w0 = ref[key].to(w.device, torch.float32)
            m = re.match(r'blocks\.(\d+)\.(attn|mlp|norm1|norm2|ls1|ls2)', key)
            group = f'block{int(m.group(1)):02d}.' + ('norm' if m.group(2).startswith(('norm', 'ls')) else m.group(2)) \
                if m else key.split('.')[0]
            num[group] += (w - w0).pow(2).sum().item()
            den[group] += w0.pow(2).sum().item()
            num['_total'] += (w - w0).pow(2).sum().item()
            den['_total'] += w0.pow(2).sum().item()
    return {k: (num[k] / max(den[k], 1e-30)) ** 0.5 for k in sorted(num)}


@torch.no_grad()
def layer_features(backbone, images: torch.Tensor, bs: int = 64) -> dict:
    """所有 block 输出的 CLS 与 patch 均值 (B, depth, D), fp16; images 为已预处理的 (N,3,H,W)。"""
    taps_bak, pd_bak = backbone.tap_layers, backbone.patch_drop_prob
    backbone.tap_layers = tuple(range(1, backbone.depth + 1))
    backbone.set_patch_drop(0.0)
    was_training = backbone.training
    backbone.eval()
    cls, mean = [], []
    for i in range(0, len(images), bs):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = backbone(images[i:i + bs].cuda(non_blocking=True))
        cls.append(torch.stack([out['taps'][l][:, 0] for l in backbone.tap_layers], 1).half().cpu())
        mean.append(torch.stack([out['taps'][l][:, 1:].mean(1) for l in backbone.tap_layers], 1).half().cpu())
    backbone.tap_layers = taps_bak
    backbone.set_patch_drop(pd_bak)
    backbone.train(was_training)
    return dict(cls=torch.cat(cls), mean=torch.cat(mean))


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float() - x.float().mean(0)
    y = y.float() - y.float().mean(0)
    hsic = (x.t() @ y).pow(2).sum()
    return (hsic / ((x.t() @ x).pow(2).sum().sqrt() * (y.t() @ y).pow(2).sum().sqrt())).item()


def save_analysis(exp_dir, backbone, ref_dir, images=None):
    exp_dir = Path(exp_dir)
    wc = weight_change(backbone, ref_dir)
    (exp_dir / 'weight_change.json').write_text(json.dumps(wc, indent=1))
    if images is not None:
        torch.save(layer_features(backbone, images), exp_dir / 'layer_feats.pt')
    return wc
