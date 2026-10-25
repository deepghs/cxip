"""把 CCIP v3 / CSIP v3 导出成 transformers 可直接加载的格式 (trust_remote_code, 代码在 v3/release/)。

CCIP: final/ccip_teacher_v1 (冻结的前 16 层从 AnimeDINOv3-timm 补齐), 输入白底 pad 方形 → 512。
CSIP: csip_p2/dft_r640_tag001_x2 + 事后线性擦除 posthoc_kc3_kh4 (x @ M + b 折进 head.proj 最后一层), 输入原尺度。
权重存 bf16 (评测时整模型就是 bf16)。
用法: python -m v3.scripts.export_hf {ccip,csip} --out /data/neko/cxip_v3/release/<repo>
"""
import argparse
import json
from pathlib import Path

import torch

from v3.eval_exp import load_exp_model
from v3.release.configuration_cxip import CxIPConfig
from v3.release.image_processing_cxip import CxIPImageProcessor
from v3.release.modeling_cxip import CxIPModel

EX = Path('/data/neko/cxip_v3/exps')
EV = Path('/data/neko/cxip_v3/eval')
SPECS = {
    'ccip': dict(exp=EX / 'final/ccip_teacher_v1', erase=None,
                 proc=dict(mode='pad_square', size=512)),
    'csip': dict(exp=EX / 'csip_p2/dft_r640_tag001_x2', erase=EX / 'csip_p2/dft_r640_tag001_x2/eval/posthoc_kc3_kh4.pt',
                 proc=dict(mode='native', short_side=640, max_side=1280)),
}


def threshold(task):
    """默认判定阈值 (logit): CCIP 取开放集 FULL 的 best-F1 阈值; CSIP 两个测试集最优阈值差得远, 取两者均值 (共享阈值)。"""
    if task == 'ccip':
        return json.loads((SPECS['ccip']['exp'] / 'eval/suite.json').read_text())['datasets']['openset_full']['best_threshold']
    return json.loads((EV / 'csip_full/v3_x2_erased.json').read_text())['threshold_drift']['shared_threshold']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('task', choices=list(SPECS))
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    spec = SPECS[args.task]
    src, cfg = load_exp_model(spec['exp'])
    sd = {}
    for k, v in src.state_dict().items():
        if k.startswith('backbone.vit.'):
            sd['backbone.' + k[len('backbone.vit.'):]] = v.float()
        elif k.startswith(('head.', 'sim.')):
            sd[k] = v.float()
    if spec['erase']:
        e = torch.load(spec['erase'], weights_only=True)
        M, b = e['M'].float(), e['b'].float()
        W, c = sd['head.proj.2.weight'], sd['head.proj.2.bias']  # y = x W^T + c → (x W^T + c) M + b
        sd['head.proj.2.weight'] = M.t() @ W
        sd['head.proj.2.bias'] = c @ M + b
    bb = src.backbone.config
    mc = cfg['model']
    config = CxIPConfig(
        task=args.task, image_size=bb['image_size'], patch_size=bb['patch_size'], embed_dim=bb['embed_dim'],
        depth=bb['depth'], num_heads=bb['num_heads'], mlp_ratio=bb['mlp_ratio'], num_queries=mc['num_queries'],
        out_dim=mc['out_dim'], matryoshka_dims=mc['matryoshka'], logit_scale_init=float(sd['sim.logit_scale']),
        logit_bias_init=float(sd['sim.logit_bias']), threshold=round(float(threshold(args.task)), 4),
        torch_dtype='bfloat16')
    del src
    CxIPConfig.register_for_auto_class()
    CxIPModel.register_for_auto_class('AutoModel')
    CxIPImageProcessor.register_for_auto_class('AutoImageProcessor')
    model = CxIPModel(config)
    model.load_state_dict(sd, strict=True)
    model = model.to(torch.bfloat16).eval()
    model.config.architectures = ['CxIPModel']
    model.save_pretrained(args.out, safe_serialization=True, max_shard_size='10GB')
    CxIPImageProcessor(**spec['proc']).save_pretrained(args.out)
    print('saved', args.out, sum(p.numel() for p in model.parameters()) / 1e9, 'B params')


if __name__ == '__main__':
    main()
