"""对训练实验 (或冻结骨干基线) 跑评测套件 v2: 开放集 FULL/FAST (按 subset / backbone_seen 分组)、Matryoshka 前缀、I_style。

python -m v3.eval_exp /data/neko/cxip_v3/exps/ccip_p1/base [more exps...] [--sets openset_full openset_fast] [--no-bias]
python -m v3.eval_exp frozen --res 384      # 冻结 CLS 基线 (纯 cos)
结果写到 <exp>/eval_full.json (冻结基线写到 /data/neko/cxip_v3/eval/results/frozen_cls_r{res}.json)。
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from v3.eval import runner
from v3.models.backbone import AnimeDINOBackbone
from v3.transforms import EvalTransform, NativeCropEval, NativeFullEval

MODEL_PATH = '/data/neko/cxip_v3/models/AnimeDINOv3-timm'
RESULTS = Path('/data/neko/cxip_v3/eval/results')
DEFAULT_SETS = ['openset_full', 'openset_fast']
CSIP_SETS = ['test100_0918', 'test100', 'csip_eval_public']  # test100v2 = 0918 切分未过滤版, 过滤缺图后与 test100_0918 相同


def load_exp_model(exp, ckpt='last.pt'):
    from v3.train_cxip import build_model
    cfg = json.loads((exp / 'cfg.json').read_text())
    mc = dict(cfg['model'], patch_drop=0.0)
    model = build_model(mc)
    state = torch.load(exp / ckpt, map_location='cpu', weights_only=True)['model']
    missing, unexpected = model.load_state_dict(state, strict=False)
    frozen = {n for n, p in model.named_parameters() if n.startswith('backbone.') and not p.requires_grad}
    bad = [k for k in missing if k not in frozen]
    assert not bad and not unexpected, (bad[:5], unexpected[:5])
    return model, cfg


def make_encode(fn, tf, bs_device='cuda', workers=16):
    pool = ThreadPoolExecutor(workers)  # PIL resize 释放 GIL, 预处理并行; 主线程只做前向

    @torch.no_grad()
    def encode(imgs):
        xs = list(pool.map(tf, imgs))
        if all(x.shape == xs[0].shape for x in xs):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                return fn(torch.stack(xs).to(bs_device, non_blocking=True)).float()
        # 尺寸不一 (native_full): 按尺寸分桶, 每批 token 数不超过 len(imgs) 张 384² 图
        out, budget = [None] * len(xs), len(xs) * 24 * 24
        buckets = {}
        for i, x in enumerate(xs):
            buckets.setdefault(tuple(x.shape), []).append(i)
        for shape, idx in buckets.items():
            bs = max(1, budget // ((shape[1] // 16) * (shape[2] // 16)))
            for k in range(0, len(idx), bs):
                sel = idx[k:k + bs]
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    f = fn(torch.stack([xs[i] for i in sel]).to(bs_device, non_blocking=True)).float()
                for i, row in zip(sel, f):
                    out[i] = row
        return torch.stack(out)
    return encode


def derive_subset_feats(feats_dir, src, dst):
    """dst 集合的样本是 src 的子集时, 按 sample_key 从 src 的缓存特征里切出 dst 的缓存, 免去重复提取。"""
    sp, dp = Path(feats_dir) / f'{src}.npz', Path(feats_dir) / f'{dst}.npz'
    if dp.exists() or not sp.exists():
        return
    d = np.load(sp, allow_pickle=False)
    pos = dict(zip(d['sample_key'], range(len(d['sample_key']))))
    keys = runner.load_set(dst)[0]['sample_key'].values.astype(str)
    if all(k in pos for k in keys):
        idx = np.array([pos[k] for k in keys])
        np.savez(dp, feats=d['feats'][idx], sample_key=keys)


def matryoshka_eval(feats_dir, dims, scale, bias, name='openset_full'):
    """用缓存的完整特征切前缀, 只算 pairwise + zeroshot。"""
    df, _ = runner.load_set(name)
    d = np.load(Path(feats_dir) / f'{name}.npz', allow_pickle=False)
    out = {}
    for k in dims:
        f = d['feats'][:, :k].astype(np.float32)
        r = runner.evaluate_set(name, f, df, scale, bias, tasks=('pairwise', 'zeroshot'))
        out[k] = {m: r[m] for m in ('pr_auc', 'best_f1', 'zs_ref1_top1') if m in r}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('exps', nargs='+')
    ap.add_argument('--sets', nargs='+', default=DEFAULT_SETS)
    ap.add_argument('--res', type=int)
    ap.add_argument('--no-bias', action='store_true')
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--protocol', default=None, choices=[None, 'native_full'],
                    help='native_full: 整图原尺度 (长边≤1280, 按尺寸分桶), 仅适合尺寸无关的头; 结果目录加 _native')
    ap.add_argument('--ckpt', default='last.pt', help='last_raw.pt = 开 EMA 时的非 EMA 权重 (结果目录加 _last_raw)')
    ap.add_argument('--short-side', type=int, default=None,
                    help='CSIP 预处理: 短边先缩到不超过该值 (默认 640 = 训练数据尺度; 0 关闭, 结果目录加 _ss0)')
    ap.add_argument('--upscale', action='store_true',
                    help='native_full: 短边不足 short_side 的小图也放大到该尺度 (结果目录加 _up)')
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(4)  # 224 核默认开满线程, 小张量运算互相抢占反而慢

    for e in args.exps:
        if e == 'frozen':
            res = args.res or 384
            bb = AnimeDINOBackbone(MODEL_PATH).cuda().eval().to(torch.bfloat16)
            encode = make_encode(lambda x: F.normalize(bb(x.to(torch.bfloat16))['cls'].float(), dim=-1), EvalTransform(res))
            is_csip, sets = False, args.sets
            name, scale, bias = f'frozen_cls_r{res}', None, None
            out_dir = RESULTS / name
            dims = ()
        else:
            exp = Path(e)
            model, cfg = load_exp_model(exp, args.ckpt)
            model = model.cuda().eval().to(torch.bfloat16)
            if getattr(model.head, 'uses_taps', False):
                model.head.float()  # 画风头的统计量是 fp32 (含 BN), 与训练一致: 头保持 fp32, Linear 由 autocast 降到 bf16
            sched = cfg['data'].get('res_schedule') or []
            res = args.res or cfg.get('eval', {}).get('res') or (int(sched[-1][1]) if sched else cfg['data']['res'])
            # 按任务判断 (CSIP 也可能用 query 头)
            is_csip = cfg['data'].get('transform') == 'csip' or cfg['model'].get('head') == 'style'
            ss = (640 if args.short_side is None else args.short_side) if is_csip else None
            if args.protocol == 'native_full':
                tf = NativeFullEval(short_side=ss, upscale=args.upscale)
            elif cfg['data'].get('eval_transform') == 'native_crop':
                tf = NativeCropEval(res, short_side=ss)
            else:
                tf = EvalTransform(res)
            encode = make_encode(lambda x: model(x.to(torch.bfloat16))['embedding'], tf)
            sets = CSIP_SETS if is_csip and args.sets == DEFAULT_SETS else args.sets
            suffix = (f'_r{res}' if args.res else '') + ('_native' if args.protocol else '') + \
                (f'_ss{args.short_side}' if is_csip and args.short_side not in (None, 640) else '') + \
                ('_up' if args.upscale else '') + ('' if args.ckpt == 'last.pt' else '_' + Path(args.ckpt).stem)
            name = exp.name + suffix
            scale, bias = model.sim.logit_scale.exp().item(), model.sim.logit_bias.item()
            out_dir = exp / f'eval{suffix}'
            dims = [d for d in model.matryoshka if d != model.matryoshka[0]]
        result = None
        for set_name in sets:  # 逐个集合跑, 以便 FAST 直接复用 FULL 的特征
            if set_name == 'openset_fast':
                derive_subset_feats(out_dir / 'feats', 'openset_full', 'openset_fast')
            r = runner.run_suite(encode, name, scale, bias, sets=[set_name], feats_dir=out_dir / 'feats',
                                 batch_size=args.bs)
            if result is None:
                result = r
            else:
                result['datasets'].update(r['datasets'])
        if dims:
            result['matryoshka'] = matryoshka_eval(out_dir / 'feats', dims, scale, bias, name=sets[0])
        if not args.no_bias:
            result['bias'] = runner.run_bias(encode, 'csip' if is_csip else 'ccip', name, feats_dir=out_dir / 'feats',
                                             batch_size=args.bs)
        (out_dir / 'suite.json').write_text(json.dumps(result, indent=1, default=float))
        print(json.dumps(summarize(result), ensure_ascii=False), flush=True)
        del encode
        torch.cuda.empty_cache()


def summarize(r, keys=('pr_auc', 'best_f1', 'zs_ref1_top1', 'zs_top1', 'map_at_r', 'r_at_1')):
    """抽取核心数字: 各集合的标量指标 (按 keys 过滤, 找不到就全给), 分组 PR-AUC, Matryoshka, I_style。"""
    s = dict(model=r['model'])
    for name, d in r['datasets'].items():
        scalars = {k: round(v, 4) for k, v in d.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        s[name] = {k: v for k, v in scalars.items() if k in keys} or scalars
        for g, gd in (d.get('groups') or {}).items():
            if 'pr_auc' in gd:
                s[name][f'{g}/pr_auc'] = round(gd['pr_auc'], 4)
    if 'matryoshka' in r:
        s['matryoshka_pr_auc'] = {k: round(v.get('pr_auc', float('nan')), 4) for k, v in r['matryoshka'].items()}
    if 'bias' in r:
        s['bias_style'] = r['bias'].get('style')
    return s


if __name__ == '__main__':
    main()
