"""CCIP teacher 训练 (AnimeDINOv3 2B + QueryPoolHead + SubCenterArcFace/SigLIP, P×K 采样, LP-FT)。

torchrun --nproc_per_node 4 -m v3.train_ccip --config v3/configs/ccip_p1_base.py [key=value ...]
"""
import argparse
import ast
import json
import math
import os
import runpy
import time
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')  # 须在 CUDA 初始化前; LP→FT 换形状后减少碎片

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from v3.losses import MatryoshkaLoss, dynamic_margins
from v3.models.backbone import AnimeDINOBackbone
from v3.models.heads import CxIPModel, QueryPoolHead
from v3.transforms import CCIPTrainTransform, EvalTransform


# ---------------- config ----------------
def load_cfg(path, overrides):
    cfg = runpy.run_path(path)['CFG']
    for item in overrides:
        key, val = item.split('=', 1)
        try:
            val = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            pass
        d = cfg
        *parents, last = key.split('.')
        for p in parents:
            d = d[p]
        d[last] = val
    return cfg


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def log(*args):
    if is_main():
        print(time.strftime('%H:%M:%S'), *args, flush=True)


# ---------------- data ----------------
def collate(batch):
    return torch.stack([b['image'] for b in batch]), torch.tensor([b['label'] for b in batch], dtype=torch.long)


def load_train_df(dc):
    """读训练索引 (可多个 parquet), 剔除开放集留出类与样本, 标签重映射为 0..C-1。"""
    paths = dc['index'] if isinstance(dc['index'], (list, tuple)) else [dc['index']]
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    for p in paths:  # 记录索引指纹: 便于发现数据在实验之间被改动
        st = os.stat(p)
        log(f'index {p} size {st.st_size} mtime {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))}')
    if dc.get('exclude'):
        ex = json.loads(Path(dc['exclude']).read_text())
        bad_keys = set(ex.get('exclude_sample_keys', [])) | set(ex.get('zid_overlap_sample_keys', []))
        n0 = len(df)
        df = df[~df['label'].isin(set(ex['heldout_labels'])) & ~df['sample_key'].isin(bad_keys)]
        log(f'exclude open-set: {n0} -> {len(df)} images')
    df = df.reset_index(drop=True)
    df['orig_label'] = df['label']
    df['label'], classes = pd.factorize(df['orig_label'], sort=True)
    group = None
    gk = dc.get('group_key')
    if gk and gk in df.columns:
        group = df[gk].values.astype(np.int64) if np.issubdtype(df[gk].dtype, np.integer) \
            else np.where(df[gk].fillna('').values == '', -1, pd.factorize(df[gk].fillna(''))[0])
    return df, len(classes), group


def make_train_loader(dc, df, group, seed, steps):
    from v3.data.indexed_tar import IndexedTarDataset
    from v3.data.pk_sampler import PKBatchSampler
    rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
    ds = IndexedTarDataset(dc['root'], df, transform=CCIPTrainTransform(dc['res'], **dc.get('aug', {})))
    sampler = PKBatchSampler(df['label'].values, dc['P'], dc['K'], alpha=dc.get('alpha', 0.5), group_ids=group,
                             group_frac=dc.get('group_frac', 0.25) if group is not None else 0.0,
                             rank=rank, world_size=world, seed=seed, num_batches=steps)
    return DataLoader(ds, batch_sampler=sampler, num_workers=dc.get('workers', 16), collate_fn=collate,
                      pin_memory=True, persistent_workers=True, prefetch_factor=4)


def load_eval_sets(ec, res):
    from v3.data.indexed_tar import IndexedTarDataset
    sets = []
    for s in ec.get('sets', []):
        if not Path(s['index']).exists():
            log(f'[eval] skip {s["name"]}: {s["index"]} missing')
            continue
        df = pd.read_parquet(s['index'])
        if s.get('columns_map'):  # 例: 开放集打包 tar 的 pack_tar/pack_offset -> shard/offset
            df = df.drop(columns=[v for v in s['columns_map'].values() if v in df.columns])
            df = df.rename(columns=s['columns_map'])
        if s.get('max_classes'):
            rng = np.random.RandomState(0)
            keep = rng.permutation(df['label'].unique())[:s['max_classes']]
            df = df[df['label'].isin(keep)]
        if s.get('max_per_class'):
            df = df.groupby('label', group_keys=False).head(s['max_per_class'])
        df = df.reset_index(drop=True)
        sets.append((s['name'], IndexedTarDataset(s['root'], df, transform=EvalTransform(res)), df['label'].values))
        log(f'[eval] {s["name"]}: {len(df)} images / {df.label.nunique()} classes')
    return sets


# ---------------- eval ----------------
@torch.no_grad()
def encode(model, ds, bs=128):
    rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
    idx = np.array_split(np.arange(len(ds)), world)[rank]
    loader = DataLoader(torch.utils.data.Subset(ds, idx), batch_size=bs, num_workers=8, collate_fn=collate)
    feats = []
    for x, _ in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            feats.append(model(x.cuda(non_blocking=True))['embedding'].float())
    f = torch.cat(feats) if feats else torch.zeros(0, model.head.proj[-1].out_features, device='cuda')
    if world > 1:
        n = torch.tensor([len(f)], device='cuda')
        ns = [torch.zeros_like(n) for _ in range(world)]
        dist.all_gather(ns, n)
        mx = int(max(x.item() for x in ns))
        pad = torch.zeros(mx, f.shape[1], device='cuda')
        pad[:len(f)] = f
        out = [torch.zeros_like(pad) for _ in range(world)]
        dist.all_gather(out, pad)
        f = torch.cat([o[:int(k.item())] for o, k in zip(out, ns)])
    return f


@torch.no_grad()
def pair_metrics(feat, labels, sim):
    """全部两两配对: PR-AUC (AP)、best F1、zero-shot top1 (每类 1 张参考)。优先用 v3.eval。"""
    try:
        from v3.eval import metrics as M  # eval agent 的实现
        return M.quick_metrics(feat, labels, scale=sim.logit_scale.exp().item(), bias=sim.logit_bias.item())
    except Exception:
        pass
    from sklearn.metrics import average_precision_score, precision_recall_curve
    z = torch.nn.functional.normalize(feat, dim=-1)
    s = (sim.logit_scale.exp() * z @ z.t() + sim.logit_bias).float()
    y = torch.as_tensor(labels, device=z.device)
    same = y[:, None] == y[None]
    iu = torch.triu_indices(len(z), len(z), 1, device=z.device)
    sc, tg = s[iu[0], iu[1]].cpu().numpy(), same[iu[0], iu[1]].cpu().numpy()
    ap = float(average_precision_score(tg, sc))
    p, r, _ = precision_recall_curve(tg, sc)
    f1 = float(np.max(2 * p * r / np.clip(p + r, 1e-12, None)))
    # zero-shot: 每类第一张做参考
    labels = np.asarray(labels)
    _, first = np.unique(labels, return_index=True)
    ref = z[first]
    rest = np.setdiff1d(np.arange(len(z)), first)
    pred = labels[first][(z[rest] @ ref.t()).argmax(1).cpu().numpy()]
    return dict(pr_auc=ap, best_f1=f1, zs_top1=float((pred == labels[rest]).mean()))


def run_eval(model, sets, sim):
    model.eval()
    res = {}
    for name, ds, labels in sets:
        f = encode(model, ds)
        if is_main():
            m = pair_metrics(f, labels, sim)
            res.update({f'{name}/{k}': v for k, v in m.items()})
    model.train()
    return res


# ---------------- train ----------------
class TrainWrapper(nn.Module):
    def __init__(self, model, loss_fn):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, x, y):
        out = self.model(x)
        return self.loss_fn(out['embedding'], y, self.model.sim)


def lr_factor(step, warmup, total, min_ratio):
    if step < warmup:
        return (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


def build_optimizer(model, loss_fn, oc, backbone_trainable):
    groups = []
    if backbone_trainable:
        groups += model.backbone.param_groups(oc['lr_backbone'], oc.get('layer_decay', 1.0), oc.get('weight_decay', 0.05))
    head_params = [p for p in list(model.head.parameters()) + list(loss_fn.parameters()) if p.requires_grad]
    groups.append(dict(params=head_params, lr=oc['lr_head'], lr_scale=oc['lr_head'] / oc['lr_backbone'],
                       weight_decay=oc.get('weight_decay', 0.05), name='head'))
    sim = [p for p in model.sim.parameters() if p.requires_grad]
    if sim:
        groups.append(dict(params=sim, lr=oc['lr_head'], lr_scale=oc['lr_head'] / oc['lr_backbone'],
                           weight_decay=0.0, name='sim'))
    return torch.optim.AdamW(groups, lr=oc['lr_backbone'], betas=tuple(oc.get('betas', (0.9, 0.999))), fused=True)


def save_ckpt(path, model, loss_fn, cfg, step):
    if not is_main():
        return
    # 冻结的骨干参数不存 (可从 AnimeDINOv3 原权重恢复), 只存可训练部分 + 头
    params = dict(model.named_parameters())
    state = {'model': {k: v for k, v in model.state_dict().items()
                       if not k.startswith('backbone.') or (k in params and params[k].requires_grad)},
             'loss': loss_fn.state_dict(), 'cfg': cfg, 'step': step}
    tmp = str(path) + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('overrides', nargs='*')
    args = ap.parse_args()
    cfg = load_cfg(args.config, args.overrides)

    if 'RANK' in os.environ:
        dist.init_process_group('nccl')
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    world = dist.get_world_size() if dist.is_initialized() else 1
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(cfg.get('seed', 0) + (dist.get_rank() if dist.is_initialized() else 0))

    exp = Path(cfg['exp_dir'])
    if is_main():
        exp.mkdir(parents=True, exist_ok=True)
        (exp / 'cfg.json').write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    dc, mc, lc, oc, ec = cfg['data'], cfg['model'], cfg['loss'], cfg['optim'], cfg.get('eval', {})

    df, num_classes, group = load_train_df(dc)
    counts = np.bincount(df['label'].values, minlength=num_classes)
    global_bs = dc['P'] * dc['K']
    total_steps = int(oc['total_samples'] // global_bs)
    lp_steps = int(oc.get('lp_samples', 0) // global_bs)
    log(f'train images {len(df)} classes {num_classes} global_bs {global_bs} steps {total_steps} (lp {lp_steps})')

    backbone = AnimeDINOBackbone(mc['backbone'], patch_drop=mc.get('patch_drop', 0.0))
    backbone.setup_finetune(mc['finetune'], n_last=mc.get('n_last', 6), blocks=mc.get('blocks', ()),
                            lora_rank=mc.get('lora_rank', 32))
    head = QueryPoolHead(backbone.embed_dim, mc.get('out_dim', 768), pool=mc.get('pool', 'query'),
                         num_queries=mc.get('num_queries', 16))
    model = CxIPModel(backbone, head, mc.get('matryoshka', (768, 512, 256, 128))).cuda()
    margins = dynamic_margins(counts, lc.get('m_min', 0.05), lc.get('m_max', 0.5)) if lc.get('dynamic_margin', True) else None
    loss_fn = MatryoshkaLoss(model.matryoshka, num_classes, lc.get('arc_weight', 1.0), lc.get('siglip_weight', 1.0),
                             margins=margins, k=lc.get('subcenters', 3), scale=lc.get('scale', 30.0),
                             siglip_gather=lc.get('siglip_gather', True), imprint=lc.get('imprint', True)).cuda()
    if lc.get('siglip_weight', 1.0) == 0:
        model.sim.requires_grad_(False)
    if mc.get('compile'):
        backbone.compile_blocks()

    bb_trainable = [p for p in backbone.parameters() if p.requires_grad]
    bb_first = backbone.first_trainable
    wrapper = TrainWrapper(model, loss_fn)

    def make_ddp():
        if world == 1:
            return wrapper
        return DDP(wrapper, device_ids=[torch.cuda.current_device()], broadcast_buffers=False, gradient_as_bucket_view=True)

    def set_phase(lp):
        for p in bb_trainable:
            p.requires_grad_(not lp)
        backbone.first_trainable = backbone.depth if lp else bb_first
        if not lp:
            backbone.set_grad_checkpointing(mc.get('grad_ckpt', False))
        return make_ddp(), build_optimizer(model, loss_fn, oc, backbone_trainable=not lp)

    in_lp = lp_steps > 0
    ddp, opt = set_phase(in_lp)
    eval_sets = load_eval_sets(ec, dc['res'])
    loader = make_train_loader(dc, df, group, cfg.get('seed', 0), total_steps)
    log_f = open(exp / 'train.jsonl', 'a') if is_main() else None
    eval_every = int(ec.get('every_samples', 100_000) // global_bs) or 1
    pd_off_step = int(total_steps * (1 - mc.get('patch_drop_off_frac', 0.1)))

    model.train()
    t0, seen = time.time(), 0
    for step, (x, y) in enumerate(loader):
        if step >= total_steps:
            break
        if in_lp and step == lp_steps:
            in_lp = False
            del ddp, opt  # 释放旧 DDP 的 reducer hooks
            ddp, opt = set_phase(False)
            log(f'step {step}: LP done, unfreeze backbone ({mc["finetune"]})')
        if step == pd_off_step and backbone.patch_drop_prob > 0:
            backbone.set_patch_drop(0.0)  # FLIP: 末段不丢 token, 消除训练/推理的分布差
            if mc.get('ckpt_when_no_drop', True):  # token 翻倍, 激活显存翻倍: 换成 checkpointing
                backbone.set_grad_checkpointing(True)
            log(f'step {step}: patch drop off')
        f = lr_factor(step - (0 if in_lp else lp_steps), oc.get('warmup_steps', 100),
                      lp_steps if in_lp else total_steps - lp_steps, oc.get('min_lr_ratio', 0.01))
        for g in opt.param_groups:
            g['lr'] = oc['lr_backbone'] * g['lr_scale'] * f
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, logs = ddp(x, y)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']], oc.get('grad_clip', 1.0))
        opt.step()
        opt.zero_grad(set_to_none=True)
        seen += global_bs
        if step % cfg.get('log_every', 20) == 0 and is_main():
            dt = time.time() - t0
            rec = dict(step=step, samples=seen, loss=loss.item(), grad_norm=gn.item(), lr_factor=f, img_s=seen / dt,
                       scale=model.sim.logit_scale.exp().item(), bias=model.sim.logit_bias.item(),
                       **{k: v.item() for k, v in logs.items()})
            log_f.write(json.dumps(rec) + '\n'); log_f.flush()
            log(f'step {step}/{total_steps} loss {rec["loss"]:.4f} ' +
                ' '.join(f'{k} {v.item():.4f}' for k, v in logs.items()) + f' {rec["img_s"]:.0f} img/s')
        if (step + 1) % eval_every == 0 or step + 1 == total_steps:
            res = run_eval(model, eval_sets, model.sim)
            if is_main() and res:
                rec = dict(step=step + 1, samples=seen, **res)
                (exp / 'eval.jsonl').open('a').write(json.dumps(rec) + '\n')
                log('eval', json.dumps(res))
        if cfg.get('save_every_samples') and (step + 1) % int(cfg['save_every_samples'] // global_bs) == 0 \
                and step + 1 < total_steps:
            save_ckpt(exp / 'last.pt', model, loss_fn, cfg, step + 1)
    if cfg.get('save_final', True):
        save_ckpt(exp / 'last.pt', model, loss_fn, cfg, total_steps)
    if is_main() and cfg.get('analysis', True):  # C′: 逐层权重变化 + 固定 512 张评测图的逐层特征 (离线 CKA)
        from v3.analysis import save_analysis
        imgs = None
        if eval_sets:
            ds = eval_sets[0][1]
            imgs = torch.stack([ds[i]['image'] for i in range(min(512, len(ds)))])
        wc = save_analysis(exp, backbone, mc['backbone'], imgs)
        log('weight change total', round(wc['_total'], 6))
    log('done')
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
