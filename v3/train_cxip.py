"""CxIP teacher 训练 (CCIP / CSIP 共用): AnimeDINOv3 2B + 头 + SubCenterArcFace/SigLIP, P×K 采样, LP-FT。

CCIP: QueryPoolHead, CCIPTrainTransform, 作品成组难负例。
CSIP: StyleStatsHead (多层 patch 统计), CSIPTrainTransform (原尺度裁剪), "同角色不同画师"交叉分组。

torchrun --nproc_per_node 4 -m v3.train_cxip --config v3/configs/csip_p2_base.py [key=value ...]
"""
import argparse
import ast
import contextlib
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
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from v3.losses import MatryoshkaLoss, dynamic_margins
from v3.models.backbone import AnimeDINOBackbone
from v3.models.heads import CxIPModel, QueryPoolHead
from v3.models.style_head import StyleStatsHead
from v3.transforms import CCIPTrainTransform, CSIPTrainTransform, EvalTransform, NativeCropEval

TRAIN_TRANSFORMS = dict(ccip=CCIPTrainTransform, csip=CSIPTrainTransform)
EVAL_TRANSFORMS = dict(pad_resize=EvalTransform, native_crop=NativeCropEval)


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


def train_collate(batch):
    x, y = collate(batch)
    return x, y, np.array([b['index'] for b in batch])  # 数据集下标: 去偏时按它查每样本的概念标签


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
    ids = dict(group=_ids(df, dc.get('group_key')), cross=_ids(df, dc.get('cross_key')),
               avoid=_ids(df, dc.get('avoid_key')), spread=_ids(df, dc.get('spread_key')))
    return df, len(classes), ids


def _ids(df, key):
    """列 -> 每样本整数 id (< 0 表示无): 整数列直接用; 列表列取唯一元素 (多个/空 = -1); 其他列 factorize。"""
    if not key or key not in df.columns:
        return None
    from v3.data.pk_sampler import encode_ids, single_value_ids
    col = df[key]
    if pd.api.types.is_integer_dtype(col.dtype):  # 含 pandas 可空整数 (Int64, 例: danbooru 的 year 有缺失)
        return col.fillna(-1).to_numpy(dtype=np.int64)
    if pd.api.types.is_float_dtype(col.dtype) and np.all(np.mod(col.dropna().values, 1) == 0):
        return col.fillna(-1).to_numpy(dtype=np.int64)  # 多个索引拼接时某些索引缺这列 (例: CCIP 合并集没有 artist_id)
    first = col.dropna().iloc[0] if col.notna().any() else None
    if isinstance(first, (list, tuple, np.ndarray)):
        return single_value_ids(col.values)
    return encode_ids(col.values)


def make_train_loader(dc, df, ids, seed, steps, start_batch=0):
    from v3.data.indexed_tar import IndexedTarDataset
    from v3.data.pk_sampler import PKBatchSampler
    rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
    tf = TRAIN_TRANSFORMS[dc.get('transform', 'ccip')](dc['res'], **dc.get('aug', {}))
    ds = IndexedTarDataset(dc['root'], df, transform=tf)
    sampler = PKBatchSampler(df['label'].values, dc['P'], dc['K'], alpha=dc.get('alpha', 0.5),
                             group_ids=ids['group'], group_frac=dc.get('group_frac', 0.25) if ids['group'] is not None else 0.0,
                             cross_ids=ids['cross'], cross_frac=dc.get('cross_frac', 0.0) if ids['cross'] is not None else 0.0,
                             cross_size=dc.get('cross_size', 4), avoid_same=ids['avoid'],
                             rank=rank, world_size=world, seed=seed, num_batches=steps, start_batch=start_batch,
                             spread=ids['spread'])
    return DataLoader(ds, batch_sampler=sampler, num_workers=dc.get('workers', 16), collate_fn=train_collate,
                      pin_memory=True, persistent_workers=True, prefetch_factor=4)


def load_eval_sets(ec, res, eval_tf='pad_resize'):
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
        tf = EVAL_TRANSFORMS[s.get('transform', eval_tf)](res)
        sets.append((s['name'], IndexedTarDataset(s['root'], df, transform=tf), df['label'].values))
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
    def __init__(self, model, loss_fn, debiaser=None):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.debiaser = debiaser  # v3.debias.Debiaser: 擦除器在 DDP 包装内, 梯度随主干一起 all-reduce

    def forward(self, x, y, targets=None):
        out = self.model(x)
        loss, logs = self.loss_fn(out['embedding'], y, self.model.sim)
        if self.debiaser is not None:
            l_db, db_logs = self.debiaser(out['embedding'], targets)
            loss = loss + l_db
            logs.update(db_logs)
        with torch.no_grad():  # 坍缩监控: 本卡 batch 内不同类样本对的平均 cos
            z = F.normalize(out['embedding'].detach().float(), dim=-1)
            neg = y[:, None] != y[None]
            logs['emb_cos'] = (z @ z.t())[neg].mean()
        return loss, logs


def lr_factor(step, warmup, total, min_ratio):
    if step < warmup:
        return (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


def build_optimizer(model, loss_fn, oc, backbone_trainable, debiaser=None):
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
    if debiaser is not None:  # 擦除器要跟得上编码器: 固定较大 lr (仍随全局日程衰减)
        lr = oc.get('lr_debias', 1e-3)
        groups.append(dict(params=list(debiaser.parameters()), lr=lr, lr_scale=lr / oc['lr_backbone'],
                           weight_decay=0.0, name='debias'))
    return torch.optim.AdamW(groups, lr=oc['lr_backbone'], betas=tuple(oc.get('betas', (0.9, 0.999))), fused=True)


def build_model(mc):
    """按配置构建 backbone + 头 (训练与评测共用)。"""
    style = mc.get('head', 'query') == 'style'
    layers = tuple(mc.get('style_layers', (4, 8, 12, 16, 20, 24)))
    backbone = AnimeDINOBackbone(mc['backbone'], tap_layers=layers if style else (), patch_drop=mc.get('patch_drop', 0.0))
    backbone.setup_finetune(mc['finetune'], n_last=mc.get('n_last', 6), blocks=mc.get('blocks', ()),
                            lora_rank=mc.get('lora_rank', 32))
    if style:
        # 不含最后一层时后面的 block 没有梯度 (DDP 未用参数), 暂不支持
        assert max(layers) == backbone.depth, f'style_layers must include {backbone.depth}'
        head = StyleStatsHead(layers, backbone.embed_dim, mc.get('out_dim', 768), token_dim=mc.get('token_dim', 512),
                              stats=mc.get('style_stats', 'mean_std'), use_cls=mc.get('style_use_cls', False),
                              center=mc.get('style_center', 'bn'))
        backbone.vit.norm.requires_grad_(False)  # 画风头只读中间层 tap, 最终 norm 无梯度 (DDP 会报未用参数)
    else:
        head = QueryPoolHead(backbone.embed_dim, mc.get('out_dim', 768), pool=mc.get('pool', 'query'),
                             num_queries=mc.get('num_queries', 16))
    return CxIPModel(backbone, head, mc.get('matryoshka', (768, 512, 256, 128)))


def trainable_state(model, trainable_keys):
    """冻结的骨干参数不存 (可从 AnimeDINOv3 原权重恢复), 只存可训练部分 + 头。

    trainable_keys: FT 阶段会训练的骨干参数名 (LP 阶段它们暂时 requires_grad=False, 不能按当前状态判断)。"""
    return {k: v for k, v in model.state_dict().items() if not k.startswith('backbone.') or k in trainable_keys}


def _atomic_save(state, path):
    tmp = str(path) + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)


def save_ckpt(path, model, loss_fn, cfg, step, trainable_keys):
    if not is_main():
        return
    sd = trainable_state(model, trainable_keys)
    if cfg.get('save_dtype', 'fp32') == 'bf16':  # 消融用: 评测本来就把模型转成 bf16, 结果不变, 体积减半
        sd = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in sd.items()}
    _atomic_save({'model': sd, 'loss': loss_fn.state_dict(), 'cfg': cfg, 'step': step}, path)


def save_resume(path, model, loss_fn, opt, ema, cfg, step, trainable_keys):
    """断点续训状态 (fp32 可训练参数 + 优化器 + EMA 影子), 训练正常结束后删除。"""
    if not is_main():
        return
    _atomic_save({'model': trainable_state(model, trainable_keys), 'loss': loss_fn.state_dict(), 'opt': opt.state_dict(),
                  'ema': None if ema is None else ema.shadow, 'ema_updates': 0 if ema is None else ema.updates,
                  'cfg': cfg, 'step': step}, path)


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

    df, num_classes, ids = load_train_df(dc)
    counts = np.bincount(df['label'].values, minlength=num_classes)
    global_bs = dc['P'] * dc['K']
    total_steps = int(oc['total_samples'] // global_bs)
    lp_steps = int(oc.get('lp_samples', 0) // global_bs)
    log(f'train images {len(df)} classes {num_classes} global_bs {global_bs} steps {total_steps} (lp {lp_steps})')

    model = build_model(mc).cuda()
    backbone = model.backbone
    margins = dynamic_margins(counts, lc.get('m_min', 0.05), lc.get('m_max', 0.5)) if lc.get('dynamic_margin', True) else None
    loss_fn = MatryoshkaLoss(model.matryoshka, num_classes, lc.get('arc_weight', 1.0), lc.get('siglip_weight', 1.0),
                             margins=margins, k=lc.get('subcenters', 3), scale=lc.get('scale', 30.0),
                             siglip_gather=lc.get('siglip_gather', True), imprint=lc.get('imprint', True),
                             arc_negatives=lc.get('arc_negatives', 'all')).cuda()
    if lc.get('siglip_weight', 1.0) == 0:
        model.sim.requires_grad_(False)
    if cfg.get('init_ckpt'):  # 从已训练的 teacher 继续 (去偏微调等): 只加载权重, 优化器/步数从头开始
        st = torch.load(cfg['init_ckpt'], map_location='cpu', weights_only=False)
        missing, unexpected = model.load_state_dict({k: v.float() for k, v in st['model'].items()}, strict=False)
        assert not unexpected, unexpected[:5]
        assert not [k for k in missing if not k.startswith('backbone.')], missing[:5]
        loss_fn.load_state_dict(st['loss'])
        log(f'init from {cfg["init_ckpt"]} (step {st.get("step")}, {len(missing)} frozen backbone keys from pretrain)')
        del st
    debiaser = concepts = None
    db = cfg.get('debias') or {}
    if db.get('tag_weight', 0) > 0 or db.get('char_weight', 0) > 0:
        from v3.debias import ConceptTargets, Debiaser
        concepts = ConceptTargets(df, db.get('char_min_count', 20), use_tags=db.get('tag_weight', 0) > 0,
                                  use_chars=db.get('char_weight', 0) > 0, conditional=db.get('conditional', False),
                                  smooth=db.get('smooth', 2.0))
        debiaser = Debiaser(mc.get('out_dim', 768), concepts, db.get('tag_weight', 0), db.get('char_weight', 0),
                            hidden=db.get('hidden', 1024)).cuda()
        log(f'debias tags {len(concepts.tag_vocab) if concepts.tags else 0} '
            f'chars {len(concepts.char_names) if concepts.chars else 0} weights {db}')
    if mc.get('compile'):
        backbone.compile_blocks()

    bb_trainable = [p for p in backbone.parameters() if p.requires_grad]
    trainable_keys = {f'backbone.{n}' for n, p in backbone.named_parameters() if p.requires_grad}
    bb_first = backbone.first_trainable
    wrapper = TrainWrapper(model, loss_fn, debiaser)

    def make_ddp():
        if world == 1:
            return wrapper
        # 有 BatchNorm 时广播 rank0 的 buffer, 保证各卡评测用同一套 running stats
        has_bn = any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules())
        return DDP(wrapper, device_ids=[torch.cuda.current_device()], broadcast_buffers=has_bn, gradient_as_bucket_view=True)

    def set_phase(lp):
        for p in bb_trainable:
            p.requires_grad_(not lp)
        backbone.first_trainable = backbone.depth if lp else bb_first
        if not lp:
            backbone.set_grad_checkpointing(mc.get('grad_ckpt', False))
        return make_ddp(), build_optimizer(model, loss_fn, oc, backbone_trainable=not lp, debiaser=debiaser)

    in_lp = lp_steps > 0
    ddp, opt = set_phase(in_lp)
    # 分辨率日程: data.res_schedule=[[frac, res], ...], 在 frac·total_steps 处换分辨率 (batch 结构不变, 从同一 batch 序号续上)
    res_phases = [(0, dc['res'])] + [(int(total_steps * f), int(r)) for f, r in dc.get('res_schedule', [])]
    eval_sets = load_eval_sets(ec, ec.get('res', res_phases[-1][1]), dc.get('eval_transform', 'pad_resize'))

    def batch_iter(start_step=0):
        for k, (start, res) in enumerate(res_phases):
            end = res_phases[k + 1][0] if k + 1 < len(res_phases) else total_steps
            start = max(start, start_step)  # 断点续训: 跳过已训练的 batch (batch 序列由 seed/epoch/序号决定)
            if end <= start:
                continue
            if k > 0:
                log(f'step {start}: res -> {res}')
                if res > res_phases[k - 1][1]:
                    backbone.set_grad_checkpointing(True)  # token 变多, 换成全量 checkpointing
            it = iter(make_train_loader(dict(dc, res=res), df, ids, cfg.get('seed', 0), total_steps, start_batch=start))
            for b in range(start, end):
                yield b, next(it)
            del it

    ema = None
    if oc.get('ema'):  # 跟踪 FT 阶段会训练的全部参数 (头 + sim + 可训练骨干)
        from v3.ema import ModelEMA
        ema = ModelEMA(list(model.head.parameters()) + list(model.sim.parameters()) + bb_trainable, oc['ema'])
    ema_ctx = (lambda: ema.swapped()) if ema else contextlib.nullcontext

    # 断点续训: 设置 resume_every_samples 后定期写 exp/resume.pt, 重新启动同一实验时自动从中恢复
    resume_every = int(cfg['resume_every_samples'] // global_bs) if cfg.get('resume_every_samples') else 0
    resume_path, start_step = exp / 'resume.pt', 0
    if resume_every and resume_path.exists():
        st = torch.load(resume_path, map_location='cpu', weights_only=False)
        _, unexpected = model.load_state_dict(st['model'], strict=False)
        assert not unexpected, unexpected[:5]
        loss_fn.load_state_dict(st['loss'])
        start_step = st['step']
        if in_lp and start_step >= lp_steps:
            in_lp = False
            del ddp, opt
            ddp, opt = set_phase(False)
        opt.load_state_dict(st['opt'])
        if ema is not None:
            ema.shadow, ema.updates = [t.cuda() for t in st['ema']], st['ema_updates']
        del st
        log(f'resumed from step {start_step}')
    log_f = open(exp / 'train.jsonl', 'a') if is_main() else None
    eval_every = int(ec.get('every_samples', 100_000) // global_bs) or 1
    pd_off_step = int(total_steps * (1 - mc.get('patch_drop_off_frac', 0.1)))

    model.train()
    t0, seen = time.time(), 0
    for step, (x, y, idx) in batch_iter(start_step):
        if in_lp and step == lp_steps:
            in_lp = False
            del ddp, opt  # 释放旧 DDP 的 reducer hooks
            ddp, opt = set_phase(False)
            log(f'step {step}: LP done, unfreeze backbone ({mc["finetune"]})')
        if step >= pd_off_step and backbone.patch_drop_prob > 0:  # >=: 续训起点可能已过切换点
            backbone.set_patch_drop(0.0)  # FLIP: 末段不丢 token, 消除训练/推理的分布差
            if mc.get('ckpt_when_no_drop', True):  # token 翻倍, 激活显存翻倍: 换成 checkpointing
                backbone.set_grad_checkpointing(True)
            log(f'step {step}: patch drop off')
        f = lr_factor(step - (0 if in_lp else lp_steps), oc.get('warmup_steps', 100),
                      lp_steps if in_lp else total_steps - lp_steps, oc.get('min_lr_ratio', 0.01))
        for g in opt.param_groups:
            g['lr'] = oc['lr_backbone'] * g['lr_scale'] * f
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        targets = None
        if concepts is not None:
            targets = concepts.batch(idx, x.device)
            debiaser.ramp = min(1.0, step / max(1, db.get('ramp_steps', 200)))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, logs = ddp(x, y, targets)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']], oc.get('grad_clip', 1.0))
        opt.step()
        opt.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update()
        seen += global_bs
        if step % cfg.get('log_every', 20) == 0 and is_main():
            dt = time.time() - t0
            rec = dict(step=step, samples=(step + 1) * global_bs, loss=loss.item(), grad_norm=gn.item(), lr_factor=f,
                       img_s=seen / dt,
                       scale=model.sim.logit_scale.exp().item(), bias=model.sim.logit_bias.item(),
                       **{k: v.item() for k, v in logs.items()})
            log_f.write(json.dumps(rec) + '\n'); log_f.flush()
            log(f'step {step}/{total_steps} loss {rec["loss"]:.4f} ' +
                ' '.join(f'{k} {v.item():.4f}' for k, v in logs.items()) + f' {rec["img_s"]:.0f} img/s')
        if (step + 1) % eval_every == 0 or step + 1 == total_steps:
            with ema_ctx():
                res = run_eval(model, eval_sets, model.sim)
            if is_main() and res:
                rec = dict(step=step + 1, samples=(step + 1) * global_bs, **res)
                (exp / 'eval.jsonl').open('a').write(json.dumps(rec) + '\n')
                log('eval', json.dumps(res))
        if cfg.get('save_every_samples') and (step + 1) % int(cfg['save_every_samples'] // global_bs) == 0 \
                and step + 1 < total_steps:
            with ema_ctx():
                save_ckpt(exp / 'last.pt', model, loss_fn, cfg, step + 1, trainable_keys)
        if resume_every and (step + 1) % resume_every == 0 and step + 1 < total_steps:
            save_resume(resume_path, model, loss_fn, opt, ema, cfg, step + 1, trainable_keys)
            log(f'step {step + 1}: resume state saved')
    if cfg.get('save_final', True) and ema is not None:
        save_ckpt(exp / 'last_raw.pt', model, loss_fn, cfg, total_steps, trainable_keys)  # 非 EMA 权重
    with ema_ctx():  # 最终存档与 C′ 分析都用 EMA 权重
        if cfg.get('save_final', True):
            save_ckpt(exp / 'last.pt', model, loss_fn, cfg, total_steps, trainable_keys)
        if is_main() and cfg.get('analysis', True):  # C′: 逐层权重变化 + 固定 512 张评测图的逐层特征 (离线 CKA)
            from v3.analysis import save_analysis
            imgs = None
            if eval_sets:
                ds = eval_sets[0][1]
                imgs = torch.stack([ds[i]['image'] for i in range(min(512, len(ds)))])
            wc = save_analysis(exp, backbone, mc['backbone'], imgs)
            log('weight change total', round(wc['_total'], 6))
    if is_main() and resume_path.exists():
        resume_path.unlink()  # 已正常结束, 续训状态 (数十 GB) 不再需要
    log('done')
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
