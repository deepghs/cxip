"""CCIP P1 消融基线: 全量微调 + 丢 50% patch (末 10% 关闭) + QueryPool + SubCenterArcFace/SigLIP + Matryoshka。

固定预算 300k 样本 (LP 30k), 384 分辨率, 全局 batch 512 = P128 × K4 (设计值 1024 留给最终训练, 需开 ckpt)。消融只改一个维度。
"""
MODEL = '/data/neko/cxip_v3/models/AnimeDINOv3-timm'
CCIP_ROOT = '/data/neko/cxip_repro/data/ccip_merged_nd'
INDEX = '/data/neko/cxip_v3/data/ccip-merged-v1-index/index'
OPENSET = '/data/neko/cxip_v3/eval/ccip_openset_v1'

CFG = dict(
    exp_dir='/data/neko/cxip_v3/exps/ccip_p1/base',
    seed=0,
    log_every=20,
    save_every_samples=None,
    save_final=True,
    save_dtype='bf16',  # 消融存档用 bf16 (评测本来就转 bf16, 结果不变; 磁盘只剩 ~320G)
    analysis=True,
    data=dict(
        root=CCIP_ROOT,
        index=f'{INDEX}/train.parquet',
        exclude=f'{OPENSET}/train_exclude.json',  # 开放集留出类, 训练前剔除并重映射标签
        group_key='series_id',  # 同作品难负例分组 (-1 = 未知作品; 列缺失时自动关闭)
        res=384, P=128, K=4, alpha=0.5, group_frac=0.25,  # 全局 512, 每卡 128 (丢 patch 后约 110GB)
        workers=16,
        aug=dict(),
    ),
    model=dict(
        backbone=MODEL,
        finetune='full', n_last=6, lora_rank=32,
        patch_drop=0.5, patch_drop_off_frac=0.1,
        compile=True, grad_ckpt=10,  # 每卡 bs128: 只 checkpoint 10/24 个 block (全不开时 >136GB OOM)
        pool='query', num_queries=16, out_dim=768,
        matryoshka=(768, 512, 256, 128),
    ),
    loss=dict(
        arc_weight=1.0, siglip_weight=1.0,
        subcenters=3, scale=30.0,
        dynamic_margin=True, m_min=0.05, m_max=0.5,
    ),
    optim=dict(
        lr_backbone=2e-5, lr_head=2e-4, layer_decay=0.85, weight_decay=0.05,
        warmup_steps=50, min_lr_ratio=0.01, grad_clip=1.0,
        total_samples=300_000, lp_samples=30_000,
    ),
    eval=dict(
        every_samples=100_000,
        sets=[
            dict(name='fast', root=OPENSET, index=f'{OPENSET}/fast.parquet',
                 columns_map={'pack_tar': 'shard', 'pack_offset': 'offset'}),
        ],
    ),
)
