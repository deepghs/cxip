"""CSIP P2 消融基线: 全量 + 温和 LLRD 0.9 + 多层统计画风头 + SubCenterArcFace/SigLIP + Matryoshka。

数据: danbooru-artist-v1 train (4,057,782 张 / 38,761 画师, 已剔除 0918 测试画师和 csip_eval 画师)。
采样: P×K 画师, 同一画师的 K 张尽量来自不同角色 (avoid_key); 25% 的类按"同角色不同画师"交叉成组 (cross_key),
      让模型只能靠画风区分 (数据层去偏, 真实标签版 CharBucket)。
增强: 原尺度裁剪 + 轻微缩放 + 翻转, 不动颜色/清晰度。固定预算 300k 样本 (LP 30k), 384, 全局 batch 512。
"""
MODEL = '/data/neko/cxip_v3/models/AnimeDINOv3-timm'
DANBOORU = '/data/neko/cxip_v3/data/danbooru-artist-v1'

CFG = dict(
    exp_dir='/data/neko/cxip_v3/exps/csip_p2/base',
    seed=0,
    log_every=20,
    save_every_samples=None,
    save_final=True,
    save_dtype='bf16',  # 消融存档用 bf16 (评测本来就转 bf16, 结果不变; 磁盘只剩 ~320G)
    analysis=True,
    data=dict(
        root=DANBOORU,
        index=f'{DANBOORU}/index/train.parquet',
        exclude=None,
        transform='csip', eval_transform='native_crop',
        res=384, P=128, K=4, alpha=0.5,
        group_key=None,
        cross_key='characters', cross_frac=0.25, cross_size=4,  # 单角色图的角色 id; 多角色/无角色 = -1
        avoid_key='characters',
        workers=16,
        aug=dict(zoom=(0.8, 1.25)),
    ),
    model=dict(
        backbone=MODEL,
        finetune='full', n_last=6, lora_rank=32,
        patch_drop=0.5, patch_drop_off_frac=0.1,
        compile=True, grad_ckpt=10,
        head='style', style_layers=(4, 8, 12, 16, 20, 24), style_stats='mean_std', token_dim=512,
        out_dim=768, matryoshka=(768, 512, 256, 128),
    ),
    loss=dict(
        arc_weight=1.0, siglip_weight=1.0,
        subcenters=3, scale=30.0,
        dynamic_margin=True, m_min=0.05, m_max=0.5,
        imprint=True,
        arc_negatives='batch',  # 38,761 类每类 LP 内至多访问约一次: batch 外的旧中心会诱发坍缩, 见 losses.MatryoshkaLoss
    ),
    optim=dict(
        lr_backbone=2e-5, lr_head=2e-4, layer_decay=0.9, weight_decay=0.05,
        warmup_steps=50, min_lr_ratio=0.01, grad_clip=1.0,
        total_samples=300_000, lp_samples=30_000,
    ),
    eval=dict(
        every_samples=100_000,
        sets=[
            dict(name='t100', root=DANBOORU, index=f'{DANBOORU}/index/test100_0918.parquet', max_per_class=30),
        ],
    ),
)
