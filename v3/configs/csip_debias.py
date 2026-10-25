"""CSIP 去偏微调: 从 csip_teacher_v1 (非 EMA 权重) 出发, 同配方再训 600k 样本 (约 0.1× 原预算), 只改去偏手段。

对照 dft_ctrl (同预算不去偏) 用来扣掉 "多训一会儿" 本身的效果; 各变体只改一处:
  - dft_cross50:  同角色交叉成组 0.25 → 0.5 (数据层去偏, 无额外损失)
  - dft_tag*:     内容标签混淆擦除 (v3.debias, 标签 = general ∩ 探针 320 标签 − 画风/媒介标签)
  - dft_char*:    角色混淆擦除 (单角色图, ≥20 张的 11k 个角色)
lr 取原来的 0.3 倍、不做 LP、warmup 100 步后余弦; 全程 384 (末 10% 关 patch drop), 不开 EMA (final 里 EMA 无增益)。
"""
import copy
import runpy

CFG = copy.deepcopy(runpy.run_path('/data/neko/cxip/v3/configs/csip_final.py')['CFG'])
CFG.update(
    exp_dir='/data/neko/cxip_v3/exps/csip_p2/dft_ctrl',
    init_ckpt='/data/neko/cxip_v3/exps/final/csip_teacher_v1/last_raw.pt',
    save_dtype='bf16',
    resume_every_samples=None,
    debias=dict(tag_weight=0.0, char_weight=0.0, char_min_count=20, hidden=1024),
)
CFG['data'].update(res=384, res_schedule=[])
CFG['model'].update(patch_drop_off_frac=0.1)
CFG['optim'].update(total_samples=600_000, lp_samples=0, warmup_steps=100, ema=None,
                    lr_backbone=6e-6, lr_head=6e-5, lr_debias=1e-3)
CFG['eval'].update(every_samples=200_000, res=384)
