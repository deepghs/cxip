"""CSIP v3 teacher 最终训练。

配方 = q_x4 (P2 最优): query 头 (16 queries) + 只用 SigLIP (ArcFace 在 4 倍预算下仍 −0.12 且内容偏置升高) +
Matryoshka + 同角色交叉成组; 全量微调 + LLRD 0.9 (冻结前 12 层 −0.014, 见 q_ft_last12)。
- 预算是主杠杆 (每倍约 +0.05, 1.2M 时仍在上升): 6M 样本 ≈ 1.5 epoch。
- 分辨率: 512 训练在 native 评测上 +0.03 / +0.06, 但单价 2.1 倍; 同算力下 384 多训更划算 (按预算斜率折算),
  所以只在末 15% 换 512 (FixRes: 让训练分辨率接近整图推理), 末 5% 关 patch drop。
- 评测协议: native_full (短边 ≤640 整图); 训练内 t100 固定 384 中心裁剪, 与 P2 轨迹可比。
"""
import copy
import runpy

CFG = copy.deepcopy(runpy.run_path('/data/neko/cxip/v3/configs/csip_p2_base.py')['CFG'])
CFG.update(
    exp_dir='/data/neko/cxip_v3/exps/final/csip_teacher_v1',
    save_dtype='fp32',
    resume_every_samples=500_000,  # 约 20-40 分钟一次; 续训状态约 32G, 正常结束后删除
)
CFG['data'].update(res=384, res_schedule=[[0.85, 512]])
CFG['model'].update(head='query', pool='query', num_queries=16, patch_drop_off_frac=0.05)
CFG['loss'].update(arc_weight=0.0, siglip_weight=1.0)
CFG['optim'].update(total_samples=6_000_000, lp_samples=30_000, warmup_steps=100, ema=0.999)
CFG['eval'].update(every_samples=500_000, res=384)
