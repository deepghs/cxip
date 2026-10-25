"""CCIP v3 teacher 最终训练。

配方 (P1 消融结论, 见 docs/cxip_v3_results_zh.md):
- 只训最后 8 层 (l8b 与 12 层持平, 省 1/3 反传); ArcFace 只用 batch 内负类; 去掉 LP (l12b_no_lp +0.007);
  QueryPool 16 queries; 保留同作品成组 (nogroup −0.005)。
- 伪画师交叉成组 0.25 (l12b_pa: FULL +0.007, I_style 0.140 → 0.134): 让 "同画风不同角色" 成为难负例, 抑制靠画风认角色。
- 全程 384 训练, 512 推理/评测 (l12b_seed1: 384 训练 + 512 评测 +0.015; 训练末段 512 无额外收益)。
- 预算 3M 样本 (每 4 倍约 +0.027, 1.2M 时仍在升); 不开 EMA (CSIP final 里无收益)。
"""
import copy
import runpy

CFG = copy.deepcopy(runpy.run_path('/data/neko/cxip/v3/configs/ccip_p1_base.py')['CFG'])
CFG.update(
    exp_dir='/data/neko/cxip_v3/exps/final/ccip_teacher_v1',
    save_dtype='fp32',
    resume_every_samples=500_000,
)
CFG['data'].update(index='/data/neko/cxip_v3/data/ccip-pseudo-artist-v1/train.parquet',
                   cross_key='pseudo_artist', cross_frac=0.25)
CFG['model'].update(finetune='last_n', n_last=8)
CFG['loss'].update(arc_negatives='batch')
CFG['optim'].update(total_samples=3_000_000, lp_samples=0, warmup_steps=100)
CFG['eval'].update(every_samples=500_000, res=512)
