# CCIP / CSIP v2.1 复现 (H200, /data/neko/cxip_repro)

目标: 用恢复出来的训练代码重新训练 CCIP (dghstmp/AnimeCCIP) 和 CSIP v2.1 (dghstmp/AnimeCSIP-v2.1)，
和 HF 上的模型在同一套评测上对比。

## 目录

```
/data/neko/cxip_repro/
  CxIP/                 本仓库, 分支 repro
  RainbowNekoEngine/    分支 cxip-repro (trainer_ac.py: 跳过优化器步时不推进 lr/wd scheduler)
  env.sh                conda env cxip + HF/KohakuHub token (chmod 600, 不进 git)
  data/                 数据 (见下)
  exps/                 训练输出
  results/              评测 json (pair_eval.py)
  logs/                 训练/特征提取日志
```

## 数据来源

| 用途 | 原路径 | 复现用的数据 | 一致性 |
|---|---|---|---|
| CCIP 训练 | /nfs/ccip/ccip_v1_flat | HF deepghs/character_similarity images_v1 解压后，按 hub.deepghs.org deepghs/ccip (原 /nfs/ccip 的 tar 备份) 的文件列表对齐 (`data/align_ccip_orig.py`)：原始集合是 images_v1 的子集，多出的 51,317 张挪到 ccip_v1_extra/ | **完全一致**: 241,483 张 / 3,982 角色 |
| CCIP 训练中评测 | /nfs/ccip/ccip_eval/eval | deepghs/ccip 备份里的 ccip_eval/eval | 完全一致 (993 张, 10 角色) |
| CCIP 测试 | /nfs/ccip/test_ccip | deepghs/ccip 备份里的 test_ccip | 完全一致 (328 张) |
| CCIP 测试 (卡片) | deepghs/dinov2_ccip_eval | 不可下载 (deepghs 私有库超出 HF 存储配额, 403)；用 zerochan-character-w640-ws-m50-full test 固定抽样 zc50x15 (750 张, 规模同卡片) / zc500x10 | 不同集合 |
| CSIP 训练 | /nfs/danbooru-w640-ws-full-artist | hub.deepghs.org 上同名仓库只有 README (备份没传上去)。按原流程重建 (`data/build_danbooru_artist.py`)：animetimm/danbooru-wdtagger-v4-w640-ws-full 图片 + isek-ai/danbooru-tags-2024 画师 (覆盖 99%)，单画师、画师 >= 30 张，split_labels.py 切 800 个测试画师 (seed 42)，csip_sub 取 100 个 -> test_split_100.json | 规模接近 (4.27M 张 / 39.6k 画师; test100 9,509 张 / 875k 正样本对，卡片 10,360 张 / 1.67M)，**测试画师不同** |
| CSIP 测试 (公开) | deepghs/csip_eval | 同 | 一致，但不是卡片用的集合 |
| 初始化 | /data/anime_dino/.../model-191000.safetensors | HF dghstmp/anime_dino 同名文件 (键带 `base:` 前缀，引擎会自动拆开，原 state_prefix 可直接用，已验证 387/387 个参数被加载) | 一致 |

## 与原配置的差异

- 只改了路径 (`repro/cfgs/*_repro.py` 由 `cfgs/py/train/contrastive/*.py` 机械替换生成，diff 只有路径/exp_dir/gradient_accumulation_steps)
- 原 launcher `cfgs/launcher/multi.yaml` 是 8 卡；这里 4 卡 (4-7 号) x bs128 x grad_accum 2，全局 batch 1024、步数与 8 卡相同。
  SigLIP loss 不跨卡 gather，每卡 batch 128 与原来一样
- CSIP v2.1 的角色伪标签: 原流程 ccip_save.py 每张图存一个 .pt，这里改成 `data/extract_ccip_feats.py` 分片 npy，
  `tools/ccip_cls.py --feat-npy` 读取 (已验证与逐文件 .pt 路径得到的矩阵/文件顺序完全一致)，聚类参数不变

## 注意

- **不要在 H200 上大量写小文件**: 2026-10-06 解压 427 万张图片后，共享盘回写卡死 (kernel hung task)，机器被强制重启，
  其他人 0-3 号卡上的任务被杀。之后的步骤都避免逐图写文件
- 训练中 `CXIPMetricContainer` 的 AP/ROC 用 `max(pred * same_mask)` 取最难正/负样本，logit 全为负时 (bias 初始 -10) 正负都取到 0，
  会恒等于 0.5 / cohens 0，看 loss 判断是否在学 (常数预测的 SigLIP loss 下界约 0.562)
- HF CSIP v2/v2.1 是在原 danbooru-w640-ws-full-artist 上训练的，我们重新随机切的 test100 画师大概率在它们的训练集里，
  所以 HF v2/v2.1 在 test100 上的分数偏高 (泄漏)，只有我们自己训练的模型在 test100 上是真正的 held-out

## 评测

`eval/pair_eval.py`: 测试集内全部图片两两配对，score = exp(logit_scale)·cos + logit_bias，报告 PR-AUC (卡片指标)、ROC-AUC、
best F1/阈值、zero-shot (每类 1/5 张参考图)。`eval/summarize.py results/*.json` 生成下表。

### HF 模型 (基线)

| model | dataset | images | classes | PR-AUC | ROC-AUC | best F1 | threshold | zs top1 (1 ref) | zs top1 (5 ref) |
|---|---|---|---|---|---|---|---|---|---|
| dghstmp/AnimeCCIP | ccip_eval | 993 | 10 | 1.0000 | 1.0000 | 0.9978 | 0.233 | 1.000 | 1.000 |
| dghstmp/AnimeCCIP | test_ccip | 328 | 42 | 0.9941 | 0.9998 | 0.9661 | 1.439 | 1.000 | - |
| dghstmp/AnimeCCIP | ccip_test_v1 | 338 | 43 | 0.9924 | 0.9998 | 0.9600 | 1.511 | 0.997 | - |
| dghstmp/AnimeCCIP | zc50x15 | 750 | 50 | 0.9863 | 0.9995 | 0.9476 | -0.182 | 0.981 | 0.994 |
| dghstmp/AnimeCCIP | zc500x10 | 5000 | 500 | 0.8794 | 0.9985 | 0.8243 | 2.363 | 0.872 | 0.959 |
| dghstmp/AnimeCSIP | csip_eval_public | 8952 | 122 | 0.7727 | 0.9930 | 0.7064 | 3.668 | 0.823 | 0.950 |
| dghstmp/AnimeCSIP | test100 | 9509 | 100 | 0.5879 | 0.9617 | 0.5576 | 2.168 | 0.578 | 0.803 |
| dghstmp/AnimeCSIP-v2 | csip_eval_public | 8952 | 122 | 0.8414 | 0.9964 | 0.7699 | 4.869 | 0.885 | 0.977 |
| dghstmp/AnimeCSIP-v2 | test100 | 9509 | 100 | 0.8379 | 0.9921 | 0.7584 | 2.663 | 0.745 | 0.933 |
| dghstmp/AnimeCSIP-v2.1 | csip_eval_public | 8952 | 122 | 0.8724 | 0.9968 | 0.7977 | 4.708 | 0.897 | 0.976 |
| dghstmp/AnimeCSIP-v2.1 | test100 | 9509 | 100 | 0.9246 | 0.9959 | 0.8554 | 2.257 | 0.814 | 0.950 |

卡片上的数值: CCIP PR-AUC 0.997 (阈值 -0.317)；CSIP v1 0.663 / v2 0.815 / v2.1 0.757 (都在各自的原始测试集上)。

### 复现模型

**CCIP** (`exps/ccip_dino/` 原配置 2 epoch = 1,562 步; `exps/ccip_dino_long/` 从 1,562 步继续 13 epoch = 10,153 步, cosine lr 1e-4 -> 0)

| model | dataset | PR-AUC | ROC-AUC | best F1 | threshold | zs top1 (1 ref) | zs top1 (5 ref) |
|---|---|---|---|---|---|---|---|
| HF AnimeCCIP | ccip_eval | 1.0000 | 1.0000 | 0.9978 | 0.233 | 1.000 | 1.000 |
| HF AnimeCCIP | test_ccip | 0.9941 | 0.9998 | 0.9661 | 1.439 | 1.000 | - |
| HF AnimeCCIP | ccip_test_v1 | 0.9924 | 0.9998 | 0.9600 | 1.511 | 0.997 | - |
| HF AnimeCCIP | zc50x15 | 0.9863 | 0.9995 | 0.9476 | -0.182 | 0.981 | 0.994 |
| HF AnimeCCIP | zc500x10 | 0.8794 | 0.9985 | 0.8243 | 2.363 | 0.872 | 0.959 |
| 复现 2 epoch | ccip_eval | 0.9625 | 0.9948 | 0.8906 | 0.112 | 0.982 | 0.987 |
| 复现 2 epoch | test_ccip | 0.8318 | 0.9884 | 0.7739 | 0.097 | 0.846 | - |
| 复现 2 epoch | ccip_test_v1 | 0.8160 | 0.9879 | 0.7593 | 0.107 | 0.837 | - |
| 复现 2 epoch | zc50x15 | 0.7360 | 0.9828 | 0.6954 | 0.099 | 0.757 | 0.926 |
| 复现 2 epoch | zc500x10 | 0.2884 | 0.9847 | 0.3444 | 0.540 | 0.533 | 0.684 |
| 复现 long (10,153 步) | ccip_eval | 0.9999 | 1.0000 | 0.9958 | 1.443 | 1.000 | 1.000 |
| 复现 long (10,153 步) | test_ccip | 0.9908 | 0.9996 | 0.9635 | 1.444 | 0.983 | - |
| 复现 long (10,153 步) | ccip_test_v1 | 0.9780 | 0.9994 | 0.9335 | 1.444 | 0.980 | - |
| 复现 long (10,153 步) | zc50x15 | 0.9189 | 0.9930 | 0.8672 | 0.320 | 0.881 | 0.972 |
| 复现 long (10,153 步) | zc500x10 | 0.6112 | 0.9897 | 0.5948 | 2.802 | 0.658 | 0.799 |

- 原 CCIP 测试集 (test_ccip / ccip_eval) 上基本追平 HF; 换成训练集以外分布的 zerochan 角色 (尤其 500 类) 差距仍大
- long 中间 checkpoint (step 2000) 只有 test_ccip 0.867 / zc500x10 0.273, 提升主要发生在后半段 lr 衰减时
- 学到的 sim 参数: 复现 long scale exp(2.689)=14.7 / bias -9.68, HF 26.3 / -13.9 -> HF 的训练量或数据仍明显多于恢复的配置,
  推测原模型用了比 ccip_v1 (3,982 角色) 更多的角色数据

**CSIP v2.1** (`exps/csip_dino_v2.1_fromv2/`, HF AnimeCSIP-v2 初始化, split v2 + 新聚类的角色伪标签, 1 epoch = 3,894 步): 训练中
