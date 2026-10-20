# CCIP / CSIP v2.1 复现 (H200, /data/neko/cxip_repro)

目标: 用恢复出来的训练代码重新训练 CCIP (dghstmp/AnimeCCIP) 和 CSIP v2.1 (dghstmp/AnimeCSIP-v2.1)，
和 HF 上的模型在同一套评测上对比。


**状态 (2026-10-20): CCIP 和 CSIP v2.1 视为已恢复。**
- CCIP: HF AnimeCCIP 是用 zerochan 数据 + `cfgs/py/train/contrastive/ccip_wds.py` 训练的 (已确认)。下文 ccip_long 用的是 ccip_v1 (`ccip.py`), 所以 zerochan 测试集上的差距来自训练数据, 不是代码。
- CSIP v2.1: 从 HF v2 微调 1 epoch, test100_0918 PR-AUC 0.7451 (HF 0.7445), csip_eval_public 0.8785 (HF 0.8723), content/char bias 0.086/0.024 (HF 0.091/0.036)。最终配置 `repro/cfgs/fix_csip_0918_g003_full.py`, 流程 `repro/run_fix_full.sh`。

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

**原始画师划分 (2025-09) 已恢复**
- 规则：witha 的 `artist_tags[0]`，witha train→train、test→test，val 不用。
- 指纹：test 182,810 / test100 22,613，与 2025-09-18 对话日志的记录完全一致。
- 脚本 `repro/data/match_split_0918.py`，输出 `data/danbooru-artist-split-0918/`。

HF 模型在原始 test100 上的结果（22,559 张，缺 54 张），数值为 PR-AUC：

| | v1 | v2 | v2.1 |
|-|-|-|-|
| test100_0918 | 0.7455 | 0.7919 | 0.7445 |
| 卡片 | 0.663 | 0.815 | 0.757 |

排序与卡片一致。

注意：下面的 v2.1 微调用的是 split v2（isek，37k 个画师），训练集包含原始 test100 的大部分画师，所以它在 test100_0918 上的分数有泄漏。

**CSIP v2.1 微调**（`exps/csip_dino_v2.1_fromv2_0918/`）
- 初始化：HF AnimeCSIP-v2。
- 数据：原始划分 0918（2,079,184 张），伪标签为 30,658 个角色簇。
- 训练：1 epoch = 2,043 步。
- 之前在 split v2 上的那次运行已停止（训练集与原始测试画师重叠），见下面"划分"一节。

| model | test100_0918 PR-AUC | csip_eval_public PR-AUC |
|-|-|-|
| HF v2（初始化） | 0.7919 | 0.8414 |
| v2 初始化经 trans_csip 导出（对照） | 0.7919 | - |
| 微调 step 1000 | 0.2170 | - |
| 微调 step 2043 | **0.1868** | **0.2966** |

**结论：重建的 debias 训练会破坏 v2 的风格特征。**
- 导出流程没有问题：对照组与 HF v2 完全一致。
- 崩溃发生在前 1000 步。权重变化不大（backbone 相对变化 1～4%），sim 参数几乎不变。
- 训练 loss 的变化：siglip 从 0.008 升到约 0.03；char 从 0.42 降到 0.04；content 约 1.2，没有被擦除。
- 推测原因：重建配置中的 GRL alpha=1、两个 eraser 的 loss 权重=1，加上 content eraser 是 12,129 个标签的 BCE，反转梯度远大于 siglip 的梯度。
- 这些超参都是猜的（原文丢失）。原版 v2.1 显然不是这样训练的：HF v2.1 在 test100_0918 上是 0.7445。

**消融**（结果在 `results/abl_*.json`）：

| 设置 | 步数 | test100_0918 PR-AUC |
|-|-|-|
| v2 初始化 | 0 | 0.7919 |
| 只用 siglip，loss_csip(0, 0) | 300 | 0.7629 |
| debias 权重 0.1 | 300 | 0.6569 |
| debias 权重 1（正式运行） | 1000 / 2043 | 0.2170 / 0.1868 |

结论：
- 数据、标签和 siglip 本身基本正常，只用 siglip 时 300 步后只有小幅下降。
- 退化程度随 debias 权重增大而加重，所以问题出在重建的 debias 部分：eraser 的位置、GRL 强度或调度、loss 权重。
- 原版 v2/v2.1 用的应该是弱得多的对抗强度。也可能用了 DANN 式的 alpha 预热，或者 eraser 接在别的位置。这些都没有留下记录。


## debias 实现对比与修正 (2026-10-10)

**结构与原版一致**: 从 Lark 白板 (W1vgwKraWh3QCfbg4CDj5ZDwpTc) 的形状和连线还原出拓扑: raw style emb (各 stage mean/std 拼接) -> 共用一个 `-grad` -> Content Eraser MLP -> probs -> Lbce <- 冻结 Anime Tagger probs;
同一个 `-grad` -> Character Eraser MLP -> probs -> Lkl <- MLP(Train) <- CCIP(Freeze)。与 `model/csip_debias.py` 一致; `loss/kl.py` 与原文件快照逐字相同。白板没有 LayerNorm, eraser 接在 norm 前/后仍未知。

**问题 1 (根因) eraser 根本没学**: 重建配置把 eraser 与 backbone 放在同一参数组 (lr 1e-5, wd 1e-2)。
- 2043 步全程 content BCE ≈ 1.2-1.4 (= 2 个数据集 x ln2, 即输出全 0 的水平)
- `csip_dino_v2.1_fromv2_0918/model-2043` 中 `content_eraser.2.weight` std 0.0256 / `.0.weight` std 0.0100, 与随机初始化 (0.0255 / 0.0100) 相同, `.2.bias` 均值 -0.01 (拟合稀疏标签需要约 -5)
- 于是 GRL 只是在推 backbone 去骗一个随机 MLP: style PR-AUC 掉 (0.79 -> 0.66 / 0.19), bias 基本不降 (I_c 0.127 -> 0.124)

**问题 2 teacher 塌缩**: char teacher MLP 可训练且在 KL 中不 detach, teacher 与 char eraser 可一起变成同一个近均匀分布 (char KL 0.42 -> 0.04 而权重仍是初始值)。

**诊断 `diag_eraser_0918`** (冻结 v2 backbone, eraser lr 1e-3, 300 步): content BCE 1.39 -> 0.023 (两数据集之和), 说明 v2 特征里内容信息很容易被读出, 之前完全是 lr 问题。
(同时训练的常数 prior 在 lr 1e-3 下 300 步来不及收敛, 不能当无信息基线。)

**修正 `fix_csip_0918_*`**: eraser/teacher 单独参数组 lr 1e-3 无 wd, 用诊断的 eraser 权重初始化 (`repro/data/merge_eraser_init.py`), `detach_teacher=True, teacher_scale=3`; 搜 grl_alpha 1/0.3/0.1 + siglip-only 对照。

## 训练提速 (2026-10-10)

| 1 卡 H200, caformer_b36@384, bs128 | backbone 训练吞吐 |
|-|-|
| fp32 (原配置 mixed_precision=None, allow_tf32=False) | 17.8 img/s |
| tf32 | 57 img/s |
| bf16 | 107 img/s |
| bf16 + 按 stage torch.compile | 275 img/s |

- profile: bf16 下 70%+ 时间在逐元素/LayerNorm/dtype 拷贝等访存型 kernel, matmul+attention 不到 10%, 所以 compile 融合收益大; channels_last 无效; 整体 compile 会被取 style 的 forward hook 打断, 需按 stage compile (`compile_model=True`)
- 数据加载: 每个数据集 2 worker 只有 98 img/s (32 worker 1098 img/s), 改为 12
- 实际训练 (含冻结 tagger + CCIP 前向): 68 s/step -> 16.8 s/step (全局 batch 2048 图), 300 步 5.7h -> 1.4h
- 测速脚本 `repro/bench_speed.py <cfg> model|data|opt`

### 修正后搜索结果 (从 v2 训练 300 步, 1 卡 bf16+compile; 2026-10-10)

| 模型 | test100_0918 PR-AUC | content I_c | char I_char |
|-|-|-|-|
| HF v2 (起点) | 0.7919 | 0.1274 | 0.0603 |
| HF v2.1 (目标) | 0.7445 | 0.0909 | 0.0360 |
| siglip only (fp32, 旧) | 0.7629 | 0.1284 | 0.0678 |
| siglip only (bf16+compile) | 0.7636 | 0.1159 | 0.0504 |
| fix grl_alpha 1 | 0.3432 | 0.0627 | 0.0127 |
| fix grl_alpha 0.3 | 0.5675 | 0.0741 | 0.0146 |
| fix grl_alpha 0.1 | 0.6870 | 0.0791 | 0.0189 |
| fix grl_alpha 0.03 | 0.7344 | 0.0938 | 0.0293 |
| fix grl_alpha 0.01 | 0.7522 | 0.1046 | 0.0387 |

- 修正后 debias 确实在去偏: alpha 0.1 的 content/char bias 已低于 HF v2.1, 但 PR-AUC 也比 v2.1 低 0.06; alpha 越大越去偏、风格越差
- 两个 siglip-only 对照 bias 相差约 0.01, 可作为 bias 指标的噪声量级
- alpha 0.03 与 HF v2.1 几乎重合 (PR-AUC -0.010, I_c +0.003, I_char -0.007), 用它跑完整 1 epoch: `fix_csip_0918_g003_full` (`repro/run_fix_full.sh`)

### grl_alpha 0.03 完整 1 epoch (`fix_csip_0918_g003_full`, 2043 步, 2026-10-11)

| 模型 | test100_0918 PR-AUC | content I_c | char I_char |
|-|-|-|-|
| HF v2 (起点) | 0.7919 | 0.1274 | 0.0603 |
| **HF v2.1 (目标)** | **0.7445** | **0.0909** | **0.0360** |
| step 500 | 0.7540 | 0.0920 | 0.0276 |
| step 1000 | 0.7430 | 0.0865 | 0.0251 |
| step 1500 | 0.7432 | 0.0860 | 0.0240 |
| step 2000 | 0.7451 | 0.0848 | 0.0241 |
| **step 2043 (最终)** | **0.7451** | **0.0856** | **0.0241** |

- 最终模型 test100 PR-AUC 0.7451 与 HF v2.1 的 0.7445 几乎相同; content bias 相当 (-0.005, 在噪声内), char bias 更低 (-0.012)
- csip_eval_public: 最终模型 0.8785, HF v2.1 0.8723 (v1 0.7727, v2 0.8414); zero-shot top1 (1 ref) 0.873 vs 0.897, (5 ref) 0.977 vs 0.976 (HF v2.1 在 CPU 上 fp32 测, 其余为 GPU bf16)
- 1000 步后指标基本稳定, 说明 v2.1 很可能就是从 v2 出发、弱 GRL 的约 1 epoch 微调
