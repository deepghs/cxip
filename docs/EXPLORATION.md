# CxIP 探索路线还原

整理时间：2026-10-07。

本文按时间顺序还原 CCIP（角色特征）和 CSIP（画风特征）的探索过程：每条路线试了什么、用了哪些参数，是保留了还是放弃了，代码现在在哪里。
代码文件的恢复情况见 [RECOVERY.md](RECOVERY.md)，复现实验见 [../experiments/README.md](../experiments/README.md)。`recovered/raw/` 为本地原始材料，不入库。

## 来源与可信度

| 来源 | 覆盖时间 | 内容 | 可信度 |
|-|-|-|-|
| git 历史（master、`origin/myself`、`origin/tmpbs`，共 105 个提交） | 2024-01 ~ 2025-08-22 | 2024 年 CSIP 的各条路线；2025 年 ccip、dino、siglip 三个提交 | 原文 |
| Lark 文档《AnimeDINO + cxip训练》和 5 张白板（`recovered/raw/lark_whiteboards/`） | 2025-08 ~ 2025-12 | CCIP、CSIP v1/v2/v2.1 的结构图、结论和指标 | 原文（结论层面） |
| Cursor 对话 21 份（`recovered/raw/cursor_transcripts/`）和文件快照（`recovered/raw/snapshots/`） | 2025-08-05 ~ 2025-12-03 | 数据处理、评测协议、bias 指标、聚类、CharBucket 的实现过程 | 原文（过程层面） |
| HF 仓库（`recovered/raw/hf/`） | 2025-10 ~ 2025-12 | 发布的权重、导出代码、阈值、bias 数值 | 原文 |

注意：
- Cursor 导出文件名中的时间是会话的创建时间。正文里的 "USER 2025-09-22" 是导出时间，不能当作对话时间。INDEX 里有些快照因此被标成了 0922，实际要早得多。
- 20251001、20251010、20251030、20260310 四份对话讲的是 NekoDiffusion 和 UNet，与 CxIP 无关。

---

## 总览

```
2024-01 ─ CSIP 基线 (caformer_m36 + MLCE)
          ├─ InfoNCE ─ info_nce_weak ══▶ 2024 版 CSIP v1 (deepghs/csip_model, m36-12000)
          │                 ├─ noisy + patch (打碎内容只留笔触) ─ swinv2
          │                 ├─ cwip (服装/cosplay?)  · caip (模板)
          │                 └─ auto cluster / 阈值分析 (tmpbs 分支)
          ├─ style loss (Gram/三元组) ─ vgg style loss          [放弃]
          └─ cat (拼图 BCE) ─ cross attn                         [放弃]
2025-08 ─ AnimeDINO (caformer_b36, model-191000) 作为统一初始化
          ├─ CCIP: 线性探测 → 全量微调 → SigLIP ══▶ dghstmp/AnimeCCIP (2025-10-08)
          └─ CSIP-dino: 各 stage mean/std 风格头 + SigLIP ══▶ dghstmp/AnimeCSIP (v1)
2025-08~10 ─ 评测体系: pairwise 阈值 / UMAP+HDBSCAN 聚类 / zero-shot N 类 A 个 anchor / OOD / 与 fscls 对比
2025-11 ─ content bias 指标 (linear probe) → CSIP v2: Anime Tagger 对抗 + GRL ══▶ AnimeCSIP-v2
2025-11~12 ─ character bias 指标 → CSIP v2.1: 冻结 CCIP + Character Eraser + KL + CharBucket ══▶ AnimeCSIP-v2.1
```

---

## 一、2024：CSIP 的早期探索（git）

所有代码都能用 `git show <hash>:<path>` 取回。master 历史中没有删除过文件，"不在 HEAD" 的情况分三种：被就地改写、只在未合并分支上、配置被覆盖。

| # | 日期 | 路线 | 思路 | 关键提交 / 文件 | 结果 |
|-|-|-|-|-|-|
| 1 | 01-19～01-31 | CSIP 基线 | caformer_m36（timm 预训练），384 输入，AttentionPool → AvgAttnPooling2d，BatchCosineSimilarity，MLCEImageLoss，PosNegBucket(0.5) | `e61076a`、`cd6bd03`、`79dae0d:cfgs/py/train/contrastive/csip.py`（即现在的 `csip_old.py`） | 演变为 InfoNCE |
| 1b | 01-24～01-29 | tiny 数据调参（narugo） | lr 2e-4，500 epoch，bs 32，cosine_with_restarts | `origin/myself` 分支 `d5d8cc6` | 未合并 |
| 2 | 01-29 | AP 评估 | 去掉对角线后计算 AP | `83d80e7`、`277d897:evaluate/csip_map.py` | 已改写为 CXIPMetricContainer |
| 3 | 01-31 | InfoNCE | 改用 FeatWrapper，同时输出 (sim, feat)；exp `full_ds_csip_v1_exp_info_nce_0` | `1041b68`、`csip_info_nce.py` | 主线 |
| 4 | 02-03～02-07 | style loss | CAFormer 四个 stage 的特征接 StyleLoss，TripletBucket，评估用 CSIPTripletAPContainer | `45ca7a0`、`e4092ea:model/caformer.py`（原版 CAFormerStyleBackbone）、`40f7557:csip_style_loss.py` | 放弃 |
| 5 | 02-09 | cat | anchor 与 pos/neg 沿高度方向拼成 2H 的图，接 Linear + Sigmoid，用 StyleBCELoss；后来换成 s18 和 64px patch | `de9e1e6`、`64bc04f`、`ef25cf9`，ckpt `csip_cat_s18/csip-caformer-s18-21000.ckpt` | 放弃 |
| 6 | 02-09 | caformer_t15 | 自定义的小模型 | `00525b4` | 未使用 |
| 7 | 02～03 | flow | 推理脚本 | `scripts/infer/csip_flow.py` | 在 |
| 8 | 02-21～03-17 | cross attn | SDP_Attention 做 anchor→pos/neg 的交叉注意力，有 attn_v1/v2/v3 三种；backbone 和 head 用不同 lr | `8c85016`、`ef25cf9`、`771d3fa:csip_cross.py`（m36，exp `csip_cross_m36`） | 放弃 |
| 9 | 03-15～03-16 | **info_nce_weak** | m36，384 输入，只做 Flip 和 RandomCrop 弱增强，InfoNCE，lr 1e-4，bs 128，30～50 epoch，数据 `/data/csip_v1` | `329dd99`、`26d74cb:csip_info_nce_weak.py`（HEAD 中这份文件已被 0403af9 改成 noisy + s18） | **成为 2024 版 CSIP v1**：`exps/csip_v1_info_nce_m36-p384-weak/.../csip-caformer-m36-12000.ckpt`，即 HF `deepghs/csip_model` |
| 10 | 03-15～03-19 | cwip | 同一套 InfoNCE，换成强增强 + HeadCutAugment（遮头），数据 `/data/cwip`、`cwip_v0.1` | `329dd99`、`6d533af:cwip_info_nce.py`（含 HeadCut）、HF `deepghs/cwip_models/cwip-caformer-m36-8100.ckpt` | 用途推测是服装/cosplay，没有确认 |
| 11 | 03-16～03-18 | auto cluster / 阈值 | ContrastiveAnalysisAction 扫阈值求 F1，ClusterTestAction 用 OPTICS + ARI；tmpbs 上还有样本平衡、lighttuner HPO、分批提特征 | `69918d1`（master）；`762fa51:actions/postprocess.py`、`csip_feats.py`、`cwip_feats.py`、`cwip_threshold.py`、`1d86be1:model/caformer.py`（tmpbs） | tmpbs 部分未合并 |
| 12 | 03-17 | vgg style loss | VGG19 前 4 个 block 的特征接 StyleLoss | `35fa77e:model/vgg.py`、`csip_style_loss.py`（exp `csip_vgg19_styleloss-2`） | 放弃 |
| 13 | 03-18～03-19 | noisy + patch | CropAndStitch：随机裁 12×12 个 32px 小块拼成 384 的图，打碎内容只留笔触；NoisyInfoNCE(T=1/16) | `75b19c3`、`01e1efd:csip_info_nce_patch.py`，ckpt `csip_v1_noisy_info_nce_s18-p384-patch/...-12000` | 支线 |
| 14 | 03-19 | caip | 只建了模板，数据仍是 csip_v1 | `01e1efd:caip_info_nce.py` | 未展开 |
| 15 | 03-25～03-28 | swinv2 | 用 wd-swinv2-tagger-v3 做 backbone，448 输入，14×14 patch | `9e4e240:model/swinv2.py`、`050cbf2:csip_info_nce_patch.py`，ckpt `csip-swinv2-base-22000` | 支线 |
| 16 | 05-15 | ONNX 版余弦相似度 | BatchCosineSimilarityONNX | `8e874bc` | 已合并 |

结论：成对判别（cat、cross attn）和 Gram 风格损失两条思路都被放弃。最终保留的是"对比学习 + 弱增强"。打碎内容（patch）这条支线是 debias 思想的雏形。

---

## 二、2025-08：以 AnimeDINO 为统一初始化

| 日期 | 事件 | 细节 | 来源 |
|-|-|-|-|
| 08-05 | 查看 AnimeDINO 权重 | `/data/anime_dino/exps/anime_dinov2_bs128-lr1e-3/ckpts/model-191000.safetensors`。`scripts/ckpt_cvt.py` 在 0810 时有 37 行的转换版，这一版已丢失，仓库中只有 11 行的读 keys 版 | 对话 0805-194503 |
| 08-05 | CCIP 线性探测 | `ea34744:ccip.py`：从 `anime_dinov2_bs256-v2.1-lr4e-4/model-4` 加载，只训练 attnpool 和 sim，10 epoch，lr 1e-4，MLCE，使用 ANIME_MEAN/STD | git |
| 08-05 | CCIP v1 数据拍平 | `scripts/data/proc_data.py`：把 `/nfs/ccip/ccip_v1/{cls}/{sub}/*.jpg` 软链到 `ccip_v1_flat/{cls}_{sub}_...` | 对话 0805-201257 |
| 08-09 | CCIP 全量微调 | `cfgs/py/eval/ccip.py` 评估的是 `/data/cxip/exps/2025-08-09-00-28-22/ckpts/model-4000.safetensors` | git 271d271 |
| 08-10 | webdataset 过滤 | `scripts/data/filter_webdataset.py`：基于 `danbooru-wdtagger-v4-w640-ws-full-witha`，按 character、tags、artist 过滤，每个 shard 50000 条。实际用的过滤条件没有留下 | 对话 0810 |
| 08-12 | CSIP 画师数据切分 | `scripts/data/split_labels.py`：合并 train/test labels 后随机取 800 个画师作为 test，seed 42 | 对话 0812 |
| 08-21 | update to dino | `ccip.py` 全量微调 2 epoch（`ema_teacher.backbone.`），数据 `/nfs/ccip/ccip_v1_flat`。<br>**`ccip_wds.py`**：zerochan-character-w640-ws-m50-full，train 00001-00045，5031 个角色桶，MLCE。<br>`csip_wds.py` 和 `csip.py`：CAFormerStyleBackbone(b36)，**各 stage 的 mean/std 拼接后接 MLP**（AdaIN 思路），lr 1e-5，exp `csip_dino`、`csip_dino_style` | git 271d271 |
| 08-22 | siglip | 加入 SiglipLoss，BatchCosineSimilarity 加上 logit_bias。HEAD 中的 SiglipLoss 有 bug（引用了未定义的 `self.reduction`，mask 用了 long），现已修复 | git a945654 |

Lark 文档记录的训练量：
- 新版 CCIP 总共迭代 **6.14M** 张图，旧版 CCIP 是 20.07M。
- 2025 版 CSIP（dino）是 **4.71M**。

发布模型学到的 sim 参数（scale 取 exp 后的值 / bias）：

| 模型 | scale / bias | 来源 |
|-|-|-|
| 旧版 CCIP（ccip-caformer_b36-24） | 29.72 / -14.92 | 对话 0922 |
| 新版 CCIP（0918 的 ckpt） | 23.52 / -12.26 | 对话 0918 |
| HF AnimeCCIP（10-08） | 26.29 / -13.95 | HF 权重 |
| CSIP-dino（0831 的 ckpt） | 14.52 / -8.24 | 对话 0831 |
| HF AnimeCSIP（v1） | 17.07 / -8.63 | HF 权重 |

> 复现中发现：用 `ccip.py`（ccip_v1_flat，3,982 个角色）训练 10,153 步（约 10.4M 张）之后，scale 只到 14.7，zerochan 500 类的 PR-AUC 只有 0.61（HF 是 0.88）。
> **已确认（用户提供，2026-10-20）：HF AnimeCCIP 是用 zerochan 数据、通过 `cfgs/py/train/contrastive/ccip_wds.py` 训练的**（zerochan-character-w640-ws-m50-full，5031 个角色），不是 ccip_v1。复现用 ccip_v1 训练的差距（zerochan 500 类 0.61 vs 0.88）由此解释。

注意区分两个 "CSIP v1"：
- 2024 年的 `deepghs/csip_model`：m36 + InfoNCE。
- 2025 年的 `dghstmp/AnimeCSIP`：AnimeDINO b36 + mean/std 风格头 + SigLIP。

Lark 文档和 HF 卡片中的 v1 指后者。

---

## 三、2025-08～10：评测体系

| 日期 | 内容 | 关键参数 | 代码 |
|-|-|-|-|
| 08-31 | CSIP 聚类评估 | 先试 DBSCAN（eps 2.6～2.7）和 OPTICS，后改用 **UMAP + HDBSCAN**。距离矩阵 `D = scale + bias + 0.1 - pred`。t-SNE 用 perplexity 30，init random | `scripts/eval/eval_csip.py`（**已更新为快照最新版**，即带 UMAP 的 5a20bce7 版） |
| 08-31 | UMAP 参数网格搜索 | n_components 5～100 × n_neighbors 5～50 × mcs {10,15,20}，共 960 组。最终定为 **n_neighbors=11、n_components=35、min_dist=0、cosine**，mcs 在 2～7 之间扫 | 搜索结果 csv 已丢失 |
| 08-31 | zero-shot 第一版 | 每类随机取 1 个 anchor，重复 20 次（seed 42），指标为 macro 的 Acc/P/R/F1 | `scripts/eval/eval_csip_zero.py` |
| 09-01 | 加 Top-5 | 向量化的 top-k | 同上 |
| 09-14 | zerochan CCIP 测试集 | `small_val` 按 character 分文件夹（`scripts/data/proc_zerochan.py`），随机保留 100 个角色（`scripts/data/random_select_folders.py`，seed 未知），特征存为 `feat_ccip_zero` | 在 |
| 09-17 | 手动扫 OOD 阈值 | 在 -5～0 之间步长 0.1，最终定为 `ood_thr=-1.3`（HF 卡片中的 CCIP OOD 阈值也是 -1.3） | `scripts/eval/eval.sh` |
| 09-18 | **N 类 + A 个 anchor + OOD** | (N, seen) ∈ {(40,20), (60,40), (100,60)}，A ∈ {1,3,5}。多个 anchor 的 cos 取平均；`max score < thr` 判为未知类 | `scripts/eval/eval_ccip_zero.py`（最终版） |
| 09-18 | CSIP 套用 OOD 框架 | 结果较差（N=100、A=5 时 top1 只有 0.49），**没有写进 Lark**，推测被放弃 | `scripts/eval/eval_csip_zero.py`（**已更新为快照最新版** 528989） |
| 09-18 | 雷达图 | 新 CCIP 在 A=1 时已超过旧 CCIP 在 A=5 时的表现（面积 0.975 vs 0.977） | `scripts/eval/plot_radar_charts.py`、`scripts/eval/plot_csip_radar_charts.py` |
| 09-22 | 最佳 pairwise 阈值和最佳 OOD 阈值 | 取上三角，按 F1 选；OOD 用 `-scores` 作为概率 | `scripts/eval/eval_ccip.py`、`scripts/eval/eval_ccip_zero.py` |
| 10-08 | 指标曲线工具 | PR 曲线、阈值–P/R 曲线；所有阈值存为 float16 npy，即 HF 上的 `metrics_vs_threshold` | `scripts/eval/metrics_visualization.py` |
| 10-20 | 与 fscls 对比 | `deeplol/char_fscls_v1.0`；导出最差的 100 对 FN；fscls 在 zerochan 上 mcs=3 时 NMI 0.908 | `scripts/eval/eval_ccip.py`（当前为 fscls 版） |

已知问题：
- 0918 的 OOD 对比中，旧 CCIP 用的是新 CCIP 的 logit 参数和阈值。argmax 分类不受影响，但 OOD 判断对旧 CCIP 可能不公平。
- 0918 时 `test_split.json` 共 182,810 张，前 100 类 22,613 张。这和 HF v2.1 卡片上的测试集（10,360 张）对不上。

### 原始画师划分已完全恢复（2026-10-07）

用 0918 记录的两个数字作为指纹，在 witha 元数据上枚举划分规则，**精确命中**（182,810 / 22,613）。规则如下：
- 图片和 split 来自 `danbooru-wdtagger-v4-w640-ws-full-witha`：train 作为 train_labels，test 作为 test_labels，val 不用。
- 画师取 `artist_tags[0]`，有多个画师时取第一个。这与 `data/source.py` 中 `StyleWebDatasetSource` 的逻辑一致。
- 每个画师的最少图片数在 1～20 之间都不影响结果。
- 之后按 `scripts/data/split_labels.py`（seed 42，800 个测试画师）切分，再按 `csip_sub.py` 取前 100 个画师。

由此得到：
- train_split 2,085,298 张 / 8,680 个画师。
- test_split 182,810 张 / 800 个画师。
- test_split_100 22,613 张，正样本对 4,558,917 个。

脚本是 `experiments/data_prep/match_split_0918.py`，输出在服务器的 `data/danbooru-artist-split-0918/`。

几条推论：
- Lark 记录 CSIP 迭代了 4.71M 张，约等于这个 train_split 的 2.3 个 epoch。
- 11 月给约 2M 张图提 CCIP 特征，也与这个 train_split 的规模一致。所以 v2.1 很可能也是在这个划分上训练的。
- HF v2.1 卡片的测试集（10,360 张 / 1.67M 对）仍然对不上，可能做过额外筛选，原因未知。
- 复现时用的 isek 划分（3.93M 张 / 37k 个画师）比原始划分大得多。

HF 三个模型在恢复出的 test100 上的结果（去掉 54 张缺图后为 22,559 张），数值为 PR-AUC：

| | v1 | v2 | v2.1 |
|-|-|-|-|
| test100_0918（原始划分） | 0.746 | **0.792** | 0.745 |
| HF 卡片 | 0.663 | **0.815** | 0.757 |
| test100v2（isek 划分，有泄漏） | 0.825 | 0.929 | 0.952 |

- 在原始划分上，三个模型的排序和卡片一致：v2 最好，v2.1 略低于 v2。
- 在有泄漏的划分上，v2.1 反而最高。
- 这说明这个测试集对三个 HF 模型都是 held-out 的，也印证了 v2.1 是在同一个划分上训练的。
- 绝对值和卡片不同，是因为卡片只用了其中 10,360 张。

---

## 四、2025-11：content bias 指标 → CSIP v2

**指标（content bias I_c）**
- 用 linear probe 从 style embedding 预测 tagger 的软概率：MLP 768→256→320，BCEWithLogits，AdamW lr 1e-3，batch 128，100 epoch，按 0.1/0.1 划分 val/test。
- 计算公式：`I_c = 1 - L(f(x), y) / L(ε, y)`。
- 评估数据：
  - 内容标签来自 `/nfs/danbooru-wdtagger-v4-w640-ws-30k`：`scripts/tagger/tag_map.py` 把 323 个标签映射到 `selected_tags`，其中 3 个是 NaN，剩下 320 个。
  - 概率由 `scripts/tagger/img2prob.py` 用 `SmilingWolf/wd-vit-tagger-v3` 生成，存为 `train/probs.pkl`。
  - style 特征在 `/data/cxip/feat_csip_dbv4`。
- 注意：评估用的 tagger 是 wd-vit-v3，训练对抗用的是 animetimm caformer_b36.dbv4-full，两者不同。

**基线的演变**

| 版本 | 基线做法 | 状态 |
|-|-|-|
| 11-04 `4266de` | randperm 打乱配对 | 已被替换 |
| 11-05 `af302c` | `random.shuffle(style_list)`。但 content 按 stem 查找，打乱后配对关系不变；另外引用了不存在的属性，运行会报错 | 原先仓库里就是这一版 |
| 11-05 AI 修复（`a14e7c` 等） | 打乱 content_keys | 被拒绝 |
| **最终（Lark / HF 卡片）** | **ε ~ N(μ, σ) 输入** | 没有快照，**已按文档重建**：`evaluate/content_bias.py --baseline gauss`（默认），μ、σ 取训练集 style 特征的逐维统计 |

11-14 讨论过：BCE 近似 H(P_c | Z_s)，ΔL 是互信息的下界，但线性比值不能直接换算成信息量。

**CSIP v2 结构**（白板 OYRAwjZA）
- backbone 各 stage 的 (mean, std) 拼成 raw style emb，经 MLP 得到 style emb，用 Lsiglip 训练。
- raw style emb 同时经 **GRL（-grad）** 接入 Content Eraser，用 Lbce 拟合 Anime Tagger 的 probs。
- HF 原版代码中，v2 在 head 前加了 `LayerNorm(3328)`。

**结果**（Lark）
- I_c：0.1391（v1）→ 0.0783（v2）。
- 50 类聚类：NMI 0.912 → 0.956，ARI 0.824 → 0.897。
- zero-shot 提升约 10%。

---

## 五、2025-11～12：character bias → CSIP v2.1

**动机**：v2 做了 content debias 之后，style embedding 里仍有角色信息（画师常反复画同一个角色）。

**方案**（Lark 和白板 W1vgwKra、NEWIwvdr）
1. **冻结的 CCIP + Character Eraser + GRL + KL**：raw style emb 经 GRL 进入 Character Eraser，与冻结 CCIP 输出的分布做 Lkl。白板上 CCIP 标的是 Freeze，它后面的 MLP 标的是 Train。Content Eraser 保持不变。
2. **并行的第二个数据集**：用 CharBucket 按角色分组，一个 batch 中有相同角色、不同画风的样本，label 仍然是画师（`'style -> label'`）。白板上的示意是 Dataset1 按画风分组，Dataset2 按角色分组。

**角色伪标签**（11-17～11-27）
- 用 CCIP 给训练集提特征，存到 `/nfs/danbooru-w640-ws-full-artist/train_ccip_feats`，约 2M 个 `.pt`。
- 聚类脚本是 `scripts/char_labels/ccip_cls_orig.py`：memmap 流式读取、fp16、chunk 32768，然后 UMAP → HDBSCAN。
- 默认参数为 UMAP(15, 0.0, 35, cosine)，HDBSCAN(mcs=5)，在约 2M 张图上得到 3 万多个类。11-27 嫌"太碎"，AI 给的建议是 neighbors 60、mcs 256、min_samples 64、eps 0.15，**是否采用没有记录**。
- 复现时用默认参数在 3.93M 张图上得到 48,743 个簇，其中 48% 是噪声。
- 在恢复出的原始划分（2,079,184 张）上用默认参数复现，得到 **30,658 个簇**，噪声 45%，有标签的图 1,149,258 张。这与 11-27 对话中"约 2M 张图上 3 万多类"一致，说明当时至少用过默认参数。

**CharBucket**（11-27）
- 原版做法：每个角色凑满 min_cat_size=8 个样本，不足时重复采样；batch 128 即 16 个角色块。
- AI 建议加 `min_styles_per_char` 和按画风均衡，**六次修改全部被拒绝**，保留原版。被拒绝的方案在 `recovered/raw/snapshots/data/char_bucket.py/20251127-181014__35c897f4-edit-rejected-af5a6c.py`。

**character bias 指标的演变**（11-28～12-03）

| 版本 | 做法 | 状态 |
|-|-|-|
| 初版 | style 和 char 各接一个可训练探针（768→512→64，后改为 100），中间用 LogitsKLLoss；随机配对作为基线；train/val/test 是同一份数据 | 有数据泄露，且缺少真值监督 |
| 12-03 `88ab9e` | 按角色分层划分，val 占 0.2 | 就是原先仓库里的版本 |
| **最终（Lark / HF 卡片）** | **从 style 回归 CCIP embedding，L = √MSE，基线为 ε ~ N(μ, σ)** | 没有快照，**已按文档重建**：`evaluate/char_bias.py --method rmse --baseline gauss`（默认）；旧做法用 `--method kl --baseline shuffle` |

评估数据：`/data/cxip/feats/feat_csip_v2.1-char` 和 `feats/feat_ccip_zero`（zerochan 角色验证集）。

**结果**（Lark / HF）

| | content I_c | character I_char | 50 类 NMI / ARI |
|-|-|-|-|
| v1 | 0.1391 | 0.2452 | 0.912 / 0.824 |
| v2 | 0.0783 | 0.1991 | 0.956 / 0.897 |
| v2.1 | 0.0600 | 0.0937 | 0.934 / 0.836 |

v2.1 的聚类指标略有下降。文档的解释是画师前后期画风不一致。

**训练配置**
- 原文只剩 `csip2_1.py` 第 180–254 行：两个数据集各 bs 128，CategoryBucket(4) 和 CharBucket(8)，WeakRandAugment2，evaluator 每 2000 步评一次。
- lr、loss 权重、GRL alpha、初始化 ckpt、步数都已丢失。
- 从 HF 权重推断：v2.1 的 logit_bias 是 -23.8，比 v2 的 -18.4 更远离初始值 -10，所以 v2.1 很可能是在 v2 基础上继续训练的。

**复现结果（H200，2026-10-11，详见 [../experiments/README.md](../experiments/README.md)）**：从 HF v2 出发、在恢复出的原始划分上微调 1 epoch（2043 步），各项指标与 HF v2.1 吻合。

| | test100_0918 PR-AUC | csip_eval_public PR-AUC | content I_c | char I_char |
|-|-|-|-|-|
| HF v2.1 | 0.7445 | 0.8723 | 0.0909 | 0.0360 |
| 复现 | 0.7451 | 0.8785 | 0.0856 | 0.0241 |

（bias 为本仓库重建的指标，绝对值与 Lark 不同，相对排序一致。）关键设置：
- **eraser 和 teacher head 单独一个参数组，lr 1e-3，无 wd**。与 backbone 共用 lr 1e-5 时 eraser 2000 步都停在随机初始化，GRL 只会破坏特征而不去偏。
- eraser 先在冻结的 v2 backbone 上预热 300 步；teacher 输出 detach 并做 LayerNorm×3，防止与 char eraser 一起塌缩。
- grl_alpha 0.03，loss 权重均为 1。300 步搜索：alpha 1/0.3/0.1/0.03/0.01 的 PR-AUC 为 0.34/0.57/0.69/0.73/0.75，0.03 与 v2.1 最接近。
- 1000 步后指标基本稳定。

---

## 六、代码现状

| 路线 | 代码位置 | 状态 |
|-|-|-|
| 2024 年各条路线 | git（见第一节的 hash）；配置仍在 `cfgs/py/train/contrastive/` | 原文；tmpbs、myself 两个分支未合并 |
| CCIP / CSIP-dino / siglip | `cfgs/py/train/contrastive/{ccip,ccip_wds,csip,csip_wds}.py`，`loss/siglip.py` | 原文，siglip 的 bug 已修复 |
| 评测体系 | `scripts/eval/`（`eval_ccip*.py`、`eval_csip*.py`、`metrics_visualization.py`、`plot_*radar*.py`）；复现用的统一评测 `experiments/common/pair_eval.py`、`bias_eval.py` | 原文。**本次把 `scripts/eval/eval_csip.py` 和 `scripts/eval/eval_csip_zero.py` 换成了快照中的最新版**，原先是 0831 的早期版本，那个版本仍保存在 snapshots 里 |
| content bias | `evaluate/content_bias.py`、`scripts/tagger/tag_map.py`、`scripts/tagger/img2prob.py` | 探针部分是原文；**高斯基线是按文档重建的**，并修复了基线的 bug |
| character bias | `evaluate/char_bias.py` | KL 版是原文；**√MSE 加高斯基线是按文档重建的** |
| v2 / v2.1 模型 | `model/csip_debias.py`、`model/content.py`、`model/grl.py`、`loss/kl.py` | 按白板和 HF 原版代码重建，结构与白板连线一致，`loss/kl.py` 与原文件一致；**复现指标与 HF v2.1 吻合，视为已恢复** |
| v2.1 数据 | `data/source.py`（StyleCharImageLabelSource）、`data/char_bucket.py` | 原文 |
| 伪标签 | `scripts/char_labels/ccip_cls_orig.py`（原文），`scripts/char_labels/ccip_cls.py`（加了 `--feat-npy`、`--umap-parallel`），`scripts/char_labels/clusters_to_char_labels.py`（重建） | |
| v2.1 配置 | `cfgs/py/train/contrastive/csip2_1.py`；可复现的最终配置 `experiments/csip_v2.1/final/fix_csip_0918_g003_full.py` | 后半部分是原文，前半部分是重建 |

**仍然缺失**：
- `scripts/ckpt_cvt.py` 的转换版
- `wds_flat.py`、`data1.py`、`speed_test.py`、`datatest.py`
- `tools/style_encode.py`
- UMAP 参数搜索的结果 csv
- v2 的训练超参和训练日志（v2.1 已用复现确定一套等效超参）
- 所有实验目录（`/data/cxip/exps/*`）

---

## 七、待确认的问题

1. ~~HF AnimeCCIP 的训练数据是 ccip_v1 还是 zerochan webdataset~~：已确认是 zerochan + `ccip_wds.py`。
2. ~~danbooru-w640-ws-full-artist 是怎么筛选的~~：已恢复，见第三节。剩下的问题是：HF v2.1 卡片上的测试集（10,360 张）是怎么从 test_split_100（22,613 张）里筛出来的？
3. v2 的 lr、loss 权重、GRL alpha 和步数。v2.1 已由复现给出等效设置（从 v2 继续训练、eraser lr 1e-3、alpha 0.03、1 epoch），但原版的具体数值和 teacher 的训练方式仍未知。
4. 伪标签最终用的聚类参数和簇数。`clusters_to_char_labels` 是否过滤了小簇？
5. 高斯基线中 μ、σ 的具体取法（逐维统计还是全局统计）。char bias 的回归目标是否就是 CCIP 的 768 维特征？
6. `scripts/data/random_select_folders.py` 用的 seed，以及最终 100 个角色的名单。
