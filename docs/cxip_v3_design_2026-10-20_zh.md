# CxIP v3：基于 AnimeDINOv3 的新一代 CCIP / CSIP 设计

日期：2026-10-20。状态：设计已确认，M0 执行中。

## 0. 目标与约束

- 目标：基于 AnimeDINOv3 (2.09B ViT) 训练新一代 CCIP (角色) 和 CSIP (画师/画风)，用新的训练方法替换 2024-2025 的方案。
- 交付：分阶段。先训 2B teacher (也可作为 GPU 版发布)，再蒸馏出与旧版同量级 (~100-300M) 的 student (可 ONNX / imgutils)。
- 下游用途 (全部要覆盖)：生成模型的 condition encoder、数据集清洗/筛选、大规模伪标注/聚类、检索/归因、生成模型的评测/奖励。
- 算力：本机 H200 的 4-7 号卡 (7 号卡上的 ComfyUI 服务在需要 4 卡时停掉)。不碰 0-3 号卡。
- 磁盘：本机剩余约 745G；不在共享盘上大量写小文件 (2026-10-06 事故)。数据以 R2 (`sizigi-deepghs`) 为准。
- 许可：AnimeDINOv3 是 CC BY-NC 4.0，衍生的 CCIP/CSIP (含蒸馏 student) 发布前需要确认许可口径。

### AnimeDINOv3 关键事实

| | |
|-|-|
| 仓库 | hub.deepghs.org `deepghs/AnimeDINOv3-timm` (推理版，`model.safetensors` 8.35GB + `anime_dino.py`)；`deepghs/AnimeDINOv3` (训练状态，含 DINO/iBOT/tag head) |
| 结构 | timm ViT，24 层，宽 2688，21 头，SwiGLU，LayerScale，P16，1 个 CLS，**无 register**，**学习式绝对位置编码** (32×32 网格，`dynamic_img_size`) |
| 输入 | 白底 pad 成方形，bicubic 到 512 (也训过 256)，ImageNet mean/std |
| 输出 | CLS (2688)、patch (H/16×W/16×2688)、embedding = L2[L2(CLS); L2(mean patch)] (5376) |
| 训练 | WebSSL-2B → P16 蒸馏 DINOv3-7B → 100M 动漫 SSL → Danbooru 标签监督 (含 **3,247 个角色标签**，**无画师标签**) → 512 适配 → Q4 |
| 冻结评测 @512 | Zerochan 角色 kNN 83.98 (旧骨干 40.13)；CSIP 线性 91.30 (旧 82.31) |

推论：骨干已经被角色标签监督过，Zerochan 上的角色评测不是 zero-shot；画风信息只来自 SSL，而 DINO 系最后一层特征偏内容 (WikiArt 画风检索 DINOv2 40.0 vs CLIP 58.4)。

## 1. 总体阶段、模型接口、评测

### 1.1 阶段

| 阶段 | 内容 | 依赖 |
|-|-|-|
| M0 基础设施 | 数据上 R2；带索引的 tar 随机读取 + P×K 采样器；评测套件 v2；2B 加载/推理；吞吐实测；冻结基线；"旧配方 + 新骨干"对照组 | 权重 |
| M1 CCIP teacher | 短预算消融 → 长训 | M0 |
| M2 CSIP teacher | 短预算消融 → 长训 | M1 (角色去偏用 CCIP teacher) |
| M3 蒸馏 | 2B teacher → student | M1, M2 |
| M4 发布 | HF 格式、ONNX、阈值和指标曲线、模型卡 | M3 |

teacher 训练策略 (C′)：冻结特征只用于基线、LP-FT 的头部初始化和 bias 探针；所有消融都用短预算微调 (约 20 万张图) 来做，因为冻结特征上的结论 (取哪层、怎么池化) 不一定能迁移到微调后的模型。每次微调后分析逐层的权重相对变化和 CKA，事后判断哪些层发生了适配。最终配置在 512 下长训。

### 1.2 统一模型接口 (teacher 和 student 共用)

```
image → backbone → (CLS + patch tokens，可取多层)
      → head ─┬─ cond_tokens  K×D    生成模型的 condition
              ├─ embedding    768-d  Matryoshka (前缀 512/256/128 可单独用)
              └─ sim          exp(scale)·cos + bias   校准分数 (阈值/OOD/奖励)
```

- CCIP cond_tokens：K≈16 个可学习 query 对 patch tokens 做注意力池化。
- CSIP cond_tokens：每个取样层一个画风 token (由粗到细)。
- HF 导出保持旧接口 (`AutoModel.from_pretrained(..., trust_remote_code=True)`，score 公式不变)。

### 1.3 评测体系

1. **CCIP 开放集测试 (新)**：现有测试都是闭集 (ccip_merged_v1 的 test 4,953 类全部在 train 里；zc500x10 的角色也在 m50-full 训练集里)。
   - 从所有训练源整类剔除约 1,000 个角色，按作品分层抽样；其中一部分按 copyright 整组剔除 ("新作品"子集)。
   - 按"是否在 AnimeDINOv3 的 3,247 个 danbooru 角色标签里"拆成"骨干见过 / 没见过"分别报告。
   - 旧闭集测试 (ccip_eval、test_ccip、ccip_test_v1、zc50x15、zc500x10) 保留，用于与 HF AnimeCCIP 对比。
2. **CSIP**：保留 test100_0918、csip_eval_public (与 HF v1/v2/v2.1 可比)；新增跨角色评测 (同角色不同画师的对作为困难负样本的 PR-AUC)。
3. **OOD zero-shot**：沿用 Lark 文档的 (N, X) ∈ {(20,20), (40,20), (60,40)}、A ∈ {1,3,5}；新增 (N=500, X=500)，按作品成组抽样。基线 (新版 CCIP，OOD F1)：N=40/60/100 时 0.916-0.979。
4. **按用途的指标**：

| 用途 | 指标 |
|-|-|
| 清洗 | 固定阈值 P/R/F1；F1 平台宽度 (F1 ≥ 0.98·max 的阈值区间)；最佳阈值的跨数据集漂移；OOD AUROC |
| 伪标注 | UMAP+HDBSCAN 的 NMI/ARI、簇纯度 |
| 检索 | zero-shot top-1/5 (1/5 张参考图)、mAP@R |
| 条件/奖励 | CSIP 的 content bias I_c、char bias I_char；CCIP 的画风泄漏 I_style (探针从 CCIP 向量预测画师) |

5. **成功标准**：
   - CCIP：闭集测试不低于 HF AnimeCCIP；开放集和 OOD 协议上明显超过 HF AnimeCCIP；I_style 进入 < 0.05 档。
   - CSIP：检索指标不低于 v2 (test100_0918 PR-AUC 0.792)，同时 I_c < 0.05 且 I_char < 0.05 (v2.1 是 0.060 / 0.0937)。
6. **两档速度**：消融用以困难负样本为主的快速子集 (10 分钟内)，里程碑跑全集；固定 seed，结果写 json 后自动汇总 (沿用 `experiments/common/summarize.py`)。

现有评测已经饱和 (danbooru CCIP 聚类 NMI=1.0；N=20 zero-shot F1=1.0)，快速子集必须以困难负样本为主，否则消融无法区分配置。

## 2. Teacher 设计

### 2.1 CCIP teacher

- **数据**：ccip_merged_v1 (33,173 类 / 981,963 张训练图，每类中位数 6 张，最多 821 张)，先剔除开放集角色。扩展 (消融)：danbooru ws-full 中恰好 1 个角色标签的单人图 (带画师，可强制跨画师正样本)；类别名对齐复用 ccip_merge 的 LLM 判定流程，跨源去重用 sha256 + embedding 近重复。
- **跨画风正样本**：合并数据没有画师字段，类内按 CSIP 向量 (先用 HF v2.1，之后换新 CSIP) 做最远点采样选 K 张。
- **采样**：P×K，全局 batch 1024 = 256 类 × 4 张；类别采样概率 ∝ n_c^0.5；约 25% 的类按作品成组 (同作品困难负样本)；`variants.json` 里未合并的变体作为困难负样本。
- **增强**：pad 成方形、RandomResizedCrop (0.4-1)、翻转、轻微旋转、webp/JPEG 压缩、轻微模糊；多粒度视图 (整图/全身/上半身/头部，用 imgutils 检测器)。**禁用色相/饱和度抖动和灰度化** (发色、瞳色是身份特征)。
- **模型**：骨干微调方式由消融决定 (所有层 LoRA / 全量 + LLRD 0.85 / 只解冻最后 N 层)。头部：16 个 query 注意力池化 → cond_tokens；embedding = MLP([CLS; mean(cond_tokens)]) → 768 Matryoshka。池化消融：只用 CLS、CLS 拼 patch 均值。
- **损失**：SubCenterArcFace (每类 3 个子中心，动态 margin m_c = a·n_c^(-1/4) + b，限制在 [0.05, 0.5]) + λ·SigLIP (4 卡 all-gather，每行 3 个正样本，bias 重新初始化)，每个 Matryoshka 前缀各算一次。LP-FT：先冻结骨干训头，再解冻。损失消融：只 SigLIP / 只 ArcFace / 混合。
- **训练**：消融用短预算 (分辨率由 M0 吞吐和位置编码插值测试决定，候选 256/384)；最终大部分步数用 384，最后 20-30% 用 512，EMA，总量约 5-10M 张。

### 2.2 CSIP teacher

- **数据**：全量重建数据 (39,634 个画师 / 4,270,807 张) 减去 0918 的 800 个测试画师和 csip_eval_public 的 122 个画师。标签全部来自 isek-ai/danbooru-tags-2024：画师 (目标)、角色、作品、created_at、general 标签。
- **采样**：P×K 个画师，同一画师的 K 张尽量来自不同角色/作品；一部分 batch 按"同角色或同作品、不同画师"成组 (真实标签版 CharBucket)。这是主要的去偏手段。
- **增强**：原始像素尺度随机裁剪 (只允许 0.8-1.25 的轻微缩放)；只用翻转；**禁用颜色、模糊、锐化**。多裁剪自一致性 (2 个全局 + 4 个局部裁剪，同图互为正样本，CSD 式 0.2 权重 SSL 项)。消融：tile 拼接视图。
- **模型**：多层 patch tokens (候选 {4,8,12,16,20,24}，排除 padding，逐层归一化) → 每层 mean/std (消融：低秩协方差) → 每层 MLP 得到画风 token (= cond_tokens) → MLP(concat) → 768 Matryoshka。
- **损失**：SubCenterArcFace (画师，每类 3 个子中心) + λ·SigLIP + 0.2·多裁剪 SSL + 去偏项。去偏消融顺序：(1) 只靠数据层同内容负样本；(2) 画风向量与冻结 CCIP 向量、内容标签向量的批内交叉协方差惩罚 (对照：repro 里修好的 GRL)；(3) 训练后在最终向量上做 LEACE。CCIP teacher 向量离线计算 (约 6.5GB)。
- **微调方式假设**：CSIP 需要调整中浅层 (Surgical Fine-Tuning)。比较：所有层 LoRA / 全量 + 温和 LLRD 0.9 / 分块组解冻 {0-7}、{8-15}、{16-23}。
- 两个 teacher 都要有"旧配方 + 新骨干"对照组 (注意力池化 + 单卡 SigLIP)，用来区分骨干和配方各自的贡献。

## 3. 蒸馏、数据与 R2、代码组织

### 3.1 蒸馏 student

- 候选：CAFormer-B36 (用 HF AnimeCCIP / AnimeCSIP-v2.1 热启动，imgutils 管线不用改，先做)；ViT-B/16 (可做 token 级蒸馏)。
- 损失：监督损失 + α·embedding 余弦回归 + β·批内相似度矩阵蒸馏 (迁移阈值和 OOD 行为) + γ·cond_tokens MSE。
- 数据：有标签训练集 + R2 上的大规模无标签数据 (`aio-selfsupervised-raw/v1` 等)。teacher 输出离线缓存 (每图 2-3 个固定视图)。
- 消融：两个独立 student vs 一个共享骨干双头 student。
- 导出：HF + ONNX fp32/fp16/int8，逐样本数值核对，重新算阈值和 `metrics_vs_threshold`。

### 3.2 数据与 R2

R2 bucket `sizigi-deepghs`，沿用 anime_dino 的 `datasets/khub/<名称>/<版本>/` 约定：

| 路径 | 内容 | 大小 |
|-|-|-|
| `datasets/khub/cxip-ccip-merged/v1` | ccip_merged_nd 原样 | 58G |
| `datasets/khub/cxip-ccip-merged/v2` | 剔除开放集角色 (+ danbooru 扩展) | 约 60-150G |
| `datasets/khub/cxip-danbooru-artist/v1` | 427 万张按画师分组打包，关联 isek 标签 | 约 250G |
| `datasets/khub/cxip-eval/v1` | 全部评测集 + OOD 协议索引 | 约 5G |
| `cxip/features/...` | teacher 向量缓存 | 约 50-300G |
| `cxip/models/...` | checkpoint (本地只留 best/last) | 每个 4-9G |

- 有标签的度量学习：带索引的 tar 随机读取。parquet 里记录每个样本的 `(shard, offset, size)`；训练前把数据集整体同步到本地，按偏移 `pread`。支持精确 P×K 采样，不产生小文件。
- 无标签的蒸馏：neko-data 现有的流式读取。
- 本地磁盘：danbooru 散文件目录 (253G) 在远端校验通过后删除；`ccip_merged` (55G，被 `_nd` 取代) 和 `zips` (54G) 归档后删除。`/data/.hf` 里的 danbooru 重复副本 (353G + 334G) 由用户决定。每次删除前先列清单给用户确认。
- 凭证：`~/.config/cxip/env.sh` (权限 600，含 R2 的 `AWS_*`、`S3_ENDPOINT_URL` 和 hub token)，不进 git。

### 3.3 代码组织与运行目录

- 代码：`cxip` 仓库新建 `v3/` 包，复用现有评测、导出和 HF 格式代码；2B 加载复用 anime_dino `models/dinov3_timm.py`；训练器沿用 anime_dino 训练 2B 时的模式 (RainbowNeko 只负责配置和实例化，自己写 torchrun 循环，bf16，按需 FSDP/ZeRO-2 和 activation checkpointing)。
- 运行目录：`/data/neko/cxip_v3/` (`models/`、`exps/`、`logs/`、`cache/`、`features/`)。

## 4. 里程碑与风险

| 里程碑 | 预计时间 |
|-|-|
| M0 基础设施 | 2-3 天 |
| M1 CCIP | 3-4 天 |
| M2 CSIP | 4-5 天 |
| M3 蒸馏 | 3-4 天 |
| M4 发布 | 1-2 天 |

| 风险 | 应对 |
|-|-|
| 2B 微调吞吐/显存比估算差 | M0 实测卡关；退到 LoRA / 只调最后 N 层 / 只用低分辨率 |
| 位置编码插值到 384 后特征下降 | M0 用冻结特征比较 256/384/512 |
| 开放集测试被骨干间接见过 | 按骨干见过/没见过分别报告 |
| CSIP 去偏与检索互相拉扯 | 画 Pareto 曲线按成功标准选点；LEACE 兜底 |
| zerochan↔danbooru 类别对齐出错 | 扩展数据只作为消融项；LLM 判定 + CCIP 复核 |
| GPU 争用 | 只用 4-7 号卡 |
| CC BY-NC 许可 | 发布前确认 |

## 5. 加速手段 (按 M0 吞吐实测结果取舍)

- 流程并行：下载、数据上 R2、评测套件、吞吐测试同时进行。
- 训练：bf16 + SDPA (flash attention)；按 block torch.compile；显存允许时不开 activation checkpointing；冻结的前若干层放在 `no_grad` 下；FLIP 式随机丢弃 50% patch token 做大部分训练，最后无丢弃微调；消融优先用骨干原生训练过的 256 分辨率 (CCIP)。
- 数据：tar 随机读取 + 多 worker webp 解码 (本机 224 核)。
- 评测：评测集特征缓存，bf16 批量推理。

### 5.1 M0-0.5 吞吐实测 (2026-10-20，单张 H200，合成输入，bf16 autocast，含 fused AdamW step)

原始数据：`/data/neko/cxip_v3/logs/bench/results.jsonl`。单位 img/s，括号内为峰值显存 GB。

| 配置 | 256 | 384 | 512 |
|-|-|-|-|
| 推理 (compile) | 500 | 217 | 118 |
| 最后 6 层 (compile) | 318 (32) | 139 (35) | 75 (32) |
| 最后 6 层 + 丢 50% patch | — | 275 (24) | — |
| 最后 12 层 (compile) | — | 103 (60) | — |
| 全量，不开 ckpt (compile) | 150 (64，bs64) | 66 (68，bs32) | 35 (64，bs16) |
| 全量 + ckpt (compile) | 120 (36，bs128) | 52 (37，bs64) | 28 (36，bs32) |
| **全量 + 丢 50% patch，不开 ckpt** | **299 (65，bs128)** | **133 (69，bs64)；137 (90，bs96)** | **72 (64，bs32)；74 (83，bs48)** |
| 全量 + 丢 50% patch + ckpt | — | 102 (34，bs64) | 57 (37，bs64) |
| LoRA r32，不开 ckpt (compile) | — | 83 (52，bs32) | — |
| LoRA r32 + 丢 50% patch | — | 168 (52，bs64) | — |

结论：
- 默认配置为**全量 + 丢 50% patch + compile + 不开 ckpt + DDP (不用 FSDP)**。
  - 384 bs64/卡，4 卡约 530 img/s；300k 样本的消融约 10 分钟。
  - 最后 10% 步数关闭丢 patch。
- activation checkpointing 只在需要更大单卡 batch 时开，代价约 25%。
- LoRA 只省显存和优化器状态，不省反传计算 (梯度仍要穿过全部 24 层)。它在 FT 方式消融里作为一个选项，不作为提速手段。
- 数据侧：`CCIPTrainTransform(384)` 单核 27 img/s，每 rank 16 个 worker 足够。

## 6. M0 任务拆分

| # | 任务 | 产出 |
|-|-|-|
| 0.1 | 下载 AnimeDINOv3-timm，sha256 校验 | `/data/neko/cxip_v3/models/AnimeDINOv3-timm` |
| 0.2 | ccip_merged_nd 上传 R2 并校验 | `datasets/khub/cxip-ccip-merged/v1` |
| 0.3 | danbooru-artist 打包 (关联 isek 标签) 上传 R2 并校验 | `datasets/khub/cxip-danbooru-artist/v1` |
| 0.4 | tar 偏移索引 + map-style 数据集 + P×K BatchSampler (n_c^α、作品成组) | `v3/data/` + 测试 |
| 0.5 | 2B 加载、吞吐实测 (256/384/512 × 冻结/LoRA/最后 N 层/全量 × compile/ckpt/token 丢弃) | `v3/bench/` + 结果表 |
| 0.6 | 冻结特征在 256/384/512 下的快速评测 (位置编码插值影响) | 结果表 |
| 0.7 | CCIP 开放集划分 (剔除角色、作品成组、骨干见过/没见过) | `cxip-eval/v1` + 划分脚本 |
| 0.8 | 评测套件 v2 (OOD 协议、F1 平台宽度、阈值漂移、I_style、跨角色 CSIP 评测) | `v3/eval/` |
| 0.9 | 冻结基线 + "旧配方 + 新骨干"对照组 | 结果表 |
