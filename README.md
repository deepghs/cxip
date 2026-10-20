# CxIP: 动漫角色 / 画风特征 (CCIP & CSIP)

CxIP 训练两类图像特征模型，backbone 都是 CAFormer-B36 (anime_dino 初始化)，用 SigLIP 对比损失训练：

| 模型 | 作用 | HF |
|-|-|-|
| **CCIP** (Contrastive Character Image Pretraining) | 判断两张图是不是同一个角色 | `dghstmp/AnimeCCIP` |
| **CSIP** (Contrastive Style Image Pretraining) | 判断两张图是不是同一个画师/画风 | `dghstmp/AnimeCSIP` (v1)、`AnimeCSIP-v2`、`AnimeCSIP-v2.1` |

CSIP 的三个版本：
- **v1**：各 stage 特征的 (mean, std) 拼接 → MLP → style embedding。
- **v2**：在 head 前加 LayerNorm，并用 **Content Eraser** 去掉内容信息：style embedding 经梯度反转层 (GRL) 预测冻结 tagger 的标签概率。
- **v2.1**：再加 **Character Eraser** 去掉角色信息 (经 GRL 拟合冻结 CCIP 的输出分布)，并增加一个按角色分组的数据集 (CharBucket)，让同一 batch 里出现"同角色、不同画师"的样本。

> 2025-09 后的服务器代码曾丢失，2026-10 从 Cursor 缓存、Lark 文档和 HF 仓库恢复，并在 H200 上复现，CCIP 和 CSIP v2.1 已恢复。
> 恢复过程见 [docs/RECOVERY.md](docs/RECOVERY.md)，完整的探索路线 (2024 年至今试过的方案、参数和结论) 见 [docs/EXPLORATION.md](docs/EXPLORATION.md)。

## 安装

```bash
pip install -r requirements.txt
```

另需安装 [RainbowNekoEngine](https://github.com/IrisRainbowNeko/RainbowNekoEngine) (提供 `neko_train` / `neko_run` 等命令)。所有命令都在仓库根目录下运行。

## 目录结构

```
model/          模型: CAFormer 骨干 (caformer.py)、v2/v2.1 训练模型 (csip_debias.py)、冻结的 tagger/CCIP (content.py)、GRL
data/           数据源 (source.py)、按角色分组的 CharBucket (char_bucket.py)、content eraser 使用的标签表 (content_tags/)
loss/           SigLIP、KL 损失
evaluate/       训练中的指标；content / character bias 探针 (content_bias.py、char_bias.py)
actions/        推理 workflow 动作 (保存特征等)
cfgs/
  launcher/     accelerate 启动配置 (multi 8 卡、launcher_4gpu、launcher_1gpu 等)
  py/train/contrastive/   训练配置: ccip_wds.py (CCIP)、csip2_1.py (CSIP v2.1)、csip*.py (v1 及 2024 年的各条路线)
  py/infer/     提取特征 (ccip_save.py、csip_save.py)
  py/eval/      引擎内评测
export/         导出为 HF 格式: trans_ccip.py、trans_csip.py，模板 ccip/、csip/，以及 HF 上各版本的原版文件 (*_model/)
scripts/
  data/         数据处理: 画师划分 (split_labels.py)、zerochan 整理、webdataset 过滤等
  tagger/       content bias 用的 tagger 概率和标签选择 (content_tags.py、img2prob.py、tag_map.py、csip_sub.py)
  char_labels/  CSIP v2.1 的角色伪标签: CCIP 特征 UMAP+HDBSCAN 聚类 (ccip_cls.py) → train_char_labels.json
  eval/         特征级评测: 聚类 NMI/ARI、zero-shot、OOD 阈值、雷达图 (eval_ccip*.py、eval_csip*.py)
  infer/        CSIP v1 推理示例
experiments/    H200 上的复现实验: 配置、运行脚本、统一评测和全部结果，见 experiments/README.md
docs/           探索路线、恢复记录、评测可视化说明
```

## 训练

### CCIP

HF `AnimeCCIP` 用的是 zerochan 角色数据 (zerochan-character-w640-ws-m50-full，5031 个角色) 和下面这份配置：

```bash
neko_train --cfg cfgs/py/train/contrastive/ccip_wds.py
```

`cfgs/py/train/contrastive/ccip.py` 是同一模型在 ccip_v1 (3982 个角色) 上训练的版本；用它训练时 zerochan 测试集明显偏低 (见 experiments/README.md)。

### CSIP v2.1

1. **角色伪标签**：用 CCIP 给训练集提特征，再聚类成角色伪标签。
   ```bash
   neko_run --cfg cfgs/py/infer/ccip_save.py test_img_root=<训练集图片> out_dir_name=<特征目录>
   python scripts/char_labels/ccip_cls.py --feat-dir <特征目录> --output-dir <输出目录>
   python scripts/char_labels/clusters_to_char_labels.py --clusters <ccip_clusters_*.json> --style-label-file <train_split.json>
   ```
2. **内容标签** (可选，仓库已带生成好的 `data/content_tags/caformer_b36.dbv4-full.csv`)：`python scripts/tagger/content_tags.py`。
3. **训练**：v2.1 是在 v2 的基础上微调 1 epoch。可复现的完整配置和一键脚本 (训练 → 导出 → 评测)：
   ```bash
   neko_train --launch_cfg cfgs/launcher/launcher_1gpu.yaml --cfg experiments/csip_v2.1/final/fix_csip_0918_g003_full.py
   bash experiments/csip_v2.1/final/run_fix_full.sh      # 或者直接跑整个流程
   ```
   配置中的数据路径是 H200 上的路径，换机器时需要修改。关键设置：content/char eraser 单独一个参数组 (lr 1e-3，无 weight decay)，GRL alpha 0.03，char teacher 输出 detach；原因见 experiments/README.md 的"debias 实现对比与修正"。
   `cfgs/py/train/contrastive/csip2_1.py` 是恢复出的原始配置 (后半部分为原文)，超参数待填。

单卡调试可用 `neko_train_1gpu --cfg ...`。

`cfgs/py/train/contrastive/` 下 2024 年各条路线的配置 (`csip_cat.py`、`csip_info_nce*.py`、`caip/cwip_info_nce.py` 等) 用的是旧版引擎接口 (`ClsEvaluatorContainer`、`CkptManagerPKL`)，在当前 RainbowNekoEngine 下无法加载，仅作为历史记录保留。

## 导出为 HF 格式

```bash
cd export
python trans_csip.py --ckpt <训练 ckpt.safetensors> --out <输出目录>
python trans_ccip.py --ckpt <训练 ckpt.safetensors> --out <输出目录>
```

导出结果用 HF 上的原版代码，可通过 `AutoModel.from_pretrained(..., trust_remote_code=True)` 加载。导出时会丢掉 eraser、冻结的 tagger/CCIP 等训练期模块。

## 评测

| 脚本 | 内容 |
|-|-|
| `experiments/common/pair_eval.py` | 测试集内全部图片两两配对：PR-AUC (HF 卡片指标)、ROC-AUC、best F1、zero-shot top-1；直接读 HF 格式模型 |
| `experiments/common/bias_eval.py` | content bias I_c 和 character bias I_char (探针相对高斯噪声基线的 loss 下降比例，越低越好) |
| `experiments/common/summarize.py` | 把多个结果 json 汇总成 markdown 表 |
| `scripts/eval/` | 基于已提取特征的评测：聚类 NMI/ARI、zero-shot N 类 A 个 anchor、OOD 阈值扫描、雷达图 |

```bash
python experiments/common/pair_eval.py --model dghstmp/AnimeCSIP-v2.1 --data csip_eval_public=folder:<csip_eval 目录> --out result.json
```

## 复现结果

CSIP v2.1 (从 HF v2 微调 1 epoch，`experiments/csip_v2.1/final/`)：

| | test100 PR-AUC | csip_eval_public PR-AUC | content bias I_c | char bias I_char |
|-|-|-|-|-|
| HF v2 | 0.7919 | 0.8414 | 0.127 | 0.060 |
| HF v2.1 | 0.7445 | 0.8723 | 0.091 | 0.036 |
| 复现 | 0.7451 | 0.8785 | 0.086 | 0.024 |

test100 是按指纹精确恢复的原始测试画师划分；bias 为本仓库重建的指标，绝对值与 HF 卡片不同，版本间的相对关系一致。
CCIP、debias 超参搜索、训练提速等全部记录见 [experiments/README.md](experiments/README.md)。

## 文档

- [docs/EXPLORATION.md](docs/EXPLORATION.md)：探索路线 (2024 年的 16 条路线、评测体系、content/character bias、v2/v2.1 的由来)
- [docs/RECOVERY.md](docs/RECOVERY.md)：代码恢复记录，哪些文件是原文、哪些是重建
- [experiments/README.md](experiments/README.md)：复现实验的数据、配置、结果和问题诊断
- [docs/metrics_visualization.md](docs/metrics_visualization.md)、[docs/radar_charts.md](docs/radar_charts.md)：评测可视化脚本说明
