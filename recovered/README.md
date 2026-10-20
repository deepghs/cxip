# CxIP 代码恢复（2026-10-06）

服务器 `/data/cxip` 丢失。git 最后提交为 a945654（2025-08-22），之后到 2025-12 的工作需要找回。
恢复来源：
- 本机 Cursor 缓存（composer、bubble、codeBlockDiff）
- Lark 文档《AnimeDINO + cxip训练》及其白板
- HF 私有仓库：`dghstmp/AnimeCSIP-v2.1`、`AnimeCSIP-v2`、`AnimeCSIP`、`AnimeCCIP`
- 本机的 RainbowNekoEngine、AnimeDiffusion、AnimeModels 三个项目

探索路线（各阶段试过什么、参数、结果、代码在哪）见 [EXPLORATION.md](EXPLORATION.md)。

**状态（2026-10-20）：CCIP 和 CSIP v2.1 视为已恢复。**
- CCIP：训练代码为原文 `cfgs/py/train/contrastive/ccip_wds.py`，HF AnimeCCIP 用的就是它和 zerochan 数据（已确认）。
- CSIP v2.1：模型和损失按白板重建，在恢复出的原始划分上从 v2 微调 1 epoch，PR-AUC 与 content/char bias 均与 HF v2.1 吻合。复现流程、配置和结果见 [../repro/README.md](../repro/README.md)。

## 目录

| 目录 | 内容 | 可信度 |
|-|-|-|
| `files/` | 原样恢复的文件（来自 Cursor 缓存和 HF 仓库） | 原文 |
| `reconstructed/` | 依据文档、白板、HF 原版代码和残片重建的文件，可按相同相对路径合并回项目 | 重建，需复核 |
| `raw/` | 全部历史版本快照、Cursor 会话记录、Lark 白板图片、HF 仓库原始下载（`raw/hf/`，含 README） | 原始材料 |

## files/（原文）

- **HF 导出代码**（来自 HF 仓库）
  - `transfs/csip/{modeling_csip,configuration_csip}.py`：v2/v2.1 导出模板。共 72 行，与 Cursor 记录的服务器文件行数（73，含末尾空行）吻合。
  - `transfs/csipv2_model/`、`transfs/csipv2.1_model/`：config、preprocessor 和 modeling 代码。
  - `csip_model/`：v1 的导出代码。
  - `ccip_model/`：新版 CCIP 的导出代码，以及 `metrics_vs_threshold.csv`。
- **CSIP v2.1 数据**
  - `data/source.py`：本地版本加上 `StyleCharImageLabelSource`。
  - `data/char_bucket.py`：`CharBucket`。
- **v2.1 训练配置残片**：`cfgs/py/train/contrastive/csip2_1.py`，只有原文件第 180–254 行。
- **损失**：`loss/kl.py`（`LogitsKLLoss`）、`model/loss.py`。
- **偏差评估**：`evaluate/content_bias.py`、`evaluate/char_bias.py`。
- **角色伪标签**：`tools/ccip_cls.py`、`csip_v2.1/ccip_cls.py`，生成 `train_char_labels.json`。
- **Tagger 相关工具**：`tools/img2prob.py`、`tools/tag_map.py`、`tools/csip_sub.py`。
- **评估脚本**：`eval_ccip.py`、`eval_ccip_zero.py`、`eval_csip.py`、`eval_csip_zero.py`、`eval.sh`。
- **绘图**：`metrics_visualization.py`、`plot_csip_radar_charts.py` 等。
- **数据处理工具**：`split_labels.py`、`filter_webdataset.py`、`proc_zerochan.py`、`ckpt_cvt.py` 等。

## 从 HF 原版代码确认的事实

1. **v2 起在 head 前加了 `LayerNorm(3328, eps=1e-6)`**，即 `feat = head(norm(style_all))`。v1 没有这一层，本地提交的代码和 AnimeDiffusion 里的 `style_encoder.py` 也都没有。
   - 注意：AnimeDiffusion 的 `CAFormerStyleBackbone.from_pretrained` 用 `strict=False` 加载。如果换成 v2/v2.1 的权重，`norm` 会被静默丢弃，得到的特征是错的。
2. **`BatchCosineSimilarity` 的初始化为 `logit_scale=log(10)`、`logit_bias=-10`**，即 SigLIP 的标准初始化。本地 `model/caformer.py` 中是 ones/zeros。
3. HF 端模块名叫 `backbone.*`，训练端叫 `caformer.*`，导出时会重命名。
4. HF 权重只含 `backbone`/`norm`/`head`/`sim`，训练期的 eraser 已被剥离，无法从权重反推 eraser 的结构。
5. 冻结的 CCIP 可以直接用 `dghstmp/AnimeCCIP`（2025-10-08，基于 anime_dino）。它的预处理和 CSIP 完全相同（384×384，ANIME_MEAN/STD）。
6. 同画风阈值：v2 为 `sim_score > 1.7`，v2.1 为 `sim_score > 0.423`。v2.1 的 OOD 阈值在 (1.536, 1.993) 之间。以上取自 HF README。

## reconstructed/（重建）

| 文件 | 说明 |
|-|-|
| `model/grl.py` | 梯度反转层 |
| `model/content.py` | `load_preprocessor`、`FrozenTagger`、`FrozenCCIP`。`FrozenCCIP` 默认从 HF `dghstmp/AnimeCCIP` 加载 |
| `model/csip_debias.py` | v2/v2.1 训练模型，已按 HF 原版加上 LayerNorm 和 sim 初始化 |
| `loss/siglip.py` | 修复本地版本的 bug |
| `data/`、`loss/`、`model/` 下的 `__init__.py` | 补充导出 |
| `cfgs/py/train/contrastive/csip2_1.py` | 前半部分重建，后半部分为原文 |
| `transfs/trans_csip.py`、`trans_ccip.py` | 把训练权重导出到原版 HF 代码。产出的 config、preprocessor 和 modeling 与 HF 上的文件一致 |

已验证（CPU，pt25 环境）：训练模型能正常前向、反向并计算三个 loss，冻结模块没有梯度。训练 ckpt（含 eraser 等多余 key）经 `trans_csip.py` 导出后，用**原版 HF 代码**通过 `AutoModel` 加载，logits 与训练模型完全一致（max diff 0）。

需要人工确认（配置中标了 TODO）：
- loss 权重
- GRL 的 alpha
- ~~tagger 是否只取部分标签~~：原版做法无证据，现改为只用内容标签，见下文「content eraser 的标签选择」
- dataset1 的 batch size
- eraser 接在 LayerNorm 之前还是之后（当前接在之后）

设计风险：CCIP 后面的 MLP 是可训练的，可能和 char_eraser 一起塌缩成同一个常数分布。可以考虑冻结这个 MLP，或加熵约束。

## 仍无法恢复

- `tools/style_encode.py`
- `cfgs/py/infer/*_save.py` 的新版
- `csip.py` 的服务器新版
- 训练日志
- v2/v2.1 训练时 eraser 的具体实现和超参数

## 合并进项目的情况（2026-10-06）

`reconstructed/` 和 `files/` 中的代码已全部合并进项目，只有以下内容留在 `recovered/`：
`files/.claude`（无关文件）、`files/csip_zero*_metrics.csv`（评估结果数据）和 `raw/`。

### 目录调整

HF 导出相关的代码统一放到 `transfs/`：

| 路径 | 内容 |
|-|-|
| `transfs/csip/`、`transfs/ccip/` | 导出模板（原文） |
| `transfs/csipv1_model/`、`csipv2_model/`、`csipv2.1_model/`、`ccip_model/` | HF 原版参考文件 |
| `transfs/trans_csip.py`、`transfs/trans_ccip.py` | 导出脚本，在 `transfs/` 目录下运行 |

### 合并时新增或修改的内容

| 文件 | 改动 |
|-|-|
| `model/caformer.py` | `BatchCosineSimilarity` 初始化改为 log(10)/-10，`csip_debias.py` 中重复的初始化已删去 |
| `cfgs/py/train/contrastive/ccip.py` | 新版 CCIP：MLCE 改为 SigLIP，用 `LossGroup` 包装（见下文引擎兼容性），`exp_dir=exps/ccip_dino` |
| `cfgs/py/train/contrastive/csip2_1.py` | `ckpt_saver(layers='trainable')`，不保存冻结的 tagger/ccip |
| `cfgs/py/infer/ccip_save.py`、`csip_save.py` | 特征提取配置，每张图存为 `{id}.pt` |
| `actions/save.py` | `SaveFeatAction`，提取配置中用来保存特征 |
| `cfgs/py/eval/csip.py` | `eval.sh` 中引用但已丢失的 CSIP 评估配置 |
| `tools/clusters_to_char_labels.py` | 把 `ccip_cls.py` 的聚类结果转成 `train_char_labels.json` |
| `actions/__init__.py` | `postprocess.py` 依赖的 `MemoryMixin` 已从新版引擎移除，改为 try-import |
| `eval.sh`、`train.sh` | 合并恢复版本，补充新流程的命令 |
| `tools/content_tags.py`、`data/content_tags/caformer_b36.dbv4-full.csv` | content eraser 只用内容标签，见下文 |
| `model/content.py` | `FrozenTagger(tag_file=...)`；不装 imgutils 时按 `preprocess.json` 搭预处理（含 pad_to_size）；冻结模型构造后即为 eval |

### content eraser 的标签选择

content eraser 应擦除"画了什么"，以及文字、元数据。签名、水印、用户名、语种、年份会成为识别画师的捷径，同样要 debias。画风等属性不应擦除。`animetimm/caformer_b36.dbv4-full` 只有三类标签：

- general 9225 个，画风、构图、元数据类标签都混在里面；
- character 3247 个；
- rating 4 个。

`tools/content_tags.py` 的做法：

- **general 按 Danbooru tag_group wiki 分节归类。** 数据来自 `deepghs/danbooru_wikis_full` 的 `wiki_pages.csv`，再补充命名规则和人工列表。
- **文字、元数据参与 debias，共 89 个，优先级高于排除规则：**
  - 签名、水印、`*_username`、平台 logo、web_address；
  - 各语种文字、`!`、`?` 等文字片段；
  - 年份（2005–2025、dated）。

  `watermark`、`web_address`、`logo` 同时出现在 image_composition 的 Flaws/Format 里，按此优先级仍算作文字、元数据。`--drop-text-meta` 可去掉这部分。
- **排除 347 个非内容标签：**
  - 构图：视角、取景（`upper_body`、`cowboy_shot` 等）、format（comic、4koma、cover）；
  - 画风与技法：sketch、realistic、`*_(style)`、flat_color、景深、bokeh、光效，以及 dot_nose / lips / eyelashes 等五官画法；
  - 背景：`*_background`、simple_background；
  - 色调：monochrome、greyscale、`*_theme`；
  - 漫画元素：对话框、`spoken_*`、拟声词；
  - 后期打码（马赛克、黑条）；
  - 画质缺陷：bad_anatomy、blurry 等。
- **保留的内容标签：**
  - 画面中的实物和遮挡：光源实物（灯、火把）、身上的文字、遮挡式和谐（头发、蒸汽）；
  - `*_focus`、花纹；
  - 全部 character 和 rating 标签。

最终 12129/12476 个标签作为 content eraser 的目标，csv 里每个标签都有 `content` 和 `reason` 两列，便于人工审核。`csip2_1.py` 用 `FrozenTagger(TAGGER, tag_file=CONTENT_TAGS)` 读取，content eraser 的输出维度随之变为 12129。

如果只想让 char_eraser 处理角色，可以用 `--drop-character` 重新生成，这样会去掉 3247 个角色标签。

### 引擎兼容性

本机 RainbowNekoEngine（2026-01-04）有两点需要注意：

- **criterion 必须返回 dict。** 自 7908b54（2025-12-08）起如此，`LossContainer` 要包一层 `LossGroup`。已提交的 v1 配置 `csip.py` 等仍直接写 `LossContainer`，在新引擎下会报错。
- **`NekoModelLoader` 会对 ckpt 中每个 key 取 `states[k]`。** 如果 ckpt 中有模型里没有的 key（例如 eraser），会报 KeyError。CSIP 的推理和评估配置因此都指定了 `layers=['caformer','norm','head','sim']`。

### 验证

CPU、pt25 环境、随机权重，验证脚本在 `/tmp/cxip_verify/`：

- `ccip.py` 配置可解析，SigLIP loss 可反传，`logit_bias` 有梯度。
- `csip2_1.py` 配置可解析，两个数据集的 handler 输出和三个 loss 正常：
  - tagger/ccip 保持冻结且无梯度，GRL 能翻转梯度；
  - `CfgWDModelParser` 只打开了配置中列出的层。
- `CharBucket` 每 8 个样本属于同一角色。
- `ccip_save` / `csip_save` 两个 workflow 端到端跑通：
  - 文件名与 `eval_*` 脚本、`ccip_cls.py` 的读取约定一致；
  - 保存的特征与直接前向的结果一致；
  - 含 eraser 的训练 ckpt 能正常加载。
- 真实 tagger（`verify_content_tags.py`）：`tag_file` 切片与全量输出取下标结果一致，content eraser 输出 12129；`pad_to_size` 行为与 imgutils 一致。
- `trans_ccip.py`、`trans_csip.py` 导出后，用原版 HF 代码通过 `AutoModel` 加载，与训练模型的 max diff 为 0。导出的 config、preprocessor 和 modeling 与 HF 原版一致（只有 `transformers_version` 不同）。

## 2026-10-07 补充

- `eval_csip.py`、`eval_csip_zero.py` 换成快照中的最新版（分别为 0831 带 UMAP 版和 0918 的 OOD 版）。之前恢复的是 0831 的早期版本，原因是 INDEX 把早期的 read 记录误标成了 0922。
- `evaluate/content_bias.py`、`evaluate/char_bias.py` 按 Lark 文档和 HF 卡片的定义补了最终版：
  - 基线改为 ε~N(μ,σ)，char bias 改为 √MSE 回归 CCIP embedding，这两项都是默认设置。
  - 旧做法保留为选项：`--baseline shuffle`、`--method kl`。
  - 修复了 content_bias 中基线引用不存在属性导致的报错。
  - 已在合成数据上跑通。
- 原始画师划分（2025-09）已按指纹精确恢复，见 EXPLORATION.md 第三节。
