# CSIP v4 P0：数据 v2 + benchmark v2 + 基线 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal：** 完成设计文档 §6 的 P0。具体是：
- 建好带原发日期校正的 `danbooru-artist-v2` 和 `pixiv-artist-v1`；
- 建好 B1 / B2 / B3 测试集 (含 dev / test / sealed 划分)；
- 写好 benchmark v2 评测代码；
- 用 v3 发布版、v3 x4 和 Kaloscope 跑出带 bootstrap CI 的基线表。

最后停在用户检查点。

**Architecture：**

数据侧有两个阶段：
1. **prepare**：用 DuckDB / pandas 处理元数据，输出 parquet 计划表。
2. **resize_parts + neko-data build**：顺序扫描本地 4MP tar，缩到短边 640 后写成少量大 tar，再按画师连续打包。

测试集不复制图片。index.parquet 直接记录数据集 tar 里的 (tar, offset, size)，评测时 pread 读取。

评测侧以 "每画师直方图" 为核心：
- 正对按 (画师, 时间层) 统计；
- 负对按 (画师, 子集) 统计，用 2^14 个 cos 箱。

所有 PR 类指标和按画师的 bootstrap 都由这些直方图加权求和得到。逐图指标 (年代误差、MAP@R、注入 AUROC) 先按画师汇总，再做加权平均。

**Tech Stack：**
- Python 3 (`/home/ubuntu/miniconda3/envs/cxip/bin/python`)；
- pandas / pyarrow / DuckDB / numpy / torch (CPU)；
- neko-data (`neko_data.build.job.BuildJob`)；
- Pillow；
- pytest；
- curl (内部 hub)。

**Spec：** `docs/csip_next_plan_2026-10-27_zh.md` (已批准)。实施者要同时读 spec 的 §4.1、§5 和本计划。

## Global Constraints

- 所有 python 命令都显式设 `CUDA_VISIBLE_DEVICES`：CPU 任务设为空，GPU 任务只能用 4-7 号卡。用 GPU 前先跑 `nvidia-smi`，把占用情况给用户看并得到确认。
- 运行方式：`cd /data/neko/cxip && CUDA_VISIBLE_DEVICES= PYTHONPATH=. /home/ubuntu/miniconda3/envs/cxip/bin/python ...`。测试用 `... -m pytest v3/tests/<file> -q`。
- 新数据、特征一律放 `/scratch/cxip_v4/` (下文记作 `V4`)。不写大量小文件，图片一律打包进 tar。
- 所有构建脚本在输出目录已有 `manifest.json` 时直接退出 (`sys.exit`)，要换数据就升版本号，不覆盖。
- 不删除任何已有数据。确实要删时，先列清单，得到用户确认后再删。
- 图片规格同 v1：短边 ≤ 640 (不放大)，webp quality 95，保留透明通道。
- Danbooru 帖子过滤规则：
  - 单画师 (`tag_count_artist == 1`)、未删除、`file_ext ∈ {jpg, jpeg, png, webp}`；
  - 不含 `ai-generated` / `ai-assisted`；
  - 作品族 = `parent_id` 或自身 `id`。
- Pixiv 作品过滤规则：
  - `type == 'illust'`、`illust_ai_type != 2`、非 ugoira；
  - 只取 Danbooru 快照中 `pixiv_id` 没出现过的作品。
- Kaloscope 只用于评测，不参与任何训练数据的选择，B2 的排除名单除外。
- hub token 只在 `~/.config/cxip/env.sh` 里。curl 通过 `--config -` 从 stdin 读取 token，不能出现在 argv 或日志中。并发 ≤ 8。
- `deepghs/cxip` 是公开仓库，仓库里不能放画师名、画师图片或来源链接：
  - 测试用合成数据；
  - 划分名单和画师名都放在 `V4` 下；
  - 报告里不写画师名。
- git：只在用户明确同意后才 commit，每次只 add 本任务的文件。不要动 `style_flow_sdxl_*.py`。
- bootstrap：按画师重采样 1000 次，种子 0，报告 95% CI (2.5 / 97.5 分位)。
- 计划里的数量 (300 / 250 / 250、≤80 张等) 都是 spec §5.2 范围内由我定下的值，按原样实现。

## Review Focus

1. **画师改名**：2024 年的 v1 数据和 2026 快照之间画师 tag 被改名。B1 画师必须仍能找到，且仍被排除在训练外。做法是按共同帖子 id 做多数映射 (Task 4 有测试)。
2. **不可能的原发日期**：
   - 非 snowflake 的小数字 id、Pixiv id 超出索引范围时，不能得到 1970 年之类的日期，也不能外推；
   - 原发时间晚于 Danbooru 上传时间的不采用 (Task 1 有测试)。
3. **候选为空、层为空**：
   - 某画师全部图同一天，或图太少，导致年代估计没有候选；
   - B2 的短职业画师没有 ≥6 年对。

   这两种情况都要返回 None / NaN 并计数，不能崩，也不能让 NaN 进入均值或 bootstrap (Task 10、11 有测试)。
4. **跨来源泄漏**：评测画师或 B3-P2 账号对应的 Pixiv 账号，以及 Danbooru 的评测画师，不能出现在任何训练 index 里 (Task 6、7 的 verify 步骤要断言不相交)。
5. **sealed 被误用**：`load_set('bench2_b1')` 等常规名字只能返回 dev / test 行。sealed 用单独的名字和目录，P0 不提取 sealed 的特征 (Task 8 有测试)。

---

## 文件结构

| 文件 | 职责 |
|-|-|
| `v3/data/dates.py` | 原发日期：Twitter snowflake、Pixiv 作品时间 (精确 / 单调插值)、合并取最早 |
| `v3/data/fetch_pixiv_s1.py` | 下载 `deepghs/pixiv_ai_s1-webp-4Mpixel` 的 json 索引和 tar，并校验 |
| `v3/data/pixiv_meta.py` | Pixiv 图片 → 作品 → 账号表；账号 ↔ Danbooru 画师唯一映射；可用作品表 |
| `v3/data/build_danbooru_artist_v2.py` | Danbooru2026 prepare (过滤、作品族去重、日期、标志位、改名映射) + build + verify |
| `v3/data/bench2_splits.py` | B1 / B2 / B3-P2 的画师划分与训练排除名单 |
| `v3/data/resize_parts.py` | 通用：顺序扫 tar → 缩图 → 写成大 tar + offset parquet |
| `v3/data/build_pixiv_artist.py` | pixiv-artist-v1 prepare + build + verify |
| `v3/data/build_index.py` (改) | 加 `build_danbooru_v2` 和 `build_pixiv_v1`，生成训练 index |
| `v3/data/bench2_sets.py` | 选 B 集合的图 (时间轴均匀取)，写 index.parquet，生成注入名单 (b / c / d) |
| `v3/eval/evalsets.py` (改) | 注册 bench2 集合；`load_index_set` 支持 tar pread |
| `v3/eval/bootstrap.py` | 按画师 bootstrap、配对 bootstrap、总体重加权权重 |
| `v3/eval/timestrata.py` | 每画师直方图 (时间层、跨内容、同角色 / 同作品负样本)、加权曲线指标、画师内 Spearman |
| `v3/eval/dateest.py` | `style_dates`、中位数基线、`conflict_scores`、年代指标 |
| `v3/eval/retrieval.py` (改) | 时间窗检索 `window_retrieval` |
| `v3/eval/clustering.py` (改) | `bcubed` 与混合池聚类 |
| `v3/eval/inject.py` | 清洗注入：生成文件夹 (4 类)、kNN 离群打分、AUROC |
| `v3/eval/bench2.py` | benchmark v2 编排：多模型、共享 bootstrap 抽样、阈值迁移、输出 JSON |
| `v3/scripts/bench2_eval.py` | CLI：读缓存特征，跑 `bench2.run`，打印 markdown 表 |
| `docs/csip_v4_p0_baseline_zh.md` | 检查点报告：集合构成 + 基线表 (不含画师名) |

路径常量 (各模块各自定义，值必须一致)：

```python
V4 = Path('/scratch/cxip_v4')
SNAP = '/scratch/anime_tagger/dl/nyanko_danbooru2026/posts-snapshot.parquet'
SHARDS = Path('/scratch/anime_tagger/shards')        # train_index.parquet: id, ds (0=danbooru-full-4mpixel-v1, 1=danbooru-newest-4mpixel-v1), shard
PIXIV_INDEX = Path('/scratch/pixiv_probe/pixiv_index')  # tables/{1..4}/{images,artworks}.parquet, users.parquet
PIXIV_RAW = V4 / 'data/pixiv-s1'
V1_PLAN = Path('/data/neko/cxip_v3/data/danbooru-artist-v1-work/plan.parquet')
KAL_CLASSES = '/data/neko/ext_models/kaloscope3_preview/hf/v1-artist-classifier/class_mapping.csv'
UNSEEN_V1 = Path('/data/neko/cxip_v3/data/csip_unseen_v1/index.parquet')
BENCH = V4 / 'data/bench2_v1'
```

---

### Task 1：原发日期校正 `v3/data/dates.py`

**Files：**
- Create: `v3/data/dates.py`
- Test: `v3/tests/test_dates.py`

**Interfaces：**
- Produces:
  - `twitter_time(source: pd.Series) -> pd.Series[datetime64[ns, UTC]]`：没有可用时间的为 NaT。
  - `pixiv_time(pixiv_ids: pd.Series, index_ids: np.ndarray, index_times: np.ndarray) -> tuple[pd.Series, pd.Series]`：返回 (时间, 来源)，来源为 `'pixiv'`、`'pixiv_interp'` 或 None。
  - `correct_dates(upload: pd.Series, candidates: list[tuple[pd.Series, pd.Series | str]]) -> pd.DataFrame[date, date_source]`
  - `to_year(ts: pd.Series) -> np.ndarray[float]`：小数年，即 year + (day_of_year − 1) / 365.25。
  - `to_day(ts: pd.Series) -> np.ndarray[int64]`：自 1970-01-01 起的天数。
  - `SOURCES = ('twitter', 'pixiv', 'pixiv_interp', 'danbooru_upload')`

- [ ] **Step 1：写失败测试**

```python
import numpy as np, pandas as pd
from v3.data import dates as D

def test_twitter_time():
    s = pd.Series(['https://twitter.com/foo/status/1234567890123456789',
                   'https://x.com/a/statuses/1500000000000000000/photo/1',
                   'https://twitter.com/foo/status/20',            # pre-snowflake id
                   'https://www.pixiv.net/artworks/123', None])
    t = D.twitter_time(s)
    assert t[0] == pd.Timestamp('2020-03-02 19:54:56.824', tz='UTC')
    assert t[1] == pd.Timestamp('2022-03-05 06:47:23.309', tz='UTC')
    assert t[2:].isna().all()

def test_pixiv_time_exact_interp_and_range():
    ids = np.array([100, 200, 300]); ts = pd.to_datetime(['2010-01-01', '2012-01-01', '2011-01-01'], utc=True).values
    t, src = D.pixiv_time(pd.Series([200, 150, 250, 50, 400, None]), ids, ts)
    assert t[0] == pd.Timestamp('2012-01-01', tz='UTC') and src[0] == 'pixiv'
    assert abs((t[1] - pd.Timestamp('2011-01-01', tz='UTC')).days) <= 1 and src[1] == 'pixiv_interp'
    assert t[2] == pd.Timestamp('2012-01-01', tz='UTC')          # running max makes times monotone
    assert t[3:].isna().all() and src[3:].isna().all()          # no extrapolation

def test_correct_dates_earliest_valid():
    up = pd.to_datetime(['2020-01-01', '2020-01-01', '2020-01-01'], utc=True).to_series().reset_index(drop=True)
    tw = pd.to_datetime(['2018-05-01', None, None], utc=True).to_series().reset_index(drop=True)
    px = pd.to_datetime(['2019-01-01', '2021-06-01', None], utc=True).to_series().reset_index(drop=True)
    out = D.correct_dates(up, [(tw, 'twitter'), (px, 'pixiv')])
    assert out.date_source.tolist() == ['twitter', 'danbooru_upload', 'danbooru_upload']   # pixiv later than upload is rejected
    assert out.date[0] == pd.Timestamp('2018-05-01', tz='UTC')

def test_year_and_day():
    ts = pd.to_datetime(['2020-01-01', '2020-07-02'], utc=True).to_series()
    assert np.allclose(D.to_year(ts), [2020.0, 2020 + 183 / 365.25])
    assert D.to_day(ts).tolist() == [18262, 18445]
```

- [ ] **Step 2：确认测试失败**
  - 运行：`CUDA_VISIBLE_DEVICES= PYTHONPATH=. $PY -m pytest v3/tests/test_dates.py -q`
  - 预期：`ModuleNotFoundError: v3.data.dates`

- [ ] **Step 3：实现**
  - **Twitter**：用正则 `(?:twitter\.com|x\.com)/[^/]+/status(?:es)?/(\d+)` 取 id。ms = (id >> 22) + 1288834974657。只接受 id > 2^32 且 ms < 1.83e12 的结果。
  - **Pixiv**：在 `index_ids` 中精确匹配到的直接用，来源记 `'pixiv'`。其余先把索引按 id 排序，时间取累计最大值使其单调，再在 [min, max] 区间内 `np.interp`，来源记 `'pixiv_interp'`；区间外为 NaT。
  - **合并**：候选时间早于 2005-01-01，或晚于 upload + 1 天，都视为无效。有效候选与 upload 一起取最早的；来源按取到的那个候选标注，取到 upload 时标 `'danbooru_upload'`。

- [ ] **Step 4：确认通过**
  - 运行同 Step 2，预期 4 passed。

- [ ] **Step 5：Commit** (需用户同意)

```bash
git add v3/data/dates.py v3/tests/test_dates.py && git commit -m "feat(v4): original-date correction (twitter snowflake / pixiv id interpolation)"
```

### Task 2：下载 Pixiv 数据 `v3/data/fetch_pixiv_s1.py`

**Files：**
- Create: `v3/data/fetch_pixiv_s1.py`
- Test: `v3/tests/test_fetch_pixiv.py`

**Interfaces：**
- Produces:
  - `PIXIV_RAW/images/NNNN.tar` 与 `NNNN.json`
  - `PIXIV_RAW/done.jsonl`
  - `parse_tar_index(json_path: Path) -> pd.DataFrame[image_id:int64, tar:str, offset:int64, size:int64, sha256:str]`
  - `image_table(raw: Path = PIXIV_RAW) -> pd.DataFrame`：合并全部 json，结果缓存到 `raw/image_table.parquet`。

- [ ] **Step 1：写失败测试** (只测纯函数，不联网)

```python
import json
from v3.data.fetch_pixiv_s1 import parse_tar_index

def test_parse_tar_index(tmp_path):
    p = tmp_path / '0007.json'
    p.write_text(json.dumps({'filesize': 10, 'files': {'43000.webp': {'offset': 1536, 'size': 82478, 'sha256': 'ab'},
                                                        'x.txt': {'offset': 0, 'size': 1, 'sha256': 'cd'}}}))
    df = parse_tar_index(p)
    assert df.to_dict('records') == [dict(image_id=43000, tar=str(tmp_path / '0007.tar'), offset=1536, size=82478, sha256='ab')]
```

- [ ] **Step 2：确认失败** (`ModuleNotFoundError`)

- [ ] **Step 3：实现**
  - 从 `~/.config/cxip/env.sh` 读取 endpoint 和 token，token 只经 stdin 的 `--config -` 传给 curl。
  - 列目录：`{endpoint}/api/datasets/deepghs/pixiv_ai_s1-webp-4Mpixel/tree/main/images`。
  - 下载：`.../resolve/main/images/{name}`，用 `curl -fL -C - --retry 5`，线程池 8 路并发 (不用 fork 进程池)。
  - 先下全部 `.json`，再下 `.tar`。每个 tar 下完后按 json 里的 `hash_lfs` 做 sha256 校验，通过才追加一行到 `done.jsonl`。重跑时跳过已完成的文件。
  - 只保留 `.webp` 成员；image_id 取文件名的数字部分。
  - `main()` 子命令：`index` (只下 json)、`tars`、`table`。

- [ ] **Step 4：确认测试通过**

- [ ] **Step 5：先拉索引，确认格式**
  - 运行：`CUDA_VISIBLE_DEVICES= PYTHONPATH=. $PY -m v3.data.fetch_pixiv_s1 index && ... table`
  - 预期：
    - `image_table.parquet` 行数约 3.05M；
    - 与 `/scratch/pixiv_probe/exist_ids.json` 的 id 集合完全相同 (脚本打印对比结果)。

- [ ] **Step 6：后台下载 tar** (约 360 GB)
  - 运行：`nohup ... fetch_pixiv_s1 tars > V4/logs/fetch_pixiv.log 2>&1 &`
  - 下载与 Task 3-6 并行；Task 7 开始前必须完成。
  - 完成标准：`done.jsonl` 行数等于 tar 数，且没有校验失败。

- [ ] **Step 7：Commit** (需用户同意)

### Task 3：Pixiv 元数据与账号映射 `v3/data/pixiv_meta.py`

**Files：**
- Create: `v3/data/pixiv_meta.py`
- Test: `v3/tests/test_pixiv_meta.py`

**Interfaces：**
- Consumes:
  - `fetch_pixiv_s1.image_table()`
  - `dates.to_year`、`dates.to_day`
- Produces：
  - `map_accounts(shared: pd.DataFrame[user_id, artist, n_shared], min_shared=3) -> pd.DataFrame[user_id, artist]`
    - 一个账号只有一个画师的 n_shared ≥ 3 时才映射；
    - 多个账号映射到同一画师是允许的。
  - `pixiv_medium(tags: list[str]) -> str`：取值 `'comic'`、`'mono'`、`'color'`。
    - 标签含 `漫画` 或 `4コマ` → comic；
    - 含 `モノクロ` 或 `白黒` → mono。
  - `prepare(out=V4/'data/pixiv-artist-v1-work')` 写出两张表：
    - `works.parquet`，每张可用图一行，列为：image_id、artwork_id、page (order)、user_id、t_year、date_day、date_source='pixiv'、x_restrict、medium、tar、offset、size、on_danbooru (bool)。
    - `accounts.parquet`，列为：user_id、artist (映射到的 Danbooru 画师当前名，没有则为 None)、n_shared、n_works (可用且不在 Danbooru 的作品数)、n_images、t_first、t_last、span、pixiv_only (bool：该账号没有任何作品在 Danbooru)。

- [ ] **Step 1：写失败测试**

```python
import pandas as pd
from v3.data.pixiv_meta import map_accounts, pixiv_medium

def test_map_accounts_unique_only():
    shared = pd.DataFrame(dict(user_id=[1, 2, 2, 3, 4], artist=['x', 'x', 'y', 'z', 'x'], n_shared=[3, 3, 4, 2, 5]))
    m = map_accounts(shared)
    assert dict(zip(m.user_id, m.artist)) == {1: 'x', 4: 'x'}   # 2 is one-to-many, 3 below threshold

def test_pixiv_medium():
    assert pixiv_medium(['オリジナル', '漫画']) == 'comic'
    assert pixiv_medium(['モノクロ']) == 'mono'
    assert pixiv_medium(['女の子']) == 'color'
```

- [ ] **Step 2：确认失败**

- [ ] **Step 3：实现**
  - `shared` 表：Danbooru 快照中满足 Global Constraints 过滤规则、且有 `pixiv_id` 的帖子，与 pixiv_index 的 artworks 按作品 id 连接，再按 (user_id, tag_string_artist) 统计不同作品数。
  - `on_danbooru` = artwork_id ∈ 快照 pixiv_id 集合。
  - 作品级过滤按 Global Constraints。

- [ ] **Step 4：确认测试通过**

- [ ] **Step 5：跑 prepare 并核对 spec §0.6 的量级**
  - 运行：`CUDA_VISIBLE_DEVICES= PYTHONPATH=. $PY -m v3.data.pixiv_meta prepare`
  - 预期 (打印的统计在 spec 数字 ±5% 内)：
    - 映射账号 ≈ 24,567，对应画师 ≈ 24,391；
    - pixiv_only 且 n_works ≥ 30 的账号 ≈ 746。

- [ ] **Step 6：Commit** (需用户同意)

### Task 4：Danbooru2026 prepare `v3/data/build_danbooru_artist_v2.py`

**Files：**
- Create: `v3/data/build_danbooru_artist_v2.py`
- Test: `v3/tests/test_danbooru_v2.py`

**Interfaces：**
- Consumes:
  - `dates.twitter_time`、`dates.pixiv_time`、`dates.correct_dates`、`dates.to_year`、`dates.to_day`
  - pixiv_index 的 artworks 表 (id, created_at)
- Produces：
  - `filter_posts(snap: pd.DataFrame) -> pd.DataFrame`：应用 Global Constraints 的 Danbooru 规则。
  - `dedupe_families(posts: pd.DataFrame) -> pd.DataFrame`：每个 (artist, family) 只保留 id 最小的一张；调用方已先筛成本地有图的帖子。
  - `v1_name_map(v1: pd.DataFrame[id, artist], snap: pd.DataFrame[id, tag_string_artist]) -> dict[str, str]`：v1 名 → 当前名，按共同帖子多数决定；没有共同帖子时映射到自身。
  - `medium_of(general: set[str]) -> str`：规则同 `gap_buckets.MEDIUM`，取值 `'comic'`、`'sketch'`、`'mono'`、`'color'`。
  - `prepare()` 在 `WORK = V4/'data/danbooru-artist-v2-work'` 下写出：
    - `posts.parquet`，列为：
      - id、artist (当前名)、family、ds、shard；
      - date、date_source、t_year、date_day、upload_day；
      - characters (list)、copyrights (list)、general (str)、medium、rating；
      - orig_width、orig_height、pixiv_id、n_artist_local；
      - 标志位：eval_0918_test、eval_test100、eval_public、eval_unseen_v1、in_v1、in_kaloscope。
    - `artists.parquet`，列为：artist、n_snapshot、n_local、t_first、t_last、span 及以上各标志位。
    - `prepare_stats.json`，内容含各 date_source 的占比。

- [ ] **Step 1：写失败测试**

```python
import pandas as pd
from v3.data import build_danbooru_artist_v2 as B

def _snap():
    return pd.DataFrame(dict(
        id=[1, 2, 3, 4, 5, 6], tag_count_artist=[1, 1, 2, 1, 1, 1], is_deleted=[False, False, False, True, False, False],
        file_ext=['jpg', 'png', 'jpg', 'jpg', 'gif', 'webp'], parent_id=[None, 1, None, None, None, None],
        tag_string=['a', 'b', 'c', 'd', 'e', 'ai-generated x'], tag_string_artist=['p', 'p', 'p q', 'p', 'p', 'p']))

def test_filter_posts_rules():
    assert B.filter_posts(_snap()).id.tolist() == [1, 2]

def test_dedupe_families_keeps_min_id():
    p = pd.DataFrame(dict(id=[2, 1, 7], artist=['p', 'p', 'p'], family=[1, 1, 7]))
    assert sorted(B.dedupe_families(p).id) == [1, 7]

def test_v1_name_map_follows_rename():
    v1 = pd.DataFrame(dict(id=[1, 2, 3, 9], artist=['old_name', 'old_name', 'old_name', 'gone']))
    snap = pd.DataFrame(dict(id=[1, 2, 3], tag_string_artist=['new_name', 'new_name', 'other']))
    assert B.v1_name_map(v1, snap) == {'old_name': 'new_name', 'gone': 'gone'}

def test_medium_of():
    assert B.medium_of({'comic', 'monochrome'}) == 'comic'
    assert B.medium_of({'greyscale'}) == 'mono'
    assert B.medium_of({'1girl'}) == 'color'
```

- [ ] **Step 2：确认失败**

- [ ] **Step 3：实现 `prepare()`**
  - **本地图**：`SHARDS/train_index.parquet` 的 (id, ds, shard)。只保留本地有图的帖子，再做 `dedupe_families`；n_artist_local 在去重之后计算。
  - **日期**：`correct_dates(created_at, [(twitter_time(source), 'twitter'), pixiv_time(pixiv_id, ...)])`。pixiv_time 的来源列逐行传入。
  - **画师范围**：只保留 n_artist_local ≥ 10 的画师，A6 消融要用 10-29 张这一档。
  - **评测标志位**：
    - 0918 测试画师 = `V1_PLAN` 中 `artist_0918_test` 为真的 v1 名；test100 = `split_0918 == 'test100'` 的 v1 名；public = `csip_eval_public_artist` 为真的 v1 名。三者都经 `v1_name_map` 转成当前名。
    - `eval_unseen_v1` 来自 `UNSEEN_V1.label_name`。
    - `in_v1`：当前名在映射后的 v1 名集合中。
    - `in_kaloscope`：当前名或任一映射前的 v1 名小写后出现在 `KAL_CLASSES` 中。
  - **upload_day**：Danbooru created_at 的日期 (天)。

- [ ] **Step 4：确认测试通过**

- [ ] **Step 5：跑 prepare 并检查**
  - 运行：`CUDA_VISIBLE_DEVICES= PYTHONPATH=. $PY -m v3.data.build_danbooru_artist_v2 prepare`
  - 预期：
    - eval_0918_test 画师数 = 798；如有缺失，打印缺失名单并停下报告；
    - eval_test100 画师数 = 100；
    - n_artist_local ≥ 30 的画师在 45k-62k 之间；
    - `date_source != 'danbooru_upload'` 的占比 ≥ 0.45；
    - 打印 `v1_name_map` 中多对一 (多个 v1 名 → 同一当前名，即 tag 合并) 的个数和例子 (spec §7 的重名 / 合并风险)。例子只打印到日志，不进仓库。

- [ ] **Step 6：Commit** (需用户同意)

### Task 5：benchmark v2 画师划分 `v3/data/bench2_splits.py`

**Files：**
- Create: `v3/data/bench2_splits.py`
- Test: `v3/tests/test_bench2_splits.py`

**Interfaces：**
- Consumes:
  - `WORK/artists.parquet` (Task 4)
  - `pixiv-artist-v1-work/accounts.parquet` (Task 3)
- Produces：
  - `span_bin(span: np.ndarray, edges: tuple) -> np.ndarray[int]`
  - `assign_splits(keys: np.ndarray, strata: np.ndarray, sizes: dict[str, int], seed: int, forced: dict[str, set] = {}) -> dict[key, split]`
    - 在每个层内按比例分配，强制项先占名额；
    - sizes 的总和等于 len(keys)。
  - `BENCH/splits.json`：
    - `{'b1': {artist: split}, 'b2': {artist: split}, 'b3p2': {user_id: split}, 'b3p1_accounts': {user_id: artist}, 'exclude_danbooru_artists': [...], 'exclude_pixiv_users': [...]}`
    - 同时写 `manifest.json`。

- [ ] **Step 1：写失败测试**

```python
import numpy as np
from v3.data.bench2_splits import assign_splits, span_bin

def test_assign_splits_sizes_forced_and_deterministic():
    keys = np.array([f'a{i}' for i in range(100)]); strata = np.arange(100) % 3
    forced = {'test': {'a0', 'a1', 'a2'}}
    s = assign_splits(keys, strata, dict(dev=40, test=30, sealed=30), seed=0, forced=forced)
    assert sorted(set(s.values())) == ['dev', 'sealed', 'test']
    assert [list(s.values()).count(k) for k in ('dev', 'test', 'sealed')] == [40, 30, 30]
    assert all(s[k] == 'test' for k in forced['test'])
    assert s == assign_splits(keys, strata, dict(dev=40, test=30, sealed=30), seed=0, forced=forced)
    for z in range(3):  # stratified: each stratum ~40/30/30
        sub = [s[k] for k in keys[strata == z]]
        assert abs(sub.count('dev') / len(sub) - 0.4) < 0.1

def test_span_bin():
    assert span_bin(np.array([0, 2.9, 3, 8, 20]), (3, 8)).tolist() == [0, 0, 1, 2, 2]
```

- [ ] **Step 2：确认失败**

- [ ] **Step 3：实现 `main()`** (种子都为 0)
  - **B1**：798 个 eval_0918_test 画师，sizes 为 dev 300、test 250、sealed = 余下。按 span 分层，edges (3, 8)。test100 的画师强制进 test。
  - **B2**：
    - 候选池：n_local ≥ 30，且非 in_v1、非 in_kaloscope、非 eval_unseen_v1、非 eval_public、非 eval_0918_test。
    - 按 span edges (2, 5) 每层抽 200 人；某层不足时，由相邻层补足到总数 600。
    - 再按同一分层划成 200 / 200 / 200。
    - 年份不设配额 (spec 写的是 "按年份和职业跨度配额")，年份偏差由 §5.1 的总体重加权处理 (我的决定，因为 Kaloscope 未见画师的年份分布本身就偏新)。
  - **B3-P2**：
    - 候选池：`accounts` 中 pixiv_only 且 n_works ≥ 30 的账号。
    - 抽 300 个，按 span edges (2, 4) 分层划成 100 / 100 / 100。
  - **b3p1_accounts**：artist 属于 B1 的映射账号，并且 n_works ≥ 1。
  - **exclude_danbooru_artists**：B1 ∪ B2 ∪ eval_public ∪ eval_unseen_v1 的画师当前名。
  - **exclude_pixiv_users**：映射到上述排除画师的账号 ∪ B3-P2 账号。
  - 写出前断言：B1、B2 两两不相交，且都不含 in_kaloscope 与 in_v1 冲突的项 (B2 只看其候选条件)。

- [ ] **Step 4：确认测试通过**

- [ ] **Step 5：生成 splits.json**
  - 运行：`CUDA_VISIBLE_DEVICES= PYTHONPATH=. $PY -m v3.data.bench2_splits`
  - 预期打印：
    - B1 dev 300 / test 250 / sealed 248；
    - B2 600 人；
    - B3-P2 300 个账号；
    - B3-P1 账号数接近 493 个 B1 画师。

- [ ] **Step 6：Commit** (需用户同意)

### Task 6：缩图提取 + `danbooru-artist-v2` 打包 + 训练 index

**Files：**
- Create: `v3/data/resize_parts.py`
- Modify: `v3/data/build_danbooru_artist_v2.py` (加 `jobs`、`build`、`verify` 子命令)
- Modify: `v3/data/build_index.py` (加 `build_danbooru_v2`)
- Test: `v3/tests/test_resize_parts.py`

**Interfaces：**
- Consumes:
  - `posts.parquet` (Task 4)
  - `splits.json` (Task 5)
- Produces：
  - `iter_tar_members(path: Path) -> Iterator[tuple[str, int, int]]`：(name, data_offset, size)。处理 GNU 长文件名 (L)、pax 头 (x，其中的 `path=` 记录作为下一个成员名；Python tarfile 默认用这种格式写长文件名) 和 base-256 size，其余逻辑同 `build_csip_unseen.read_members`。
  - `resize_image(data: bytes, short_side=640, quality=95) -> tuple[bytes, int, int]`
  - `process_tar(src: Path, want: dict[str, str], out_tar: Path) -> pd.DataFrame[key, offset, size, width, height, error]`
    - `want`：成员名 → 输出 key；
    - 输出成员名为 `{key}.webp`；
    - 先写 `.tmp` 再原子改名，`out_tar.with_suffix('.parquet')` 同时写出；
    - 已存在则跳过。
  - `run_jobs(jobs: pd.DataFrame[src_tar, member, key], out_dir: Path, workers: int) -> pd.DataFrame`：合并后的 parts 表。
  - `danbooru-artist-v2` 数据集：
    - neko-data，单 split `all`，画师连续，classes.json 按名字排序；
    - 每样本的 metadata 字段：label、artist、danbooru_id、family、date (ISO)、date_source、t_year、date_day、upload_day、characters、copyrights、general、medium、rating、orig_width、orig_height、n_artist_local，以及 Task 4 的全部标志位；
    - sample_key = str(id)。
  - index 文件，三个都含上述 metadata 列：
    - `index/all.parquet`
    - `index/train_ge30.parquet`：artist ∉ exclude_danbooru_artists，且 n_artist_local ≥ 30
    - `index/train_ge10.parquet`：同上，阈值改为 ≥ 10

- [ ] **Step 1：写失败测试**

```python
import io, tarfile
import numpy as np
from PIL import Image
from v3.data.resize_parts import iter_tar_members, resize_image, process_tar

def _webp(w, h, mode='RGB'):
    b = io.BytesIO(); Image.new(mode, (w, h), (10, 20, 30, 128)[:len(mode)]).save(b, 'WEBP'); return b.getvalue()

def _tar(path, items):
    with tarfile.open(path, 'w') as t:
        for name, data in items:
            ti = tarfile.TarInfo(name); ti.size = len(data); t.addfile(ti, io.BytesIO(data))

def test_resize_rules():
    b, w, h = resize_image(_webp(1000, 2000)); assert (w, h) == (640, 1280)
    b, w, h = resize_image(_webp(300, 200)); assert (w, h) == (300, 200)
    b, w, h = resize_image(_webp(800, 800, 'RGBA')); assert Image.open(io.BytesIO(b)).mode == 'RGBA'

def test_iter_tar_members_offsets(tmp_path):
    a, c = _webp(10, 10), _webp(20, 20)
    long = 'x' * 150 + '.webp'
    _tar(tmp_path / 's.tar', [('1.webp', a), (long, c)])
    raw = (tmp_path / 's.tar').read_bytes()
    got = {n: raw[o:o + s] for n, o, s in iter_tar_members(tmp_path / 's.tar')}
    assert got['1.webp'] == a and got[long] == c

def test_process_tar_roundtrip(tmp_path):
    _tar(tmp_path / 's.tar', [('5.webp', _webp(1280, 960)), ('6.webp', _webp(64, 64))])
    df = process_tar(tmp_path / 's.tar', {'5.webp': '5'}, tmp_path / 'out' / 'p.tar')
    assert df.key.tolist() == ['5'] and df.error.isna().all()
    raw = (tmp_path / 'out' / 'p.tar').read_bytes()
    r = df.iloc[0]
    assert Image.open(io.BytesIO(raw[r.offset:r.offset + r['size']])).size == (853, 640)
```

- [ ] **Step 2：确认失败**

- [ ] **Step 3：实现 `resize_parts.py`**
  - 先把 P / LA / CMYK 等模式转成 RGB 或 RGBA。缩放用 BICUBIC，算法同 `build_csip_unseen.save`。
  - `run_jobs` 用 ProcessPool，进程数 = workers，每个源 tar 一个任务，按源 tar 顺序整读，不做随机读。
  - 每完成 1% 打印一次进度和 img/s。

- [ ] **Step 4：确认测试通过**

- [ ] **Step 5：在 `build_danbooru_artist_v2.py` 加子命令**
  - `jobs`：用 posts.parquet 生成 (src_tar, member=`{id}.webp`, key=`{id}`)。ds=0 的 src_tar 为 `danbooru-full-4mpixel-v1/wds/train/train-{shard:06d}.tar`，ds=1 为 `danbooru-newest-4mpixel-v1/...`。
  - `resize`：调用 `run_jobs`，输出到 `WORK/resized/`。
  - `build`：
    - 按 v1 `iter_samples` 的写法，用线程池 pread `WORK/resized/*.tar` 生成有序的 `NormalizedSample` 流；
    - `BuildJob` 参数同 v1，`dataset_id='cxip-danbooru-artist'`、`version='v2'`、`group_key='label'`、`resume=True`；
    - metadata 中 `split_columns` 列出全部标志位；
    - resize 出错的样本跳过，并记录到 `WORK/resize_errors.csv`。
  - `verify`：
    - 检查 manifest 与文件一致，样本数 = plan 中成功缩图的数量，画师连续；
    - 随机抽 2000 个样本，与 resized parts 做字节级比对。

- [ ] **Step 6：在 `build_index.py` 加 `build_danbooru_v2(root)`**
  - 写法同现有 `build_danbooru`，列表列为 `('characters', 'copyrights')`。
  - 写出 3 个 index 文件。
  - 断言 train 文件与 `splits.json['exclude_danbooru_artists']` 不相交。

- [ ] **Step 7：试跑**
  - 运行：`... build_danbooru_artist_v2 jobs && ... resize --limit-tars 4`
  - 预期：
    - 4 个 parts 生成，error 占比 < 0.1%；
    - 打印的速度 > 2000 img/s (workers=160)。

- [ ] **Step 8：全量**
  - 运行：`nohup ... resize --workers 160 && ... build && ... verify && $PY -m v3.data.build_index danbooru_v2 > V4/logs/dbv2.log 2>&1 &`
  - 预期：
    - verify 打印 OK；
    - train_ge30 在 4.5M-5.3M 张之间；
    - 不相交断言通过。

- [ ] **Step 9：Commit** (需用户同意)

### Task 7：`pixiv-artist-v1` 打包 + 训练 index

**Files：**
- Create: `v3/data/build_pixiv_artist.py`
- Modify: `v3/data/build_index.py` (加 `build_pixiv_v1`)
- Test: `v3/tests/test_pixiv_artist.py`

**Interfaces：**
- Consumes:
  - `works.parquet`、`accounts.parquet` (Task 3)
  - `splits.json` (Task 5)
  - `resize_parts.run_jobs` (Task 6)
- Produces：
  - `select_works(works, accounts, exclude_users: set) -> pd.DataFrame`：保留两类账号的不在 Danbooru 上的作品：
    - 映射账号 (artist 非 None)；
    - pixiv_only 且 n_works ≥ 30 的账号。

    命中 exclude_users 的行也保留 (B3 要用)，并把 `eval_excluded` 置为 True。
  - 类名：映射账号用 Danbooru 画师当前名，纯 Pixiv 账号用 `pixiv:{user_id}`。Plan 2 按类名与 danbooru-artist-v2 合并。
  - neko-data 数据集 `V4/data/pixiv-artist-v1`：
    - `dataset_id='cxip-pixiv-artist'`、`version='v1'`；
    - sample_key = `p{image_id}`；
    - metadata 字段：label、artist、user_id、artwork_id (= family)、page、date、date_source、t_year、date_day、medium、x_restrict、eval_excluded。
  - `index/all.parquet`；`index/train.parquet` (eval_excluded == False)。

- [ ] **Step 1：写失败测试**

```python
import pandas as pd
from v3.data.build_pixiv_artist import select_works

def test_select_works_rules():
    works = pd.DataFrame(dict(image_id=[1, 2, 3, 4, 5], artwork_id=[10, 10, 11, 12, 13], user_id=[7, 7, 7, 8, 9],
                              on_danbooru=[False, False, True, False, False]))
    accounts = pd.DataFrame(dict(user_id=[7, 8, 9], artist=['x', None, None], pixiv_only=[False, True, False],
                                 n_works=[2, 30, 40]))
    s = select_works(works, accounts, exclude_users={8})
    assert s.image_id.tolist() == [1, 2, 4]                  # on_danbooru dropped; user 9 neither mapped nor pixiv_only
    assert s.set_index('image_id').eval_excluded.to_dict() == {1: False, 2: False, 4: True}
    assert s.set_index('image_id').artist.to_dict() == {1: 'x', 2: 'x', 4: 'pixiv:8'}
```

- [ ] **Step 2：确认失败**

- [ ] **Step 3：实现 prepare / jobs / resize / build / verify**
  - 写法同 Task 6；源 tar 来自 `image_table()`。
  - 开始前检查 `PIXIV_RAW/done.jsonl` 是否覆盖了全部需要的 tar，缺了就退出并列出缺的 tar。

- [ ] **Step 4：确认测试通过**

- [ ] **Step 5：全量构建并验证**
  - 运行：`... build_pixiv_artist prepare && ... jobs && ... resize --workers 160 && ... build && ... verify && $PY -m v3.data.build_index pixiv_v1`
  - 预期：
    - verify 通过；
    - `train.parquet` 的 user_id 与 `exclude_pixiv_users` 不相交 (断言)；
    - 打印的类数中，纯 Pixiv 类 ≤ 746 − 300 (B3-P2 已排除)。

- [ ] **Step 6：Commit** (需用户同意)

### Task 8：B 集合打包、读取、注册

**Files：**
- Create: `v3/data/bench2_sets.py`
- Modify: `v3/eval/evalsets.py`：`INDEX_SETS`；`load_index_set` 在 df 有 `tar` 列时用 pread 读取。
- Test: `v3/tests/test_bench2_sets.py`

**Interfaces：**
- Consumes:
  - `danbooru-artist-v2/index/all.parquet`
  - `pixiv-artist-v1/index/all.parquet`
  - `splits.json`
- Produces：
  - `spread_pick(t_year: np.ndarray, n: int) -> np.ndarray[int]`：在时间排序后按 `np.linspace(0, len-1, n)` 取整、去重，结果是选中项在原数组中的下标。
  - 每个集合目录 `BENCH/{b1,b2,b3p1,b3p2}` 只放 dev 与 test 行；`BENCH/{name}_sealed` 只放 sealed 行。目录里有 `index.parquet` 与 `manifest.json`。
  - index 列：
    - 读取：path (= tar)、tar、offset、size；
    - 标识：sample_key (`d{id}` / `p{image_id}`)、label (0..n-1，集合内编号)、label_name、id、source (`danbooru` / `pixiv`)、split；
    - 时间：family、t_year、date_day、date_source；
    - 内容：characters、copyrights、medium、rating。
  - `evalsets.INDEX_SETS` 新增：
    - `bench2_b1`、`bench2_b2`、`bench2_b3p1`、`bench2_b3p2`；
    - 对应的 `*_sealed` 四项。
  - 每画师张数上限：B1 ≤ 80，B2 ≤ 60，B3-P2 ≤ 60 (每个作品只取 page 最小的一页)，B3-P1 ≤ 40 (同样每作品一页)。
  - B3-P1 的 label 与 label_name 用 B1 中对应画师的值，以便和 B1 拼接。

- [ ] **Step 1：写失败测试**

```python
import io, tarfile
import numpy as np, pandas as pd
from PIL import Image
from v3.data.bench2_sets import spread_pick
from v3.eval import evalsets

def test_spread_pick_covers_span():
    t = np.array([2010.0, 2010.1, 2010.2, 2015.0, 2020.0, 2020.1])
    idx = spread_pick(t, 3)
    assert sorted(t[idx]) == [2010.0, 2015.0, 2020.1]
    assert len(spread_pick(t, 10)) == 6

def test_load_index_set_pread(tmp_path, monkeypatch):
    b = io.BytesIO(); Image.new('RGB', (8, 8)).save(b, 'WEBP'); data = b.getvalue()
    with tarfile.open(tmp_path / 'a.tar', 'w') as t:
        ti = tarfile.TarInfo('1.webp'); ti.size = len(data); t.addfile(ti, io.BytesIO(data))
    d = tmp_path / 'set'; d.mkdir()
    pd.DataFrame(dict(path=[str(tmp_path / 'a.tar')], tar=[str(tmp_path / 'a.tar')], offset=[512], size=[len(data)],
                      label_name=['x'], label=[0], sample_key=['d1'], split=['dev'])).to_parquet(d / 'index.parquet')
    monkeypatch.setitem(evalsets.INDEX_SETS, 'tmp_set', d)
    df, read = evalsets.load_index_set('tmp_set')
    assert read(0).size == (8, 8)

def test_sealed_never_in_regular_sets():
    for name in ('bench2_b1', 'bench2_b2', 'bench2_b3p1', 'bench2_b3p2'):
        assert evalsets.INDEX_SETS[name] != evalsets.INDEX_SETS[name + '_sealed']
```

- [ ] **Step 2：确认失败**

- [ ] **Step 3：实现 `bench2_sets.py build`**
  - 按 Interfaces 选图，种子 0。
  - B1、B2 的画师在 v2 all 中有 < 30 张时照样收入 (全部取)，并在 manifest 里计数。
  - B1 的图只取 Danbooru。
  - B3-P1 = b3p1_accounts 中账号的、不在 Danbooru 上的作品。
  - manifest 内容：
    - 每个集合 × 划分的画师数、图片数；
    - span 的分位数、t_year 的分位数；
    - date_source 的占比；
    - 规则列表与种子。

- [ ] **Step 4：实现 `bench2_sets.py inject`**
  - 调用 Task 13 的 `inject.make_folders`，生成 B1、B2 (dev + test) 的三类文件夹：`same_char`、`far_period`、`bad_date`。
  - 写到 `BENCH/inject/{set}_{kind}.parquet`。
  - `near_style` 一类在 Task 15 补。
  - 本步依赖 Task 13，执行顺序见 Task 13 Step 5。

- [ ] **Step 5：确认测试通过**，并跑 build
  - 运行：`... -m v3.data.bench2_sets build`
  - 预期打印：
    - B1 约 5 万张 (≤ 80 × 798)；
    - B3-P1 约 5-7 千张；
    - `load_set('bench2_b1')` 返回的 split 只有 dev 和 test。

- [ ] **Step 6：Commit** (需用户同意)

### Task 9：按画师 bootstrap `v3/eval/bootstrap.py`

**Files：**
- Create: `v3/eval/bootstrap.py`
- Test: `v3/tests/test_eval_bootstrap.py`

**Interfaces：**
- Produces：
  - `draw_weights(n_artists: int, n_boot: int, seed: int) -> np.ndarray[(n_boot, n_artists), int32]`：每行是多项分布抽样的重数，行和 = n_artists。
  - `summarize(point: dict[str, float | None], draws: list[dict]) -> dict[str, dict]`：
    - 每个键输出 `{point, lo, hi, n_valid}`，lo / hi 为 2.5 / 97.5 分位；
    - 抽样结果为 None 或 NaN 时丢弃；
    - point 为 None 时，lo 和 hi 也为 None。
  - `artist_bootstrap(stat: Callable[[np.ndarray], dict], n_artists, n_boot=1000, seed=0, base_w=None) -> dict`
    - point = stat(base_w 或全 1)；
    - 每次抽样调用 stat(W[b] * base_w)。
  - `paired_bootstrap(stat_a, stat_b, n_artists, n_boot=1000, seed=0, base_w=None) -> dict`：对 b − a，输出 `{point, lo, hi, p_gt0}`。
  - `population_weights(span: np.ndarray, debut_year: np.ndarray, ref_span: np.ndarray, ref_year: np.ndarray, span_edges=(1, 3, 6), year_edges=(2013, 2018, 2022)) -> tuple[np.ndarray, float]`
    - 每画师权重 = p_ref(格) / p_set(格)，并归一化到均值 1；
    - 第二个返回值是集合中没有覆盖到的格在参考分布里的质量。

- [ ] **Step 1：写失败测试**

```python
import numpy as np
from v3.eval import bootstrap as BS

def test_draw_weights_are_multiplicities():
    W = BS.draw_weights(5, 20, 1)
    assert W.shape == (20, 5) and (W.sum(1) == 5).all() and (W >= 0).all()

def test_artist_bootstrap_mean():
    v = np.array([0., 1., 2., 3.])
    r = BS.artist_bootstrap(lambda w: {'m': float((w * v).sum() / w.sum())}, 4, n_boot=2000)
    assert r['m']['point'] == 1.5 and r['m']['lo'] < 1.5 < r['m']['hi'] and r['m']['n_valid'] == 2000

def test_paired_identical_is_zero():
    v = np.random.default_rng(0).random(30)
    f = lambda w: {'m': float((w * v).sum() / w.sum())}
    d = BS.paired_bootstrap(f, f, 30, n_boot=200)
    assert d['m']['point'] == 0 and d['m']['lo'] == 0 and d['m']['hi'] == 0

def test_none_handling():
    r = BS.artist_bootstrap(lambda w: {'x': None, 'y': 1.0}, 3, n_boot=10)
    assert r['x'] == dict(point=None, lo=None, hi=None, n_valid=0) and r['y']['point'] == 1.0

def test_population_weights():
    w, miss = BS.population_weights(np.array([0.5, 0.5, 10]), np.array([2020, 2020, 2020]),
                                    np.array([0.5, 10, 10, 10]), np.array([2020, 2020, 2020, 2010]))
    assert np.isclose(w.mean(), 1) and w[2] > w[0] and np.isclose(miss, 0.25)
```

- [ ] **Step 2：确认失败**
- [ ] **Step 3：实现**
- [ ] **Step 4：确认通过**
- [ ] **Step 5：Commit** (需用户同意)

### Task 10：时间分层验证 `v3/eval/timestrata.py`

**Files：**
- Create: `v3/eval/timestrata.py`
- Test: `v3/tests/test_eval_timestrata.py`

**Interfaces：**
- Consumes:
  - `pairwise._curves_from_counts`、`pairwise._summarize`
  - `bootstrap.artist_bootstrap`
- Produces：
  - 常量：
    - `STRATA = ('<1y', '1-3y', '3-6y', '>=6y')`
    - `EDGES = (1.0, 3.0, 6.0)`
    - `POS_GROUPS = STRATA + ('all', 'cross_content')`
    - `NEG_GROUPS = ('all', 'same_char', 'same_copy')`
    - `N_BINS = 1 << 14`
  - `@dataclass ArtistHists`：
    - `pos: np.ndarray[(A, len(POS_GROUPS), N_BINS), int64]`
    - `neg: np.ndarray[(A, len(NEG_GROUPS), N_BINS), int64]`
    - `scale: float`、`bias: float`
    - neg 中每个负对在两侧画师各计一次。
  - `tag_ids(lists: Sequence[list[str]], k=4, drop=('original',)) -> np.ndarray[(N, k), int32]`：每图最多 k 个标签 id，按全集频次从高到低取，用 -1 补齐。
  - `artist_histograms(feats, labels, t_year, family, chars=None, copys=None, scale=1.0, bias=0.0, rows=None, cols=None, chunk=2048, device='cpu') -> ArtistHists`
    - 计算 rows × cols 的对，rows / cols 是布尔掩码，默认全部。
    - B3-P1 的用法是 rows = Pixiv 查询、cols = Danbooru 库，只统计跨来源对。
    - 同一 family 的对不算正也不算负；自身对排除。
    - rows 与 cols 都为全部时，只计 i < j。
  - `curve_metrics(pos: np.ndarray[B], neg: np.ndarray[B], scale, bias, fprs=(1e-3, 1e-4)) -> dict | None`
    - 返回 pr_auc、ap、best_f1、best_threshold (score 单位)，以及 `recall@fpr{f}`；
    - 另有 `curve` 字段，格式与 `threshold_drift` 兼容；
    - pos 或 neg 总数为 0 时返回 None。
  - `stratified_metrics(h: ArtistHists, w: np.ndarray | None = None) -> dict`
    - 键为每个 POS_GROUPS 中的组，值为 curve_metrics(Σ w·pos[:, g], Σ w·neg[:, 0] / 2)，为空则是 None；
    - 另有三个顶层浮点键 (分母为 0 时为 None)，都在 all 组的 best 阈值下计算：
      - `cross_content_recall`：跨内容正样本的召回；
      - `same_char_fpr`、`same_copy_fpr`：对应负样本子集的 FPR。
  - `drift_spearman(feats, labels, t_year, family, min_images=10, min_span=3.0) -> np.ndarray`：每个合格画师一个值，即画师内对的 cos 与 |Δt| 的 Spearman；同 family 的对不计。

- [ ] **Step 1：写失败测试**

```python
import numpy as np
from v3.eval import timestrata as T
from v3.eval.pairwise import metrics_exact

def _toy():
    f = np.eye(4)[[0, 0, 0, 1, 1, 2, 2]] + 0.01 * np.random.default_rng(0).standard_normal((7, 4))
    lab = np.array([0, 0, 0, 1, 1, 2, 2]); t = np.array([2010, 2010.5, 2018, 2012, 2013.5, 2020, 2020.2])
    return f, lab, t

def test_counts_per_stratum():
    f, lab, t = _toy()
    h = T.artist_histograms(f, lab, t, np.arange(7), scale=10, bias=-5)
    P = T.POS_GROUPS
    assert h.pos[0, P.index('<1y')].sum() == 1 and h.pos[0, P.index('>=6y')].sum() == 2
    assert h.pos[1, P.index('1-3y')].sum() == 1 and h.pos[:, P.index('all')].sum() == 5
    assert h.neg[:, 0].sum() == 2 * (21 - 5)

def test_same_family_excluded():
    f, lab, t = _toy()
    fam = np.array([0, 0, 2, 3, 4, 5, 6])
    h = T.artist_histograms(f, lab, t, fam)
    assert h.pos[0, T.POS_GROUPS.index('<1y')].sum() == 0

def test_binned_matches_exact():
    rng = np.random.default_rng(1)
    lab = np.repeat(np.arange(10), 30); f = rng.standard_normal((300, 16)) + 3 * rng.standard_normal((10, 16))[lab]
    h = T.artist_histograms(f, lab, np.full(300, 2020.0), np.arange(300), scale=10, bias=-5)
    z = f / np.linalg.norm(f, axis=1, keepdims=True); iu = np.triu_indices(300, 1)
    ex = metrics_exact(10 * (z @ z.T)[iu] - 5, lab[iu[0]] == lab[iu[1]])
    assert abs(T.stratified_metrics(h)['all']['pr_auc'] - ex['pr_auc']) < 2e-3

def test_empty_stratum_is_none():
    f, lab, t = _toy()
    t = np.full(7, 2020.0)
    assert T.stratified_metrics(T.artist_histograms(f, lab, t, np.arange(7)))['>=6y'] is None

def test_content_groups():
    f, lab, t = _toy()
    chars = T.tag_ids([['a'], ['b'], ['a'], ['a'], [], [], []])
    h = T.artist_histograms(f, lab, t, np.arange(7), chars=chars, copys=T.tag_ids([[]] * 7))
    assert h.neg[:, T.NEG_GROUPS.index('same_char')].sum() == 2 * 2      # (0,3), (2,3)
    assert h.pos[0, T.POS_GROUPS.index('cross_content')].sum() == 2      # (0,1), (1,2)

def test_rows_cols_cross_source():
    f, lab, t = _toy()
    q = np.array([1, 0, 0, 1, 0, 0, 0], bool)
    h = T.artist_histograms(f, lab, t, np.arange(7), rows=q, cols=~q)
    assert h.pos[:, T.POS_GROUPS.index('all')].sum() == 2 + 1          # img0->{1,2}, img3->{4}
```

- [ ] **Step 2：确认失败**

- [ ] **Step 3：实现**
  - 特征先 L2 归一化。按行分块做 `z[rows_chunk] @ z[cols].T`，把 cos 映射成箱号：`bin = clip(((cos + 1) / 2 * N_BINS).long(), 0, N_BINS-1)`。
  - 正对用 `torch.bincount(label_i * (G * N_BINS) + g * N_BINS + bin)` 累加。负对和内容组用同样方式，按掩码分别累加。
  - 标签重叠：k × k 次逐元素相等比较，且要求 id ≥ 0。
  - `curve_metrics`：取非空箱、从高到低排列，交给 `_curves_from_counts` 和 `_summarize`。阈值 = 箱下沿换算成 score。权重可以是浮点数。

- [ ] **Step 4：确认通过**
- [ ] **Step 5：Commit** (需用户同意)

### Task 11：年代估计 `v3/eval/dateest.py`

**Files：**
- Create: `v3/eval/dateest.py`
- Test: `v3/tests/test_eval_dateest.py`

**Interfaces：**
- Produces：
  - `style_dates(feats, labels, t_year, day, family, k=5, temp=10.0, min_candidates=10, block=None, query=None, gallery=None) -> np.ndarray[float]`
    - 对每个查询图 i，候选 = 同画师、gallery 内、≠ i、day 不同、family 不同的图。block 不为 None 时 (形状 (N, k) 的标签 id)，再排除与 i 共享任一 id 的图。
    - 估计值 = top-k 余弦近邻日期的加权平均，权重 exp(temp · cos)。
    - 候选数 < min_candidates 时为 NaN；不在 query 中的图也为 NaN。
  - `median_baseline(labels, t_year, day, family, min_candidates=10, query=None, gallery=None) -> np.ndarray`：候选的日期中位数，候选规则同上。
  - `conflict_scores(est, t_claimed, labels, floor=0.25) -> np.ndarray`：|t − est| / max(同画师 median|t − est|, floor)；est 为 NaN 时结果也为 NaN。
  - `date_errors(est, t_true, reliable) -> np.ndarray`：|est − t|，reliable 为 False 或 est 为 NaN 的位置为 NaN。
  - `per_artist_mean(values, labels, n_artists) -> tuple[np.ndarray, np.ndarray]`：(每画师的和, 每画师的有效个数)，供加权 bootstrap 使用 (MAE = Σw·sum / Σw·cnt)。

- [ ] **Step 1：写失败测试**

```python
import numpy as np
from v3.eval import dateest as DE

def _toy():
    early, late = np.array([1., 0, 0]), np.array([0., 1, 0])
    f = np.stack([early, early * 1.01, early * 0.99, late, late * 1.01, late * 0.99]) + 1e-3
    t = np.array([2010.0, 2010.2, 2010.4, 2018.0, 2018.2, 2018.4])
    return f, np.zeros(6, int), t, np.arange(6), np.arange(6)

def test_style_dates_uses_similar_images():
    f, lab, t, day, fam = _toy()
    est = DE.style_dates(f, lab, t, day, fam, k=2, min_candidates=2)
    assert abs(est[0] - 2010.3) < 0.15 and abs(est[3] - 2018.3) < 0.15

def test_same_day_and_family_excluded():
    f, lab, t, day, fam = _toy()
    day = np.array([0, 0, 0, 1, 2, 3])
    assert np.isnan(DE.style_dates(f, lab, t, day, fam, k=2, min_candidates=4)[0])   # only 3 candidates left
    est = DE.style_dates(f, lab, t, day, fam, k=2, min_candidates=2)
    assert est[0] > 2015                                                                # same-day early images excluded

def test_block_and_query_gallery():
    f, lab, t, day, fam = _toy()
    block = np.array([[0], [0], [0], [-1], [-1], [-1]])
    assert DE.style_dates(f, lab, t, day, fam, k=2, min_candidates=2, block=block)[0] > 2015
    q = np.array([1, 0, 0, 0, 0, 0], bool)
    est = DE.style_dates(f, lab, t, day, fam, k=2, min_candidates=2, query=q, gallery=~q)
    assert np.isnan(est[1:]).all() and abs(est[0] - 2010.3) < 0.15

def test_conflict_and_baseline():
    est = np.array([2010., 2010., 2010., 2010.]); t = np.array([2010.2, 2009.9, 2010.1, 2016.0])
    s = DE.conflict_scores(est, t, np.zeros(4, int))
    assert s.argmax() == 3
    f, lab, t, day, fam = _toy()
    assert np.allclose(DE.median_baseline(lab, t, day, fam, min_candidates=2)[0], np.median(t[1:]))
```

- [ ] **Step 2：确认失败**

- [ ] **Step 3：实现**
  - 逐画师处理，每个画师内算一个 n_a × n_a 的 cos 矩阵 (B1 每人 ≤ 80 张，代价很小)。
  - B3-P1 的用法：把 B1 的 Danbooru 图和 B3-P1 的 Pixiv 图拼在一起，query = Pixiv、gallery = Danbooru。

- [ ] **Step 4：确认通过**

- [ ] **Step 5：复现 spec §0.3 的参考值** (CPU，用已缓存的特征)
  - 写一个一次性检查函数 `dateest.check_reference()`，`__main__` 中可调用。它在 test100_0918 上运行：
    - 日期 `t` 取 `style_drift.load_meta` 的 Danbooru created_at 小数年；day = created_at 的日期；family = arange；
    - 只算跨度 ≥ 3 年的画师；k=5，temp=10，min_candidates=10。
  - 运行：`CUDA_VISIBLE_DEVICES= PYTHONPATH=. $PY -m v3.eval.dateest check`
  - 预期：
    - v3_release MAE 0.95 ± 0.02；
    - kaloscope 0.89 ± 0.02；
    - baseline 2.51 ± 0.02；
    - 图片数 21,710。
  - 特征路径同 `/tmp/gran_sheets/dateknn.py` 中的 F 字典。

- [ ] **Step 6：Commit** (需用户同意)

### Task 12：时间窗检索 + BCubed 聚类

**Files：**
- Modify: `v3/eval/retrieval.py` (加 `window_retrieval`)
- Modify: `v3/eval/clustering.py` (加 `bcubed`、`mixed_pool_clustering`)
- Test: `v3/tests/test_eval_retrieval_window.py`、`v3/tests/test_eval_clustering.py` (追加)

**Interfaces：**
- Produces：
  - `window_retrieval(feats, labels, t_year, family, window=1.0, query=None, gallery=None, chunk=1024) -> dict[str, np.ndarray]`
    - 返回逐查询数组 `map_at_r`、`r1`，没有相关项的查询为 NaN。
    - 查询 i 的库 = gallery 中除自身、同 family，以及 "同画师且 |Δt| ≥ window" 之外的全部图。
    - 相关项 = 同画师、|Δt| < window。
    - window=None 时，同画师任意时期都算相关 ("认画师" 口径)。
  - `bcubed(y_true, y_pred) -> dict(precision, recall, f)`：噪声 (−1) 视为各自独立的单例；y_true 为 −1 的干扰图也是单例真类。
  - `mixed_pool_clustering(feats, labels, n_pools=5, n_artists=100, max_per_artist=40, n_distractors=1000, seed=0) -> dict`
    - 每个池：从集合中抽 n_artists 个画师作为真类，再从其余画师抽 n_distractors 张图，标签为 −1。
    - 两种模式：`reduce='umap'` 与 `reduce=None` (直接 HDBSCAN)，各自调用 `clustering_eval` 扫描 mcs，用 nmi 选出最佳 mcs。
    - 输出每种模式的 bcubed_f、ari 在各池上的均值和标准差。
    - 聚类不做 bootstrap，用池间标准差表示波动 (我的决定，聚类太慢)。

- [ ] **Step 1：写失败测试**

```python
import numpy as np
from v3.eval.retrieval import window_retrieval
from v3.eval.clustering import bcubed

def test_window_retrieval_excludes_far_same_artist():
    f = np.array([[1, 0], [1, 0.01], [0.9, 0.1], [0, 1.]])
    lab = np.array([0, 0, 0, 1]); t = np.array([2010, 2010.5, 2016, 2010])
    r = window_retrieval(f, lab, t, np.arange(4), window=1.0)
    assert r['map_at_r'][0] == 1.0 and r['r1'][0] == 1.0
    assert np.isnan(r['map_at_r'][2]) and np.isnan(r['map_at_r'][3])   # no in-window positives
    r2 = window_retrieval(f, lab, t, np.arange(4), window=None)
    assert r2['map_at_r'][2] == 1.0

def test_bcubed():
    assert bcubed([0, 0, 1, 1], [5, 5, 7, 7])['f'] == 1.0
    r = bcubed([0, 0, 1, 1], [3, 3, 3, 3])
    assert np.isclose(r['precision'], 0.5) and r['recall'] == 1.0
    assert bcubed([0, 0, -1], [1, 1, -1])['f'] == 1.0
```

- [ ] **Step 2：确认失败**
- [ ] **Step 3：实现** (检索部分照搬 `retrieval_eval` 的分块 topk 写法，掩码改为时间窗)
- [ ] **Step 4：确认通过** (原有的 `test_eval_clustering.py` 也要继续通过)
- [ ] **Step 5：Commit** (需用户同意)

### Task 13：清洗注入 `v3/eval/inject.py`

**Files：**
- Create: `v3/eval/inject.py`
- Test: `v3/tests/test_eval_inject.py`

**Interfaces：**
- Consumes:
  - `dateest.style_dates`、`dateest.conflict_scores`
  - `timestrata.tag_ids`
- Produces：
  - `KINDS = ('near_style', 'same_char', 'far_period', 'bad_date')`
  - `make_folders(df: pd.DataFrame, kind: str, frac=0.05, seed=0, feats: np.ndarray | None = None) -> pd.DataFrame[folder, sample_key, injected, t_claimed]`
    - 每个文件夹的注入数 = max(1, round(frac × 原始张数))；df 是一个 B 集合的 index (dev 和 test 一起)。
    - **near_style**：需要 feats。文件夹 = 画师 A 的全部图；注入 = 离 A 的特征质心最近的其他画师的图，每个画师最多取 1 张。
    - **same_char**：文件夹 = A 的全部图；注入 = 其他画师中带 A 最常见角色的图。A 没有角色标签时跳过该画师。
    - **far_period**：只用跨度 ≥ 7 年的画师。文件夹 = A 在图最多的 2 年窗口内的图 (要求 ≥ 10 张)；注入 = A 中距窗口中心 ≥ 6 年的图。
    - **bad_date**：只用跨度 ≥ 6 年的画师。文件夹 = A 的全部图；随机选注入图，把 t_claimed 改成距真值 ≥ 4 年、且仍在 A 的 [t_min, t_max] 内的值；能满足条件才改，否则换一张图。其余图的 t_claimed = t_year。
  - `knn_outlier_scores(feats, folders: pd.DataFrame, key_to_row: dict, k=10) -> np.ndarray`：每行的分数 = −(到文件夹内其他图的 top-k cos 均值)，分数越高越像离群。
  - `date_conflict_scores(feats, folders, meta: pd.DataFrame) -> np.ndarray`：style_dates 用 t_claimed 当日期、文件夹当画师算出，再交给 conflict_scores。
  - `folder_auroc(scores, folders) -> np.ndarray`：每个文件夹一个 AUROC，没有注入的文件夹为 NaN。指标 = 按画师加权的宏平均。

- [ ] **Step 1：写失败测试**

```python
import numpy as np, pandas as pd
from v3.eval import inject as I

def _df():
    rng = np.random.default_rng(0)
    n = 40
    t = np.r_[np.linspace(2010, 2011.5, 14), np.linspace(2017.5, 2020, 6), np.linspace(2012, 2014, 20)]
    return pd.DataFrame(dict(sample_key=[f'd{i}' for i in range(n)], label=np.repeat([0, 1], 20), t_year=t,
                             family=np.arange(n), date_day=(t * 365).astype(int),
                             characters=[['c1']] * 10 + [[]] * 10 + [['c1']] * 20, copyrights=[[]] * n))

def test_far_period_and_bad_date_rules():
    df = _df()
    fp = I.make_folders(df, 'far_period')
    win = fp[~fp.injected]; inj = fp[fp.injected]
    t = df.set_index('sample_key').t_year
    assert (t[win.sample_key].max() - t[win.sample_key].min()) <= 2
    assert len(inj) >= 1 and (abs(t[inj.sample_key].values - t[win.sample_key].mean()) >= 6).all()
    bd = I.make_folders(df, 'bad_date')
    m = bd[bd.injected]
    assert set(bd.folder) == {0}                                     # artist 1 spans 2y
    assert (abs(m.t_claimed.values - t[m.sample_key].values) >= 4).all()

def test_same_char_injects_other_artists():
    df = _df()
    sc = I.make_folders(df, 'same_char')
    lab = df.set_index('sample_key').label
    assert (lab[sc[sc.injected & (sc.folder == 0)].sample_key] == 1).all()

def test_auroc_perfect_when_outliers_far():
    f = np.r_[np.tile([1., 0], (19, 1)), [[0., 1]]] + 1e-3 * np.random.default_rng(0).standard_normal((20, 2))
    folders = pd.DataFrame(dict(folder=0, sample_key=[f'd{i}' for i in range(20)], injected=[False] * 19 + [True]))
    s = I.knn_outlier_scores(f, folders, {f'd{i}': i for i in range(20)}, k=5)
    assert I.folder_auroc(s, folders)[0] == 1.0
```

- [ ] **Step 2：确认失败**
- [ ] **Step 3：实现**
- [ ] **Step 4：确认通过**
- [ ] **Step 5：生成 b / c / d 三类注入名单** (即 Task 8 Step 4)
  - 运行：`... -m v3.data.bench2_sets inject`
  - 预期：每类在 B1 上都有 ≥ 100 个文件夹，数量打印出来。

- [ ] **Step 6：Commit** (需用户同意)

### Task 14：benchmark v2 编排与 CLI

**Files：**
- Create: `v3/eval/bench2.py`、`v3/scripts/bench2_eval.py`
- Test: `v3/tests/test_eval_bench2.py`

**Interfaces：**
- Consumes：Task 9-13 的全部函数，以及 `runner.load_set`、`pairwise.threshold_drift`
- Produces：
  - `@dataclass ModelFeats`：`name`、`feats: dict[str, np.ndarray]` (集合名 → 已与 `load_set` 的行对齐的特征)、`scale`、`bias`、`threshold` (默认阈值，score 单位)。
  - `evaluate_set(name: str, df: pd.DataFrame, models: list[ModelFeats], split: str, n_boot=1000, seed=0, reweight: tuple | None = None, inject_dir=BENCH/'inject') -> dict`
    - 先把 df 按 split 过滤。`ModelFeats.feats` 与过滤前的 df 逐行对齐，过滤时要用同一个掩码。
    - 年代指标的真值只用 `date_source != 'danbooru_upload'` 的图，候选近邻可以用全部图。只有上传日期的图另报 `date_mae_upload_only` (spec §4.1)。
    - 每个模型算一次 ArtistHists、逐查询检索值、逐图年代误差、注入 AUROC。
    - 所有模型共用同一组 `draw_weights`。
    - 输出：
      - `{model: {metric: {point, lo, hi, n_valid}}}`；
      - `'diff'`：`{f'{m}-vs-{models[0].name}': {metric: {point, lo, hi, p_gt0}}}`；
      - `'clustering'`：仅 split == 'dev' 时计算；
      - `'composition'`：画师数、图片数、各层的正对数。
    - metric 键名 (固定)：
      - `pr_auc/<group>`、`recall@1e-3/<stratum>`、`ap/<1y` (= µAP)、`ap/all`；
      - `map_at_r/1y`、`r1/1y`、`map_at_r/any`；
      - `drift_spearman_median`；
      - `date_mae`、`date_hit1y`、`date_mae_content`、`date_mae_baseline`、`date_mae_upload_only`；
      - `cross_content_recall`、`same_char_fpr`、`same_copy_fpr`；
      - `inject_auroc/<kind>`；
      - 传入 reweight 时，再给每个 pr_auc 加一份 `.../reweighted`。
  - `threshold_transfer(sets: dict[str, dict[model, ArtistHists]]) -> dict[model, dict]`
    - 对每个模型，用各集合的 all-组 `curve_metrics` 结果调用 `pairwise.threshold_drift`；
    - 另报告默认阈值下的 F1，以及 "源集合 FPR = 1e-3 的阈值" 迁移到目标集合后的实际 FPR；
    - `max_cross_f1_loss` 只在 {bench2_b1, bench2_b2, bench2_b3p2, csip_eval_public} 上取 (G2 口径)；
    - B3-P1 单独报告：用 B1 的 best 阈值在 B3-P1 跨来源对上算 F1 损失。
  - `run(models: list[ModelFeats], split='dev', sets=('bench2_b1', 'bench2_b2', 'bench2_b3p1', 'bench2_b3p2'), legacy=('test100_0918', 'csip_eval_public', 'csip_unseen_v1'), n_boot=1000) -> dict`
    - B3-P1 的评测组合：B1 + B3-P1，rows = Pixiv、cols = Danbooru。
    - B2 与 B3-P2 传入 reweight，参考分布 = v2 中 n_artist_local ≥ 30 的全部画师的 (span, 首年)。
    - legacy 集合只参与 threshold_transfer。
  - CLI：
    - 参数：`--model NAME FEATS_DIR SCALE BIAS THR` (可重复，第一个是参照模型)、`--split dev|test`、`--n-boot`、`--out json`、`--md md`；
    - 特征用 `csip_full_eval.load_aligned` 读取。

- [ ] **Step 1：写失败测试** (合成集合，monkeypatch 掉 `load_set` 和注入目录)

```python
import numpy as np, pandas as pd
from v3.eval import bench2 as B2

def _set(n_art=12, per=10, seed=0):
    rng = np.random.default_rng(seed)
    lab = np.repeat(np.arange(n_art), per)
    t = 2010 + rng.random(len(lab)) * 10
    df = pd.DataFrame(dict(sample_key=[f'd{i}' for i in range(len(lab))], label=lab, t_year=t, family=np.arange(len(lab)),
                           date_day=(t * 365.25).astype(int), date_source='twitter', characters=[[]] * len(lab),
                           copyrights=[[]] * len(lab), split='dev', source='danbooru'))
    f = rng.standard_normal((len(lab), 8)) + 3 * rng.standard_normal((n_art, 8))[lab]
    return df, f

def test_evaluate_set_shapes_and_self_diff(tmp_path):
    df, f = _set()
    m = B2.ModelFeats('a', {'s': f}, 10.0, -5.0, 0.0)
    r = B2.evaluate_set('s', df, [m, B2.ModelFeats('b', {'s': f}, 10.0, -5.0, 0.0)], 'dev', n_boot=50, inject_dir=tmp_path)
    x = r['a']['pr_auc/<1y']
    assert x['lo'] <= x['point'] <= x['hi']
    assert r['diff']['b-vs-a']['pr_auc/all']['point'] == 0
    assert 'date_mae' in r['a'] and 'map_at_r/1y' in r['a']
    assert r['composition']['n_artists'] == 12

def test_split_filter(tmp_path):
    df, f = _set()
    df.loc[df.label < 6, 'split'] = 'test'
    r = B2.evaluate_set('s', df, [B2.ModelFeats('a', {'s': f}, 10.0, -5.0, 0.0)], 'dev', n_boot=10, inject_dir=tmp_path)
    assert r['composition']['n_artists'] == 6
```

- [ ] **Step 2：确认失败**

- [ ] **Step 3：实现 `bench2.py`**
  - 注入目录里缺某类文件时，跳过该类并在输出中注明。
  - 所有 None 和 NaN 原样进入 summarize，由它处理。

- [ ] **Step 4：实现 CLI**
  - markdown 表的行 = 主指标，列 = 各模型，值为 point [lo, hi]，参照模型以外的列再加一列 diff。
  - 主指标：`pr_auc/<1y`、`pr_auc/1-3y`、`pr_auc/>=6y`、`map_at_r/1y`、`ap/<1y`、`date_mae`、`inject_auroc/bad_date`、`max_cross_f1_loss`、`same_char_fpr`。

- [ ] **Step 5：确认通过**，并跑全部测试：`$PY -m pytest v3/tests -q --ignore=v3/tests/smoke_train.py`，预期全部通过。

- [ ] **Step 6：Commit** (需用户同意)

### Task 15：基线、报告、检查点

**Files：**
- Create: `docs/csip_v4_p0_baseline_zh.md`
- Modify: `v3/data/bench2_sets.py` (`inject-near` 子命令)

**Interfaces：**
- Consumes:
  - `extract_release_feats`、`gap_feats exp`、`eval_kaloscope`
  - `bench2_eval` CLI
- Produces：
  - 特征目录：`V4/eval/bench2_v1/{v3_release, v3_x4, csip_v21}/feats/*.npz`
  - Kaloscope 的特征写到它的固定 OUT 目录 `/data/neko/cxip_v3/eval/kaloscope3/proj256/feats`。
  - 结果：`V4/eval/bench2_v1/results_{dev,test}.json/.md`
  - 报告文档

- [ ] **Step 1：申请 GPU**
  - 跑 `nvidia-smi`，把 4-7 号卡的占用发给用户，确认后再继续。
  - 当前 7 号卡有约 120 GB 别人的占用，默认只用 4、5、6 号卡。

- [ ] **Step 2：提取 4 个模型的特征** (每个模型一张卡，并行)
  - 集合：bench2_b1、bench2_b2、bench2_b3p1、bench2_b3p2。

```bash
S="bench2_b1 bench2_b2 bench2_b3p1 bench2_b3p2"
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. $PY -m v3.scripts.extract_release_feats --model /data/neko/cxip_v3/release/AnimeCSIP-v3 --sets $S --out-dir /scratch/cxip_v4/eval/bench2_v1/v3_release
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. $PY -m v3.scripts.gap_feats exp /data/neko/cxip_v3/exps/csip_p2/dft_r640_tag001_x4 /scratch/cxip_v4/eval/bench2_v1/v3_x4 --sets $S
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. $PY -m v3.scripts.eval_kaloscope --sets $S
```

  - 然后在 6 号卡上提取 AnimeCSIP-v2.1：`extract_release_feats --model dghstmp/AnimeCSIP-v2.1`，hub 访问照 env.sh 的方式。
    - 先用 `--sets bench2_b3p1` 试跑 1 批；
    - AutoModel 加载失败时，改用 `/data/neko/cxip_repro/CxIP/experiments/common/pair_eval.py` 的加载方式，写一个 encode 函数后调用 `runner.cached_features`。
  - 预期：每个 npz 的 sample_key 与 `load_set` 一致。`bench2_eval` 读取时由 `load_aligned` 断言。

- [ ] **Step 3：生成 near_style 注入名单**
  - 运行：`CUDA_VISIBLE_DEVICES= PYTHONPATH=. $PY -m v3.data.bench2_sets inject-near --feats /scratch/cxip_v4/eval/bench2_v1/csip_v21/feats`
  - 预期：写出 `BENCH/inject/bench2_b1_near_style.parquet` 和 `bench2_b2_near_style.parquet`。

- [ ] **Step 4：跑 dev 和 test 的基线** (CPU)
  - 参照模型 = v3 发布版，各模型参数：

    | 模型 | SCALE | BIAS | THR |
    |-|-|-|-|
    | v3 发布版 | 18.2881 | -9.875 | -0.5819 |
    | v3 x4 | 18.875 | -9.9375 | 0 |
    | Kaloscope | `kaloscope3/meta.json` | `kaloscope3/meta.json` | 0 |

  - 运行：`CUDA_VISIBLE_DEVICES= PYTHONPATH=. $PY -m v3.scripts.bench2_eval --model v3_release ... --model v3_x4 ... --model kaloscope ... --split dev --out .../results_dev.json --md .../results_dev.md`，test 同样跑一遍。
  - 合理性检查：
    - 三个模型在 B1 上都有 `pr_auc/<1y` > `pr_auc/>=6y`；
    - `date_mae_baseline` > 2 年；
    - v3 发布版在 test100_0918 上 legacy 部分的 best_f1 与 `/data/neko/cxip_v3/eval/csip_full/` 中的旧值相差 < 0.01 (分箱误差)。

- [ ] **Step 5：收集偏置数值**
  - I_content / I_char 来源：
    - v3 发布版：0.087 / 0.031 (spec §0.2)；
    - Kaloscope：`kaloscope3/bias_proj256.json`；
    - v3 x4：在其 exp 目录里找已有的 run_bias 结果，找不到时在 GPU 上用 `runner.run_bias(encode, 'csip')` 补跑一次。

- [ ] **Step 6：写 `docs/csip_v4_p0_baseline_zh.md`** (不含画师名，不放图片)，内容为：
  1. 数据 v2 与 Pixiv 的规模、date_source 占比；
  2. 各 B 集合按 split 的画师数、图片数、跨度分布，以及各时间层的正对数；
  3. dev 与 test 上三个模型的主指标表 (含 CI 和配对 diff)；
  4. 阈值迁移表 (G2 口径)；
  5. 偏置；
  6. 与 spec 的偏离 (例如画师缺失、数量与预期不符)，以及需要用户决定的事项。

  B1 上 Kaloscope 的分数要注明 "训练中见过这些画师" (spec §5.2)。

- [ ] **Step 7：检查点**
  - 把报告链接发给用户，等用户确认集合构成和基线后，才进入 Plan 2。P1 起才用 GPU 训练。

- [ ] **Step 8：Commit** (需用户同意)

---

## 后续计划 (P0 检查点后再写成完整计划)

**Plan 2：阶段 0、漂移估计、v4 训练 (spec §4.2-4.6、§6 的 P1-P3)**

| 部分 | 内容 |
|-|-|
| 数据合并 | `load_train_df` 支持 danbooru-artist-v2 + pixiv-artist-v1 按类名合并 |
| 采样器 | `PKBatchSampler` 加 "时间轴分散" 模式：K 张中一半在锚点 1 年内、一半更远；Pixiv 每作品每 epoch 最多 2 页 |
| 损失 | `GatheredSigLIP` 支持软目标：all-gather 每样本的 (label, t_year, family, curve_id)，按查找表算 y = r(\|Δt\|)，同族的对权重为 0 |
| 阶段 0 配置 | 1 年窗口，忽略远对 |
| 漂移估计 | `v3/eval/drift.py`：同日合并、仅彩色、内容控制、等渗回归、经验贝叶斯收缩，输出查找表 |
| 消融配置 | A0-A7 (A3g / A3a)，各约 1M 样本 @384 |
| 扩量与长训 | 扩量曲线和最终长训 |
| 评测接入 | 每个实验都用本计划的 `bench2_eval` 在 dev 上评测 |

**Plan 3：读出、发布评测、年代工具 (P4-P5)**
- CSLS / s-norm 可选读出；
- 按来源给推荐阈值；
- 跑一次 sealed；
- 只在需要时做事后擦除；
- 年代工具：`style_year`、`date_conflict`、每画师速度、阶段 tag，以及 CLI 和接触表 (只放 g/s)。工具复用本计划的 `dateest.style_dates` 与 `dateest.conflict_scores`。
