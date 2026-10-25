"""新旧 CCIP / CSIP 对比图 (docs/figures/)。数据全部读评测 JSON, 不手抄数字。

fig1_ccip_dumbbell   CCIP: HF AnimeCCIP → v3 teacher_v1 (开放集任务 / 按数据来源 / 旧闭集评测集 / 画风泄漏)
fig2_csip_dumbbell   CSIP: 旧版逐行最优 (v1/v2/v2.1) → v3 x2+擦除 (发布候选, 偏置已 ≤ v2.1)
fig3_csip_heatmap    CSIP: 7 个模型 × 全部指标, 逐行归一着色 (深 = 好), 饱和指标置灰
fig4_csip_tradeoff   CSIP: 精度 vs 内容/角色偏置
用法: python -m v3.scripts.plot_cxip_compare [--out docs/figures]
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

EV = Path('/data/neko/cxip_v3/eval')
EX = Path('/data/neko/cxip_v3/exps')
REPRO = Path('/data/neko/cxip_repro/results')

# 参考调色板 (dataviz/references/palette.md, light)
SURFACE, INK, INK2, MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#898781'
GRID, AXIS, NEUTRAL = '#e1e0d9', '#c3c2b7', '#f0efec'
NEW, OLD = '#2a78d6', '#a3a29b'   # v3 = 蓝 (强调色), 旧版 = 灰 (弱化)
RAMP = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf',
        '#1c5cab', '#184f95']


def setup_fonts():
    """matplotlib 只读 .ttc 的第一个字面 (JP 字形), 抽出 SC 字面存成 otf 再注册。"""
    cache = Path.home() / '.cache/cxip_fonts'
    for w in ('Regular', 'Bold'):
        dst = cache / f'NotoSansCJKsc-{w}.otf'
        if not dst.exists():
            from fontTools.ttLib import TTCollection
            cache.mkdir(parents=True, exist_ok=True)
            for f in TTCollection(f'/usr/share/fonts/opentype/noto/NotoSansCJK-{w}.ttc').fonts:
                if f['name'].getDebugName(1) == 'Noto Sans CJK SC':
                    f.save(dst)
        font_manager.fontManager.addfont(str(dst))
    plt.rcParams.update({
        'font.family': 'Noto Sans CJK SC', 'font.size': 10, 'axes.unicode_minus': False,
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE, 'svg.fonttype': 'none',
        'text.color': INK, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': INK,
        'axes.edgecolor': AXIS, 'axes.linewidth': 0.8, 'xtick.major.size': 0, 'ytick.major.size': 0,
    })


def J(p):
    return json.loads(Path(p).read_text())


# ---------------------------------------------------------------- 数据

def ccip_data():
    hf = J(EV / 'hf_ccip/suite.json')['datasets']['openset_full']
    v3s = J(EX / 'final/ccip_teacher_v1/eval/suite.json')
    v3 = v3s['datasets']['openset_full']
    cf, cu = J(EV / 'clustering_openset_full.json'), J(EV / 'clustering_openset_fast.json')

    def tasks(d, who):
        return [d['pr_auc'], d['best_f1'], d['zero_shot_ref1']['top1'], d['nway_A1']['closed_top1'],
                d['nway_A5']['closed_top1'], d['retrieval']['map_at_r'], d['retrieval']['recall_at_1'],
                d['ood']['N500_X500']['A1']['ood_auroc'], d['ood']['N500_X500']['A1']['ood_f1_best'],
                cf[who]['nmi'], cf[who]['ari'], cu[who]['ari']]
    task_names = ['验证 PR-AUC', '验证 best F1', '零样本识别 top1', '识别 top1 (1 张参考)', '识别 top1 (5 张参考)',
                  '检索 MAP@R', '检索 R@1', '未知角色拒识 AUROC', '未知角色拒识 F1', '聚类 NMI (FULL)',
                  '聚类 ARI (FULL)', '聚类 ARI (FAST)']
    src = ccip_source_split()
    legacy_v3 = J(EX / 'final/ccip_teacher_v1/eval_r512/suite.json')['datasets']
    legacy_hf = J(EV / 'hf_ccip_legacy/suite.json')['datasets'] | J(EV / 'hf_ccip_zc/suite.json')['datasets']
    leg = [('zc500x10', 'zc500x10 (500 类)'), ('zc50x15', 'zc50x15 (50 类)'), ('ccip_eval', 'ccip_eval (10 类)'),
           ('test_ccip', 'test_ccip (42 类)'), ('ccip_test_v1', 'ccip_test_v1 (43 类)')]
    hb = J(EV / 'hf_ccip/bias.json')
    return dict(
        tasks=(task_names, tasks(hf, 'HF'), tasks(v3, 'v3')),
        source=([f"{s['name']}  · {s['n_classes']} 类" for s in src], [s['HF'] for s in src], [s['v3'] for s in src]),
        legacy=([n for _, n in leg], [legacy_hf[k]['pr_auc'] for k, _ in leg], [legacy_v3[k]['pr_auc'] for k, _ in leg]),
        style=(['回归版', '分类版 (猜 100 个画师)'], [hb['style']['I'], hb['style_ce']['I']],
               [v3s['bias']['style']['I'], v3s['bias']['style_ce']['I']]))


def ccip_source_split():
    """开放集按数据来源拆分的 PR-AUC (HF AnimeCCIP 用 zerochan 训练, zerochan 来源的评测类它可能见过)。结果缓存。"""
    cache = EV / 'ccip_source_split.json'
    if cache.exists():
        return J(cache)
    from v3.eval.runner import load_set
    from v3.scripts.posthoc_erase import fast_ap
    meta = {c['label']: c for c in J(EV / 'ccip_openset_v1/heldout_classes.json')['classes']}
    df, _ = load_set('openset_full')
    lab = dict(zip(df['sample_key'].astype(str), df['label']))
    F = {}
    for n, p in [('HF', EV / 'hf_ccip/feats/openset_full.npz'), ('v3', EX / 'final/ccip_teacher_v1/eval/feats/openset_full.npz')]:
        d = np.load(p)
        F[n] = (d['feats'], np.array([lab[str(k)] for k in d['sample_key']]))
    zc = lambda c: 'zc800k' in meta[c]['sources']  # noqa: E731
    seen = lambda c: meta[c]['backbone_seen']  # noqa: E731
    groups = [('zerochan 来源 (HF 可能训练过)', lambda c: zc(c)), ('非 zerochan 来源 (HF 没训练过)', lambda c: not zc(c)),
              ('骨干见过 ∩ zerochan', lambda c: seen(c) == 'seen' and zc(c)),
              ('骨干见过 ∩ 非 zerochan', lambda c: seen(c) == 'seen' and not zc(c)),
              ('骨干没见过 ∩ zerochan', lambda c: seen(c) == 'unseen' and zc(c)),
              ('骨干没见过 ∩ 非 zerochan', lambda c: seen(c) == 'unseen' and not zc(c))]
    out = []
    for g, f in groups:
        r = dict(name=g)
        for n, (x, y) in F.items():
            m = np.array([f(c) for c in y])
            r[n] = fast_ap(x[m], y[m])
            r['n_classes'] = int(len(np.unique(y[m])))
        out.append(r)
    cache.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


CSIP_MODELS = [('HF v1', 'hf_AnimeCSIP'), ('HF v2', 'hf_AnimeCSIP-v2'), ('HF v2.1', 'hf_AnimeCSIP-v2.1'),
               ('v3 teacher_v1', 'v3_teacher_v1'), ('v3 x2', 'v3_x2'), ('v3 x4', 'v3_x4'), ('v3 x2+擦除', 'v3_x2_erased')]


def csip_data():
    R = {n: J(EV / f'csip_full/{f}.json') for n, f in CSIP_MODELS}
    hb = J(REPRO / 'bias_hf.json')
    bias = {'HF v1': hb['dghstmp/AnimeCSIP'], 'HF v2': hb['dghstmp/AnimeCSIP-v2'], 'HF v2.1': hb['dghstmp/AnimeCSIP-v2.1']}
    B = {k: (v['content']['I'], v['char']['I']) for k, v in bias.items()}
    for n, p in [('v3 teacher_v1', 'final/csip_teacher_v1'), ('v3 x2', 'csip_p2/dft_r640_tag001_x2'),
                 ('v3 x4', 'csip_p2/dft_r640_tag001_x4')]:
        b = J(EX / p / 'eval/suite.json')['bias']
        B[n] = (b['content']['I'], b['char']['I'])
    # x2+擦除: 事后擦除 kc3/kh4 在偏置探针留出划分上的 I (posthoc_erase 网格, 见结果文档)
    B['v3 x2+擦除'] = (0.087, 0.031)
    S = lambda n, s, k: R[n]['summary'][s][k]  # noqa: E731
    b0 = lambda n, k, f: R[n]['buckets_0918'][k][f]  # noqa: E731
    # (分组, 名称, 取值函数, 越高越好?, 饱和?)
    rows = []
    for s, tag in (('test100_0918', '0918'), ('csip_eval_public', 'public')):
        rows += [
            (tag, '验证 PR-AUC', lambda n, s=s: S(n, s, 'pr_auc'), True, False),
            (tag, '验证 best F1', lambda n, s=s: S(n, s, 'best_f1'), True, False),
            (tag, '画师识别 top1 (1 张参考)', lambda n, s=s: S(n, s, 'nway_A1'), True, False),
            (tag, '画师识别 top1 (5 张参考)', lambda n, s=s: S(n, s, 'nway_A5'), True, False),
            (tag, '检索 MAP@R', lambda n, s=s: S(n, s, 'map_at_r'), True, False),
            (tag, '检索 R@1', lambda n, s=s: S(n, s, 'r_at_1'), True, True),
            (tag, '聚类 NMI', lambda n, s=s: S(n, s, 'clus_nmi'), True, tag == 'public'),
            (tag, '聚类 ARI', lambda n, s=s: S(n, s, 'clus_ari'), True, tag == 'public'),
            (tag, '未知画师拒识 AUROC (1 张参考)', lambda n, s=s: S(n, s, 'ood_auroc_A1'), True, False),
            (tag, '未知画师拒识 F1 (1 张参考)', lambda n, s=s: S(n, s, 'ood_f1_A1'), True, False),
        ]
    rows += [
        ('难例 (0918)', '召回：年份差 ≥6 年', lambda n: b0(n, 'ygap/6+', 'recall'), True, False),
        ('难例 (0918)', '召回：彩色 vs 黑白', lambda n: b0(n, 'medium/color-mono', 'recall'), True, False),
        ('难例 (0918)', '召回：彩色 vs 漫画', lambda n: b0(n, 'medium/color-comic', 'recall'), True, False),
        ('难例 (0918)', '召回：彩色 vs 草图', lambda n: b0(n, 'medium/color-sketch', 'recall'), True, False),
        ('难例 (0918)', '误报：同角色不同画师', lambda n: b0(n, 'char/same', 'fp'), False, False),
        ('难例 (0918)', '误报：同作品不同画师', lambda n: b0(n, 'copyright/same', 'fp'), False, False),
        ('部署', '阈值迁移：共享阈值 F1 损失', lambda n: R[n]['threshold_drift']['shared_threshold_mean_f1_loss'], False, False),
        ('部署', '阈值迁移：最大交叉 F1 损失', lambda n: R[n]['threshold_drift']['max_cross_f1_loss'], False, False),
        ('内容偏置', 'I_content', lambda n: B[n][0], False, False),
        ('内容偏置', 'I_char', lambda n: B[n][1], False, False),
    ]
    names = [n for n, _ in CSIP_MODELS]
    table = [(g, lab, [f(n) for n in names], hb_, sat) for g, lab, f, hb_, sat in rows]
    return names, table, B, R


# ---------------------------------------------------------------- 画图部件

def dumbbell(ax, labels, old, new, title, xlim, lower_better=False, fmt='{:+.3f}', old_tag=None):
    """每行一个指标: 灰点 = 旧版, 蓝点 = v3, 右侧差值。lower_better 时差值按 "好" 的方向着正负号。"""
    y = np.arange(len(labels))[::-1]
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    for yy in y:
        ax.axhline(yy, color=GRID, lw=0.8, zorder=0)
    for yy, o, n in zip(y, old, new):
        ax.plot([o, n], [yy, yy], color=AXIS, lw=2.5, solid_capstyle='round', zorder=1)
        ax.scatter(o, yy, s=70, color=OLD, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.scatter(n, yy, s=70, color=NEW, edgecolor=SURFACE, linewidth=2, zorder=4)
    ax.set_yticks(y, labels)
    ax.tick_params(axis='y', pad=6, labelsize=9.5)
    ax.tick_params(axis='x', labelsize=8.5)
    ax.grid(axis='x', color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    # 差值列 (在轴外右侧, 文字用墨色; 变好加粗)
    for i, (yy, o, n) in enumerate(zip(y, old, new)):
        d = (o - n) if lower_better else (n - o)
        ax.annotate(fmt.format(d), xy=(1.0, yy), xycoords=('axes fraction', 'data'), xytext=(10, 0),
                    textcoords='offset points', va='center', ha='left', fontsize=9.5,
                    color=INK if d > 0 else INK2, fontweight='bold' if d > 0 else 'normal')
        if old_tag:
            ax.annotate(old_tag[i], xy=(o, yy), xytext=(0, 7), textcoords='offset points', ha='center',
                        fontsize=7.5, color=MUTED)
    ax.set_title(title, loc='left', fontsize=11.5, fontweight='bold', pad=10)


def figure_legend(fig, items, y=0.985):
    handles = [plt.Line2D([], [], marker='o', ls='', ms=9, mfc=c, mec=SURFACE, mew=2) for _, c in items]
    fig.legend(handles, [t for t, _ in items], loc='upper right', bbox_to_anchor=(0.985, y), ncol=len(items),
               frameon=False, fontsize=10, handletextpad=0.3, columnspacing=1.4)


# ---------------------------------------------------------------- 图

def note(ax, text):
    ax.text(0, -0.13, text, transform=ax.transAxes, fontsize=8.5, color=MUTED, va='top', ha='left', linespacing=1.5)


def fig1_ccip(d, out):
    fig = plt.figure(figsize=(14, 10))
    kw = dict(height_ratios=[12, 6], hspace=0.5, top=0.86, bottom=0.1)
    gl = fig.add_gridspec(2, 1, left=0.14, right=0.44, **kw)
    gr = fig.add_gridspec(2, 1, left=0.68, right=0.93, **kw)
    names, o, n = d['tasks']
    dumbbell(fig.add_subplot(gl[0]), names, o, n, 'a  开放集 FULL：1,002 个没训练过的角色', (0.2, 1.0))
    names, o, n = d['source']
    ax = fig.add_subplot(gr[0])
    dumbbell(ax, names, o, n, 'b  按数据来源拆分 (验证 PR-AUC)', (0.68, 1.0))
    ax.axhline(len(names) - 2.5, color=AXIS, lw=0.8, ls=(0, (2, 2)))
    note(ax, 'HF AnimeCCIP 用 zerochan 训练，zerochan 来源的评测类它可能见过；\n在它没训练过的子集上 v3 全面领先。')
    names, o, n = d['legacy']
    ax = fig.add_subplot(gl[1])
    dumbbell(ax, names, o, n, 'c  旧闭集评测集 (验证 PR-AUC)', (0.86, 1.005))
    note(ax, 'test_ccip / ccip_test_v1 的文件名是 zerochan id，HF 可能训练过 (零样本 100%)；\nv3 训练集与之 0 张重合。')
    names, o, n = d['style']
    ax = fig.add_subplot(gr[1])
    dumbbell(ax, names, o, n, 'd  画风泄漏 I_style (越低越好)', (0.0, 0.8), lower_better=True)
    note(ax, '同口径 9,509 张：从 CCIP 特征能猜出多少画师信息。\nv3 的画风泄漏高于旧版，是目前唯一退步的方向。')
    fig.text(0.02, 0.965, 'CCIP 新旧版对比：HF AnimeCCIP → v3 teacher_v1', fontsize=17, fontweight='bold')
    fig.text(0.02, 0.928, '同一套 v3 评测，两边各用自己的预处理与打分参数；右侧数字 = v3 相对旧版的变化 (加粗 = 变好)。',
             fontsize=9.5, color=INK2)
    figure_legend(fig, [('HF AnimeCCIP (旧)', OLD), ('v3 teacher_v1 @512', NEW)])
    save(fig, out / 'fig1_ccip_dumbbell')


def fig2_csip_dumbbell(names, table, out):
    old_idx, new_i = [0, 1, 2], names.index('v3 x2+擦除')
    pick = [('0918', '验证 PR-AUC'), ('0918', '画师识别 top1 (1 张参考)'), ('0918', '检索 MAP@R'), ('0918', '聚类 ARI'),
            ('0918', '未知画师拒识 AUROC (1 张参考)'),
            ('public', '验证 PR-AUC'), ('public', '画师识别 top1 (1 张参考)'), ('public', '检索 MAP@R'),
            ('public', '未知画师拒识 AUROC (1 张参考)'),
            ('难例 (0918)', '召回：年份差 ≥6 年'), ('难例 (0918)', '召回：彩色 vs 黑白'), ('难例 (0918)', '召回：彩色 vs 漫画'),
            ('难例 (0918)', '召回：彩色 vs 草图')]
    lower = [('难例 (0918)', '误报：同角色不同画师'), ('难例 (0918)', '误报：同作品不同画师'),
             ('部署', '阈值迁移：共享阈值 F1 损失'), ('内容偏置', 'I_content'), ('内容偏置', 'I_char')]
    T = {(g, lab): (v, hb) for g, lab, v, hb, _ in table}

    def build(keys):
        labs, old, new, tag = [], [], [], []
        for k in keys:
            v, hb = T[k]
            ov = [v[i] for i in old_idx]
            j = int(np.argmax(ov) if hb else np.argmin(ov))
            labs.append(f'{k[0]} · {k[1]}' if k[0] in ('0918', 'public') else k[1])
            old.append(ov[j]); new.append(v[new_i]); tag.append(names[old_idx[j]].replace('HF ', ''))
        return labs, old, new, tag
    fig = plt.figure(figsize=(13.5, 8.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.75, left=0.2, right=0.93, top=0.85, bottom=0.07)
    gr = gs[0, 1].subgridspec(2, 1, height_ratios=[2, 3], hspace=0.6)
    labs, o, n, tag = build(pick)
    ax = fig.add_subplot(gs[0, 0])
    dumbbell(ax, labs, o, n, 'a  精度与鲁棒性 (越高越好)', (0.4, 1.0), old_tag=tag)
    for yy in (len(labs) - 5.5, len(labs) - 9.5):
        ax.axhline(yy, color=AXIS, lw=0.8, ls=(0, (2, 2)))
    ax.text(0.41, 3.62, '0918 上的难例召回 ↓', fontsize=8.5, color=MUTED, va='bottom')
    labs, o, n, tag = build(lower[:2])
    dumbbell(fig.add_subplot(gr[0]), labs, o, n, 'b  困难负样本误报 (越低越好)', (0.0, 0.012), lower_better=True,
             fmt='{:+.4f}', old_tag=tag)
    labs, o, n, tag = build(lower[2:])
    dumbbell(fig.add_subplot(gr[1]), labs, o, n, 'c  部署与内容偏置 (越低越好)', (0.0, 0.14), lower_better=True,
             old_tag=tag)
    fig.text(0.02, 0.965, 'CSIP 新旧版对比：旧版逐项最优 → v3 x2 + 事后擦除', fontsize=17, fontweight='bold')
    fig.text(0.02, 0.928, '灰点 = HF CSIP v1 / v2 / v2.1 在该项上最好的一个 (点上方小字标明版本)；蓝点 = v3 发布候选 '
             '(640 续训 1.2M + 内容去偏 + 3/4 维线性擦除，偏置已低于 v2.1)。v3 用整图推理，HF 用各自预处理。',
             fontsize=9.5, color=INK2)
    figure_legend(fig, [('HF CSIP 旧版 (逐项最优)', OLD), ('v3 x2+擦除', NEW)])
    save(fig, out / 'fig2_csip_dumbbell')


GROUP_NAMES = {'0918': '0918 · Danbooru 100 画师', 'public': 'public · 122 画师', '难例 (0918)': '0918 难例'}


def fig3_csip_heatmap(names, table, out):
    nr, nc = len(table), len(names)
    fig = plt.figure(figsize=(12.5, 0.36 * nr + 2.6))
    H = 0.36 * nr + 2.0
    fig.set_size_inches(12.5, H)
    ax = fig.add_axes([0.27, 0.02, 0.70, 1 - 0.02 - 1.55 / H])
    ax.set_xlim(0, nc)
    ax.set_ylim(nr, 0)
    ax.axis('off')
    gap = 0.04
    prev_group = None
    for i, (g, lab, v, hb, sat) in enumerate(table):
        v = np.array(v, float)
        if sat:
            score = np.full(nc, np.nan)
        else:
            lo, hi = np.nanmin(v), np.nanmax(v)
            score = (v - lo) / (hi - lo) if hi > lo else np.ones(nc)
            if not hb:
                score = 1 - score
        best = np.nanargmax(v) if hb else np.nanargmin(v)
        digits = 4 if np.nanmax(np.abs(v)) < 0.05 else 3
        for j in range(nc):
            if np.isnan(score[j]):
                fc, tc = NEUTRAL, MUTED
            else:
                k = int(round(score[j] * (len(RAMP) - 1)))
                fc = RAMP[k]
                tc = '#ffffff' if k >= 6 else INK
            ax.add_patch(FancyBboxPatch((j + gap, i + gap), 1 - 2 * gap, 1 - 2 * gap,
                                        boxstyle='round,pad=0,rounding_size=0.08', fc=fc, ec='none'))
            txt = f'{v[j]:.{digits}f}'
            ax.text(j + 0.5, i + 0.52, txt, ha='center', va='center', fontsize=9, color=tc,
                    fontweight='bold' if (j == best and not sat) else 'normal')
        arrow = '↑' if hb else '↓'
        ax.text(-0.12, i + 0.52, f'{lab}  {arrow}' + ('  (饱和)' if sat else ''), ha='right', va='center',
                fontsize=9.5, color=MUTED if sat else INK)
        if g != prev_group:
            if prev_group is not None:
                ax.plot([-2.55, nc], [i, i], color=AXIS, lw=0.8, clip_on=False)
            n_g = sum(1 for t in table if t[0] == g)
            ax.text(-2.45, i + n_g / 2, GROUP_NAMES.get(g, g), ha='center', va='center', rotation=90, fontsize=10.5,
                    fontweight='bold', color=INK2)
            prev_group = g
    for j, n in enumerate(names):
        ax.text(j + 0.5, -0.35, n.replace('v3 ', 'v3\n').replace('HF ', 'HF\n'), ha='center', va='bottom',
                fontsize=9.5, fontweight='bold', color=INK if n.startswith('v3') else INK2)
    ax.plot([3, 3], [-0.1, nr], color=AXIS, lw=1.2, clip_on=False)
    top = 1 - 0.3 / H
    fig.text(0.02, top, 'CSIP 七个模型 × 全部评测指标', fontsize=17, fontweight='bold', va='top')
    fig.text(0.02, top - 0.5 / H, '每行单独着色：颜色越深该项越好 (↑ 越高越好，↓ 越低越好)，加粗 = 该行最优；'
             '灰底 = 已饱和、区分不了模型的指标。v3 用整图推理，HF 用各自预处理。', fontsize=9.5, color=INK2, va='top')
    save(fig, out / 'fig3_csip_heatmap')


def fig4_csip_tradeoff(names, table, B, out):
    T = {(g, lab): v for g, lab, v, _, _ in table}
    acc = (np.array(T[('0918', '验证 PR-AUC')]) + np.array(T[('public', '验证 PR-AUC')])) / 2
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.13, wspace=0.22)
    off = {'HF v1': (-10, 10), 'HF v2': (8, 4), 'HF v2.1': (8, -12), 'v3 teacher_v1': (8, 6), 'v3 x2': (-6, 10),
           'v3 x4': (8, -11), 'v3 x2+擦除': (8, -12)}
    for ax, k, xl, ref in ((axes[0], 0, 'I_content (内容标签偏置，越低越好)', 0.091),
                           (axes[1], 1, 'I_char (角色偏置，越低越好)', 0.036)):
        x = np.array([B[n][k] for n in names])
        ax.axvline(ref, color=AXIS, lw=1.2, ls=(0, (4, 3)), zorder=0)
        ax.text(ref, 0.735, ' HF v2.1 偏置水平', fontsize=8.5, color=MUTED, va='bottom')
        v3m = np.array([n.startswith('v3') for n in names])
        ax.plot(x[~v3m], acc[~v3m], color=OLD, lw=1, alpha=0.5, zorder=1)
        ax.set_xlim(x.min() - 0.08 * np.ptp(x), x.max() + 0.06 * np.ptp(x))
        ax.plot(x[v3m][[0, 1, 2]], acc[v3m][[0, 1, 2]], color=NEW, lw=1, alpha=0.35, zorder=1)
        ax.annotate('', xy=(x[names.index('v3 x2+擦除')], acc[names.index('v3 x2+擦除')]),
                    xytext=(x[names.index('v3 x2')], acc[names.index('v3 x2')]),
                    arrowprops=dict(arrowstyle='-|>', color=NEW, lw=1.2, alpha=0.6, shrinkA=7, shrinkB=7), zorder=1)
        for i, n in enumerate(names):
            c = NEW if n.startswith('v3') else OLD
            ax.scatter(x[i], acc[i], s=110 if n == 'v3 x2+擦除' else 75, color=c, edgecolor=SURFACE, linewidth=2,
                       marker='D' if n == 'v3 x2+擦除' else 'o', zorder=3)
            ax.annotate(n, (x[i], acc[i]), xytext=off[n], textcoords='offset points', fontsize=9.5,
                        ha='right' if off[n][0] < 0 else 'left', va='center', color=INK,
                        fontweight='bold' if n == 'v3 x2+擦除' else 'normal')
        ax.set_xlabel(xl, fontsize=10)
        ax.set_ylim(0.735, 0.915)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=9)
        ax.text(0.98, 0.97, '← 更干净    更准 ↑', transform=ax.transAxes, ha='right', va='top', fontsize=9, color=MUTED)
    axes[0].set_ylabel('验证 PR-AUC (0918 与 public 平均)', fontsize=10)
    fig.text(0.02, 0.95, 'CSIP 精度 vs 内容偏置', fontsize=17, fontweight='bold')
    fig.text(0.02, 0.89, '旧版 v2 → v2.1 拿精度换偏置；v3 同时更准、更干净。箭头 = 事后线性擦除 (x2 → x2+擦除)：'
             '精度 −0.02，两项偏置都降到 v2.1 以下。', fontsize=9.5, color=INK2)
    figure_legend(fig, [('HF CSIP 旧版', OLD), ('v3', NEW)], y=0.97)
    save(fig, out / 'fig4_csip_tradeoff')


def save(fig, stem):
    fig.savefig(f'{stem}.png', dpi=200)
    fig.savefig(f'{stem}.svg')
    plt.close(fig)
    print('wrote', stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/data/neko/cxip/docs/figures')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    setup_fonts()
    fig1_ccip(ccip_data(), out)
    names, table, B, _ = csip_data()
    fig2_csip_dumbbell(names, table, out)
    fig3_csip_heatmap(names, table, out)
    fig4_csip_tradeoff(names, table, B, out)


if __name__ == '__main__':
    main()
