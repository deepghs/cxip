"""新旧 CCIP / CSIP 展示图 (docs/figures/showcase_*)。数据与 plot_cxip_compare 同源 (评测 JSON)。

showcase_1_overview    错误率相对下降, 按幅度排序 (CCIP: HF → v3; CSIP: 旧版逐项最优 → v3 发布版), 退步项同图保留
showcase_2_csip_radar  CSIP 4 个版本雷达图 (每轴在 4 个模型间归一)
showcase_3_error_bars  关键任务错误率柱状图, v3 柱上标相对下降
showcase_4_pareto      CSIP 精度 vs 内容 / 角色偏置
"错误率" = 1 − 指标 (越高越好的指标) 或指标本身 (误报率、偏置、阈值损失)。相对下降 = (旧 − 新) / 旧。
用法: python -m v3.scripts.plot_cxip_showcase [--out docs/figures]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from v3.scripts.plot_cxip_compare import ccip_data, csip_data, setup_fonts  # noqa: E402

WHITE, INK, INK2, MUTED = '#ffffff', '#0b0b0b', '#52514e', '#898781'
GRID, AXIS = '#ebeae5', '#c3c2b7'
BLUE, BLUE_L, BLUE_XL = '#2a78d6', '#86b6ef', '#e8f1fc'
ORANGE, ORANGE_L = '#eb6834', '#f6c3ad'
OLDS = {'HF v1': '#dcdbd4', 'HF v2': '#bdbcb4', 'HF v2.1': '#8f8e87'}   # 旧版: 浅 → 深灰, 按发布顺序
REL = 'v3 x2+擦除'                                                         # CSIP 发布候选
REL_NAME = 'v3 (发布版)'


def style():
    setup_fonts()
    plt.rcParams.update({'figure.facecolor': WHITE, 'axes.facecolor': WHITE, 'savefig.facecolor': WHITE,
                         'axes.edgecolor': AXIS, 'xtick.color': MUTED, 'ytick.color': INK})


def err(v, hb):
    return 1 - v if hb else v


def rel_drop(old, new, hb):
    eo, en = err(old, hb), err(new, hb)
    return (eo - en) / eo


def save(fig, stem):
    fig.savefig(f'{stem}.png', dpi=220)
    fig.savefig(f'{stem}.svg')
    plt.close(fig)
    print('wrote', stem)


def clean(ax, left=False):
    for s in ('top', 'right', 'left', 'bottom'):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def pill(ax, x, y, text, fc, tc=WHITE, fs=8.5, transform=None, ha='center'):
    ax.text(x, y, text, ha=ha, va='center', fontsize=fs, color=tc, fontweight='bold', zorder=6,
            transform=transform or ax.transData,
            bbox=dict(boxstyle='round,pad=0.28,rounding_size=0.75', fc=fc, ec='none'))


# ---------------------------------------------------------------- 数据整理 (任务名 + 一句说明)
# 每项: (长名称, 雷达短名称, 各模型取值, 越高越好?, 'err' 错误率类 / 'fp' 误报率 / 'bias' 偏差分)
# CCIP 取值顺序 [旧版, v3]; CSIP 取值顺序 [v1, v2, v2.1, v3], 两个测试集 (0918 / public) 先取平均。

def ccip_spec(d):
    tn, to, tv = d['tasks']
    ti = {k: i for i, k in enumerate(tn)}
    so, sv = d['source'][1], d['source'][2]
    lo, lv = d['legacy'][1], d['legacy'][2]
    sto, stv = d['style'][1], d['style'][2]

    def t(k):
        return [to[ti[k]], tv[ti[k]]]
    return [
        ('验证：两张图是否同一角色', '', t('验证 PR-AUC'), True, 'err'),
        ('识别 (1-shot)：给 1 张参考图认出角色', '', t('识别 top1 (1 张参考)'), True, 'err'),
        ('识别 (5-shot)：给 5 张参考图认出角色', '', t('识别 top1 (5 张参考)'), True, 'err'),
        ('检索 R@1：以图搜图，首个结果命中', '', t('检索 R@1'), True, 'err'),
        ('检索 MAP@R：同角色图片整体排序', '', t('检索 MAP@R'), True, 'err'),
        ('开集拒识：识别出未见过的角色', '', t('未知角色拒识 AUROC'), True, 'err'),
        ('聚类：无标注图片自动按角色分组', '', t('聚类 ARI (FAST)'), True, 'err'),
        ('验证 · 旧版未训练过的 657 个角色', '', [so[1], sv[1]], True, 'err'),
        ('验证 · 旧版训练过的 345 个角色', '', [so[0], sv[0]], True, 'err'),
        ('旧版测试集 zc500x10 (500 个角色)', '', [lo[0], lv[0]], True, 'err'),
        ('旧版测试集 zc50x15 (50 个角色)', '', [lo[1], lv[1]], True, 'err'),
        ('画风泄漏：角色特征中混入的画风信息', '', [sto[0], stv[0]], False, 'bias'),
    ]


def csip_spec(names, table, B):
    T = {(g, lab): np.array(v, float) for g, lab, v, _, _ in table}
    mi = [names.index(m) for m in list(OLDS) + [REL]]

    def both(k):
        return ((T[('0918', k)] + T[('public', k)]) / 2)[mi]

    def hard(k):
        return T[('难例 (0918)', k)][mi]
    drift = T[('部署', '阈值迁移：共享阈值 F1 损失')][mi]
    drift_max = T[('部署', '阈值迁移：最大交叉 F1 损失')][mi]
    return [
        ('验证：两张图是否同一画师', '验证\n(是否同一画师)', both('验证 PR-AUC'), True, 'err'),
        ('识别 (1-shot)：给 1 张参考图认出画师', '识别\n(1 张参考图)', both('画师识别 top1 (1 张参考)'), True, 'err'),
        ('检索 MAP@R：以图搜同画师作品', '检索\n(以图搜图)', both('检索 MAP@R'), True, 'err'),
        ('开集拒识：识别出未见过的画师', '开集拒识\n(未见过的画师)', both('未知画师拒识 AUROC (1 张参考)'), True, 'err'),
        ('聚类：无标注图片自动按画师分组', '聚类\n(自动分组)', T[('0918', '聚类 ARI')][mi], True, 'err'),
        ('跨年份召回：同一画师相隔 6 年以上', '', hard('召回：年份差 ≥6 年'), True, 'err'),
        ('跨媒介召回：彩图 vs 黑白', '', hard('召回：彩色 vs 黑白'), True, 'err'),
        ('跨媒介召回：彩图 vs 漫画', '', hard('召回：彩色 vs 漫画'), True, 'err'),
        ('跨媒介召回：彩图 vs 草图', '', hard('召回：彩色 vs 草图'), True, 'err'),
        ('同角色误报：不同画师画同一角色', '同角色误报\n(不同画师画同一角色)', hard('误报：同角色不同画师'), False, 'fp'),
        ('同作品误报：不同画师画同一作品', '', hard('误报：同作品不同画师'), False, 'fp'),
        ('内容偏差：受画面内容 (标签) 影响', '内容偏差\n(受画面内容影响)', np.array([B[n][0] for n in names])[mi], False, 'bias'),
        ('角色偏差：受所画角色影响', '角色偏差\n(受所画角色影响)', np.array([B[n][1] for n in names])[mi], False, 'bias'),
        ('阈值迁移：换数据集沿用同一阈值的 F1 损失', '', drift, False, 'err'),
        ('阈值迁移：最差一组的 F1 损失', '', drift_max, False, 'err'),
    ]


def best_old(vals, hb):
    o = np.asarray(vals[:-1])
    j = int(np.argmax(o) if hb else np.argmin(o))
    return o[j], j


def fmt_val(v, hb, kind):
    if kind == 'bias':
        return f'{v:.3f}'
    e = err(v, hb) * 100
    return f'{e:.2f}%' if e < 1 else f'{e:.1f}%'


# ---------------------------------------------------------------- 图 1: 错误率降幅

def overview_panel(ax, spec, title, sub, xmax, old_names):
    rows = []
    for lab, _, vals, hb, kind in spec:
        o, j = best_old(vals, hb)
        rows.append((lab, o, vals[-1], hb, kind, old_names[j]))
    rows.sort(key=lambda t: rel_drop(t[1], t[2], t[3]))
    y = np.arange(len(rows))
    h = 0.62
    for yy, (lab, o, n, hb, kind, on) in zip(y, rows):
        r = rel_drop(o, n, hb) * 100
        c = BLUE if r >= 0 else ORANGE
        x0, w = (0, r) if r >= 0 else (r, -r)
        ax.add_patch(FancyBboxPatch((x0, yy - h / 2), w, h, boxstyle='round,pad=0,rounding_size=0.18',
                                    mutation_aspect=1 / (xmax / 6), fc=c, ec='none', zorder=3))
        ax.text(r + (1.2 if r >= 0 else -1.2), yy, f'↓{r:.0f}%' if r >= 0 else f'↑{-r:.0f}%',
                va='center', ha='left' if r >= 0 else 'right', fontsize=9, fontweight='bold', color=INK, zorder=4)
        ax.text(1.09, yy, f'{on} {fmt_val(o, hb, kind)}  →  v3 {fmt_val(n, hb, kind)}', va='center', ha='left',
                fontsize=8.5, color=MUTED, transform=ax.get_yaxis_transform())
    ax.set_yticks(y, [t[0] for t in rows], fontsize=10)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(min(-xmax * 0.4, min(rel_drop(t[1], t[2], t[3]) * 100 for t in rows) - 22), xmax)
    ax.axvline(0, color=INK2, lw=0.9, zorder=2)
    ax.grid(axis='x', color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks([])
    clean(ax)
    n_win = sum(rel_drop(t[1], t[2], t[3]) > 0 for t in rows)
    ax.annotate(title, (0, 1), xycoords='axes fraction', xytext=(0, 26), textcoords='offset points', fontsize=15,
                fontweight='bold', va='bottom')
    ax.text(0, 1.01, f'{sub}　·　{n_win} / {len(rows)} 项更优', transform=ax.transAxes, fontsize=9.5,
            color=INK2, va='bottom')


def fig_overview(cd, names, table, B, out):
    cs, ss = ccip_spec(cd), csip_spec(names, table, B)
    n = max(len(cs), len(ss))
    H = 0.42 * n + 2.9
    fig = plt.figure(figsize=(18, H))
    top, bot = 1 - 2.3 / H, 0.75 / H
    ax1 = fig.add_axes([0.175, top - (top - bot) * len(cs) / n, 0.2, (top - bot) * len(cs) / n])
    ax2 = fig.add_axes([0.65, top - (top - bot) * len(ss) / n, 0.2, (top - bot) * len(ss) / n])
    overview_panel(ax1, cs, 'CCIP · 角色识别', '对比旧版 AnimeCCIP；开放集测试，1,002 个角色均未参与 v3 训练', 62, ['旧版'])
    overview_panel(ax2, ss, 'CSIP · 画师风格识别', '对比旧版 v1 / v2 / v2.1 中每项的最优者', 52,
                   ['v1', 'v2', 'v2.1'])
    fig.text(0.015, 1 - 0.35 / H, 'v3 相对旧版的错误率降幅', fontsize=21, fontweight='bold', va='top')
    fig.text(0.015, 1 - 0.9 / H, '每行一项任务，条长 = v3 错误率相对旧版下降的比例 (↓50% 即错误减半)。'
             '蓝色 = v3 更优，橙色 = v3 更差；右侧灰字为旧版 → v3 的错误率 (偏差类为偏差分)。',
             fontsize=10.5, color=INK2, va='top')
    fig.text(0.015, 0.28 / H, '错误率 = 1 − 指标得分 (验证: PR-AUC；识别: top-1 准确率；检索: R@1 / MAP@R；开集拒识: AUROC；聚类: ARI；召回: 召回率)；'
             '误报率、偏差、阈值损失直接用原值。旧版另外两个测试集未列入：旧版训练时见过其中图片。', fontsize=9, color=MUTED)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=BLUE), plt.Rectangle((0, 0), 1, 1, fc=ORANGE)]
    fig.legend(handles, ['v3 更优', 'v3 更差'], loc='upper right', bbox_to_anchor=(0.985, 1 - 0.3 / H), ncol=2,
               frameon=False, fontsize=11, handlelength=1.1, handleheight=1.1)
    save(fig, out / 'showcase_1_overview')


# ---------------------------------------------------------------- 图 2: CSIP 雷达

# 雷达图轴旁写性能得分 (越高越好), 误报率 / 偏差保留原值并注明越低越好
RADAR_METRIC = {'验证': ('PR-AUC', 'dec'), '识别': ('top-1 准确率', 'pct'), '检索': ('MAP@R', 'dec'),
                '开集拒识': ('AUROC', 'dec'), '聚类': ('ARI', 'dec'), '跨年份': ('召回率', 'pct')}


def score_text(lab, v, kind, name=True):
    if kind == 'bias':
        return f'偏差 {v:.3f}' if name else f'{v:.3f} (越低越好)'
    if kind == 'fp':
        return f'误报率 {v * 100:.2f}%' if name else f'{v * 100:.2f}% (越低越好)'
    metric, unit = next(m for k, m in RADAR_METRIC.items() if lab.startswith(k))
    val = f'{v * 100:.1f}%' if unit == 'pct' else f'{v:.3f}'
    return f'{metric} {val}' if name else val


def fig_radar(names, table, B, out):
    spec = csip_spec(names, table, B)
    axes_def = [(short, np.asarray(v, float), hb, kind) for _, short, v, hb, kind in spec if short]
    # 难认的画 (跨年份 / 黑白 / 漫画 / 草图) 四项召回平均成一条轴
    hard = np.mean([np.asarray(v, float) for lab, _, v, _, _ in spec if lab.startswith(('跨年份', '跨媒介'))], axis=0)
    axes_def.insert(5, ('跨年份 / 跨媒介\n召回', hard, True, 'err'))
    K = len(axes_def)
    th = np.linspace(0, 2 * np.pi, K, endpoint=False)
    fig = plt.figure(figsize=(11, 10.8))
    ax = fig.add_axes([0.21, 0.15, 0.58, 0.6], projection='polar')
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.0)
    ax.axis('off')
    for r in (0.2, 0.4, 0.6, 0.8, 1.0):
        ax.plot(np.r_[th, th[:1]], np.full(K + 1, r), color=GRID if r < 1 else AXIS, lw=0.8 if r < 1 else 1.1, zorder=0)
    for t in th:
        ax.plot([t, t], [0, 1], color=GRID, lw=0.8, zorder=0)
    R = np.zeros((4, K))
    for k, (_, vv, hb, _) in enumerate(axes_def):
        s = (vv - vv.min()) / (vv.max() - vv.min())
        R[:, k] = 0.15 + 0.85 * (s if hb else 1 - s)
    tt = np.r_[th, th[:1]]
    for mi, m in enumerate(list(OLDS) + [REL]):
        rr = np.r_[R[mi], R[mi][:1]]
        if m == REL:
            ax.fill(tt, rr, color=BLUE, alpha=0.16, zorder=4)
            ax.plot(tt, rr, color=BLUE, lw=2.6, zorder=5)
            ax.scatter(th, R[mi], s=42, color=BLUE, edgecolor=WHITE, linewidth=1.8, zorder=6)
        else:
            c = OLDS[m]
            ax.fill(tt, rr, color=c, alpha=0.18, zorder=2)
            ax.plot(tt, rr, color=c, lw=1.8, zorder=3, ls=(0, (5, 2)) if m == 'HF v1' else '-')
            ax.scatter(th, R[mi], s=22, color=c, edgecolor=WHITE, linewidth=1.2, zorder=3)
    for k, (lab, vv, hb, kind) in enumerate(axes_def):
        o, j = best_old(vv, hb)
        t = th[k]
        ha = 'center' if abs(np.sin(t)) < 0.2 else ('left' if np.sin(t) > 0 else 'right')
        below = np.cos(t) < -0.3        # 下半圈: 名称和数值都放在锚点下方, 免得压进图形
        ny = -2 if below else 1
        vy = -2 - 15.5 * (lab.count('\n') + 1) if below else -1
        ax.annotate(lab, (t, 1.13), xytext=(0, ny), textcoords='offset points', ha=ha, va='top' if below else 'bottom',
                    fontsize=11, fontweight='bold', color=INK, linespacing=1.2, annotation_clip=False)
        ax.annotate(f'{score_text(lab, o, kind)} → {score_text(lab, vv[-1], kind, name=False)}', (t, 1.13), xytext=(0, vy),
                    textcoords='offset points', ha=ha, va='top', fontsize=9, color=MUTED, annotation_clip=False)
    fig.text(0.04, 0.962, 'CSIP 画师风格识别：四个版本的能力对比', fontsize=21, fontweight='bold')
    fig.text(0.04, 0.93, '每条轴在 4 个模型间归一，越往外越好 (误报、偏差类已反向)。v3 新版在全部 9 项上超过三个旧版。轴旁小字：旧版最优得分 → v3 得分。',
             fontsize=10.5, color=INK2)
    fig.text(0.04, 0.008, '数据：Danbooru 100 位画师 + 公开数据 122 位画师两个测试集的平均。'
             '偏差：画风模型应只依据绘画风格判断，不应受画面内容或所画角色影响。',
             fontsize=9, color=MUTED)
    handles = [plt.Line2D([], [], color=OLDS[m], lw=2, ls=(0, (5, 2)) if m == 'HF v1' else '-', marker='o', ms=5,
                          mec=WHITE) for m in OLDS] + [plt.Line2D([], [], color=BLUE, lw=2.6, marker='o', ms=6, mec=WHITE)]
    fig.legend(handles, ['旧版 v1', '旧版 v2', '旧版 v2.1', 'v3 新版'], loc='upper left', ncol=4,
               frameon=False, fontsize=11, bbox_to_anchor=(0.035, 0.912), handlelength=2.2, columnspacing=2)
    save(fig, out / 'showcase_2_csip_radar')


# ---------------------------------------------------------------- 图 3: 错误率柱状图

def bars_panel(ax, labels, vals, colors, title, kind):
    vals = np.asarray(vals, float)
    x = np.arange(len(vals))
    w = 0.7
    top = vals.max() * 1.42
    for xi, v, c in zip(x, vals, colors):
        ax.add_patch(FancyBboxPatch((xi - w / 2, 0), w, v, boxstyle='round,pad=0,rounding_size=0.12',
                                    mutation_aspect=top / 4.2, fc=c, ec='none', zorder=3))
        txt = f'{v:.3f}' if kind == 'bias' else (f'{v * 100:.2f}' if v < 0.01 else f'{v * 100:.1f}')
        ax.text(xi, v + top * 0.025, txt, ha='center', va='bottom', fontsize=9,
                color=INK if c == BLUE else INK2, fontweight='bold' if c == BLUE else 'normal')
    ref = vals[:-1].min()
    d = (ref - vals[-1]) / ref * 100
    pill(ax, x[-1], top * 0.95, f'↓{d:.0f}%' if d >= 0 else f'↑{-d:.0f}%',
         BLUE if d >= 0 else ORANGE, fs=8.5)
    ax.set_xlim(-0.6, len(vals) - 0.4)
    ax.set_ylim(0, top)
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_yticks([])
    clean(ax)
    ax.axhline(0, color=AXIS, lw=1, zorder=4)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=4, loc='center', linespacing=1.3)


def fig_error_bars(cd, names, table, B, out):
    cs = {lab: (v, hb, kind) for lab, _, v, hb, kind in ccip_spec(cd)}
    ss = {lab: (v, hb, kind) for lab, _, v, hb, kind in csip_spec(names, table, B)}
    ccip = [('验证\n是否同一角色', '验证：两张图是否同一角色'),
            ('识别 (1-shot)\n给 1 张参考图', '识别 (1-shot)：给 1 张参考图认出角色'),
            ('检索 R@1\n以图搜图首个命中', '检索 R@1：以图搜图，首个结果命中'),
            ('开集拒识\n未见过的角色', '开集拒识：识别出未见过的角色'),
            ('聚类\n自动按角色分组', '聚类：无标注图片自动按角色分组'),
            ('验证 · 旧版\n未训练过的角色', '验证 · 旧版未训练过的 657 个角色')]
    csip = [('验证\n是否同一画师', '验证：两张图是否同一画师'),
            ('识别 (1-shot)\n给 1 张参考图', '识别 (1-shot)：给 1 张参考图认出画师'),
            ('检索 MAP@R\n以图搜同画师作品', '检索 MAP@R：以图搜同画师作品'),
            ('开集拒识\n未见过的画师', '开集拒识：识别出未见过的画师'),
            ('跨年份召回\n相隔 6 年以上', '跨年份召回：同一画师相隔 6 年以上'),
            ('同角色误报\n不同画师画同一角色', '同角色误报：不同画师画同一角色'),
            ('内容偏差 (偏差分)\n受画面内容影响', '内容偏差：受画面内容 (标签) 影响'),
            ('角色偏差 (偏差分)\n受所画角色影响', '角色偏差：受所画角色影响')]
    fig = plt.figure(figsize=(16.5, 9))
    gs1 = fig.add_gridspec(1, 6, left=0.04, right=0.98, top=0.75, bottom=0.54, wspace=0.35)
    gs2 = fig.add_gridspec(1, 8, left=0.04, right=0.98, top=0.35, bottom=0.07, wspace=0.35)
    for k, (t, key) in enumerate(ccip):
        v, hb, kind = cs[key]
        e = [err(x, hb) if kind == 'err' else x for x in v]
        bars_panel(fig.add_subplot(gs1[0, k]), ['旧版', 'v3'], e, [OLDS['HF v2.1'], BLUE], t, kind)
    for k, (t, key) in enumerate(csip):
        v, hb, kind = ss[key]
        e = [err(x, hb) if kind == 'err' else x for x in v]
        bars_panel(fig.add_subplot(gs2[0, k]), ['v1', 'v2', 'v2.1', 'v3'], e,
                   [OLDS['HF v1'], OLDS['HF v2'], OLDS['HF v2.1'], BLUE], t, kind)
    fig.text(0.04, 0.955, '关键任务错误率 (越低越好)', fontsize=21, fontweight='bold')
    fig.text(0.04, 0.918, '柱顶数字为错误率 / 误报率 (%)，最后两项为偏差分；蓝色标签 = v3 相对旧版最优的错误率降幅。',
             fontsize=10.5, color=INK2)
    fig.text(0.04, 0.83, 'CCIP · 角色识别', fontsize=15, fontweight='bold')
    fig.text(0.175, 0.832, '对比旧版 AnimeCCIP；开放集测试，1,002 个角色均未参与 v3 训练', fontsize=10.5, color=INK2)
    fig.text(0.04, 0.43, 'CSIP · 画师风格识别', fontsize=15, fontweight='bold')
    fig.text(0.22, 0.432, '对比旧版 v1 / v2 / v2.1；Danbooru 100 位画师 + 公开数据 122 位画师两个测试集', fontsize=10.5,
             color=INK2)
    save(fig, out / 'showcase_3_error_bars')


# ---------------------------------------------------------------- 图 4: 精度 vs 偏差

def fig_pareto(names, table, B, out):
    T = {(g, lab): np.array(v, float) for g, lab, v, _, _ in table}
    acc = (T[('0918', '验证 PR-AUC')] + T[('public', '验证 PR-AUC')]) / 2
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.12, wspace=0.08)
    show = list(OLDS) + ['v3 teacher_v1', 'v3 x2', 'v3 x4', REL]
    lab = {'HF v1': '旧版 v1', 'HF v2': '旧版 v2', 'HF v2.1': '旧版 v2.1', 'v3 teacher_v1': 'v3 早期版',
           'v3 x2': 'v3 加长训练', 'v3 x4': 'v3 再加长', REL: 'v3 新版'}
    off = {'HF v1': (0, -16), 'HF v2': (0, -16), 'HF v2.1': (0, -16), 'v3 teacher_v1': (12, 0), 'v3 x2': (-6, 13),
           'v3 x4': (12, -4), REL: (0, -19)}
    for ax, k, xl in ((axes[0], 0, '内容偏差：受画面内容影响的程度'), (axes[1], 1, '角色偏差：受所画角色影响的程度')):
        x = np.array([B[n][k] for n in names])
        old = [names.index(m) for m in OLDS]
        # 优于全部旧版的区域: 偏差 ≤ 旧版最低, 精度 ≥ 旧版最高
        xr, yr = x[old].min(), acc[old].max()
        xlo, xhi = x.min() - 0.16 * np.ptp(x), x.max() + 0.08 * np.ptp(x)
        ax.add_patch(plt.Rectangle((xlo, yr), xr - xlo, 1, fc=BLUE_XL, ec='none', zorder=0))
        ax.text(xlo + 0.015 * (xhi - xlo), 0.912, '优于全部\n旧版', fontsize=9.5, color=BLUE, va='top', ha='right',
                fontweight='bold', linespacing=1.3)
        ax.axvline(xr, color=BLUE_L, lw=0.9, ls=(0, (3, 3)), zorder=1)
        ax.axhline(yr, color=BLUE_L, lw=0.9, ls=(0, (3, 3)), zorder=1)
        ox, oy = x[old], acc[old]
        for a, b in ((0, 1), (1, 2)):
            ax.annotate('', xy=(ox[b], oy[b]), xytext=(ox[a], oy[a]),
                        arrowprops=dict(arrowstyle='-|>', color=OLDS['HF v2.1'], lw=1.2, shrinkA=8, shrinkB=8), zorder=2)
        v3i = [names.index(m) for m in ('v3 teacher_v1', 'v3 x2', 'v3 x4')]
        ax.plot(x[v3i], acc[v3i], color=BLUE_L, lw=1.2, zorder=2)
        ri, x2i = names.index(REL), names.index('v3 x2')
        ax.annotate('', xy=(x[ri], acc[ri]), xytext=(x[x2i], acc[x2i]),
                    arrowprops=dict(arrowstyle='-|>', color=BLUE, lw=1.4, shrinkA=8, shrinkB=11), zorder=2)
        for n in show:
            i = names.index(n)
            if n == REL:
                ax.scatter(x[i], acc[i], s=330, marker='*', color=BLUE, edgecolor=WHITE, linewidth=1.5, zorder=6)
            elif n.startswith('v3'):
                ax.scatter(x[i], acc[i], s=90, color=BLUE_L, edgecolor=WHITE, linewidth=2, zorder=5)
            else:
                ax.scatter(x[i], acc[i], s=90, color=OLDS[n], edgecolor=WHITE, linewidth=2, zorder=5)
            ax.annotate(lab[n], (x[i], acc[i]), xytext=off[n], textcoords='offset points', fontsize=10.5,
                        ha='left' if off[n][0] > 0 else ('right' if off[n][0] < 0 else 'center'), va='center',
                        color=INK if n == REL else INK2, fontweight='bold' if n == REL else 'normal')
        ax.set_xlim(xhi, xlo)   # 反转: 越往右偏差越低
        ax.set_ylim(0.745, 0.915)
        ax.set_xlabel(f'{xl}　(越往右越低 →)', fontsize=11, color=INK2, labelpad=8)
        ax.set_xticks([])
        ax.grid(color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        clean(ax)
        ax.tick_params(labelsize=9, colors=MUTED)
    axes[0].set_ylabel('验证精度 PR-AUC (两测试集平均) ↑', fontsize=11, color=INK2, labelpad=8)
    fig.text(0.02, 0.94, 'CSIP：精度与内容偏差的权衡', fontsize=21, fontweight='bold')
    fig.text(0.02, 0.885, '画风模型应只依据绘画风格判断，不应受画面内容或所画角色影响。'
             '纵轴为验证精度，横轴为偏差 (越往右越低)，右上角最优。', fontsize=10.5, color=INK2)
    fig.text(0.02, 0.85, '旧版 v2 → v2.1 以精度换取更低偏差；v3 各版本精度均高于全部旧版，'
             'v3 新版 (★，经去偏处理) 两项偏差也低于全部旧版。', fontsize=10.5, color=INK2)
    save(fig, out / 'showcase_4_pareto')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/data/neko/cxip/docs/figures')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    style()
    cd = ccip_data()
    names, table, B, _ = csip_data()
    fig_overview(cd, names, table, B, out)
    fig_radar(names, table, B, out)
    fig_error_bars(cd, names, table, B, out)
    fig_pareto(names, table, B, out)


if __name__ == '__main__':
    main()
