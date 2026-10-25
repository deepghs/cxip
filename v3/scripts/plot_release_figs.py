"""发布仓库 README 用的英文图 (图内只有坐标轴、刻度、图例和数值, 无标题/说明文字)。数据与 plot_cxip_showcase 同源。

AnimeCCIP-v3:  ccip_radar (任务得分) / ccip_error_reduction (相对旧版的错误率降幅)
AnimeCSIP-v3:  csip_radar (4 个版本任务得分) / csip_error_reduction (相对旧版逐项最优) / csip_bias_tradeoff (精度 vs 偏置)
用法: python -m v3.scripts.plot_release_figs [--out /data/neko/cxip_v3/release]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from v3.scripts.plot_cxip_compare import ccip_data, csip_data  # noqa: E402
from v3.scripts.plot_cxip_showcase import (AXIS, BLUE, BLUE_L, BLUE_XL, GRID, INK, INK2, MUTED, OLDS, ORANGE, WHITE,  # noqa: E402
                                           best_old, ccip_spec, clean, csip_spec, rel_drop, style)

OLD_NAMES = {'HF v1': 'AnimeCSIP', 'HF v2': 'AnimeCSIP-v2', 'HF v2.1': 'AnimeCSIP-v2.1'}
OLDS = {'HF v1': '#c9c8c0', 'HF v2': '#a3a29a', 'HF v2.1': '#706f68'}
SHORT = {'HF v1': 'v1', 'HF v2': 'v2', 'HF v2.1': 'v2.1'}

# 中文 spec 名 → 英文 (行名, 雷达轴名, 雷达数值格式)
EN = {
    '验证：两张图是否同一角色': 'Verification (PR-AUC)',
    '识别 (1-shot)：给 1 张参考图认出角色': 'Identification 1-shot (top-1)',
    '识别 (5-shot)：给 5 张参考图认出角色': 'Identification 5-shot (top-1)',
    '检索 R@1：以图搜图，首个结果命中': 'Retrieval (R@1)',
    '检索 MAP@R：同角色图片整体排序': 'Retrieval (MAP@R)',
    '开集拒识：识别出未见过的角色': 'Open-set rejection (AUROC)',
    '聚类：无标注图片自动按角色分组': 'Clustering (ARI)',
    '验证 · 旧版未训练过的 657 个角色': 'Verification, 657 chars new to AnimeCCIP',
    '验证 · 旧版训练过的 345 个角色': 'Verification, 345 chars AnimeCCIP trained on',
    '旧版测试集 zc500x10 (500 个角色)': 'zc500x10 (legacy) (PR-AUC)',
    '旧版测试集 zc50x15 (50 个角色)': 'zc50x15 (legacy) (PR-AUC)',
    '画风泄漏：角色特征中混入的画风信息': 'Artist-style leakage',
    '验证：两张图是否同一画师': 'Verification (PR-AUC)',
    '识别 (1-shot)：给 1 张参考图认出画师': 'Identification 1-shot (top-1)',
    '检索 MAP@R：以图搜同画师作品': 'Retrieval (MAP@R)',
    '开集拒识：识别出未见过的画师': 'Open-set rejection (AUROC)',
    '聚类：无标注图片自动按画师分组': 'Clustering (ARI)',
    '跨年份召回：同一画师相隔 6 年以上': 'Recall: works 6+ years apart',
    '跨媒介召回：彩图 vs 黑白': 'Recall: color vs. monochrome',
    '跨媒介召回：彩图 vs 漫画': 'Recall: color vs. comic',
    '跨媒介召回：彩图 vs 草图': 'Recall: color vs. sketch',
    '同角色误报：不同画师画同一角色': 'False positives: same character',
    '同作品误报：不同画师画同一作品': 'False positives: same series',
    '内容偏差：受画面内容 (标签) 影响': 'Content bias',
    '角色偏差：受所画角色影响': 'Character bias',
    '阈值迁移：换数据集沿用同一阈值的 F1 损失': 'Threshold transfer: mean F1 loss',
    '阈值迁移：最差一组的 F1 损失': 'Threshold transfer: worst F1 loss',
}

CCIP_RADAR = [('验证：两张图是否同一角色', 'Verification\nPR-AUC'),
              ('识别 (1-shot)：给 1 张参考图认出角色', 'Identification\n1-shot top-1'),
              ('识别 (5-shot)：给 5 张参考图认出角色', 'Identification\n5-shot top-1'),
              ('检索 R@1：以图搜图，首个结果命中', 'Retrieval\nR@1'),
              ('检索 MAP@R：同角色图片整体排序', 'Retrieval\nMAP@R'),
              ('开集拒识：识别出未见过的角色', 'Open-set rejection\nAUROC'),
              ('聚类：无标注图片自动按角色分组', 'Clustering\nARI')]
CSIP_RADAR = [('验证：两张图是否同一画师', 'Verification\nPR-AUC'),
              ('识别 (1-shot)：给 1 张参考图认出画师', 'Identification\n1-shot top-1'),
              ('检索 MAP@R：以图搜同画师作品', 'Retrieval\nMAP@R'),
              ('开集拒识：识别出未见过的画师', 'Open-set rejection\nAUROC'),
              ('聚类：无标注图片自动按画师分组', 'Clustering\nARI'),
              ('跨年份召回：同一画师相隔 6 年以上', 'Recall\n6+ years apart'),
              ('跨媒介召回：彩图 vs 黑白', 'Recall\ncolor vs. mono'),
              ('跨媒介召回：彩图 vs 漫画', 'Recall\ncolor vs. comic'),
              ('跨媒介召回：彩图 vs 草图', 'Recall\ncolor vs. sketch')]


def save(fig, path):
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print('wrote', path)


def radar(spec, axes_keys, models, colors, rmin, path, per_axis=False):
    """绝对得分雷达, 轴名下方是 旧版最优 → v3 的数值。
    per_axis=False: 所有轴共用半径 rmin..1。
    per_axis=True: 外圈 = 1.0 (满分), 每轴下端按错误率定: lo = worst - 0.5 (1 - worst), 最差旧版落在 1/3 半径处;
    于是任一轴上相同的径向距离 = 相同的相对错误率降幅, 不会把小差距放大成大差距。"""
    S = {lab: np.asarray(v, float) for lab, _, v, _, _ in spec}
    K = len(axes_keys)
    if per_axis:
        return radar_per_axis(S, axes_keys, models, colors, path)
    th = np.linspace(0, 2 * np.pi, K, endpoint=False)
    fig = plt.figure(figsize=(9, 8.6))
    ax = fig.add_axes([0.2, 0.15, 0.6, 0.65], projection='polar')
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    rad = lambda v: (np.asarray(v) - rmin) / (1 - rmin)  # noqa: E731
    rings = np.arange(np.ceil(rmin * 10) / 10, 1.001, 0.1)
    for r in rings:
        ax.plot(np.r_[th, th[:1]], np.full(K + 1, rad(r)), color=AXIS if r > 0.999 else GRID,
                lw=1.1 if r > 0.999 else 0.8, zorder=0)
        ax.annotate(f'{r:.1f}', (0, rad(r)), xytext=(4, 2), textcoords='offset points', fontsize=7.5, color=MUTED,
                    ha='left', va='bottom', zorder=1)
    for t in th:
        ax.plot([t, t], [0, 1], color=GRID, lw=0.8, zorder=0)
    tt = np.r_[th, th[:1]]
    for mi, (m, c) in enumerate(zip(models, colors)):
        r = rad([S[k][mi] for k, _ in axes_keys])
        rr = np.r_[r, r[:1]]
        last = mi == len(models) - 1
        ax.fill(tt, rr, color=c, alpha=0.16 if last else 0.12, zorder=4 if last else 2)
        ax.plot(tt, rr, color=c, lw=2.6 if last else 1.8, zorder=5 if last else 3)
        ax.scatter(th, r, s=40 if last else 22, color=c, edgecolor=WHITE, linewidth=1.6, zorder=6 if last else 3)
    for k, (key, lab) in enumerate(axes_keys):
        v = S[key]
        o, _ = best_old(v, True)
        t = th[k]
        ha = 'center' if abs(np.sin(t)) < 0.2 else ('left' if np.sin(t) > 0 else 'right')
        below = np.cos(t) < -0.3
        ax.annotate(lab, (t, 1.1), xytext=(0, -2 if below else 13), textcoords='offset points', ha=ha,
                    va='top' if below else 'bottom', fontsize=10.5, fontweight='bold', color=INK, linespacing=1.15,
                    annotation_clip=False)
        ax.annotate(f'{o:.3f} → {v[-1]:.3f}', (t, 1.1), xytext=(0, -2 - 13.5 * (lab.count('\n') + 1) if below else 0),
                    textcoords='offset points', ha=ha, va='top' if below else 'bottom', fontsize=9, color=MUTED,
                    annotation_clip=False)
    handles = [plt.Line2D([], [], color=c, lw=2.6 if i == len(models) - 1 else 1.8, marker='o', ms=5, mec=WHITE)
               for i, c in enumerate(colors)]
    fig.legend(handles, models, loc='upper center', ncol=len(models), frameon=False, fontsize=10.5,
               bbox_to_anchor=(0.5, 0.985), handlelength=2.2, columnspacing=1.8)
    save(fig, path)


def radar_per_axis(S, axes_keys, models, colors, path):
    K = len(axes_keys)
    th = np.linspace(0, 2 * np.pi, K, endpoint=False)
    fig = plt.figure(figsize=(9, 8.6))
    ax = fig.add_axes([0.2, 0.15, 0.6, 0.65], projection='polar')
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    lo = np.array([S[k].min() - 0.5 * (1 - S[k].min()) for k, _ in axes_keys])
    for r in (0.25, 0.5, 0.75, 1.0):
        ax.plot(np.r_[th, th[:1]], np.full(K + 1, r), color=AXIS if r == 1 else GRID, lw=1.1 if r == 1 else 0.8,
                zorder=0)
    ax.annotate('1.0', (0, 1), xytext=(4, 2), textcoords='offset points', fontsize=7.5, color=MUTED, ha='left',
                va='bottom', zorder=1)
    for t in th:
        ax.plot([t, t], [0, 1], color=GRID, lw=0.8, zorder=0)
    tt = np.r_[th, th[:1]]
    for mi, (m, c) in enumerate(zip(models, colors)):
        r = np.array([(S[k][mi] - lo[i]) / (1 - lo[i]) for i, (k, _) in enumerate(axes_keys)])
        rr = np.r_[r, r[:1]]
        last = mi == len(models) - 1
        ax.fill(tt, rr, color=c, alpha=0.16 if last else 0.12, zorder=4 if last else 2)
        ax.plot(tt, rr, color=c, lw=2.6 if last else 1.8, zorder=5 if last else 3)
        ax.scatter(th, r, s=40 if last else 22, color=c, edgecolor=WHITE, linewidth=1.6, zorder=6 if last else 3)
    for k, (key, lab) in enumerate(axes_keys):
        v = S[key]
        o, _ = best_old(v, True)
        t = th[k]
        ha = 'center' if abs(np.sin(t)) < 0.2 else ('left' if np.sin(t) > 0 else 'right')
        below = np.cos(t) < -0.3
        ax.annotate(lab, (t, 1.1), xytext=(0, -2 if below else 13), textcoords='offset points', ha=ha,
                    va='top' if below else 'bottom', fontsize=10.5, fontweight='bold', color=INK, linespacing=1.15,
                    annotation_clip=False)
        ax.annotate(f'{o:.3f} → {v[-1]:.3f}', (t, 1.1), xytext=(0, -2 - 13.5 * (lab.count('\n') + 1) if below else 0),
                    textcoords='offset points', ha=ha, va='top' if below else 'bottom', fontsize=9, color=MUTED,
                    annotation_clip=False)
    handles = [plt.Line2D([], [], color=c, lw=2.6 if i == len(models) - 1 else 1.8, marker='o', ms=5, mec=WHITE)
               for i, c in enumerate(colors)]
    fig.legend(handles, models, loc='upper center', ncol=len(models), frameon=False, fontsize=10.5,
               bbox_to_anchor=(0.5, 0.985), handlelength=2.2, columnspacing=1.8)
    save(fig, path)


def error_reduction(spec, old_names, xlabel, path, xmax):
    """相对错误率降幅 (旧版逐项最优 → v3), 按幅度排序; 右侧为 旧 → 新 的原始值。"""
    rows = []
    for lab, _, vals, hb, kind in spec:
        o, j = best_old(vals, hb)
        rows.append((EN[lab], o, vals[-1], hb, kind, old_names[j]))
    rows.sort(key=lambda t: rel_drop(t[1], t[2], t[3]))
    n = len(rows)
    H = 0.4 * n + 1.6
    fig = plt.figure(figsize=(11.5, H))
    ax = fig.add_axes([0.33, 0.95 / H, 0.4, 1 - 1.25 / H])
    y = np.arange(n)
    h = 0.62
    lo = min(-xmax * 0.3, min(rel_drop(t[1], t[2], t[3]) * 100 for t in rows) - 18)
    for yy, (lab, o, nv, hb, kind, on) in zip(y, rows):
        r = rel_drop(o, nv, hb) * 100
        c = BLUE if r >= 0 else ORANGE
        x0, w = (0, r) if r >= 0 else (r, -r)
        ax.add_patch(FancyBboxPatch((x0, yy - h / 2), w, h, boxstyle='round,pad=0,rounding_size=0.18',
                                    mutation_aspect=1 / ((xmax - lo) / 7), fc=c, ec='none', zorder=3))
        ax.text(r + (1.2 if r >= 0 else -1.2), yy, f'−{-r:.0f}%' if r < 0 else f'{r:.0f}%',
                va='center', ha='left' if r >= 0 else 'right', fontsize=9, fontweight='bold', color=INK, zorder=4)
        fv = (lambda v: f'{v:.3f}') if kind == 'bias' else (lambda v: f'{v:.4f}' if kind == 'fp' else f'{v:.3f}')
        ax.text(1.04, yy, f'{fv(o)}' + (f' ({on})' if len(set(old_names)) > 1 else ''), va='center', ha='left', fontsize=8.5, color=MUTED,
                transform=ax.get_yaxis_transform())
        ax.text(1.21, yy, '→', va='center', ha='left', fontsize=8.5, color=MUTED, transform=ax.get_yaxis_transform())
        ax.text(1.27, yy, f'{fv(nv)}', va='center', ha='left', fontsize=8.5, color=INK if r >= 0 else ORANGE,
                fontweight='bold', transform=ax.get_yaxis_transform())
    ax.set_yticks(y, [t[0] for t in rows], fontsize=10)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlim(lo, xmax)
    ax.axvline(0, color=INK2, lw=0.9, zorder=2)
    ax.grid(axis='x', color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))
    ax.tick_params(axis='x', labelsize=8.5, colors=MUTED)
    ax.set_xlabel(xlabel, fontsize=10, color=INK2, labelpad=6)
    clean(ax)
    ax.tick_params(axis='x', length=0)
    tr = ax.get_yaxis_transform()
    hdr = n - 0.3 + 0.25
    ax.text(1.04, hdr, 'previous', fontsize=8.5, color=MUTED, transform=tr, va='bottom', fontweight='bold')
    ax.text(1.27, hdr, 'v3', fontsize=8.5, color=INK, transform=tr, va='bottom', fontweight='bold')
    save(fig, path)


def bias_tradeoff(names, table, B, path):
    T = {(g, lab): np.array(v, float) for g, lab, v, _, _ in table}
    acc = (T[('0918', '验证 PR-AUC')] + T[('public', '验证 PR-AUC')]) / 2
    REL = 'v3 x2+擦除'
    show = list(OLDS) + [REL]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.14, wspace=0.07)
    lab = {**OLD_NAMES, REL: 'AnimeCSIP-v3'}
    off = {'HF v1': (0, -15), 'HF v2': (0, 13), 'HF v2.1': (0, -15), REL: (0, -18)}
    for ax, k, xl in ((axes[0], 0, 'Content bias ↓'), (axes[1], 1, 'Character bias ↓')):
        x = np.array([B[n][k] for n in names])
        idx = [names.index(m) for m in show]
        xs = x[idx]
        xlo, xhi = 0, xs.max() * 1.12
        old = idx[:3]
        xr, yr = x[old].min(), acc[old].max()
        ax.add_patch(plt.Rectangle((xlo, yr), xr - xlo, 1, fc=BLUE_XL, ec='none', zorder=0))
        ax.axvline(xr, color=BLUE_L, lw=0.9, ls=(0, (3, 3)), zorder=1)
        ax.axhline(yr, color=BLUE_L, lw=0.9, ls=(0, (3, 3)), zorder=1)
        for a, b in ((0, 1), (1, 2)):
            ax.annotate('', xy=(x[old[b]], acc[old[b]]), xytext=(x[old[a]], acc[old[a]]),
                        arrowprops=dict(arrowstyle='-|>', color=OLDS['HF v2.1'], lw=1.1, shrinkA=8, shrinkB=8), zorder=2)
        for n in show:
            i = names.index(n)
            if n == REL:
                ax.scatter(x[i], acc[i], s=320, marker='*', color=BLUE, edgecolor=WHITE, linewidth=1.5, zorder=6)
            else:
                ax.scatter(x[i], acc[i], s=85, color=OLDS[n], edgecolor=WHITE, linewidth=2, zorder=5)
            ax.annotate(lab[n], (x[i], acc[i]), xytext=off[n], textcoords='offset points', fontsize=9.5, ha='center',
                        va='center', color=INK if n == REL else INK2, fontweight='bold' if n == REL else 'normal')
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(0.74, 0.9)
        ax.set_xlabel(xl, fontsize=10.5, color=INK2, labelpad=6)
        ax.grid(color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        clean(ax)
        ax.tick_params(labelsize=8.5, colors=MUTED)
    axes[0].set_ylabel('Verification PR-AUC ↑', fontsize=10.5, color=INK2, labelpad=6)
    save(fig, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/data/neko/cxip_v3/release')
    args = ap.parse_args()
    out = Path(args.out)
    style()
    cd = ccip_data()
    names, table, B, _ = csip_data()
    cs, ss = ccip_spec(cd), csip_spec(names, table, B)
    d = out / 'AnimeCCIP-v3' / 'assets'
    d.mkdir(parents=True, exist_ok=True)
    radar(cs, CCIP_RADAR, ['AnimeCCIP', 'AnimeCCIP-v3'], [OLDS['HF v2.1'], BLUE], 0.5, d / 'ccip_radar.png',
          per_axis=True)
    error_reduction(cs, ['AnimeCCIP'], 'Relative error reduction vs. AnimeCCIP', d / 'ccip_error_reduction.png', 62)
    d = out / 'AnimeCSIP-v3' / 'assets'
    d.mkdir(parents=True, exist_ok=True)
    radar(ss, CSIP_RADAR, list(OLD_NAMES.values()) + ['AnimeCSIP-v3'], list(OLDS.values()) + [BLUE], 0.4,
          d / 'csip_radar.png', per_axis=True)
    error_reduction(ss, [SHORT[m] for m in OLDS], 'Relative error reduction vs. best previous AnimeCSIP',
                    d / 'csip_error_reduction.png', 52)
    bias_tradeoff(names, table, B, d / 'csip_bias_tradeoff.png')


if __name__ == '__main__':
    main()
