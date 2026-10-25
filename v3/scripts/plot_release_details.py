"""发布 README 的细节图 (英文, 图内只有坐标轴 / 图例 / 数值) + metrics_vs_threshold.npy。数据来自 release_analysis。

每个仓库 assets/: *_score_distribution, *_pr_curve, *_threshold_metrics, *_ood, *_calibration, *_fewshot,
*_matryoshka, *_tsne (数据来自 release_tsne.py); CSIP 另有 csip_hard_cases (按年份差 / 媒介的召回)。
用法: python -m v3.scripts.plot_release_details
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from v3.scripts.plot_cxip_showcase import AXIS, BLUE, GRID, INK, INK2, MUTED, ORANGE, WHITE, style  # noqa: E402

REL = Path('/data/neko/cxip_v3/release')
AN = REL / '_analysis'
EV = Path('/data/neko/cxip_v3/eval')
AQUA = '#1baf7a'
CAT8 = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
GRAYS = {'AnimeCCIP': '#706f68', 'AnimeCSIP': '#c9c8c0', 'AnimeCSIP-v2': '#a3a29a', 'AnimeCSIP-v2.1': '#706f68'}
SET_NAME = {'openset_full': 'Open-set (1,002 characters)', 'test100_0918': 'Danbooru (100 artists)',
            'csip_eval_public': 'Public (122 artists)'}


def color(m):
    return BLUE if m.endswith('-v3') else GRAYS[m]


def save(fig, path):
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print('wrote', path)


def axstyle(ax, grid='both'):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3, color=AXIS)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def panel_title(ax, text):
    ax.set_title(text, fontsize=10.5, color=INK, fontweight='bold', loc='left', pad=8)


def legend(ax, **kw):
    lg = ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, **kw)
    return lg


def top_legend(ax, ncol=2):
    """图例放在坐标区上方 (直方图 / 曲线占满坐标区时不遮挡数据)"""
    return ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc='lower left', bbox_to_anchor=(0, 1.01),
                     ncol=ncol, borderaxespad=0, columnspacing=1.6, handlelength=1.8)


def panel_title_above(ax, text, rows):
    ax.set_title(text, fontsize=10.5, color=INK, fontweight='bold', loc='left', pad=10 + 15 * rows)


# ---------------------------------------------------------------- 由直方图得到曲线

class Pairs:
    def __init__(self, d, meta, k):
        self.pos, self.neg = d[k + '/hist_pos'].astype(np.float64), d[k + '/hist_neg'].astype(np.float64)
        self.scale, self.bias = meta['scale'][k], meta['bias'][k]
        n = len(self.pos)
        self.score = self.scale * (np.linspace(-1, 1, n + 1)[:-1] + 1 / n) + self.bias   # 箱中心
        self.lo = self.scale * np.linspace(-1, 1, n + 1)[:-1] + self.bias                 # 箱下沿 = 阈值
        nz = np.nonzero(self.pos + self.neg)[0][::-1]
        tp, fp = np.cumsum(self.pos[nz]), np.cumsum(self.neg[nz])
        self.thr = self.lo[nz]
        self.P = tp / np.maximum(tp + fp, 1e-300)
        self.R = tp / self.pos.sum()
        self.F1 = 2 * self.P * self.R / np.maximum(self.P + self.R, 1e-12)
        self.best = int(np.argmax(self.F1))
        self.auc = float(np.trapezoid(np.r_[1.0, self.P], np.r_[0.0, self.R]))

    @property
    def best_thr(self):
        return float(self.thr[self.best])

    def coarse(self, edges):
        idx = np.clip(np.searchsorted(edges, self.score) - 1, 0, len(edges) - 2)
        ok = (self.score >= edges[0]) & (self.score < edges[-1])
        return (np.bincount(idx[ok], self.pos[ok], len(edges) - 1),
                np.bincount(idx[ok], self.neg[ok], len(edges) - 1))

    def at(self, t):
        i = np.searchsorted(-self.thr, -t)       # thr 从高到低
        i = min(i, len(self.thr) - 1)
        return self.P[i], self.R[i], self.F1[i]

    def grid(self, step=0.01):
        """规则阈值网格上的 (threshold, precision, recall), 给 metrics_vs_threshold.npy"""
        g = np.arange(np.floor(self.thr.min()), np.ceil(self.thr.max()) + step / 2, step)
        rows = [(t, *self.at(t)[:2]) for t in g]
        return np.array(rows, dtype=np.float32)


def load(task):
    return np.load(AN / f'{task}.npz'), json.loads((AN / f'{task}.json').read_text())


# ---------------------------------------------------------------- 图

def fig_distribution(panels, path, default_thr=None):
    """panels: [(title, Pairs)]: 同类 / 不同类分数密度 (各自归一), 最优阈值线。"""
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.2), squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(left=0.07 if len(panels) > 1 else 0.11, right=0.98, top=0.78 if len(panels) > 1 else 0.83,
                        bottom=0.14, wspace=0.18)
    for ax, (title, p) in zip(axes, panels):
        cpos, cneg = np.cumsum(p.pos) / p.pos.sum(), np.cumsum(p.neg) / p.neg.sum()
        lo = p.score[np.searchsorted(cneg, 0.0005)]
        hi = p.score[min(np.searchsorted(cpos, 0.9995), len(cpos) - 1)]
        edges = np.linspace(lo, hi, 71)
        pos, neg = p.coarse(edges)
        w = np.diff(edges)
        ax.stairs(neg / neg.sum() / w, edges, fill=True, color='#bdbcb4', alpha=0.75,
                  label=f'Different {"class"} ({int(p.neg.sum()):,} pairs)')
        ax.stairs(pos / pos.sum() / w, edges, fill=True, color=BLUE, alpha=0.6,
                  label=f'Same class ({int(p.pos.sum()):,} pairs)')
        ax.axvline(p.best_thr, color=INK, lw=1.2, ls=(0, (4, 2)), label=f'Best threshold ({p.best_thr:.3f})')
        if default_thr is not None and abs(default_thr - p.best_thr) > 1e-3:
            ax.axvline(default_thr, color=ORANGE, lw=1.2, ls=(0, (1, 1.5)), label=f'Default threshold ({default_thr:.3f})')
        ax.set_xlim(lo, hi)
        ax.set_xlabel('Score (logit)', fontsize=9.5, color=INK2)
        axstyle(ax, 'y')
        lg = top_legend(ax)
        if len(panels) > 1:
            panel_title_above(ax, title, (len(lg.get_texts()) + 1) // 2)
    axes[0].set_ylabel('Density', fontsize=9.5, color=INK2)
    save(fig, path)


def fig_pr(panels, path):
    """panels: [(title, [(model, Pairs)])]"""
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.3), squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(left=0.09 if len(panels) > 1 else 0.13, right=0.98, top=0.88, bottom=0.13, wspace=0.15)
    for ax, (title, curves) in zip(axes, panels):
        for m, p in curves:
            k = slice(None, None, max(1, len(p.R) // 4000))
            ax.plot(p.R[k], p.P[k], color=color(m), lw=2.2 if m.endswith('-v3') else 1.5,
                    label=f'{m} (AUC {p.auc:.3f})', zorder=3 if m.endswith('-v3') else 2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel('Recall', fontsize=9.5, color=INK2)
        axstyle(ax)
        legend(ax, loc='lower left')
        if len(panels) > 1:
            panel_title(ax, title)
    axes[0].set_ylabel('Precision', fontsize=9.5, color=INK2)
    save(fig, path)


def fig_threshold(panels, path, default_thr=None):
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.2), squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(left=0.07 if len(panels) > 1 else 0.1, right=0.98, top=0.78 if len(panels) > 1 else 0.83,
                        bottom=0.14, wspace=0.12)
    for ax, (title, p) in zip(axes, panels):
        k = slice(None, None, max(1, len(p.thr) // 4000))
        ax.plot(p.thr[k], p.P[k], color=BLUE, lw=1.8, label='Precision')
        ax.plot(p.thr[k], p.R[k], color=ORANGE, lw=1.8, label='Recall')
        ax.plot(p.thr[k], p.F1[k], color=AQUA, lw=2.2, label=f'F1 (max {p.F1[p.best]:.3f})')
        ax.axvline(p.best_thr, color=INK, lw=1.2, ls=(0, (4, 2)), label=f'Best threshold ({p.best_thr:.3f})')
        if default_thr is not None and abs(default_thr - p.best_thr) > 1e-3:
            ax.axvline(default_thr, color=INK2, lw=1.2, ls=(0, (1, 1.5)), label=f'Default threshold ({default_thr:.3f})')
        cpos, cneg = np.cumsum(p.pos) / p.pos.sum(), np.cumsum(p.neg) / p.neg.sum()
        ax.set_xlim(p.score[np.searchsorted(cneg, 0.02)], p.score[min(np.searchsorted(cpos, 0.999), len(cpos) - 1)])
        ax.set_ylim(0, 1.02)
        ax.set_xlabel('Threshold (logit)', fontsize=9.5, color=INK2)
        axstyle(ax)
        lg = top_legend(ax, ncol=3)
        if len(panels) > 1:
            panel_title_above(ax, title, (len(lg.get_texts()) + 2) // 3)
    axes[0].set_ylabel('Score', fontsize=9.5, color=INK2)
    save(fig, path)


def ood_f1_curve(mx, unk):
    """未知 = max score < t; 返回 (t, F1) 从低到高"""
    o = np.argsort(mx)
    s, u = mx[o], unk[o]
    tp = np.cumsum(u)                 # t 取 s[i] 之后 (含 i) 预测为未知
    fp = np.cumsum(~u)
    P, R = tp / (tp + fp), tp / u.sum()
    F1 = 2 * P * R / np.maximum(P + R, 1e-12)
    return s, F1


def fig_ood(panels, path):
    """panels: [(title, d, key)]: 上: 未知检测 F1 vs 阈值 (1 / 5 张参考); 下: 已知 / 未知查询的最大类分数分布 (1 张参考)"""
    n = len(panels)
    fig, axes = plt.subplots(2, n, figsize=(5.6 * n, 7.2), squeeze=False)
    fig.subplots_adjust(left=0.08 if n > 1 else 0.11, right=0.98, top=0.93, bottom=0.07, wspace=0.15, hspace=0.42)
    out = {}
    for j, (title, d, k) in enumerate(panels):
        ax = axes[0, j]
        info = {}
        for A, ls in ((1, '-'), (5, (0, (4, 2)))):
            mx, unk, tau = d[f'{k}/ood{A}_mx'], d[f'{k}/ood{A}_unk'].astype(bool), d[f'{k}/ood{A}_tau']
            t, F1 = ood_f1_curve(mx, unk)
            b = int(np.argmax(F1))
            k2 = slice(None, None, max(1, len(t) // 3000))
            ax.plot(t[k2], F1[k2], color=BLUE, lw=2 if A == 1 else 1.6, ls=ls,
                    label=f'{A} reference image{"s" if A > 1 else ""} (best {t[b]:.2f}, F1 {F1[b]:.3f})')
            ax.scatter([t[b]], [F1[b]], s=30, color=BLUE, edgecolor=WHITE, linewidth=1.2, zorder=4)
            info[A] = dict(best=float(t[b]), f1=float(F1[b]), lo=float(np.min(tau)), hi=float(np.max(tau)))
        out[k] = info
        lo, hi = np.percentile(d[f'{k}/ood1_mx'], [0.5, 99.5])
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Threshold on max class score (logit)', fontsize=9.5, color=INK2)
        axstyle(ax)
        legend(ax, loc='lower right')
        if n > 1:
            panel_title(ax, title)
        ax = axes[1, j]
        mx, unk = d[f'{k}/ood1_mx'], d[f'{k}/ood1_unk'].astype(bool)
        edges = np.linspace(lo, hi, 61)
        ax.hist(mx[~unk], edges, density=True, color=BLUE, alpha=0.6, label=f'Known class ({(~unk).sum():,} queries)')
        ax.hist(mx[unk], edges, density=True, color=ORANGE, alpha=0.55, label=f'Unknown class ({unk.sum():,} queries)')
        ax.axvline(info[1]['best'], color=INK, lw=1.2, ls=(0, (4, 2)), label=f'Best threshold ({info[1]["best"]:.2f})')
        ax.set_xlim(lo, hi)
        ax.set_xlabel('Max class score, 1 reference image (logit)', fontsize=9.5, color=INK2)
        axstyle(ax, 'y')
        top_legend(ax)
    axes[0, 0].set_ylabel('Unknown-class F1', fontsize=9.5, color=INK2)
    axes[1, 0].set_ylabel('Density', fontsize=9.5, color=INK2)
    save(fig, path)
    return out


def fig_calibration(panels, path):
    """sigmoid(logit) 分箱 vs 实际同类比例 (测试集全部图片对)。"""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.4), squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(left=0.1 if n > 1 else 0.14, right=0.98, top=0.88, bottom=0.13, wspace=0.15)
    edges = np.linspace(0, 1, 11)
    for ax, (title, curves) in zip(axes, panels):
        ax.plot([0, 1], [0, 1], color=AXIS, lw=1, ls=(0, (3, 3)), label='Perfect calibration')
        for m, p in curves:
            prob = 1 / (1 + np.exp(-p.score))
            idx = np.clip(np.searchsorted(edges, prob) - 1, 0, 9)
            tot = np.bincount(idx, p.pos + p.neg, 10)
            pos = np.bincount(idx, p.pos, 10)
            mean_p = np.bincount(idx, prob * (p.pos + p.neg), 10) / np.maximum(tot, 1)
            ok = tot >= 200
            v3 = m.endswith('-v3')
            ax.plot(mean_p[ok], pos[ok] / tot[ok], color=color(m), lw=2.2 if v3 else 1.5, marker='o',
                    ms=5 if v3 else 4, mec=WHITE, label=m, zorder=3 if v3 else 2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_xlabel('Predicted probability  sigmoid(logit)', fontsize=9.5, color=INK2)
        axstyle(ax)
        if n > 1:
            panel_title(ax, title)
    # 图例统一放在图上方: v3 曲线在对角线上方, 会穿过坐标区左上角
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8.5, labelcolor=INK2, loc='upper center', ncol=len(l),
               bbox_to_anchor=(0.5, 0.995), columnspacing=1.6, handlelength=1.8)
    fig.subplots_adjust(top=0.84 if n > 1 else 0.86)
    axes[0].set_ylabel('Observed same-class rate', fontsize=9.5, color=INK2)
    save(fig, path)


def fig_fewshot(panels, path):
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 3.9), squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(left=0.09 if n > 1 else 0.13, right=0.98, top=0.86, bottom=0.15, wspace=0.18)
    for ax, (title, curves) in zip(axes, panels):
        for m, a in curves:
            v3 = m.endswith('-v3')
            ax.errorbar(a[:, 0], a[:, 1], yerr=a[:, 2], color=color(m), lw=2.2 if v3 else 1.5, marker='o',
                        ms=5 if v3 else 4, mec=WHITE, capsize=0, label=m, zorder=3 if v3 else 2)
            if v3:
                for x, yv in a[:, :2]:
                    ax.annotate(f'{yv:.3f}', (x, yv), xytext=(0, 7), textcoords='offset points', ha='center',
                                fontsize=8, color=INK)
        ax.set_xscale('log')
        ax.set_xticks(curves[0][1][:, 0], [f'{int(x)}' for x in curves[0][1][:, 0]])
        ax.minorticks_off()
        ax.set_xlabel('Reference images per class', fontsize=9.5, color=INK2)
        axstyle(ax)
        legend(ax, loc='lower right')
        if n > 1:
            panel_title(ax, title)
    axes[0].set_ylabel('Identification top-1', fontsize=9.5, color=INK2)
    save(fig, path)


def fig_matryoshka(panels, path):
    """panels: [(title, {dim: pr_auc} of v3, [(old model, pr_auc)])]"""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 3.7), squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(left=0.1 if n > 1 else 0.15, right=0.97, top=0.86, bottom=0.16, wspace=0.2)
    for ax, (title, v3, olds, name) in zip(axes, panels):
        dims = sorted(v3)
        ax.plot(dims, [v3[k] for k in dims], color=BLUE, lw=2.2, marker='o', ms=6, mec=WHITE, label=name, zorder=3)
        for k in dims:
            ax.annotate(f'{v3[k]:.3f}', (k, v3[k]), xytext=(0, 7), textcoords='offset points', ha='center',
                        fontsize=8, color=INK)
        for m, v in olds:
            ax.axhline(v, color=GRAYS[m], lw=1.4, ls=(0, (4, 2)), label=f'{m} (768-d)')
        ax.set_xscale('log', base=2)
        ax.set_xticks(dims, [str(k) for k in dims])
        ax.minorticks_off()
        lo = min(min(v3.values()), min(v for _, v in olds))
        ax.set_ylim(lo - 0.03, max(v3.values()) + 0.025)
        ax.set_xlabel('Embedding dimensions used', fontsize=9.5, color=INK2)
        axstyle(ax, 'y')
        legend(ax, loc='lower right')
        if n > 1:
            panel_title(ax, title)
    axes[0].set_ylabel('Verification PR-AUC', fontsize=9.5, color=INK2)
    save(fig, path)


def fig_tsne(task, models, path):
    """t-SNE (release_tsne.py): 24 类着色 (8 色 x 3 形状), 其余类浅灰作背景"""
    d = np.load(REL / '_analysis/tsne_v1.npz')
    y, hl = d[f'{task}/y'], d[f'{task}/hl']
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 5.0), squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.03, wspace=0.06)
    bg = ~np.isin(y, hl)
    for ax, m in zip(axes, models):
        X = d[f'{task}/{m}']
        ax.scatter(X[bg, 0], X[bg, 1], s=4, color='#d6d5cf', linewidths=0)
        for i, c in enumerate(hl):
            sel = y == c
            ax.scatter(X[sel, 0], X[sel, 1], s=12, color=CAT8[i % 8], marker='os^'[i // 8], alpha=0.9,
                       linewidths=0)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.set_title(m, fontsize=11, color=INK, fontweight='bold')
    save(fig, path)


def fig_hard_cases(path):
    models = [('AnimeCSIP', 'hf_AnimeCSIP'), ('AnimeCSIP-v2', 'hf_AnimeCSIP-v2'), ('AnimeCSIP-v2.1', 'hf_AnimeCSIP-v2.1'),
              ('AnimeCSIP-v3', 'v3_x2_erased')]
    B = {m: json.loads((EV / f'csip_full/{f}.json').read_text())['buckets_0918'] for m, f in models}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.9), gridspec_kw=dict(width_ratios=[1, 1.25]))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.17, wspace=0.18)
    ax = axes[0]
    yg = [('ygap/0', 'same year'), ('ygap/1-2', '1-2'), ('ygap/3-5', '3-5'), ('ygap/6+', '6+')]
    for m, _ in models:
        v3 = m.endswith('-v3')
        ax.plot(range(4), [B[m][k]['recall'] for k, _ in yg], color=color(m), lw=2.2 if v3 else 1.5, marker='o',
                ms=5 if v3 else 4, mec=WHITE, label=m, zorder=3 if v3 else 2)
    ax.set_xticks(range(4), [t for _, t in yg])
    ax.set_xlabel('Years between the two works', fontsize=9.5, color=INK2)
    ax.set_ylabel('Same-artist recall', fontsize=9.5, color=INK2)
    ax.set_ylim(0.35, 1)
    axstyle(ax, 'y')
    legend(ax, loc='lower left')
    ax = axes[1]
    med = [('medium/color-color', 'color\nvs. color'), ('medium/color-mono', 'color\nvs. mono'),
           ('medium/color-comic', 'color\nvs. comic'), ('medium/color-sketch', 'color\nvs. sketch'),
           ('medium/mono-mono', 'mono\nvs. mono')]
    w = 0.19
    for i, (m, _) in enumerate(models):
        x = np.arange(len(med)) + (i - 1.5) * w
        ax.bar(x, [B[m][k]['recall'] for k, _ in med], w * 0.9, color=color(m), label=m)
    for xi, (k, _) in enumerate(med):
        v = B['AnimeCSIP-v3'][k]['recall']
        ax.annotate(f'{v:.2f}', (xi + 1.5 * w, v), xytext=(0, 3), textcoords='offset points', ha='center',
                    fontsize=7.5, color=INK)
    ax.set_xticks(range(len(med)), [t for _, t in med], fontsize=8.5)
    ax.set_ylim(0, 1)
    axstyle(ax, 'y')
    ax.tick_params(axis='x', length=0)
    save(fig, path)


# ---------------------------------------------------------------- main

def ccip():
    d, meta = load('ccip')
    s, out = 'openset_full', REL / 'AnimeCCIP-v3/assets'
    P = {m: Pairs(d, meta, f'{s}/{m}') for m in meta['models']}
    v3 = P['AnimeCCIP-v3']
    fig_distribution([('', v3)], out / 'ccip_score_distribution.png')
    fig_pr([('', [(m, P[m]) for m in meta['models']])], out / 'ccip_pr_curve.png')
    fig_threshold([('', v3)], out / 'ccip_threshold_metrics.png')
    ood = fig_ood([('', d, f'{s}/AnimeCCIP-v3')], out / 'ccip_ood.png')
    fig_calibration([('', [(m, P[m]) for m in meta['models']])], out / 'ccip_calibration.png')
    fig_fewshot([('', [(m, d[f'{s}/{m}/fewshot']) for m in meta['models']])], out / 'ccip_fewshot.png')
    mat = {768: v3.auc, **{k: float(d[f'{s}/AnimeCCIP-v3/matry_{k}']) for k in (512, 256, 128)}}
    fig_matryoshka([('', mat, [('AnimeCCIP', P['AnimeCCIP'].auc)], 'AnimeCCIP-v3')], out / 'ccip_matryoshka.png')
    fig_tsne('ccip', meta['models'], out / 'ccip_tsne.png')
    np.save(REL / 'AnimeCCIP-v3/metrics_vs_threshold.npy', v3.grid())
    return dict(best_thr=v3.best_thr, p=float(v3.P[v3.best]), r=float(v3.R[v3.best]), f1=float(v3.F1[v3.best]),
                ood=ood, matryoshka=mat, fewshot=d[f'{s}/AnimeCCIP-v3/fewshot'].tolist(),
                fewshot_old=d[f'{s}/AnimeCCIP/fewshot'].tolist(), fewshot_classes=meta['fewshot_classes'])


def csip():
    d, meta = load('csip')
    out = REL / 'AnimeCSIP-v3/assets'
    default = json.loads((REL / 'AnimeCSIP-v3/config.json').read_text())['threshold']
    sets = meta['sets']
    P = {(s, m): Pairs(d, meta, f'{s}/{m}') for s in sets for m in meta['models']}
    V = [(SET_NAME[s], P[(s, 'AnimeCSIP-v3')]) for s in sets]
    fig_distribution(V, out / 'csip_score_distribution.png', default)
    fig_pr([(SET_NAME[s], [(m, P[(s, m)]) for m in meta['models']]) for s in sets], out / 'csip_pr_curve.png')
    fig_threshold(V, out / 'csip_threshold_metrics.png', default)
    ood = fig_ood([(SET_NAME[s], d, f'{s}/AnimeCSIP-v3') for s in sets], out / 'csip_ood.png')
    fig_calibration([(SET_NAME[s], [(m, P[(s, m)]) for m in meta['models']]) for s in sets], out / 'csip_calibration.png')
    fig_fewshot([(SET_NAME[s], [(m, d[f'{s}/{m}/fewshot']) for m in meta['models']]) for s in sets],
                out / 'csip_fewshot.png')
    mats = []
    for s in sets:
        mat = {768: P[(s, 'AnimeCSIP-v3')].auc, **{k: float(d[f'{s}/AnimeCSIP-v3/matry_{k}']) for k in (512, 256, 128)}}
        best_old = max(meta['models'][:-1], key=lambda m: P[(s, m)].auc)
        mats.append((SET_NAME[s], mat, [(best_old, P[(s, best_old)].auc)], 'AnimeCSIP-v3'))
    fig_matryoshka(mats, out / 'csip_matryoshka.png')
    fig_tsne('csip', ['AnimeCSIP-v2.1', 'AnimeCSIP-v3'], out / 'csip_tsne.png')
    fig_hard_cases(out / 'csip_hard_cases.png')
    tag = {'test100_0918': 'danbooru', 'csip_eval_public': 'public'}
    info = {}
    for s in sets:
        p = P[(s, 'AnimeCSIP-v3')]
        np.save(REL / f'AnimeCSIP-v3/metrics_vs_threshold_{tag[s]}.npy', p.grid())
        info[s] = dict(best_thr=p.best_thr, p=float(p.P[p.best]), r=float(p.R[p.best]), f1=float(p.F1[p.best]),
                       at_default=[float(x) for x in p.at(default)], fewshot=d[f'{s}/AnimeCSIP-v3/fewshot'].tolist())
    return dict(sets=info, ood=ood, matryoshka=[m[1] for m in mats])


def main():
    style()
    info = dict(ccip=ccip(), csip=csip())
    (AN / 'figure_numbers.json').write_text(json.dumps(info, indent=1, default=float))
    print(json.dumps(info, default=float)[:3000])


if __name__ == '__main__':
    main()
