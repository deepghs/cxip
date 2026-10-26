"""README 二次元图效果展示 (选例见 release_showcase_mine.py 与人工审阅后的 showcase_v1/picks.json)。

{ccip,csip}_grid.png         画师 x 角色 3x3 对照, 每个模型一张: 同一参考图, CCIP 高分落在同一列 (同角色), CSIP 落在同一行 (同画师)
                             (分数来自发布的 HF 模型; 图中有简单标注: 浅蓝底带 = 真实同类, 蓝框 = 超过 config.threshold, ✓/✗ + 图例)
{ccip,csip}_hard_pairs.png   难例对: 旧版 (最新一版) 与 v3 的 logit 和对错 (各自 best-F1 阈值), 最后一行为 v3 仍判错的例子
{ccip,csip}_clustering.png   UMAP + HDBSCAN 自动分组: 每行一个预测簇, 角上色点 = 真实类别
画师完全匿名 (图中只有 Artist A/B/C), 不附出处。
用法: python -m v3.scripts.plot_release_showcase
"""
import json
from concurrent.futures import ThreadPoolExecutor

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from v3.eval.runner import load_set  # noqa: E402
from v3.scripts.plot_cxip_showcase import BLUE, GRID, INK, INK2, MUTED, style  # noqa: E402
from v3.scripts.plot_release_details import CAT8, REL  # noqa: E402
from v3.scripts.release_showcase_mine import OUT as SHOW, thumb  # noqa: E402

GOOD, BAD = '#008300', '#e34948'
SYM = dict(fontfamily='DejaVu Sans')  # CJK 字体缺 ✗
CHAR_NAME = {'komeiji_koishi': 'Komeiji Koishi', 'hakurei_reimu': 'Hakurei Reimu', 'flandre_scarlet': 'Flandre Scarlet'}


def J(name):
    return json.loads((SHOW / name).read_text())


def save(fig, path):
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print('wrote', path)


def load_thumbs(read, idx, size=360):
    with ThreadPoolExecutor(16) as ex:
        return list(ex.map(lambda i: np.asarray(thumb(read(int(i)), size)), idx))


def show(fig, rect, im, edge=GRID, lw=0.8):
    ax = fig.add_axes(rect)
    ax.imshow(im)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(edge)
        sp.set_linewidth(lw)
    return ax


# ------------------------------------------------------------------------------------------------ A
def fig_grid_single(repo, path):
    """单模型 3x3: 浅蓝底带 = 真实同类 (CCIP 同一列 / CSIP 同一行), 蓝框 = 分数超过 config.threshold, ✓/✗ = 判断对错。"""
    g = J('grid.json')
    _, read = load_set('test100_0918')
    ims = dict(zip(g['keys'], load_thumbs(read, g['idx'])))
    chars, arts = g['chars'], g['artists']
    logit, thr = dict(zip(g['keys'], g[repo]['logits'])), g[repo]['threshold']
    ccip = repo.startswith('AnimeCCIP')
    a_ch, a_art = g['anchor'].split('|')
    same = (lambda ch, a: ch == a_ch) if ccip else (lambda ch, a: a == a_art)
    cell, gap, sgap = 1.4, 0.14, 0.5
    left, top = 1.0, 1.05
    gw = 3 * cell + 2 * gap
    W = left + gw + 1.25
    H = top + 3 * (cell + sgap) + (0.45 if ccip else 0.05) + 0.85
    fig = plt.figure(figsize=(W, H))
    X = lambda c: left + c * (cell + gap)
    Y = lambda r: H - top - r * (cell + sgap) - cell
    fig.text((left + gw / 2) / W, (H - 0.35) / H, repo, ha='center', fontsize=13, fontweight='bold', color=INK)
    band = dict(color='#e3eefb', transform=fig.transFigure, zorder=-1, lw=0)
    pad = 0.08
    if ccip:
        c = chars.index(a_ch)
        bx, by = X(c) - pad, Y(2) - sgap - 0.4
        fig.add_artist(plt.Rectangle((bx / W, by / H), (cell + 2 * pad) / W, (H - top + 0.32 - by) / H, **band))
        fig.text((X(c) + cell / 2) / W, (by + 0.2) / H, 'Same character', ha='center', va='center', fontsize=9.5,
                 color=BLUE, fontweight='bold')
    else:
        r = arts.index(int(a_art) if isinstance(arts[0], int) else a_art)
        by = Y(r) - sgap + 0.06
        fig.add_artist(plt.Rectangle(((left - 0.95) / W, by / H), (gw + 2.1) / W, (cell + sgap + pad - 0.06) / H, **band))
        fig.text((left + gw + 0.62) / W, (Y(r) + cell / 2) / H, 'Same\nartist', ha='center', va='center', fontsize=9.5,
                 color=BLUE, fontweight='bold', linespacing=1.3)
    for c, ch in enumerate(chars):
        fig.text((X(c) + cell / 2) / W, (H - top + 0.14) / H, CHAR_NAME[ch], ha='center', fontsize=9.5, color=INK2)
    for r, a in enumerate(arts):
        fig.text((left - 0.14) / W, (Y(r) + cell / 2) / H, f'Artist {"ABC"[r]}', ha='right', va='center', fontsize=9.5,
                 color=INK2)
        for c, ch in enumerate(chars):
            k, x, y = f'{ch}|{a}', X(c), Y(r)
            anchor, hit = k == g['anchor'], logit[k] > thr
            show(fig, [x / W, y / H, cell / W, cell / H], ims[k],
                 edge=INK if anchor else (BLUE if hit else GRID), lw=3.2 if anchor else (2.6 if hit else 0.8))
            ty = (y - 0.22) / H
            if anchor:
                fig.text((x + cell / 2) / W, ty, 'Reference', ha='center', va='center', fontsize=9.5, color=INK,
                         fontweight='bold')
                continue
            ok = hit == same(ch, str(a))
            fig.text((x + cell / 2 + 0.06) / W, ty, f'{logit[k]:.2f}', ha='right', va='center', fontsize=9.5,
                     color=INK if hit else MUTED, fontweight='bold' if hit else 'normal')
            fig.text((x + cell / 2 + 0.14) / W, ty, '✓' if ok else '✗', ha='left', va='center', fontsize=11,
                     color=GOOD if ok else BAD, fontweight='bold', **SYM)
    # 图例 (两行)
    what = 'character' if ccip else 'artist'
    rows = [[('ref', 'Reference image'), ('hit', f'Score > {thr:.3g}: predicted match')],
            [('band', f'Same {what} as the reference'), ('mark', 'Prediction right / wrong')]]
    for i, items in enumerate(rows):
        ly = 0.62 - i * 0.36
        for j, (kind, lab) in enumerate(items):
            x = 0.3 + j * (W / 2 - 0.1)
            if kind == 'band':
                fig.add_artist(plt.Rectangle((x / W, (ly - 0.1) / H), 0.32 / W, 0.2 / H, **band))
            elif kind == 'mark':
                fig.text((x + 0.02) / W, ly / H, '✓', va='center', fontsize=11, color=GOOD, fontweight='bold', **SYM)
                fig.text((x + 0.17) / W, ly / H, '✗', va='center', fontsize=11, color=BAD, fontweight='bold', **SYM)
            else:
                fig.add_artist(plt.Rectangle((x / W, (ly - 0.1) / H), 0.32 / W, 0.2 / H, transform=fig.transFigure,
                                             fill=False, ec=INK if kind == 'ref' else BLUE, lw=2.2))
            fig.text((x + 0.45) / W, ly / H, lab, va='center', fontsize=9, color=INK2)
    save(fig, path)


# ------------------------------------------------------------------------------------------------ D
def fig_pairs(task, path):
    d = J(f'pairs_{task}.json')
    picks = J('picks.json')[f'{task}_pairs']
    new, old = d['new'], d['last_old']
    ps = [(cat, d['pairs'][cat][i]) for cat, i in picks]
    _, read = load_set('openset_full' if task == 'ccip' else 'test100_0918')
    ims = load_thumbs(read, [x for _, p in ps for x in (p['a'], p['b'])])
    same_lab = 'Same character' if task == 'ccip' else 'Same artist'
    diff_lab = 'Different characters' if task == 'ccip' else 'Different artists'
    cols, rows = 2, (len(ps) + 1) // 2
    cell, cw, ch = 1.55, 6.0, 2.2           # 缩略图边长 / 卡片宽高 (英寸)
    W, H = cols * cw + 0.3, rows * ch + 0.15
    fig = plt.figure(figsize=(W, H))
    for n, (cat, p) in enumerate(ps):
        r, c = n // cols, n % cols
        x0, y0 = 0.2 + c * cw, H - (r + 1) * ch
        same = p['label_a'] == p['label_b']
        fig.text(x0 / W, (y0 + 0.28 + cell + 0.16) / H, same_lab if same else diff_lab, fontsize=9, color=MUTED,
                 va='center')
        for j in range(2):
            show(fig, [(x0 + j * (cell + 0.06)) / W, (y0 + 0.28) / H, cell / W, cell / H], ims[2 * n + j])
        for j, m in enumerate([old, new]):
            v = p[m]
            ok = (v > d['thr'][m]) == same
            ty = y0 + 0.28 + cell / 2 + 0.21 - j * 0.42
            fig.text((x0 + 2 * cell + 0.3) / W, ty / H, m, fontsize=9.5, color=INK2, va='center',
                     fontweight='bold' if m == new else 'normal')
            fig.text((x0 + 2 * cell + 2.2) / W, ty / H, f'{v:.2f}', fontsize=9.5, color=INK, va='center', ha='right')
            fig.text((x0 + 2 * cell + 2.32) / W, ty / H, '✓' if ok else '✗', fontsize=12, va='center',
                     color=GOOD if ok else BAD, fontweight='bold', **SYM)
        if cat.startswith('fail') and (n == 0 or not ps[n - 1][0].startswith('fail')) and c == 0:
            fig.add_artist(plt.Line2D([0.15 / W, (W - 0.15) / W], [(y0 + ch + 0.02) / H] * 2, color=GRID, lw=1))
    save(fig, path)


# ------------------------------------------------------------------------------------------------ E
def fig_clusters(task, path):
    d = J('cluster.json')[task]
    _, read = load_set('openset_full' if task == 'ccip' else 'test100_0918')
    ims = load_thumbs(read, d['idx'], 240)
    pred, lab = np.array(d['pred']), np.array(d['label'])
    order = {c: i for i, c in enumerate(dict.fromkeys(lab[np.argsort(pred, kind='stable')]))}   # 真实类 → 颜色
    clusters = sorted(set(pred) - {-1}, key=lambda c: (-(pred == c).sum(), c)) + ([-1] if (pred < 0).any() else [])
    cell, gap = 0.82, 0.05
    ncol = max(int((pred == c).sum()) for c in clusters)
    W, H = 0.25 + ncol * (cell + gap) + 0.1, 0.15 + len(clusters) * (cell + 0.14)
    fig = plt.figure(figsize=(W, H))
    for r, c in enumerate(clusters):
        mem = np.nonzero(pred == c)[0]
        mem = mem[np.argsort([order[lab[i]] for i in mem], kind='stable')]
        y = H - 0.1 - (r + 1) * (cell + 0.14) + 0.07
        for j, i in enumerate(mem):
            x = 0.25 + j * (cell + gap)
            ax = show(fig, [x / W, y / H, cell / W, cell / H], ims[i])
            ax.add_patch(plt.Circle((40, 40), 26, color=CAT8[order[lab[i]] % 8], ec='white', lw=1.5,
                                    transform=ax.transData, zorder=5))
        if c == -1:
            fig.text(0.12 / W, (y + cell / 2) / H, '−', ha='center', va='center', fontsize=10, color=MUTED)
    save(fig, path)


def main():
    style()
    for t, repo in (('ccip', 'AnimeCCIP-v3'), ('csip', 'AnimeCSIP-v3')):
        fig_grid_single(repo, REL / f'{repo}/assets/{t}_grid.png')
        fig_pairs(t, REL / f'{repo}/assets/{t}_hard_pairs.png')
        fig_clusters(t, REL / f'{repo}/assets/{t}_clustering.png')


if __name__ == '__main__':
    main()
