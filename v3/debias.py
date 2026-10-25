"""CSIP 去偏: 混淆式对抗擦除 (Alvi et al. 2018, "Turning a Blind Eye")。

每个概念 (内容标签 / 角色) 一个擦除器 MLP(embedding → 概念), 同一次反向里两路:
  - 擦除器路: 在 embedding.detach() 上用真实标签训练擦除器 (它尽量从画风向量里读出概念);
  - 混淆路:   擦除器参数 detach, 编码器让擦除器的输出贴近概念先验 (BCE→标签频率 / CE→角色频率),
              最优解 = 擦除器对每张图都只能报先验 = 向量里读不出该概念。
比 GRL (直接最大化擦除器损失) 稳: GRL 会把预测推向 "反着猜" 的极端, 那本身也是信息, 且梯度无界
(v2/v2.1 的 GRL 擦除需要很小的 λ 才不坍缩, 见 model/csip_debias.py)。

内容标签: danbooru-artist-v1 的 general 标签 ∩ 偏置探针的 320 个 tagger 标签, 去掉画风/媒介/元信息类
(monochrome、comic、signature …: 这些本来就是画风的一部分, 擦掉会伤画风识别)。
角色: 单角色图的角色名, 只保留训练集里 ≥ char_min_count 张的角色, 其余样本不参与角色擦除。

conditional=True (条件去偏): 混淆目标从全局先验换成 "该画师的概念分布" (留一、向全局先验平滑)。
Danbooru 里角色/作品与画师强相关, 全局版会连画师信息一起擦掉 (角色擦除 λ=0.01 已让 0918 掉到 0.61);
条件版只要求 "给定画师后, 向量里读不出这张图画的是谁/什么", 即只擦画师内部的内容差异 —— 内容偏置的定义本身。
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

# 偏置探针的标签表 (cxip_repro/results/bias_targets.pt 的 tag_names), 训练时只用它和 general 标签的交集
BIAS_TARGETS = '/data/neko/cxip_repro/results/bias_targets.pt'
# 画风 / 媒介 / 构图习惯 / 元信息: 属于画师习惯, 不擦
STYLE_TAGS = {
    'monochrome', 'greyscale', 'comic', '4koma', 'chibi', 'sketch', 'lineart', 'traditional_media',
    'signature', 'artist_name', 'twitter_username', 'dated', 'english_text', 'speech_bubble', 'character_name',
    'multiple_views', 'simple_background', 'white_background', 'grey_background', 'blurry', 'blurry_background',
    'censored', 'uncensored', 'mosaic_censoring', 'bar_censor',
}


def _tag_vocab(tag_vocab=None):
    if tag_vocab is None:
        tag_vocab = torch.load(BIAS_TARGETS, weights_only=True)['tag_names']
    return [t for t in tag_vocab if t not in STYLE_TAGS]


def build_tag_targets(general, tag_vocab=None):
    """general: 每样本 "a, b c, ..." 字符串 (可为 None)。返回 (packed (N, ceil(T/8)) uint8, has (N,) bool, vocab)。"""
    vocab = _tag_vocab(tag_vocab)
    pos = {t.replace('_', ' '): i for i, t in enumerate(vocab)}
    bits = np.zeros((len(general), len(vocab)), np.bool_)
    has = np.zeros(len(general), np.bool_)
    for n, g in enumerate(general):
        if not g:
            continue
        has[n] = True
        for t in g.split(', '):
            j = pos.get(t)
            if j is not None:
                bits[n, j] = True
    return np.packbits(bits, axis=1), has, vocab


def build_char_targets(characters, min_count=20):
    """characters: 每样本角色名列表。单角色且该角色 ≥ min_count 张 → 类 id, 否则 -1。返回 (ids (N,), names)。"""
    single = np.array([c[0] if c is not None and len(c) == 1 else None for c in characters], dtype=object)
    names, inv, cnt = np.unique(single[single != None].astype(str), return_inverse=True, return_counts=True)  # noqa: E711
    keep = cnt >= min_count
    remap = np.full(len(names), -1, np.int64)
    remap[keep] = np.arange(int(keep.sum()))
    ids = np.full(len(single), -1, np.int64)
    ids[single != None] = remap[inv]  # noqa: E711
    return ids, names[keep].tolist()


class ConceptTargets:
    """按训练 df 的行号 (= 数据集下标) 取每样本的概念标签; 数据留在 CPU, 每步只把本 batch 搬上 GPU。"""

    def __init__(self, df, char_min_count=20, use_tags=True, use_chars=True, conditional=False, smooth=2.0):
        self.tags = self.chars = None
        self.conditional, self.smooth = conditional, float(smooth)
        if conditional:
            self.artist = df['label'].values.astype(np.int64)
            n_art = int(self.artist.max()) + 1
        if use_tags:
            self.tag_packed, self.tag_has, self.tag_vocab = build_tag_targets(df['general'].values)
            bits = np.unpackbits(self.tag_packed[self.tag_has], axis=1, count=len(self.tag_vocab))
            self.tag_prior = torch.tensor(bits.mean(0), dtype=torch.float32).clamp(1e-4, 1 - 1e-4)
            self.tags = True
            if conditional:  # 每画师: 有标签图的张数 + 各标签计数
                a = self.artist[self.tag_has]
                self.tag_n = np.bincount(a, minlength=n_art).astype(np.float32)
                self.tag_sum = np.zeros((n_art, len(self.tag_vocab)), np.float32)
                np.add.at(self.tag_sum, a, bits.astype(np.float32))
        if use_chars:
            self.char_ids, self.char_names = build_char_targets(df['characters'].values, char_min_count)
            c = np.bincount(self.char_ids[self.char_ids >= 0], minlength=len(self.char_names))
            self.char_prior = torch.tensor(c / c.sum(), dtype=torch.float32)
            self.chars = True
            if conditional:  # 每画师 × 角色计数 (稀疏)
                ok = self.char_ids >= 0
                self.char_count = sp.csr_matrix((np.ones(int(ok.sum()), np.float32), (self.artist[ok], self.char_ids[ok])),
                                                shape=(n_art, len(self.char_names)))
                self.char_n = np.asarray(self.char_count.sum(1)).ravel().astype(np.float32)

    def batch(self, idx, device):
        idx = np.asarray(idx)
        out = {}
        if self.tags:
            bits = np.unpackbits(self.tag_packed[idx], axis=1, count=len(self.tag_vocab))
            out['tags'] = torch.from_numpy(bits).to(device, torch.float32, non_blocking=True)
            out['tag_mask'] = torch.from_numpy(self.tag_has[idx]).to(device, non_blocking=True)
            if self.conditional:
                a, own = self.artist[idx], bits.astype(np.float32) * self.tag_has[idx, None]
                n = self.tag_n[a][:, None] - self.tag_has[idx, None]
                tgt = (self.tag_sum[a] - own + self.smooth * self.tag_prior.numpy()) / (n + self.smooth)
                out['tag_target'] = torch.from_numpy(np.clip(tgt, 1e-4, 1 - 1e-4)).to(device, non_blocking=True)
        if self.chars:
            c = self.char_ids[idx]
            out['char'] = torch.from_numpy(c).to(device, non_blocking=True)
            if self.conditional:
                a = self.artist[idx]
                cnt = self.char_count[a].toarray()
                has = c >= 0
                cnt[np.nonzero(has)[0], c[has]] -= 1  # 留一: 不含本图
                n = self.char_n[a] - has
                tgt = (cnt + self.smooth * self.char_prior.numpy()) / (n + self.smooth)[:, None]
                out['char_target'] = torch.from_numpy(tgt.astype(np.float32)).to(device, non_blocking=True)
        return out


def eraser_mlp(in_dim, hidden, out_dim):
    return nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                         nn.GELU(), nn.Linear(hidden, out_dim))


class Debiaser(nn.Module):
    """擦除器集合 + 混淆损失。forward(emb, targets) → (加到主损失上的标量, logs)。"""

    def __init__(self, in_dim, concepts: ConceptTargets, tag_weight=0.0, char_weight=0.0, hidden=1024):
        super().__init__()
        self.tag_weight, self.char_weight = tag_weight, char_weight
        self.ramp = 1.0  # 混淆权重的爬坡系数 (训练循环每步设置): 开头擦除器还是随机的, 先让它学会读概念
        self.erasers = nn.ModuleDict()
        if tag_weight > 0:
            self.erasers['tags'] = eraser_mlp(in_dim, hidden, len(concepts.tag_vocab))
            self.register_buffer('tag_prior', concepts.tag_prior.clone())
        if char_weight > 0:
            self.erasers['char'] = eraser_mlp(in_dim, hidden, len(concepts.char_names))
            self.register_buffer('char_prior', concepts.char_prior.clone())

    @staticmethod
    def _frozen(m):
        return {k: v.detach() for k, v in m.named_parameters()}

    def forward(self, emb, t):
        z = emb.float()
        total, logs = z.new_zeros(()), {}
        if 'tags' in self.erasers:
            m, e = t['tag_mask'], self.erasers['tags']
            if m.any():
                y, zm = t['tags'][m], z[m]
                l_er = F.binary_cross_entropy_with_logits(e(zm.detach()), y)
                tgt = t['tag_target'][m] if 'tag_target' in t else self.tag_prior.expand_as(y)
                l_conf = F.binary_cross_entropy_with_logits(functional_call(e, self._frozen(e), (zm,)), tgt)
                with torch.no_grad():  # 擦除器相对 "只报先验" 的增益: ≈ 在线版 I_content (越接近 0 越干净)
                    l_prior = F.binary_cross_entropy_with_logits(torch.logit(self.tag_prior).expand_as(y), y)
                total = total + l_er + self.ramp * self.tag_weight * l_conf
                logs.update(tag_er=l_er.detach(), tag_I=(1 - l_er / l_prior).detach())
            else:  # DDP 要求每张卡上所有参数都有梯度
                total = total + 0 * sum(p.sum() for p in e.parameters())
        if 'char' in self.erasers:
            c, e = t['char'], self.erasers['char']
            m = c >= 0
            if m.any():
                zm, y = z[m], c[m]
                l_er = F.cross_entropy(e(zm.detach()), y)
                logp = F.log_softmax(functional_call(e, self._frozen(e), (zm,)), dim=-1)
                tgt = t['char_target'][m] if 'char_target' in t else self.char_prior
                l_conf = -(tgt * logp).sum(-1).mean()  # CE(先验 → 擦除器), 最优 = 输出 (条件) 先验
                with torch.no_grad():
                    l_prior = -torch.log(self.char_prior[y].clamp_min(1e-12)).mean()
                total = total + l_er + self.ramp * self.char_weight * l_conf
                logs.update(char_er=l_er.detach(), char_I=(1 - l_er / l_prior).detach())
            else:
                total = total + 0 * sum(p.sum() for p in e.parameters())
        return total, logs
