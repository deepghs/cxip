import numpy as np
import pandas as pd
import torch

from v3.debias import ConceptTargets, Debiaser, build_char_targets, build_tag_targets

VOCAB = ['1girl', 'long_hair', 'monochrome', 'red_eyes']


def test_tag_targets_drop_style_tags_and_none():
    packed, has, vocab = build_tag_targets(['1girl, long hair, monochrome', None, 'red eyes, unknown tag'], VOCAB)
    assert vocab == ['1girl', 'long_hair', 'red_eyes']  # monochrome 属于画风, 不擦
    bits = np.unpackbits(packed, axis=1, count=len(vocab))
    assert bits.tolist() == [[1, 1, 0], [0, 0, 0], [0, 0, 1]]
    assert has.tolist() == [True, False, True]


def test_char_targets_single_and_min_count():
    chars = [np.array(['a']), np.array(['a']), np.array(['b']), np.array(['a', 'b']), None, np.array([])]
    ids, names = build_char_targets(chars, min_count=2)
    assert names == ['a']
    assert ids.tolist() == [0, 0, -1, -1, -1, -1]


def _concepts(monkeypatch):
    import v3.debias as D
    monkeypatch.setattr(D, '_tag_vocab', lambda v=None: [t for t in VOCAB if t not in D.STYLE_TAGS])
    df = pd.DataFrame(dict(general=['1girl, long hair', '1girl, red eyes', None, 'long hair'] * 4,
                           characters=[np.array(['a']), np.array(['b']), np.array(['a']), None] * 4))
    return ConceptTargets(df, char_min_count=2)


def test_debiaser_gradient_routing(monkeypatch):
    torch.manual_seed(0)
    c = _concepts(monkeypatch)
    db = Debiaser(8, c, tag_weight=1.0, char_weight=1.0, hidden=16)
    t = c.batch(np.arange(16), 'cpu')
    assert t['tags'].shape == (16, 3) and t['char'].tolist()[:4] == [0, 1, 0, -1]
    emb = torch.randn(16, 8, requires_grad=True)
    loss, logs = db(emb, t)
    assert {'tag_er', 'tag_I', 'char_er', 'char_I'} <= set(logs)
    # 擦除器参数只收到擦除器路 (detach 的 embedding) 的梯度; embedding 只收到混淆路的梯度
    g_emb = torch.autograd.grad(loss, emb, retain_graph=True)[0]
    l_conf_only = loss - logs['tag_er'] - logs['char_er']
    g_emb2 = torch.autograd.grad(l_conf_only, emb, retain_graph=True)[0]
    assert torch.allclose(g_emb, g_emb2)
    loss.backward()
    assert all(p.grad is not None for p in db.parameters())


def test_confusion_optimum_is_prior(monkeypatch):
    """擦除器恰好输出先验时, 混淆路对 embedding 的梯度为 0"""
    c = _concepts(monkeypatch)
    db = Debiaser(8, c, tag_weight=1.0, char_weight=1.0, hidden=16)
    for name, prior in (('tags', torch.logit(db.tag_prior)), ('char', torch.log(db.char_prior))):
        last = db.erasers[name][-1]
        torch.nn.init.zeros_(last.weight)
        with torch.no_grad():
            last.bias.copy_(prior)
    emb = torch.randn(16, 8, requires_grad=True)
    loss, logs = db(emb, c.batch(np.arange(16), 'cpu'))
    g = torch.autograd.grad(loss - logs['tag_er'] - logs['char_er'], emb)[0]
    assert g.abs().max() < 1e-6


def test_conditional_targets_leave_one_out(monkeypatch):
    import v3.debias as D
    monkeypatch.setattr(D, '_tag_vocab', lambda v=None: [t for t in VOCAB if t not in D.STYLE_TAGS])
    # 画师 0: 3 张都是角色 a / 1girl; 画师 1: a, b
    df = pd.DataFrame(dict(label=[0, 0, 0, 1, 1],
                           general=['1girl', '1girl', '1girl', 'red eyes', 'long hair'],
                           characters=[np.array(['a'])] * 3 + [np.array(['a']), np.array(['b'])]))
    c = ConceptTargets(df, char_min_count=1, conditional=True, smooth=0.0)
    t = c.batch(np.arange(5), 'cpu')
    # 画师 0 的其余两张都是 a / 1girl → 条件目标就是 one-hot
    assert torch.allclose(t['char_target'][0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(t['tag_target'][0], torch.tensor([1 - 1e-4, 1e-4, 1e-4]))
    # 画师 1 的第 4 张 (a) 留一后只剩 b
    assert torch.allclose(t['char_target'][3], torch.tensor([0.0, 1.0]))
    db = Debiaser(8, c, tag_weight=1.0, char_weight=1.0, hidden=16)
    loss, logs = db(torch.randn(5, 8, requires_grad=True), t)
    loss.backward()
