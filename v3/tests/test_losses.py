import math

import torch

from v3.losses import GatheredSigLIP, MatryoshkaLoss, SubCenterArcFace, dynamic_margins
from v3.models.heads import CalibratedSim, QueryPoolHead


def test_dynamic_margins_range_and_order():
    m = dynamic_margins([2, 6, 100, 821])
    assert torch.isclose(m[0], torch.tensor(0.5)) and torch.isclose(m[-1], torch.tensor(0.05))
    assert torch.all(m[:-1] >= m[1:])


def test_arcface_margin_increases_loss_and_fallback_monotonic():
    torch.manual_seed(0)
    x = torch.randn(8, 16)
    y = torch.randint(0, 5, (8,))
    a0 = SubCenterArcFace(16, 5, k=3, margin=0.0)
    a1 = SubCenterArcFace(16, 5, k=3, margin=0.4)
    a1.weight.data.copy_(a0.weight.data)
    assert a1(x, y) > a0(x, y)
    # θ 接近 π 时兜底分支仍随 cos 单调
    arc = SubCenterArcFace(2, 1, k=1, margin=0.5)
    arc.weight.data = torch.tensor([[1.0, 0.0]])
    feats = torch.tensor([[math.cos(t), math.sin(t)] for t in (2.5, 2.8, 3.1)])
    cos = arc.cosine(feats)[:, 0]
    m = torch.full((3,), 0.5)
    theta = cos.acos()
    out = torch.where(theta + m < math.pi, (theta + m).cos(), cos - torch.sin(math.pi - m) * m)
    assert torch.all(out[:-1] >= out[1:])


def test_siglip_single_process_ignores_self_pairs():
    sim = CalibratedSim()
    emb = torch.randn(6, 8, requires_grad=True)
    y = torch.tensor([0, 0, 1, 1, 2, 2])
    loss = GatheredSigLIP()(emb, y, sim)
    loss.backward()
    assert torch.isfinite(loss) and emb.grad is not None and sim.logit_bias.grad is not None


def test_matryoshka_and_head_shapes():
    head = QueryPoolHead(in_dim=32, out_dim=24, num_queries=4, dim=16, heads=4, hidden=32)
    out = head(torch.randn(3, 32), torch.randn(3, 10, 32))
    assert out['embedding'].shape == (3, 24) and out['cond_tokens'].shape == (3, 4, 16)
    loss_fn = MatryoshkaLoss((24, 12), num_classes=5, margins=dynamic_margins([3, 4, 5, 6, 7]))
    loss, logs = loss_fn(out['embedding'], torch.tensor([0, 1, 1]), CalibratedSim())
    loss.backward()
    assert set(logs) == {'arc24', 'arc12', 'siglip'}


def test_imprint_initializes_new_classes_once():
    torch.manual_seed(0)
    arc = SubCenterArcFace(8, 4, k=3)
    centers = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    y = torch.tensor([0, 0, 2, 2, 2])
    emb = centers[y] + 0.05 * torch.randn(5, 8)
    arc.imprint(emb, y)
    assert arc.seen.tolist() == [True, False, True, False]
    assert arc.cosine(centers[[0, 2]])[[0, 1], [0, 2]].gt(0.95).all()  # 新类中心对准类均值
    w = arc.weight.detach().clone()
    arc.imprint(emb + 1.0, y)  # 已见过的类不再改动
    assert torch.equal(w, arc.weight.detach())


def test_batch_negatives_ignore_absent_classes():
    """arc_negatives='batch': 只有当前 (全局) batch 内出现的类参与 softmax, 不在 batch 的类中心改动不影响 loss。"""
    torch.manual_seed(0)
    x, y = torch.randn(6, 16), torch.tensor([0, 0, 1, 1, 2, 2])
    for mode, should_change in (('batch', False), ('all', True)):
        loss_fn = MatryoshkaLoss((16, 8), num_classes=10, siglip_weight=0.0, imprint=False, arc_negatives=mode)
        before = loss_fn(x, y, CalibratedSim())[0]
        with torch.no_grad():
            for a in loss_fn.arcs:  # 把 batch 外的类 7 的子中心设成与第一个样本完全重合 (最难负例)
                a.weight[7 * a.k:(7 + 1) * a.k] = x[0, :a.weight.shape[1]]
        after = loss_fn(x, y, CalibratedSim())[0]
        assert (not torch.isclose(before, after)) == should_change, mode
