import json
import os

import numpy as np
import pytest

from v3.eval import featio as FI


def test_save_feats_refuses_existing_unless_overwrite(tmp_path):
    p = tmp_path / 'feats' / 's.npz'
    FI.save_feats(p, np.ones((2, 3)), ['a', 'b'])
    d = np.load(p)
    assert d['feats'].dtype == np.float16 and list(d['sample_key']) == ['a', 'b']
    with pytest.raises(FI.RefuseWrite, match='--overwrite'):
        FI.save_feats(p, np.zeros((2, 3)), ['a', 'b'])
    assert np.load(p)['feats'][0, 0] == 1
    FI.save_feats(p, np.zeros((2, 3)), ['a', 'b'], overwrite=True)
    assert np.load(p)['feats'][0, 0] == 0


@pytest.mark.parametrize('overwrite', [False, True])
def test_never_writes_through_symlink(tmp_path, overwrite):
    real = tmp_path / 'old' / 's.npz'
    real.parent.mkdir()
    np.savez(real, feats=np.ones((1, 2), np.float16))
    before = real.read_bytes()
    link = tmp_path / 'new' / 's.npz'
    link.parent.mkdir()
    os.symlink(real, link)
    with pytest.raises(FI.RefuseWrite, match='symlink'):
        FI.save_feats(link, np.zeros((1, 2)), overwrite=overwrite)
    dangling = tmp_path / 'new' / 'd.npz'
    os.symlink(tmp_path / 'nowhere.npz', dangling)
    with pytest.raises(FI.RefuseWrite, match='symlink'):
        FI.save_feats(dangling, np.zeros((1, 2)), overwrite=overwrite)
    with pytest.raises(FI.RefuseWrite, match='symlink'):
        FI.write_json(link, {'a': 1}, overwrite=overwrite)
    assert real.read_bytes() == before and not (tmp_path / 'nowhere.npz').exists()


def test_write_json_identical_is_noop_and_different_needs_overwrite(tmp_path):
    p = tmp_path / 'meta.json'
    assert FI.write_json(p, dict(scale=1.0, bias=-2.0)) is True
    mtime = p.stat().st_mtime_ns
    assert FI.write_json(p, dict(bias=-2.0, scale=1.0)) is False and p.stat().st_mtime_ns == mtime
    with pytest.raises(FI.RefuseWrite, match='different content'):
        FI.write_json(p, dict(scale=2.0, bias=-2.0))
    assert json.loads(p.read_text())['scale'] == 1.0
    assert FI.write_json(p, dict(scale=2.0, bias=-2.0), overwrite=True) is True
    assert json.loads(p.read_text())['scale'] == 2.0


def test_check_targets_fails_on_first_bad(tmp_path):
    (tmp_path / 'b.npz').write_bytes(b'x')
    FI.check_targets([tmp_path / 'a.npz'])
    with pytest.raises(FI.RefuseWrite):
        FI.check_targets([tmp_path / 'a.npz', tmp_path / 'b.npz'])
    FI.check_targets([tmp_path / 'a.npz', tmp_path / 'b.npz'], overwrite=True)


# ---------------------------------------------------------------- the three writer scripts refuse before loading a model
def _boom(*a, **k):
    raise AssertionError('model must not be loaded when a target is refused')


def _existing_and_link(tmp_path, rel):
    """tmp out dir with <rel>/s1.npz existing and <rel>/s2.npz a symlink to an outside file"""
    out = tmp_path / 'out'
    (out / rel).mkdir(parents=True)
    np.savez(out / rel / 's1.npz', feats=np.ones((1, 2), np.float16))
    outside = tmp_path / 'outside.npz'
    np.savez(outside, feats=np.ones((1, 2), np.float16))
    os.symlink(outside, out / rel / 's2.npz')
    return out, outside.read_bytes()


def test_extract_release_feats_guard(tmp_path, monkeypatch):
    from v3.scripts import extract_release_feats as X
    monkeypatch.setattr(X.AutoModel, 'from_pretrained', _boom)
    out, outside = _existing_and_link(tmp_path, 'feats')
    base = ['--model', 'm', '--out-dir', str(out)]
    with pytest.raises(FI.RefuseWrite, match='exists'):
        X.main(base + ['--sets', 's1'])
    for ow in ([], ['--overwrite']):
        with pytest.raises(FI.RefuseWrite, match='symlink'):
            X.main(base + ['--sets', 's2'] + ow)
    with pytest.raises(AssertionError, match='must not be loaded'):      # --overwrite on a regular file passes the guard
        X.main(base + ['--sets', 's1', '--overwrite'])
    assert (tmp_path / 'outside.npz').read_bytes() == outside


def test_gap_feats_guard(tmp_path, monkeypatch):
    from v3.scripts import gap_feats as G
    monkeypatch.setattr(G, 'load_exp_model', _boom)
    out, outside = _existing_and_link(tmp_path, 'feats')
    with pytest.raises(FI.RefuseWrite, match='exists'):
        G.main(['exp', 'e', str(out), '--sets', 's1'])
    for ow in ([], ['--overwrite']):
        with pytest.raises(FI.RefuseWrite, match='symlink'):
            G.main(['exp', 'e', str(out), '--sets', 's2'] + ow)
    os.symlink(tmp_path / 'outside.npz', out / 'meta.json')
    with pytest.raises(FI.RefuseWrite, match='symlink'):
        G.main(['exp', 'e', str(out), '--sets', 's3', '--overwrite'])
    assert (tmp_path / 'outside.npz').read_bytes() == outside


def test_eval_kaloscope_guard_and_meta(tmp_path, monkeypatch):
    try:
        from v3.scripts import eval_kaloscope as K
    except Exception as e:                    # third-party Kaloscope code not on this machine
        pytest.skip(f'kaloscope code unavailable: {e}')
    out, outside = _existing_and_link(tmp_path, 'proj256/feats')
    monkeypatch.setattr(K, 'load_model', _boom)
    with pytest.raises(FI.RefuseWrite, match='exists'):
        K.main(['--out', str(out), '--sets', 's1'])
    for ow in ([], ['--overwrite']):
        with pytest.raises(FI.RefuseWrite, match='symlink'):
            K.main(['--out', str(out), '--sets', 's2'] + ow)
    assert (tmp_path / 'outside.npz').read_bytes() == outside
    # meta.json: identical -> not rewritten; different -> refused unless --overwrite (--print-cmds skips extraction)
    monkeypatch.setattr(K, 'load_model', lambda: (None, 2.0, -1.0))
    K.main(['--out', str(out), '--print-cmds'])
    meta = out / 'meta.json'
    mtime = meta.stat().st_mtime_ns
    K.main(['--out', str(out), '--print-cmds'])
    assert meta.stat().st_mtime_ns == mtime
    monkeypatch.setattr(K, 'load_model', lambda: (None, 3.0, -1.0))
    with pytest.raises(FI.RefuseWrite, match='different content'):
        K.main(['--out', str(out), '--print-cmds'])
    assert json.loads(meta.read_text())['scale'] == 2.0
    K.main(['--out', str(out), '--print-cmds', '--overwrite'])
    assert json.loads(meta.read_text())['scale'] == 3.0
