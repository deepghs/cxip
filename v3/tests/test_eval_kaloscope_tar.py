import io
import tarfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from v3.eval import evalsets


def _kaloscope():
    try:
        from v3.scripts import eval_kaloscope
    except Exception as e:                    # third-party Kaloscope code / checkpoint dir not on this machine
        pytest.skip(f'kaloscope code unavailable: {e}')
    return eval_kaloscope


def test_images_reads_tar_packed_set(tmp_path, monkeypatch):
    K = _kaloscope()
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(40, 60, 4), dtype=np.uint8)
    arr[..., 3] = np.where(np.arange(60) < 30, 255, 0)[None, :]         # right half transparent -> white
    b = io.BytesIO(); Image.fromarray(arr, 'RGBA').save(b, 'PNG'); data = b.getvalue()
    (tmp_path / 'one.png').write_bytes(data)
    with tarfile.open(tmp_path / 'a.tar', 'w') as t:
        for name, payload in (('pad.bin', b'\0' * 700), ('1.png', data)):
            ti = tarfile.TarInfo(name); ti.size = len(payload); t.addfile(ti, io.BytesIO(payload))
    with tarfile.open(tmp_path / 'a.tar') as t:
        off = t.getmember('1.png').offset_data
    d = tmp_path / 'set'; d.mkdir()
    tar = str(tmp_path / 'a.tar')
    pd.DataFrame(dict(path=[tar], tar=[tar], offset=[off], size=[len(data)], label_name=['x'], label=[0],
                      sample_key=['d1'], split=['dev'])).to_parquet(d / 'index.parquet')
    monkeypatch.setitem(evalsets.INDEX_SETS, 'tmp_kal_set', d)
    df, read = evalsets.load_index_set('tmp_kal_set')
    x = K.Images(df['path'].tolist(), lambda i: read(i, decode=False))[0]
    ref = K.Images([str(tmp_path / 'one.png')])[0]                       # single-file path, same preprocessing
    assert x.shape == (3, 512, 512) and str(x.dtype) == 'torch.uint8'
    assert (x == ref).all()
    assert int(x[:, :, -1].min()) == 255                                # transparent side composited on white
