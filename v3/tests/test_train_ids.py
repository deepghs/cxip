import numpy as np
import pandas as pd

from v3.train_cxip import _ids


def test_ids_handles_nullable_int_float_lists_and_strings():
    df = pd.DataFrame(dict(
        year=pd.array([2010, None, 2020], dtype='Int64'),
        y_np=np.array([1, 2, 3]),
        f=[1.0, np.nan, 3.0],
        chars=[np.array(['a']), np.array(['a', 'b']), None],
        s=['x', None, 'x']))
    assert _ids(df, 'year').tolist() == [2010, -1, 2020]
    assert _ids(df, 'y_np').tolist() == [1, 2, 3]
    assert _ids(df, 'f').tolist() == [1, -1, 3]
    assert _ids(df, 'chars').tolist() == [0, -1, -1]
    assert _ids(df, 's').tolist() == [0, -1, 0]
    assert _ids(df, 'missing') is None and _ids(df, None) is None
