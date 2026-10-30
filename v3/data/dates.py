"""Original-publication date correction for Danbooru posts.

Danbooru ``created_at`` is the upload time, often years after the work first appeared. Better estimates:
Twitter status id (snowflake timestamp), Pixiv artwork time (exact from the pixiv index, else monotone
interpolation over artwork id). ``correct_dates`` takes the earliest plausible candidate (never later than upload).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

SOURCES = ('twitter', 'pixiv', 'pixiv_interp', 'danbooru_upload')
_TW_RE = r'(?:twitter\.com|x\.com)/[^/]+/status(?:es)?/(\d{1,19})'
_TW_EPOCH_MS = 1288834974657
_MIN_DATE = pd.Timestamp('2005-01-01', tz='UTC')


def _utc(s) -> pd.Series:
    return pd.to_datetime(s, utc=True).astype('datetime64[ns, UTC]')


def _ns(s: pd.Series) -> np.ndarray:
    return _utc(s).dt.tz_localize(None).to_numpy('datetime64[ns]').view(np.int64)      # NaT -> int64 min


def twitter_time(source: pd.Series) -> pd.Series:
    """Snowflake time from a twitter/x status url; NaT if no url or the id is not a snowflake."""
    x = source.astype('object').where(source.notna(), '').astype(str).str.extract(_TW_RE)[0]
    tid = x.map(lambda v: int(v) if isinstance(v, str) and int(v) < 2 ** 63 else -1).astype(np.int64)
    ms = (tid // 2 ** 22) + _TW_EPOCH_MS
    ok = (tid > 2 ** 32) & (ms < 1.83e12)
    return pd.to_datetime(ms.where(ok).astype('Int64'), unit='ms', utc=True).astype('datetime64[ns, UTC]').rename(None)


def pixiv_time(pixiv_ids: pd.Series, index_ids: np.ndarray, index_times: np.ndarray) -> tuple[pd.Series, pd.Series]:
    """(time, source): exact index hit -> 'pixiv'; inside [min id, max id] -> monotone interp 'pixiv_interp'; else NaT/None."""
    it = np.asarray(index_times)
    it = (it if it.dtype.kind == 'M' else it.astype('datetime64[s]')).astype('datetime64[ns]').astype(np.int64)
    ids, first = np.unique(np.asarray(index_ids, dtype=np.int64), return_index=True)
    raw = it[first]; mono = np.maximum.accumulate(raw)
    q = pd.to_numeric(pixiv_ids, errors='coerce').to_numpy(dtype=np.float64)
    pos = np.clip(np.searchsorted(ids, q), 0, len(ids) - 1)
    hit = ids[pos] == q                                    # NaN never equals
    inside = (q >= ids[0]) & (q <= ids[-1])
    val = np.where(hit, raw[pos], np.interp(np.nan_to_num(q), ids, mono.astype(np.float64)).astype(np.int64))
    ok = hit | inside
    t = pd.Series(pd.to_datetime(pd.Series(np.where(ok, val, 0), index=pixiv_ids.index), unit='ns', utc=True).where(ok))
    src = pd.Series(np.where(hit, 'pixiv', np.where(inside, 'pixiv_interp', None)), index=pixiv_ids.index, dtype=object)
    return t.astype('datetime64[ns, UTC]'), src


def correct_dates(upload: pd.Series, candidates: list[tuple[pd.Series, pd.Series | str]]) -> pd.DataFrame:
    """Earliest of (upload, valid candidates); valid = in [2005-01-01, upload + 1 day]. Ties go to the candidate."""
    up = _utc(upload)
    cols, srcs = [], []
    for t, s in candidates:
        t = _utc(t)
        t = t.where((t >= _MIN_DATE) & (t <= up + pd.Timedelta(days=1)))
        cols.append(_ns(t))
        srcs.append(np.full(len(t), s, dtype=object) if isinstance(s, str) else s.to_numpy(dtype=object))
    cols.append(_ns(up)); srcs.append(np.full(len(up), 'danbooru_upload', dtype=object))
    T = np.stack(cols, 1); S = np.stack(srcs, 1)
    nat = np.iinfo(np.int64).min
    k = np.where(T == nat, np.iinfo(np.int64).max, T).argmin(1)    # upload is last, so a tie picks the candidate
    r = np.arange(len(up))
    ok = T[r, k] != nat
    date = pd.Series(T[r, k].view('datetime64[ns]'), index=up.index).dt.tz_localize('UTC').where(ok)
    return pd.DataFrame({'date': date, 'date_source': pd.Series(S[r, k], index=up.index, dtype=object).where(ok)})


def to_year(ts: pd.Series) -> np.ndarray:
    return (ts.dt.year + (ts.dt.dayofyear - 1) / 365.25).to_numpy(dtype=np.float64)


def to_day(ts: pd.Series) -> np.ndarray:
    return _ns(ts) // 86_400_000_000_000
