import numpy as np, pandas as pd
from v3.data import dates as D

def test_twitter_time():
    s = pd.Series(['https://twitter.com/foo/status/1234567890123456789',
                   'https://x.com/a/statuses/1500000000000000000/photo/1',
                   'https://twitter.com/foo/status/20',            # pre-snowflake id
                   'https://www.pixiv.net/artworks/123', None])
    t = D.twitter_time(s)
    assert t[0] == pd.Timestamp('2020-03-02 19:54:56.824', tz='UTC')
    assert t[1] == pd.Timestamp('2022-03-05 06:47:23.309', tz='UTC')
    assert t[2:].isna().all()

def test_pixiv_time_exact_interp_and_range():
    ids = np.array([100, 200, 300]); ts = pd.to_datetime(['2010-01-01', '2012-01-01', '2011-01-01'], utc=True).values
    t, src = D.pixiv_time(pd.Series([200, 150, 250, 50, 400, None]), ids, ts)
    assert t[0] == pd.Timestamp('2012-01-01', tz='UTC') and src[0] == 'pixiv'
    assert abs((t[1] - pd.Timestamp('2011-01-01', tz='UTC')).days) <= 1 and src[1] == 'pixiv_interp'
    assert t[2] == pd.Timestamp('2012-01-01', tz='UTC')          # running max makes times monotone
    assert t[3:].isna().all() and src[3:].isna().all()          # no extrapolation

def test_correct_dates_earliest_valid():
    up = pd.to_datetime(['2020-01-01', '2020-01-01', '2020-01-01'], utc=True).to_series().reset_index(drop=True)
    tw = pd.to_datetime(['2018-05-01', None, None], utc=True).to_series().reset_index(drop=True)
    px = pd.to_datetime(['2019-01-01', '2021-06-01', None], utc=True).to_series().reset_index(drop=True)
    out = D.correct_dates(up, [(tw, 'twitter'), (px, 'pixiv')])
    assert out.date_source.tolist() == ['twitter', 'danbooru_upload', 'danbooru_upload']   # pixiv later than upload is rejected
    assert out.date[0] == pd.Timestamp('2018-05-01', tz='UTC')

def test_year_and_day():
    ts = pd.to_datetime(['2020-01-01', '2020-07-02'], utc=True).to_series()
    assert np.allclose(D.to_year(ts), [2020.0, 2020 + 183 / 365.25])
    assert D.to_day(ts).tolist() == [18262, 18445]
