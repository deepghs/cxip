"""
match_witha_split.py 的补充: witha 只标了 9,480 个画师, 指纹对不上; 这里改用 isek-ai/danbooru-tags-2024
(+ danbooru_newest_index 兜底) 的完整画师信息, 图片仍是 ws-full, 用同样的切分逻辑和指纹搜索筛选条件。
"""
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_danbooru_artist import load_artists  # noqa: E402
from match_witha_split import DATA, TARGET, build_labels, fingerprint, split_and_sub  # noqa: E402


def write(rows, out, tr=('train',), te=('test',), multi='drop', k=30):
    """写出选定组合的 label 文件 (格式同原 /nfs/danbooru-w640-ws-full-artist/*.json)"""
    train, test = build_labels(rows, tr, te, multi, k)
    merged, test_split, sub = split_and_sub(train, test)
    test_classes = set(test_split.values())
    train_split = {i: a for i, a in merged.items() if a not in test_classes}
    out.mkdir(parents=True, exist_ok=True)
    for name, d in [('train_labels', train), ('test_labels', test), ('train_split', train_split),
                    ('test_split', test_split), ('test_split_100', sub)]:
        with open(out / f'{name}.json', 'w') as f:
            json.dump(d, f, ensure_ascii=False)
    print(f'{out}: train_split {len(train_split)} / {len(set(train_split.values()))} artists, '
          f'test_split {len(test_split)} / 800, test100 fingerprint {fingerprint(sub)} (card {TARGET})')


def main():
    with open(DATA / 'wsfull_id_split.json') as f:
        wsfull = {int(k): v for k, v in json.load(f).items()}
    art = load_artists()
    art = art[art['id'].isin(wsfull.keys())].dropna(subset=['artist'])
    # isek: 多画师用 ', ' 分隔 (空格已换成 _); newest_index 的 tag_string_artist 用空格分隔
    rows = {int(i): (wsfull[int(i)], s.split(', ') if ', ' in s else s.split(' '))
            for i, s in zip(art['id'], art['artist']) if s}
    print(f'{len(rows)} ws-full images with artist info', flush=True)
    if len(sys.argv) > 1 and sys.argv[1] == '--write':  # 最接近卡片指纹的组合: train/test (不用 val), 单画师, >= 30 张
        write(rows, Path(sys.argv[2]))
        return
    results = []
    for (tr, te), multi, k in itertools.product(
            [(('train',), ('test',)), (('train',), ('test', 'val')), (('train', 'val'), ('test',))],
            ['drop', 'first'], [1, 2, 5, 10, 20, 30, 40, 50, 100]):
        train, test = build_labels(rows, tr, te, multi, k)
        merged, _, sub = split_and_sub(train, test)
        fp = fingerprint(sub)
        name = f'isek|tr={"+".join(tr)}|te={"+".join(te)}|multi={multi}|min={k}'
        results.append((fp == TARGET, abs(fp[0] - TARGET[0]), name, fp))
        print(('MATCH ' if fp == TARGET else '      ') + f'{name}: images={len(merged)} '
              f'artists={len(set(merged.values()))} test100={fp}', flush=True)
    print('closest:', *sorted(results, key=lambda r: (not r[0], r[1]))[:5], sep='\n  ')


if __name__ == '__main__':
    main()
