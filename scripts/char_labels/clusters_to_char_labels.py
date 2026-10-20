"""
[RECONSTRUCTED] 把 scripts/char_labels/ccip_cls.py 的聚类结果转成 CSIP v2.1 dataset_dechar 用的角色伪标签 train_char_labels.json
(服务器上做这一步的脚本已丢失)

输入:  ccip_clusters_<ts>.json  {"meta": ..., "files": ["12345.webp.pt", ...], "labels": [3, -1, ...]}
       files 是相对 --feat-dir 的特征文件路径，由 cfgs/py/infer/ccip_save.py 保存为 {id}.pt
输出:  {"12345.webp": "c3", ...}，噪声点 (-1) 丢弃，只保留 style_label_file 里有的 id
       StyleCharImageLabelSource 按 '||' 切分多角色，这里每张图只有一个簇

用法:
  python scripts/char_labels/clusters_to_char_labels.py \
    --clusters /nfs/danbooru-w640-ws-full-artist/ccip_clusters_xxx.json \
    --style-label-file /nfs/danbooru-w640-ws-full-artist/train_split.json \
    --output /nfs/danbooru-w640-ws-full-artist/train_char_labels.json
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def file_to_id(name: str) -> str:
    name = Path(name).name
    return name[:-len('.pt')] if name.endswith('.pt') else name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clusters', type=Path, required=True)
    parser.add_argument('--style-label-file', type=Path, required=True, help='train_split.json, {id: artist}')
    parser.add_argument('--output', type=Path, required=True)
    # 以下两个过滤默认关闭 (原脚本是否过滤未知)
    parser.add_argument('--min-cluster-size', type=int, default=1,
                        help='丢弃小于该大小的簇 (CharBucket 会把小簇重复采样补齐到 min_cat_size)')
    parser.add_argument('--min-styles', type=int, default=1, help='丢弃画师数少于该值的簇 (单画师的簇对去角色没有帮助)')
    args = parser.parse_args()

    with open(args.clusters, 'r', encoding='utf-8') as f:
        clusters = json.load(f)
    with open(args.style_label_file, 'r', encoding='utf-8') as f:
        style_dict = json.load(f)

    noise = clusters.get('meta', {}).get('noise_label', -1)
    stem_map = {Path(k).stem: k for k in style_dict}  # 兼容按 stem 保存的特征

    char_labels = {}
    n_noise = n_missing = 0
    for file, label in zip(clusters['files'], clusters['labels']):
        if label == noise:
            n_noise += 1
            continue
        img_id = file_to_id(file)
        if img_id not in style_dict:
            img_id = stem_map.get(img_id, stem_map.get(Path(img_id).stem))
            if img_id is None:
                n_missing += 1
                continue
        char_labels[img_id] = f'c{label}'

    sizes = Counter(char_labels.values())
    styles = {}
    for img_id, c in char_labels.items():
        styles.setdefault(c, set()).add(style_dict[img_id])
    keep = {c for c, n in sizes.items() if n >= args.min_cluster_size and len(styles[c]) >= args.min_styles}
    char_labels = {k: v for k, v in char_labels.items() if v in keep}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(char_labels, f, ensure_ascii=False)

    print(f'total: {len(clusters["files"])}, noise: {n_noise}, not in style labels: {n_missing}')
    print(f'clusters kept: {len(keep)}/{len(sizes)}, images: {len(char_labels)} -> {args.output}')


if __name__ == '__main__':
    main()
