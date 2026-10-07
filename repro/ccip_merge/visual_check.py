"""
合并 CCIP 训练集: 用 HF AnimeCCIP 对 codex 的判定做视觉复核
每个出现在 same/variant/different 候选对里的类别抽 <=8 张图提特征, 对每个候选对算跨类平均同角色概率
sigmoid(exp(logit_scale)*cos + logit_bias), 以及各自类内的平均概率作参照。
输出 data/ccip_merge/visual_check.parquet
python repro/ccip_merge/visual_check.py --device cuda:0
"""
import argparse
import glob
import io
import json
import os
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

R = Path(os.environ.get('REPRO_ROOT', '/data/neko/cxip_repro'))
SRC = R / 'data/ccip_merge'
ZC = R / 'data/zerochan_m50_800k'
CS = R / 'data/zips/character_similarity'
PER_CLASS = 8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--model', default='dghstmp/AnimeCCIP')
    args = parser.parse_args()
    classes = json.load(open(SRC / 'classes.json'))
    cands = json.load(open(SRC / 'candidates.json'))
    res = {x['id']: x for f in glob.glob(str(SRC / 'judge/batch_*.json')) for x in json.load(open(f))['results']}
    need = sorted({cands[i][k] for i in res for k in 'ab'})
    img = pd.read_parquet(SRC / 'images.parquet')
    img = img[img['cid'].isin(set(need))]
    img['crop'] = img['path'].str.contains(r'_person\d+\.', regex=True)
    img = img.sample(frac=1, random_state=0).sort_values(['cid', 'crop'], kind='stable')
    img = img.drop_duplicates(['cid', 'zid']).groupby('cid').head(PER_CLASS)
    print(len(need), 'classes,', len(img), 'images', flush=True)
    want = {(s, p): i for i, (s, p) in enumerate(zip(img['source'], img['path']))}
    data = [None] * len(img)
    for tp_name, group in img[img['source'] == 'zc800k'].groupby(img['path'].str.split(':').str[0]):
        with tarfile.open(ZC / tp_name) as tf:
            members = {m.name.rsplit('.', 1)[0]: m for m in tf if not m.name.endswith('.json')}
            for p in group['path']:
                data[want[('zc800k', p)]] = tf.extractfile(members[p.split(':')[1]]).read()
    print('zc800k loaded', flush=True)
    for source, arc, cmd in [('v1', 'images_pruned_v1.tar.xz', ['xz', '-T8', '-dc']), ('v2', 'images_v2.tar.gz', ['pigz', '-dc'])]:
        proc = subprocess.Popen(cmd + [str(CS / arc)], stdout=subprocess.PIPE, bufsize=1 << 24)
        with tarfile.open(fileobj=proc.stdout, mode='r|') as tf:
            for m in tf:
                i = want.get((source, m.name)) if m.isfile() else None
                if i is not None:
                    data[i] = tf.extractfile(m).read()
        proc.wait()
        print(source, 'loaded', flush=True)
    assert all(d is not None for d in data)

    model = AutoModel.from_pretrained(args.model, trust_remote_code=True).to(args.device).eval()
    processor = AutoImageProcessor.from_pretrained(args.model, trust_remote_code=True)
    scale, bias = float(model.sim.logit_scale.exp().detach()), float(model.sim.logit_bias.detach())
    feats = []
    with torch.no_grad():
        for s in range(0, len(data), 256):
            ims = [Image.open(io.BytesIO(d)).convert('RGB') for d in data[s:s + 256]]
            x = processor(ims, return_tensors='pt')['pixel_values'].to(args.device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                f = model(x)['features']
            feats.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu())
            if s % 25600 == 0:
                print('feat', s, flush=True)
    feats = torch.cat(feats)
    rows_of = {c: np.flatnonzero(img['cid'].values == c) for c in need}

    def prob(a, b, same):
        s = feats[rows_of[a]] @ feats[rows_of[b]].T
        p = torch.sigmoid(s * scale + bias)
        if same:
            if len(rows_of[a]) < 2:
                return float('nan')
            p = p[~torch.eye(len(p), dtype=torch.bool)]
        return float(p.mean())
    within = {c: prob(c, c, True) for c in need}
    out = []
    for i, r in res.items():
        a, b = cands[i]['a'], cands[i]['b']
        out.append(dict(id=i, verdict=r['verdict'], reason=r['reason'], shared=cands[i]['shared'],
                        a=classes[a]['source'] + ':' + classes[a]['key'], b=classes[b]['source'] + ':' + classes[b]['key'],
                        na=classes[a]['n'], nb=classes[b]['n'], cross=prob(a, b, False), within_a=within[a], within_b=within[b]))
    df = pd.DataFrame(out)
    df.to_parquet(SRC / 'visual_check.parquet')
    for v in ['same', 'variant', 'different', 'unsure']:
        d = df[df.verdict == v]['cross']
        print(v, len(d), 'cross prob quantiles', d.quantile([.01, .05, .25, .5, .75, .95]).round(3).tolist())


if __name__ == '__main__':
    main()
