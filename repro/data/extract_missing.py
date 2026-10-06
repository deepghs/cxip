"""把 split 里有、但 train/ 目录里还没有的图片从 ws-full tar 里补出来 (只写缺的几千个文件)"""
import glob
import json
import os
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATA = Path(os.environ.get('REPRO_ROOT', '/data/neko/cxip_repro')) / 'data'
ROOT = '/data/.hf/hub/datasets--animetimm--danbooru-wdtagger-v4-w640-ws-full/snapshots/47c6dd537119785c8dc4eb8c0fb647aa9264f2ca'
IMG = DATA / 'danbooru-w640-ws-full-artist/train'


def main(split_dir):
    have = set(os.listdir(IMG))
    need = set()
    for name in ['train_split', 'test_split']:
        with open(Path(split_dir) / f'{name}.json') as f:
            need.update(k for k in json.load(f) if k not in have)
    print(f'{len(need)} images to extract', flush=True)

    def scan(t):
        n = 0
        with tarfile.open(t) as tf:
            for m in tf:
                if m.name in need:
                    tmp = IMG / (m.name + '.tmp')
                    tmp.write_bytes(tf.extractfile(m).read())
                    tmp.rename(IMG / m.name)
                    n += 1
        return n

    with ThreadPoolExecutor(16) as ex:
        total = sum(ex.map(scan, sorted(glob.glob(ROOT + '/*/*.tar'))))
    print(f'extracted {total}/{len(need)}')


if __name__ == '__main__':
    main(sys.argv[1])
