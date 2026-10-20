"""
CCIP 数据:
- 训练: character_similarity images_v1.tar.xz (series/character/*.jpg) -> ccip_v1/, 再按原 scripts/data/proc_data.py 的意图
  建 ccip_v1_flat/{series}_{character}/ (目录软链, 原脚本对单个文件建链会冲突)
- 评测 (原 deepghs/dinov2_ccip_eval 不可访问): 从 zerochan-character-w640-ws-m50-full 的 test split 固定抽样
  zerochan_test_c50x15 (~750 张, 规模同 HF 卡片) 和 zerochan_test_c500x10 (5000 张)
"""
import json
import os
import random
import subprocess
import tarfile
from collections import defaultdict
from pathlib import Path

from huggingface_hub import snapshot_download

DATA = Path(os.environ.get('REPRO_ROOT', '/data/neko/cxip_repro')) / 'data'


def prepare_train():
    src = DATA / 'ccip_v1'
    if not src.exists():
        src.mkdir()
        subprocess.run(['tar', 'xJf', str(DATA / 'zips/images_v1.tar.xz'), '-C', str(src)], check=True)
    flat = DATA / 'ccip_v1_flat'
    flat.mkdir(exist_ok=True)
    n = 0
    for char_dir in sorted(src.glob('*/*')):
        if char_dir.is_dir():
            link = flat / f'{char_dir.parent.name}_{char_dir.name}'
            if not link.exists():
                link.symlink_to(char_dir.resolve(), target_is_directory=True)
            n += 1
    print(f'ccip_v1_flat: {n} characters')


def prepare_eval(seed=42):
    root = Path(snapshot_download('animetimm/zerochan-character-w640-ws-m50-full', repo_type='dataset',
                                  allow_patterns=['test/*']))
    by_char = defaultdict(list)
    for t in sorted((root / 'test').glob('*.tar')):
        with tarfile.open(t) as tf:
            for m in tf:
                if m.name.endswith('.json'):
                    by_char[json.load(tf.extractfile(m))['character']].append((t, m.name[:-5]))
    print(f'zerochan test: {sum(map(len, by_char.values()))} images, {len(by_char)} characters')

    for n_char, n_img in [(50, 15), (500, 10)]:
        rng = random.Random(seed)
        chars = rng.sample(sorted(c for c, v in by_char.items() if len(v) >= n_img), n_char)
        picks = {c: rng.sample(sorted(by_char[c]), n_img) for c in chars}
        out = DATA / f'zerochan_test_c{n_char}x{n_img}'
        need = defaultdict(dict)
        for c, items in picks.items():
            d = out / c.replace('/', '_')
            d.mkdir(parents=True, exist_ok=True)
            for t, key in items:
                need[t][key + '.webp'] = d / f'{key}.webp'
        for t, files in need.items():
            with tarfile.open(t) as tf:
                for m in tf:
                    if m.name in files:
                        files[m.name].write_bytes(tf.extractfile(m).read())
        print(f'{out.name}: {n_char * n_img} images')


if __name__ == '__main__':
    prepare_train()
    prepare_eval()
