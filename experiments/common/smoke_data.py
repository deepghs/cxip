"""CPU 上构建训练配置里的各个数据集并取一个样本, 检查数据/标签文件是否对得上: python experiments/common/smoke_data.py <cfg.py>"""
import sys
import time

import hydra
from rainbowneko.parser import load_config

_, cfg = load_config(sys.argv[1])
for name, ds_cfg in cfg.data_train.items():
    t = time.time()
    ds = hydra.utils.instantiate(ds_cfg)
    ds = ds() if callable(ds) and not hasattr(ds, 'build_bucket') else ds
    ds.build_bucket(bs=128, world_size=4)
    print(name, 'len', len(ds), f'build {time.time() - t:.0f}s', flush=True)
    item = ds[0]
    print({k: getattr(v, 'shape', v) for k, v in item.items()}, flush=True)
