"""把 HF 上的 CSIP (dghstmp/AnimeCSIP-v2) 权重改成 CAFormerStyleDebiasBackbone 的参数名, 作为 v2.1 debias 微调的初始化
(trans_csip.py 的逆映射: backbone.* -> caformer.*, norm/head/sim 不变; eraser 等新模块保持随机初始化)
python repro/data/hf_csip_to_init.py dghstmp/AnimeCSIP-v2 /data/neko/cxip_repro/exps/init/csip_v2_init.safetensors
"""
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

repo, out = sys.argv[1], Path(sys.argv[2])
state = load_file(hf_hub_download(repo, 'model.safetensors'))
state = {('caformer.' + k[len('backbone.'):] if k.startswith('backbone.') else k): v for k, v in state.items()}
assert all(k.split('.')[0] in ('caformer', 'norm', 'head', 'sim') for k in state), sorted({k.split('.')[0] for k in state})
out.parent.mkdir(parents=True, exist_ok=True)
save_file(state, str(out))
print(f'{repo}: {len(state)} tensors -> {out}')
