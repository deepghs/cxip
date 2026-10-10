"""把 HF v2 初始化 (caformer/norm/head/sim) 和 diag_eraser_0918 预热好的 eraser / teacher head 合成 fix_csip_0918_* 的初始化权重
python repro/data/merge_eraser_init.py <v2_init.safetensors> <diag model-300.safetensors> <out.safetensors>"""
import sys

from safetensors.torch import load_file, save_file

init = load_file(sys.argv[1])
diag = load_file(sys.argv[2])
eraser = {k: v for k, v in diag.items() if k.startswith(('content_eraser.', 'char_eraser.', 'char_teacher_head.'))}
assert not set(eraser) & set(init)
print(f'init {len(init)} tensors + eraser {len(eraser)} tensors ({sorted({k.split(".")[0] for k in eraser})})')
save_file({**init, **eraser}, sys.argv[3])
