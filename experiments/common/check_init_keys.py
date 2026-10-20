"""检查初始化 safetensors 对 CSIP debias 模型的覆盖: python experiments/common/check_init_keys.py <cfg.py> <init.safetensors>"""
import importlib.util
import sys

from safetensors.torch import load_file

spec = importlib.util.spec_from_file_location('cfg', sys.argv[1])
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)
model = cfg.CAFormerStyleDebiasBackbone('caformer_b36', input_resolution=384, tagger=cfg.FrozenTagger(cfg.TAGGER, tag_file=cfg.CONTENT_TAGS),
                                        ccip=cfg.FrozenCCIP(cfg.CCIP), num_char_cls=100, grl_alpha=1.0)
state = load_file(sys.argv[2])
own = model.state_dict()
loaded = [k for k in state if k in own and own[k].shape == state[k].shape]
bad_shape = [k for k in state if k in own and own[k].shape != state[k].shape]
unexpected = [k for k in state if k not in own]
missing = sorted({'.'.join(k.split('.')[:2]) for k in own if k not in state})
print(f'loaded {len(loaded)}/{len(state)}, shape mismatch {bad_shape}, unexpected {unexpected[:5]}')
print('model keys not in init (grouped):', missing)
