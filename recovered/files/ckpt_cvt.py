from safetensors.torch import load_file


ckpt = load_file('/data/anime_dino/exps/anime_dinov2_bs128-lr1e-3/ckpts/model-191000.safetensors')

print(ckpt.keys())





