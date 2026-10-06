# neko_train_1gpu --cfg cfgs/py/train/contrastive/ccip_wds.py
# CUDA_VISIBLE_DEVICES=1 neko_train_1gpu --cfg cfgs/py/train/contrastive/csip.py
# neko_train --cfg cfgs/py/train/contrastive/ccip_wds.py

# 新版 CCIP (SigLIP, anime_dino 初始化)
# neko_train --cfg cfgs/py/train/contrastive/ccip.py
# CSIP v2.1 (content eraser + character eraser)
neko_train --cfg cfgs/py/train/contrastive/csip2_1.py
