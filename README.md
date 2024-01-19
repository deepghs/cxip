# CSIP
修改`cfgs/py/train/contrastive/csip.py`中数据和训练相关配置。

启动训练
```bash
# 多卡
neko_train --cfg cfgs/py/train/contrastive/csip.py
# 单卡
neko_train_1gpu --cfg cfgs/py/train/contrastive/csip.py
```