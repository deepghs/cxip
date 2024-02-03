# CSIP
安装[RainbowNeko Engine](https://github.com/IrisRainbowNeko/RainbowNekoEngine).

修改`cfgs/py/train/contrastive/csip.py`中数据和训练相关配置。

## 训练
```bash
# 多卡
neko_train --cfg cfgs/py/train/contrastive/csip.py
# 单卡
neko_train_1gpu --cfg cfgs/py/train/contrastive/csip.py
```

换InfoNCE Loss:
```bash
neko_train --cfg cfgs/py/train/contrastive/csip_info_nce.py
```

## 推理

```bash
python csip_flow.py
```
输出两两图片的相似度