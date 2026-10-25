"""CxIP v3 数据增强。

CCIP: 只做不改变角色身份的增强 (禁用色相/饱和度/灰度: 发色、瞳色是身份特征)。
"""
import io
import random

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision.transforms import v2 as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def to_rgb_white(img: Image.Image) -> Image.Image:
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        rgba = img.convert('RGBA')
        canvas = Image.new('RGB', rgba.size, (255, 255, 255))
        canvas.paste(rgba, mask=rgba.getchannel('A'))
        return canvas
    return img.convert('RGB')


def pad_square(img: Image.Image) -> Image.Image:
    side = max(img.size)
    out = Image.new('RGB', (side, side), (255, 255, 255))
    out.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    return out


def to_tensor(img: Image.Image) -> torch.Tensor:
    x = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1).float().div_(255)
    return T.functional.normalize(x, IMAGENET_MEAN, IMAGENET_STD)


class EvalTransform:
    """与发布版推理一致: 白底 pad 方形 → bicubic resize。"""

    def __init__(self, res: int):
        self.res = res

    def __call__(self, img):
        return to_tensor(pad_square(to_rgb_white(img)).resize((self.res, self.res), Image.Resampling.BICUBIC))


class CCIPTrainTransform:
    def __init__(self, res: int, scale=(0.4, 1.0), ratio=(3 / 4, 4 / 3), rotate: float = 8.0, bc: float = 0.05,
                 p_compress: float = 0.3, p_blur: float = 0.1):
        self.res = res
        self.crop = T.RandomResizedCrop(1, scale=scale, ratio=ratio)  # 只用它的 get_params
        self.scale, self.ratio = scale, ratio
        self.rotate = rotate
        self.bc = bc
        self.p_compress = p_compress
        self.p_blur = p_blur

    def __call__(self, img):
        img = to_rgb_white(img)
        # 先在原图上裁 (避免裁到大片白边), 再 pad 成方形
        i, j, h, w = T.RandomResizedCrop.get_params(img, self.scale, self.ratio)
        img = img.crop((j, i, j + w, i + h))
        if random.random() < 0.5:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        img = pad_square(img).resize((self.res, self.res), Image.Resampling.BICUBIC)
        if self.rotate > 0:
            img = img.rotate(random.uniform(-self.rotate, self.rotate), Image.Resampling.BILINEAR, fillcolor=(255, 255, 255))
        if self.p_blur > 0 and random.random() < self.p_blur:
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.1, 1.0)))
        if self.p_compress > 0 and random.random() < self.p_compress:
            buf = io.BytesIO()
            fmt = random.choice(['JPEG', 'WEBP'])
            img.save(buf, fmt, quality=random.randint(40, 95))
            img = Image.open(io.BytesIO(buf.getvalue())).convert('RGB')
        x = to_tensor(img)
        if self.bc > 0:  # 亮度/对比度, 每张一个随机系数 (在归一化后的空间里等价于仿射)
            x = x * (1 + random.uniform(-self.bc, self.bc)) + random.uniform(-self.bc, self.bc)
        return x


# ------------------------------------------------------------------ CSIP: 原始像素尺度裁剪, 不动颜色/清晰度
def _crop_or_pad(img: Image.Image, res: int, i: int, j: int) -> Image.Image:
    """从 (j, i) 处裁 res×res, 超出原图的部分用白色填充。"""
    out = Image.new('RGB', (res, res), (255, 255, 255))
    out.paste(img.crop((j, i, min(j + res, img.width), min(i + res, img.height))), (0, 0))
    return out


class CSIPTrainTransform:
    """画风增强: 只允许轻微缩放 (zoom) + 原尺度随机裁剪 + 水平翻转。

    禁用颜色、模糊、锐化、压缩: 用色、笔触锐度、纹理本身就是画风信号。
    p_gray (默认 0, 消融用): 以该概率转灰度 —— 同画师彩色/黑白图配对的召回明显偏低 (0.64 vs 0.77)。
    """

    def __init__(self, res: int, zoom=(0.8, 1.25), p_flip: float = 0.5, p_gray: float = 0.0):
        self.res, self.zoom, self.p_flip, self.p_gray = res, zoom, p_flip, p_gray

    def __call__(self, img):
        img = to_rgb_white(img)
        z = random.uniform(*self.zoom)
        if abs(z - 1) > 1e-3:
            img = img.resize((max(1, round(img.width * z)), max(1, round(img.height * z))), Image.Resampling.BICUBIC)
        i = random.randint(0, max(0, img.height - self.res))
        j = random.randint(0, max(0, img.width - self.res))
        img = _crop_or_pad(img, self.res, i, j)
        if random.random() < self.p_flip:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if self.p_gray and random.random() < self.p_gray:
            img = img.convert('L').convert('RGB')
        return to_tensor(img)


def limit_short_side(img: Image.Image, short_side, upscale: bool = False) -> Image.Image:
    """短边超过 short_side 时等比缩小 (只缩不放)。

    CSIP 训练数据 (danbooru-w640-ws) 已统一为短边 640, "原尺度" 指的是这个尺度; 尺寸更大的外部图
    (csip_eval_public 短边中位数 800) 必须先缩到同一尺度, 否则笔触被放大 1.3-1.7 倍。"""
    m = min(img.size)
    if not short_side or m == short_side or (m < short_side and not upscale):  # upscale: 小图也放大到该尺度
        return img
    s = short_side / m
    return img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.Resampling.BICUBIC)


class NativeCropEval:
    """CSIP 评测: 原尺度中心裁 res×res (与训练的尺度一致)。short_side: 先把短边限制到训练数据的尺度。"""

    def __init__(self, res: int, short_side: int = None):
        self.res, self.short_side = res, short_side

    def __call__(self, img):
        img = limit_short_side(to_rgb_white(img), self.short_side)
        return to_tensor(_crop_or_pad(img, self.res, max(0, (img.height - self.res) // 2),
                                      max(0, (img.width - self.res) // 2)))


class NativeFullEval:
    """CSIP 评测 (可选协议): 整图原尺度, 长边超过 max_side 时等比缩小, 边长取 16 的倍数 (形状不一, 需分桶批处理)。"""

    def __init__(self, max_side: int = 1280, patch: int = 16, short_side: int = None, upscale: bool = False):
        self.max_side, self.patch, self.short_side, self.upscale = max_side, patch, short_side, upscale

    def __call__(self, img):
        img = limit_short_side(to_rgb_white(img), self.short_side, self.upscale)
        w, h = img.size
        s = min(1.0, self.max_side / max(w, h))
        p = self.patch
        W, H = max(p, round(w * s / p) * p), max(p, round(h * s / p) * p)
        return to_tensor(img.resize((W, H), Image.Resampling.BICUBIC))
