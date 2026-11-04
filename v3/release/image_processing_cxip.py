# SPDX-License-Identifier: AGPL-3.0-only
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature


def _to_rgb_white(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        canvas = Image.new("RGB", rgba.size, (255, 255, 255))
        canvas.paste(rgba, mask=rgba.getchannel("A"))
        return canvas
    return img.convert("RGB")


class CxIPImageProcessor(BaseImageProcessor):
    """Two input contracts:

    mode="pad_square" (CCIP): composite on white, pad to a centered white square, bicubic resize to size x size.
    mode="native" (CSIP): keep the native pixel scale (brush strokes and texture are style signal): shrink so the
        short side is at most `short_side` (never upscale), cap the long side at `max_side`, round both sides to a
        multiple of `patch_size`. Outputs differ in size; a list of tensors is returned unless all shapes match.
    Both normalize with ImageNet mean / std.
    """

    model_input_names = ["pixel_values"]

    def __init__(self, mode: str = "pad_square", size: int = 512, short_side: int = 640, max_side: int = 1280,
                 patch_size: int = 16, image_mean=(0.485, 0.456, 0.406), image_std=(0.229, 0.224, 0.225), **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.size = size
        self.short_side = short_side
        self.max_side = max_side
        self.patch_size = patch_size
        self.image_mean = list(image_mean)
        self.image_std = list(image_std)

    def _resize(self, img: Image.Image) -> Image.Image:
        if self.mode == "pad_square":
            side = max(img.size)
            sq = Image.new("RGB", (side, side), (255, 255, 255))
            sq.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
            return sq.resize((self.size, self.size), Image.Resampling.BICUBIC)
        if self.mode == "native":
            m = min(img.size)
            if self.short_side and m > self.short_side:
                s = self.short_side / m
                img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))),
                                 Image.Resampling.BICUBIC)
            w, h = img.size
            s = min(1.0, self.max_side / max(w, h))
            p = self.patch_size
            return img.resize((max(p, round(w * s / p) * p), max(p, round(h * s / p) * p)), Image.Resampling.BICUBIC)
        raise ValueError(f"unknown mode {self.mode!r}")

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        x = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1).float().div_(255)
        mean = torch.tensor(self.image_mean)[:, None, None]
        std = torch.tensor(self.image_std)[:, None, None]
        return (x - mean) / std

    def preprocess(self, images: Union[Image.Image, List[Image.Image]], return_tensors: Optional[str] = "pt",
                   **kwargs) -> BatchFeature:
        if isinstance(images, (Image.Image, np.ndarray)):
            images = [images]
        xs = [self._to_tensor(self._resize(_to_rgb_white(im if isinstance(im, Image.Image) else Image.fromarray(im))))
              for im in images]
        if all(x.shape == xs[0].shape for x in xs):
            return BatchFeature({"pixel_values": torch.stack(xs)})
        return BatchFeature({"pixel_values": xs})
