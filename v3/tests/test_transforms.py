from PIL import Image

from v3.transforms import NativeCropEval, NativeFullEval, limit_short_side


def test_limit_short_side_only_downscales():
    assert limit_short_side(Image.new('RGB', (1200, 800)), 640).size == (960, 640)
    assert limit_short_side(Image.new('RGB', (800, 1600)), 640).size == (640, 1280)
    assert limit_short_side(Image.new('RGB', (500, 900)), 640).size == (500, 900)
    assert limit_short_side(Image.new('RGB', (1200, 800)), None).size == (1200, 800)
    assert limit_short_side(Image.new('RGB', (500, 900)), 640, upscale=True).size == (640, 1152)


def test_eval_transforms_apply_short_side():
    img = Image.new('RGB', (1280, 960), (10, 20, 30))
    assert NativeCropEval(384, short_side=640)(img).shape == (3, 384, 384)
    # 1280×960 → 853×640 → 16 的倍数
    assert NativeFullEval(short_side=640)(img).shape == (3, 640, 848)
    assert NativeFullEval()(img).shape == (3, 960, 1280)


def test_csip_train_gray_aug():
    import torch

    from v3.transforms import IMAGENET_MEAN, IMAGENET_STD, CSIPTrainTransform
    unnorm = lambda x: x * torch.tensor(IMAGENET_STD)[:, None, None] + torch.tensor(IMAGENET_MEAN)[:, None, None]  # noqa: E731
    img = Image.new('RGB', (500, 400), (200, 30, 30))
    x = unnorm(CSIPTrainTransform(128, zoom=(1, 1), p_gray=1.0)(img))
    assert x.shape == (3, 128, 128) and torch.allclose(x[0], x[1], atol=1e-5) and torch.allclose(x[1], x[2], atol=1e-5)
    y = unnorm(CSIPTrainTransform(128, zoom=(1, 1))(img))
    assert not torch.allclose(y[0], y[1], atol=1e-2)
