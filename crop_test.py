import numpy as np
from PIL import Image

def uniform_patch_crop_vectorized(image, target_size, patch_size):
    """
    更高效的向量化版本，使用numpy的高级索引
    """
    image = np.asarray(image)
    target_w, target_h = target_size
    patch_w, patch_h = patch_size

    if image.ndim == 3:
        orig_h, orig_w, channels = image.shape
    else:
        orig_h, orig_w = image.shape
        channels = None

    # 计算patch数量
    n_patches_x = target_w // patch_w
    n_patches_y = target_h // patch_h

    # 计算均匀采样位置
    if n_patches_x == 1:
        x_starts = np.array([(orig_w - patch_w) // 2])
    else:
        x_starts = np.linspace(0, orig_w - patch_w, n_patches_x, dtype=int)

    if n_patches_y == 1:
        y_starts = np.array([(orig_h - patch_h) // 2])
    else:
        y_starts = np.linspace(0, orig_h - patch_h, n_patches_y, dtype=int)

    # 生成patch内的相对坐标
    patch_y_offsets = np.arange(patch_h)
    patch_x_offsets = np.arange(patch_w)

    # 生成所有patch的绝对坐标
    Y_starts, X_starts = np.meshgrid(y_starts, x_starts, indexing='ij')
    Y_offsets, X_offsets = np.meshgrid(patch_y_offsets, patch_x_offsets, indexing='ij')

    # 展开成一维数组用于高级索引
    Y_starts_flat = Y_starts.flatten()  # shape: (n_patches_y * n_patches_x,)
    X_starts_flat = X_starts.flatten()

    # 为每个patch生成完整的坐标索引
    Y_indices = Y_starts_flat[:, None, None] + Y_offsets[None, :, :]  # shape: (n_patches, patch_h, patch_w)
    X_indices = X_starts_flat[:, None, None] + X_offsets[None, :, :]

    if channels is not None:
        # 彩色图像：使用高级索引提取所有patch
        patches = image[Y_indices, X_indices, :]  # shape: (n_patches, patch_h, patch_w, channels)
        # 重新排列成目标形状
        patches = patches.reshape(n_patches_y, n_patches_x, patch_h, patch_w, channels)
        result = patches.transpose(0, 2, 1, 3, 4).reshape(target_h, target_w, channels)
    else:
        # 灰度图像
        patches = image[Y_indices, X_indices]  # shape: (n_patches, patch_h, patch_w)
        patches = patches.reshape(n_patches_y, n_patches_x, patch_h, patch_w)
        result = patches.transpose(0, 2, 1, 3).reshape(target_h, target_w)

    result = Image.fromarray(result)
    return result

if __name__ == '__main__':
    image = Image.open('/mnt/others/dataset/csip/eval_dataset_v0_tiny/36/7443856cfd9f7ab67d48f27cbc5e865b.jpg')
    # image = uniform_patch_crop_vectorized(image, (384, 384), (32, 32))
    image = image.resize((384, 384), resample=Image.Resampling.LANCZOS)
    image.save('crop.png')