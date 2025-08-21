import numpy as np
from PIL import Image

class UniformPatch:
    def __init__(self, target_size, patch_size):
        self.target_size = target_size
        self.patch_size = patch_size
        self.n_patches_x = target_size[0] // patch_size[0]
        self.n_patches_y = target_size[1] // patch_size[1]
        self.patch_y_offsets = np.arange(patch_size[1])
        self.patch_x_offsets = np.arange(patch_size[0])
    
    def __call__(self, image):
        image = np.asarray(image)
        target_w, target_h = self.target_size
        patch_w, patch_h = self.patch_size

        if image.ndim == 3:
            orig_h, orig_w, channels = image.shape
        else:
            orig_h, orig_w = image.shape
            channels = None

        # 计算均匀采样位置
        if self.n_patches_x == 1:
            x_starts = np.array([(orig_w - patch_w) // 2])
        else:
            x_starts = np.linspace(0, orig_w - patch_w, self.n_patches_x, dtype=int)

        if self.n_patches_y == 1:
            y_starts = np.array([(orig_h - patch_h) // 2])
        else:
            y_starts = np.linspace(0, orig_h - patch_h, self.n_patches_y, dtype=int)

        # 生成所有patch的绝对坐标
        Y_starts, X_starts = np.meshgrid(y_starts, x_starts, indexing='ij')
        Y_offsets, X_offsets = np.meshgrid(self.patch_y_offsets, self.patch_x_offsets, indexing='ij')

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
            patches = patches.reshape(self.n_patches_y, self.n_patches_x, patch_h, patch_w, channels)
            result = patches.transpose(0, 2, 1, 3, 4).reshape(target_h, target_w, channels)
        else:
            # 灰度图像
            patches = image[Y_indices, X_indices]  # shape: (n_patches, patch_h, patch_w)
            patches = patches.reshape(self.n_patches_y, self.n_patches_x, patch_h, patch_w)
            result = patches.transpose(0, 2, 1, 3).reshape(target_h, target_w)

        result = Image.fromarray(result)
        return result
