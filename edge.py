import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt


def resize_image(image, short_edge_size=1024):
    # 获取图像的高度和宽度
    height, width = image.shape[:2]

    # 计算宽高比
    aspect_ratio = width / height

    # 根据短边计算长边的大小
    if width < height:
        new_width = short_edge_size
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = short_edge_size
        new_width = int(new_height * aspect_ratio)

    # 调整图像大小
    resized_image = cv2.resize(image, (new_width, new_height))

    return resized_image

def calculate_contour_widths(image_path, gamma=1/3.):
    # 读取图像
    image = cv2.imread(image_path)
    image = resize_image(image, 1024)
    #image = cv2.boxFilter(image, -1, (5, 5), normalize=True)
    image = cv2.medianBlur(image, 7)
    #image = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)

    # 提取线稿
    image_blur = cv2.GaussianBlur(image, (3, 3), 0)
    image_line = np.abs(image_blur.astype(float)-image.astype(float))
    image_line = ((image_line/255.)**gamma)*255. # gamma矫正，提高线稿辨识度
    image_line = image_line.astype(np.uint8)

    # 灰度化、中值滤波、二值化
    image_line = cv2.cvtColor(image_line, cv2.COLOR_BGR2GRAY)
    image_line = cv2.medianBlur(image_line, 5)
    image_line = cv2.threshold(image_line, 40, 255, cv2.THRESH_BINARY)[1]

    # 提取线条的中心线
    skeleton = cv2.ximgproc.thinning(image_line)

    cv2.imshow('image_line', image_line)
    cv2.imshow('skeleton', skeleton)
    cv2.waitKey()

    distance_map = distance_transform_edt(image_line/255.) #计算线条上点到边缘距离
    skeleton_points = np.argwhere(skeleton > 0) # 获取线条中心到边缘距离（平均宽度）

    widths = []

    # 遍历骨架上的每个点
    for point in skeleton_points:
        y, x = point
        distance = distance_map[y, x]
        width = 2 * distance  # 宽度为距离的两倍
        widths.append(width)

    return widths

if __name__ == '__main__':

    # 示例用法
    image_path = '/mnt/others/dataset/csip/32410994_p0_master1200.jpg'
    contour_widths = calculate_contour_widths(image_path)

    # 打印轮廓宽度分布
    print("Contour Width Distribution:")
    #print(f"{contour_widths:.2f} pixels")
    print(np.array(contour_widths).mean())

    plt.figure(figsize=(8, 6))
    plt.hist(contour_widths, bins=20, edgecolor='black')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Frequency')
    plt.title('Contour Width Distribution')
    plt.grid(True)
    plt.show()

