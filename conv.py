
# Author: Zhengxiong Li
# Email: zhengxiong_li@foxmail.com

import numba as nb
import numpy as np
import taichi as ti


# Functions
@ti.func
def loc_conv(img_expand: ti.types.ndarray(), mask: ti.types.ndarray(), i: ti.i32, j: ti.i32) -> ti.float32:
    """
    Function:
        在指定位置进行卷积
    Args:
        img: 输入的图像，已经经过边界复制的处理
        mask: 卷积核
        i: 要卷积的位置高度方向上的坐标
        j: 要卷积的位置宽度方向上的坐标
    Return:
        卷积值
    """
    result = 0.0
    radius = int((mask.shape[0] - 1) / 2)
    size = mask.shape[0]
    for k, l in ti.ndrange((0, size), (0, size)):
        img_value = img_expand[i-radius+k, j-radius+l]
        mask_value = mask[k, l]
        result += img_value * mask_value
    return result

@ti.kernel
def _convolve(img_expand: ti.types.ndarray(), mask: ti.types.ndarray(), result: ti.types.ndarray()):
    """
    Function:
        针对已经扩展过的img计算卷积后的矩阵
    Args:
        img_expand: 经过边界扩展之后的图像矩阵
        mask: 卷积核
        result: 通过传引用输入，用来保存结果的矩阵
    Return:
        None
    """
    radius = int((mask.shape[0] - 1) / 2)
    for i, j in ti.ndrange((0, result.shape[0]), (0, result.shape[1])):
        result[i, j] = loc_conv(img_expand, mask, i+radius, j+radius)
    
    return

def convolve(img: np.ndarray, mask: ti.ndarray) -> np.ndarray:
    """
    Function:
        将输入的图像进行边界复制，然后送给Taichi函数计算卷积结果
    Args:
        img: 要被卷积的图像矩阵
        mask: 卷积核
    Return:
        None
    """
    radius = int((mask.shape[0] - 1) / 2)
    pad_width = ((radius, radius), (radius, radius))
    img_expand = np.pad(img, pad_width, mode="edge")
    img_expand = img_expand.astype(np.float32)
    result = np.zeros_like(img, dtype=np.float32)
    _convolve(img_expand, mask, result)

    return result

@nb.jit(nopython=True)
def build_ring(radius: int, thickness: int, width: int) -> np.ndarray:
    """
    Function:
        创建一个环形的卷积核
    Args:
        radius: 环形卷积核的内圆半径
        thickness: 环的厚度
    Returns:
        卷积核矩阵
    """
    mask = np.zeros((width, width)).astype(np.float32)
    center = (width - 1) / 2
    count = 0
    # calculate which pixel should be 1
    for i in range(0, width):
        for j in range(0, width):
            distance = np.sqrt((i-center)**2 + (j-center)**2)
            if (distance <= radius + thickness and distance > radius):
                mask[i,j] = 1
                count = count + 1
    mask = mask / count
    
    return mask

def get_mask_list(radius_min: int, radius_max: int, thickness: int) -> list:
    """
    Function:
        获取卷积核序列，半径从radius_min到radius_max，左闭右闭
        不同的环之间有三分之一的区域是重叠的
    Args:
        radius_min: 半径的下限
        radius_max: 半径的上限
    Returns:
        卷积核矩阵序列
    """
    result_list = list()
    step = thickness // 3
    width = 2 * (radius_max + thickness) + 1
    for i in range(radius_min, radius_max+1, step):
        mask = build_ring(i, thickness, width)
        result_list.append(mask)
    
    return result_list

def get_conv_list(img: np.ndarray, mask_list: list) -> list:
    """
    Function:
        获取卷积图像金字塔
    Args:
        img: 预处理后的图像
        mask_list: 卷积核矩阵序列
    Returns:
        卷积结果图序列
    """
    conv_list = list()
    radius = int((mask_list[0].shape[0] - 1) / 2)
    pad_width = ((radius, radius), (radius, radius))
    img_expand = np.pad(img, pad_width, mode="edge")
    for mask in mask_list:
        img_conv = convolve(img, mask)
        img_conv = img_conv.astype(np.uint8)
        conv_list.append(img_conv)
        
    return conv_list, img_expand

def get_diff_list(conv_list: list) -> list:
    """
    Functions:
        由卷积金字塔，获取差分图像金字塔
    Args:
        conv_list: list[np.ndarray], 卷积结果图序列
    Returns:
        list[np.ndarray], 差分结果图序列
    """
    diff_list = list()
    depth = len(conv_list) - 1
    for i in range(0, depth):
        img_diff = conv_list[i + 1] - conv_list[i]
        diff_list.append(img_diff)
        
    return diff_list

if __name__ == "__main__":
    ti.init(arch = ti.cpu)
    import cv2 as cv
    img = cv.imread("/mnt/Disk1/whole slide image analysis/Lizhengxiong/manual_thumbnail/biopsy_img_1_1.png", cv.IMREAD_GRAYSCALE)
    mask_list = get_mask_list(10, 20, 7)
    temp = get_conv_list(img, mask_list)
    print(len(mask_list))
    print(len(temp))
    # for i in range(0, len(temp)):
    #     img = temp[i]
    #     cv.imwrite("img_conv_{0}.png".format(i), img)
    print("Get convolve list successfully!")
    for j in mask_list:
        print(j.shape)
    np.savetxt("mask.csv", mask_list[0], fmt="%.6f")
    