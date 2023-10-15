
# Author: Zhengxiong Li
# Email: zhengxiong_li@foxmail.com

import numba as nb
import numpy as np
from scipy import ndimage
import cv2 as cv
import rembg

# Functions declaration
@nb.jit(nopython=True)
def inverse(img: np.ndarray) -> np.ndarray: 
    """
    Function:
        翻转灰度图像
    Args:
        img: 输入的图像
    Returns:
        处理过的图像
    """
    height = img.shape[0]
    width = img.shape[1]
    result_img = np.zeros((height, width), dtype=np.uint8)
    height = img.shape[0]
    width = img.shape[1]
    for i in range(0, height):
        for j in range(0, width):
            result_img[i, j] = 255 - img[i, j]
            
    return result_img

def fill(img: np.ndarray, h_max: int = 255) -> np.ndarray:
    """
    Function:
        填充灰度图像中的空洞
    Args:
        img: 输入的图像
        h_max: 默认参数
    Returns:
        填补空洞后的图像
    """
    input_array = np.copy(img) 
    el = ndimage.generate_binary_structure(2,2).astype(int)
    inside_mask = ndimage.binary_erosion(~np.isnan(input_array), structure=el)
    output_array = np.copy(input_array)
    output_array[inside_mask]=h_max
    output_old_array = np.copy(input_array)
    output_old_array.fill(0)   
    el = ndimage.generate_binary_structure(2,1).astype(int)
    while not np.array_equal(output_old_array, output_array):
        output_old_array = np.copy(output_array)
        output_array = np.maximum(input_array,ndimage.grey_erosion(output_array, footprint=el))
    return output_array

def remove_background2(img: np.ndarray) -> np.ndarray:
    """
    Function:
        利用rembg库，去除单个物体的背景，然后转换为灰度图输出
    Args:
        img: 要去除背景的图像
    Returns:
        去除背景之后的图像
    """
    img_nobg = rembg.remove(img)
    img_nobg = cv.cvtColor(img_nobg, cv.COLOR_BGR2GRAY)
    
    return img_nobg

@nb.jit(nopython=True)
def remove_background(img: np.ndarray) -> np.ndarray:
    """
    Function:
        从原始图像中根据BGR的差值去除背景
    Args:
        img: 原始图像
    Returns:
        取出背景后的图像
    """
    BOUND = 3
    result_img = img.copy()
    height = img.shape[0] 
    width = img.shape[1]
    for i in range(0, height):
        for j in range(0, width):
            b = int(img[i, j][0])
            g = int(img[i, j][1])
            r = int(img[i, j][2])
            YELLOW = (r > g) and (g > b)
            BLUE = (b > g) and (g > r)
            
            if (YELLOW or BLUE):
                pass
            else:
                result_img[i, j] = [255,255, 255]
    
    return result_img

def preprocess(img_original: np.ndarray) -> np.ndarray:
    """
    Functions:
        对图像进行预处理，输入原图输出预处理的图
    Args:
        img_original: 原始图像
    Returns:
        经过预处理后的图像
    """
    img_nobg = remove_background(img_original)              # remove background
    img_gray = cv.cvtColor(img_nobg, cv.COLOR_BGR2GRAY)     # transform into gray scale
    # 经过去背景后，图像中有空洞，所以进行填充
    # 先翻转灰度图
    img_gray_inverse = inverse(img_gray)
    # 填充空洞
    img_fill = fill(img_gray_inverse)
    # 再翻转回来
    img_preprocessed = inverse(img_fill)
    return img_preprocessed

# 测试用例
if __name__ == "__main__":
    img = cv.imread("/mnt/Disk1/whole slide image analysis/Lizhengxiong/manual_thumbnail/biopsy_img_1_1_rotate.png")
    # img_result = preprocess(img)
    img_result = remove_background2(img)
    cv.imwrite("result.png", img_result)