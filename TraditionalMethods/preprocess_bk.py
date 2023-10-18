
# Author: Zhengxiong Li
# Email: zhengxiong_li@foxmail.com

import numba as nb
import numpy as np
from scipy import ndimage
import cv2 as cv
import rembg
import openslide

# Functions declaration
def read_slide(path):
    """
    Read slide file in .mrxs format.
    """
    slide = openslide.open_slide(path)
    print(slide.level_dimensions)
    
    return


@nb.jit(nopython=True)
def inverse(img: np.ndarray): 
    """
    翻转灰度图像
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

def fill(test_array, h_max=255):
    """
    填充灰度图像中的空洞
    """
    input_array = np.copy(test_array) 
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

@nb.jit(nopython=True)
def remove_background(img: np.ndarray):
    """
    从原始图像中根据BGR的差值去除背景
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
            delta_bg = abs(b - g)
            delta_br = abs(r - b)
            delta_rg = abs(r - g)
            if (delta_bg < BOUND and delta_br < BOUND and delta_rg < BOUND):
                result_img[i, j] = [255, 255, 255]
    
    return result_img

def preprocess(img_original):
    """
    对图像进行预处理，输入原图输出预处理的图
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

def single_preprocess(img: np.ndarray):
    """
    对单体组织切片图像进行背景去除，调用rembg库即可
    """
    img_nobg = rembg.remove(img)
    img_gray = cv.cvtColor(img_nobg, cv.COLOR_BGR2GRAY)
    img_balance = cv.equalizeHist(img_gray)
    
    return img_balance

# 测试用例
if __name__ == "__main__":
    img = cv.imread("../biopsy_img_2_3.png")
    img_result = preprocess(img)
    